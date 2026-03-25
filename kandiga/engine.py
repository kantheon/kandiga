"""Kandiga inference engine \u2014 Selective Expert Materialization (SEM).

Runs 35B MoE models in 1.5GB RAM by:
1. Loading only shared layers (attention, norms) into GPU memory (~1.5GB)
2. Loading expert MLP weights from disk on demand (8 of 256 per token per layer)
3. Computing expert MLP on CPU with NEON vectorization + GCD parallelism
4. GPU handles attention via MLX while CPU handles experts in parallel

Architecture (KTransformers-inspired, Apple Silicon optimized):
  GPU (MLX): attention, norms, routing, shared expert, weight blending
  CPU (C/NEON): routed expert MLP (gate + up + SwiGLU + down)

Why CPU experts beat GPU experts on Apple Silicon:
  1. Zero Metal dispatch overhead (no command buffer creation/commit/wait)
  2. Direct pread from OS page cache (no staging buffer copy)
  3. NEON-vectorized 4-bit dequant matvec (competitive for small matrices)
  4. GCD parallel dispatch across P-cores (8 experts on 4 cores)

Performance (M4 Mac Mini, 16GB):
  Quality mode (K=8): ~3.5 tok/s, 1.5GB RSS
  Fast mode (K=4):    ~6.5 tok/s, 1.5GB RSS
"""

from __future__ import annotations

import gc
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def mem_stats() -> Dict[str, float]:
    """Current memory statistics in MB."""
    stats = {"gpu_active_mb": mx.get_active_memory() / 1e6}
    try:
        import psutil
        stats["rss_mb"] = psutil.Process().memory_info().rss / 1e6
    except ImportError:
        import resource
        stats["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return stats


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def _find_layers(model: nn.Module) -> list:
    """Auto-detect the transformer layers list from the model."""
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Cannot find transformer layers in model")


# ---------------------------------------------------------------------------
# CPU Expert Library (ctypes wrapper)
# ---------------------------------------------------------------------------

class _CPUExpertLib:
    """Thin ctypes wrapper around libkandiga_cpu_expert[_lg].dylib."""

    def __init__(self, large: bool = False):
        import ctypes

        # Look for the dylib in the package's metal/ directory
        metal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal")
        dylib_name = "libkandiga_cpu_expert_lg.dylib" if large else "libkandiga_cpu_expert.dylib"
        dylib_path = os.path.join(metal_dir, dylib_name)
        self._large = large

        if not os.path.exists(dylib_path):
            raise FileNotFoundError(
                f"CPU expert library not found at {dylib_path}. "
                f"Build it with: cd kandiga/metal && make"
            )

        self._lib = ctypes.CDLL(dylib_path)

        # bakan_cpu_expert_init(packed_dir, num_layers) -> engine*
        self._lib.bakan_cpu_expert_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.bakan_cpu_expert_init.restype = ctypes.c_void_p

        # bakan_cpu_expert_mlp_f16(engine, layer, x_f16, indices, K, output_f16)
        self._lib.bakan_cpu_expert_mlp_f16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.bakan_cpu_expert_mlp_f16.restype = ctypes.c_int

        # bakan_cpu_expert_mlp_bf16
        self._lib.bakan_cpu_expert_mlp_bf16.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_void_p,
        ]
        self._lib.bakan_cpu_expert_mlp_bf16.restype = ctypes.c_int

        # bakan_cpu_expert_destroy(engine)
        self._lib.bakan_cpu_expert_destroy.argtypes = [ctypes.c_void_p]
        self._lib.bakan_cpu_expert_destroy.restype = None

        # bakan_cpu_expert_num_layers(engine) -> int
        self._lib.bakan_cpu_expert_num_layers.argtypes = [ctypes.c_void_p]
        self._lib.bakan_cpu_expert_num_layers.restype = ctypes.c_int

    def init(self, packed_dir: str, num_layers: int):
        """Create an opaque engine handle."""
        engine = self._lib.bakan_cpu_expert_init(packed_dir.encode(), num_layers)
        if engine is None:
            raise RuntimeError(f"Failed to initialize CPU expert engine from {packed_dir}")
        return engine

    def expert_mlp_f16(
        self,
        engine,
        layer_idx: int,
        x_f16: np.ndarray,
        expert_indices: np.ndarray,
        num_experts: int,
        hidden_size: int = 2048,
    ) -> np.ndarray:
        """Run expert MLP on CPU with float16 I/O."""
        import ctypes

        output = np.empty((num_experts, hidden_size), dtype=np.float16)
        indices_c = np.ascontiguousarray(expert_indices.astype(np.int32))

        ret = self._lib.bakan_cpu_expert_mlp_f16(
            engine,
            layer_idx,
            x_f16.ctypes.data_as(ctypes.c_void_p),
            indices_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            num_experts,
            output.ctypes.data_as(ctypes.c_void_p),
        )
        if ret != 0:
            raise RuntimeError(f"CPU expert MLP failed for layer {layer_idx}")
        return output

    def expert_mlp_bf16(
        self, engine, layer_idx: int, x_bf16: np.ndarray,
        expert_indices: np.ndarray, num_experts: int, hidden_size: int = 2048,
    ) -> np.ndarray:
        """Run expert MLP with bfloat16 input. Returns float16 output."""
        import ctypes
        output = np.empty((num_experts, hidden_size), dtype=np.float16)
        indices_c = np.ascontiguousarray(expert_indices.astype(np.int32))
        ret = self._lib.bakan_cpu_expert_mlp_bf16(
            engine, layer_idx,
            x_bf16.ctypes.data_as(ctypes.c_void_p),
            indices_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            num_experts,
            output.ctypes.data_as(ctypes.c_void_p),
        )
        if ret != 0:
            raise RuntimeError(f"CPU expert MLP (bf16) failed for layer {layer_idx}")
        return output

    def destroy(self, engine):
        """Release engine resources."""
        if engine is not None:
            self._lib.bakan_cpu_expert_destroy(engine)

    def num_layers(self, engine) -> int:
        return self._lib.bakan_cpu_expert_num_layers(engine)


# ---------------------------------------------------------------------------
# CPU SwitchGLU wrapper (replaces MLX expert MLP with CPU computation)
# ---------------------------------------------------------------------------

class _CPUSwitchGLU(nn.Module):
    """Replaces SwitchGLU with CPU-computed expert MLP.

    When called by MLX during forward pass:
    1. mx.eval(x, indices) to sync GPU and get concrete values
    2. Convert to numpy (zero-copy on unified memory)
    3. Call C library for CPU expert computation (NEON + GCD parallel)
    4. Return result as mx.array for MLX to continue lazy evaluation
    """

    def __init__(
        self,
        original: nn.Module,
        layer_idx: int,
        cpu_lib: _CPUExpertLib,
        cpu_engine,
        hidden_size: int = 2048,
    ):
        super().__init__()
        self.original = original
        self._layer_idx = layer_idx
        self._cpu_lib = cpu_lib
        self._cpu_engine = cpu_engine
        self._hidden_size = hidden_size

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """Forward pass: expert MLP computed on CPU."""
        original_shape = indices.shape
        K = indices.shape[-1]

        x_flat = x.reshape(-1, self._hidden_size)
        idx_i32 = indices.reshape(-1, K).astype(mx.int32)
        mx.eval(x_flat, idx_i32)

        # Use bfloat16 path if available (avoids extra astype)
        is_bf16 = x_flat.dtype == mx.bfloat16
        if not is_bf16:
            x_flat = x_flat.astype(mx.float16)
            mx.eval(x_flat)

        x_np = np.array(x_flat.view(mx.uint16) if is_bf16 else x_flat, copy=False)
        idx_np = np.array(idx_i32, copy=False).astype(np.int32)

        num_tokens = x_np.shape[0]
        out_np = np.empty((num_tokens, K, self._hidden_size), dtype=np.float16)

        for t in range(num_tokens):
            if is_bf16 and hasattr(self._cpu_lib, '_lib') and hasattr(self._cpu_lib._lib, 'bakan_cpu_expert_mlp_bf16'):
                result_t = self._cpu_lib.expert_mlp_bf16(
                    self._cpu_engine, self._layer_idx,
                    np.ascontiguousarray(x_np[t]), idx_np[t], K,
                    hidden_size=self._hidden_size,
                )
            else:
                result_t = self._cpu_lib.expert_mlp_f16(
                    self._cpu_engine, self._layer_idx,
                    np.ascontiguousarray(x_np[t]), idx_np[t], K,
                    hidden_size=self._hidden_size,
                )
            out_np[t] = result_t

        result = mx.array(out_np.reshape(-1, K, self._hidden_size))
        target_shape = list(original_shape) + [self._hidden_size]
        return result.reshape(target_shape)


# ---------------------------------------------------------------------------
# Thinking-tag stripping
# ---------------------------------------------------------------------------

def _strip_thinking(text: str, thinking_enabled: bool = False) -> str:
    """Remove Qwen thinking tags from output."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    elif thinking_enabled and text:
        return ""
    return text


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class KandigaEngine:
    """Main inference engine for Kandiga.

    Loads a 35B MoE model with Selective Expert Materialization:
    - Shared parameters (attention, norms, embeddings) loaded to GPU (~1.5GB)
    - Expert weights read from packed binary files on CPU per token
    - NEON-vectorized 4-bit dequant + GCD parallel dispatch
    """

    DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
    CACHE_DIR = os.path.expanduser("~/.kandiga/experts")
    EXPERT_PATTERN = "switch_mlp"

    def __init__(
        self,
        model_path: str | None = None,
        fast_mode: bool = False,
        log_memory: bool = False,
    ):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.fast_mode = fast_mode
        self._log_memory = log_memory
        self._model = None
        self._tokenizer = None
        self._cpu_lib: _CPUExpertLib | None = None
        self._cpu_engine = None
        self._ready = False

    def _log(self, msg: str):
        if self._log_memory:
            stats = mem_stats()
            print(
                f"[kandiga] {msg} | "
                f"RSS={stats['rss_mb']:.0f}MB "
                f"GPU={stats['gpu_active_mb']:.0f}MB",
                flush=True,
            )

    def _model_cache_dir(self) -> str:
        """Return the expert cache directory for the current model."""
        model_name = self.model_path.split("/")[-1]
        return os.path.join(self.CACHE_DIR, model_name)

    def load(self):
        """Load the model with SEM engine.

        Steps:
        1. Verify packed binary expert files exist
        2. Initialize CPU expert library (NEON + GCD)
        3. Load model structure with lazy weights
        4. Eagerly load shared (non-expert) parameters to GPU
        5. Count MoE layers and initialize CPU engine
        6. Install CPU SwitchGLU wrappers on every MoE layer
        7. Optionally apply fast mode (K=4)
        """
        if self._ready:
            return

        # Step 1: Verify packed binary files exist
        cache_dir = self._model_cache_dir()
        packed_dir = os.path.join(cache_dir, "packed")
        if not os.path.isdir(packed_dir):
            raise FileNotFoundError(
                f"Packed expert files not found at {packed_dir}.\n"
                f"Run 'kandiga setup' first to download and prepare the model."
            )

        # Step 2: Load model with lazy weights
        self._model, self._tokenizer = load(self.model_path, lazy=True)
        self._log("Model structure loaded (lazy)")

        # Step 3: Detect model dimensions
        hidden_size = 2048  # default (35B)
        is_large = False
        # Try model.language_model.args.hidden_size (Qwen3.5 structure)
        lm = getattr(self._model, 'language_model', None)
        if lm and hasattr(lm, 'args'):
            hs = getattr(lm.args, 'hidden_size', 2048)
            if hs != 2048:
                hidden_size = hs
                is_large = True
        if not is_large and hasattr(self._model, 'args'):
            hs = getattr(self._model.args, 'hidden_size', 2048)
            if hs != 2048:
                hidden_size = hs
                is_large = True
        if is_large:
            self._log(f"Large model: hidden_size={hidden_size}")

        # Step 4: Initialize CPU expert library (pick right dylib)
        self._cpu_lib = _CPUExpertLib(large=is_large)
        self._log(f"CPU expert library loaded ({'lg' if is_large else 'standard'})")

        # Step 5: Eagerly load shared (non-expert) parameters
        flat = tree_flatten(self._model.parameters())
        shared = [v for k, v in flat if self.EXPERT_PATTERN not in k]
        if shared:
            mx.eval(*shared)
        self._log("Shared parameters loaded to GPU")

        # Step 5b: Apply ZMLX fused kernels (deltanet, norms, softmax)
        try:
            from zmlx.patch import patch as zmlx_patch
            zmlx_patch(self._model)
            self._log("ZMLX fused kernels applied")
        except ImportError:
            self._log("ZMLX not installed (pip install zmlx for +15% speed)")
        except Exception as e:
            self._log(f"ZMLX patch failed: {e}")

        # Step 6: Count MoE layers and initialize CPU engine
        layers = _find_layers(self._model)
        moe_count = sum(
            1
            for layer in layers
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp")
        )

        self._cpu_engine = self._cpu_lib.init(packed_dir, moe_count)
        self._log(f"CPU engine initialized ({moe_count} MoE layers)")

        # Step 7: Install CPU SwitchGLU wrappers
        moe_idx = 0
        for layer in layers:
            mlp = layer.mlp if hasattr(layer, "mlp") else None
            if mlp is not None and hasattr(mlp, "switch_mlp"):
                wrapper = _CPUSwitchGLU(
                    original=mlp.switch_mlp,
                    layer_idx=moe_idx,
                    cpu_lib=self._cpu_lib,
                    cpu_engine=self._cpu_engine,
                    hidden_size=hidden_size,
                )
                mlp.switch_mlp = wrapper
                moe_idx += 1

        # Step 7: Apply fast mode (K=4) if requested
        if self.fast_mode:
            for layer in layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "top_k"):
                    layer.mlp.top_k = 4
            self._log("Fast mode: K=4 experts")
        else:
            self._log("Quality mode: K=8 experts")

        self._log(f"Engine ready ({moe_idx} MoE layers, CPU expert wrappers installed)")
        self._ready = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temp: float = 0.0,
        stream: bool = True,
    ):
        """Generate a response.

        If stream=True, yields tokens one at a time.
        If stream=False, returns the full response string.
        """
        if not self._ready:
            self.load()

        messages = [{"role": "user", "content": prompt}]

        # Apply chat template with thinking disabled (saves token budget)
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if stream:
            return self._stream_generate(formatted, max_tokens, temp)
        else:
            sampler = make_sampler(temp=temp)
            response = generate(
                self._model,
                self._tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            return _strip_thinking(response)

    def _stream_generate(self, formatted_prompt: str, max_tokens: int, temp: float):
        """Stream tokens using mlx_lm's generate with a callback."""
        try:
            from mlx_lm.utils import stream_generate
        except ImportError:
            from mlx_lm import stream_generate

        sampler = make_sampler(temp=temp)

        for response in stream_generate(
            self._model,
            self._tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            text = response.text
            if text:
                # Filter out thinking tags in streaming
                if "<think>" in text or "</think>" in text:
                    continue
                yield text

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def stats(self) -> Dict[str, Any]:
        """Return current memory and speed stats."""
        s = mem_stats()
        return {
            "rss_mb": s["rss_mb"],
            "gpu_active_mb": s["gpu_active_mb"],
            "model": self.model_path,
            "fast_mode": self.fast_mode,
            "mode": "K=4 (fast)" if self.fast_mode else "K=8 (quality)",
        }

    def __del__(self):
        if self._cpu_engine is not None and self._cpu_lib is not None:
            self._cpu_lib.destroy(self._cpu_engine)
            self._cpu_engine = None
