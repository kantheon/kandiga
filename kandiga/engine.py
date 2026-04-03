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

        # bakan_cpu_expert_pread_batch
        self._lib.bakan_cpu_expert_pread_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.bakan_cpu_expert_pread_batch.restype = ctypes.c_int

        # bakan_cpu_expert_mlp_prefill_f16 (35B only — lg library doesn't have it)
        try:
            self._lib.bakan_cpu_expert_mlp_prefill_f16.argtypes = [
            ctypes.c_void_p, ctypes.c_int,  # engine, layer
            ctypes.c_void_p,                 # x_f16
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_tokens, K, hidden
            ctypes.POINTER(ctypes.c_int32),  # sorted_token_indices
            ctypes.POINTER(ctypes.c_int32),  # sorted_expert_indices
            ctypes.POINTER(ctypes.c_int32),  # sorted_k_positions
            ctypes.POINTER(ctypes.c_int32),  # group_starts
            ctypes.POINTER(ctypes.c_int32),  # unique_experts
            ctypes.c_int,                     # num_unique
            ctypes.c_void_p,                 # output
        ]
            self._lib.bakan_cpu_expert_mlp_prefill_f16.restype = ctypes.c_int
        except (AttributeError, OSError):
            pass  # lg library doesn't have this function

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
# Cross-layer expert speculation (shared state)
# ---------------------------------------------------------------------------
import threading

_prefetch_state = {
    "predicted": {},     # layer_idx -> list of expert indices
    "thread": None,      # background prefetch thread
    "hits": 0,
    "misses": 0,
}


# ---------------------------------------------------------------------------
# Custom Metal kernels for batched expert MLP (zero Python loop overhead)
# ---------------------------------------------------------------------------
_DEQUANT_HEADER = '''
inline float dq(const device uint32_t* w, const device half* s, const device half* b,
                const device half* x, uint row, uint hd, uint gs, uint ppr, uint gpr) {
    float acc=0; const device uint32_t* wr=w+row*ppr;
    const device half* sr=s+row*gpr; const device half* br=b+row*gpr;
    for(uint g=0;g<gpr;g++){float sc=float(sr[g]),bi=float(br[g]);
    for(uint p=0;p<gs/8;p++){uint32_t pk=wr[g*(gs/8)+p];uint xb=g*gs+p*8;
    acc+=(float((pk    )&0xF)*sc+bi)*float(x[xb])+(float((pk>>4)&0xF)*sc+bi)*float(x[xb+1])
        +(float((pk>> 8)&0xF)*sc+bi)*float(x[xb+2])+(float((pk>>12)&0xF)*sc+bi)*float(x[xb+3])
        +(float((pk>>16)&0xF)*sc+bi)*float(x[xb+4])+(float((pk>>20)&0xF)*sc+bi)*float(x[xb+5])
        +(float((pk>>24)&0xF)*sc+bi)*float(x[xb+6])+(float((pk>>28)&0xF)*sc+bi)*float(x[xb+7]);}}
    return acc;
}
inline float dqf(const device uint32_t* w, const device half* s, const device half* b,
                 const device float* x, uint row, uint ed, uint gs, uint ppr, uint gpr) {
    float acc=0; const device uint32_t* wr=w+row*ppr;
    const device half* sr=s+row*gpr; const device half* br=b+row*gpr;
    for(uint g=0;g<gpr;g++){float sc=float(sr[g]),bi=float(br[g]);
    for(uint p=0;p<gs/8;p++){uint32_t pk=wr[g*(gs/8)+p];uint xb=g*gs+p*8;
    acc+=(float((pk    )&0xF)*sc+bi)*x[xb]+(float((pk>>4)&0xF)*sc+bi)*x[xb+1]
        +(float((pk>> 8)&0xF)*sc+bi)*x[xb+2]+(float((pk>>12)&0xF)*sc+bi)*x[xb+3]
        +(float((pk>>16)&0xF)*sc+bi)*x[xb+4]+(float((pk>>20)&0xF)*sc+bi)*x[xb+5]
        +(float((pk>>24)&0xF)*sc+bi)*x[xb+6]+(float((pk>>28)&0xF)*sc+bi)*x[xb+7];}}
    return acc;
}
'''

_metal_phase1 = mx.fast.metal_kernel(
    name='expert_gate_up_silu',
    input_names=['x_input','gw','gs_','gb','uw','us_','ub','slots','tokens','dims'],
    output_names=['activated'],
    header=_DEQUANT_HEADER,
    source='''
    uint tid=thread_position_in_grid.x;
    uint ed=dims[0],hd=dims[1],total=dims[2],gs_v=64,ppr=hd/8,gpr=hd/gs_v;
    uint ai=tid/ed, row=tid%ed;
    if(ai>=total||row>=ed)return;
    uint sl=slots[ai],tk=tokens[ai];
    const device half* xt=x_input+tk*hd;
    float g=dq(gw+sl*ed*ppr,gs_+sl*ed*gpr,gb+sl*ed*gpr,xt,row,hd,gs_v,ppr,gpr);
    float u=dq(uw+sl*ed*ppr,us_+sl*ed*gpr,ub+sl*ed*gpr,xt,row,hd,gs_v,ppr,gpr);
    activated[ai*ed+row]=(g/(1.0f+exp(-g)))*u;
    ''',
)

_metal_phase2 = mx.fast.metal_kernel(
    name='expert_down_proj',
    input_names=['activated','dw','ds_','db','slots','dims'],
    output_names=['output'],
    header=_DEQUANT_HEADER,
    source='''
    uint tid=thread_position_in_grid.x;
    uint hd=dims[0],ed=dims[1],total=dims[2],gs_v=64,ppr=ed/8,gpr=ed/gs_v;
    uint ai=tid/hd, row=tid%hd;
    if(ai>=total||row>=hd)return;
    uint sl=slots[ai];
    float val=dqf(dw+sl*hd*ppr,ds_+sl*hd*gpr,db+sl*hd*gpr,activated+ai*ed,row,ed,gs_v,ppr,gpr);
    output[ai*hd+row]=half(val);
    ''',
)


@mx.compile
def _compiled_expert_mlp(x, gw, gs, gb, uw, us, ub, dw, ds, db):
    """Compiled expert MLP: dequantize + matmul + SiLU + down. Graph reused."""
    g = mx.dequantize(gw, gs, gb, group_size=64, bits=4)
    u = mx.dequantize(uw, us, ub, group_size=64, bits=4)
    d = mx.dequantize(dw, ds, db, group_size=64, bits=4)
    go = x @ g.T
    uo = x @ u.T
    act = (go / (1 + mx.exp(-go))) * uo
    return (act @ d.T).astype(mx.float16)


def _prefetch_experts_to_page_cache(packed_dir: str, layer_idx: int,
                                     expert_indices: list, expert_size: int):
    """Background thread: pread experts to warm OS page cache."""
    import struct
    path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.bin")
    try:
        fd = os.open(path, os.O_RDONLY)
        header_size = 4096
        for eidx in expert_indices:
            offset = header_size + eidx * expert_size
            os.pread(fd, expert_size, offset)
        os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# CPU SwitchGLU wrapper (replaces MLX expert MLP with CPU computation)
# ---------------------------------------------------------------------------

class _CPUSwitchGLU(nn.Module):
    """Replaces SwitchGLU with CPU-computed expert MLP.

    Includes cross-layer expert speculation: predicts next layer's experts
    and prefetches them into OS page cache during current layer's compute.
    ~79% prediction accuracy, reducing on-demand disk reads significantly.
    """

    def __init__(
        self,
        original: nn.Module,
        layer_idx: int,
        cpu_lib: _CPUExpertLib,
        cpu_engine,
        hidden_size: int = 2048,
        next_gate_weights=None,
        top_k: int = 8,
        packed_dir: str = "",
        expert_size: int = 0,
        num_moe_layers: int = 0,
    ):
        super().__init__()
        self.original = original
        self._layer_idx = layer_idx
        self._cpu_lib = cpu_lib
        self._cpu_engine = cpu_engine
        self._hidden_size = hidden_size
        self._next_gate_np = None
        self._top_k = top_k
        self._packed_dir = packed_dir
        self._expert_size = expert_size
        self._num_moe_layers = num_moe_layers

        # Pre-compute next layer's router gate as numpy for fast CPU matmul
        if next_gate_weights is not None:
            try:
                # next_gate_weights is a QuantizedLinear module, not raw weights
                # Call it with identity-like input to get the dequantized output
                # Or just extract and dequant manually
                gate_mod = next_gate_weights  # this is actually the gate module
                if hasattr(gate_mod, 'weight') and hasattr(gate_mod, 'scales'):
                    # Dequantize: weight is packed uint32, scales/biases are bf16
                    w = gate_mod.weight
                    s = gate_mod.scales
                    b = getattr(gate_mod, 'biases', None)
                    mx.eval(w, s)
                    if b is not None:
                        mx.eval(b)
                    # Use MLX's built-in dequantize
                    dequant = mx.dequantize(w, s, b, group_size=64, bits=4)
                    mx.eval(dequant)
                    self._next_gate_np = np.array(dequant.astype(mx.float16), copy=False).astype(np.float32)
                elif hasattr(gate_mod, 'weight'):
                    w = gate_mod.weight
                    mx.eval(w)
                    self._next_gate_np = np.array(w.astype(mx.float16), copy=False).astype(np.float32)
            except Exception:
                pass

    def _gpu_prefill_experts(self, x_flat, idx_i32, num_tokens, K, out_np):
        """Prefill: GPU batched expert MLP. One pread + one GPU matmul per expert.

        43x faster than CPU sequential. Groups tokens by expert,
        preads each expert once, GPU batched dequant+matmul.
        """
        h = self._hidden_size
        idx_np = np.array(idx_i32, copy=False).astype(np.int32)
        gs = 64
        esz = self._expert_size

        if esz == 0:
            # No expert size info — fallback to CPU
            mlp_func = self._cpu_lib.expert_mlp_f16
            x_np_raw = np.array(x_flat.astype(mx.float16), copy=False)
            mx.eval(x_flat)
            for t in range(num_tokens):
                out_np[t] = mlp_func(self._cpu_engine, self._layer_idx,
                    np.ascontiguousarray(x_np_raw[t]), idx_np[t], K, hidden_size=h)
            return

        # Vectorized grouping using numpy (no Python dicts)
        flat_tok = np.repeat(np.arange(num_tokens, dtype=np.int32), K)
        flat_k = np.tile(np.arange(K, dtype=np.int32), num_tokens)
        flat_exp = idx_np.reshape(-1)
        sort_idx = np.argsort(flat_exp, kind='mergesort')
        s_tok = flat_tok[sort_idx]
        s_k = flat_k[sort_idx]
        s_exp = flat_exp[sort_idx]

        # Find group boundaries
        changes = np.concatenate([[0], np.where(np.diff(s_exp))[0] + 1, [len(s_exp)]])
        unique_experts = s_exp[changes[:-1]]

        # Get x on GPU
        x_gpu = x_flat.astype(mx.float16)
        mx.eval(x_gpu)

        import ctypes
        num_unique = len(unique_experts)

        mx.clear_cache()

        # ONE C call: merged contiguous reads of unique experts.
        # Sorts expert IDs, groups adjacent into ranges, reads each range
        # as one pread. Reduces USB round-trips by ~7x vs individual reads.
        raw_buf = np.empty(num_unique * esz, dtype=np.uint8)
        unique_i32 = np.ascontiguousarray(unique_experts.astype(np.int32))
        self._cpu_lib._lib.bakan_cpu_expert_pread_batch(
            self._cpu_engine, self._layer_idx,
            unique_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            num_unique,
            raw_buf.ctypes.data_as(ctypes.c_void_p),
        )

        # Compute expert tensor layout (cached after first call)
        if not hasattr(self, '_edim'):
            if h == 2048:
                self._edim = 512
            elif h == 3072:
                self._edim = 1024
            else:
                self._edim = h // 4
            edim = self._edim
            packed_h = h // 8
            groups_h = h // gs
            packed_e = edim // 8
            groups_e = edim // gs
            off = 0
            self._OFS = []
            for rows, cols, dt in [
                (edim, packed_h, np.uint32), (edim, groups_h, np.uint16), (edim, groups_h, np.uint16),
                (edim, packed_h, np.uint32), (edim, groups_h, np.uint16), (edim, groups_h, np.uint16),
                (h, packed_e, np.uint32), (h, groups_e, np.uint16), (h, groups_e, np.uint16),
            ]:
                isz = 4 if dt == np.uint32 else 2
                self._OFS.append((off, rows, cols, dt))
                off += rows * cols * isz
        edim = self._edim

        # Vectorized stack: reshape raw_buf as (num_unique, esz), slice columns
        experts_2d = raw_buf.reshape(num_unique, esz)

        def stack_v(off, rows, cols, dt):
            isz = 4 if dt == np.uint32 else 2
            size = rows * cols * isz
            return mx.array(experts_2d[:, off:off+size].ravel().view(dt))

        all_gw = stack_v(*self._OFS[0])
        all_gs = stack_v(*self._OFS[1]).view(mx.bfloat16).astype(mx.float16)
        all_gb = stack_v(*self._OFS[2]).view(mx.bfloat16).astype(mx.float16)
        all_uw = stack_v(*self._OFS[3])
        all_us = stack_v(*self._OFS[4]).view(mx.bfloat16).astype(mx.float16)
        all_ub = stack_v(*self._OFS[5]).view(mx.bfloat16).astype(mx.float16)
        all_dw = stack_v(*self._OFS[6])
        all_ds = stack_v(*self._OFS[7]).view(mx.bfloat16).astype(mx.float16)
        all_db = stack_v(*self._OFS[8]).view(mx.bfloat16).astype(mx.float16)

        # Build assignment arrays
        total_assigns = len(s_tok)
        eidx_to_slot = np.zeros(256, dtype=np.int32)
        eidx_to_slot[unique_experts] = np.arange(num_unique, dtype=np.int32)
        slot_arr = mx.array(eidx_to_slot[s_exp].astype(np.int32))
        tok_arr = mx.array(s_tok.astype(np.int32))
        dims_p1 = mx.array([edim, h, total_assigns], dtype=mx.int32)
        dims_p2 = mx.array([h, edim, total_assigns], dtype=mx.int32)

        # Phase 1: gate+up+SiLU — ONE Metal dispatch
        activated = _metal_phase1(
            inputs=[x_gpu, all_gw, all_gs, all_gb, all_uw, all_us, all_ub,
                    slot_arr, tok_arr, dims_p1],
            output_shapes=[(total_assigns * edim,)],
            output_dtypes=[mx.float32],
            grid=(total_assigns * edim, 1, 1),
            threadgroup=(min(256, edim), 1, 1),
        )[0]

        # Phase 2: down projection — ONE Metal dispatch
        flat_output = _metal_phase2(
            inputs=[activated, all_dw, all_ds, all_db, slot_arr, dims_p2],
            output_shapes=[(total_assigns * h,)],
            output_dtypes=[mx.float16],
            grid=(total_assigns * h, 1, 1),
            threadgroup=(min(256, h), 1, 1),
        )[0]

        mx.eval(flat_output)

        # Vectorized scatter
        flat_out_np = np.array(flat_output, copy=False).reshape(total_assigns, h)
        out_np[s_tok, s_k] = flat_out_np

    def _batched_prefill_experts(self, x_view, idx_np, actual_tokens, K, out_np):
        """Prefill: one C call per layer, grouped by expert.

        Groups tokens by expert ID, preads each expert ONCE,
        computes all tokens with expert weights hot in L1/L2 cache.
        """
        import ctypes

        # Build CSR-style grouped index
        from collections import defaultdict
        expert_groups = defaultdict(list)
        for t in range(actual_tokens):
            for k in range(K):
                eidx = int(idx_np[t, k])
                expert_groups[eidx].append((t, k))

        # Sort by expert ID for sequential disk access
        sorted_experts = sorted(expert_groups.keys())
        unique_arr = np.array(sorted_experts, dtype=np.int32)
        num_unique = len(sorted_experts)

        # Build flat arrays
        total = sum(len(v) for v in expert_groups.values())
        tok_indices = np.empty(total, dtype=np.int32)
        exp_indices = np.empty(total, dtype=np.int32)
        k_positions = np.empty(total, dtype=np.int32)
        starts = np.empty(num_unique + 1, dtype=np.int32)

        pos = 0
        for gi, eidx in enumerate(sorted_experts):
            starts[gi] = pos
            for (t, k) in expert_groups[eidx]:
                tok_indices[pos] = t
                exp_indices[pos] = eidx
                k_positions[pos] = k
                pos += 1
        starts[num_unique] = pos

        # One C call for the entire layer
        x_contig = np.ascontiguousarray(np.array(x_view, copy=False))
        out_flat = np.zeros((actual_tokens * K * self._hidden_size,), dtype=np.float16)

        ret = self._cpu_lib._lib.bakan_cpu_expert_mlp_prefill_f16(
            self._cpu_engine,
            self._layer_idx,
            x_contig.ctypes.data_as(ctypes.c_void_p),
            actual_tokens, K, self._hidden_size,
            tok_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            exp_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            k_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            unique_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            num_unique,
            out_flat.ctypes.data_as(ctypes.c_void_p),
        )
        if ret != 0:
            raise RuntimeError(f"Prefill expert MLP failed layer {self._layer_idx}")
        out_np[:] = out_flat.reshape(actual_tokens, K, self._hidden_size)

    def _predict_and_prefetch(self, x_np):
        """Predict next layer's experts and prefetch in background.

        For single token (decode): predict top-K from last hidden state.
        For multi-token (prefill): predict top-K per token, union all unique.
        Background thread preads predicted experts into OS page cache.
        """
        if self._next_gate_np is None or self._layer_idx + 1 >= self._num_moe_layers:
            return

        try:
            if x_np.ndim == 1 or x_np.shape[0] == 1:
                # Single token: original path
                x_f32 = x_np[-1].astype(np.float32) if x_np.ndim > 1 else x_np.astype(np.float32)
                logits = x_f32 @ self._next_gate_np.T
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                predicted = np.argsort(probs)[-self._top_k:].tolist()
            else:
                # Multi-token prefill: predict for ALL tokens, union unique experts
                x_f32 = x_np.astype(np.float32)
                logits = x_f32 @ self._next_gate_np.T  # (num_tokens, num_experts)
                # Top-K per token
                top_k_per_token = np.argpartition(logits, -self._top_k, axis=1)[:, -self._top_k:]
                predicted = list(set(top_k_per_token.ravel().tolist()))
        except (ValueError, IndexError):
            return

        next_idx = self._layer_idx + 1
        _prefetch_state["predicted"][next_idx] = predicted

        # Start background prefetch into OS page cache
        t = threading.Thread(
            target=_prefetch_experts_to_page_cache,
            args=(self._packed_dir, next_idx, predicted, self._expert_size),
            daemon=True,
        )
        t.start()
        _prefetch_state["thread"] = t

    def _predict_and_prefetch_multi(self, x_np, lookahead=3):
        """Predict experts for the next N layers and prefetch all into page cache.

        Uses current hidden state to predict which experts each of the next
        N layers will need. Prefetches all in one background thread.
        More lookahead = more pages warm when those layers run.
        """
        if self._next_gate_np is None:
            return

        try:
            # Use mean hidden state across all tokens for prediction
            x_f32 = x_np.mean(axis=0).astype(np.float32) if x_np.ndim > 1 else x_np.astype(np.float32)

            # Collect all experts to prefetch across next N layers
            all_prefetch = []  # list of (layer_idx, [expert_ids])

            # We only have the NEXT layer's gate weights. For layers beyond that,
            # we use the same prediction (approximate but still warms relevant pages)
            logits = x_f32 @ self._next_gate_np.T
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()

            # For prefill, predict MORE experts per layer (union across tokens)
            if x_np.ndim > 1 and x_np.shape[0] > 1:
                all_logits = x_np.astype(np.float32) @ self._next_gate_np.T
                top_k_all = np.argpartition(all_logits, -self._top_k, axis=1)[:, -self._top_k:]
                predicted = list(set(top_k_all.ravel().tolist()))
            else:
                predicted = np.argsort(probs)[-self._top_k:].tolist()

            # Prefetch for next N layers (same predicted experts — approximate)
            for offset in range(1, lookahead + 1):
                next_idx = self._layer_idx + offset
                if next_idx < self._num_moe_layers:
                    all_prefetch.append((next_idx, predicted))

            if all_prefetch:
                def prefetch_multi():
                    for layer_idx, experts in all_prefetch:
                        _prefetch_experts_to_page_cache(
                            self._packed_dir, layer_idx, experts, self._expert_size
                        )

                t = threading.Thread(target=prefetch_multi, daemon=True)
                t.start()
                _prefetch_state["thread"] = t

        except (ValueError, IndexError):
            pass

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """Forward pass: expert MLP computed on CPU.

        Fast prefill: when num_tokens > 1, returns zeros (shared expert
        handles context encoding). CPU experts only run during decode (S=1).
        This turns 24,000 pread calls into 0 during prefill.
        """
        original_shape = indices.shape
        K = indices.shape[-1]
        num_tokens = x.shape[-2] if x.ndim > 2 else x.reshape(-1, self._hidden_size).shape[0]

        # FAST PREFILL: reduce expert computation during multi-token prefill.
        # Large models (122B+): skip all routed experts (shared expert sufficient).
        # Small models (35B): run experts on every 8th token only (8x faster).
        # No prefill shortcuts — full expert computation for every token.
        # Quality over speed. Use prompt caching for repeated queries.

        # === EXPERT COMPUTE PATH ===

        x_flat = x.reshape(-1, self._hidden_size)
        idx_i32 = indices.reshape(-1, K).astype(mx.int32)
        is_bf16 = x_flat.dtype == mx.bfloat16

        if is_bf16:
            x_view = x_flat.view(mx.uint16)
            mx.eval(x_view, idx_i32)
        else:
            x_f16 = x_flat.astype(mx.float16)
            mx.eval(x_f16, idx_i32)
            x_view = x_f16


        idx_np = np.array(idx_i32, copy=False).astype(np.int32)

        # Track speculation accuracy (decode only)
        if num_tokens == 1:
            predicted = _prefetch_state["predicted"].get(self._layer_idx)
            if predicted:
                actual = set(idx_np[0].tolist()) if idx_np.ndim > 1 else set(idx_np.tolist())
                _prefetch_state["hits"] += len(actual & set(predicted))
                _prefetch_state["misses"] += len(actual - set(predicted))

        x_np = np.array(x_view, copy=False)
        actual_tokens = x_np.shape[0]
        out_np = np.empty((actual_tokens, K, self._hidden_size), dtype=np.float16)

        # Predict + prefetch NEXT layer (decode only)
        if num_tokens == 1:
            self._predict_and_prefetch(x_np)

        # GPU prefill only works with 4-bit experts (Metal kernel is 4-bit only)
        # 3-bit experts detected by size: 1344 KB (35B) or similar
        is_4bit_experts = self._expert_size in (1769472, 5308416)  # 35B, 122B 4-bit sizes
        if actual_tokens > 1 and self._expert_size > 0 and is_4bit_experts:
            # PREFILL: GPU batched expert MLP (43x faster than CPU sequential)
            # Load each unique expert, dequantize on GPU, batched matmul
            self._gpu_prefill_experts(x_flat, idx_i32, actual_tokens, K, out_np)
        else:
            # DECODE: single token on CPU (fastest — no Metal dispatch overhead)
            mlp_func = self._cpu_lib.expert_mlp_bf16 if is_bf16 else self._cpu_lib.expert_mlp_f16
            for t in range(actual_tokens):
                out_np[t] = mlp_func(
                    self._cpu_engine, self._layer_idx,
                    np.ascontiguousarray(x_np[t]), idx_np[t], K,
                    hidden_size=self._hidden_size,
                )

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
    EXPERT_PATTERN = "switch_mlp"  # Qwen3.5; Gemma4 uses "switch_glu"

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

        # Step 1: Verify packed binary files exist (prefer 3-bit over 4-bit)
        cache_dir = self._model_cache_dir()
        packed_dir_3bit = os.path.join(cache_dir, "packed_3bit")
        packed_dir_4bit = os.path.join(cache_dir, "packed")
        if os.path.isdir(packed_dir_3bit):
            packed_dir = packed_dir_3bit
        else:
            packed_dir = packed_dir_4bit
        is_dense = not os.path.isdir(packed_dir)

        if is_dense:
            # Dense model (non-MoE) — load all weights, skip SEM
            self._model, self._tokenizer = load(self.model_path)
            self._log("Dense model loaded (all weights in memory)")
            self._cpu_lib = None
            self._cpu_engine = None
            self._ready = True
            return

        # Step 2: Load model with lazy weights (MoE path — unchanged)
        self._model, self._tokenizer = load(self.model_path, lazy=True)
        self._log("Model structure loaded (lazy)")

        # Step 2b: Auto-detect expert pattern (Qwen=switch_mlp, Gemma4=switch_glu)
        flat_keys = [k for k, _ in tree_flatten(self._model.parameters())]
        if any("switch_glu" in k for k in flat_keys):
            self.EXPERT_PATTERN = "switch_glu"
            self._log("Expert pattern: switch_glu (Gemma 4)")
        else:
            self.EXPERT_PATTERN = "switch_mlp"

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

        # Step 5: Load shared (non-expert) parameters to GPU
        # Check for 3-bit shared layers first (21% faster, 22% smaller)
        model_name = self.model_path.split("/")[-1]
        base_name = re.sub(r"-\d+bit$", "", model_name)
        shared_3bit_path = os.path.join(
            os.path.expanduser("~/.kandiga/models"),
            base_name + "-3bit-shared",
            "shared_3bit.safetensors",
        )
        if os.path.isfile(shared_3bit_path):
            # Replace 4-bit lazy weights with 3-bit BEFORE evaluating
            from mlx.utils import tree_unflatten
            weights_3bit = mx.load(shared_3bit_path)
            shared_3bit = {
                k: v for k, v in weights_3bit.items()
                if self.EXPERT_PATTERN not in k
            }
            # Convert Embedding → QuantizedEmbedding if 3-bit file has quantized embedding
            for name, module in self._model.named_modules():
                if isinstance(module, nn.Embedding) and f"{name}.scales" in shared_3bit:
                    qembed = nn.QuantizedEmbedding(
                        num_embeddings=module.weight.shape[0],
                        dims=module.weight.shape[1],
                        group_size=64, bits=3,
                    )
                    parts = name.split(".")
                    parent = self._model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], qembed)
            self._model.update(tree_unflatten(list(shared_3bit.items())))
            # Update bits on every quantized module whose weights were replaced
            # Detect actual bits from weight vs scales shape ratio
            replaced_prefixes = {}
            for k in shared_3bit:
                if k.endswith(".weight"):
                    prefix = k[:-len(".weight")]
                    replaced_prefixes[prefix] = shared_3bit[k]
            bits_updated = 0
            for name, module in self._model.named_modules():
                if name in replaced_prefixes and hasattr(module, "bits"):
                    w = replaced_prefixes[name]
                    s_key = f"{name}.scales"
                    if s_key in shared_3bit:
                        s = shared_3bit[s_key]
                        # Infer bits from weight cols vs scales cols ratio
                        # 4-bit: weight_cols = scales_cols * group_size / 8
                        # 3-bit: weight_cols = scales_cols * group_size * 3 / 32
                        w_cols = w.shape[-1]
                        s_cols = s.shape[-1]
                        expected_4bit = s_cols * module.group_size // 8
                        expected_3bit = s_cols * module.group_size * 3 // 32
                        if w_cols == expected_3bit:
                            module.bits = 3
                        elif w_cols == expected_4bit:
                            module.bits = 4
                        # else: leave as-is
                    bits_updated += 1
            del weights_3bit, shared_3bit
            gc.collect()
            self._log(f"3-bit shared layers applied ({bits_updated} modules, {os.path.getsize(shared_3bit_path) / 1e9:.2f}GB)")
        else:
            self._log("No 3-bit shared layers found, using 4-bit")

        # Evaluate shared parameters (either 3-bit or 4-bit)
        flat = tree_flatten(self._model.parameters())
        shared = [v for k, v in flat if self.EXPERT_PATTERN not in k]
        if shared:
            mx.eval(*shared)
        self._log("Shared parameters loaded to GPU")

        # Step 5a: Free lazy expert weight mmap pages from safetensors.
        # Expert weights come from our packed binary files (pread), not safetensors.
        # Freeing these mmap references gives the OS more page cache for expert pread.
        expert_params = [(k, v) for k, v in flat if self.EXPERT_PATTERN in k]
        if expert_params:
            # Delete lazy weight references from the model's parameter tree
            from mlx.utils import tree_unflatten
            zeroed = {k: mx.zeros((1,), dtype=mx.float16) for k, v in expert_params}
            self._model.update(tree_unflatten(list(zeroed.items())))
            gc.collect()
            self._log(f"Freed {len(expert_params)} lazy expert weight refs (page cache freed)")

        # Step 5b: Apply ZMLX fused kernels — deltanet + norms
        try:
            from zmlx.patch import patch as zmlx_patch
            zmlx_patch(
                self._model,
                patterns=["deltanet", "rmsnorm", "softmax"],
            )
            self._log("ZMLX fused kernels: deltanet + rmsnorm + softmax")
        except ImportError:
            self._log("ZMLX not installed (pip install zmlx for faster attention)")
        except Exception as e:
            self._log(f"ZMLX patch failed: {e}")

        # Step 5d: Install TurboQuant KV cache compression
        try:
            from kandiga.kv_compress import install_kv_compression
            n = install_kv_compression(self._model)
            self._log(f"TurboQuant KV compression active ({n} layers)")
        except Exception as e:
            self._log(f"TurboQuant failed: {e}")

        # Step 6: Count MoE layers and initialize CPU engine
        # Detect MoE structure: Qwen = layer.mlp.switch_mlp, Gemma4 = layer.experts.switch_glu
        layers = _find_layers(self._model)
        is_gemma_moe = any(hasattr(layer, "experts") and hasattr(getattr(layer, "experts", None), "switch_glu") for layer in layers)
        moe_count = 0
        for layer in layers:
            if is_gemma_moe:
                if hasattr(layer, "experts") and hasattr(layer.experts, "switch_glu"):
                    moe_count += 1
            else:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                    moe_count += 1

        self._cpu_engine = self._cpu_lib.init(packed_dir, moe_count)
        self._log(f"CPU engine initialized ({moe_count} MoE layers)")

        # Step 7: Extract router gates for cross-layer speculation
        top_k = 8
        try:
            lm_args = getattr(getattr(self._model, 'language_model', self._model), 'args', None)
            if lm_args:
                if hasattr(lm_args, 'num_experts_per_tok'):
                    top_k = lm_args.num_experts_per_tok
                elif hasattr(lm_args, 'top_k_experts'):
                    top_k = lm_args.top_k_experts
        except Exception:
            pass

        # Detect expert_size from first packed binary header
        expert_size = 0
        first_bin = os.path.join(packed_dir, "layer_00.bin")
        if os.path.exists(first_bin):
            import struct
            with open(first_bin, "rb") as f:
                hdr = f.read(20)
                if hdr[:4] == b"BKEX":
                    expert_size = struct.unpack("<Q", hdr[12:20])[0]

        # Collect MoE layers and their router gate modules
        moe_layers_info = []
        for i, layer in enumerate(layers):
            if is_gemma_moe:
                if hasattr(layer, "experts") and hasattr(layer.experts, "switch_glu"):
                    gate = getattr(getattr(layer, 'router', None), 'proj', None)
                    moe_layers_info.append((i, layer, gate))
            else:
                mlp = layer.mlp if hasattr(layer, "mlp") else None
                if mlp is not None and hasattr(mlp, "switch_mlp"):
                    gate = getattr(mlp, 'gate', None)
                    moe_layers_info.append((i, layer, gate))

        # Install CPU SwitchGLU wrappers with speculation
        _prefetch_state["predicted"].clear()
        _prefetch_state["hits"] = 0
        _prefetch_state["misses"] = 0

        moe_idx = 0
        for pos, (layer_i, layer, _) in enumerate(moe_layers_info):
            # Get NEXT layer's router gate for speculation
            next_gate = None
            if pos + 1 < len(moe_layers_info):
                next_gate = moe_layers_info[pos + 1][2]

            if is_gemma_moe:
                original_expert = layer.experts.switch_glu
            else:
                original_expert = layer.mlp.switch_mlp

            wrapper = _CPUSwitchGLU(
                original=original_expert,
                layer_idx=moe_idx,
                cpu_lib=self._cpu_lib,
                cpu_engine=self._cpu_engine,
                hidden_size=hidden_size,
                next_gate_weights=next_gate,
                top_k=top_k if not self.fast_mode else 4,
                packed_dir=packed_dir,
                expert_size=expert_size,
                num_moe_layers=moe_count,
            )
            if is_gemma_moe:
                layer.experts.switch_glu = wrapper
            else:
                layer.mlp.switch_mlp = wrapper
            moe_idx += 1

        self._log(f"Expert speculation enabled ({moe_count} layers, top_k={top_k})")

        # Step 8: Apply fast mode (K=4) if requested
        actual_k = top_k
        if self.fast_mode:
            actual_k = max(top_k // 2, 4)
            for layer in layers:
                # Qwen: layer.mlp.top_k, Gemma4: layer.experts has no top_k (handled by router)
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "top_k"):
                    layer.mlp.top_k = actual_k
                # Also check wrapper directly
                if is_gemma_moe and hasattr(layer, "experts") and hasattr(layer.experts, "switch_glu"):
                    sw = layer.experts.switch_glu
                    if hasattr(sw, "_top_k"):
                        sw._top_k = actual_k
            self._log(f"Fast mode: K={actual_k} experts")
        else:
            self._log(f"Quality mode: K={top_k} experts")

        # Step 9: Expert layer skipping for large models (turbo mode)
        # Skip routed experts every 3rd layer — shared expert handles those layers.
        # Reduces mx.eval syncs and disk I/O by 33%. Quality stays high.
        if is_large:
            skip_count = 0
            for i, layer in enumerate(layers):
                # Find the expert module (Qwen: mlp.switch_mlp, Gemma: experts.switch_glu)
                sw = None
                parent = None
                attr_name = None
                if is_gemma_moe and hasattr(layer, 'experts') and hasattr(layer.experts, 'switch_glu'):
                    sw = layer.experts.switch_glu
                    parent = layer.experts
                    attr_name = 'switch_glu'
                elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'switch_mlp'):
                    sw = layer.mlp.switch_mlp
                    parent = layer.mlp
                    attr_name = 'switch_mlp'

                if sw and hasattr(sw, '_cpu_lib') and i % 3 == 0:
                    hs = sw._hidden_size
                    class _SkipExpert(nn.Module):
                        def __init__(self, orig, hidden):
                            super().__init__()
                            self.original = orig
                            self._h = hidden
                        def __call__(self, x, indices):
                            return mx.zeros(list(indices.shape) + [self._h], dtype=x.dtype)
                    setattr(parent, attr_name, _SkipExpert(sw, hs))
                    skip_count += 1
            if skip_count > 0:
                self._log(f"Layer skipping: {skip_count}/{moe_count} layers use shared expert only")

        # Step 10: Pre-warm page cache via background thread.
        # Reads a sample of expert pages into OS page cache without generating.
        if expert_size > 0:
            self._prewarm_cache(packed_dir, moe_count, expert_size, hidden_size)

        self._log(f"Engine ready ({moe_idx} MoE layers, CPU expert wrappers installed)")
        self._ready = True

    def _prewarm_cache(self, packed_dir: str, num_layers: int,
                        expert_size: int, hidden_size: int):
        """Pre-read the most popular experts into OS page cache.

        Runs in background thread during model setup. By the time the
        first token generates, many experts are already in page cache.
        Reduces cold-start latency by 50-70%.
        """
        import threading

        def _warmup():
            import struct
            # Read a sample of experts from each layer
            # Focus on experts 0-15 (often the most popular in uniform routing)
            experts_per_layer = 8  # warm 8 experts per layer
            for layer_idx in range(num_layers):
                path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.bin")
                try:
                    fd = os.open(path, os.O_RDONLY)
                    for eidx in range(experts_per_layer):
                        offset = 4096 + eidx * expert_size
                        os.pread(fd, expert_size, offset)
                    os.close(fd)
                except OSError:
                    pass

        t = threading.Thread(target=_warmup, daemon=True)
        t.start()
        self._log(f"Page cache pre-warming started (background)")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temp: float = 0.0,
        stream: bool = True,
    ):
        """Generate a response (single turn, no history).

        If stream=True, yields tokens one at a time.
        If stream=False, returns the full response string.
        """
        if not self._ready:
            self.load()

        messages = [{"role": "user", "content": prompt}]

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

    # ------------------------------------------------------------------
    # Persistent KV cache for multi-turn conversations
    # ------------------------------------------------------------------

    def start_session(self):
        """Start a new conversation session with persistent KV cache.

        The KV cache is maintained between calls to session_generate().
        The model only processes NEW tokens each turn — previous turns
        are already cached. This eliminates re-processing the entire
        conversation history on every message.

        Call end_session() or start_session() again to reset.
        """
        if not self._ready:
            self.load()
        self._session_cache = self._model.make_cache()
        self._session_history = []
        self._session_tokens_fed = []  # exact tokens that went through the model

    def end_session(self):
        """End the conversation session and free the KV cache."""
        self._session_cache = None
        self._session_history = []
        self._session_tokens_fed = []

    def save_session(self, path: str):
        """Save the conversation state to disk.

        Saves the KV cache + history so the conversation can be
        resumed later with load_session(). The model reads the
        document once, user saves, closes app, reopens next day,
        loads — picks up exactly where they left off.

        File size: ~2-10MB depending on conversation length.
        """
        import json
        if not hasattr(self, '_session_cache') or self._session_cache is None:
            raise RuntimeError("No active session to save")

        # Save KV cache arrays (preserving exact shapes and dtypes)
        save_dict = {}
        dtype_map = {}
        layer_info = {}  # cache type + array count per layer
        for i, layer_cache in enumerate(self._session_cache):
            # Unwrap CompressedArraysCache if present
            raw_cache = layer_cache
            if hasattr(layer_cache, '_original'):
                raw_cache = layer_cache._original

            # Determine cache type and extract state
            cache_type = type(raw_cache).__name__
            if cache_type == 'KVCache':
                # KVCache.state getter crashes on fresh cache (keys=None),
                # so access keys/values directly
                arrays = [raw_cache.keys, raw_cache.values]
                layer_info[str(i)] = {'type': 'KVCache', 'n': 2,
                                      'offset': raw_cache.offset}
            elif cache_type == 'ArraysCache':
                arrays = raw_cache.cache
                layer_info[str(i)] = {'type': 'ArraysCache',
                                      'n': len(arrays)}
            else:
                continue

            for j, arr in enumerate(arrays):
                if arr is not None:
                    # For KVCache, trim to offset (don't save padding)
                    if cache_type == 'KVCache' and raw_cache.offset < arr.shape[2]:
                        arr = arr[..., :raw_cache.offset, :]
                    mx.eval(arr)
                    key = f"cache_{i}_{j}"
                    if arr.dtype == mx.float32:
                        np_arr = np.array(arr, copy=False)
                    else:
                        np_arr = np.array(arr.view(mx.uint16), copy=False)
                    save_dict[key] = np_arr
                    dtype_map[key] = {
                        'shape': list(arr.shape),
                        'dtype': str(arr.dtype),
                    }

        np.savez_compressed(path, **save_dict)

        # Save metadata alongside
        meta_path = path.replace('.npz', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'history': self._session_history,
                'tokens_fed': self._session_tokens_fed,
                'model': self.model_path,
                'dtype_map': dtype_map,
                'layer_info': layer_info,
            }, f)

        size_kb = os.path.getsize(path) / 1024
        self._log(f"Session saved: {path} ({size_kb:.0f}KB)")

    def load_session(self, path: str):
        """Load a conversation state from disk.

        Restores the KV cache + history from a previous save_session().
        The model picks up the conversation exactly where it left off.
        Loading is near-instant (<0.1s) regardless of conversation length.
        """
        import json
        if not self._ready:
            self.load()

        # Load metadata
        meta_path = path.replace('.npz', '_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)

        if meta['model'] != self.model_path:
            raise ValueError(f"Session was for {meta['model']}, not {self.model_path}")

        # Create fresh cache
        self._session_cache = self._model.make_cache()
        self._session_history = meta['history']
        self._session_tokens_fed = meta['tokens_fed']
        dtype_map = meta.get('dtype_map', {})

        # Load arrays with exact shapes and dtypes
        data = np.load(path)
        layer_info = meta.get('layer_info', {})

        for i, layer_cache in enumerate(self._session_cache):
            raw_cache = layer_cache
            if hasattr(layer_cache, '_original'):
                raw_cache = layer_cache._original

            # Use saved layer_info to know array count (avoids crashing
            # KVCache.state getter on fresh cache where keys=None)
            info_i = layer_info.get(str(i))
            if info_i is None:
                continue
            num_arrays = info_i['n']

            state = []
            for j in range(num_arrays):
                key = f"cache_{i}_{j}"
                if key in data and key in dtype_map:
                    info = dtype_map[key]
                    orig_shape = tuple(info['shape'])
                    orig_dtype = info['dtype']
                    np_arr = data[key]
                    mx_arr = mx.array(np_arr)
                    if 'float32' in orig_dtype:
                        mx_arr = mx_arr.reshape(orig_shape)
                    else:
                        dt = mx.bfloat16 if 'bfloat' in orig_dtype else mx.float16
                        mx_arr = mx_arr.view(dt).reshape(orig_shape)
                    state.append(mx_arr)
                else:
                    state.append(None)

            # For KVCache: set keys/values/offset directly (state setter
            # handles this, and offset is derived from keys.shape[2])
            if any(s is not None for s in state):
                raw_cache.state = state

        size_kb = os.path.getsize(path) / 1024
        self._log(f"Session loaded: {path} ({size_kb:.0f}KB, {len(self._session_history)} messages)")

    def session_generate(
        self,
        user_message: str,
        max_tokens: int = 2048,
        temp: float = 0.0,
    ):
        """Generate a response in an active session (persistent KV cache).

        Only processes the NEW user message through the model.
        Previous conversation turns are already in the KV cache.
        TTFT stays constant regardless of conversation length.

        Yields tokens one at a time (streaming).
        """
        if not self._ready:
            self.load()
        if not hasattr(self, '_session_cache') or self._session_cache is None:
            self.start_session()

        # Build the FULL conversation including new message
        self._session_history.append({"role": "user", "content": user_message})

        try:
            full_text = self._tokenizer.apply_chat_template(
                self._session_history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            full_text = self._tokenizer.apply_chat_template(
                self._session_history,
                tokenize=False,
                add_generation_prompt=True,
            )

        full_tokens = self._tokenizer.encode(full_text)

        # Only process tokens that the KV cache hasn't seen yet
        num_cached = len(self._session_tokens_fed)
        new_tokens = full_tokens[num_cached:]
        if not new_tokens:
            return

        new_input = mx.array(new_tokens).reshape(1, -1)

        # Prefill new tokens (KV cache extends, doesn't restart)
        out = self._model(new_input, cache=self._session_cache)
        mx.eval(out)

        self._session_tokens_fed.extend(new_tokens)

        # Decode loop: generate tokens one at a time
        sampler = make_sampler(temp=temp)
        logits = out[0, -1]

        eos_ids = set()
        if hasattr(self._tokenizer, 'eos_token_id'):
            eid = self._tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids = set(eid)
            elif eid is not None:
                eos_ids = {eid}
        # Qwen3.5: also stop on <|endoftext|> and <|im_start|> (new turn = stop)
        for stop_tok in ['<|endoftext|>', '<|im_start|>']:
            try:
                ids = self._tokenizer.encode(stop_tok)
                if len(ids) == 1:
                    eos_ids.add(ids[0])
            except:
                pass

        response_tokens = []
        for _ in range(max_tokens):
            token = mx.argmax(logits) if temp == 0 else sampler(mx.expand_dims(logits, 0))[0]
            mx.eval(token)
            tid = token.item()

            if tid in eos_ids:
                break

            response_tokens.append(tid)
            text = self._tokenizer.decode([tid])

            # Filter thinking tags
            if "<think>" in text or "</think>" in text:
                continue
            yield text

            # Next token through model (extends cache)
            out = self._model(token.reshape(1, 1), cache=self._session_cache)
            mx.eval(out)
            logits = out[0, -1]
            self._session_tokens_fed.append(tid)

        # Add assistant response to history for future turns
        full_response = self._tokenizer.decode(response_tokens)
        self._session_history.append({"role": "assistant", "content": full_response})

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
