"""Metal kernel runner for TQ3 inference on Apple Silicon.

Compiles and runs the TQ3 Metal compute shader for fused
WHT + centroid matrix-vector products.

The key optimization: activations are pre-rotated into WHT domain ONCE,
then all weight rows compute dot products directly against 3-bit centroids.
No full dequantization needed during inference.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


METAL_SRC = os.path.join(os.path.dirname(__file__), "tq3_metal.metal")
DYLIB_PATH = os.path.join(os.path.dirname(__file__), "libtq3_metal.dylib")


def compile_metal_kernel(force: bool = False) -> bool:
    """Compile the TQ3 Metal kernel into a dylib.

    Uses MLX's metal compiler pipeline or falls back to xcrun.
    Returns True if compilation succeeded.
    """
    if os.path.exists(DYLIB_PATH) and not force:
        # Check if source is newer
        if os.path.getmtime(DYLIB_PATH) >= os.path.getmtime(METAL_SRC):
            return True

    try:
        # Compile Metal → metallib → dylib
        # Step 1: Metal to AIR
        air_path = METAL_SRC.replace(".metal", ".air")
        result = subprocess.run([
            "xcrun", "-sdk", "macosx", "metal",
            "-c", METAL_SRC,
            "-o", air_path,
            "-std=metal3.0",
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Metal compilation failed: {result.stderr}")
            return False

        # Step 2: AIR to metallib
        metallib_path = METAL_SRC.replace(".metal", ".metallib")
        result = subprocess.run([
            "xcrun", "-sdk", "macosx", "metallib",
            air_path,
            "-o", metallib_path,
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Metallib creation failed: {result.stderr}")
            return False

        # Clean up AIR
        os.remove(air_path)

        return True

    except FileNotFoundError:
        print("xcrun not found — Xcode Command Line Tools required")
        return False
    except Exception as e:
        print(f"Metal compilation error: {e}")
        return False


class TQ3MetalEngine:
    """Runs TQ3 inference using Metal compute shaders via MLX custom kernels."""

    def __init__(self):
        self._compiled = False

    def ensure_compiled(self) -> bool:
        if self._compiled:
            return True
        self._compiled = compile_metal_kernel()
        return self._compiled

    def matvec_tq3(
        self,
        weights_data: bytes,
        activation: np.ndarray,
        out_features: int,
        in_features: int,
    ) -> np.ndarray:
        """Compute weights @ activation using TQ3 Metal kernel.

        Args:
            weights_data: TQ3-packed weight bytes
            activation: float32 activation vector (in_features,)
            out_features: number of output features (rows)
            in_features: number of input features (cols, must be multiple of 32)

        Returns:
            float32 output vector (out_features,)
        """
        if not HAS_MLX:
            # Fallback: dequantize and compute in numpy
            return self._fallback_matvec(weights_data, activation, out_features, in_features)

        # Use MLX for the computation
        # For now, dequantize to MLX array and use mx.matmul
        # TODO: Replace with custom Metal kernel dispatch
        from kandiga.tq3.quantize import dequantize_tensor
        W = dequantize_tensor(weights_data, (out_features, in_features),
                              out_features * in_features // 32)
        W_mx = mx.array(W)
        act_mx = mx.array(activation)
        out = act_mx @ W_mx.T
        mx.eval(out)
        return np.array(out)

    def _fallback_matvec(self, weights_data, activation, out_features, in_features):
        """Pure numpy fallback."""
        from kandiga.tq3.quantize import dequantize_tensor
        W = dequantize_tensor(weights_data, (out_features, in_features),
                              out_features * in_features // 32)
        return activation @ W.T

    def matvec_tq3_fused(
        self,
        weights_data: bytes,
        activation: np.ndarray,
        out_features: int,
        in_features: int,
    ) -> np.ndarray:
        """Fused TQ3 matvec — pre-rotate activation, compute in WHT domain.

        This is the fast path: activation is rotated into WHT domain ONCE,
        then dot products are computed directly against centroids.
        No full weight dequantization.
        """
        from kandiga.tq3.quantize import CENTROIDS, SIGNS, BLOCK_SIZE

        nblocks_per_row = in_features // BLOCK_SIZE
        num_blocks = out_features * nblocks_per_row

        # Pre-rotate activation into WHT domain (one-time cost)
        act_rotated = self._rotate_activation(activation, in_features)

        # Compute dot products in WHT domain
        output = np.zeros(out_features, dtype=np.float32)

        # Parse all blocks
        for row in range(out_features):
            dot = 0.0
            for b in range(nblocks_per_row):
                block_idx = row * nblocks_per_row + b
                offset = block_idx * 16

                d0 = np.frombuffer(weights_data[offset:offset+2], dtype=np.float16).astype(np.float32)[0]
                d1 = np.frombuffer(weights_data[offset+2:offset+4], dtype=np.float16).astype(np.float32)[0]
                qs = weights_data[offset+4:offset+16]

                act_block = act_rotated[b * 32: (b + 1) * 32]

                # Unpack and dot product
                for g in range(4):
                    packed = qs[g*3] | (qs[g*3+1] << 8) | (qs[g*3+2] << 16)
                    for r in range(8):
                        idx_pos = g * 8 + r
                        qi = (packed >> (3 * r)) & 7
                        scale = d0 if idx_pos < 16 else d1
                        w_val = CENTROIDS[qi] * scale
                        dot += w_val * act_block[idx_pos]

            output[row] = dot

        return output

    def _rotate_activation(self, activation: np.ndarray, n: int) -> np.ndarray:
        """Pre-rotate activation into WHT domain."""
        from kandiga.tq3.quantize import _wht_forward_batch, BLOCK_SIZE
        blocks = activation.reshape(-1, BLOCK_SIZE)
        return _wht_forward_batch(blocks).flatten()
