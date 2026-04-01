"""Fused TQ3 GEMV kernel — matrix-vector product directly on packed 3-bit data.

No dequantization. The Metal kernel reads packed TQ3 blocks and computes
dot products directly against centroids in the WHT domain.

Based on:
  - exllama-ish (Infatoshi): mx.fast.metal_kernel for packed 4-bit GEMV
  - turboquant-mlx (sharpner): Metal kernels for quantized operations
  - llama.cpp-tq3 (turbo-tan): TQ3 native dot product kernel

Usage:
    from kandiga.tq3.fused_kernel import tq3_gemv

    # weight_packed: (N, num_blocks_per_row * 5) uint32 — packed TQ3 blocks
    # scales: (N, num_blocks_per_row * 2) float16 — d0,d1 per block
    # x: (K,) float16 — input vector
    output = tq3_gemv(x, weight_packed, scales, K, N)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from kandiga.tq3.quantize import CENTROIDS, SIGNS, BOUNDARIES


# Lloyd-Max centroids as string for Metal source
_CENTROIDS_STR = ", ".join(f"{c:.6f}f" for c in CENTROIDS)
_SIGNS_STR = ", ".join(f"{s:.0f}.0f" for s in SIGNS)


def _build_tq3_gemv_kernel(K: int, N: int, block_size: int = 32):
    """Build fused TQ3 GEMV kernel for specific dimensions.

    The kernel:
    1. Reads packed 3-bit indices from uint32 words
    2. Looks up centroids
    3. Applies per-half-block scale + mean
    4. Multiplies with (pre-rotated) input
    5. Reduces via SIMD

    One thread per output row. Each thread processes all blocks in its row.
    """
    blocks_per_row = K // block_size
    # Each TQ3_1S_shift block: 4x fp16 (d0,d1,m0,m1) + 12 bytes packed = 20 bytes
    # Store as: scales array (4 fp16 per block) + indices array (3 uint32 per block)

    source = f"""
    uint row = thread_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint simd_lane = thread_index_in_simdgroup;

    const uint K = {K};
    const uint N = {N};
    const uint BLOCKS_PER_ROW = {blocks_per_row};

    // Lloyd-Max centroids
    const float CENTROIDS[8] = {{{_CENTROIDS_STR}}};

    if (row >= N) return;

    float acc = 0.0f;

    // Process blocks assigned to this thread
    for (uint b = tid; b < BLOCKS_PER_ROW; b += threads_per_threadgroup.x) {{
        // Load scales: d0, d1, m0, m1 (4 fp16 values)
        uint scale_idx = row * BLOCKS_PER_ROW * 4 + b * 4;
        float d0 = float(scales[scale_idx]);
        float d1 = float(scales[scale_idx + 1]);
        float m0 = float(scales[scale_idx + 2]);
        float m1 = float(scales[scale_idx + 3]);

        // Load packed indices: 4 uint32 words, one per group of 8 indices
        // Each word has 24 valid bits (8 * 3 bits)
        uint pack_idx = row * BLOCKS_PER_ROW * 4 + b * 4;
        uint base = b * 32;

        // Group 0 (indices 0-7, scale=d0, mean=m0)
        uint32_t pk = packed[pack_idx];
        for (uint r = 0; r < 8; r++) {{
            uint idx = (pk >> (r * 3)) & 7;
            float w = CENTROIDS[idx] * d0 + m0;
            acc += w * float(input[base + r]);
        }}

        // Group 1 (indices 8-15, scale=d0, mean=m0)
        pk = packed[pack_idx + 1];
        for (uint r = 0; r < 8; r++) {{
            uint idx = (pk >> (r * 3)) & 7;
            float w = CENTROIDS[idx] * d0 + m0;
            acc += w * float(input[base + 8 + r]);
        }}

        // Group 2 (indices 16-23, scale=d1, mean=m1)
        pk = packed[pack_idx + 2];
        for (uint r = 0; r < 8; r++) {{
            uint idx = (pk >> (r * 3)) & 7;
            float w = CENTROIDS[idx] * d1 + m1;
            acc += w * float(input[base + 16 + r]);
        }}

        // Group 3 (indices 24-31, scale=d1, mean=m1)
        pk = packed[pack_idx + 3];
        for (uint r = 0; r < 8; r++) {{
            uint idx = (pk >> (r * 3)) & 7;
            float w = CENTROIDS[idx] * d1 + m1;
            acc += w * float(input[base + 24 + r]);
        }}
    }}

    // SIMD reduction
    acc = simd_sum(acc);

    threadgroup float partial[8];
    uint simd_group = simdgroup_index_in_threadgroup;
    if (simd_lane == 0) {{
        partial[simd_group] = acc;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < 8) {{
        acc = partial[simd_lane];
        acc = simd_sum(acc);
        if (simd_lane == 0) {{
            output[row] = half(acc);
        }}
    }}
    """

    return mx.fast.metal_kernel(
        name=f"tq3_gemv_{K}_{N}",
        input_names=["input", "packed", "scales"],
        output_names=["output"],
        source=source,
    )


# Cache compiled kernels
_kernel_cache = {}


def tq3_gemv(
    x: mx.array,
    packed: mx.array,
    scales: mx.array,
    K: int,
    N: int,
) -> mx.array:
    """Fused TQ3 GEMV: y = x @ W^T where W is TQ3-quantized.

    Args:
        x: Input vector (K,) or (B, K) float16
        packed: Packed 3-bit indices (N, blocks_per_row * 3) uint32
        scales: Per-block scales+means (N, blocks_per_row * 4) float16
        K: Input dimension
        N: Output dimension

    Returns:
        (N,) or (B, N) float16
    """
    key = (K, N)
    if key not in _kernel_cache:
        _kernel_cache[key] = _build_tq3_gemv_kernel(K, N)

    kernel = _kernel_cache[key]

    # Handle batched input
    if x.ndim == 2:
        B = x.shape[0]
        results = []
        for i in range(B):
            out = kernel(
                inputs=[x[i], packed, scales],
                grid=(N, 1, 1),
                threadgroup=(min(256, N), 1, 1),
                output_shapes=[(N,)],
                output_dtypes=[mx.float16],
            )[0]
            results.append(out)
        return mx.stack(results)

    return kernel(
        inputs=[x, packed, scales],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.float16],
    )[0]


def pack_tq3_for_metal(weight_data: bytes, shape: tuple, nblocks: int) -> Tuple[mx.array, mx.array]:
    """Convert TQ3 block data into Metal-friendly packed format.

    Splits the 20-byte blocks into:
    - scales: (N, blocks_per_row * 4) float16 — d0,d1,m0,m1
    - packed: (N, blocks_per_row * 3) uint32 — 3 uint32 words of packed indices

    Returns:
        (packed, scales) ready for tq3_gemv()
    """
    out_features = shape[0]
    in_features = shape[1] if len(shape) > 1 else 1
    blocks_per_row = nblocks // out_features

    BLOCK_BYTES = 20  # d0(2) + d1(2) + m0(2) + m1(2) + qs(12)

    all_scales = np.zeros((out_features, blocks_per_row * 4), dtype=np.float16)
    # 4 uint32 words per block (one per group of 8 indices)
    all_packed = np.zeros((out_features, blocks_per_row * 4), dtype=np.uint32)

    # Use 4 uint32 words per block (one per group of 8 indices)
    all_packed = np.zeros((out_features, blocks_per_row * 4), dtype=np.uint32)

    for row in range(out_features):
        for b in range(blocks_per_row):
            block_idx = row * blocks_per_row + b
            offset = block_idx * BLOCK_BYTES

            d0 = np.frombuffer(weight_data[offset:offset+2], dtype=np.float16)[0]
            d1 = np.frombuffer(weight_data[offset+2:offset+4], dtype=np.float16)[0]
            m0 = np.frombuffer(weight_data[offset+4:offset+6], dtype=np.float16)[0]
            m1 = np.frombuffer(weight_data[offset+6:offset+8], dtype=np.float16)[0]
            all_scales[row, b*4:b*4+4] = [d0, d1, m0, m1]

            qs = weight_data[offset+8:offset+20]
            # 4 groups of 3 bytes → 4 uint32 words (lower 24 bits each)
            for g in range(4):
                word = int(qs[g*3]) | (int(qs[g*3+1]) << 8) | (int(qs[g*3+2]) << 16)
                all_packed[row, b*4 + g] = word

    return mx.array(all_packed), mx.array(all_scales)
