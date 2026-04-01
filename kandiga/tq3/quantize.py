"""TQ3 weight quantization — 3.5 bits per weight.

Port of turbo-tan/llama.cpp-tq3 quantization to Python/NumPy.
This runs offline to convert model weights. Not speed-critical.

Block format (TQ3_1S — 4.0 bpw, best quality):
  - 2x fp16 scale factors (d0, d1) — 4 bytes
  - 12 bytes of packed 3-bit indices
  - Total: 16 bytes per 32 weights = 4.0 bits per weight

Block format (TQ3_0 — 3.5 bpw, smallest):
  - 1x fp16 scale factor (d) — 2 bytes
  - 12 bytes of packed 3-bit indices
  - Total: 14 bytes per 32 weights = 3.5 bits per weight
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Lloyd-Max optimal centroids for standard normal distribution (8 levels)
# Computed via 178 iterations of Lloyd's algorithm
# Theoretical MSE = 0.03455
CENTROIDS = np.array([
    -1.996684, -1.291398, -0.740341, -0.247508,
     0.230106,  0.725222,  1.277503,  1.988943,
], dtype=np.float32)

# Decision boundaries (midpoints between adjacent centroids)
BOUNDARIES = np.array([
    -1.644041, -1.015870, -0.493924, -0.008701,
     0.477664,  1.001362,  1.633223,
], dtype=np.float32)

# Deterministic sign flip pattern for randomized Hadamard transform
# Generated from golden ratio hash: sign[i] = ((i * 0x9E3779B9) >> 31) ? -1 : +1
SIGNS = np.array([
    +1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, -1, -1, +1,
    -1, +1, +1, -1, +1, -1, -1, +1,
], dtype=np.float32)

BLOCK_SIZE = 32


def _wht_forward(x: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard Transform with sign flips (forward).

    Distributes information uniformly across the block so that
    scalar quantization preserves most of the signal.
    """
    out = x.copy() * SIGNS

    # In-place butterfly (5 stages for n=32)
    step = 1
    while step < 32:
        for i in range(0, 32, step * 2):
            for j in range(i, i + step):
                a, b = out[j], out[j + step]
                out[j] = a + b
                out[j + step] = a - b
        step *= 2

    return out / math.sqrt(32.0)


def _wht_inverse(x: np.ndarray) -> np.ndarray:
    """Inverse Walsh-Hadamard Transform."""
    out = x.copy()

    step = 1
    while step < 32:
        for i in range(0, 32, step * 2):
            for j in range(i, i + step):
                a, b = out[j], out[j + step]
                out[j] = a + b
                out[j + step] = a - b
        step *= 2

    return out * SIGNS / math.sqrt(32.0)


def _choose_index(v: float) -> int:
    """Map a rotated value to the nearest Lloyd-Max centroid index (0-7)."""
    idx = 0
    for b in range(7):
        if v > BOUNDARIES[b]:
            idx = b + 1
    return idx


def _pack_indices(indices: np.ndarray) -> np.ndarray:
    """Pack 32 3-bit indices into 12 bytes."""
    packed = np.zeros(12, dtype=np.uint8)
    for g in range(4):
        idx = indices[g * 8: g * 8 + 8]
        qp_offset = g * 3
        packed[qp_offset + 0] = (int(idx[0]) | (int(idx[1]) << 3) | (int(idx[2]) << 6)) & 0xFF
        packed[qp_offset + 1] = ((int(idx[2]) >> 2) | (int(idx[3]) << 1) | (int(idx[4]) << 4) | (int(idx[5]) << 7)) & 0xFF
        packed[qp_offset + 2] = ((int(idx[5]) >> 1) | (int(idx[6]) << 2) | (int(idx[7]) << 5)) & 0xFF
    return packed


def _unpack_indices(packed: np.ndarray) -> np.ndarray:
    """Unpack 12 bytes into 32 3-bit indices."""
    indices = np.zeros(32, dtype=np.uint8)
    for g in range(4):
        qp = packed[g * 3: g * 3 + 3]
        p = int(qp[0]) | (int(qp[1]) << 8) | (int(qp[2]) << 16)
        for r in range(8):
            indices[g * 8 + r] = (p >> (3 * r)) & 7
    return indices


@dataclass
class TQ3Block:
    """A single TQ3_1S_shift block: 32 weights in 18 bytes.

    Stores per-half means for shift variant (higher quality).
    """
    d0: np.float16      # scale for elements 0-15
    d1: np.float16      # scale for elements 16-31
    m0: np.float16      # mean for elements 0-15
    m1: np.float16      # mean for elements 16-31
    qs: np.ndarray       # 12 bytes of packed 3-bit indices

    def to_bytes(self) -> bytes:
        return self.d0.tobytes() + self.d1.tobytes() + self.m0.tobytes() + self.m1.tobytes() + self.qs.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> 'TQ3Block':
        d0 = np.frombuffer(data[0:2], dtype=np.float16)[0]
        d1 = np.frombuffer(data[2:4], dtype=np.float16)[0]
        m0 = np.frombuffer(data[4:6], dtype=np.float16)[0]
        m1 = np.frombuffer(data[6:8], dtype=np.float16)[0]
        qs = np.frombuffer(data[8:20], dtype=np.uint8).copy()
        return cls(d0=d0, d1=d1, m0=m0, m1=m1, qs=qs)


def quantize_block_tq3_1s(weights: np.ndarray) -> TQ3Block:
    """Quantize a block of 32 float weights to TQ3_1S format.

    Steps:
    1. Compute RMS scale per half-block (16 elements each)
    2. Normalize to unit variance
    3. Apply Randomized Hadamard Transform
    4. Quantize to 8-level Lloyd-Max codebook
    5. Refine scale via grid search
    6. Pack 3-bit indices
    """
    assert len(weights) == 32

    # Apply WHT to the full block
    rotated = _wht_forward(weights.astype(np.float32))

    # Split into two halves for dual-scale
    half0 = rotated[:16]
    half1 = rotated[16:]

    # Compute RMS per half
    rms0 = max(np.sqrt(np.mean(half0 ** 2)), 1e-10)
    rms1 = max(np.sqrt(np.mean(half1 ** 2)), 1e-10)

    # Quantize each half with scale refinement
    indices = np.zeros(32, dtype=np.uint8)
    d0 = _quantize_half_refine(half0, rms0, indices[:16])
    d1 = _quantize_half_refine(half1, rms1, indices[16:])

    # Clamp scales to fp16 range
    max_fp16 = 65504.0
    d0 = min(d0, max_fp16)
    d1 = min(d1, max_fp16)

    return TQ3Block(
        d0=np.float16(d0),
        d1=np.float16(d1),
        qs=_pack_indices(indices),
    )


def _quantize_halves_shift_vec(halves: np.ndarray, rms: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Vectorized quantization with mean subtraction (TQ3_1S_shift).

    Subtracts per-half mean before quantizing → better centering → higher quality.
    """
    N = halves.shape[0]
    scale_mults = np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50])

    # Center the data
    centered = halves - means[:, np.newaxis]

    best_scales = rms.copy()
    best_errors = np.full(N, np.inf)

    for mult in scale_mults:
        scales = np.maximum(rms * mult, 1e-10)
        inv = 1.0 / scales
        normalized = centered * inv[:, np.newaxis]

        indices = np.zeros((N, 16), dtype=np.uint8)
        for b in range(7):
            indices[normalized > BOUNDARIES[b]] = b + 1

        deq = CENTROIDS[indices] * scales[:, np.newaxis] + means[:, np.newaxis]
        errors = np.sum((halves - deq) ** 2, axis=1)

        better = errors < best_errors
        best_errors[better] = errors[better]
        best_scales[better] = scales[better]

    # 6 iterations of refinement
    scales = best_scales.copy()
    for _ in range(6):
        inv = 1.0 / np.maximum(scales, 1e-10)
        normalized = centered * inv[:, np.newaxis]

        indices = np.zeros((N, 16), dtype=np.uint8)
        for b in range(7):
            indices[normalized > BOUNDARIES[b]] = b + 1

        centroids_vals = CENTROIDS[indices]
        numer = np.sum(centered * centroids_vals, axis=1)
        denom = np.sum(centroids_vals ** 2, axis=1)
        new_scales = np.where(denom > 1e-12, numer / denom, scales)
        new_scales = np.maximum(new_scales, 1e-10)

        deq = CENTROIDS[indices] * new_scales[:, np.newaxis] + means[:, np.newaxis]
        errors = np.sum((halves - deq) ** 2, axis=1)
        better = errors < best_errors
        best_errors[better] = errors[better]
        best_scales[better] = new_scales[better]
        scales = new_scales

    return best_scales


def _quantize_halves_vec(halves: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Vectorized quantization of N half-blocks (N, 16) with scale refinement.

    Returns optimal scales (N,).
    """
    N = halves.shape[0]
    scale_mults = np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50])

    # Grid search: try each scale multiplier for all blocks at once
    best_scales = rms.copy()
    best_errors = np.full(N, np.inf)

    for mult in scale_mults:
        scales = np.maximum(rms * mult, 1e-10)  # (N,)
        inv = 1.0 / scales  # (N,)
        # Normalize: (N, 16) * (N, 1) = (N, 16)
        normalized = halves * inv[:, np.newaxis]

        # Quantize each element to nearest centroid
        indices = np.zeros((N, 16), dtype=np.uint8)
        for b in range(7):
            indices[normalized > BOUNDARIES[b]] = b + 1

        # Dequantize and compute error
        deq = CENTROIDS[indices] * scales[:, np.newaxis]  # (N, 16)
        errors = np.sum((halves - deq) ** 2, axis=1)  # (N,)

        better = errors < best_errors
        best_errors[better] = errors[better]
        best_scales[better] = scales[better]

    # 6 iterations of least-squares refinement
    scales = best_scales.copy()
    for _ in range(6):
        inv = 1.0 / np.maximum(scales, 1e-10)
        normalized = halves * inv[:, np.newaxis]

        indices = np.zeros((N, 16), dtype=np.uint8)
        for b in range(7):
            indices[normalized > BOUNDARIES[b]] = b + 1

        centroids_vals = CENTROIDS[indices]  # (N, 16)
        numer = np.sum(halves * centroids_vals, axis=1)
        denom = np.sum(centroids_vals ** 2, axis=1)
        new_scales = np.where(denom > 1e-12, numer / denom, scales)
        new_scales = np.maximum(new_scales, 1e-10)

        deq = CENTROIDS[indices] * new_scales[:, np.newaxis]
        errors = np.sum((halves - deq) ** 2, axis=1)
        better = errors < best_errors
        best_errors[better] = errors[better]
        best_scales[better] = new_scales[better]
        scales = new_scales

    return best_scales


def _quantize_half_search(src: np.ndarray, init_scale: float,
                           out_indices: np.ndarray) -> float:
    """Grid search for initial scale (matches llama.cpp tq3_1s_quantize_half_search)."""
    scale_search = [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50]
    best_scale = init_scale
    best_err = float('inf')
    best_idx = np.zeros(16, dtype=np.uint8)

    for mult in scale_search:
        scale = max(init_scale * mult, 1e-10)
        inv = 1.0 / scale
        trial_idx = np.zeros(16, dtype=np.uint8)
        err = 0.0
        for j in range(16):
            idx = _choose_index(src[j] * inv)
            trial_idx[j] = idx
            d = src[j] - CENTROIDS[idx] * scale
            err += d * d
        if err < best_err:
            best_err = err
            best_scale = scale
            best_idx[:] = trial_idx

    out_indices[:] = best_idx
    return best_scale


def _quantize_half_refine(src: np.ndarray, init_scale: float,
                           out_indices: np.ndarray) -> float:
    """Grid search + 6 iterations of least-squares refinement.
    Exact match to llama.cpp tq3_1s_quantize_half_refine."""
    trial_idx = np.zeros(16, dtype=np.uint8)
    scale = _quantize_half_search(src, init_scale, trial_idx)

    best_scale = scale
    best_err = float('inf')
    best_idx = np.zeros(16, dtype=np.uint8)

    for _ in range(6):
        inv = 1.0 / max(scale, 1e-10)
        numer = 0.0
        denom = 0.0

        for j in range(16):
            idx = _choose_index(src[j] * inv)
            trial_idx[j] = idx
            c = CENTROIDS[idx]
            numer += src[j] * c
            denom += c * c

        if denom > 1e-12:
            scale = max(numer / denom, 1e-10)

        err = 0.0
        for j in range(16):
            d = src[j] - CENTROIDS[trial_idx[j]] * scale
            err += d * d

        if err < best_err:
            best_err = err
            best_scale = scale
            best_idx[:] = trial_idx

    out_indices[:] = best_idx
    return best_scale


def _wht_forward_batch(blocks: np.ndarray) -> np.ndarray:
    """Vectorized WHT on (N, 32) array."""
    out = blocks * SIGNS[np.newaxis, :]
    step = 1
    while step < 32:
        for i in range(0, 32, step * 2):
            for j in range(i, i + step):
                a = out[:, j].copy()
                b = out[:, j + step].copy()
                out[:, j] = a + b
                out[:, j + step] = a - b
        step *= 2
    return out / np.sqrt(32.0)


def _choose_index_vec(values: np.ndarray) -> np.ndarray:
    """Vectorized centroid index selection. values shape: (N,)"""
    indices = np.zeros(len(values), dtype=np.uint8)
    for b in range(7):
        indices[values > BOUNDARIES[b]] = b + 1
    return indices


def quantize_tensor(weights: np.ndarray) -> tuple:
    """Quantize a full weight tensor to TQ3_1S format (vectorized).

    Args:
        weights: float array of any shape

    Returns:
        (blocks_data: bytes, shape: tuple, num_blocks: int)
    """
    flat = weights.flatten().astype(np.float32)
    pad_len = (32 - len(flat) % 32) % 32
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])

    num_blocks = len(flat) // 32
    blocks = flat.reshape(num_blocks, 32)

    # Batch WHT
    rotated = _wht_forward_batch(blocks)

    # Split halves
    half0 = rotated[:, :16]
    half1 = rotated[:, 16:]

    # RMS per half
    rms0 = np.sqrt(np.mean(half0 ** 2, axis=1))
    rms1 = np.sqrt(np.mean(half1 ** 2, axis=1))
    rms0[rms0 < 1e-10] = 1.0
    rms1[rms1 < 1e-10] = 1.0

    # Compute per-block mean in rotated domain for shift variant
    block_means = np.mean(rotated, axis=1)  # (N,)
    half0_means = np.mean(half0, axis=1)
    half1_means = np.mean(half1, axis=1)

    # RMS of centered data (after mean subtraction)
    centered0 = half0 - half0_means[:, np.newaxis]
    centered1 = half1 - half1_means[:, np.newaxis]
    rms0_c = np.sqrt(np.mean(centered0 ** 2, axis=1))
    rms1_c = np.sqrt(np.mean(centered1 ** 2, axis=1))
    rms0_c[rms0_c < 1e-10] = 1.0
    rms1_c[rms1_c < 1e-10] = 1.0

    # Use shift variant for better quality
    all_indices = np.zeros((num_blocks, 32), dtype=np.uint8)
    all_d0 = _quantize_halves_shift_vec(half0, rms0_c, half0_means)
    all_d1 = _quantize_halves_shift_vec(half1, rms1_c, half1_means)

    # Compute final indices with optimal scales (with mean subtraction)
    for j in range(16):
        all_indices[:, j] = _choose_index_vec((half0[:, j] - half0_means) / np.maximum(all_d0, 1e-10))
        all_indices[:, 16+j] = _choose_index_vec((half1[:, j] - half1_means) / np.maximum(all_d1, 1e-10))

    # Clamp to fp16 range and pack (20 bytes per block: 4 scales + 4 means + 12 indices)
    BLOCK_BYTES = 20
    max_fp16 = 65504.0
    all_d0 = np.clip(all_d0, -max_fp16, max_fp16)
    all_d1 = np.clip(all_d1, -max_fp16, max_fp16)
    half0_means = np.clip(half0_means, -max_fp16, max_fp16)
    half1_means = np.clip(half1_means, -max_fp16, max_fp16)

    blocks_data = bytearray(num_blocks * BLOCK_BYTES)
    for i in range(num_blocks):
        offset = i * BLOCK_BYTES
        blocks_data[offset:offset+2] = np.float16(all_d0[i]).tobytes()
        blocks_data[offset+2:offset+4] = np.float16(all_d1[i]).tobytes()
        blocks_data[offset+4:offset+6] = np.float16(half0_means[i]).tobytes()
        blocks_data[offset+6:offset+8] = np.float16(half1_means[i]).tobytes()
        packed = _pack_indices(all_indices[i])
        blocks_data[offset+8:offset+20] = packed.tobytes()

    return bytes(blocks_data), weights.shape, num_blocks


def dequantize_block_tq3_1s(block: TQ3Block) -> np.ndarray:
    """Dequantize a TQ3_1S_shift block back to 32 float values."""
    indices = _unpack_indices(block.qs)
    d0 = float(block.d0)
    d1 = float(block.d1)
    m0 = float(block.m0)
    m1 = float(block.m1)

    # Reconstruct rotated values: centroid * scale + mean
    rotated = np.zeros(32, dtype=np.float32)
    for j in range(16):
        rotated[j] = CENTROIDS[indices[j]] * d0 + m0
    for j in range(16):
        rotated[16 + j] = CENTROIDS[indices[16 + j]] * d1 + m1

    # Inverse WHT to get original weight space
    return _wht_inverse(rotated)


def dequantize_tensor(blocks_data: bytes, shape: tuple, num_blocks: int) -> np.ndarray:
    """Dequantize a TQ3 tensor. Auto-detects block format (16B or 20B)."""
    total_data = len(blocks_data)
    if num_blocks > 0:
        block_bytes = total_data // num_blocks
    else:
        block_bytes = 20

    flat = np.zeros(num_blocks * 32, dtype=np.float32)

    if block_bytes == 16:
        # Old format: d0, d1, qs (no means)
        for i in range(num_blocks):
            data = blocks_data[i * 16: (i + 1) * 16]
            d0 = np.frombuffer(data[0:2], dtype=np.float16).astype(np.float32)[0]
            d1 = np.frombuffer(data[2:4], dtype=np.float16).astype(np.float32)[0]
            qs = np.frombuffer(data[4:16], dtype=np.uint8)
            indices = _unpack_indices(qs)
            rotated = np.zeros(32, dtype=np.float32)
            for j in range(16):
                rotated[j] = CENTROIDS[indices[j]] * d0
            for j in range(16):
                rotated[16 + j] = CENTROIDS[indices[16 + j]] * d1
            flat[i * 32: (i + 1) * 32] = _wht_inverse(rotated)
    else:
        # New format: d0, d1, m0, m1, qs (with means)
        for i in range(num_blocks):
            block = TQ3Block.from_bytes(blocks_data[i * block_bytes: (i + 1) * block_bytes])
            flat[i * 32: (i + 1) * 32] = dequantize_block_tq3_1s(block)

    total_elements = 1
    for s in shape:
        total_elements *= s
    return flat[:total_elements].reshape(shape)
