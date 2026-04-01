"""Convert MoE expert weights from 4-bit packed binary to TQ3 format.

Expert binary layout (per expert, 1,769,472 bytes):
  gate_proj weights: 512×2048 at 4-bit = 524,288 bytes (offset 0)
  gate_proj scales:  512×32 bf16       = 32,768 bytes  (offset 524288)
  gate_proj biases:  512×32 bf16       = 32,768 bytes  (offset 557056)
  up_proj weights:   512×2048 at 4-bit = 524,288 bytes (offset 589824)
  up_proj scales:    512×32 bf16       = 32,768 bytes  (offset 1114112)
  (up biases implicit)
  down_proj weights: 2048×512 at 4-bit = 524,288 bytes (offset 1179648)
  down_proj scales:  2048×8 bf16       = 32,768 bytes  (offset 1703936)

4-bit format: uint32 packs 8 4-bit nibbles, bf16 scale per group of 64
"""

from __future__ import annotations

import os
import struct
import time
from typing import Optional

import numpy as np

from kandiga.tq3.quantize import quantize_tensor


# Expert layout constants (from kandiga_cpu_expert.m)
HIDDEN_SIZE = 2048
EXPERT_DIM = 512
GROUP_SIZE = 64

GATE_WEIGHT_OFF = 0
GATE_SCALES_OFF = 524288
GATE_BIASES_OFF = 557056
UP_WEIGHT_OFF = 589824
UP_SCALES_OFF = 1114112
DOWN_WEIGHT_OFF = 1179648
DOWN_SCALES_OFF = 1703936
EXPERT_SIZE = 1769472


def _bf16_to_float(data: bytes) -> np.ndarray:
    """Convert bfloat16 bytes to float32."""
    u16 = np.frombuffer(data, dtype=np.uint16)
    # bfloat16 is upper 16 bits of float32
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _dequant_4bit(weight_bytes: bytes, scale_bytes: bytes, bias_bytes: Optional[bytes],
                   out_dim: int, in_dim: int) -> np.ndarray:
    """Dequantize 4-bit packed weights to float32.

    Weight format: uint32 packs 8 4-bit nibbles
    Scale format: bf16, one per group of 64
    """
    weights_u32 = np.frombuffer(weight_bytes, dtype=np.uint32)
    scales = _bf16_to_float(scale_bytes)
    biases = _bf16_to_float(bias_bytes) if bias_bytes else np.zeros_like(scales)

    num_groups = in_dim // GROUP_SIZE
    packed_per_group = GROUP_SIZE // 8  # 8 nibbles per uint32

    output = np.zeros((out_dim, in_dim), dtype=np.float32)
    packed_per_row = in_dim // 8  # total uint32s per row

    for row in range(out_dim):
        w_row = weights_u32[row * packed_per_row: (row + 1) * packed_per_row]
        for g in range(num_groups):
            scale = scales[row * num_groups + g]
            bias = biases[row * num_groups + g] if biases is not None else 0.0
            for p in range(packed_per_group):
                packed = int(w_row[g * packed_per_group + p])
                for n in range(8):
                    nibble = (packed >> (n * 4)) & 0xF
                    col = g * GROUP_SIZE + p * 8 + n
                    output[row, col] = float(nibble) * scale + bias

    return output


def _dequant_4bit_fast(weight_bytes: bytes, scale_bytes: bytes, bias_bytes: Optional[bytes],
                        out_dim: int, in_dim: int) -> np.ndarray:
    """Vectorized 4-bit dequantization."""
    weights_u32 = np.frombuffer(weight_bytes, dtype=np.uint32).reshape(out_dim, in_dim // 8)
    scales = _bf16_to_float(scale_bytes).reshape(out_dim, in_dim // GROUP_SIZE)

    if bias_bytes:
        biases = _bf16_to_float(bias_bytes).reshape(out_dim, in_dim // GROUP_SIZE)
    else:
        biases = np.zeros_like(scales)

    # Unpack 8 nibbles from each uint32
    output = np.zeros((out_dim, in_dim), dtype=np.float32)
    for n in range(8):
        nibbles = (weights_u32 >> (n * 4)) & 0xF  # (out_dim, in_dim/8)
        nibbles_f = nibbles.astype(np.float32)

        # Map nibbles to column positions
        col_offset = n
        for p_idx in range(weights_u32.shape[1]):
            g = (p_idx * 8 + n) // GROUP_SIZE
            col = p_idx * 8 + n
            if col < in_dim:
                output[:, col] = nibbles_f[:, p_idx] * scales[:, g] + biases[:, g]

    return output


def convert_expert_layer(layer_path: str, output_path: str, verbose: bool = True) -> dict:
    """Convert one layer's expert binary from 4-bit to TQ3.

    Args:
        layer_path: Path to e.g. layer_00.bin
        output_path: Path for output TQ3 expert file

    Returns:
        Stats dict
    """
    with open(layer_path, "rb") as f:
        header = f.read(4096)
        all_data = f.read()

    magic = header[:4]
    assert magic == b"BKEX", f"Invalid magic: {magic}"
    num_experts = struct.unpack("<I", header[8:12])[0]
    expert_size = struct.unpack("<Q", header[12:20])[0]

    if verbose:
        print(f"  Layer: {os.path.basename(layer_path)} ({num_experts} experts, {expert_size:,}B each)")

    # Process each expert
    tq3_experts = []
    total_orig = 0
    total_tq3 = 0

    for eidx in range(num_experts):
        expert_data = all_data[eidx * expert_size: (eidx + 1) * expert_size]

        # Extract tensors
        gate_w = expert_data[GATE_WEIGHT_OFF:GATE_SCALES_OFF]
        gate_s = expert_data[GATE_SCALES_OFF:GATE_BIASES_OFF]
        gate_b = expert_data[GATE_BIASES_OFF:UP_WEIGHT_OFF]

        up_w = expert_data[UP_WEIGHT_OFF:UP_SCALES_OFF]
        up_s = expert_data[UP_SCALES_OFF:DOWN_WEIGHT_OFF]

        down_w = expert_data[DOWN_WEIGHT_OFF:DOWN_SCALES_OFF]
        down_s = expert_data[DOWN_SCALES_OFF:]

        # Dequantize gate (512×2048)
        gate_float = _dequant_4bit_fast(gate_w, gate_s, gate_b, EXPERT_DIM, HIDDEN_SIZE)
        # Dequantize up (512×2048)
        up_float = _dequant_4bit_fast(up_w, up_s, None, EXPERT_DIM, HIDDEN_SIZE)
        # Dequantize down (2048×512)
        down_float = _dequant_4bit_fast(down_w, down_s, None, HIDDEN_SIZE, EXPERT_DIM)

        # Requantize to TQ3
        gate_tq3, _, _ = quantize_tensor(gate_float)
        up_tq3, _, _ = quantize_tensor(up_float)
        down_tq3, _, _ = quantize_tensor(down_float)

        tq3_expert = gate_tq3 + up_tq3 + down_tq3
        tq3_experts.append(tq3_expert)

        total_orig += expert_size
        total_tq3 += len(tq3_expert)

        if verbose and eidx % 32 == 0:
            comp = expert_size / len(tq3_expert) if len(tq3_expert) > 0 else 0
            print(f"    Expert {eidx}/{num_experts}: {len(tq3_expert):,}B ({comp:.1f}x compression)")

    # Write TQ3 expert file
    tq3_expert_size = len(tq3_experts[0]) if tq3_experts else 0

    with open(output_path, "wb") as f:
        # Same header format but with new expert_size
        new_header = bytearray(4096)
        new_header[0:4] = b"TQ3X"  # TQ3 expert magic
        struct.pack_into("<I", new_header, 4, 1)  # version
        struct.pack_into("<I", new_header, 8, num_experts)
        struct.pack_into("<Q", new_header, 12, tq3_expert_size)
        # Store original dimensions for dequant
        struct.pack_into("<I", new_header, 20, HIDDEN_SIZE)
        struct.pack_into("<I", new_header, 24, EXPERT_DIM)
        struct.pack_into("<I", new_header, 28, GROUP_SIZE)
        f.write(new_header)

        for expert_data in tq3_experts:
            f.write(expert_data)

    if verbose:
        comp = total_orig / total_tq3 if total_tq3 > 0 else 0
        print(f"    Saved: {os.path.getsize(output_path)/1e6:.0f}MB (was {total_orig/1e6:.0f}MB, {comp:.1f}x)")

    return {
        "num_experts": num_experts,
        "original_bytes": total_orig,
        "tq3_bytes": total_tq3,
        "compression": total_orig / total_tq3 if total_tq3 > 0 else 0,
    }


def convert_all_expert_layers(
    packed_dir: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Convert all expert layers to TQ3."""
    if output_dir is None:
        output_dir = packed_dir.rstrip("/") + "_tq3"
    os.makedirs(output_dir, exist_ok=True)

    layer_files = sorted(f for f in os.listdir(packed_dir) if f.startswith("layer_") and f.endswith(".bin"))

    if verbose:
        print(f"Converting {len(layer_files)} expert layers to TQ3...")
        print(f"Input: {packed_dir}")
        print(f"Output: {output_dir}")

    total_stats = {"layers": 0, "original_bytes": 0, "tq3_bytes": 0}

    for lf in layer_files:
        t0 = time.time()
        stats = convert_expert_layer(
            os.path.join(packed_dir, lf),
            os.path.join(output_dir, lf),
            verbose=verbose,
        )
        elapsed = time.time() - t0

        total_stats["layers"] += 1
        total_stats["original_bytes"] += stats["original_bytes"]
        total_stats["tq3_bytes"] += stats["tq3_bytes"]

        if verbose:
            print(f"  {lf}: {elapsed:.0f}s")

    if verbose and total_stats["tq3_bytes"] > 0:
        comp = total_stats["original_bytes"] / total_stats["tq3_bytes"]
        print(f"\n=== Expert Conversion Complete ===")
        print(f"Layers: {total_stats['layers']}")
        print(f"Original: {total_stats['original_bytes']/1e9:.1f} GB")
        print(f"TQ3: {total_stats['tq3_bytes']/1e9:.1f} GB")
        print(f"Compression: {comp:.1f}x")
        print(f"Saved: {(total_stats['original_bytes'] - total_stats['tq3_bytes'])/1e9:.1f} GB")

    return total_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    convert_all_expert_layers(args.packed_dir, args.output_dir)
