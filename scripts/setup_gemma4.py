#!/usr/bin/env python3
"""Split and pack Gemma 4 26B-A4B experts for SEM inference.

Reads the single safetensors file, splits the fused 3D expert tensors
into per-expert binary files for pread()-based loading.

Usage:
    python scripts/setup_gemma4.py philtrem/gemma-4-26b-a4b-it-MLX-4bit
"""

import argparse
import gc
import os
import struct
import time

import mlx.core as mx
import numpy as np

HEADER_SIZE = 4096
NUM_LAYERS = 30
NUM_EXPERTS = 128

# Tensor order in the binary file (must match C kernel expectations)
# gate_proj and up_proj split from the fused gate_up_proj
TENSOR_ORDER = [
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
]


def _tensor_to_bytes(t):
    """Convert MLX tensor to raw bytes."""
    mx.eval(t)
    if t.dtype == mx.uint32:
        return np.array(t, copy=False).tobytes()
    else:
        # bf16 → view as uint16 → bytes
        u16 = t.view(mx.uint16)
        mx.eval(u16)
        return np.array(u16, copy=False).tobytes()


def _build_header(num_experts, expert_size, tensor_info):
    """Build BKEX binary header."""
    buf = bytearray(HEADER_SIZE)
    buf[0:4] = b"BKEX"
    struct.pack_into("<I", buf, 4, 1)  # version
    struct.pack_into("<I", buf, 8, num_experts)
    struct.pack_into("<Q", buf, 12, expert_size)
    struct.pack_into("<I", buf, 20, len(tensor_info))

    pos = 24
    for name, offset, nbytes, shape, dtype_str in tensor_info:
        dtype_code = 0 if dtype_str == "uint32" else 1
        name_bytes = name.encode("ascii")
        buf[pos] = len(name_bytes); pos += 1
        buf[pos:pos+len(name_bytes)] = name_bytes; pos += 24
        struct.pack_into("<I", buf, pos, offset); pos += 4
        struct.pack_into("<I", buf, pos, nbytes); pos += 4
        struct.pack_into("<I", buf, pos, shape[0]); pos += 4
        struct.pack_into("<I", buf, pos, shape[1] if len(shape) > 1 else 0); pos += 4
        buf[pos] = dtype_code; pos += 1

    return bytes(buf)


def setup_gemma4(model_path: str):
    """Split and pack Gemma 4 experts for SEM."""

    # Resolve model path
    from huggingface_hub import snapshot_download
    local_path = snapshot_download(model_path, local_files_only=True)
    safetensors_path = os.path.join(local_path, "model.safetensors")
    assert os.path.isfile(safetensors_path), f"Not found: {safetensors_path}"

    # Output directory
    cache_dir = os.path.expanduser("~/.kandiga/experts/gemma-4-26b-a4b-it-4bit")
    packed_dir = os.path.join(cache_dir, "packed")
    os.makedirs(packed_dir, exist_ok=True)

    print(f"Loading weights from {safetensors_path}...")
    t0 = time.time()
    weights = mx.load(safetensors_path)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Process each layer
    total_start = time.time()
    expert_size = None
    tensor_info = None

    for layer_idx in range(NUM_LAYERS):
        layer_start = time.time()
        prefix = f"model.language_model.layers.{layer_idx}.experts"

        # Get fused expert tensors for this layer
        gate_up_w = weights[f"{prefix}.gate_up_proj"]       # (128, 1408, cols_w)
        gate_up_s = weights[f"{prefix}.gate_up_proj_scales"] # (128, 1408, cols_s)
        gate_up_b = weights[f"{prefix}.gate_up_proj_biases"] # (128, 1408, cols_b)
        down_w = weights[f"{prefix}.down_proj"]              # (128, 2816, cols_w2)
        down_s = weights[f"{prefix}.down_proj_scales"]       # (128, 2816, cols_s2)
        down_b = weights[f"{prefix}.down_proj_biases"]       # (128, 2816, cols_b2)

        mx.eval(gate_up_w, gate_up_s, gate_up_b, down_w, down_s, down_b)

        # Split gate_up along dim 1: first half = gate, second half = up
        half = gate_up_w.shape[1] // 2  # 704

        # Compute expert size and header from first expert of first layer
        if expert_size is None:
            # Build tensor info for header
            expert_tensors = {
                "gate_proj.weight": gate_up_w[0, :half],
                "gate_proj.scales": gate_up_s[0, :half],
                "gate_proj.biases": gate_up_b[0, :half],
                "up_proj.weight": gate_up_w[0, half:],
                "up_proj.scales": gate_up_s[0, half:],
                "up_proj.biases": gate_up_b[0, half:],
                "down_proj.weight": down_w[0],
                "down_proj.scales": down_s[0],
                "down_proj.biases": down_b[0],
            }

            tensor_info = []
            offset = 0
            for name in TENSOR_ORDER:
                t = expert_tensors[name]
                dtype_str = "uint32" if t.dtype == mx.uint32 else "bfloat16"
                elem_size = 4 if dtype_str == "uint32" else 2
                nbytes = t.size * elem_size
                tensor_info.append((name, offset, nbytes, tuple(t.shape), dtype_str))
                offset += nbytes
            expert_size = offset
            print(f"Expert size: {expert_size/1024:.0f} KB ({expert_size} bytes)")

        # Pack all experts for this layer
        out_path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.bin")
        header = _build_header(NUM_EXPERTS, expert_size, tensor_info)

        with open(out_path, "wb") as f:
            f.write(header)
            for eidx in range(NUM_EXPERTS):
                # Split this expert's tensors
                parts = [
                    _tensor_to_bytes(gate_up_w[eidx, :half]),   # gate weight
                    _tensor_to_bytes(gate_up_s[eidx, :half]),   # gate scales
                    _tensor_to_bytes(gate_up_b[eidx, :half]),   # gate biases
                    _tensor_to_bytes(gate_up_w[eidx, half:]),   # up weight
                    _tensor_to_bytes(gate_up_s[eidx, half:]),   # up scales
                    _tensor_to_bytes(gate_up_b[eidx, half:]),   # up biases
                    _tensor_to_bytes(down_w[eidx]),             # down weight
                    _tensor_to_bytes(down_s[eidx]),             # down scales
                    _tensor_to_bytes(down_b[eidx]),             # down biases
                ]
                f.write(b"".join(parts))

        # Verify file size
        expected = HEADER_SIZE + NUM_EXPERTS * expert_size
        actual = os.path.getsize(out_path)
        assert actual == expected, f"Layer {layer_idx}: {actual} != {expected}"

        elapsed = time.time() - layer_start
        total = time.time() - total_start
        eta = (total / (layer_idx + 1)) * (NUM_LAYERS - layer_idx - 1)
        print(f"  Layer {layer_idx:2d}/{NUM_LAYERS-1}: {elapsed:.1f}s (ETA {eta:.0f}s)")

        gc.collect()

    total_elapsed = time.time() - total_start
    layer_size_mb = (HEADER_SIZE + NUM_EXPERTS * expert_size) / 1e6
    total_size_gb = layer_size_mb * NUM_LAYERS / 1e3
    print(f"\nDone! {NUM_LAYERS} layers packed in {total_elapsed:.1f}s")
    print(f"Expert: {expert_size/1024:.0f} KB, Layer file: {layer_size_mb:.0f} MB")
    print(f"Total disk: {total_size_gb:.1f} GB")
    print(f"Output: {packed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="HuggingFace model ID or local path")
    args = parser.parse_args()
    setup_gemma4(args.model_path)
