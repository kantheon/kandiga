"""Pack per-expert safetensors into single raw binary files per layer.

Reads the split per-expert safetensors files and packs them into a compact
binary format that can be read with zero parsing overhead using pread().

Binary format per file:
  Header (4096 bytes):
    magic:       4 bytes  "BKEX"
    version:     uint32   1
    num_experts: uint32   (auto-detected)
    expert_size: uint64   (auto-detected from tensor shapes)
    num_tensors: uint32   9
    tensor descriptors...
    padding to 4096 bytes

  Expert data (num_experts x expert_size bytes):
    expert_000: [gate.weight][gate.scales][gate.biases]
                [up.weight][up.scales][up.biases]
                [down.weight][down.scales][down.biases]
    ...
"""

from __future__ import annotations

import os
import struct
import time

import mlx.core as mx
import numpy as np

HEADER_SIZE = 4096

# Tensor names in canonical order (must match C library)
TENSOR_NAMES = [
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


def _detect_layout(sample_expert_path: str):
    """Auto-detect tensor shapes and dtypes from a sample expert file.

    Returns:
        tensor_order: list of (name, shape, dtype_str)
        expert_size: total bytes per expert
        num_experts: number of experts per layer (detected from directory)
    """
    tensors = mx.load(sample_expert_path)
    mx.eval(*tensors.values())

    tensor_order = []
    expert_size = 0

    for name in TENSOR_NAMES:
        t = tensors[name]
        shape = tuple(t.shape)
        # Determine dtype
        if t.dtype in (mx.uint32, mx.int32):
            dtype_str = "uint32"
            itemsize = 4
        else:
            dtype_str = "bfloat16"
            itemsize = 2

        nbytes = 1
        for dim in shape:
            nbytes *= dim
        nbytes *= itemsize

        tensor_order.append((name, shape, dtype_str, nbytes))
        expert_size += nbytes

    del tensors
    return tensor_order, expert_size


def _count_experts(layer_dir: str) -> int:
    """Count expert files in a layer directory."""
    return len([f for f in os.listdir(layer_dir) if f.startswith("expert_") and f.endswith(".safetensors")])


def _count_layers(input_dir: str) -> int:
    """Count layer directories."""
    return len([d for d in os.listdir(input_dir) if d.startswith("layer_") and os.path.isdir(os.path.join(input_dir, d))])


def _build_header(num_experts: int, expert_size: int, tensor_order: list) -> bytes:
    """Build the 4096-byte binary header."""
    buf = bytearray(HEADER_SIZE)

    buf[0:4] = b"BKEX"
    struct.pack_into("<I", buf, 4, 1)  # version
    struct.pack_into("<I", buf, 8, num_experts)
    struct.pack_into("<Q", buf, 12, expert_size)
    struct.pack_into("<I", buf, 20, len(tensor_order))

    offset_in_expert = 0
    pos = 24
    for name, shape, dtype_str, nbytes in tensor_order:
        dtype_code = 0 if dtype_str == "uint32" else 1

        name_bytes = name.encode("ascii")
        struct.pack_into("<B", buf, pos, len(name_bytes))
        pos += 1
        buf[pos: pos + len(name_bytes)] = name_bytes
        pos += 24
        struct.pack_into("<I", buf, pos, offset_in_expert)
        pos += 4
        struct.pack_into("<I", buf, pos, nbytes)
        pos += 4
        # Store up to 4 dims
        for d in range(2):
            val = shape[d] if d < len(shape) else 0
            struct.pack_into("<I", buf, pos, val)
            pos += 4
        struct.pack_into("<B", buf, pos, dtype_code)
        pos += 1

        offset_in_expert += nbytes

    assert offset_in_expert == expert_size, (
        f"Tensor sizes don't sum to expert_size: {offset_in_expert} != {expert_size}"
    )
    return bytes(buf)


def _expert_to_bytes(tensors: dict[str, mx.array], tensor_order: list, expert_size: int) -> bytes:
    """Convert an expert's tensor dict to raw bytes in canonical order."""
    parts = []
    for name, shape, dtype_str, expected_nbytes in tensor_order:
        tensor = tensors[name]
        mx.eval(tensor)

        if dtype_str == "uint32":
            np_arr = np.array(tensor, copy=False)
            raw = np_arr.tobytes()
        else:
            u16 = tensor.view(mx.uint16)
            mx.eval(u16)
            np_arr = np.array(u16, copy=False)
            raw = np_arr.tobytes()

        assert len(raw) == expected_nbytes, f"{name}: got {len(raw)} bytes, expected {expected_nbytes}"
        parts.append(raw)

    data = b"".join(parts)
    assert len(data) == expert_size, f"Expert data {len(data)} != {expert_size}"
    return data


def _pack_layer(layer_idx: int, input_dir: str, output_dir: str,
                num_experts: int, expert_size: int, tensor_order: list) -> None:
    """Pack all experts for one layer into a single binary file."""
    layer_dir = os.path.join(input_dir, f"layer_{layer_idx:02d}")
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    header = _build_header(num_experts, expert_size, tensor_order)

    with open(out_path, "wb") as f:
        f.write(header)
        for expert_idx in range(num_experts):
            st_path = os.path.join(layer_dir, f"expert_{expert_idx:03d}.safetensors")
            tensors = mx.load(st_path)
            mx.eval(*tensors.values())
            raw = _expert_to_bytes(tensors, tensor_order, expert_size)
            f.write(raw)
            del tensors

    # Verify file size
    expected_size = HEADER_SIZE + num_experts * expert_size
    actual_size = os.path.getsize(out_path)
    assert actual_size == expected_size, (
        f"File size mismatch: {actual_size} != {expected_size}"
    )


def pack_experts(input_dir: str, output_dir: str, num_layers: int | None = None) -> None:
    """Pack all layers from split expert files into binary format.

    Auto-detects model dimensions, expert count, and layer count.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect layout from first expert of first layer
    if num_layers is None:
        num_layers = _count_layers(input_dir)

    layer0_dir = os.path.join(input_dir, "layer_00")
    num_experts = _count_experts(layer0_dir)
    sample_path = os.path.join(layer0_dir, "expert_000.safetensors")

    tensor_order, expert_size = _detect_layout(sample_path)

    expert_size_kb = expert_size / 1024
    print(f"  Detected: {num_layers} layers, {num_experts} experts, {expert_size_kb:.0f}KB per expert")

    total_start = time.time()
    for layer_idx in range(num_layers):
        layer_start = time.time()
        print(f"  Packing layer {layer_idx:2d}/{num_layers - 1}...", end=" ", flush=True)
        _pack_layer(layer_idx, input_dir, output_dir, num_experts, expert_size, tensor_order)
        elapsed = time.time() - layer_start
        total_elapsed = time.time() - total_start
        eta = (total_elapsed / (layer_idx + 1)) * (num_layers - layer_idx - 1)
        print(f"done ({elapsed:.1f}s, ETA {eta:.0f}s)")

    total_elapsed = time.time() - total_start
    print(f"  {num_layers} layer files packed in {total_elapsed:.1f}s")
