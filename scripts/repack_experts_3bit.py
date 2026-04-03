#!/usr/bin/env python3
"""Repack 4-bit expert binary files to 3-bit.

Reads existing packed binary files (layer_XX.bin), dequantizes expert weights
from 4-bit → float → 3-bit, and writes new binary files with 22% less I/O.

Usage:
    python scripts/repack_experts_3bit.py ~/.kandiga/experts/Qwen3.5-35B-A3B-4bit/packed

Creates: packed_3bit/ directory alongside packed/ with 3-bit binary files.
"""

import argparse
import os
import struct
import time

import mlx.core as mx
import numpy as np

HEADER_SIZE = 4096
GROUP_SIZE = 64


def parse_header(path: str):
    """Parse BKEX header from a packed binary file."""
    with open(path, "rb") as f:
        hdr = f.read(HEADER_SIZE)

    assert hdr[:4] == b"BKEX", f"Bad magic: {hdr[:4]}"
    version = struct.unpack_from("<I", hdr, 4)[0]
    num_experts = struct.unpack_from("<I", hdr, 8)[0]
    expert_size = struct.unpack_from("<Q", hdr, 12)[0]
    num_tensors = struct.unpack_from("<I", hdr, 20)[0]

    tensors = []
    pos = 24
    for _ in range(num_tensors):
        name_len = hdr[pos]; pos += 1
        name = hdr[pos:pos+name_len].decode("ascii"); pos += 24
        offset = struct.unpack_from("<I", hdr, pos)[0]; pos += 4
        nbytes = struct.unpack_from("<I", hdr, pos)[0]; pos += 4
        d0 = struct.unpack_from("<I", hdr, pos)[0]; pos += 4
        d1 = struct.unpack_from("<I", hdr, pos)[0]; pos += 4
        dtype_code = hdr[pos]; pos += 1
        tensors.append({
            "name": name, "offset": offset, "nbytes": nbytes,
            "shape": (d0, d1), "dtype": "uint32" if dtype_code == 0 else "bfloat16",
        })

    return num_experts, expert_size, tensors


def read_expert(fd: int, expert_idx: int, expert_size: int, tensor_descs: list):
    """Read one expert's tensors from a packed binary file."""
    file_offset = HEADER_SIZE + expert_idx * expert_size
    data = os.pread(fd, expert_size, file_offset)

    result = {}
    for td in tensor_descs:
        start = td["offset"]
        end = start + td["nbytes"]
        raw = data[start:end]

        if td["dtype"] == "uint32":
            arr = np.frombuffer(raw, dtype=np.uint32).reshape(td["shape"])
            result[td["name"]] = mx.array(arr)
        else:
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(td["shape"])
            # Reinterpret uint16 bits as bfloat16 (NOT cast!)
            result[td["name"]] = mx.array(arr).view(mx.bfloat16)

    return result


def convert_expert_3bit(tensors_4bit: dict, tensor_descs: list):
    """Convert one expert's tensors from 4-bit to 3-bit.

    Returns dict of 3-bit tensors and the new tensor layout.
    """
    result = {}
    new_layout = []

    proj_names = ["gate_proj", "up_proj", "down_proj"]
    for proj in proj_names:
        w = tensors_4bit[f"{proj}.weight"]
        s = tensors_4bit[f"{proj}.scales"]
        b = tensors_4bit[f"{proj}.biases"]

        # Dequantize 4-bit → float
        w_float = mx.dequantize(w, s, b, group_size=GROUP_SIZE, bits=4)

        # Requantize float → 3-bit
        w3, s3, b3 = mx.quantize(w_float, group_size=GROUP_SIZE, bits=3)
        mx.eval(w3, s3, b3)

        result[f"{proj}.weight"] = w3
        result[f"{proj}.scales"] = s3
        result[f"{proj}.biases"] = b3

        del w_float

    return result


def expert_to_bytes_3bit(tensors: dict):
    """Convert 3-bit expert tensors to raw bytes."""
    parts = []
    total = 0
    layout = []

    for proj in ["gate_proj", "up_proj", "down_proj"]:
        for suffix in ["weight", "scales", "biases"]:
            key = f"{proj}.{suffix}"
            t = tensors[key]
            mx.eval(t)

            if t.dtype == mx.uint32:
                # 3-bit weights packed as uint32
                raw = np.array(t, copy=False).tobytes()
                dtype_str = "uint32"
            else:
                # scales/biases — must be bfloat16 for C kernel
                # mx.quantize returns float32, so cast to bf16 first
                bf16 = t.astype(mx.bfloat16)
                mx.eval(bf16)
                u16 = bf16.view(mx.uint16)
                mx.eval(u16)
                raw = np.array(u16, copy=False).tobytes()
                dtype_str = "bfloat16"

            layout.append({
                "name": key,
                "offset": total,
                "nbytes": len(raw),
                "shape": tuple(t.shape),
                "dtype": dtype_str,
            })
            parts.append(raw)
            total += len(raw)

    return b"".join(parts), total, layout


def build_header_3bit(num_experts: int, expert_size: int, layout: list):
    """Build BKEX header for 3-bit packed file."""
    buf = bytearray(HEADER_SIZE)
    buf[0:4] = b"BKEX"
    struct.pack_into("<I", buf, 4, 1)  # version
    struct.pack_into("<I", buf, 8, num_experts)
    struct.pack_into("<Q", buf, 12, expert_size)
    struct.pack_into("<I", buf, 20, len(layout))

    pos = 24
    for td in layout:
        name_bytes = td["name"].encode("ascii")
        dtype_code = 0 if td["dtype"] == "uint32" else 1
        buf[pos] = len(name_bytes); pos += 1
        buf[pos:pos+len(name_bytes)] = name_bytes; pos += 24
        struct.pack_into("<I", buf, pos, td["offset"]); pos += 4
        struct.pack_into("<I", buf, pos, td["nbytes"]); pos += 4
        struct.pack_into("<I", buf, pos, td["shape"][0]); pos += 4
        struct.pack_into("<I", buf, pos, td["shape"][1] if len(td["shape"]) > 1 else 0); pos += 4
        buf[pos] = dtype_code; pos += 1

    return bytes(buf)


def repack_layer(input_path: str, output_path: str,
                 num_experts: int, expert_size_4bit: int, tensor_descs: list):
    """Repack one layer from 4-bit to 3-bit."""
    fd = os.open(input_path, os.O_RDONLY)

    # Convert first expert to get layout
    expert0 = read_expert(fd, 0, expert_size_4bit, tensor_descs)
    expert0_3bit = convert_expert_3bit(expert0, tensor_descs)
    data0, expert_size_3bit, layout = expert_to_bytes_3bit(expert0_3bit)
    del expert0, expert0_3bit

    header = build_header_3bit(num_experts, expert_size_3bit, layout)

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(data0)  # expert 0 already converted

        for eidx in range(1, num_experts):
            expert = read_expert(fd, eidx, expert_size_4bit, tensor_descs)
            expert_3bit = convert_expert_3bit(expert, tensor_descs)
            data, _, _ = expert_to_bytes_3bit(expert_3bit)
            f.write(data)
            del expert, expert_3bit

    os.close(fd)

    # Verify
    expected = HEADER_SIZE + num_experts * expert_size_3bit
    actual = os.path.getsize(output_path)
    assert actual == expected, f"Size mismatch: {actual} != {expected}"

    return expert_size_3bit


def main():
    parser = argparse.ArgumentParser(description="Repack 4-bit experts to 3-bit")
    parser.add_argument("packed_dir", help="Path to packed/ directory with 4-bit layer_XX.bin files")
    args = parser.parse_args()

    packed_dir = args.packed_dir
    assert os.path.isdir(packed_dir), f"Not a directory: {packed_dir}"

    # Output alongside packed/
    output_dir = os.path.join(os.path.dirname(packed_dir), "packed_3bit")
    os.makedirs(output_dir, exist_ok=True)

    # Find layer files
    layer_files = sorted([f for f in os.listdir(packed_dir) if f.startswith("layer_") and f.endswith(".bin")])
    num_layers = len(layer_files)

    # Parse header from first file
    first_path = os.path.join(packed_dir, layer_files[0])
    num_experts, expert_size_4bit, tensor_descs = parse_header(first_path)

    print(f"Input: {num_layers} layers, {num_experts} experts, {expert_size_4bit/1024:.0f} KB/expert (4-bit)")
    print(f"Output: {output_dir}")

    total_start = time.time()
    for i, layer_file in enumerate(layer_files):
        input_path = os.path.join(packed_dir, layer_file)
        output_path = os.path.join(output_dir, layer_file)
        t0 = time.time()
        expert_size_3bit = repack_layer(input_path, output_path,
                                        num_experts, expert_size_4bit, tensor_descs)
        elapsed = time.time() - t0
        total = time.time() - total_start
        eta = (total / (i + 1)) * (num_layers - i - 1)
        print(f"  Layer {i:2d}/{num_layers-1}: {elapsed:.1f}s "
              f"({expert_size_3bit/1024:.0f} KB/expert, ETA {eta:.0f}s)")

    total_elapsed = time.time() - total_start
    savings = 1 - expert_size_3bit / expert_size_4bit
    print(f"\nDone! {num_layers} layers repacked in {total_elapsed:.1f}s")
    print(f"4-bit: {expert_size_4bit/1024:.0f} KB/expert → 3-bit: {expert_size_3bit/1024:.0f} KB/expert ({savings*100:.0f}% smaller)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
