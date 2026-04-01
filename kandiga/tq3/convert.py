"""Convert a model's weights from safetensors/fp16 to TQ3 format.

Usage:
    python -m kandiga.tq3.convert --model mlx-community/Qwen3.5-4B-4bit --output ~/.kandiga/tq3/Qwen3.5-4B
    python -m kandiga.tq3.convert --model mlx-community/Qwen3.5-27B --output ~/.kandiga/tq3/Qwen3.5-27B

This converts all linear layer weights to TQ3_1S format (4.0 bpw).
Non-linear weights (embeddings, norms, biases) are kept as-is.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def convert_safetensors_to_tq3(
    model_path: str,
    output_dir: str,
    skip_layers: Optional[list] = None,
    verbose: bool = True,
) -> Dict[str, any]:
    """Convert a model's weights to TQ3 format.

    Args:
        model_path: HuggingFace model ID or local path
        output_dir: Directory to save TQ3 weights
        skip_layers: Layer name patterns to skip (keep as fp16)
        verbose: Print progress

    Returns:
        Dict with conversion stats
    """
    from kandiga.tq3.quantize import quantize_tensor

    if skip_layers is None:
        skip_layers = ["embed", "norm", "lm_head", "rotary", "bias"]

    os.makedirs(output_dir, exist_ok=True)

    # Load model weights
    if verbose:
        print(f"Loading model: {model_path}")

    try:
        from safetensors import safe_open
        from huggingface_hub import snapshot_download

        # Download if needed
        if not os.path.isdir(model_path):
            model_dir = snapshot_download(model_path)
        else:
            model_dir = model_path

        # Find safetensors files
        st_files = sorted(Path(model_dir).glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files in {model_dir}")

        if verbose:
            print(f"Found {len(st_files)} safetensors files")

    except ImportError:
        print("Error: pip install safetensors huggingface_hub")
        return {}

    stats = {
        "model": model_path,
        "total_params": 0,
        "quantized_params": 0,
        "skipped_params": 0,
        "original_bytes": 0,
        "tq3_bytes": 0,
        "layers": {},
    }

    # Process each safetensors file
    tq3_data = {}
    fp16_data = {}

    for st_file in st_files:
        if verbose:
            print(f"Processing {st_file.name}...")

        with safe_open(str(st_file), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                num_params = tensor.size
                stats["total_params"] += num_params
                stats["original_bytes"] += tensor.nbytes

                # Decide: quantize or keep as-is
                should_skip = any(s in key.lower() for s in skip_layers)
                is_1d = tensor.ndim < 2
                is_small = num_params < 256  # too small to quantize

                if should_skip or is_1d or is_small:
                    # Keep as fp16
                    fp16_data[key] = tensor.astype(np.float16)
                    stats["skipped_params"] += num_params
                    if verbose and num_params > 1000:
                        print(f"  SKIP {key} ({tensor.shape})")
                else:
                    # Quantize to TQ3
                    t0 = time.time()
                    data, shape, nblocks = quantize_tensor(tensor.astype(np.float32))
                    qt = time.time() - t0

                    tq3_data[key] = {
                        "data": data,
                        "shape": shape,
                        "nblocks": nblocks,
                    }
                    stats["quantized_params"] += num_params
                    stats["tq3_bytes"] += len(data)
                    stats["layers"][key] = {
                        "shape": list(shape),
                        "original_bytes": tensor.nbytes,
                        "tq3_bytes": len(data),
                        "compression": tensor.nbytes / len(data),
                        "time": qt,
                    }

                    if verbose:
                        comp = tensor.nbytes / len(data)
                        print(f"  TQ3  {key} ({tensor.shape}) {comp:.1f}x compression, {qt:.1f}s")

    # Save TQ3 weights
    tq3_path = os.path.join(output_dir, "weights.tq3")
    if verbose:
        print(f"\nSaving TQ3 weights to {tq3_path}...")

    with open(tq3_path, "wb") as f:
        # Magic + version
        f.write(b"TQ3W")
        f.write(struct.pack("<I", 1))  # version

        # Number of TQ3 tensors
        f.write(struct.pack("<I", len(tq3_data)))

        # Index: name -> offset
        index = {}
        data_offset = 0
        for key in sorted(tq3_data.keys()):
            entry = tq3_data[key]
            index[key] = {
                "offset": data_offset,
                "length": len(entry["data"]),
                "shape": entry["shape"],
                "nblocks": entry["nblocks"],
            }
            data_offset += len(entry["data"])

        # Write index as JSON
        index_json = json.dumps(index).encode()
        f.write(struct.pack("<I", len(index_json)))
        f.write(index_json)

        # Write tensor data
        for key in sorted(tq3_data.keys()):
            f.write(tq3_data[key]["data"])

    # Save fp16 weights that weren't quantized
    fp16_path = os.path.join(output_dir, "weights_fp16.npz")
    if fp16_data:
        if verbose:
            print(f"Saving {len(fp16_data)} fp16 tensors to {fp16_path}...")
        np.savez_compressed(fp16_path, **fp16_data)

    # Save stats
    stats_path = os.path.join(output_dir, "tq3_stats.json")
    total_tq3 = stats["tq3_bytes"] + sum(t.nbytes for t in fp16_data.values())
    stats["total_tq3_bytes"] = total_tq3
    stats["overall_compression"] = stats["original_bytes"] / total_tq3 if total_tq3 > 0 else 0

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    if verbose:
        print(f"\n=== Conversion Complete ===")
        print(f"Model: {model_path}")
        print(f"Total params: {stats['total_params']:,}")
        print(f"Quantized: {stats['quantized_params']:,} ({stats['quantized_params']/stats['total_params']*100:.1f}%)")
        print(f"Skipped: {stats['skipped_params']:,}")
        print(f"Original: {stats['original_bytes']/1e9:.2f} GB")
        print(f"TQ3 total: {total_tq3/1e9:.2f} GB")
        print(f"Compression: {stats['overall_compression']:.2f}x")
        print(f"Output: {output_dir}")

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert model weights to TQ3")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--skip", nargs="*", default=None, help="Layer patterns to skip")
    args = parser.parse_args()

    convert_safetensors_to_tq3(args.model, args.output, skip_layers=args.skip)


if __name__ == "__main__":
    main()
