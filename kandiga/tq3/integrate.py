"""Integrate TQ3 into Kandiga's inference pipeline.

Converts model weights to TQ3 and replaces MLX linear layers
with TQ3Linear layers for reduced memory and (eventually) faster inference.

Usage:
    # Convert 4B model
    python -m kandiga.tq3.integrate --model mlx-community/Qwen3.5-4B-4bit

    # Convert 35B MoE shared layers only (experts stay on SSD)
    python -m kandiga.tq3.integrate --model mlx-community/Qwen3.5-35B-A3B-4bit --moe-shared-only
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from kandiga.tq3.quantize import quantize_tensor, dequantize_tensor
from kandiga.tq3.engine import TQ3Linear


TQ3_CACHE_DIR = os.path.expanduser("~/.kandiga/tq3")


def get_model_layers(model) -> List[Tuple[str, any]]:
    """Get all linear layer weights from an MLX model."""
    flat = tree_flatten(model.parameters())
    linear_layers = []
    for key, tensor in flat:
        if tensor.ndim == 2 and tensor.size >= 512:
            linear_layers.append((key, tensor))
    return linear_layers


def convert_model_to_tq3(
    model_path: str,
    output_dir: Optional[str] = None,
    moe_shared_only: bool = False,
    skip_patterns: Optional[List[str]] = None,
    verbose: bool = True,
) -> str:
    """Convert a model's linear layers to TQ3 format.

    Args:
        model_path: HuggingFace model ID
        output_dir: Where to save (default: ~/.kandiga/tq3/<model_name>)
        moe_shared_only: Only convert shared layers (skip expert weights)
        skip_patterns: Layer name patterns to skip

    Returns:
        Path to the saved TQ3 model directory
    """
    if not HAS_MLX:
        raise ImportError("MLX required for model conversion")

    from mlx_lm import load

    model_name = model_path.split("/")[-1]
    if output_dir is None:
        output_dir = os.path.join(TQ3_CACHE_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)

    if skip_patterns is None:
        skip_patterns = ["embed", "norm", "lm_head", "rotary", "bias"]
    if moe_shared_only:
        skip_patterns.append("switch_mlp")  # skip expert weights
        skip_patterns.append("expert")

    if verbose:
        print(f"Loading model: {model_path}")

    model, tokenizer = load(model_path)

    flat = tree_flatten(model.parameters())
    total_params = sum(t.size for _, t in flat)
    if verbose:
        print(f"Total parameters: {total_params:,}")

    # Convert each linear layer
    tq3_layers = {}
    skipped_layers = {}
    converted_params = 0
    skipped_params = 0
    original_bytes = 0
    tq3_bytes = 0

    # Build a map of quantized layers: weight + scales + biases → full matrix
    # MLX QuantizedLinear stores: weight (packed uint), scales (fp16), biases (fp16)
    # We need to dequantize via MLX to get the full float matrix
    quant_groups = {}  # parent_path → {weight, scales, biases}
    for key, tensor in flat:
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            parent, attr = parts
            if attr in ("weight", "scales", "biases"):
                quant_groups.setdefault(parent, {})[attr] = (key, tensor)

    for key, tensor in flat:
        should_skip = (
            tensor.ndim < 2 or
            tensor.size < 512 or
            any(p in key.lower() for p in skip_patterns)
        )

        # Skip scales/biases — they're handled with their parent weight
        if key.endswith(".scales") or key.endswith(".biases"):
            skipped_layers[key] = tensor
            skipped_params += tensor.size
            continue

        if should_skip:
            skipped_layers[key] = tensor
            skipped_params += tensor.size
            continue

        # Check if this is a quantized weight that needs dequantization
        parent_path = key.rsplit(".", 1)[0] if "." in key else ""
        group = quant_groups.get(parent_path, {})

        if "scales" in group and key.endswith(".weight"):
            # This is a QuantizedLinear weight — dequantize via MLX first
            scales_tensor = group["scales"][1]
            biases_tensor = group.get("biases", (None, None))[1]

            try:
                w_full = mx.dequantize(tensor, scales_tensor, biases_tensor)
                mx.eval(w_full)
                w = np.array(w_full.astype(mx.float32))
                del w_full  # free GPU memory immediately
                mx.clear_cache()
            except Exception as e:
                if verbose:
                    print(f"  WARN: dequantize failed for {key}: {e}, using raw")
                w = np.array(tensor.astype(mx.float32))
        else:
            # Regular tensor — convert directly
            w = np.array(tensor.astype(mx.float32))

        orig_size = w.size * 2  # fp16 equivalent
        original_bytes += orig_size

        t0 = time.time()
        data, shape, nblocks = quantize_tensor(w)
        qt = time.time() - t0

        tq3_bytes += len(data)
        converted_params += w.size

        tq3_layers[key] = {
            "data": data,
            "shape": shape,
            "nblocks": nblocks,
        }

        if verbose:
            comp = orig_size / len(data)
            print(f"  TQ3 {key} ({w.shape}) {comp:.1f}x {qt:.1f}s")

    # Save TQ3 weights
    tq3_path = os.path.join(output_dir, "tq3_weights.bin")
    index = {}
    offset = 0

    with open(tq3_path, "wb") as f:
        f.write(b"TQ3M")  # magic
        f.write(len(tq3_layers).to_bytes(4, "little"))

        # Write all layer data
        for key in sorted(tq3_layers.keys()):
            entry = tq3_layers[key]
            data = entry["data"]
            index[key] = {
                "offset": offset,
                "length": len(data),
                "shape": list(entry["shape"]),
                "nblocks": entry["nblocks"],
            }
            f.write(data)
            offset += len(data)

    # Save index
    index_path = os.path.join(output_dir, "tq3_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Save skipped layers as npz
    if skipped_layers:
        skip_path = os.path.join(output_dir, "skipped_weights.npz")
        skip_np = {}
        for key, tensor in skipped_layers.items():
            if HAS_MLX and isinstance(tensor, mx.array):
                # Convert MLX array to numpy — handle bfloat16
                if tensor.dtype == mx.bfloat16:
                    skip_np[key] = np.array(tensor.astype(mx.float16))
                else:
                    try:
                        skip_np[key] = np.array(tensor)
                    except (RuntimeError, TypeError):
                        skip_np[key] = np.array(tensor.astype(mx.float32))
            else:
                skip_np[key] = np.array(tensor)
        np.savez_compressed(skip_path, **skip_np)

    # Save metadata
    meta = {
        "model_path": model_path,
        "model_name": model_name,
        "total_params": total_params,
        "converted_params": converted_params,
        "skipped_params": skipped_params,
        "original_bytes": original_bytes,
        "tq3_bytes": tq3_bytes,
        "compression": original_bytes / tq3_bytes if tq3_bytes > 0 else 0,
        "moe_shared_only": moe_shared_only,
        "skip_patterns": skip_patterns,
    }
    meta_path = os.path.join(output_dir, "tq3_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\n=== Conversion Complete ===")
        print(f"Converted: {converted_params:,} params ({converted_params/total_params*100:.1f}%)")
        print(f"Skipped: {skipped_params:,} params")
        print(f"Original (fp16): {original_bytes/1e6:.1f} MB")
        print(f"TQ3: {tq3_bytes/1e6:.1f} MB")
        print(f"Compression: {original_bytes/tq3_bytes:.1f}x")
        print(f"Output: {output_dir}")

    return output_dir


def load_tq3_model(tq3_dir: str) -> Dict:
    """Load a TQ3-converted model.

    Returns dict with 'tq3_layers' (TQ3Linear objects) and 'skipped_layers' (numpy arrays).
    """
    # Load index
    index_path = os.path.join(tq3_dir, "tq3_index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Load TQ3 weight data
    tq3_path = os.path.join(tq3_dir, "tq3_weights.bin")
    with open(tq3_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"TQ3M"
        num_layers = int.from_bytes(f.read(4), "little")
        all_data = f.read()

    tq3_layers = {}
    for key, info in index.items():
        data = all_data[info["offset"]:info["offset"] + info["length"]]
        shape = tuple(info["shape"])
        nblocks = info["nblocks"]
        tq3_layers[key] = TQ3Linear(data, shape, nblocks)

    # Load skipped layers
    skipped = {}
    skip_path = os.path.join(tq3_dir, "skipped_weights.npz")
    if os.path.exists(skip_path):
        with np.load(skip_path) as npz:
            for key in npz.files:
                skipped[key] = npz[key]

    # Load metadata
    meta = {}
    meta_path = os.path.join(tq3_dir, "tq3_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return {
        "tq3_layers": tq3_layers,
        "skipped_layers": skipped,
        "meta": meta,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert model to TQ3")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--moe-shared-only", action="store_true",
                        help="Only convert shared layers (skip experts)")
    args = parser.parse_args()

    convert_model_to_tq3(args.model, args.output, args.moe_shared_only)


if __name__ == "__main__":
    main()
