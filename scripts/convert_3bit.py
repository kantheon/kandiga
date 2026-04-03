#!/usr/bin/env python3
"""Convert a 4-bit MLX model to 3-bit.

Usage:
    python scripts/convert_3bit.py mlx-community/Qwen3.5-27B-4bit ~/.kandiga/models/Qwen3.5-27B-3bit
    python scripts/convert_3bit.py mlx-community/Qwen3.5-4B-4bit ~/.kandiga/models/Qwen3.5-4B-3bit

Layer-by-layer conversion to minimize peak memory:
  For each QuantizedLinear: dequant 4-bit → requant 3-bit → replace in model tree
  Then save entire model as safetensors.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load


def convert_to_3bit(model_path: str, output_path: str, group_size: int = 64, bits: int = 3):
    """Convert a 4-bit quantized model to N-bit."""

    print(f"Loading {model_path} (lazy)...")
    t0 = time.time()
    model, tokenizer = load(model_path, lazy=True)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Collect all quantized modules
    quant_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            quant_modules.append((name, mod, "linear"))
        elif isinstance(mod, nn.QuantizedEmbedding):
            quant_modules.append((name, mod, "embedding"))

    print(f"Found {len(quant_modules)} quantized modules to convert")

    # Convert each module: dequant 4-bit → requant 3-bit
    converted = {}
    for i, (name, mod, mod_type) in enumerate(quant_modules):
        # Evaluate this module's weights (loads from lazy)
        mx.eval(mod.weight, mod.scales, mod.biases)

        # Dequantize: 4-bit packed → float
        w_float = mx.dequantize(
            mod.weight, mod.scales, mod.biases,
            group_size=mod.group_size, bits=mod.bits,
        )

        # Requantize: float → 3-bit packed
        w_3bit, scales_3bit, biases_3bit = mx.quantize(w_float, group_size=group_size, bits=bits)
        mx.eval(w_3bit, scales_3bit, biases_3bit)

        # Store in converted dict
        converted[f"{name}.weight"] = w_3bit
        converted[f"{name}.scales"] = scales_3bit
        converted[f"{name}.biases"] = biases_3bit

        # Update module in-place
        mod.weight = w_3bit
        mod.scales = scales_3bit
        mod.biases = biases_3bit
        mod.bits = bits
        mod.group_size = group_size

        # Free intermediate
        del w_float

        if (i + 1) % 50 == 0 or i == len(quant_modules) - 1:
            gc.collect()
            print(f"  [{i+1}/{len(quant_modules)}] converted")

    # Also need non-quantized parameters (norms, biases, etc.)
    flat = tree_flatten(model.parameters())
    all_params = {}
    for k, v in flat:
        if k in converted:
            all_params[k] = converted[k]
        else:
            mx.eval(v)
            all_params[k] = v

    print(f"Total parameters: {len(all_params)}")

    # Save
    os.makedirs(output_path, exist_ok=True)

    # Save weights
    out_file = os.path.join(output_path, "model.safetensors")
    print(f"Saving to {out_file}...")
    mx.save_safetensors(out_file, all_params)

    size_gb = os.path.getsize(out_file) / 1e9
    print(f"Saved: {size_gb:.2f} GB")

    # Copy config with updated quantization
    source_dir = None
    # Resolve HuggingFace cache path
    from huggingface_hub import snapshot_download
    source_dir = snapshot_download(model_path, local_files_only=True)

    # Copy tokenizer and config files
    for fname in os.listdir(source_dir):
        if fname.endswith(('.json', '.jinja', '.txt', '.model')) and not fname.startswith('model'):
            src = os.path.join(source_dir, fname)
            dst = os.path.join(output_path, fname)
            shutil.copy2(src, dst)

    # Update config.json with quantization settings
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["quantization"] = {"bits": bits, "group_size": group_size}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Updated config.json with bits={bits}")

    print(f"\nDone! 3-bit model saved to {output_path}")
    print(f"Disk: {size_gb:.2f} GB")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 4-bit MLX model to 3-bit")
    parser.add_argument("model_path", help="HuggingFace model ID or local path")
    parser.add_argument("output_path", help="Output directory for 3-bit model")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits (2, 3, or 4)")
    args = parser.parse_args()

    convert_to_3bit(args.model_path, args.output_path, args.group_size, args.bits)
