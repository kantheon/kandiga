#!/usr/bin/env python3
"""Convert a 4-bit MLX model to mixed 2/3-bit quantization.

Strategy: Keep quality-critical layers at 3-bit, compress bulk MLP at 2-bit.
- First/last N layers: 3-bit (most sensitive to quantization)
- Attention projections: 3-bit (quality-critical for reasoning)
- Embeddings / LM head: 3-bit (vocabulary precision matters)
- MLP layers (middle): 2-bit (largest layers, least sensitive)

Usage:
    python scripts/convert_mixed.py mlx-community/Qwen3.5-27B-4bit ~/.kandiga/models/Qwen3.5-27B-mixed
"""

import argparse
import gc
import json
import os
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load


def convert_mixed(model_path: str, output_path: str,
                  protect_layers: int = 4, group_size: int = 64):
    """Convert with mixed quantization: 3-bit for critical, 2-bit for bulk."""

    print(f"Loading {model_path} (lazy)...")
    model, tokenizer = load(model_path, lazy=True)

    # Detect number of layers
    num_layers = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            if 'layers.' in name:
                layer_num = int(name.split('layers.')[1].split('.')[0])
                num_layers = max(num_layers, layer_num + 1)

    print(f"Model has {num_layers} layers")
    print(f"Protected layers (3-bit): 0-{protect_layers-1} and {num_layers-protect_layers}-{num_layers-1}")
    print(f"Middle layers (2-bit MLP, 3-bit attention): {protect_layers}-{num_layers-protect_layers-1}")

    quant_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            quant_modules.append((name, mod, "linear"))
        elif isinstance(mod, nn.QuantizedEmbedding):
            quant_modules.append((name, mod, "embedding"))

    print(f"Converting {len(quant_modules)} quantized modules...")

    stats = {"3bit": 0, "2bit": 0}
    bits_map = {}  # module_name → bits (for loading)

    for i, (name, mod, mod_type) in enumerate(quant_modules):
        # Decide bits for this module
        bits = 3  # default: 3-bit

        if 'layers.' in name:
            layer_num = int(name.split('layers.')[1].split('.')[0])
            is_protected = (layer_num < protect_layers or
                          layer_num >= num_layers - protect_layers)
            is_mlp = 'mlp' in name

            if is_mlp and not is_protected:
                bits = 2  # Bulk MLP in middle layers → 2-bit
            # Everything else stays 3-bit (attention, protected layers, norms)

        bits_map[name] = bits
        stats[f"{bits}bit"] += 1

        # Evaluate and convert
        mx.eval(mod.weight, mod.scales, mod.biases)
        w_float = mx.dequantize(
            mod.weight, mod.scales, mod.biases,
            group_size=mod.group_size, bits=mod.bits,
        )
        w_q, s_q, b_q = mx.quantize(w_float, group_size=group_size, bits=bits)
        mx.eval(w_q, s_q, b_q)

        mod.weight = w_q
        mod.scales = s_q
        mod.biases = b_q
        mod.bits = bits
        mod.group_size = group_size

        del w_float

        if (i + 1) % 50 == 0 or i == len(quant_modules) - 1:
            gc.collect()
            print(f"  [{i+1}/{len(quant_modules)}] converted "
                  f"(3-bit: {stats['3bit']}, 2-bit: {stats['2bit']})")

    # Collect all parameters
    flat = tree_flatten(model.parameters())
    all_params = {}
    for k, v in flat:
        mx.eval(v)
        all_params[k] = v

    print(f"\nTotal parameters: {len(all_params)}")
    print(f"3-bit modules: {stats['3bit']}, 2-bit modules: {stats['2bit']}")

    # Save
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "model.safetensors")
    print(f"Saving to {out_file}...")
    mx.save_safetensors(out_file, all_params)
    size_gb = os.path.getsize(out_file) / 1e9
    print(f"Saved: {size_gb:.2f} GB")

    # Copy config files
    from huggingface_hub import snapshot_download
    source_dir = snapshot_download(model_path, local_files_only=True)
    for fname in os.listdir(source_dir):
        if fname.endswith(('.json', '.jinja', '.txt', '.model')) and not fname.startswith('model'):
            shutil.copy2(os.path.join(source_dir, fname), os.path.join(output_path, fname))

    # Update config — use 3-bit as the "base" since mlx_lm needs a single bits value
    # The mixed bits are baked into the weight shapes; mlx_lm's QuantizedLinear
    # auto-detects bits from weight shape at load time
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        # Store as 3-bit — modules with 2-bit weights will have different shapes
        # and mlx_lm handles this correctly
        config["quantization"] = {"bits": 3, "group_size": group_size}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Updated config.json (mixed 2/3-bit, group_size={group_size})")

    # Save bits mapping for loader to fix per-module bits
    bits_map_path = os.path.join(output_path, "bits_map.json")
    # Only save 2-bit entries (3-bit is the default)
    bits_2bit = {k: v for k, v in bits_map.items() if v == 2}
    with open(bits_map_path, "w") as f:
        json.dump(bits_2bit, f)
    print(f"Saved bits_map.json ({len(bits_2bit)} 2-bit modules)")

    print(f"\nDone! Mixed-quant model saved to {output_path}")
    print(f"Disk: {size_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert to mixed 2/3-bit quantization")
    parser.add_argument("model_path", help="HuggingFace model ID or local path")
    parser.add_argument("output_path", help="Output directory")
    parser.add_argument("--protect-layers", type=int, default=4,
                       help="Number of first/last layers to keep at 3-bit")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    convert_mixed(args.model_path, args.output_path, args.protect_layers, args.group_size)
