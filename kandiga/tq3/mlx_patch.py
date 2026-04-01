"""Patch MLX model to use TQ3 weights.

Replaces QuantizedLinear layers in an MLX model with TQ3-backed layers.
The model runs normally through MLX but weight storage uses TQ3 format.

Usage:
    from mlx_lm import load
    from kandiga.tq3.mlx_patch import patch_model_tq3

    model, tok = load("mlx-community/Qwen3.5-4B-4bit")
    stats = patch_model_tq3(model, "~/.kandiga/tq3/Qwen3.5-4B-4bit")
    # Model now uses TQ3 weights — runs through MLX as normal
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from kandiga.tq3.quantize import dequantize_tensor


def patch_model_tq3(model, tq3_dir: str, verbose: bool = True) -> Dict:
    """Replace model weights with TQ3-dequantized weights.

    Loads TQ3 weights, dequantizes to float16, and patches them
    into the MLX model. The model runs through MLX as normal but
    with TQ3-quality weights that used less disk space.

    For true memory savings during inference, the Metal kernel
    integration (future work) will keep weights in TQ3 format
    and compute directly.

    Args:
        model: MLX model instance
        tq3_dir: Path to TQ3 model directory
        verbose: Print progress

    Returns:
        Dict with patch stats
    """
    if not HAS_MLX:
        raise ImportError("MLX required")

    tq3_dir = os.path.expanduser(tq3_dir)

    # Load TQ3 index
    index_path = os.path.join(tq3_dir, "tq3_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"TQ3 index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    # Load TQ3 weight data
    tq3_path = os.path.join(tq3_dir, "tq3_weights.bin")
    with open(tq3_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"TQ3M", f"Invalid TQ3 file: {magic}"
        num_layers = int.from_bytes(f.read(4), "little")
        all_data = f.read()

    if verbose:
        print(f"Loading TQ3 weights: {len(index)} layers from {tq3_dir}")

    # Dequantize and patch each layer
    patched = 0
    skipped = 0
    updates = {}

    for key, info in index.items():
        data = all_data[info["offset"]:info["offset"] + info["length"]]
        shape = tuple(info["shape"])
        nblocks = info["nblocks"]

        # Dequantize to float32
        w_float = dequantize_tensor(data, shape, nblocks)

        # Convert to MLX array (float16, clamp to range)
        w_clamped = np.clip(w_float, -65504, 65504)
        w_mx = mx.array(w_clamped.astype(np.float16))

        updates[key] = w_mx
        patched += 1

    # Replace QuantizedLinear layers with regular Linear using TQ3 weights
    if updates:
        _replace_quantized_layers(model, updates, verbose)
        if verbose:
            total_bytes = sum(info["length"] for info in index.values())
            print(f"Patched {patched} layers ({total_bytes/1e6:.0f}MB TQ3 → fp16 in GPU)")

    return {
        "patched": patched,
        "skipped": skipped,
        "tq3_layers": len(index),
    }


def _replace_quantized_layers(model, updates: Dict[str, mx.array], verbose: bool = True):
    """Replace QuantizedLinear layers with Linear layers using TQ3 weights.

    MLX's QuantizedLinear expects uint32 packed data. We need regular
    Linear layers that work with our float16 dequantized weights.
    """
    replaced = 0

    # Group updates by parent module path
    # e.g., "language_model.model.layers.0.self_attn.q_proj.weight"
    # → parent = "language_model.model.layers.0.self_attn.q_proj"
    weight_updates = {}
    for key, tensor in updates.items():
        if key.endswith(".weight"):
            parent_path = key[:-len(".weight")]
            weight_updates[parent_path] = tensor
        elif key.endswith(".scales"):
            # TQ3 already handled scales — skip
            pass

    for parent_path, weight in weight_updates.items():
        # Navigate to the parent module
        parts = parent_path.split(".")
        obj = model
        parent = None
        last_attr = None
        try:
            for part in parts:
                parent = obj
                last_attr = part
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
        except (AttributeError, IndexError, TypeError):
            continue

        # Check if it's a QuantizedLinear
        if hasattr(obj, 'weight') and type(obj).__name__ == 'QuantizedLinear':
            # Create a replacement Linear layer
            out_features, in_features = weight.shape
            has_bias = hasattr(obj, 'bias') and obj.bias is not None

            new_linear = nn.Linear(in_features, out_features, bias=has_bias)
            new_linear.weight = weight

            if has_bias:
                new_linear.bias = obj.bias

            # Replace in parent
            if last_attr.isdigit():
                parent[int(last_attr)] = new_linear
            else:
                setattr(parent, last_attr, new_linear)
            replaced += 1

    if verbose:
        print(f"Replaced {replaced} QuantizedLinear → Linear layers")


def get_tq3_memory_savings(tq3_dir: str) -> Dict:
    """Calculate memory savings from TQ3 conversion."""
    tq3_dir = os.path.expanduser(tq3_dir)
    meta_path = os.path.join(tq3_dir, "tq3_meta.json")

    if not os.path.isfile(meta_path):
        return {}

    with open(meta_path) as f:
        meta = json.load(f)

    return {
        "original_mb": meta.get("original_bytes", 0) / 1e6,
        "tq3_mb": meta.get("tq3_bytes", 0) / 1e6,
        "compression": meta.get("compression", 0),
        "converted_params": meta.get("converted_params", 0),
        "model": meta.get("model_name", ""),
    }
