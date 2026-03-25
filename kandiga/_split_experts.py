"""Split stacked MoE expert weights into per-expert safetensors files.

Reads the downloaded Qwen3.5-35B-A3B-4bit model's stacked expert weights
(256 experts per layer, 40 MoE layers) and splits them into individual
per-expert files for selective loading.

Output structure:
  ~/.kandiga/experts/Qwen3.5-35B-A3B-4bit/
    layer_00/
      expert_000.safetensors
      ...
      expert_255.safetensors
    layer_01/
      ...
    layer_39/
      ...
"""

from __future__ import annotations

import json
import os
import time

import mlx.core as mx

WEIGHT_PREFIX = "language_model.model.layers"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
COMPONENTS = ("weight", "scales", "biases")
def _build_weight_map(model_dir: str) -> dict[str, str]:
    """Map tensor names to absolute shard file paths."""
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"No index file found at {index_file}")
    with open(index_file) as f:
        data = json.load(f)
    return {k: os.path.join(model_dir, v) for k, v in data["weight_map"].items()}


def _detect_moe_layers(weight_map: dict[str, str]) -> list[int]:
    """Auto-detect which layers have MoE experts from weight keys."""
    import re
    moe_layers = set()
    for key in weight_map:
        m = re.match(r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.", key)
        if m:
            moe_layers.add(int(m.group(1)))
    return sorted(moe_layers)


def _detect_num_experts(weight_map: dict[str, str], layer_idx: int) -> int:
    """Auto-detect number of experts from tensor shape in weight map."""
    key = f"{WEIGHT_PREFIX}.{layer_idx}.mlp.switch_mlp.gate_proj.weight"
    if key in weight_map:
        t = mx.load(weight_map[key], return_metadata=False)
        if isinstance(t, dict):
            t = t[key] if key in t else list(t.values())[0]
        return t.shape[0]  # first dim is num_experts
    return 256  # fallback


def _split_layer(
    layer_idx: int,
    weight_map: dict[str, str],
    output_dir: str,
) -> None:
    """Split one layer's stacked expert weights into per-expert files."""
    layer_dir = os.path.join(output_dir, f"layer_{layer_idx:02d}")
    os.makedirs(layer_dir, exist_ok=True)

    prefix = f"{WEIGHT_PREFIX}.{layer_idx}.mlp.switch_mlp."

    # Collect all 9 tensor keys for this layer's experts
    tensor_keys = {}
    for proj in PROJECTIONS:
        for comp in COMPONENTS:
            key = f"{prefix}{proj}.{comp}"
            if key not in weight_map:
                raise KeyError(f"Missing weight key: {key}")
            tensor_keys[(proj, comp)] = key

    # Group by shard file to minimize file opens
    shards: dict[str, list[tuple[str, str, str]]] = {}
    for (proj, comp), key in tensor_keys.items():
        shard_file = weight_map[key]
        shards.setdefault(shard_file, []).append((proj, comp, key))

    # Load all stacked tensors for this layer
    stacked: dict[tuple[str, str], mx.array] = {}
    for shard_file, entries in shards.items():
        shard_data = mx.load(shard_file)
        for proj, comp, key in entries:
            tensor = shard_data[key]
            mx.eval(tensor)
            stacked[(proj, comp)] = tensor
        del shard_data

    # Detect num experts from tensor shape (first dim)
    first_tensor = next(iter(stacked.values()))
    num_experts = first_tensor.shape[0]

    # Split and save per-expert files
    for expert_idx in range(num_experts):
        expert_tensors = {}
        for proj in PROJECTIONS:
            for comp in COMPONENTS:
                full_tensor = stacked[(proj, comp)]
                sliced = full_tensor[expert_idx]
                mx.eval(sliced)
                expert_tensors[f"{proj}.{comp}"] = sliced

        out_path = os.path.join(layer_dir, f"expert_{expert_idx:03d}.safetensors")
        mx.save_safetensors(out_path, expert_tensors)
        del expert_tensors

    del stacked


def split_experts(
    model_dir: str,
    output_dir: str,
    num_layers: int | None = None,
) -> None:
    """Split all layers' expert weights into per-expert files."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Building weight map...")
    weight_map = _build_weight_map(model_dir)

    # Auto-detect MoE layers
    moe_layers = _detect_moe_layers(weight_map)
    if not moe_layers:
        raise RuntimeError("No MoE layers found in model weights")

    num_experts = _detect_num_experts(weight_map, moe_layers[0])
    print(f"  Detected {len(moe_layers)} MoE layers, {num_experts} experts per layer")

    total_start = time.time()
    for i, layer_idx in enumerate(moe_layers):
        layer_start = time.time()
        print(f"  Splitting layer {layer_idx:2d}/{moe_layers[-1]}...", end=" ", flush=True)
        _split_layer(layer_idx, weight_map, output_dir)
        elapsed = time.time() - layer_start
        total_elapsed = time.time() - total_start
        eta = (total_elapsed / (i + 1)) * (len(moe_layers) - i - 1)
        print(f"done ({elapsed:.1f}s, ETA {eta:.0f}s)")

    total_elapsed = time.time() - total_start
    total_files = len(moe_layers) * num_experts
    print(f"  {total_files:,} expert files created in {total_elapsed:.1f}s")
