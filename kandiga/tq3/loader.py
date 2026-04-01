"""Load a model with TQ3 weights — drop-in replacement for mlx_lm.load().

Usage:
    from kandiga.tq3.loader import load_tq3_model
    model, tok = load_tq3_model("mlx-community/Qwen3.5-4B-4bit")
    # model has TQ3 weights in all linear layers
    # use exactly like a normal MLX model
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except ImportError:
    raise ImportError("MLX required")

from kandiga.tq3.quantize import dequantize_tensor


TQ3_DIR = os.path.expanduser("~/.kandiga/tq3")


def load_tq3_model(
    model_path: str,
    tq3_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple:
    """Load a model with TQ3 weights.

    If TQ3 weights exist for this model, loads them.
    Otherwise falls back to the original model.

    Returns:
        (model, tokenizer) — same as mlx_lm.load()
    """
    from mlx_lm import load

    model_name = model_path.split("/")[-1]
    if tq3_dir is None:
        tq3_dir = os.path.join(TQ3_DIR, model_name)

    # Check if TQ3 conversion exists
    index_path = os.path.join(tq3_dir, "tq3_index.json")
    if not os.path.isfile(index_path):
        if verbose:
            print(f"No TQ3 weights found for {model_name}, using original")
        return load(model_path)

    if verbose:
        print(f"Loading {model_name} with TQ3 weights...")

    t_start = time.time()

    # Load model structure (lazy — no weights in GPU)
    model, tok = load(model_path, lazy=True)

    # Load TQ3 index and data
    with open(index_path) as f:
        index = json.load(f)

    weights_path = os.path.join(tq3_dir, "tq3_weights.bin")
    with open(weights_path, "rb") as f:
        f.read(8)  # skip header
        all_data = f.read()

    # Load skipped weights
    skip_path = os.path.join(tq3_dir, "skipped_weights.npz")
    skip = np.load(skip_path) if os.path.isfile(skip_path) else {}

    # Materialize skipped weights (embeds, norms, etc)
    flat = list(tree_flatten(model.parameters()))
    for key in skip.files if hasattr(skip, 'files') else []:
        for k, t in flat:
            if k == key:
                mx.eval(t)
                break

    # Replace QuantizedLinear layers with TQ3 dequantized Linear
    replaced = 0
    for key, info in index.items():
        if not key.endswith(".weight"):
            continue

        data = all_data[info["offset"]:info["offset"] + info["length"]]
        shape = tuple(info["shape"])
        nblocks = info["nblocks"]

        w = dequantize_tensor(data, shape, nblocks)
        w_mx = mx.array(np.clip(w, -65504, 65504).astype(np.float16))

        parent = key.rsplit(".", 1)[0]
        parts = parent.split(".")
        obj = model
        parent_obj = None
        last_attr = None
        for part in parts:
            parent_obj = obj
            last_attr = part
            obj = getattr(obj, part) if not part.isdigit() else obj[int(part)]

        if type(obj).__name__ == "QuantizedLinear":
            has_bias = hasattr(obj, "bias") and obj.bias is not None
            new_lin = nn.Linear(w_mx.shape[1], w_mx.shape[0], bias=has_bias)
            new_lin.weight = w_mx
            if has_bias:
                new_lin.bias = obj.bias
            if last_attr.isdigit():
                parent_obj[int(last_attr)] = new_lin
            else:
                setattr(parent_obj, last_attr, new_lin)
            replaced += 1

        del w
        if replaced % 50 == 0:
            mx.clear_cache()

    # Eval everything
    for _, t in tree_flatten(model.parameters()):
        mx.eval(t)

    load_time = time.time() - t_start
    if verbose:
        gpu = mx.get_active_memory() / 1e6
        print(f"  {replaced} layers replaced with TQ3 ({load_time:.0f}s, GPU: {gpu:.0f}MB)")

    return model, tok
