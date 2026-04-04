#!/usr/bin/env python3
"""Measure cross-token expert cache hit rate.

For each token, checks how many of its selected experts were also
selected by the PREVIOUS token in the same layer. This tells us
how much I/O we could save with a per-layer expert cache.

No engine modifications — pure measurement via logging wrappers.
"""

import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, "/Volumes/Crucial/Users/mousears1090/projects/kandiga")

from kandiga.engine import KandigaEngine

MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
PROMPTS = [
    "Explain how photosynthesis works in detail, covering the light-dependent and light-independent reactions.",
    "Write a Python function to implement a binary search tree with insert, delete, and search operations.",
    "What were the main political, economic, and social causes that led to World War I?",
    "Describe the process of protein folding, including primary through quaternary structure.",
    "Compare and contrast TCP and UDP protocols, including use cases for each.",
    "Explain how a transformer neural network processes text, step by step.",
    "What is the history of jazz music from its origins to modern day?",
    "Describe how CRISPR gene editing works and its potential applications.",
]
MAX_TOKENS = 80  # more tokens = better statistics

# ─── Logging ───
_log_buffer = {}          # layer_idx -> expert list (current token)
_prev_token = {}          # layer_idx -> expert set (previous token)
_hits_per_layer = defaultdict(int)
_total_per_layer = defaultdict(int)
_hit_rates_per_token = []  # per-token hit rate across all layers


def flush_token():
    """Compare current token's experts against previous token's, then advance."""
    global _log_buffer, _prev_token

    if not _log_buffer:
        return

    token_hits = 0
    token_total = 0

    for layer_idx, experts in _log_buffer.items():
        current_set = set(experts)
        prev_set = _prev_token.get(layer_idx, set())

        if prev_set:
            hits = len(current_set & prev_set)
            _hits_per_layer[layer_idx] += hits
            _total_per_layer[layer_idx] += len(current_set)
            token_hits += hits
            token_total += len(current_set)

        _prev_token[layer_idx] = current_set

    if token_total > 0:
        _hit_rates_per_token.append(token_hits / token_total)

    _log_buffer = {}


def reset_for_new_prompt():
    """Reset previous token state between prompts."""
    global _prev_token
    _prev_token = {}


def patch_engine(engine):
    """Patch wrappers to log expert selections."""
    model = engine._model
    if hasattr(model, 'language_model'):
        model = model.language_model
    layers = model.model.layers if hasattr(model, 'model') else model.layers

    patched = 0
    for i, layer in enumerate(layers):
        wrapper = None
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'switch_mlp'):
            wrapper = layer.mlp.switch_mlp
        elif hasattr(layer, 'experts') and hasattr(layer.experts, 'switch_glu'):
            wrapper = layer.experts.switch_glu

        if wrapper is None or not hasattr(wrapper, '_layer_idx'):
            continue

        layer_idx = wrapper._layer_idx

        class LoggingWrapper:
            def __init__(self, inner, lidx):
                self.__dict__['_inner'] = inner
                self.__dict__['_lidx'] = lidx

            def __call__(self, x, indices):
                idx_np = np.array(
                    indices.reshape(-1, indices.shape[-1]), copy=False
                ).astype(np.int32)
                if idx_np.shape[0] == 1:  # decode only
                    _log_buffer[self._lidx] = idx_np[0].tolist()
                return self._inner(x, indices)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        logging_wrapper = LoggingWrapper(wrapper, layer_idx)

        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'switch_mlp'):
            layer.mlp.switch_mlp = logging_wrapper
        elif hasattr(layer, 'experts') and hasattr(layer.experts, 'switch_glu'):
            layer.experts.switch_glu = logging_wrapper

        patched += 1

    print(f"Patched {patched} layers")


def main():
    print("=" * 60)
    print("CROSS-TOKEN EXPERT CACHE HIT RATE MEASUREMENT")
    print("=" * 60)

    engine = KandigaEngine(MODEL)
    engine.load()
    patch_engine(engine)

    total_tokens = 0
    for i, prompt in enumerate(PROMPTS):
        reset_for_new_prompt()
        print(f"\nPrompt {i+1}/{len(PROMPTS)}: {prompt[:60]}...")
        t0 = time.time()
        count = 0
        for token in engine.generate(prompt, max_tokens=MAX_TOKENS, temp=0.0):
            flush_token()
            count += 1
            if count >= MAX_TOKENS:
                break
        elapsed = time.time() - t0
        total_tokens += count
        print(f"  {count} tokens, {elapsed:.1f}s")

    # ─── Results ───
    print("\n" + "=" * 60)
    print("RESULTS: Cross-Token Expert Cache Hit Rate")
    print("=" * 60)

    print(f"\nTokens analyzed: {total_tokens}")
    print(f"K=8 experts per layer, 40 MoE layers")

    # Per-layer hit rates
    print(f"\nPer-layer cache hit rate (% of experts reused from previous token):")
    layer_rates = []
    for lidx in sorted(_hits_per_layer.keys()):
        hits = _hits_per_layer[lidx]
        total = _total_per_layer[lidx]
        rate = hits / total if total > 0 else 0
        layer_rates.append(rate)
        bar = "█" * int(rate * 50)
        print(f"  Layer {lidx:2d}: {rate*100:5.1f}% {bar}")

    avg_rate = np.mean(layer_rates) if layer_rates else 0
    print(f"\n  Average hit rate: {avg_rate*100:.1f}%")

    # Per-token distribution
    if _hit_rates_per_token:
        rates = np.array(_hit_rates_per_token)
        print(f"\nPer-token hit rate distribution:")
        print(f"  Mean:   {rates.mean()*100:.1f}%")
        print(f"  Median: {np.median(rates)*100:.1f}%")
        print(f"  Std:    {rates.std()*100:.1f}%")
        print(f"  Min:    {rates.min()*100:.1f}%")
        print(f"  Max:    {rates.max()*100:.1f}%")

        # Distribution buckets
        print(f"\n  Token distribution by hit rate:")
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                       (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]:
            pct = ((rates >= lo/100) & (rates < hi/100)).mean() * 100
            bar = "█" * int(pct)
            print(f"    {lo:2d}-{hi:2d}%: {pct:5.1f}% of tokens {bar}")

    # I/O savings estimate
    print(f"\n" + "=" * 60)
    print("I/O SAVINGS ESTIMATE")
    print("=" * 60)
    expert_size_kb = 1344  # 3-bit expert size
    experts_per_token = 40 * 8  # 40 layers × K=8
    io_per_token_mb = experts_per_token * expert_size_kb / 1024
    saved_mb = io_per_token_mb * avg_rate
    cache_ram_mb = 40 * 8 * expert_size_kb / 1024

    print(f"  I/O per token (no cache):    {io_per_token_mb:.0f} MB")
    print(f"  I/O per token (with cache):  {io_per_token_mb - saved_mb:.0f} MB")
    print(f"  I/O savings:                 {saved_mb:.0f} MB/token ({avg_rate*100:.0f}%)")
    print(f"  Cache RAM cost:              {cache_ram_mb:.0f} MB")
    print(f"")

    # Speed estimate
    current_tps = 12  # current tok/s estimate
    # Speed is roughly proportional to 1/IO (I/O bound)
    if avg_rate < 1.0:
        estimated_tps = current_tps / (1 - avg_rate)
        print(f"  Current speed (estimated):   ~{current_tps} tok/s")
        print(f"  With cache (estimated):      ~{estimated_tps:.0f} tok/s")
        print(f"  Speedup:                     {estimated_tps/current_tps:.1f}x")
    else:
        print(f"  100% hit rate — all experts cached (impossible in practice)")


if __name__ == "__main__":
    main()
