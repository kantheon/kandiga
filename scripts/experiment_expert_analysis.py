#!/usr/bin/env python3
"""Research experiment: Cross-layer expert correlation & adaptive K analysis.

Instruments SEM inference to log every expert selection decision,
then analyzes:
1. Cross-layer correlation — why does speculation get 77% accuracy?
2. Router confidence distribution — can we predict when fewer experts suffice?
3. Adaptive K potential — what % of tokens could use K=2 or K=1?
4. Expert popularity — are some experts "universal" vs "specialized"?

Usage:
    python scripts/experiment_expert_analysis.py
"""

import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, "/Volumes/Crucial/Users/mousears1090/projects/kandiga")

from kandiga.engine import KandigaEngine

# ─── Configuration ───
MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
PROMPTS = [
    "Explain how photosynthesis works in detail.",
    "Write a Python function to find the longest common subsequence.",
    "What were the main causes of World War I?",
    "Describe the process of protein folding.",
    "Compare and contrast TCP and UDP protocols.",
]
MAX_TOKENS = 50  # per prompt — enough for statistical analysis


# ─── Logging infrastructure ───
class ExpertLogger:
    """Captures expert selections per layer per token."""

    def __init__(self):
        self.selections = []       # list of {layer_idx: [expert_ids]} per token
        self.gate_logits = []      # list of {layer_idx: softmax_probs} per token
        self._current_token = {}
        self._current_logits = {}
        self._token_count = 0

    def log_selection(self, layer_idx, expert_indices, gate_probs=None):
        self._current_token[layer_idx] = list(expert_indices)
        if gate_probs is not None:
            self._current_logits[layer_idx] = gate_probs

    def next_token(self):
        if self._current_token:
            self.selections.append(dict(self._current_token))
            self.gate_logits.append(dict(self._current_logits))
            self._current_token = {}
            self._current_logits = {}
            self._token_count += 1

    @property
    def num_tokens(self):
        return len(self.selections)


logger = ExpertLogger()


def patch_engine(engine):
    """Monkey-patch _CPUSwitchGLU wrappers to log expert selections."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    # Find all _CPUSwitchGLU instances
    patched = 0
    layers = None
    model = engine._model
    if hasattr(model, 'language_model'):
        model = model.language_model
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers

    if layers is None:
        print("ERROR: Could not find model layers")
        return

    for i, layer in enumerate(layers):
        # Find the switch_mlp or switch_glu wrapper
        wrapper = None
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'switch_mlp'):
            wrapper = layer.mlp.switch_mlp
        elif hasattr(layer, 'experts') and hasattr(layer.experts, 'switch_glu'):
            wrapper = layer.experts.switch_glu

        if wrapper is None or not hasattr(wrapper, '_layer_idx'):
            continue  # Not a SEM wrapper

        # Store reference to gate weights for recomputing probabilities
        gate_np = getattr(wrapper, '_next_gate_np', None)

        # Patch __call__ to add logging
        original_call = wrapper.__class__.__call__

        def make_patched(orig_call, w, gate):
            def patched_call(self_inner, x, indices):
                # Log the expert selections
                idx_np = np.array(indices.reshape(-1, indices.shape[-1]), copy=False).astype(np.int32)
                num_tokens = idx_np.shape[0]

                if num_tokens == 1:
                    # Decode: log single token selection
                    experts = idx_np[0].tolist()

                    # Recompute router probabilities from hidden state × gate weights
                    probs = None
                    if gate is not None:
                        try:
                            x_np = np.array(x.reshape(-1, self_inner._hidden_size), copy=False)
                            x_f32 = x_np[-1].astype(np.float32)
                            logits = x_f32 @ gate.T
                            logits -= logits.max()
                            exp_logits = np.exp(logits)
                            probs = exp_logits / exp_logits.sum()
                        except Exception:
                            pass

                    logger.log_selection(self_inner._layer_idx, experts, probs)

                # Call original
                return orig_call(self_inner, x, indices)
            return patched_call

        # We need the CURRENT layer's gate, not next layer's
        # The wrapper stores next_gate_np. For the current layer, we need
        # the gate from the wrapper installed on the PREVIOUS layer.
        # For now, use next_gate_np from previous wrapper as current layer's gate.
        # Actually, let's extract the gate directly from the model.
        current_gate_np = None
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            gate_mod = layer.mlp.gate
            if hasattr(gate_mod, 'weight') and hasattr(gate_mod, 'scales'):
                try:
                    w = gate_mod.weight
                    s = gate_mod.scales
                    b = getattr(gate_mod, 'biases', None)
                    mx.eval(w, s)
                    if b is not None:
                        mx.eval(b)
                    dequant = mx.dequantize(w, s, b, group_size=64, bits=gate_mod.bits)
                    mx.eval(dequant)
                    current_gate_np = np.array(dequant.astype(mx.float16), copy=False).astype(np.float32)
                except Exception:
                    pass
        elif hasattr(layer, 'router') and hasattr(layer.router, 'proj'):
            gate_mod = layer.router.proj
            if hasattr(gate_mod, 'weight') and hasattr(gate_mod, 'scales'):
                try:
                    w = gate_mod.weight
                    s = gate_mod.scales
                    b = getattr(gate_mod, 'biases', None)
                    mx.eval(w, s)
                    if b is not None:
                        mx.eval(b)
                    dequant = mx.dequantize(w, s, b, group_size=64, bits=gate_mod.bits)
                    mx.eval(dequant)
                    current_gate_np = np.array(dequant.astype(mx.float16), copy=False).astype(np.float32)
                except Exception:
                    pass

        wrapper.__class__.__call__ = make_patched(original_call, wrapper, current_gate_np)
        patched += 1

    print(f"Patched {patched} MoE layers for logging")


def analyze_cross_layer_correlation(selections, num_layers):
    """Compute Jaccard similarity between expert sets of adjacent layers."""
    print("\n" + "=" * 60)
    print("1. CROSS-LAYER EXPERT CORRELATION")
    print("=" * 60)

    # Build correlation matrix
    layer_indices = sorted(set(l for s in selections for l in s.keys()))
    n = len(layer_indices)
    jaccard_matrix = np.zeros((n, n))
    jaccard_counts = np.zeros((n, n))

    for token_sel in selections:
        for i, li in enumerate(layer_indices):
            if li not in token_sel:
                continue
            set_i = set(token_sel[li])
            for j, lj in enumerate(layer_indices):
                if lj not in token_sel:
                    continue
                set_j = set(token_sel[lj])
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                if union > 0:
                    jaccard_matrix[i, j] += intersection / union
                    jaccard_counts[i, j] += 1

    # Average
    mask = jaccard_counts > 0
    jaccard_matrix[mask] /= jaccard_counts[mask]

    # Print adjacent-layer correlations
    print(f"\nAdjacent layer Jaccard similarity (K experts overlap):")
    adjacent_scores = []
    for i in range(n - 1):
        score = jaccard_matrix[i, i + 1]
        adjacent_scores.append(score)
        bar = "█" * int(score * 40)
        print(f"  Layer {layer_indices[i]:2d} → {layer_indices[i+1]:2d}: {score:.3f} {bar}")

    avg_adjacent = np.mean(adjacent_scores) if adjacent_scores else 0
    print(f"\n  Average adjacent-layer Jaccard: {avg_adjacent:.3f}")
    print(f"  This explains ~{avg_adjacent*100:.0f}% expert overlap between consecutive layers")

    # Distance analysis
    print(f"\nCorrelation by layer distance:")
    for dist in [1, 2, 3, 5, 10]:
        scores = []
        for i in range(n - dist):
            if jaccard_counts[i, i + dist] > 0:
                scores.append(jaccard_matrix[i, i + dist])
        if scores:
            print(f"  Distance {dist:2d}: {np.mean(scores):.3f} (±{np.std(scores):.3f})")

    return jaccard_matrix, layer_indices


def analyze_router_confidence(gate_logits, selections):
    """Analyze router softmax entropy and confidence."""
    print("\n" + "=" * 60)
    print("2. ROUTER CONFIDENCE ANALYSIS")
    print("=" * 60)

    layer_entropies = defaultdict(list)
    layer_top1_probs = defaultdict(list)
    layer_top2_mass = defaultdict(list)
    layer_topk_mass = defaultdict(list)  # mass in selected top-K

    for token_idx, (logits, sels) in enumerate(zip(gate_logits, selections)):
        for layer_idx, probs in logits.items():
            if probs is None:
                continue
            # Entropy
            p = probs[probs > 0]
            entropy = -np.sum(p * np.log2(p))
            layer_entropies[layer_idx].append(entropy)

            # Top-1 probability
            sorted_p = np.sort(probs)[::-1]
            layer_top1_probs[layer_idx].append(sorted_p[0])

            # Top-2 cumulative probability
            layer_top2_mass[layer_idx].append(sorted_p[:2].sum())

            # Top-K mass (how much probability mass is in the selected experts)
            if layer_idx in sels:
                selected = sels[layer_idx]
                k_mass = sum(probs[e] for e in selected if e < len(probs))
                layer_topk_mass[layer_idx].append(k_mass)

    if not layer_entropies:
        print("  No router probability data captured")
        return

    print(f"\nPer-layer router entropy (lower = more confident):")
    print(f"  Max possible entropy: {np.log2(256):.1f} bits (uniform over 256 experts)")
    for layer_idx in sorted(layer_entropies.keys()):
        ent = layer_entropies[layer_idx]
        top1 = layer_top1_probs.get(layer_idx, [0])
        top2 = layer_top2_mass.get(layer_idx, [0])
        print(f"  Layer {layer_idx:2d}: entropy={np.mean(ent):.2f}±{np.std(ent):.2f} bits  "
              f"top1={np.mean(top1):.3f}  top2={np.mean(top2):.3f}")

    # Adaptive K analysis
    print(f"\n--- ADAPTIVE K POTENTIAL ---")
    all_top1 = []
    all_top2 = []
    all_topk = []
    for layer_idx in sorted(layer_top1_probs.keys()):
        all_top1.extend(layer_top1_probs[layer_idx])
        all_top2.extend(layer_top2_mass[layer_idx])
        all_topk.extend(layer_topk_mass.get(layer_idx, []))

    if all_top1:
        all_top1 = np.array(all_top1)
        all_top2 = np.array(all_top2)

        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            pct_top1 = (all_top1 >= threshold).mean() * 100
            pct_top2 = (all_top2 >= threshold).mean() * 100
            print(f"  Confidence ≥ {threshold:.1f}: {pct_top1:.1f}% tokens could use K=1, "
                  f"{pct_top2:.1f}% could use K=2")

        print(f"\n  If adaptive K used confidence ≥ 0.7 threshold:")
        k1_pct = (all_top1 >= 0.7).mean() * 100
        k2_pct = ((all_top2 >= 0.7) & (all_top1 < 0.7)).mean() * 100
        k8_pct = 100 - k1_pct - k2_pct
        avg_k = (k1_pct * 1 + k2_pct * 2 + k8_pct * 8) / 100
        speedup = 8 / avg_k
        print(f"    K=1: {k1_pct:.1f}% of tokens")
        print(f"    K=2: {k2_pct:.1f}% of tokens")
        print(f"    K=8: {k8_pct:.1f}% of tokens")
        print(f"    Average K: {avg_k:.2f} (vs fixed K=8)")
        print(f"    Theoretical I/O speedup: {speedup:.2f}x (fewer expert reads)")


def analyze_expert_popularity(selections):
    """Which experts are used most? Are there universal vs specialized experts?"""
    print("\n" + "=" * 60)
    print("3. EXPERT POPULARITY ANALYSIS")
    print("=" * 60)

    layer_expert_counts = defaultdict(lambda: defaultdict(int))
    total_per_layer = defaultdict(int)

    for token_sel in selections:
        for layer_idx, experts in token_sel.items():
            for e in experts:
                layer_expert_counts[layer_idx][e] += 1
            total_per_layer[layer_idx] += 1

    # Per-layer analysis
    print(f"\nExpert usage distribution per layer:")
    for layer_idx in sorted(layer_expert_counts.keys())[:5]:  # First 5 layers
        counts = layer_expert_counts[layer_idx]
        total = total_per_layer[layer_idx]
        sorted_experts = sorted(counts.items(), key=lambda x: -x[1])
        top5 = sorted_experts[:5]
        num_used = len(counts)
        max_count = sorted_experts[0][1] if sorted_experts else 0
        min_count = sorted_experts[-1][1] if sorted_experts else 0

        print(f"  Layer {layer_idx:2d}: {num_used}/256 experts used, "
              f"top5: {[f'E{e}({c})' for e, c in top5]}")

    # Cross-layer "universal" experts — appear in many layers
    expert_layer_presence = defaultdict(set)
    for token_sel in selections:
        for layer_idx, experts in token_sel.items():
            for e in experts:
                expert_layer_presence[e].add(layer_idx)

    num_layers = len(set(l for s in selections for l in s.keys()))
    universal = [(e, len(layers)) for e, layers in expert_layer_presence.items()
                 if len(layers) >= num_layers * 0.8]
    universal.sort(key=lambda x: -x[1])

    if universal:
        print(f"\n  'Universal' experts (appear in ≥80% of layers): {len(universal)}")
        for e, n in universal[:10]:
            print(f"    Expert {e}: present in {n}/{num_layers} layers")
    else:
        print(f"\n  No universal experts found (all are layer-specialized)")


def main():
    print("=" * 60)
    print("KANDIGA RESEARCH EXPERIMENT: Expert Selection Analysis")
    print("=" * 60)

    # Load engine
    print(f"\nLoading {MODEL}...")
    engine = KandigaEngine(MODEL)
    engine.load()

    # Patch for logging
    patch_engine(engine)

    # Generate tokens across multiple prompts
    print(f"\nGenerating {MAX_TOKENS} tokens each for {len(PROMPTS)} prompts...")
    total_tokens = 0
    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}: {prompt[:50]}... ---")
        t0 = time.time()

        # Patch token boundary detection
        # We detect new tokens by checking if layer 0 is logged again
        seen_layers = set()
        original_log = logger.log_selection

        def boundary_aware_log(layer_idx, experts, probs=None):
            nonlocal seen_layers
            if layer_idx in seen_layers:
                # New token — layer repeated
                logger.next_token()
                seen_layers.clear()
            seen_layers.add(layer_idx)
            original_log(layer_idx, experts, probs)

        logger.log_selection = boundary_aware_log

        token_count = 0
        for token in engine.generate(prompt, max_tokens=MAX_TOKENS, temp=0.0):
            token_count += 1
            if token_count >= MAX_TOKENS:
                break

        # Flush last token
        logger.next_token()
        seen_layers.clear()
        logger.log_selection = original_log

        elapsed = time.time() - t0
        print(f"  {token_count} tokens in {elapsed:.1f}s ({token_count/elapsed:.1f} tok/s)")
        total_tokens += token_count

    print(f"\nTotal tokens logged: {logger.num_tokens}")
    print(f"Layers per token: {len(logger.selections[0]) if logger.selections else 0}")

    # Run analyses
    num_layers = len(set(l for s in logger.selections for l in s.keys()))
    jaccard, layer_ids = analyze_cross_layer_correlation(logger.selections, num_layers)
    analyze_router_confidence(logger.gate_logits, logger.selections)
    analyze_expert_popularity(logger.selections)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — PUBLISHABLE FINDINGS")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Tokens analyzed: {logger.num_tokens}")
    print(f"MoE layers: {num_layers}")
    print(f"Experts per layer: 256, K={8}")


if __name__ == "__main__":
    main()
