#!/usr/bin/env python3
"""Benchmark: cross-token expert cache impact on decode speed."""

import sys
import time

sys.path.insert(0, "/Volumes/Crucial/Users/mousears1090/projects/kandiga")

from kandiga.engine import KandigaEngine

MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
PROMPT = "Explain the complete process of how a CPU executes an instruction, from fetch to writeback, including pipelining and branch prediction."
MAX_TOKENS = 100

print("=" * 60)
print("BENCHMARK: Cross-Token Expert Cache")
print("=" * 60)

engine = KandigaEngine(MODEL)
t0 = time.time()
engine.load()
print(f"Engine loaded in {time.time()-t0:.1f}s")

# Warm up
print("\nWarm-up run (10 tokens)...")
for i, tok in enumerate(engine.generate("Hello", max_tokens=10, temp=0.0)):
    pass

# Benchmark
print(f"\nGenerating {MAX_TOKENS} tokens...")
print(f"Prompt: {PROMPT[:60]}...")

tokens = []
t0 = time.time()
for tok in engine.generate(PROMPT, max_tokens=MAX_TOKENS, temp=0.0):
    tokens.append(tok)
    if len(tokens) >= MAX_TOKENS:
        break
elapsed = time.time() - t0

text = "".join(tokens)
tps = len(tokens) / elapsed

print(f"\n--- Results ---")
print(f"Tokens: {len(tokens)}")
print(f"Time: {elapsed:.2f}s")
print(f"Speed: {tps:.2f} tok/s")
print(f"Output: {text[:200]}...")
print(f"\n(Cache stats will be printed when engine is destroyed)")

del engine  # triggers destroy → prints cache stats
