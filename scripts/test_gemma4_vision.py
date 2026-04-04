#!/usr/bin/env python3
"""Test Gemma 4 vision via SEM — one image, minimal RAM."""

import time
import sys
sys.path.insert(0, "/Volumes/Crucial/Users/mousears1090/projects/kandiga")

from kandiga.engine import KandigaEngine

IMAGE = "/tmp/test_vision.png"
PROMPT = "What do you see in this image? Describe it briefly."

print("=== Gemma 4 Vision via SEM ===")
print(f"Image: {IMAGE}")
print()

engine = KandigaEngine("mlx-community/gemma-4-26b-a4b-it-4bit")
t0 = time.time()
engine.load()
print(f"\nEngine loaded in {time.time()-t0:.1f}s")

# Check vision is available
print(f"Has _vlm_model: {hasattr(engine, '_vlm_model') and engine._vlm_model is not None}")
print(f"Has _vlm_processor: {hasattr(engine, '_vlm_processor') and engine._vlm_processor is not None}")
print(f"Has _orig_lm_call: {hasattr(engine, '_orig_lm_call')}")

import mlx.core as mx
print(f"GPU memory: {mx.get_peak_memory() / 1e9:.2f} GB")
print()

# Test vision
print("Generating with image...")
t1 = time.time()
try:
    result = engine.generate_with_image(IMAGE, PROMPT, max_tokens=300)
    elapsed = time.time() - t1
    print(f"\n--- Result ({elapsed:.1f}s) ---")
    print(result)
    print(f"\nGPU memory after vision: {mx.get_peak_memory() / 1e9:.2f} GB")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nFailed after {time.time()-t1:.1f}s: {e}")
