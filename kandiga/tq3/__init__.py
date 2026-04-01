"""TQ3 — TurboQuant 3-bit weight quantization for Apple Silicon.

Applies the same rotation + quantization technique used in KV cache compression
(TurboQuant/PolarQuant) to MODEL WEIGHTS. 3.5 bits per weight with <0.2% quality loss.

Based on: https://github.com/turbo-tan/llama.cpp-tq3
Algorithm:
  1. Block weights into groups of 32
  2. Compute RMS scale per block
  3. Normalize to unit variance
  4. Apply Randomized Hadamard Transform (sign flip + WHT butterfly)
  5. Quantize to 8-level Lloyd-Max codebook (3 bits)
  6. Pack 3-bit indices into bytes

During inference:
  - Activations are pre-rotated into WHT domain
  - Dot products computed directly against centroids
  - No inverse transform needed (WHT is self-inverse)
"""

from kandiga.tq3.quantize import quantize_tensor, dequantize_tensor, TQ3Block
from kandiga.tq3.engine import TQ3Linear

__all__ = ["quantize_tensor", "dequantize_tensor", "TQ3Block", "TQ3Linear"]
