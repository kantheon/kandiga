# Kandiga

Giant models. Tiny memory.

Kandiga is an open-source MoE inference engine for Apple Silicon. Run models that normally need 20-224GB of RAM in **2-8GB** — on any Mac. No cloud, no API keys.

## Supported Models

| Model | Parameters | Active | Disk | Kandiga RAM | Min. Mac | Decode | TTFT |
|-------|-----------|--------|------|-------------|----------|--------|------|
| Qwen3.5-35B-A3B | 35B | 3B | 20 GB | ~2 GB | 8 GB | 6.5 tok/s | 3-8s |
| Qwen3.5-122B-A10B | 122B | 10B | 70 GB | ~4 GB | 16 GB | 1.4 tok/s | 7-27s |
| Qwen3.5-397B-A17B | 397B | 17B | 224 GB | ~8 GB | 24 GB | ~1 tok/s | TBD |

Without Kandiga, these models require their full disk size in RAM. With SEM, only the shared layers load to memory — expert weights stay on disk and are read on demand.

## How it works

MoE models have hundreds of expert sub-networks per layer, but only activate a few per token. Kandiga exploits this sparsity with six techniques:

1. **Selective Expert Materialization (SEM)** — shared layers load to GPU, expert weights stay on disk. Only the 8 router-selected experts are read per token per layer.

2. **Custom Metal GPU kernels** — during prefill (processing your prompt), expert MLP runs entirely on GPU via custom Metal shaders. One dispatch handles ALL experts for ALL tokens — zero Python loop overhead.

3. **CPU NEON decode** — during generation (single token), expert MLP runs on CPU with NEON-vectorized 4-bit dequant. Faster than GPU for single-token due to zero Metal dispatch overhead.

4. **Cross-layer expert speculation** — predicts next layer's expert routing with 77% accuracy. Pre-fetches predicted experts into OS page cache during current layer's compute. Overlaps I/O with computation.

5. **TurboQuant KV compression** — compresses the KV cache from 16-bit to 3-bit (3.8x) using PolarQuant + QJL error correction. Enables longer conversations without running out of memory.

6. **ZMLX fused kernels** — third-party Metal kernel optimizations for attention and norms. +45% decode speed.

## Install

```bash
pip install kandiga

# For maximum speed (includes ZMLX fused kernels):
pip install kandiga[fast]
```

Requirements: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+

## Quick start

```bash
# One-time setup: choose model, download, prepare expert files
kandiga setup

# Interactive chat
kandiga chat

# Fast mode (K=4 experts instead of 8, ~2x speed)
kandiga chat --fast

# One-shot prompt
kandiga "What is the capital of France?"

# Start an OpenAI-compatible API server
kandiga serve

# Run benchmarks
kandiga bench

# Update to latest version
kandiga update

# View changelog
kandiga changelog
```

## Performance

Measured on M4 Mac Mini (16GB):

| Model | Mode | Decode | TTFT (first turn) | TTFT (follow-up) | RAM |
|-------|------|--------|-------------------|-------------------|-----|
| Qwen3.5-35B | K=8 | 3.7 tok/s | 5-15s | **2-4s** | ~2 GB |
| Qwen3.5-35B | K=4 | 6.3 tok/s | 3-8s | **2-4s** | ~2 GB |
| Qwen3.5-122B | K=4 | 1.0 tok/s | 11-18s | **11-15s** | ~4 GB |

Key: follow-up TTFT is constant (3-4s) regardless of conversation length thanks to persistent KV cache.

## Persistent KV Cache (Conversation Memory)

The biggest problem with local LLMs: every message re-processes the ENTIRE conversation history. Turn 30 re-reads turns 1-29 — even though the model already read them.

Kandiga solves this with persistent KV cache:

```
Without persistent cache:
  Turn 1:  8s    (reads invoice)
  Turn 5:  25s   (re-reads invoice + 4 turns)
  Turn 30: 2min+ (re-reads everything)

With persistent cache:
  Turn 1:  8s    (reads invoice once)
  Turn 2:  3s    (only reads new question)
  Turn 5:  3s    (only reads new question)
  Turn 30: 3s    (only reads new question)
```

The model reads the conversation once and keeps its understanding in memory. Every follow-up only processes the new message. TTFT stays flat at 3-4 seconds regardless of conversation length.

```python
from kandiga.engine import KandigaEngine

engine = KandigaEngine()
engine.load()
engine.start_session()

# Turn 1: send a document (8s TTFT — processes the document)
for token in engine.session_generate("Here is an invoice: ..."):
    print(token, end="")

# Turn 2: follow-up (3s TTFT — document already cached)
for token in engine.session_generate("What is the most expensive item?"):
    print(token, end="")

# Turn 30: still 3s TTFT
for token in engine.session_generate("Summarize everything"):
    print(token, end="")

engine.end_session()
```

Combined with TurboQuant (3.8x KV compression), conversations can run for 32K+ tokens before hitting memory limits.

### Save & Resume Sessions

Sessions persist to disk so conversations survive app restarts. Close your laptop, reopen tomorrow, and pick up exactly where you left off — no re-processing.

```python
# Save conversation state (KV cache + history)
engine.save_session("~/my_session.npz")   # ~2-10MB compressed

# Later: load and continue instantly
engine.load_session("~/my_session.npz")   # <0.2s load time
for token in engine.session_generate("Where were we?"):
    print(token, end="")  # model remembers everything
```

The save file includes the full KV cache state for all 40-48 layers (attention + linear), conversation history, and token tracking. Loading restores the model's exact state — no re-reading the conversation.

## GPU Metal Prefill

During prefill (processing your prompt), Kandiga uses custom Metal compute shaders instead of CPU:

- **Phase 1**: Gate + Up projection + SwiGLU activation — one Metal dispatch for ALL experts
- **Phase 2**: Down projection — one Metal dispatch

This eliminates the Python loop entirely. All expert computation for all tokens happens in two GPU dispatches per layer. Combined with C-level parallel `pread` via GCD, this gives **3-5x faster prefill** compared to the CPU-only path.

```
Prefill improvement (35B, ~150 tokens):
  CPU baseline:     24.0s
  GPU + Python loop: 7.7s
  GPU Metal kernel:  6.1s  (current)
```

## KV Cache Compression (TurboQuant)

Compresses the KV cache from 16-bit to 3-bit per element (3.8x reduction) with only 4% quality loss. Based on [Google Research TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/).

| Context Length | Standard (float16) | Kandiga (3-bit) | Compression |
|---------------|-------------------|-----------------|-------------|
| 4K tokens | 16.8 MB | 4.5 MB | 3.8x |
| 8K tokens | 33.6 MB | 8.9 MB | 3.8x |
| 16K tokens | 67.1 MB | 17.8 MB | 3.8x |
| 32K tokens | 134 MB | 35.3 MB | 3.8x |

## Architecture

```
User prompt
    |
    v
[Tokenizer + Chat Template]
    |
    v
[MLX Forward Pass — per layer:]
    |
    +---> GPU: Attention + Norms + Router (MLX lazy eval)
    |
    +---> Prefill (multi-token):
    |     GPU Metal kernel: batch pread + dequant + matmul + SiLU
    |     (one dispatch for ALL experts, ALL tokens)
    |
    +---> Decode (single-token):
    |     CPU NEON: pread + 4-bit dequant matvec + GCD parallel
    |     (faster than GPU for single vectors)
    |
    +---> Cross-layer speculation: predict next layer's experts
    |     Background prefetch into OS page cache
    |
    v
[Token Output — streaming]
```

## API Server

OpenAI-compatible HTTP API:

```bash
kandiga serve --port 8340
```

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8340/v1", api_key="unused")
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-35B-A3B-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Key Technical Details

- **Expert binary format**: packed 4-bit uint32 weights with bfloat16 scales/biases, 4096-byte header per layer file. Expert dimensions auto-detected from header.
- **Two C libraries**: `libkandiga_cpu_expert.dylib` (35B, hardcoded dims) and `libkandiga_cpu_expert_lg.dylib` (122B/397B, dynamic dims from header)
- **Custom Metal shaders**: `attention.metal` (19 kernels), `expert_mlp.metal`, `moe_block.metal` — compiled at runtime
- **Cross-layer speculation**: 77% prediction accuracy using router gate CPU matmul (~0.05ms)
- **Layer skipping**: large models skip routed experts on every 3rd layer (shared expert only). Reduces syncs by 33% with quality preserved.
- **Thinking mode disabled**: `enable_thinking=False` in chat template to avoid wasted tokens
- **Auto-update checker**: background check against PyPI, cached 24h

## Development

```bash
git clone https://github.com/kantheon/kandiga.git
cd kandiga
pip install -e ".[serve,fast]"

# Build CPU expert libraries
cd kandiga/metal && make && cd ../..

# Run tests
pytest tests/ -v
```

## License

MIT — Built by [Kantheon](https://kantheon.com)
