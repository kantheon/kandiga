# Kandiga

Giant models. Tiny memory.

Kandiga is an open-source MoE inference engine that uses **Selective Expert Materialization (SEM)** to run massive models on any Apple Silicon Mac. A 397B model that normally needs 224GB of RAM runs in 8GB. A 35B model runs in 2GB. No cloud, no API keys.

## Supported Models

| Model | Parameters | Active | Experts | Disk | Kandiga RAM | Min. Mac |
|-------|-----------|--------|---------|------|-------------|----------|
| Qwen3.5-35B-A3B | 35B | 3B | 256 | 20 GB | ~2 GB | 8 GB |
| Qwen3.5-122B-A10B | 122B | 10B | 256 | 70 GB | ~4 GB | 16 GB |
| Qwen3.5-397B-A17B | 397B | 17B | 512 | 224 GB | ~8 GB | 24 GB |

Without Kandiga, these models require their full disk size in RAM. With SEM, only the shared layers load to memory — expert weights stay on disk and are read on demand.

## How it works

MoE models have hundreds of expert sub-networks per layer, but only activate a few per token. Kandiga exploits this sparsity:

1. **Shared layers** (attention, norms, embeddings) load to GPU memory
2. **Expert weights** stay on disk in packed binary files
3. **Per token**: the router selects which experts to activate (8 of 256)
4. **CPU computes** expert MLP with NEON-vectorized 4-bit dequant + GCD parallelism
5. **GPU computes** attention simultaneously via MLX (unified memory, zero copy)

## Install

```bash
pip install kandiga
```

Requirements: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+

## Quick start

```bash
# One-time setup: download model + prepare expert files
kandiga setup

# Choose a different model
kandiga setup --model mlx-community/Qwen3.5-122B-A10B-4bit

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
```

## Performance

Measured on M4 Mac Mini (16GB), Qwen3.5-35B-A3B-4bit:

| Mode | Experts | Speed | RAM | Quality |
|------|---------|-------|-----|---------|
| Quality (K=8) | 8/256 per layer | ~3.5 tok/s | ~2 GB | Full |
| Fast (K=4) | 4/256 per layer | ~6.5 tok/s | ~2 GB | Near-equal |

## KV Cache Compression (TurboQuant)

Every token generated grows a memory buffer called the **KV cache**. On a 16GB Mac, this normally limits conversations to ~4K tokens before RAM runs out.

Kandiga implements **TurboQuant** (based on [Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)) — a compression algorithm that shrinks the KV cache from 16-bit to 3-bit per element with only 4% quality loss:

1. **PolarQuant** — randomly rotates vectors to spread information, then quantizes to 3 bits
2. **QJL** — 1-bit error correction using the Johnson-Lindenstrauss transform

| Context Length | Standard (float16) | Kandiga (3-bit) | Compression |
|---------------|-------------------|-----------------|-------------|
| 4K tokens | 16.8 MB | 4.5 MB | 3.8x |
| 8K tokens | 33.6 MB | 8.9 MB | 3.8x |
| 16K tokens | 67.1 MB | 17.8 MB | 3.8x |
| 32K tokens | 134 MB | 35.3 MB | 3.8x |

- **96% cosine similarity** — attention scores stay accurate
- **Zero speed overhead** — rotation cost is negligible vs expert I/O
- **No configuration needed** — compression is automatic

## Architecture

```
User prompt
    |
    v
[Tokenizer + Chat Template]
    |
    v
[MLX Forward Pass]
    |
    +---> GPU: Attention + Norms + Router + Shared Expert + Blending
    |
    +---> CPU: Routed Expert MLP (NEON 4-bit dequant + GCD parallel)
    |         |
    |         +-- pread expert weights from SSD (OS page cache)
    |         +-- SwiGLU activation + down projection
    |
    v
[Token Output]
```

Both CPU and GPU operate on the same physical DRAM (Apple Silicon unified memory), so there is zero data transfer overhead between them.

## API Server

Kandiga includes an OpenAI-compatible HTTP API:

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

## Development

```bash
git clone https://github.com/kantheon/kandiga.git
cd kandiga
pip install -e ".[serve]"
cd kandiga/metal && make && cd ../..
pytest tests/ -v
```

## License

MIT — Built by [Kantheon](https://kantheon.com)
