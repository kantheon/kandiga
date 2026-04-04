# Kandiga

Giant models. Tiny memory.

Kandiga is an open-source MoE inference engine + AI agent for Apple Silicon. Run models that normally need 20-224GB of RAM in **2-8GB** — on any Mac. No cloud, no API keys.

## What It Does

- **Inference engine**: Run 35B-397B parameter MoE models in 2-8GB RAM via Selective Expert Materialization (SEM)
- **AI agent**: Tool calling, web search, file operations, macOS integrations (Calendar, Reminders, Notes, Notifications) — all local
- **3-bit quantization**: 21% faster and 22% smaller than 4-bit via MLX native `mx.quantize(bits=3)`
- **Persistent KV cache**: Follow-up turns process only new tokens — turn 50 is as fast as turn 1
- **TurboQuant**: 3.8x KV cache compression for longer conversations

## Supported Models

| Model | Parameters | Active | Disk | Kandiga RAM | Decode* | Status |
|-------|-----------|--------|------|-------------|---------|--------|
| Qwen3.5-4B (3-bit) | 4B | 4B | 1.84 GB | ~1.8 GB | **31 tok/s** | Proven |
| Qwen3.5-35B-A3B (full 3-bit) | 35B | 3B | 20 GB | **~1 GB** | **~12 tok/s** | Proven |
| Qwen3.5-122B-A10B (full 3-bit) | 122B | 10B | 70 GB | **~2.7 GB** | **~4 tok/s** | Proven |
| Gemma 4 26B-A4B | 26B | 4B | 13 GB | **~1.35 GB** | **~10 tok/s** | Proven |
| Qwen3.5-397B-A17B (full 3-bit) | 397B | 17B | 224 GB | est. ~5 GB | est. ~1 tok/s | Pending |

*MoE decode speed depends on SSD bandwidth. Estimates for internal NVMe on M4.

## Install

```bash
pip install kandiga

# For maximum speed (includes ZMLX fused kernels):
pip install kandiga[fast]
```

Requirements: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+

## Quick Start

```bash
# One-time setup: choose model, download, prepare expert files
kandiga setup

# Interactive chat
kandiga chat

# Fast mode (K=4 experts, ~2x speed)
kandiga chat --fast

# AI agent mode — tools, skills, memory, macOS integrations
kandiga agent --fast

# Agent with web UI
kandiga agent --fast --web

# One-shot prompt
kandiga "What is the capital of France?"

# OpenAI-compatible API server
kandiga serve

# Benchmarks
kandiga bench
```

## Architecture

### Inference Engine (SEM)

MoE models have hundreds of expert sub-networks per layer, but only activate a few per token. Kandiga exploits this sparsity:

1. **Selective Expert Materialization** — shared layers on GPU (~1.4GB), expert weights on SSD. Only router-selected experts loaded per token.
2. **Custom Metal GPU kernels** — prefill runs expert MLP entirely on GPU. One dispatch, zero Python overhead.
3. **CPU NEON decode** — single-token expert MLP on CPU with NEON-vectorized 4-bit dequant. Faster than GPU for single tokens (no Metal dispatch overhead).
4. **Cross-layer speculation** — predicts next layer's experts with 77% accuracy. Pre-fetches into OS page cache during current compute.
5. **TurboQuant KV compression** — 3.8x compression (16-bit → 3-bit) via PolarQuant + QJL. Enables 32K context on 16GB.
6. **ZMLX fused kernels** — optimized attention and norms.

### 3-Bit Weight Quantization

MLX's native `mx.quantize(bits=3)` with `quantized_matmul(bits=3)`:

| Metric | 4-bit | 3-bit | Improvement |
|--------|-------|-------|-------------|
| Speed | 112 tok/s | **136 tok/s** | **21% faster** |
| Load time | 3.6s | **0.9s** | **4x faster** |
| GPU memory | 2,368MB | **1,842MB** | **526MB saved** |
| Disk | 2.4GB | **1.84GB** | **23% smaller** |
| Quality | ✓ correct | ✓ correct | Same |

Conversion: one-time `dequant 4-bit → requant 3-bit → save safetensors`. Model saved at `~/.kandiga/models/Qwen3.5-4B-3bit/`.

**Full 3-bit MoE** — shared layers on GPU + expert weights on SSD, both at 3-bit:

| Model | 4-bit | Full 3-bit | Speed gain | GPU savings |
|-------|-------|------------|------------|-------------|
| 35B-A3B | ~8 tok/s, 1.4 GB | **~12 tok/s, 1.0 GB** | **+50%** | **-22%** |
| 122B-A10B | ~2 tok/s, 3.5 GB | **~4 tok/s, 2.7 GB** | **+100%** | **-22%** |
| Gemma 4 26B-A4B | N/A | **~10 tok/s, 1.35 GB** | — | — |

Conversion (one-time):
```bash
# Shared layers (GPU): dequant 4-bit → requant 3-bit → save safetensors
python scripts/convert_3bit.py mlx-community/Qwen3.5-35B-A3B-4bit \
    ~/.kandiga/models/Qwen3.5-35B-A3B-3bit-shared

# Expert weights (SSD): repack binary files at 3-bit (22% smaller = 22% less I/O)
python scripts/repack_experts_3bit.py ~/.kandiga/experts/Qwen3.5-35B-A3B-4bit/packed
```

Both auto-detected on engine startup. NEON-vectorized 3-bit dequant kernel matches MLX's bit layout.

### Agent System

Kandiga includes a full AI agent with native Qwen3.5 tool calling:

**Architecture:**
- **4B (3-bit, 136 tok/s)**: tool call JSON generation, route classification
- **35B K=4 (6.7 tok/s)**: response writing, reasoning via session KV cache
- **17 tools**: filesystem (read/write/list/search), shell, web search, macOS (Calendar, Reminders, Notes, Notifications, Finder, Contacts, system info, text-to-speech)
- **Skill engine**: OpenClaw-compatible SKILL.md format
- **Memory**: MEMORY.md + daily notes + persistent KV cache sessions

**Agent performance (M4 Mac Mini 16GB):**

| Task | Time | Tools Used |
|------|------|-----------|
| Hello | 2.5s | — |
| List files | 3s | list_dir |
| Create + run script | 6s | write_file, run_shell |
| Web search | 3s | web_search |
| Math (127 × 389) | 3s | — (direct) |
| What time is it | 3s | — (injected) |
| Delete file | 5s | run_shell |
| Recall turn 1 at turn 44 | 8s | — (KV cache) |

95% accuracy on 44-turn multi-turn conversation. Persistent KV cache maintains context across all turns. 3.1s average per turn.

## Performance (M4 Mac Mini, 16GB)

| Model | Mode | Decode* | Follow-up TTFT | RAM |
|-------|------|---------|----------------|-----|
| Qwen3.5-4B (3-bit) | dense | **31 tok/s** | <1s | 1.8 GB |
| Qwen3.5-35B (full 3-bit) | K=4 | **~12 tok/s** | **2-4s** | ~1 GB |
| Qwen3.5-122B (full 3-bit) | K=4 | **~4 tok/s** | **5-10s** | ~2.7 GB |
| Gemma 4 26B-A4B | K=4 | **~10 tok/s** | **2-4s** | ~1.35 GB |

*MoE decode speed depends on SSD bandwidth. Estimates for M4 internal NVMe.

Follow-up TTFT is constant regardless of conversation length thanks to persistent KV cache.

## Persistent KV Cache

```
Without persistent cache:       With persistent cache:
  Turn 1:  8s (reads document)     Turn 1:  8s (reads once)
  Turn 5:  25s (re-reads all)      Turn 5:  3s (new tokens only)
  Turn 30: 2min+ (re-reads all)    Turn 30: 3s (new tokens only)
```

Save/load sessions to disk:
```python
engine.save_session("~/session.npz")   # Save KV cache state
engine.load_session("~/session.npz")   # Resume instantly (<0.1s)
```

## TQ3 Weight Quantization

TQ3 (TurboQuant 3-bit) applies Walsh-Hadamard Transform rotation before quantization for better quality:

- **Algorithm**: WHT rotation → Lloyd-Max 8-level codebook → 3-bit packing
- **Quality**: 0.990 cosine similarity per layer (proven across all 32 layers)
- **Metal kernel**: Fused GEMV with SIMD WHT butterfly (cosine 1.0, 62% memory savings)
- **Status**: Algorithm proven, MLX native 3-bit is faster for production use

For production: use `mx.quantize(bits=3)` (MLX native). TQ3 WHT rotation is for research/future optimization.

### Vision

Gemma 4 26B-A4B and Qwen 3.5 4B support image analysis:

- **Gemma 4 vision via SEM**: Vision encoder on GPU (~1.1 GB) + shared language layers (~1.6 GB) + experts from SSD. Total: 3.5 GB peak. Enable with `vision=True` (default off to save RAM).
- **Qwen 3.5 4B**: Natively multimodal. Loads on-demand when `analyze_image` tool is called. 36.6 tok/s, 4 GB.
- Agent tools: `analyze_image(path, question)`, `screenshot_analyze(question)`

### Research: MoE Expert Routing Analysis

Empirical analysis of expert selection patterns across 40 MoE layers, 256 experts/layer, K=8 (Qwen 3.5 35B-A3B, 249 tokens across 8 prompts):

**Finding 1: Cross-layer expert indices are random.** Adjacent layers share experts at 1.07x random chance (Jaccard 0.017 vs 0.016 expected). Layer distance doesn't matter — layers 20 apart are equally uncorrelated.

**Finding 2: Hidden state prediction works despite random indices.** Cross-layer speculation achieves 77% accuracy by predicting from `hidden_state × next_gate_weights`. The hidden representation carries enough information to predict routing even though expert *identities* are completely independent across layers.

**Finding 3: Router distributions are extremely diffuse.** Top-8 experts capture only 15.7% of total probability mass. Entropy averages 7.5/8.0 bits (near-uniform). Adaptive K via confidence thresholding is not viable for this architecture.

**Finding 4: Deeper layers specialize more.** Gini coefficient rises from 0.38 (layer 0) to 0.62 (layer 20). Early layers use 250/256 experts; deep layers use ~175/256.

**Finding 5: Cross-token expert caching shows 35% hit rate.** Consecutive tokens reuse 35% of experts within the same layer. Deep layers hit 48%. Estimated 1.5x speedup (12→18 tok/s) with 420 MB cache. Implementation pending.

Scripts: `scripts/experiment_expert_v2.py`, `scripts/experiment_cross_token_cache.py`

## File Structure

```
kandiga/
├── engine.py              # SEM inference engine (1387 lines)
├── kv_compress.py         # TurboQuant KV cache compression
├── speculative.py         # Dual-model speculative decoding
├── cli.py                 # CLI interface
├── chat.py                # Interactive chat (Rich terminal)
├── serve.py               # OpenAI-compatible API server
├── agents/                # AI agent layer
│   ├── agent_loop.py      # Native Qwen3.5 tool-calling loop
│   ├── agent_chat.py      # Agent interactive chat
│   ├── agent_serve.py     # Agent web server + UI
│   ├── dual_engine.py     # 4B + 35B dual-model engine
│   ├── pipeline.py        # Agent pipeline (routing, tools, verification)
│   ├── tools.py           # 17 tools (filesystem, shell, web, macOS)
│   ├── macos.py           # macOS native integrations via osascript
│   ├── skills.py          # OpenClaw-compatible SKILL.md engine
│   ├── memory.py          # Persistent memory (MEMORY.md + daily notes)
│   ├── cloud.py           # Cloud escalation (Kimi/Claude/OpenAI)
│   ├── protocol.py        # Typed dataclasses (ToolCall, ToolResult, AgentResult)
│   └── json_repair.py     # 5-strategy JSON repair (never crashes)
├── tq3/                   # TQ3 weight quantization
│   ├── quantize.py        # WHT + Lloyd-Max + packing (vectorized)
│   ├── engine.py          # TQ3Linear layer + save/load
│   ├── fused_kernel.py    # Metal GEMV kernel (SIMD WHT)
│   ├── integrate.py       # Model conversion pipeline
│   ├── loader.py          # TQ3 model loader
│   ├── convert_experts.py # MoE expert conversion
│   ├── mlx_patch.py       # MLX model patching
│   └── tq3_metal.metal    # Metal compute shader
├── metal/                 # C/Metal inference (6,600+ lines)
│   ├── kandiga_cpu_expert.m    # NEON expert MLP (35B)
│   ├── kandiga_cpu_expert_lg.m # NEON expert MLP (122B/397B)
│   ├── attention.metal         # GPU attention kernels
│   ├── expert_mlp.metal        # GPU expert MLP kernels
│   └── moe_block.metal         # GPU MoE block kernels
├── static/
│   └── agent.html         # Agent web UI
└── tools/                 # Optional tool integrations
```

## Development

```bash
git clone https://github.com/kantheon/kandiga.git
cd kandiga
pip install -e ".[serve,fast]"
pytest tests/ -v
```

## License

MIT
