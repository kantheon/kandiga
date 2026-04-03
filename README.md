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

| Model | Parameters | Active | Disk | Kandiga RAM | Decode | Status |
|-------|-----------|--------|------|-------------|--------|--------|
| Qwen3.5-4B (3-bit) | 4B | 4B | 1.84 GB | ~1.8 GB | **136 tok/s** | Proven |
| Qwen3.5-4B (4-bit) | 4B | 4B | 2.4 GB | ~2.4 GB | 112 tok/s | Proven |
| Qwen3.5-35B-A3B (full 3-bit) | 35B | 3B | 20 GB | **~1 GB** | **8.1 tok/s** | Proven |
| Qwen3.5-35B-A3B (4-bit) | 35B | 3B | 20 GB | ~1.4 GB | 6.7 tok/s | Proven |
| Qwen3.5-122B-A10B (full 3-bit) | 122B | 10B | 70 GB | **~2.7 GB** | **2.2 tok/s** | Proven |
| Qwen3.5-122B-A10B (4-bit) | 122B | 10B | 70 GB | ~3.5 GB | 1.3 tok/s | Proven |
| Qwen3.5-397B-A17B (full 3-bit) | 397B | 17B | 224 GB | est. ~5 GB | est. 0.3-0.5 tok/s | Pending |

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
| 35B-A3B | 6.7 tok/s, 1.4 GB | **8.1 tok/s, 1.0 GB** | **+21%** | **-22%** |
| 122B-A10B | 1.3 tok/s, 3.5 GB | **2.2 tok/s, 2.7 GB** | **+69%** | **-22%** |

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
| Hello | 2-6s | — |
| List files | 15s | list_dir |
| Read CSV + calculate | 60s | read_file |
| Create script + run | 70s | write_file, run_shell |
| Web search + notify | 49s | web_search, notify |
| Multi-step (5 tools) | 105s | list_dir, read_file ×3, write_file |

10/10 multi-turn test passed. KV cache maintains context across all turns.

## Performance (M4 Mac Mini, 16GB)

| Model | Mode | Decode | TTFT | Follow-up TTFT | RAM |
|-------|------|--------|------|----------------|-----|
| Qwen3.5-4B (3-bit) | dense | **136 tok/s** | <1s | <1s | 1.8 GB |
| Qwen3.5-35B (full 3-bit) | K=4 | **8.1 tok/s** | 3-8s | **2-4s** | ~1 GB |
| Qwen3.5-35B (4-bit) | K=4 | 6.7 tok/s | 3-8s | **2-4s** | ~1.4 GB |
| Qwen3.5-122B (full 3-bit) | K=4 | **2.2 tok/s** | 11-18s | **11-15s** | ~2.7 GB |
| Qwen3.5-122B (4-bit) | K=4 | 1.3 tok/s | 11-18s | **11-15s** | ~3.5 GB |

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
