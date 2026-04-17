# Qwen2.5-Omni Audio Static Debug Log — for Expert Review

**Author:** debugging session 2026-04-16 night → 2026-04-17 morning
**Status:** Unresolved. Asking for outside eyes.

---

## The Problem

Running **Qwen/Qwen2.5-Omni-3B** and **Qwen/Qwen2.5-Omni-7B** locally (PyTorch and MLX implementations). Speech generation works — words are correct, Whisper transcribes everything cleanly. But there's a consistent background **hiss / "mic static"** under every output. Sounds like a cheap microphone recording, present during speech and silence.

**Expected:** Clean audio like Alibaba's official demos online.
**Observed:** Audible hiss on every output, every voice, every configuration we tried.

---

## Environment

- **Hardware:** Mac Mini M4 16 GB (primary)
- **Python:** 3.11.14
- **MLX:** latest pip
- **transformers:** version installed from pip (Qwen2.5-Omni support)
- **soundfile, librosa:** latest
- **Sample rate output:** 24000 Hz mono PCM_16 (matches HF generate output)
- **System prompt:** verbatim from HF — `"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."`

---

## What We Built (all working end-to-end)

Pure-MLX port of the complete Omni pipeline (Thinker + Talker + Token2Wav DiT + BigVGAN + ECAPA-TDNN + RK4 ODE solver), parameterized by config, numerically equivalent to the PyTorch reference.

Scripts (all in `personaplex-mlx/scripts/`):
- `convert_omni_7b_q4full.py` — quantize Thinker-7B to 4-bit / 3-bit full (incl. embed + lm_head)
- `convert_omni_7b_mixed.py` — PersonaPlex-style mixed 4-bit (MLP + attn only)
- `convert_talker_4bit.py` — Talker port + quant
- `omni_pipeline_7b.py` — time-multiplexed Phase 1/2 (peak RAM 3.34 GB)
- `streaming_pipeline.py` — chunked T2W streaming (first-audio 1.06s in Phase 2)
- `baseline_omni_reference.py` — pure PyTorch Omni-3B reference, no MLX
- `pt_vs_mlx_vocoder.py` — captures PT codec/cond/ref_mel, feeds to MLX
- `t2w_steps_sweep.py` — ODE steps sweep 10/20/32/50
- `t2w_bf16_test.py` — Token2Wav at fp32 / bf16 / fp16
- `voice_ab.py` — Chelsie vs Ethan on same codecs
- `denoise_test.py` — noisereduce post-process

Audio artifacts for comparison at `~/.kandiga/pipeline-test/`.

---

## What We've Ruled Out

### 1. MLX port bug — RULED OUT
- Pure PyTorch reference (`baseline_omni_reference.py`, fp32, CPU, zero MLX): **same static**
- PT vs MLX on identical captured codec tokens: correlation 1.0 on fixed noise, same RMS/peak (0.41 vs 0.42) with different noise init → numerically equivalent

### 2. Wrong sample rate / format — RULED OUT
- All outputs: 24 kHz mono PCM_16 (verified via `soundfile.info`)
- BigVGAN upsample ratio `[5, 3, 2, 2, 2, 2] × mel@100Hz = 24000 Hz` — matches
- Official HF Space output is also 24 kHz mono PCM_16 (same format)

### 3. Wrong ODE solver — RULED OUT
- Diffed HF `RungeKutta4ODESolver` vs our MLX RK4 step **line-by-line**
- Same 4-stage formula: `delta = (k1 + 3*(k2 + k3) + k4) * dt / 8` at t0, t0+dt/3, t0+2dt/3, t1
- Same time embedding, same sway coefficient
- `Qwen2_5OmniToken2WavDiTModel.sample` doesn't expose any user-facing params

### 4. ODE num_steps too low — RULED OUT
- Ran the same codec tokens at `num_steps=10, 20, 32, 50`
- All 4 outputs: same RMS (0.057–0.061), same audible static character
- ODE converges by step 20; extra steps don't clean up anything

### 5. Wrong voice (Chelsie bug per PR #157) — RULED OUT for our symptom
- Tested both shipped voices: Chelsie and Ethan
- Same hiss on both, equivalent RMS
- PR #157 says Chelsie is "gibberish" — ours is not gibberish, it's just hissy
- Cherry/Serena not available locally (only in DashScope API)

### 6. Wrong precision (fp32/bf16/fp16) — RULED OUT
- `t2w_bf16_test.py` ran Token2Wav on identical codec tokens at all three precisions for both voices → 6 outputs
- User listened: **all 6 sound identical**
- HF docs mark fp32 "Not Recommended" but that's for memory, not quality

### 7. Wrong system prompt — RULED OUT
- Using `OMNI_SYSTEM` verbatim
- Tested custom "voice assistant" prompt → produced out-of-distribution hidden states that collapsed Talker into garbage codec tokens (foreign-language Unicode). Reverted.

### 8. Denoising as fix — INEFFECTIVE
- `noisereduce 3.0.3` (stationary spectral gating, prop_decrease=0.85, n_fft=1024): RMS dropped 3× but audio sounds "muffled" per user — artifacts, not clean

### 9. Quantization (MLX Thinker 3-bit, Talker 4-bit mixed) — RULED OUT
- The pure PT fp32 reference (zero quantization) has the same static
- Static appears equally in all quant variants

---

## The Key Red Herring: the Official HF Space

We found `Qwen/Qwen2.5-Omni-7B-Demo` Space produces **clean** audio. Captured `official_demo_cherry.wav` and `official_demo_chelsie.wav` via Playwright. Clean. No hiss.

Then we fetched the Space's `app.py` source:
```python
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# ...
completion = client.chat.completions.create(
    model="qwen-omni-turbo",
    messages=messages,
    modalities=["text", "audio"],
    audio={"voice": voice, "format": "wav"},
    stream=True,
)
```

**The Space is calling Alibaba's DashScope API (`qwen-omni-turbo`), not running the released model locally.** So its clean output tells us nothing about the local weights. DashScope very likely serves a different/internally-fine-tuned variant.

---

## What We Believe, But Can't Prove

- DashScope serves a post-training-refined version of Qwen2.5-Omni with a cleaner vocoder (or explicit post-processing) that was never released publicly.
- All "clean local demos" people have seen online are either DashScope-served or cherry-picked clips with youtube-compression-masking.
- The released weights (both 3B and 7B) produce staticky audio in any local configuration we can construct.

---

## What We DIDN'T Try (but could)

- [ ] Running the full Omni pipeline on a **real CUDA GPU with bf16** (we don't have CUDA hardware). HF docs recommend this; every clean local demo we found used this config.
- [ ] [DeepFilterNet2](https://github.com/Rikorose/DeepFilterNet) — neural denoiser, much stronger than noisereduce.
- [ ] Pulling the GPTQ-Int4 or AWQ variants and running them on CUDA (same Token2Wav under the hood, so unlikely to fix).
- [ ] Building Qwen2.5-Omni from source with a specific `transformers` commit pinned to the day the model released (possible buggy update in main).
- [ ] Qwen3-Omni-30B-A3B — the newer Qwen omni where they [explicitly replaced the vocoder stack](https://arxiv.org/html/2509.17765v1) "to alleviate hallucinations caused by noisy data and significantly improve quality of generated speech." Config is already on disk (`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`), ~15 GB weights not yet downloaded.
- [ ] MPS (PyTorch Metal) backend — unclear if all Omni ops are supported.

---

## Specific Questions for an Expert

1. **Is there a `transformers` version with a Token2Wav bug that's the common-case for installs?** We're on the current pip release; maybe an older or newer version has different behavior.
2. **Has anyone actually gotten clean audio from `Qwen/Qwen2.5-Omni-7B` on Apple Silicon or CPU-only?** Not a cherry-picked demo — actual repro-able setup with code.
3. **Is there a speaker embedding we're missing?** The shipped `spk_dict.pt` has only Chelsie and Ethan. Cherry/Serena are in the DashScope API only. Maybe Alibaba released an updated spk_dict we haven't pulled?
4. **Is the `use_audio_in_video: False` kwarg we see passed through `token2wav.forward(...)` significant when input is text-only?**
5. **Is there server-side post-processing that DashScope applies** (e.g., DeepFilterNet2, spectral gating with a trained noise profile) that Alibaba didn't include in the release?

---

## What "works" right now

- Omni-7B end-to-end pipeline: 3.34 GB peak RAM (under 5 GB target)
- Streaming: 1.06s first-audio in Phase 2
- Whisper-verified transcription of all outputs
- Quality: **staticky but intelligible**. Every listener (user + Whisper) parses the text.

---

## Numerical parity checkpoint (worth preserving for reference)

- PT Token2Wav output (captured codec, fixed seed) vs MLX Token2Wav output (same seed): correlation ≈ 1.0 across DiT blocks
- PT vs MLX with different random noise: correlation 0.034 but identical RMS/peak envelopes — flow-matching ODE converges to same mel distribution with different phase
- Token2Wav weights PT vs our saved MLX fp32: `numpy.allclose(..., atol=1e-6)` → True on every weight checked
- 3B and 7B Token2Wav weights: byte-identical (`numpy.allclose` max abs diff = 0.0)

---

## File Map

- `docs/04-16-2026.md` — full timeline of all work from 2026-04-16 evening
- `docs/04-17-2026.md` — continuation log with today's debug session
- `personaplex-mlx/scripts/` — all conversion, pipeline, and debug scripts listed above
- `personaplex-mlx/mlx_omni/token2wav_mlx.py` — ~900-line pure MLX port of the Omni Token2Wav stack

---

## Bottom Line

After exhaustive testing: we can't find a local-only configuration that eliminates the static. The static appears in the released PyTorch path, not just our MLX port. Our best guess is that Alibaba's public release doesn't match what they serve via DashScope. We're open to being wrong — if you can point to a specific local invocation or a precision setting or a hidden kwarg that gives clean output, please share.
