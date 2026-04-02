"""Tri-mode engine — 2B (JSON) + 35B K=2 (writing) + 35B K=4 (verification).

2B (45 tok/s, 0.8s):  tool JSON, route classification, structured output
35B K=2 (9-10 tok/s): response writing, summaries, content
35B K=4 (5 tok/s):    verification when tools fail (runs once)

Total GPU: ~2.4GB (1.06GB 2B + 1.38GB 35B shared).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Iterator, Optional

import mlx.nn as nn

from kandiga.engine import KandigaEngine, _find_layers


def _set_expert_k(model: nn.Module, k: int) -> int:
    try:
        layers = _find_layers(model)
    except (AttributeError, TypeError):
        return 0
    count = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if hasattr(mlp, "top_k"):
            mlp.top_k = k
            count += 1
        sw = getattr(mlp, "switch_mlp", None)
        if sw and hasattr(sw, "_top_k"):
            sw._top_k = k
    return count


class DualEngine:
    """2B for JSON + 35B K=2/K=4 for writing/verification."""

    # 3-bit is 21% faster than 4-bit, 22% less GPU
    STRUCT_MODEL_3BIT = os.path.expanduser("~/.kandiga/models/Qwen3.5-4B-3bit")
    STRUCT_MODEL_4BIT = "mlx-community/Qwen3.5-4B-4bit"
    BRAIN_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
    K_FAST = 4     # K=2 is too degraded for response writing
    K_PRECISE = 4  # same K for everything — proven quality

    def __init__(self, fast_mode: bool = True, log_memory: bool = False):
        self._brain = KandigaEngine(
            model_path=self.BRAIN_MODEL, fast_mode=fast_mode, log_memory=log_memory,
        )
        self._struct_model = None
        self._struct_tokenizer = None
        self._ready = False
        self._current_k = 0

    def load(self):
        if self._ready:
            return
        self._brain.load()
        self._switch_k(self.K_FAST)
        self._brain._log(f"Dual-K: write=K{self.K_FAST}, precise=K{self.K_PRECISE}")

        # Load 4B: try 3-bit (faster) → fall back to 4-bit
        from mlx_lm import load as mlx_load
        if os.path.isdir(self.STRUCT_MODEL_3BIT):
            try:
                self._struct_model, self._struct_tokenizer = mlx_load(self.STRUCT_MODEL_3BIT)
                self._brain._log(f"4B loaded (3-bit, 21% faster)")
            except Exception as e:
                self._struct_model, self._struct_tokenizer = mlx_load(self.STRUCT_MODEL_4BIT)
                self._brain._log(f"4B loaded (4-bit fallback: {e})")
        else:
            self._struct_model, self._struct_tokenizer = mlx_load(self.STRUCT_MODEL_4BIT)
            self._brain._log(f"4B loaded (4-bit, no 3-bit model found)")
        self._ready = True

    def _switch_k(self, k: int):
        if k == self._current_k:
            return
        _set_expert_k(self._brain._model, k)
        self._current_k = k

    def _format_brain(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self._brain._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self._brain._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def _format_struct(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self._struct_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self._struct_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    # --- 2B: Structured output (JSON, classification) ---

    def generate_fast(self, system: str, user: str,
                      max_tokens: int = 1200, temp: float = 0.0) -> str:
        """2B — tool JSON, route classification, structured output. ~45 tok/s."""
        if not self._ready:
            self.load()
        formatted = self._format_struct(system, user)
        from mlx_lm import generate as mlx_generate
        from mlx_lm.generate import make_sampler
        response = mlx_generate(
            self._struct_model, self._struct_tokenizer,
            prompt=formatted, max_tokens=max_tokens,
            sampler=make_sampler(temp=temp), verbose=False,
        )
        return _strip_thinking(response)

    # --- 35B K=2: Fast writing ---

    def generate_brain(self, system: str, user: str,
                       max_tokens: int = 2048, temp: float = 0.0) -> str:
        """35B K=2 — response writing, summaries, content. ~9-10 tok/s."""
        if not self._ready:
            self.load()
        self._switch_k(self.K_FAST)
        formatted = self._format_brain(system, user)
        return self._brain.generate(formatted, max_tokens=max_tokens, temp=temp, stream=False)

    def generate_brain_stream(self, system: str, user: str,
                              max_tokens: int = 2048, temp: float = 0.0) -> Iterator[str]:
        """35B K=2 — streaming."""
        if not self._ready:
            self.load()
        self._switch_k(self.K_FAST)
        formatted = self._format_brain(system, user)
        for token in self._brain.generate(formatted, max_tokens=max_tokens, temp=temp, stream=True):
            yield token

    # --- 35B K=4: Verification ---

    def generate_check(self, system: str, user: str,
                       max_tokens: int = 512, temp: float = 0.0) -> str:
        """35B K=4 — verification. ~5 tok/s. Only when tools fail."""
        if not self._ready:
            self.load()
        self._switch_k(self.K_PRECISE)
        formatted = self._format_brain(system, user)
        result = self._brain.generate(formatted, max_tokens=max_tokens, temp=temp, stream=False)
        self._switch_k(self.K_FAST)
        return result

    # --- Session ---

    @property
    def brain(self) -> KandigaEngine:
        return self._brain

    @property
    def tokenizer(self):
        return self._brain._tokenizer

    @property
    def model_path(self) -> str:
        return self.BRAIN_MODEL

    def start_session(self):
        self._switch_k(self.K_FAST)
        self._brain.start_session()

    def end_session(self):
        self._brain.end_session()

    def save_session(self, path: str):
        self._brain.save_session(path)

    def load_session(self, path: str):
        self._brain.load_session(path)

    def session_generate(self, user_message: str, max_tokens: int = 2048, temp: float = 0.0):
        self._switch_k(self.K_FAST)
        return self._brain.session_generate(user_message, max_tokens=max_tokens, temp=temp)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def stats(self) -> Dict[str, Any]:
        s = self._brain.stats
        s["dual_k"] = True
        s["k_fast"] = self.K_FAST
        s["k_precise"] = self.K_PRECISE
        s["current_k"] = self._current_k
        s["struct_model"] = self.STRUCT_MODEL
        return s

    def __del__(self):
        if hasattr(self, '_brain'):
            del self._brain


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text
