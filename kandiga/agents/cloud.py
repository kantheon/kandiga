"""Cloud escalation — Kimi, Claude, OpenAI, Groq, Together, Ollama.

When local models can't handle a task, escalate to cloud with PII stripped.
One engine class, multiple providers via OpenAI-compatible API.

Config: ~/.kandiga/config.json
{
    "cloud": {
        "provider": "kimi",
        "api_key": "sk-...",
        "threshold": 0.5,
        "always": false,
        "strip_pii": true
    }
}
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


CONFIG_PATH = os.path.expanduser("~/.kandiga/config.json")

PROVIDERS = {
    "kimi": {"base_url": "https://api.moonshot.ai/v1", "model": "kimi-k2.5", "temp": 0.3},
    "claude": {"base_url": "https://api.anthropic.com/v1", "model": "claude-sonnet-4-6-20250514", "native": True},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "together": {"base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3"},
}


# --- PII Stripping ---

_PII_PATTERNS = [
    (r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', '[EMAIL]'),
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]'),
    (r'\b(?:sk|pk|api)[_-][a-zA-Z0-9]{20,}\b', '[API_KEY]'),
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]'),
]


def strip_pii(text: str) -> tuple:
    """Strip PII from text. Returns (cleaned_text, replacements_map)."""
    replacements = {}
    cleaned = text
    for pattern, placeholder in _PII_PATTERNS:
        matches = re.findall(pattern, cleaned)
        for i, match in enumerate(matches):
            key = f"{placeholder}_{i+1}" if len(matches) > 1 else placeholder
            replacements[key] = match
            cleaned = cleaned.replace(match, key, 1)
    return cleaned, replacements


def restore_pii(text: str, replacements: dict) -> str:
    """Restore PII placeholders back to original values."""
    restored = text
    for placeholder, original in replacements.items():
        restored = restored.replace(placeholder, original)
    return restored


# --- Cloud Engine ---

class CloudEngine:
    """OpenAI-compatible cloud escalation engine."""

    def __init__(self, provider: str = "kimi", api_key: str = "",
                 strip_pii_enabled: bool = True):
        self.provider = provider
        self.api_key = api_key
        self.strip_pii_enabled = strip_pii_enabled

        conf = PROVIDERS.get(provider, PROVIDERS["kimi"])
        self.base_url = conf["base_url"]
        self.model = conf["model"]
        self.default_temp = conf.get("temp", 0.0)
        self._native_anthropic = conf.get("native", False)

    @classmethod
    def from_config(cls, config_path: str = CONFIG_PATH) -> Optional['CloudEngine']:
        """Load cloud config from disk. Returns None if not configured."""
        if not os.path.isfile(config_path):
            return None
        try:
            with open(config_path) as f:
                config = json.load(f)
            cloud = config.get("cloud", {})
            provider = cloud.get("provider", "")
            api_key = cloud.get("api_key", "")
            if not provider or not api_key:
                return None
            return cls(
                provider=provider,
                api_key=api_key,
                strip_pii_enabled=cloud.get("strip_pii", True),
            )
        except Exception:
            return None

    def generate(self, system: str, user: str, max_tokens: int = 4096,
                 temp: float = 0.0) -> str:
        """Generate via cloud API. Strips PII if enabled."""
        # Strip PII
        pii_map = {}
        if self.strip_pii_enabled:
            user, pii_map = strip_pii(user)
            system, sys_pii = strip_pii(system)
            pii_map.update(sys_pii)

        # Call API
        if self._native_anthropic:
            response = self._call_anthropic(system, user, max_tokens, temp)
        else:
            response = self._call_openai_compat(system, user, max_tokens, temp)

        # Restore PII
        if pii_map:
            response = restore_pii(response, pii_map)

        return response

    def _call_openai_compat(self, system: str, user: str,
                             max_tokens: int, temp: float) -> str:
        """Call OpenAI-compatible endpoint (Kimi, Groq, Together, Ollama)."""
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temp or self.default_temp,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Cloud error ({self.provider}): {e}"

    def _call_anthropic(self, system: str, user: str,
                         max_tokens: int, temp: float) -> str:
        """Call Anthropic's native API."""
        import urllib.request

        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data["content"][0]["text"]
        except Exception as e:
            return f"Cloud error (claude): {e}"
