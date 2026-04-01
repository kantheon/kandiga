"""Vision tools — analyze images and screenshots.

Uses the local model's vision capabilities when available,
or falls back to describing file metadata.
"""

from __future__ import annotations

import base64
import os
from typing import Optional

from kandiga.agents.tools import ToolRegistry


def analyze_image(path: str, question: str = "Describe this image in detail") -> str:
    """Analyze an image using the vision model."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: image not found: {path}"

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
        return f"Error: unsupported image format: {ext}"

    size = os.path.getsize(path)

    # Try MLX vision model
    try:
        return _analyze_with_mlx(path, question)
    except Exception:
        pass

    # Fallback: file metadata
    return (
        f"Image: {path}\n"
        f"Format: {ext}\n"
        f"Size: {size:,} bytes\n"
        f"(Vision model not available — install mlx-vlm for image analysis)"
    )


def analyze_screenshot(question: str = "What is shown on this screen?") -> str:
    """Take a screenshot and analyze it."""
    import subprocess
    path = "/tmp/kandiga_screen_analysis.png"
    try:
        subprocess.run(["screencapture", "-x", path], timeout=5)
        if os.path.isfile(path):
            return analyze_image(path, question)
        return "Error: screenshot failed"
    except Exception as e:
        return f"Error taking screenshot: {e}"


def _analyze_with_mlx(path: str, question: str) -> str:
    """Try to analyze with MLX vision model."""
    try:
        from mlx_vlm import load as vlm_load, generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_image

        model_path = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

        # Check if model is cached
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(model_path, local_files_only=True)

        model, processor = vlm_load(model_path)
        image = load_image(path)

        prompt = apply_chat_template(
            processor,
            config=model.config,
            prompt=question,
            num_images=1,
        )

        output = vlm_generate(
            model, processor, prompt, image,
            max_tokens=500, verbose=False,
        )
        return output

    except ImportError:
        raise ImportError("mlx-vlm not installed")
    except Exception as e:
        raise RuntimeError(f"Vision analysis failed: {e}")


def register_vision_tools(registry: ToolRegistry) -> int:
    tools = [
        ("analyze_image", "Analyze an image file", {"path": "str", "question": "str"}, analyze_image),
        ("screenshot_analyze", "Take and analyze a screenshot", {"question": "str"}, analyze_screenshot),
    ]
    for name, desc, schema, func in tools:
        registry.register(name, desc, schema, func)
    return len(tools)
