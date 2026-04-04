"""Vision tools — analyze images using Qwen 3.5 4B (natively multimodal).

Loads the vision model on first use and caches it for subsequent calls.
"""

from __future__ import annotations

import os
from typing import Optional

from kandiga.agents.tools import ToolRegistry

# Cached vision model — loaded on first use
_vision_model = None
_vision_processor = None
_VISION_MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"


def _get_vision_model():
    """Load and cache the vision model."""
    global _vision_model, _vision_processor
    if _vision_model is not None:
        return _vision_model, _vision_processor

    from mlx_vlm import load as vlm_load
    _vision_model, _vision_processor = vlm_load(_VISION_MODEL_ID)
    return _vision_model, _vision_processor


def analyze_image(path: str, question: str = "Describe this image in detail") -> str:
    """Analyze an image using Qwen 3.5 4B vision."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: image not found: {path}"

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
        return f"Error: unsupported image format: {ext}"

    try:
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_image

        model, processor = _get_vision_model()
        image = [load_image(path)]

        prompt = apply_chat_template(
            processor,
            config=model.config,
            prompt=question,
            num_images=1,
        )

        result = vlm_generate(
            model, processor, prompt, image,
            max_tokens=500, temperature=0.0, verbose=False,
        )
        return result.text if hasattr(result, 'text') else str(result)

    except ImportError:
        return "Error: mlx-vlm not installed — run: pip install mlx-vlm"
    except Exception as e:
        return f"Error analyzing image: {e}"


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


def register_vision_tools(registry: ToolRegistry) -> int:
    tools = [
        ("analyze_image", "Analyze an image file — describe contents, read text, identify objects, interpret diagrams/charts/medical images", {"path": "str", "question": "str"}, analyze_image),
        ("screenshot_analyze", "Take and analyze a screenshot of the current screen", {"question": "str"}, analyze_screenshot),
    ]
    for name, desc, schema, func in tools:
        registry.register(name, desc, schema, func)
    return len(tools)
