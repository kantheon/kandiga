"""Browser automation via Playwright — navigate, click, type, extract, screenshot.

Gives the agent the ability to interact with web pages like a human.
Hermes has this. Now we do too.
"""

from __future__ import annotations

import os
import subprocess
import json
import time
from typing import Optional

from kandiga.agents.tools import ToolRegistry

_browser = None
_page = None


def _ensure_playwright():
    """Lazy-init Playwright browser."""
    global _browser, _page
    if _page is not None:
        return True
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        _browser = pw.chromium.launch(headless=True)
        _page = _browser.new_page()
        _page.set_default_timeout(15000)
        return True
    except ImportError:
        return False
    except Exception:
        return False


def browser_navigate(url: str) -> str:
    """Navigate to a URL and return the page text content."""
    if not _ensure_playwright():
        return "Error: playwright not installed (pip install playwright && playwright install chromium)"
    try:
        _page.goto(url, wait_until="domcontentloaded", timeout=20000)
        time.sleep(1)
        title = _page.title()
        # Extract text content
        text = _page.inner_text("body")
        text = text[:5000] if text else "(empty page)"
        return f"Title: {title}\nURL: {_page.url}\n\n{text}"
    except Exception as e:
        return f"Error navigating to {url}: {e}"


def browser_click(selector: str) -> str:
    """Click an element on the current page."""
    if not _ensure_playwright():
        return "Error: playwright not available"
    try:
        _page.click(selector, timeout=5000)
        time.sleep(0.5)
        return f"Clicked: {selector}\nCurrent URL: {_page.url}"
    except Exception as e:
        return f"Error clicking {selector}: {e}"


def browser_type(selector: str, text: str) -> str:
    """Type text into an input field."""
    if not _ensure_playwright():
        return "Error: playwright not available"
    try:
        _page.fill(selector, text, timeout=5000)
        return f"Typed into {selector}: {text[:50]}"
    except Exception as e:
        return f"Error typing into {selector}: {e}"


def browser_screenshot(path: str = "/tmp/kandiga_screenshot.png") -> str:
    """Take a screenshot of the current page."""
    if not _ensure_playwright():
        return "Error: playwright not available"
    try:
        _page.screenshot(path=path, full_page=False)
        size = os.path.getsize(path)
        return f"Screenshot saved: {path} ({size} bytes)\nURL: {_page.url}"
    except Exception as e:
        return f"Error taking screenshot: {e}"


def browser_extract(selector: str = "body") -> str:
    """Extract text from a specific element."""
    if not _ensure_playwright():
        return "Error: playwright not available"
    try:
        text = _page.inner_text(selector, timeout=5000)
        return text[:5000] if text else "(empty)"
    except Exception as e:
        return f"Error extracting from {selector}: {e}"


def browser_links(filter_text: str = "") -> str:
    """List all links on the current page."""
    if not _ensure_playwright():
        return "Error: playwright not available"
    try:
        links = _page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({text: e.innerText.trim().slice(0,80), href: e.href})).filter(l => l.text)"
        )
        if filter_text:
            links = [l for l in links if filter_text.lower() in l["text"].lower()]
        if not links:
            return "(no links found)"
        return "\n".join(f"- {l['text']}: {l['href']}" for l in links[:30])
    except Exception as e:
        return f"Error listing links: {e}"


def browser_scroll(direction: str = "down") -> str:
    """Scroll the page up or down."""
    if not _ensure_playwright():
        return "Error: playwright not available"
    try:
        amount = 500 if direction == "down" else -500
        _page.evaluate(f"window.scrollBy(0, {amount})")
        return f"Scrolled {direction}"
    except Exception as e:
        return f"Error scrolling: {e}"


def browser_back() -> str:
    """Go back in browser history."""
    if not _ensure_playwright():
        return "Error: playwright not available"
    try:
        _page.go_back()
        return f"Navigated back. Current URL: {_page.url}"
    except Exception as e:
        return f"Error going back: {e}"


def register_browser_tools(registry: ToolRegistry) -> int:
    tools = [
        ("browse", "Navigate to a URL and read the page", {"url": "str"}, browser_navigate),
        ("browse_click", "Click an element on the page", {"selector": "str"}, browser_click),
        ("browse_type", "Type into an input field", {"selector": "str", "text": "str"}, browser_type),
        ("browse_screenshot", "Take a screenshot", {"path": "str"}, browser_screenshot),
        ("browse_extract", "Extract text from an element", {"selector": "str"}, browser_extract),
        ("browse_links", "List all links on the page", {"filter_text": "str"}, browser_links),
        ("browse_scroll", "Scroll the page", {"direction": "str"}, browser_scroll),
        ("browse_back", "Go back in browser history", {}, browser_back),
    ]
    for name, desc, schema, func in tools:
        registry.register(name, desc, schema, func)
    return len(tools)
