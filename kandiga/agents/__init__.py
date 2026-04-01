"""Kandiga agent layer — the full local AI assistant.

Tools: filesystem, shell, web search, browser, macOS native,
       iMessage, vision, MCP servers
Skills: OpenClaw-compatible SKILL.md, auto-creation from patterns
Memory: SQLite state store, MEMORY.md, daily notes, persistent KV cache
Scheduling: Cron jobs with delivery to iMessage/Telegram
Messaging: iMessage (native), Telegram (bot)

Usage:
    from kandiga.engine import KandigaEngine
    from kandiga.agents import AgentPipeline, full_registry

    engine = KandigaEngine(fast_mode=True)
    engine.load()

    registry = full_registry()
    pipeline = AgentPipeline(engine, registry=registry)
    result = pipeline.run("search the web for AI news and notify me")
"""

from kandiga.agents.protocol import AgentResult, ToolCall, ToolResult
from kandiga.agents.tools import ToolRegistry, default_tools
from kandiga.agents.pipeline import AgentPipeline
from kandiga.agents.skills import SkillEngine, Skill
from kandiga.agents.memory import Memory
from kandiga.agents.scheduler import Scheduler, ScheduledTask
from kandiga.agents.state import StateStore


def full_registry() -> ToolRegistry:
    """Create a registry with ALL tools — filesystem, web, browser, macOS, messaging, vision."""
    reg = default_tools()

    try:
        from kandiga.agents.macos import register_macos_tools
        register_macos_tools(reg)
    except Exception:
        pass

    try:
        from kandiga.agents.browser import register_browser_tools
        register_browser_tools(reg)
    except Exception:
        pass

    try:
        from kandiga.agents.messaging import register_messaging_tools
        register_messaging_tools(reg)
    except Exception:
        pass

    try:
        from kandiga.agents.vision import register_vision_tools
        register_vision_tools(reg)
    except Exception:
        pass

    return reg


__all__ = [
    "AgentResult", "ToolCall", "ToolResult",
    "ToolRegistry", "default_tools", "full_registry",
    "AgentPipeline",
    "SkillEngine", "Skill",
    "Memory",
    "Scheduler", "ScheduledTask",
    "StateStore",
]

try:
    from kandiga.agents.dual_engine import DualEngine
    __all__.append("DualEngine")
except ImportError:
    pass
