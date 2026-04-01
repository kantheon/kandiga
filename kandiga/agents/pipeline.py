"""Agent pipeline — dual-model or single-model, tools, verification, skills.

Dual-model mode (default):
  4B handles: tool call JSON, classification, content generation (~34 tok/s)
  35B handles: planning, final synthesis, complex reasoning (~3-6 tok/s)
  Result: tool queries ~15s instead of ~76s

Single-model mode (fallback):
  35B handles everything (slower but still works)

Pipeline:
  1. Classify query via keywords (instant, no model call)
  2. If tools: 4B generates tool JSON → execute → 35B synthesizes
  3. If multi-step: 35B plans → 4B executes each step → 35B synthesizes
  4. Verify result (deterministic cross-check)
  5. Return typed AgentResult
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set

from kandiga.agents.json_repair import (
    extract_write_file,
    parse_json,
    validate_plan,
    validate_tool_calls,
)
from kandiga.agents.protocol import AgentResult, ToolCall, ToolResult
from kandiga.agents.tools import ToolRegistry, default_tools


# ---------------------------------------------------------------------------
# Keyword detection (instant, no model call)
# ---------------------------------------------------------------------------

_ACTION_WORDS: Set[str] = {
    "create", "write", "save", "make", "read", "open", "delete",
    "remove", "list", "find", "search", "run", "execute", "grep",
    "mkdir", "copy", "move", "rename", "replace", "change", "add",
    "edit", "modify", "update", "rewrite", "fix", "correct", "improve",
    "schedule", "remind", "note", "email", "send", "check", "show",
    "notify", "reveal", "speak", "say", "look", "google", "browse",
}

_FILE_EXTENSIONS = (
    ".txt", ".py", ".md", ".json", ".csv", ".html", ".js", ".ts",
    ".yaml", ".toml", ".sh", ".xml", ".sql", ".log", ".docx", ".pdf",
)

_MULTI_STEP_SIGNALS = (" then ", " and then ", " after that ", ", then ")


def _needs_tools(query: str) -> bool:
    words = set(query.lower().split())
    q = query.lower()
    if _ACTION_WORDS & words:
        targets = {"file", "files", "folder", "directory", "dir", "code",
                    "content", "calendar", "reminder", "note", "email", "meeting",
                    "notification", "finder", "contact", "contacts", "event",
                    "web", "internet", "online", "benchmark", "performance"}
        if targets & words:
            return True
    if any(ext in q for ext in _FILE_EXTENSIONS):
        return True
    if "/" in q and any(w in words for w in ("read", "write", "save", "create", "list", "open")):
        return True
    # Direct triggers — these always need tools
    direct_triggers = [
        "search the web", "web search", "google", "look up",
        "send notification", "send me a notification", "notify me",
        "create reminder", "add reminder", "remind me",
        "create event", "schedule", "add to calendar",
        "create note", "add note", "search notes",
        "open in finder", "reveal in finder",
        "what's on my calendar", "what is on my calendar", "my reminders", "my contacts",
        "say aloud", "say hello", "speak", "read aloud", "aloud",
        "system info", "hardware info",
    ]
    if any(t in q for t in direct_triggers):
        return True

    # Current-event / factual queries that need web search
    current_triggers = [
        "news", "latest", "today", "this week", "this month",
        "current", "recent", "right now", "happening",
        "price of", "stock price", "weather", "score",
        "who won", "what happened", "trending",
        "how much is", "how much does",
    ]
    if any(t in q for t in current_triggers):
        return True

    # Action requests — user wants the agent to DO something
    action_triggers = [
        "can you start", "can you run", "can you execute",
        "start it", "run it", "execute it", "do it",
        "launch it", "open it", "stop it", "kill it",
        "start the", "run the", "execute the", "stop the",
        "install", "download", "deploy", "build",
        "is it running", "is it working", "check if",
    ]
    if any(t in q for t in action_triggers):
        return True

    return False


def _needs_multi_step(query: str) -> bool:
    q = query.lower()
    if any(c in q for c in _MULTI_STEP_SIGNALS):
        return True
    write_w = ("write ", "create ", "save ", "generate ")
    file_w = ("file", ".txt", ".py", ".md", ".json", ".csv", ".html")
    if any(w in q for w in write_w) and any(f in q for f in file_w):
        return True
    edit_w = ("replace ", "change ", "edit ", "modify ", "fix ", "rewrite ")
    if any(w in q for w in edit_w) and any(f in q for f in file_w):
        return True
    # Two or more action verbs = multi-step (even with "and" not "then")
    verbs = ["read", "write", "list", "find", "search", "run", "create", "delete",
             "save", "replace", "edit", "fix", "add", "remove", "copy", "move",
             "start", "stop", "check", "send", "open"]
    found = sum(1 for v in verbs if v in q.split())  # exact word match
    return found >= 2


# ---------------------------------------------------------------------------
# Verification (deterministic, no model call)
# ---------------------------------------------------------------------------

def _verify(response: str, context: str, tool_results: List[ToolResult]) -> tuple:
    resp = response.lower()
    ctx = context.lower()
    combined = resp + " " + ctx
    flags = []
    confidence = 0.85

    any_failed = any(not tr.success for tr in tool_results)
    any_wrote = any(tr.success and tr.tool == "write_file" for tr in tool_results)
    any_read = any(tr.success and tr.tool == "read_file" for tr in tool_results)

    error_patterns = ["error:", "file not found", "not a directory", "permission denied"]
    has_error = any(p in ctx for p in error_patterns) or any_failed

    refusal_phrases = ["i cannot", "i can't", "i'm unable", "i am unable",
                       "i don't have access", "beyond my capabilities"]
    is_refusal = any(p in resp for p in refusal_phrases)

    claims_saved = any(p in resp for p in ["has been saved", "saved the file",
                                            "saved successfully", "written to"])
    has_write_confirm = any_wrote or ("wrote" in combined and "bytes" in combined)
    false_claim = claims_saved and not has_write_confirm and not has_error

    if has_error:
        confidence = min(confidence, 0.3)
        flags.append("Tool error detected")
        for tr in tool_results:
            if not tr.success:
                flags.append(f"{tr.tool}: {tr.error}")

    if is_refusal:
        confidence = min(confidence, 0.25)
        flags.append("Response refuses to help")

    if false_claim:
        confidence = min(confidence, 0.4)
        flags.append("Claims saved but no write confirmation")

    if (any_wrote or any_read) and not has_error and not is_refusal and not false_claim:
        confidence = max(confidence, 0.92)

    if not tool_results and not has_error and not is_refusal and not false_claim:
        confidence = 0.85

    # If all tools succeeded and no errors, boost confidence
    all_ok = tool_results and all(tr.success for tr in tool_results)
    if all_ok and not has_error and not is_refusal:
        confidence = max(confidence, 0.90)

    return confidence, confidence >= 0.7, flags


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_HOME = os.path.expanduser("~")

TOOL_PROMPT = (
    "You have these tools:\n{tool_descriptions}\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"tool_calls": [{{"tool": "TOOL_NAME", "args": {{"key": "value"}}}}], "reasoning": "brief"}}\n\n'
    "Rules:\n"
    "- Use write_file to create files, NOT run_shell\n"
    "- Use read_file to read files, NOT run_shell with cat\n"
    "- Use list_dir to list dirs, NOT run_shell with ls\n"
    "- When creating a script, ALSO add a run_shell call to execute it\n"
    f"- The user's home directory is {_HOME}\n"
    f"- Downloads = {_HOME}/Downloads, Documents = {_HOME}/Documents\n"
    "- ALWAYS use full absolute paths starting with the home directory above\n"
    "- Multiple tool_calls can be in one response — do write THEN run\n"
    "- If a search finds nothing, try broader search or different path\n"
    "- Output ONLY valid JSON, nothing else"
)

PLAN_PROMPT = (
    "You are a task planner with these tools:\n{tool_descriptions}\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"plan": [{{"step": 1, "action": "tool_name", "description": "what", "depends_on": []}}]}}\n\n'
    "Rules:\n"
    "- action must be an exact tool name\n"
    "- For edit operations: plan read_file then write_file\n"
    f"- Home directory is {_HOME}. Use full absolute paths.\n"
    "- When creating a script, include a run_shell step to execute it"
)

SYNTH_PROMPT = (
    "You are Kandiga, a powerful local AI agent running on the user's Mac. "
    "You have FULL access to the filesystem, shell, browser, calendar, reminders, "
    "notes, iMessage, web search, and more. You CAN execute commands, start processes, "
    "read/write files, and interact with macOS apps. "
    f"The user's home directory is {_HOME}. "
    f"Downloads folder is {_HOME}/Downloads. "
    f"Documents folder is {_HOME}/Documents. "
    f"Desktop is {_HOME}/Desktop. "
    "ALWAYS use full absolute paths starting with this home directory. "
    "When the user asks you to DO something, DO IT using your tools. "
    "Never say 'I cannot', 'I'm unable', or 'you need to do it yourself'. "
    "If something failed, explain the error and try a different approach. "
    "Be direct, concise, and action-oriented."
)

StageCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AgentPipeline:
    """Agent layer on Kandiga. Supports dual-model (4B+35B) or single-model.

    Dual-model usage:
        from kandiga.agents.dual_engine import DualEngine
        from kandiga.agents import AgentPipeline

        engine = DualEngine(fast_mode=True)
        engine.load()
        pipeline = AgentPipeline(engine)
        result = pipeline.run("read config.yaml and fix the bug")

    Single-model usage:
        from kandiga.engine import KandigaEngine
        from kandiga.agents import AgentPipeline

        engine = KandigaEngine()
        engine.load()
        pipeline = AgentPipeline(engine)
        result = pipeline.run("explain recursion")
    """

    def __init__(
        self,
        engine,
        registry: Optional[ToolRegistry] = None,
        working_dirs: Optional[List[str]] = None,
        max_steps: int = 6,
        on_stage: Optional[StageCallback] = None,
        state_store=None,
        pattern_tracker=None,
        skill_engine=None,
    ):
        self.engine = engine
        self.registry = registry or default_tools()
        self.working_dirs = working_dirs or []
        self.max_steps = max_steps
        self.on_stage = on_stage or (lambda *_: None)

        # Detect if we have a DualEngine (4B + 35B K=2/K=4)
        self._dual = hasattr(engine, 'generate_fast') and hasattr(engine, 'generate_brain')
        self._has_check = hasattr(engine, 'generate_check')  # K=4 verification

        # Session state for multi-turn (persistent KV cache)
        self._session_active = False
        self._turn_count = 0
        self._session_id: Optional[str] = None

        # Optional systems (wired in when provided)
        self._state_store = state_store
        self._pattern_tracker = pattern_tracker
        self._skill_engine = skill_engine

        # Token streaming callback: fn(token: str) -> None
        self._on_token: Optional[Callable] = None

        # Cloud escalation
        self._cloud = None
        self._escalation_threshold = 0.3
        try:
            from kandiga.agents.cloud import CloudEngine
            self._cloud = CloudEngine.from_config()
            if self._cloud:
                # Load threshold from config
                import json as _json
                config_path = os.path.expanduser("~/.kandiga/config.json")
                if os.path.isfile(config_path):
                    with open(config_path) as f:
                        cfg = _json.load(f)
                    self._escalation_threshold = cfg.get("cloud", {}).get("threshold", 0.5)
        except Exception:
            pass

    # --- Session management (persistent KV cache) ---

    def start_session(self) -> None:
        """Start a persistent session. Follow-up turns only process new tokens.

        This is the key advantage over OpenClaw/Hermes: turn 50 takes
        the same time as turn 1 because previous turns are in the KV cache.
        """
        brain = self.engine if not self._dual else self.engine.brain
        if hasattr(brain, 'start_session'):
            brain.start_session()
        self._session_active = True
        self._turn_count = 0

        # Create DB session
        if self._state_store:
            model = ""
            if hasattr(self.engine, 'model_path'):
                model = self.engine.model_path
            self._session_id = self._state_store.create_session(model=model)

        self.on_stage("session", "started (persistent KV cache active)")

    def end_session(self) -> None:
        """End session, free KV cache memory."""
        brain = self.engine if not self._dual else self.engine.brain
        if hasattr(brain, 'end_session'):
            brain.end_session()
        self._session_active = False
        self._turn_count = 0
        self.on_stage("session", "ended")

    def save_session(self, path: str) -> None:
        """Save session to disk — survive restarts."""
        brain = self.engine if not self._dual else self.engine.brain
        if hasattr(brain, 'save_session'):
            brain.save_session(path)
            self.on_stage("session", f"saved to {path}")

    def load_session(self, path: str) -> None:
        """Load session from disk — instant resume."""
        brain = self.engine if not self._dual else self.engine.brain
        if hasattr(brain, 'load_session'):
            brain.load_session(path)
            self._session_active = True
            self.on_stage("session", f"loaded from {path}")

    def run(self, query: str, context: str = "") -> AgentResult:
        """Run the full agent pipeline.

        Routing: 4B model classifies intent (fast, ~1-2s), with keyword fallback.
        """
        t_start = time.time()

        # Expand ~ in query so models never see ambiguous paths
        query = query.replace("~/", f"{_HOME}/").replace("~ ", f"{_HOME} ")

        # Empty query — just return
        if not query.strip():
            return AgentResult(content="", confidence=1.0, verified=True, route="direct")

        full_ctx = context
        if self.working_dirs:
            dirs = "\n".join(f"  - {d}" for d in self.working_dirs)
            full_ctx = f"Accessible directories:\n{dirs}\n\n{full_ctx}" if full_ctx else f"Accessible directories:\n{dirs}"

        # Route via 4B model (fast) + keyword fallback
        route = self._classify_route(query, full_ctx)

        if route == "agentic":
            self.on_stage("route", "agentic")
            result = self._run_agentic(query, full_ctx)
        elif route == "tool":
            self.on_stage("route", "tool")
            result = self._run_tool(query, full_ctx)
        elif route == "think":
            self.on_stage("route", "think (35B)")
            result = self._run_think(query, full_ctx)
        else:
            self.on_stage("route", "direct")
            result = self._run_direct(query, full_ctx)

        result.duration_ms = int((time.time() - t_start) * 1000)

        # --- Post-pipeline hooks ---

        # Save to SQLite state store
        if self._state_store and self._session_id:
            try:
                turn = self._turn_count
                tools_used = [tr.tool for tr in result.tool_results]
                self._state_store.add_message(
                    self._session_id, turn, "user", query,
                    route=result.route, duration_ms=0,
                )
                msg_id = self._state_store.add_message(
                    self._session_id, turn, "assistant", result.content,
                    route=result.route, confidence=result.confidence,
                    verified=result.verified, tools_used=tools_used,
                    duration_ms=result.duration_ms,
                )
                for tr in result.tool_results:
                    self._state_store.add_tool_execution(
                        self._session_id, msg_id, tr.tool, tr.args,
                        str(tr.output)[:2000] if tr.output else "",
                        tr.success, tr.error or "", tr.duration_ms,
                    )
            except Exception:
                pass

        # Track pattern for auto-skill creation
        if self._pattern_tracker:
            try:
                tools_used = [tr.tool for tr in result.tool_results]
                suggestion = self._pattern_tracker.track(
                    query, result.route, tools_used, result.confidence >= 0.7,
                )
                if suggestion:
                    result.flags.append(
                        f"Pattern detected ({suggestion['count']}x): "
                        f"'{suggestion['pattern']}' — consider creating a skill"
                    )
            except Exception:
                pass

        # Match skills and inject into context for future queries
        if self._skill_engine:
            try:
                matched = self._skill_engine.match(query)
                if matched:
                    result.flags.append(f"Skill matched: {matched[0].name}")
            except Exception:
                pass

        return result

    # --- Routing ---

    ROUTE_PROMPT = (
        "You are a query classifier. Decide what kind of response is needed.\n"
        "You have tools for: files, shell, web search, browser, calendar, "
        "reminders, notes, iMessage, screenshots.\n\n"
        "Respond with ONLY one word:\n"
        "- TOOL — user wants actions done (read/write/run/search/create/list)\n"
        "- AGENTIC — user wants MULTIPLE actions in sequence\n"
        "- THINK — complex question needing deep reasoning (explain, analyze, compare, debug code, plan)\n"
        "- DIRECT — simple greeting or quick factual answer\n\n"
        "Examples:\n"
        "- 'hello' → DIRECT\n"
        "- '2+2' → DIRECT\n"
        "- 'read config.yaml' → TOOL\n"
        "- 'search the web for AI news' → TOOL\n"
        "- 'run it' → TOOL\n"
        "- 'create a file and then run it' → AGENTIC\n"
        "- 'read the file, fix the bug, save it' → AGENTIC\n"
        "- 'explain how this code works' → THINK\n"
        "- 'analyze this data and tell me what patterns you see' → THINK\n"
        "- 'what architecture should I use for this project' → THINK\n"
        "- 'debug this error message' → THINK\n"
        "- 'compare Python vs Rust for this use case' → THINK\n"
        "- 'whats the news this week' → TOOL\n"
        "- 'yes do it' → TOOL\n\n"
        "Respond with ONLY: TOOL, AGENTIC, THINK, or DIRECT"
    )

    def _classify_route(self, query: str, context: str) -> str:
        """Classify query intent using 4B model (fast) + keyword fallback."""

        # Keyword fast-path for obvious cases
        q = query.lower().strip()
        if q in ("hi", "hello", "hey", "thanks", "thank you", "ok", "yes", "no"):
            return "direct"

        # Use 4B model to classify
        self.on_stage("classify", "4B routing...")
        try:
            user_msg = query
            if context:
                # Include recent context so model knows what "it" refers to
                user_msg = f"Recent context: {context[-500:]}\n\nUser message: {query}"

            raw = self._gen_fast(self.ROUTE_PROMPT, user_msg, max_tokens=10)
            raw = raw.strip().upper()

            if "AGENTIC" in raw:
                return "agentic"
            elif "TOOL" in raw:
                return "tool"
            elif "THINK" in raw:
                # Override THINK → TOOL if keywords clearly indicate tool use
                # The model sometimes classifies action requests as "thinking"
                if _needs_tools(query):
                    return "agentic" if _needs_multi_step(query) else "tool"
                return "think"
            elif "DIRECT" in raw:
                # Override DIRECT → TOOL if keywords clearly indicate tool use
                if _needs_tools(query):
                    return "agentic" if _needs_multi_step(query) else "tool"
                return "direct"
        except Exception:
            pass

        # Keyword fallback
        if _needs_tools(query):
            if _needs_multi_step(query):
                return "agentic"
            return "tool"

        return "direct"

    # --- Generation helpers ---

    def _gen_fast(self, system: str, user: str, max_tokens: int = 1200) -> str:
        """Fast generation — 4B in dual mode, engine in single mode."""
        if self._dual:
            return self.engine.generate_fast(system, user, max_tokens=max_tokens)
        return self._gen_single(system, user, max_tokens)

    def _gen_brain(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Brain generation — 35B K=2. Always fresh (no session KV contamination)."""
        if self._dual:
            if self._on_token:
                return self._gen_brain_stream(system, user, max_tokens)
            return self.engine.generate_brain(system, user, max_tokens=max_tokens)
        return self._gen_single_stream(system, user, max_tokens)

    def _gen_brain_stream(self, system: str, user: str, max_tokens: int) -> str:
        """Stream from 35B brain, yielding tokens via callback."""
        if hasattr(self.engine, 'generate_brain_stream'):
            tokens = []
            for token in self.engine.generate_brain_stream(system, user, max_tokens=max_tokens):
                tokens.append(token)
                if self._on_token:
                    self._on_token(token)
            return "".join(tokens)
        return self.engine.generate_brain(system, user, max_tokens=max_tokens)

    def _gen_single_stream(self, system: str, user: str, max_tokens: int) -> str:
        """Stream from single engine."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            formatted = self.engine._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self.engine._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        if self._on_token and hasattr(self.engine, 'generate'):
            try:
                tokens = []
                for token in self.engine.generate(formatted, max_tokens=max_tokens, temp=0.0, stream=True):
                    tokens.append(token)
                    self._on_token(token)
                return "".join(tokens)
            except Exception:
                pass

        return self.engine.generate(formatted, max_tokens=max_tokens, temp=0.0, stream=False)

    def _gen_single(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Generate with single KandigaEngine."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            formatted = self.engine._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self.engine._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return self.engine.generate(formatted, max_tokens=max_tokens, temp=0.0, stream=False)

    # --- 35B Verification (Nightingale pattern) ---

    VERIFY_PROMPT = (
        "You are a verifier for Kandiga, a local AI agent that CAN execute commands, "
        "read/write files, and interact with macOS. Check the response below.\n\n"
        "IMPORTANT: Kandiga HAS tool access. It CAN run scripts, create files, "
        "send notifications, etc. Do NOT say 'I cannot execute' — the tools already ran.\n\n"
        "If the response is correct and complete, reply: VERIFIED\n"
        "If there are factual errors in the tool results, reply with a brief correction.\n"
        "Do NOT rewrite the entire response. Do NOT add 'I cannot' statements."
    )

    def _verify_with_brain(self, response: str, query: str, context: str) -> Optional[str]:
        """Use 35B to verify a 4B-generated response. Returns corrected text or None.

        Only returns a replacement if the 35B provides a genuine correction,
        not if it adds refusal language.
        """
        user = (
            f"User query: {query}\n\n"
            f"Context (tool results are real): {context[-1000:]}\n\n"
            f"Response to verify:\n{response}"
        )
        try:
            result = self._gen_brain(self.VERIFY_PROMPT, user, max_tokens=512)
            if not result:
                return None
            r = result.strip().upper()
            # Accept verification
            if "VERIFIED" in r[:20]:
                return None
            # Reject if the 35B added refusal language
            refusals = ["i cannot", "i can't", "i'm unable", "i am unable",
                        "i don't have", "cannot execute", "cannot run"]
            if any(ref in result.lower() for ref in refusals):
                return None  # ignore refusal — 4B's response was better
            return result
        except Exception:
            return None

    # --- Direct (no tools) ---

    def _run_direct(self, query: str, context: str) -> AgentResult:
        # 35B K=2 writes the response (fast, ~10-13 tok/s)
        self.on_stage("brain-k2", "Writing response (35B K=2)...")
        user = f"Context:\n{context}\n\nQuery: {query}" if context else query
        content = self._gen_brain(SYNTH_PROMPT, user)

        conf, verified, flags = _verify(content, context, [])
        self.on_stage("done", f"confidence={conf:.2f}")
        return AgentResult(content=content, confidence=conf, verified=verified,
                           route="direct", flags=flags)

    # --- Think (35B brain for complex reasoning) ---

    THINK_PROMPT = (
        "You are Kandiga, a local AI agent. Answer this question using your knowledge. "
        "Be thorough, accurate, and helpful. "
        "IMPORTANT: Only discuss information that is in the context provided. "
        "If you need to read files or check real data to answer, say so — "
        "do NOT make up file contents, paths, or error messages. "
        "If the context is empty or irrelevant, answer from general knowledge only."
    )

    def _run_think(self, query: str, context: str) -> AgentResult:
        """Use the 35B brain for complex reasoning tasks."""
        self.on_stage("brain", "Thinking (35B)...")
        user = f"Context:\n{context}\n\nQuery: {query}" if context else query
        content = self._gen_brain(self.THINK_PROMPT, user)

        # If brain returned empty (thinking exhausted), fall back to 4B
        if not content or not content.strip():
            self.on_stage("fast", "Fallback to 4B...")
            content = self._gen_fast(SYNTH_PROMPT, user, max_tokens=2048)

        conf = 0.9  # 35B responses get high confidence
        self.on_stage("done", f"confidence={conf:.2f}")
        return AgentResult(content=content, confidence=conf, verified=True,
                           route="think", flags=[])

    # --- Autonomous tool loop ---

    LOOP_PROMPT = (
        "You are an autonomous agent. You have completed some tool calls.\n"
        "Decide: is the task DONE, or do you need MORE tool calls?\n\n"
        "Tool results so far:\n{tool_output}\n\n"
        "Original task: {query}\n\n"
        "Respond with ONLY valid JSON:\n"
        '{{"done": true, "summary": "what was accomplished"}}\n'
        "OR\n"
        '{{"done": false, "next_tool_calls": [{{"tool": "TOOL", "args": {{}}}}], "reasoning": "why"}}\n\n'
        "Rules:\n"
        "- If the task is complete, set done=true\n"
        "- If more work is needed (search, read, write, compare, verify), set done=false and specify next tools\n"
        "- If a tool failed, try a different approach\n"
        f"- Home directory: {_HOME}\n"
        "- Output ONLY valid JSON"
    )

    def _run_tool(self, query: str, context: str) -> AgentResult:
        """Autonomous tool loop — keeps going until the task is done.

        Unlike simple one-shot: the agent executes tools, checks results,
        decides if more work is needed, and continues. Up to max_steps iterations.
        This is what OpenClaw/Hermes do — multi-turn autonomous execution.
        """
        tool_system = TOOL_PROMPT.format(tool_descriptions=self.registry.describe_tools())
        defaults = {"tool_calls": [], "reasoning": ""}
        all_results: List[ToolResult] = []
        accumulated = ""
        iteration = 0
        seen_errors: set = set()  # track errors to avoid repeating

        while iteration < self.max_steps:
            iteration += 1

            # Build context with accumulated results
            if accumulated:
                step_ctx = f"{context}\n\nPrevious results:\n{accumulated}" if context else f"Previous results:\n{accumulated}"
            else:
                step_ctx = context

            # 4B generates tool calls
            self.on_stage("fast", f"Step {iteration}: parsing tool calls...")
            user = f"Context:\n{step_ctx}\n\nTask: {query}" if step_ctx else query
            raw = self._gen_fast(tool_system, user, max_tokens=1200)

            parsed = parse_json(raw, defaults)
            calls = validate_tool_calls(parsed, self.registry.tool_names)

            if not calls and "write_file" in raw:
                extracted = extract_write_file(raw)
                if extracted:
                    calls = [extracted]

            if not calls and iteration == 1:
                calls = self._infer_tool(query)

            if not calls:
                break  # model has nothing more to do

            calls = self._fix_shell(calls)

            # Execute tools
            self.on_stage("tools", f"Step {iteration}: executing {len(calls)} tool(s)...")
            step_results = []
            for c in calls:
                tc = ToolCall(tool=c["tool"], args=c.get("args", {}))
                tr = self.registry.execute(tc)
                step_results.append(tr)
                all_results.append(tr)

            # Accumulate results + detect repeated errors
            for tr in step_results:
                status = "OK" if tr.success else "FAILED"
                output = str(tr.output)[:500] if tr.output else (tr.error or "")
                accumulated += f"\n[{tr.tool}] ({status}) {output}"
                if not tr.success and tr.error:
                    err_key = f"{tr.tool}:{tr.error[:80]}"
                    if err_key in seen_errors:
                        # Same error twice — stop looping, it won't fix itself
                        self.on_stage("tools", "Same error repeated — stopping loop")
                        iteration = self.max_steps  # force exit
                        break
                    seen_errors.add(err_key)

            # Decision: should we loop or stop?
            all_succeeded = all(tr.success for tr in step_results)

            # If all tools succeeded on first try and query isn't explicitly multi-step: STOP
            if iteration == 1 and all_succeeded and not _needs_multi_step(query):
                break

            # If all tools succeeded and we've done multiple iterations: STOP
            if iteration > 1 and all_succeeded:
                break

            # If tools failed, retry (the loop continues automatically)
            if not all_succeeded and iteration < self.max_steps:
                self.on_stage("fast", f"Some tools failed — retrying...")
                continue

            # For multi-step queries, ask the 2B if there's more to do
            if _needs_multi_step(query) and all_succeeded:
                self.on_stage("fast", "Checking if task is complete...")
                loop_check = self._gen_fast(
                    self.LOOP_PROMPT.format(tool_output=accumulated[-1500:], query=query),
                    "Is the task complete? Reply ONLY JSON: {\"done\": true} or {\"done\": false}",
                    max_tokens=100,
                )
                loop_parsed = parse_json(loop_check, {"done": True})
                if loop_parsed.get("done", True):
                    break

        # Build final tool output
        tool_output = "\n\n".join(
            f"[{tr.tool}] {'OK' if tr.success else 'FAILED'}\n{str(tr.output)[:300] if tr.output else tr.error or ''}"
            for tr in all_results
        )

        # 35B K=2 writes the response (fast but smart)
        self.on_stage("brain-k2", "Writing response (35B K=2)...")
        synth_ctx = f"{context}\n\nTool results:\n{tool_output}" if context else f"Tool results:\n{tool_output}"
        content = self._gen_brain(SYNTH_PROMPT, f"Context:\n{synth_ctx}\n\nQuery: {query}")

        conf, verified, flags = _verify(content, synth_ctx, all_results)

        # 35B K=4 verifies if there were errors (Nightingale pattern)
        if self._has_check and any(not tr.success for tr in all_results):
            self.on_stage("brain-k4", "Verifying (35B K=4)...")
            check = self.engine.generate_check(
                "Check this response against the tool results. "
                "If claims don't match tool output, reply with ONLY the corrected text. "
                "If correct, reply: VERIFIED. "
                "Do NOT add 'I cannot' or disclaimers.",
                f"Tool results:\n{tool_output}\n\nResponse:\n{content}",
            )
            if check and "VERIFIED" not in check.upper()[:20]:
                # Reject if it added refusal language
                refusals = ["i cannot", "i can't", "i'm unable"]
                if not any(r in check.lower() for r in refusals):
                    content = check
                    conf, verified, flags = _verify(content, synth_ctx, all_results)

        # Cloud escalation if confidence is low and cloud is configured
        if conf < (self._escalation_threshold or 0.3) and self._cloud:
            self.on_stage("cloud", f"Escalating to {self._cloud.provider}...")
            cloud_content = self._cloud.generate(
                SYNTH_PROMPT,
                f"Context:\n{synth_ctx}\n\nQuery: {query}",
            )
            if cloud_content and "error" not in cloud_content.lower()[:20]:
                content = cloud_content
                conf = 0.9
                verified = True
                flags.append(f"escalated to {self._cloud.provider}")

        if iteration > 1:
            flags.append(f"autonomous loop: {iteration} iterations")

        self.on_stage("done", f"confidence={conf:.2f}")
        return AgentResult(content=content, confidence=conf, verified=verified,
                           route="tool", tool_results=all_results, flags=flags)

    # --- Agentic (multi-step) ---

    def _run_agentic(self, query: str, context: str) -> AgentResult:
        # 4B creates the plan (fast — it's just JSON generation)
        self.on_stage("fast", "Planning...")
        plan_system = PLAN_PROMPT.format(tool_descriptions=self.registry.describe_tools())
        user = f"Context:\n{context}\n\nTask: {query}" if context else query
        raw = self._gen_fast(plan_system, user, max_tokens=512)
        parsed = parse_json(raw, {"plan": []})
        steps = validate_plan(parsed)

        if not steps:
            return self._run_tool(query, context)

        plan_summary = "; ".join(f"{s['action']}: {s['description']}" for s in steps)
        self.on_stage("plan", f"{len(steps)} steps")

        # 4B executes each step (fast)
        all_results: List[ToolResult] = []
        accumulated = ""
        consecutive_errors = 0

        for i, step in enumerate(steps[:self.max_steps]):
            if consecutive_errors >= 2:
                break

            self.on_stage("fast", f"Step {i+1}/{len(steps)}: {step['description'][:60]}")
            step_ctx = f"{context}\n\nPrior results:\n{accumulated}" if accumulated else context

            if step["action"] == "write_file":
                # 4B generates content (fast)
                content = self._gen_fast(
                    SYNTH_PROMPT,
                    f"Context:\n{step_ctx}\n\nGenerate content for: {step['description']}\nOriginal request: {query}",
                    max_tokens=2048,
                )
                path_m = re.search(r"(?:to|at|in)\s+(/[^\s,]+|\S+\.\w+)", step["description"])
                path = path_m.group(1) if path_m else "output.txt"
                tc = ToolCall(tool="write_file", args={"path": path, "content": content})
                tr = self.registry.execute(tc)
            else:
                # 4B generates tool call JSON (fast)
                tool_raw = self._gen_fast(
                    TOOL_PROMPT.format(tool_descriptions=self.registry.describe_tools()),
                    f"Context:\n{step_ctx}\n\nTask: {step['description']}",
                    max_tokens=1200,
                )
                step_parsed = parse_json(tool_raw, {"tool_calls": []})
                step_calls = validate_tool_calls(step_parsed, self.registry.tool_names)
                if not step_calls:
                    step_calls = [{"tool": step["action"], "args": {}}]
                tc = ToolCall(tool=step_calls[0]["tool"], args=step_calls[0].get("args", {}))
                tr = self.registry.execute(tc)

            all_results.append(tr)
            output = str(tr.output) if tr.output else (tr.error or "")
            accumulated += f"\nStep {i+1} [{tr.tool}]: {output[:500]}"

            if not tr.success:
                consecutive_errors += 1
            else:
                consecutive_errors = 0

        # 35B K=2 writes the synthesis (smart + fast)
        self.on_stage("brain-k2", "Writing response (35B K=2)...")
        synth_ctx = f"{context}\n\nExecution results:\n{accumulated}" if context else f"Execution results:\n{accumulated}"
        content = self._gen_brain(SYNTH_PROMPT, f"Context:\n{synth_ctx}\n\nQuery: {query}")

        conf, verified, flags = _verify(content, synth_ctx, all_results)

        if consecutive_errors >= 2:
            flags.append("Stopped: too many consecutive errors")

        self.on_stage("done", f"confidence={conf:.2f}")
        return AgentResult(content=content, confidence=conf, verified=verified,
                           route="agentic", tool_results=all_results, flags=flags,
                           plan_summary=plan_summary)

    # --- Helpers ---

    def _infer_tool(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        if any(w in q for w in ("find", "search")) and any(e in q for e in (".py", ".txt", ".md", "*.")):
            return [{"tool": "search_files", "args": {"pattern": "*.txt", "path": "."}}]
        if any(w in q for w in ("list", "directory", "ls")):
            return [{"tool": "list_dir", "args": {"path": "."}}]
        if "read" in q and "file" in q:
            return [{"tool": "read_file", "args": {"path": "."}}]
        return []

    def _fix_shell(self, calls: List[Dict]) -> List[Dict]:
        fixed = []
        for c in calls:
            if c.get("tool") != "run_shell":
                fixed.append(c)
                continue
            cmd = c.get("args", {}).get("command", "").lower()
            if cmd.startswith("find ") or "find " in cmd:
                m = re.search(r"-name\s+[\"']?\*?\.?(\w+)[\"']?", cmd)
                pattern = f"*.{m.group(1)}" if m else "*.txt"
                fixed.append({"tool": "search_files", "args": {"pattern": pattern, "path": "."}})
            elif cmd.startswith("ls "):
                fixed.append({"tool": "list_dir", "args": {"path": "."}})
            elif cmd.startswith("cat "):
                m = re.search(r"cat\s+(\S+)", cmd)
                fixed.append({"tool": "read_file", "args": {"path": m.group(1) if m else "."}})
            else:
                fixed.append(c)
        return fixed
