"""Native Qwen3.5 tool-calling agent loop.

Uses the model's ACTUAL chat template with tools= parameter.
The tokenizer formats everything correctly. We just:
1. Pass tool definitions to apply_chat_template(tools=...)
2. Parse <function=name><parameter=key>value</parameter></function> from output
3. Feed results back as <tool_response>...</tool_response>

This is the format the model was TRAINED on. No custom prompts.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from kandiga.agents.protocol import AgentResult, ToolCall, ToolResult
from kandiga.agents.tools import ToolRegistry, default_tools

_HOME = os.path.expanduser("~")


def _build_tool_defs(registry: ToolRegistry) -> List[Dict]:
    """Build tool definitions in Qwen3.5 JSON Schema format."""
    tools = []
    for name in sorted(registry.tool_names):
        spec = registry._tools[name]
        properties = {}
        required = []
        for arg_name, arg_type in spec.args_schema.items():
            properties[arg_name] = {"type": "string", "description": arg_name}
            required.append(arg_name)
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse Qwen3.5 native tool call format.

    Format:
    <tool_call>
    <function=tool_name>
    <parameter=param1>
    value1
    </parameter>
    </function>
    </tool_call>
    """
    match = re.search(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        text, re.DOTALL,
    )
    if not match:
        return None

    name = match.group(1)
    params_block = match.group(2)

    args = {}
    for param_match in re.finditer(
        r'<parameter=(\w+)>\s*(.*?)\s*</parameter>',
        params_block, re.DOTALL,
    ):
        key = param_match.group(1)
        value = param_match.group(2).strip()
        args[key] = value

    return {"name": name, "args": args}


def _strip_thinking(text: str) -> str:
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if '</think>' in text:
        text = text.split('</think>', 1)[-1].strip()
    return text


StageCallback = Callable[[str, str], None]


class LoopGuard:
    """Prevents infinite agent loops — repeated calls, no progress, budget exhaustion."""

    def __init__(self, max_steps: int = 12, max_repeat: int = 4, max_flat: int = 5):
        self.max_steps = max_steps
        self.max_repeat = max_repeat  # same exact call (name+args) repeated this many times
        self.max_flat = max_flat      # consecutive steps with no new information
        self.steps = 0
        self.flat_steps = 0
        self.seen: Dict[str, int] = {}

    def on_step(self) -> Optional[str]:
        self.steps += 1
        if self.steps > self.max_steps:
            return "max_steps_reached"
        return None

    def on_tool_call(self, name: str, args: Dict) -> Optional[str]:
        sig = f"{name}:{json.dumps(args, sort_keys=True)}"
        self.seen[sig] = self.seen.get(sig, 0) + 1
        if self.seen[sig] >= self.max_repeat:
            return f"loop_detected:repeated_{name}"
        return None

    def on_progress(self, has_new: bool) -> Optional[str]:
        self.flat_steps = 0 if has_new else self.flat_steps + 1
        if self.flat_steps >= self.max_flat:
            return "loop_detected:no_progress"
        return None

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.steps)


class AgentLoop:
    """Native Qwen3.5 tool-calling loop using the model's trained format.

    Production-quality features:
    - Stop generation after first </tool_call> (no repeated calls)
    - LoopGuard prevents infinite loops (repeated calls, no progress)
    - Budget warnings injected when iterations are running low
    - Tool results fed back in native format for multi-turn
    - Context compaction when conversation grows too long
    - 35B synthesis for complex results, 4B skip for simple ones
    """

    def __init__(
        self,
        engine,
        registry: Optional[ToolRegistry] = None,
        max_iterations: int = 12,
        on_stage: Optional[StageCallback] = None,
    ):
        self.engine = engine
        self.registry = registry or default_tools()
        self.max_iterations = max_iterations
        self.on_stage = on_stage or (lambda *_: None)
        self._dual = hasattr(engine, 'generate_fast') and hasattr(engine, 'generate_brain')
        self._tool_defs = _build_tool_defs(self.registry)

        # Get the tokenizer
        if self._dual:
            self._tokenizer = engine._struct_tokenizer
        else:
            self._tokenizer = engine._tokenizer

        # Session state
        self._session_active = False
        # Recent files/paths used — injected as context for follow-ups
        self._recent_paths: List[str] = []
        self._last_query = ""
        self._last_response = ""

    def start_session(self):
        """Start persistent KV cache session on the 35B."""
        brain = self.engine.brain if self._dual else self.engine
        if hasattr(brain, 'start_session'):
            brain.start_session()
            self._session_active = True
            self.on_stage("session", "KV cache active — follow-ups are instant")

    def end_session(self):
        brain = self.engine.brain if self._dual else self.engine
        if hasattr(brain, 'end_session'):
            brain.end_session()
            self._session_active = False

    def _format_and_generate(self, messages: List[Dict], max_tokens: int = 150) -> str:
        """Format messages with native tool template and generate.

        Stops generation after first </tool_call> to prevent
        repeated calls and false positives after direct answers.
        """
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                tools=self._tool_defs, enable_thinking=False,
            )
        except TypeError:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        if self._dual:
            from mlx_lm import stream_generate
            from mlx_lm.generate import make_sampler
            # Stream and stop at first </tool_call> or </function>
            result = ""
            for resp in stream_generate(
                self.engine._struct_model, self._tokenizer,
                prompt=formatted, max_tokens=max_tokens,
                sampler=make_sampler(temp=0.0),
            ):
                result += resp.text
                # Stop after first complete tool call
                if "</tool_call>" in result:
                    # Keep everything up to and including </tool_call>
                    idx = result.index("</tool_call>") + len("</tool_call>")
                    result = result[:idx]
                    break
                if resp.finish_reason:
                    break
            return result
        else:
            return self.engine.generate(formatted, max_tokens=max_tokens, temp=0.0, stream=False)

    def _generate_response(self, query: str, tool_summary: str) -> str:
        """Generate response using 35B session KV cache.

        The 35B remembers ALL prior turns via KV cache.
        Turn 50 processes the same amount of new tokens as turn 1.
        """
        message = f"Tool results for '{query}':\n{tool_summary}\n\nSummarize what was done."
        brain = self.engine.brain if self._dual else self.engine

        if self._session_active and hasattr(brain, '_session_cache') and brain._session_cache is not None:
            tokens = []
            for token in brain.session_generate(message, max_tokens=200):
                tokens.append(token)
            return _strip_thinking("".join(tokens))

        # No session — fresh generation
        if self._dual:
            return self.engine.generate_brain(
                "You are Kandiga. Summarize what was done. Be concise.",
                f"Task: {query}\n\nTool results:\n{tool_summary}",
            )
        return tool_summary

    def _verify_and_fix(self, response: str, tool_summary: str) -> str:
        """Quick deterministic fixes. K=4 only if critical issues remain."""
        has_refusal = any(r in response.lower() for r in [
            "i cannot", "i can't", "i'm unable", "i don't have access", "cannot directly",
        ])
        non_english = sum(1 for c in response[:500] if ord(c) > 0x024F and ord(c) < 0x10000) > 10
        lines = [l for l in response.split('\n') if l.strip()]
        has_repetition = len(lines) > 6 and len(set(l.strip() for l in lines)) < len(lines) * 0.4

        if not has_refusal and not non_english and not has_repetition:
            return response

        self.on_stage("fix", "Correcting output...")
        fixed = response

        if non_english:
            fixed = '\n'.join(l for l in fixed.split('\n')
                              if sum(1 for c in l if ord(c) > 0x024F and ord(c) < 0x10000) < 3)

        if has_repetition:
            seen = set()
            deduped = []
            for line in fixed.split('\n'):
                s = line.strip()
                if not s or s not in seen:
                    seen.add(s)
                    deduped.append(line)
            fixed = '\n'.join(deduped)

        if any(r in fixed.lower() for r in ["i cannot", "i can't", "i'm unable"]):
            fixed = '\n'.join(l for l in fixed.split('\n')
                              if not any(r in l.lower() for r in ["i cannot", "i can't", "i'm unable"]))

        if len(fixed.strip()) < 20 and self._dual and hasattr(self.engine, 'generate_check'):
            self.on_stage("fix", "K=4 rewriting...")
            fixed = self.engine.generate_check(
                "Summarize these tool results concisely. English only.",
                f"Tool results:\n{tool_summary}",
            )

        return fixed.strip()

    def run(self, query: str, context: str = "") -> AgentResult:
        """Run the native tool-calling loop with production-quality guardrails."""
        t_start = time.time()
        query = query.replace("~/", f"{_HOME}/").replace("~ ", f"{_HOME} ")
        guard = LoopGuard(max_steps=self.max_iterations)
        flags: List[str] = []

        # Build system prompt
        system_content = (
            f"You are Kandiga, a local AI agent on the user's Mac. "
            f"Home directory: {_HOME}. Always use full absolute paths. "
            f"Use tools when the user's request requires accessing files, running commands, "
            f"searching the web, or sending notifications. "
            f"For general knowledge, math, creative writing, and conversation, respond directly. "
            f"If a tool returns an error, try a different approach. "
            f"When the task is complete, respond with your final answer (no tool_call tags)."
        )
        if context:
            system_content += f"\n\nContext: {context}"
        if self._recent_paths:
            system_content += f"\n\nRecent files used: {', '.join(self._recent_paths[-5:])}"
        if self._last_query and self._last_response:
            system_content += f"\n\nPrevious turn:\nUser: {self._last_query[:200]}\nAgent: {self._last_response[:200]}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        all_results: List[ToolResult] = []

        for iteration in range(1, self.max_iterations + 1):
            # LoopGuard: check step limit
            stop = guard.on_step()
            if stop:
                flags.append(stop)
                break

            self.on_stage("step", f"Step {iteration}/{self.max_iterations}")

            raw = self._format_and_generate(messages)
            raw = _strip_thinking(raw)

            # Parse tool call
            tool_call = _parse_tool_call(raw)

            if not tool_call:
                # No tool call — model is giving final answer
                response = self._build_response(query, raw, all_results)
                return self._finish(query, response, all_results, t_start, flags)

            # Execute tool
            name = tool_call["name"]
            args = tool_call["args"]

            # LoopGuard: check for repeated calls
            dup = guard.on_tool_call(name, args)
            if dup:
                flags.append(dup)
                self.on_stage("guard", f"Loop detected: repeated {name}")
                response = self._build_response(query, "", all_results)
                return self._finish(query, response, all_results, t_start, flags)

            if not self.registry.has_tool(name):
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"<tool_response>\nError: unknown tool '{name}'. Available: {', '.join(sorted(self.registry.tool_names))}\n</tool_response>"})
                continue

            self.on_stage("tool", f"{name}({json.dumps(args)[:60]})")
            tc = ToolCall(tool=name, args=args)
            tr = self.registry.execute(tc)
            all_results.append(tr)

            # LoopGuard: check progress
            has_new = tr.success and tr.output and len(str(tr.output)) > 10
            prog = guard.on_progress(has_new)
            if prog:
                flags.append(prog)
                self.on_stage("guard", "No progress detected")
                response = self._build_response(query, "", all_results)
                return self._finish(query, response, all_results, t_start, flags)

            output = str(tr.output) if tr.output else (tr.error or "")
            if len(output) > 2000:
                lines = output.split('\n')
                if len(lines) > 20:
                    kept = lines[:10] + [f"... ({len(lines) - 20} lines omitted) ..."] + lines[-10:]
                    output = '\n'.join(kept)
                else:
                    output = output[:2000]

            status = "OK" if tr.success else "FAILED"
            self.on_stage("result", f"{name}: {status}")

            # Feed back in native format
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"<tool_response>\n{output}\n</tool_response>"})

            # Budget warning when running low
            if guard.remaining <= 2:
                messages[-1]["content"] += f"\n[BUDGET WARNING: {guard.remaining} iterations remaining. Wrap up.]"

        # Max iterations exhausted
        flags.append("max_iterations")
        response = self._build_response(query, "", all_results)
        return self._finish(query, response, all_results, t_start, flags)

    def _build_response(self, query: str, raw: str, all_results: List[ToolResult]) -> str:
        """Build the final response — skip 35B for simple results."""
        if not all_results:
            # No tools called — clean up tags from direct answer
            response = re.sub(r'</?(?:tool_call|function|parameter)[^>]*>', '', raw).strip()
            return response or raw

        tool_summary = "\n".join(
            f"[{tr.tool}] {'OK' if tr.success else 'FAIL'}: "
            f"{str(tr.output)[:300] if tr.output else tr.error or ''}"
            for tr in all_results
        )

        # Skip 35B for simple results — only use 35B when truly needed
        q_lower = query.lower()
        needs_reasoning = any(w in q_lower for w in [
            "summarize", "analyze", "compare", "explain why",
        ])

        # Skip 35B if all tools succeeded and we don't need deep reasoning
        successful = [tr for tr in all_results if tr.success]
        if successful and not needs_reasoning and len(successful) == len(all_results):
            # Format each tool result directly — no 35B needed
            parts = []
            for tr in all_results:
                output = str(tr.output) if tr.output else ""
                if tr.tool in ("list_dir", "search_files"):
                    parts.append(output)
                elif tr.tool == "read_file":
                    parts.append(output[:2000])
                elif tr.tool == "run_shell":
                    parts.append(output[:2000])
                elif tr.tool == "write_file":
                    parts.append(output or "File written.")
                elif tr.tool == "notify":
                    parts.append(output or "Notification sent.")
                elif tr.tool == "web_search":
                    # Web search still needs 35B to summarize
                    self.on_stage("write", "Summarizing (35B)...")
                    return self._generate_response(query, tool_summary)
                else:
                    parts.append(output or "Done.")
            return "\n".join(parts)

        # Multiple tools or failures — 35B synthesizes
        self.on_stage("write", "Writing response (35B)...")
        return self._generate_response(query, tool_summary)

    def _finish(self, query: str, response: str, all_results: List[ToolResult],
                t_start: float, flags: List[str]) -> AgentResult:
        """Finalize the result with confidence scoring and tracking."""
        duration = int((time.time() - t_start) * 1000)

        if all_results and all(tr.success for tr in all_results):
            conf = 0.90
        elif all_results:
            conf = 0.50
        else:
            conf = 0.85

        # Track for follow-ups
        self._last_query = query
        self._last_response = response[:300]
        for tr in all_results:
            for v in tr.args.values():
                if isinstance(v, str) and '/' in v:
                    if v not in self._recent_paths:
                        self._recent_paths.append(v)

        self.on_stage("done", f"conf={conf:.2f} ({len(all_results)} tools)")
        return AgentResult(
            content=response, confidence=conf, verified=conf >= 0.7,
            route="agent", tool_results=all_results, flags=flags, duration_ms=duration,
        )
