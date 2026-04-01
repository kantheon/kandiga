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


class AgentLoop:
    """Native Qwen3.5 tool-calling loop using the model's trained format."""

    def __init__(
        self,
        engine,
        registry: Optional[ToolRegistry] = None,
        max_iterations: int = 10,
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

    def _format_and_generate(self, messages: List[Dict], max_tokens: int = 800) -> str:
        """Format messages with native tool template and generate."""
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                tools=self._tool_defs, enable_thinking=False,
            )
        except TypeError:
            # Fallback without tools parameter
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        if self._dual:
            # Use the 4B model for tool decisions
            from mlx_lm import generate as mlx_generate
            from mlx_lm.generate import make_sampler
            return mlx_generate(
                self.engine._struct_model, self._tokenizer,
                prompt=formatted, max_tokens=max_tokens,
                sampler=make_sampler(temp=0.0), verbose=False,
            )
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
        """Run the native tool-calling loop."""
        t_start = time.time()
        query = query.replace("~/", f"{_HOME}/").replace("~ ", f"{_HOME} ")

        # Build initial messages
        system_content = (
            f"You are Kandiga, a local AI agent on the user's Mac. "
            f"Home directory: {_HOME}. Always use full absolute paths. "
            f"Use tools to complete tasks. Never say 'I cannot'.\n"
            f"Shell tips: use 'ls -lhS' to sort by size, 'head -N' to limit output, "
            f"'grep -r' to search in files, 'wc -l' to count lines, "
            f"'du -sh' for directory sizes, 'python3' not 'python'."
        )
        if context:
            system_content += f"\n\nContext: {context}"

        # Inject recent conversation context so follow-ups work
        if self._recent_paths:
            system_content += f"\n\nRecent files used: {', '.join(self._recent_paths[-5:])}"
        if self._last_query and self._last_response:
            system_content += f"\n\nPrevious turn:\nUser: {self._last_query[:200]}\nAgent: {self._last_response[:200]}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        all_results: List[ToolResult] = []
        seen_calls: set = set()

        for iteration in range(1, self.max_iterations + 1):
            self.on_stage("step", f"Step {iteration}...")

            raw = self._format_and_generate(messages, max_tokens=300)
            raw = _strip_thinking(raw)

            # Parse tool call
            tool_call = _parse_tool_call(raw)

            if not tool_call:
                # No tool call — model is giving final answer
                if all_results:
                    tool_summary = "\n".join(
                        f"[{tr.tool}] {'OK' if tr.success else 'FAIL'}: "
                        f"{str(tr.output)[:300] if tr.output else tr.error or ''}"
                        for tr in all_results
                    )

                    # Speed optimization: skip 35B for simple single-tool results
                    # Use 35B when: multiple tools, failures, or query needs reasoning
                    q_lower = query.lower()
                    needs_reasoning = any(w in q_lower for w in [
                        "total", "sum", "calculate", "most", "least", "biggest", "smallest",
                        "compare", "analyze", "explain", "summarize", "which", "how much",
                        "how many", "average", "best", "worst", "find", "expensive", "cheap",
                    ])

                    if len(all_results) == 1 and all_results[0].success and not needs_reasoning:
                        tr = all_results[0]
                        output = str(tr.output) if tr.output else ""
                        if tr.tool in ("list_dir", "search_files"):
                            response = f"Contents of the directory:\n{output}"
                        elif tr.tool == "read_file":
                            response = f"File contents:\n{output[:2000]}"
                        elif tr.tool == "run_shell":
                            response = f"Command output:\n{output[:2000]}"
                        elif tr.tool == "write_file":
                            response = f"File written successfully. {output}"
                        elif tr.tool == "notify":
                            response = "Notification sent."
                        elif tr.tool == "web_search":
                            # Web search needs 35B to summarize
                            self.on_stage("write", "Summarizing (35B)...")
                            response = self._generate_response(query, tool_summary)
                        else:
                            response = output or "Done."
                    else:
                        # Multiple tools or failures — 35B synthesizes
                        self.on_stage("write", "Writing response (35B)...")
                        response = self._generate_response(query, tool_summary)
                else:
                    # Clean up tags from response
                    response = re.sub(r'</?(?:tool_call|function|parameter)[^>]*>', '', raw).strip()
                    if not response:
                        response = raw

                duration = int((time.time() - t_start) * 1000)
                conf = 0.90 if all_results and all(tr.success for tr in all_results) else (0.50 if all_results else 0.85)

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
                    route="agent", tool_results=all_results, duration_ms=duration,
                )

            # Execute tool
            name = tool_call["name"]
            args = tool_call["args"]

            call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
            if call_key in seen_calls:
                messages.append({"role": "user", "content": "<tool_response>\nAlready executed. Result is above.\n</tool_response>"})
                continue
            seen_calls.add(call_key)

            if not self.registry.has_tool(name):
                messages.append({
                    "role": "assistant",
                    "content": raw,
                })
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\nError: unknown tool '{name}'. Available: {', '.join(sorted(self.registry.tool_names))}\n</tool_response>",
                })
                continue

            self.on_stage("tool", f"{name}({json.dumps(args)[:60]})")
            tc = ToolCall(tool=name, args=args)
            tr = self.registry.execute(tc)
            all_results.append(tr)

            status = "OK" if tr.success else "FAILED"
            output = str(tr.output) if tr.output else (tr.error or "")
            # Smart truncation — keep first and last lines if too long
            if len(output) > 2000:
                lines = output.split('\n')
                if len(lines) > 20:
                    kept = lines[:10] + [f"... ({len(lines) - 20} lines omitted) ..."] + lines[-10:]
                    output = '\n'.join(kept)
                else:
                    output = output[:2000]
            self.on_stage("result", f"{name}: {status}")

            # Feed back in the native format
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"<tool_response>\n{output}\n</tool_response>"})

        # Max iterations
        duration = int((time.time() - t_start) * 1000)
        if all_results:
            tool_summary = "\n".join(
                f"[{tr.tool}] {'OK' if tr.success else 'FAIL'}: "
                f"{str(tr.output)[:300] if tr.output else tr.error or ''}"
                for tr in all_results
            )
            self.on_stage("write", "Writing response...")
            response = self._generate_response(query, tool_summary)
        else:
            response = "Could not complete task within iteration limit."

        return AgentResult(
            content=response, confidence=0.50, route="agent",
            tool_results=all_results, flags=["max iterations"], duration_ms=duration,
        )
