"""Tool registry and filesystem/shell tools — typed results."""

from __future__ import annotations

import glob
import os
import subprocess
import time
from typing import Any, Callable, Dict, List, Set

from kandiga.agents.protocol import ToolCall, ToolResult


class ToolSpec:
    def __init__(self, name: str, description: str, args_schema: Dict[str, str], func: Callable):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func

    def describe(self) -> str:
        args_str = ", ".join(f"{k}: {v}" for k, v in self.args_schema.items())
        return f"- {self.name}({args_str}): {self.description}"


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, args_schema: Dict[str, str], func: Callable):
        self._tools[name] = ToolSpec(name, description, args_schema, func)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_names(self) -> Set[str]:
        return set(self._tools.keys())

    def describe_tools(self) -> str:
        return "\n".join(s.describe() for s in self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        t0 = time.time()
        if call.tool not in self._tools:
            return ToolResult(tool=call.tool, args=call.args, output=None,
                              success=False, error=f"Unknown tool: {call.tool}")
        try:
            result = self._tools[call.tool].func(**call.args)
            ms = int((time.time() - t0) * 1000)
            output_str = str(result) if result is not None else ""
            is_err = output_str.startswith("Error")
            return ToolResult(tool=call.tool, args=call.args, output=result,
                              success=not is_err, error=output_str if is_err else None, duration_ms=ms)
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            return ToolResult(tool=call.tool, args=call.args, output=None,
                              success=False, error=str(e), duration_ms=ms)


# --- Built-in tools ---

def read_file(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(os.path.expanduser("~"), path)
    if os.path.isdir(path):
        # Auto-fix: they meant list_dir, not read_file
        entries = sorted(os.listdir(path))[:50]
        return "This is a directory. Contents:\n" + "\n".join(f"  {e}{'/' if os.path.isdir(os.path.join(path,e)) else ''}" for e in entries)
    if not os.path.isfile(path):
        return f"Error: file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read() or "(empty file)"
    except Exception as e:
        return f"Error reading {path}: {e}"


def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    # Convert relative paths to absolute using home directory
    if not os.path.isabs(path):
        path = os.path.join(os.path.expanduser("~"), path)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            n = f.write(content)
        # Verify the file actually exists
        if os.path.isfile(path):
            return f"Wrote {n} bytes to {path}"
        return f"Error: wrote {n} bytes but file not found at {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def list_dir(path: str = ".") -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path) and path != ".":
        path = os.path.join(os.path.expanduser("~"), path)
    if not os.path.isdir(path):
        return f"Error: not a directory: {path}"
    try:
        entries = sorted(os.listdir(path))
        if not entries:
            return "(empty directory)"
        lines = []
        for e in entries:
            full = os.path.join(path, e)
            lines.append(f"  {e}{'/' if os.path.isdir(full) else ''}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing {path}: {e}"


def search_files(pattern: str, path: str = ".") -> str:
    path = os.path.expanduser(path)
    try:
        matches = glob.glob(os.path.join(path, "**", pattern), recursive=True)
        if not matches:
            return f"No files matching '{pattern}' in {path}"
        return "\n".join(sorted(matches)[:50])
    except Exception as e:
        return f"Error searching: {e}"


def run_shell(command: str, background: bool = False) -> str:
    if not command or not command.strip():
        return "Error: empty command"
    dangerous = ["rm -rf /", "rm -rf ~", "mkfs", ": (){ :|:& };:"]
    for d in dangerous:
        if d in command:
            return "Error: blocked dangerous command"

    home = os.path.expanduser("~")

    # Auto-detect scripts that should run in background (infinite loops, daemons)
    if not background and any(cmd in command for cmd in ["while True", "sleep", "schedule", "daemon"]):
        background = True

    if background or command.strip().endswith("&"):
        # Run in background — don't wait for completion
        cmd = command.rstrip("& ")
        try:
            proc = subprocess.Popen(
                f"nohup {cmd} > /tmp/kandiga_bg_output.log 2>&1 &",
                shell=True, cwd=home,
            )
            return f"Started in background (PID check: ps aux | grep '{cmd[:30]}')\nOutput: /tmp/kandiga_bg_output.log"
        except Exception as e:
            return f"Error starting background process: {e}"

    # Fix common command issues
    command = command.replace("python ", "python3 ").replace("python\n", "python3\n")
    if command.strip() == "python":
        command = "python3"

    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30,
                           cwd=home)
        if r.returncode != 0:
            out = f"Error (exit {r.returncode}): {r.stderr.strip()}"
            if r.stdout:
                out += f"\n{r.stdout.strip()}"
            return out
        output = r.stdout.strip() if r.stdout else "(no output)"
        # Keep output manageable but show enough data
        if len(output) > 3000:
            lines = output.split('\n')
            if len(lines) > 30:
                output = '\n'.join(lines[:15] + [f'... ({len(lines)-30} lines omitted) ...'] + lines[-15:])
        return output
    except subprocess.TimeoutExpired:
        return "Error: timed out after 30s (use background=true for long-running scripts)"
    except Exception as e:
        return f"Error: {e}"


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("read_file", "Read the contents of a file at a given path", {"path": "str"}, read_file)
    reg.register("write_file", "Write or create a file with the given content", {"path": "str", "content": "str"}, write_file)
    reg.register("list_dir", "List files and folders in a directory", {"path": "str"}, list_dir)
    reg.register("search_files", "Search for files matching a glob pattern", {"pattern": "str", "path": "str"}, search_files)
    reg.register("run_shell", "Run any shell command (ls, df, date, rm, find, python3, etc) and return output. Use this for system info, disk space, time, deleting files, running scripts, or any terminal command", {"command": "str"}, run_shell)
    # Web search — try ddgs (new name) then duckduckgo_search (old name)
    _ddgs_cls = None
    try:
        from ddgs import DDGS as _ddgs_cls
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _ddgs_cls
        except ImportError:
            pass

    if _ddgs_cls:
        _DDGS = _ddgs_cls

        def web_search(query: str, max_results: int = 3) -> str:
            """Search the web AND read the top pages. Returns real content."""
            import urllib.request
            import re as _re

            try:
                with _DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return f"No results for: {query}"

                # Read the actual pages — not just snippets
                output_parts = []
                for r in results[:3]:
                    title = r.get('title', '')
                    url = r.get('href', '')
                    snippet = r.get('body', '')

                    page_content = ""
                    if url:
                        try:
                            req = urllib.request.Request(url, headers={
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Kandiga/1.0"
                            })
                            with urllib.request.urlopen(req, timeout=8) as resp:
                                html = resp.read().decode("utf-8", errors="replace")
                            # Strip HTML to text
                            text = _re.sub(r"<script[^>]*>.*?</script>", "", html, flags=_re.DOTALL)
                            text = _re.sub(r"<style[^>]*>.*?</style>", "", text, flags=_re.DOTALL)
                            text = _re.sub(r"<[^>]+>", " ", text)
                            text = _re.sub(r"\s+", " ", text).strip()
                            page_content = text[:3000]
                        except Exception:
                            page_content = snippet

                    output_parts.append(
                        f"--- {title} ---\nSource: {url}\n{page_content or snippet}\n"
                    )

                return "\n\n".join(output_parts)
            except Exception as e:
                return f"Error searching: {e}"

        def read_webpage(url: str) -> str:
            """Fetch a webpage and extract text content."""
            import urllib.request
            import re as _re
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Kandiga/1.0"
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
                text = _re.sub(r"<script[^>]*>.*?</script>", "", html, flags=_re.DOTALL)
                text = _re.sub(r"<style[^>]*>.*?</style>", "", text, flags=_re.DOTALL)
                text = _re.sub(r"<[^>]+>", " ", text)
                text = _re.sub(r"\s+", " ", text).strip()
                return text[:5000] if text else "(empty page)"
            except Exception as e:
                return f"Error reading {url}: {e}"

        reg.register("web_search", "Search the web for current information like weather, news, prices, sports scores, or any real-time data", {"query": "str"}, web_search)
        reg.register("read_webpage", "Fetch and read the text content of a specific URL", {"url": "str"}, read_webpage)
    return reg
