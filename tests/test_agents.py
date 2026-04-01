"""Tests for the full agent layer — pipeline, skills, memory, tools."""

import json
import os
import tempfile
import pytest

from kandiga.agents.protocol import AgentResult, ToolCall, ToolResult
from kandiga.agents.json_repair import parse_json, validate_tool_calls, validate_plan, extract_write_file
from kandiga.agents.tools import ToolRegistry, default_tools, read_file, write_file, list_dir, run_shell
from kandiga.agents.pipeline import AgentPipeline, _needs_tools, _needs_multi_step, _verify
from kandiga.agents.skills import SkillEngine, parse_skill_md
from kandiga.agents.memory import Memory


# --- Fixtures ---

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def tmp_file(tmp_dir):
    path = os.path.join(tmp_dir, "test.txt")
    with open(path, "w") as f:
        f.write("Hello, World!\nLine 2\nLine 3\n")
    return path

@pytest.fixture
def registry():
    return default_tools()


# --- Mock engine ---

class _MockEngine:
    def __init__(self, responses=None):
        self._responses = responses or {}
        self._tokenizer = _MockTokenizer()
        self._ready = True

    def generate(self, prompt, max_tokens=2048, temp=0.0, stream=False):
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return "This is a helpful response."

    def generate_fast(self, system, user, max_tokens=1200, temp=0.0):
        prompt = system + user
        # Route classification
        if "TOOL, AGENTIC" in prompt and "DIRECT" in prompt:
            u = user.lower()
            if any(w in u for w in ("explain", "analyze", "debug", "compare", "plan")):
                return "THINK"
            if any(w in u for w in ("read", "write", "list", "file", "run", "search", "create")):
                return "TOOL"
            return "DIRECT"
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return '{"tool_calls": [], "reasoning": "none"}'

    def generate_brain(self, system, user, max_tokens=2048, temp=0.0):
        prompt = system + user
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return "This is a thoughtful response from the brain."

    @property
    def is_ready(self):
        return self._ready


class _MockTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)


# ===========================================================================
# Protocol
# ===========================================================================

class TestProtocol:
    def test_tool_call_immutable(self):
        tc = ToolCall(tool="read_file", args={"path": "x"})
        with pytest.raises(AttributeError):
            tc.tool = "other"

    def test_tool_result_serializable(self):
        tr = ToolResult(tool="x", args={}, output="ok", success=True, duration_ms=5)
        d = tr.to_dict()
        assert json.loads(json.dumps(d))

    def test_agent_result_serializable(self):
        ar = AgentResult(content="Done", confidence=0.9, verified=True, route="tool",
                         tool_results=[ToolResult(tool="a", args={}, output="ok", success=True)])
        d = ar.to_dict()
        s = json.dumps(d)
        assert json.loads(s)["confidence"] == 0.9

    def test_all_tools_succeeded(self):
        ar = AgentResult(content="x", tool_results=[
            ToolResult(tool="a", args={}, output="ok", success=True),
            ToolResult(tool="b", args={}, output="ok", success=True),
        ])
        assert ar.all_tools_succeeded is True

        ar2 = AgentResult(content="x", tool_results=[
            ToolResult(tool="a", args={}, output="ok", success=True),
            ToolResult(tool="b", args={}, output=None, success=False),
        ])
        assert ar2.all_tools_succeeded is False


# ===========================================================================
# JSON Repair
# ===========================================================================

class TestJsonRepair:
    def test_valid(self):
        assert parse_json('{"a": 1}', {"a": 0})["a"] == 1

    def test_preamble(self):
        r = parse_json('Sure!\n{"tool_calls": [{"tool": "read_file"}]}', {"tool_calls": []})
        assert len(r.get("tool_calls", [])) == 1

    def test_truncated(self):
        r = parse_json('{"tool_calls": [{"tool": "read_file", "args": {"path": "/tmp/x"', {"tool_calls": []})
        assert isinstance(r, dict)

    def test_garbage(self):
        assert parse_json("not json!", {"d": True}) == {"d": True}

    def test_empty(self):
        assert parse_json("", {"d": 1}) == {"d": 1}

    def test_never_crashes(self):
        full = json.dumps({"tool_calls": [{"tool": "write_file", "args": {"path": "/x", "content": "hi"}}]})
        for i in range(1, len(full)):
            r = parse_json(full[:i], {"tool_calls": []})
            assert isinstance(r, dict)

    def test_validate_tool_calls(self):
        parsed = {"tool_calls": [{"tool": "read_file", "args": {}}, "bad", {"tool": "unknown"}]}
        valid = validate_tool_calls(parsed, {"read_file"})
        assert len(valid) == 1

    def test_validate_plan(self):
        parsed = {"plan": [
            {"step": 1, "action": "read_file", "description": "Read", "depends_on": []},
            {"step": 2, "action": "write_file", "description": "Write", "depends_on": [1]},
        ]}
        assert len(validate_plan(parsed)) == 2


# ===========================================================================
# Tools
# ===========================================================================

class TestTools:
    def test_read_write(self, tmp_dir):
        path = os.path.join(tmp_dir, "out.txt")
        assert "Wrote" in write_file(path, "content")
        assert read_file(path) == "content"

    def test_read_nonexistent(self):
        assert read_file("/nonexistent").startswith("Error")

    def test_list_dir(self, tmp_dir):
        write_file(os.path.join(tmp_dir, "a.txt"), "a")
        assert "a.txt" in list_dir(tmp_dir)

    def test_run_shell(self):
        assert "hello" in run_shell("echo hello")

    def test_shell_blocked(self):
        assert "Error" in run_shell("rm -rf /")

    def test_registry_execute(self, registry, tmp_file):
        tr = registry.execute(ToolCall(tool="read_file", args={"path": tmp_file}))
        assert tr.success and "Hello" in str(tr.output)

    def test_registry_unknown(self, registry):
        tr = registry.execute(ToolCall(tool="fake", args={}))
        assert not tr.success

    def test_batch_50(self, tmp_dir, registry):
        for i in range(50):
            tr = registry.execute(ToolCall(tool="write_file", args={"path": os.path.join(tmp_dir, f"f{i}.txt"), "content": str(i)}))
            assert tr.success
        assert len(os.listdir(tmp_dir)) == 50


# ===========================================================================
# Routing
# ===========================================================================

class TestRouting:
    def test_needs_tools(self):
        assert _needs_tools("read the file") is True
        assert _needs_tools("list directory") is True
        assert _needs_tools("hello") is False
        assert _needs_tools("why is the sky blue") is False
        assert _needs_tools("schedule a meeting") is True

    def test_needs_multi_step(self):
        assert _needs_multi_step("read file then fix spelling") is True
        assert _needs_multi_step("create poem and save as poem.txt") is True
        assert _needs_multi_step("read config.yaml") is False


# ===========================================================================
# Verification
# ===========================================================================

class TestVerification:
    def test_error(self):
        tr = ToolResult(tool="read_file", args={}, output=None, success=False, error="not found")
        c, v, f = _verify("response", "Error: not found", [tr])
        assert c <= 0.3

    def test_refusal(self):
        c, v, f = _verify("I cannot help", "ctx", [])
        assert c <= 0.25

    def test_false_claim(self):
        c, v, f = _verify("The file has been saved", "no write", [])
        assert c <= 0.4

    def test_success_boost(self):
        tr = ToolResult(tool="read_file", args={}, output="data", success=True)
        c, v, f = _verify("Here is the data", "ctx", [tr])
        assert c >= 0.85

    def test_write_boost(self):
        tr = ToolResult(tool="write_file", args={}, output="Wrote 10 bytes", success=True)
        c, v, f = _verify("Written", "ctx", [tr])
        assert c >= 0.9


# ===========================================================================
# Pipeline
# ===========================================================================

class TestPipelineDirect:
    def test_simple(self):
        engine = _MockEngine()
        pipe = AgentPipeline(engine)
        r = pipe.run("what is 2+2?")
        assert r.route == "direct"
        assert len(r.content) > 0

    def test_serializable(self):
        engine = _MockEngine()
        pipe = AgentPipeline(engine)
        r = pipe.run("hello")
        assert json.loads(json.dumps(r.to_dict()))


class TestPipelineTool:
    def test_tool_call(self, tmp_file):
        tool_json = json.dumps({"tool_calls": [{"tool": "read_file", "args": {"path": tmp_file}}]})
        engine = _MockEngine(responses={"TOOL_NAME": tool_json, "Task:": tool_json})
        pipe = AgentPipeline(engine)
        r = pipe.run(f"read the file {tmp_file}")
        assert r.route == "tool"
        assert any(tr.success for tr in r.tool_results)

    def test_list_dir(self, tmp_dir):
        write_file(os.path.join(tmp_dir, "x.txt"), "x")
        tool_json = json.dumps({"tool_calls": [{"tool": "list_dir", "args": {"path": tmp_dir}}]})
        engine = _MockEngine(responses={"TOOL_NAME": tool_json, "Task:": tool_json})
        pipe = AgentPipeline(engine)
        r = pipe.run(f"list the directory {tmp_dir}")
        assert r.route == "tool"
        assert any(tr.success for tr in r.tool_results)


class TestPipelineDual:
    """Test dual-model pipeline where fast/brain are separate."""

    def test_dual_tool(self, tmp_file):
        tool_json = json.dumps({"tool_calls": [{"tool": "read_file", "args": {"path": tmp_file}}]})
        engine = _MockEngine(responses={"TOOL_NAME": tool_json, "Task:": tool_json})
        pipe = AgentPipeline(engine)
        assert pipe._dual is True  # has generate_fast + generate_brain
        r = pipe.run(f"read file {tmp_file}")
        assert r.route == "tool"

    def test_dual_think(self):
        engine = _MockEngine()
        pipe = AgentPipeline(engine)
        r = pipe.run("explain quantum computing")
        assert r.route in ("think", "direct")
        assert len(r.content) > 0


class TestPipelineStress:
    def test_50_queries(self):
        engine = _MockEngine()
        pipe = AgentPipeline(engine)
        for i in range(50):
            r = pipe.run(f"query {i}")
            assert isinstance(r, AgentResult)

    def test_garbage_response(self):
        engine = _MockEngine(responses={"TOOL_NAME": "NOT JSON <html>garbage</html>"})
        pipe = AgentPipeline(engine)
        r = pipe.run("read some file.txt")
        assert isinstance(r, AgentResult)

    def test_empty_response(self):
        engine = _MockEngine(responses={"TOOL_NAME": ""})
        pipe = AgentPipeline(engine)
        r = pipe.run("list the files")
        assert isinstance(r, AgentResult)


# ===========================================================================
# Skills
# ===========================================================================

class TestSkills:
    def test_parse_skill(self):
        content = """---
name: test-skill
description: A test skill
version: 1.0.0
tags: [test, example]
---

Do something useful here.
"""
        skill = parse_skill_md(content)
        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert "test" in skill.tags
        assert "Do something useful" in skill.instructions

    def test_parse_invalid(self):
        assert parse_skill_md("no frontmatter here") is None
        assert parse_skill_md("---\nno_name: true\n---\nbody") is None

    def test_create_and_load(self, tmp_dir):
        engine = SkillEngine(extra_dirs=[tmp_dir])

        # Create a skill
        skill_dir = os.path.join(tmp_dir, "greet")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write("---\nname: greet\ndescription: Greet the user\nversion: 1.0.0\ntags: [social]\n---\n\nSay hello warmly.\n")

        count = engine.load_all()
        assert count >= 1
        skill = engine.get("greet")
        assert skill is not None
        assert skill.description == "Greet the user"

    def test_search(self, tmp_dir):
        engine = SkillEngine(extra_dirs=[tmp_dir])

        for name, desc in [("calendar-add", "Create calendar events"), ("file-reader", "Read files")]:
            d = os.path.join(tmp_dir, name)
            os.makedirs(d)
            with open(os.path.join(d, "SKILL.md"), "w") as f:
                f.write(f"---\nname: {name}\ndescription: {desc}\nversion: 1.0.0\ntags: []\n---\n\nInstructions.\n")

        engine.load_all()
        matches = engine.match("calendar event")
        assert len(matches) >= 1
        assert matches[0].name == "calendar-add"

    def test_create_skill(self, tmp_dir):
        engine = SkillEngine(extra_dirs=[tmp_dir])
        # Override SKILLS_DIR for test
        import kandiga.agents.skills as skills_mod
        old_dir = skills_mod.SKILLS_DIR
        skills_mod.SKILLS_DIR = tmp_dir
        try:
            path = engine.create_skill("my-skill", "Does cool stuff", "Step 1: be cool", tags=["cool"])
            assert os.path.isfile(path)
            with open(path) as f:
                content = f.read()
            assert "my-skill" in content
            assert "Does cool stuff" in content
        finally:
            skills_mod.SKILLS_DIR = old_dir


# ===========================================================================
# Memory
# ===========================================================================

class TestMemory:
    def test_add_and_read(self, tmp_dir):
        mem = Memory(memory_dir=tmp_dir)
        mem.add_memory("User prefers dark mode", category="preference")
        text = mem.read_memory()
        assert "dark mode" in text
        assert "preference" in text

    def test_search(self, tmp_dir):
        mem = Memory(memory_dir=tmp_dir)
        mem.add_memory("User likes Python and FastAPI", category="tech")
        mem.add_memory("User has a cat named Whiskers", category="personal")
        results = mem.search_memory("Python programming")
        assert len(results) >= 1
        assert any("Python" in r for r in results)

    def test_daily_notes(self, tmp_dir):
        mem = Memory(memory_dir=tmp_dir)
        mem.log_daily("Had a meeting about the project")
        mem.log_daily("Reviewed PR #42")
        text = mem.read_daily()
        assert "meeting" in text
        assert "PR #42" in text

    def test_build_context(self, tmp_dir):
        mem = Memory(memory_dir=tmp_dir)
        mem.add_memory("User works on AI agents", category="work")
        mem.log_daily("Discussed Kandiga architecture")
        ctx = mem.build_context("AI agent project")
        assert len(ctx) > 0

    def test_stats(self, tmp_dir):
        mem = Memory(memory_dir=tmp_dir)
        mem.add_memory("test")
        mem.log_daily("note")
        s = mem.stats
        assert s["memory_bytes"] > 0
        assert s["daily_notes"] >= 1

    def test_clear(self, tmp_dir):
        mem = Memory(memory_dir=tmp_dir)
        mem.add_memory("to be deleted")
        assert "deleted" in mem.read_memory()
        mem.clear_memory()
        assert mem.read_memory() == ""

    def test_list_daily_notes(self, tmp_dir):
        mem = Memory(memory_dir=tmp_dir)
        mem.log_daily("today's note")
        notes = mem.list_daily_notes()
        assert len(notes) >= 1
