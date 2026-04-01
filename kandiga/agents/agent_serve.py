"""Agent HTTP server — web UI + API endpoints."""

import json
import os
import time
import uuid
from typing import Optional

from pydantic import BaseModel


class QueryBody(BaseModel):
    query: str
    context: str = ""


class MemoryBody(BaseModel):
    content: str
    category: str = "user"


_engine = None
_pipeline = None
_memory = None
_skills = None
_scheduler = None


def create_agent_app(fast: bool = False, model: Optional[str] = None):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="Kandiga Agent", version="0.1.0")

    @app.on_event("startup")
    async def startup():
        global _engine, _pipeline, _memory, _skills

        from kandiga.engine import KandigaEngine
        from kandiga.agents.pipeline import AgentPipeline
        from kandiga.agents import full_registry
        from kandiga.agents.memory import Memory
        from kandiga.agents.skills import SkillEngine
        from kandiga.agents.state import StateStore
        from kandiga.agents.scheduler import Scheduler
        from kandiga.agents.auto_skills import PatternTracker

        # Try dual engine (4B fast + 35B brain)
        try:
            from kandiga.agents.dual_engine import DualEngine
            _engine = DualEngine(fast_mode=fast, log_memory=True)
            _engine.load()
        except Exception:
            model_path = model or KandigaEngine.DEFAULT_MODEL
            _engine = KandigaEngine(model_path=model_path, fast_mode=fast, log_memory=True)
            _engine.load()

        registry = full_registry()

        # Try loading MCP servers
        try:
            from kandiga.agents.mcp_client import MCPManager
            mcp = MCPManager()
            mcp_count = mcp.load_config()
            if mcp_count > 0:
                mcp.register_all_tools(registry)
        except Exception:
            pass

        _memory = Memory()
        _skills = SkillEngine()
        _skills.load_all()

        # State store + pattern tracker
        state_store = StateStore()
        pattern_tracker = PatternTracker(skill_engine=_skills)

        _pipeline = AgentPipeline(
            _engine, registry=registry,
            state_store=state_store,
            pattern_tracker=pattern_tracker,
            skill_engine=_skills,
        )
        _pipeline.start_session()

        global _scheduler
        _scheduler = Scheduler(run_task_fn=lambda q: _pipeline.run(q))
        _scheduler.load_tasks()
        _scheduler.start()

        # Start Telegram gateway if configured
        try:
            from kandiga.agents.messaging import TelegramGateway
            tg_token = os.environ.get("KANDIGA_TELEGRAM_TOKEN")
            if tg_token:
                tg = TelegramGateway(
                    token=tg_token,
                    on_message=lambda text, chat_id: _pipeline.run(text).content,
                )
                tg.start()

                # Wire scheduler delivery
                def deliver(task, result_text):
                    # Send to iMessage if configured, else Telegram
                    from kandiga.agents.messaging import imessage_send
                    try:
                        tg.send(chat_id=int(os.environ.get("KANDIGA_TELEGRAM_CHAT_ID", "0")), text=f"[{task.name}]\n{result_text[:3000]}")
                    except Exception:
                        pass
                _scheduler.set_delivery(deliver)
        except Exception:
            pass

    @app.on_event("shutdown")
    async def shutdown():
        if _pipeline:
            _pipeline.end_session()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = os.path.join(os.path.dirname(__file__), "..", "static", "agent.html")
        if os.path.isfile(html_path):
            with open(html_path) as f:
                return f.read()
        return "<h1>Kandiga Agent</h1><p>UI not found</p>"

    @app.get("/api/status")
    async def status():
        return {
            "status": "ok" if _engine and _engine.is_ready else "loading",
            "model": _engine.model_path if _engine else None,
            "fast_mode": _engine.fast_mode if _engine else False,
            "tools": len(_pipeline.registry.tool_names) if _pipeline else 0,
            "skills": len(_skills.list_all()) if _skills else 0,
            "memory": _memory.stats if _memory else {},
            "session_active": _pipeline._session_active if _pipeline else False,
        }

    @app.post("/api/query")
    async def do_query(data: QueryBody):
        q = data.query
        context = data.context
        if _memory:
            mem_ctx = _memory.build_context(q, max_chars=1000)
            if mem_ctx:
                context = f"Memory:\n{mem_ctx}\n\n{context}" if context else f"Memory:\n{mem_ctx}"

        result = _pipeline.run(q, context=context)

        if _memory:
            _memory.log_daily(f"Q: {q[:100]}\nRoute: {result.route} Conf: {result.confidence:.2f}")

        return JSONResponse(result.to_dict())

    @app.post("/api/stream")
    async def do_stream(data: QueryBody):
        q = data.query

        import asyncio
        import threading
        import queue

        async def generate():
            context = data.context
            if _memory:
                mem_ctx = _memory.build_context(q, max_chars=1000)
                if mem_ctx:
                    context = f"Memory:\n{mem_ctx}\n\n{context}" if context else f"Memory:\n{mem_ctx}"

            event_queue = queue.Queue()
            result_holder = [None]

            def on_stage(stage, detail):
                event_queue.put({"type": "stage", "stage": stage, "detail": detail})

            def on_token(token):
                event_queue.put({"type": "token", "token": token})

            def run_pipeline():
                old_cb = _pipeline.on_stage
                old_tok = _pipeline._on_token
                _pipeline.on_stage = on_stage
                _pipeline._on_token = on_token
                try:
                    result_holder[0] = _pipeline.run(q, context=context)
                finally:
                    _pipeline.on_stage = old_cb
                    _pipeline._on_token = old_tok
                    event_queue.put(None)

            t = threading.Thread(target=run_pipeline, daemon=True)
            t.start()

            while True:
                try:
                    evt = event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if evt is None:
                    break
                yield f"data: {json.dumps(evt)}\n\n"

            t.join(timeout=5)
            if result_holder[0] is not None:
                result = result_holder[0]
                if _memory:
                    _memory.log_daily(f"Q: {q[:100]}\nRoute: {result.route} Conf: {result.confidence:.2f}")
                yield f"data: {json.dumps({'type': 'result', **result.to_dict()})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/api/session/start")
    async def session_start():
        if _pipeline:
            _pipeline.start_session()
        return {"status": "started"}

    @app.post("/api/session/end")
    async def session_end():
        if _pipeline:
            _pipeline.end_session()
        return {"status": "ended"}

    @app.get("/api/tools")
    async def list_tools():
        if not _pipeline:
            return {"tools": []}
        return {"tools": sorted(_pipeline.registry.tool_names)}

    @app.get("/api/skills")
    async def list_skills():
        if not _skills:
            return {"skills": []}
        return {"skills": [s.to_dict() for s in _skills.list_all()]}

    @app.get("/api/memory")
    async def read_memory():
        if not _memory:
            return {"memory": "", "stats": {}}
        return {"memory": _memory.read_memory(), "stats": _memory.stats}

    @app.post("/api/memory")
    async def add_memory(data: MemoryBody):
        if not data.content:
            return JSONResponse({"error": "content required"}, status_code=400)
        if _memory:
            _memory.add_memory(data.content, category=data.category)
        return {"status": "saved"}

    # --- Scheduler ---

    class ScheduleBody(BaseModel):
        name: str = ""
        query: str
        schedule: str  # "every 6h", "every day at 9:00"
        condition: str = ""
        action_on_match: str = ""

    class NaturalScheduleBody(BaseModel):
        text: str  # "every 6 hours check the news and notify me"

    @app.get("/api/schedules")
    async def list_schedules():
        if not _scheduler:
            return {"schedules": []}
        return {"schedules": [t.to_dict() for t in _scheduler.list_tasks()]}

    @app.post("/api/schedules")
    async def create_schedule(data: ScheduleBody):
        if not _scheduler:
            return JSONResponse({"error": "scheduler not running"}, status_code=500)
        task = _scheduler.add_task(
            name=data.name or data.query[:50],
            query=data.query,
            schedule=data.schedule,
            condition=data.condition,
            action_on_match=data.action_on_match,
        )
        return {"status": "created", "task": task.to_dict()}

    @app.post("/api/schedules/natural")
    async def create_natural_schedule(data: NaturalScheduleBody):
        if not _scheduler:
            return JSONResponse({"error": "scheduler not running"}, status_code=500)
        from kandiga.agents.scheduler import parse_natural_schedule
        parsed = parse_natural_schedule(data.text)
        if not parsed:
            return JSONResponse({"error": f"Could not parse schedule: {data.text}"}, status_code=400)
        task = _scheduler.add_task(**parsed)
        return {"status": "created", "task": task.to_dict()}

    @app.delete("/api/schedules/{task_id}")
    async def delete_schedule(task_id: str):
        if not _scheduler:
            return JSONResponse({"error": "scheduler not running"}, status_code=500)
        if _scheduler.remove_task(task_id):
            return {"status": "deleted"}
        return JSONResponse({"error": "task not found"}, status_code=404)

    return app


def run_agent_serve(port: int = 3000, fast: bool = False, model: Optional[str] = None):
    import uvicorn
    app = create_agent_app(fast=fast, model=model)
    print(f"Kandiga Agent — http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
