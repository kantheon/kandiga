"""Interactive agent chat with Rich — tools, skills, memory, macOS."""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()

_LOGO = r"""[bold cyan]
 ██╗  ██╗ █████╗ ███╗   ██╗██████╗ ██╗ ██████╗  █████╗
 ██║ ██╔╝██╔══██╗████╗  ██║██╔══██╗██║██╔════╝ ██╔══██╗
 █████╔╝ ███████║██╔██╗ ██║██║  ██║██║██║  ███╗███████║
 ██╔═██╗ ██╔══██║██║╚██╗██║██║  ██║██║██║   ██║██╔══██║
 ██║  ██╗██║  ██║██║ ╚████║██████╔╝██║╚██████╔╝██║  ██║
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝[/]
[dim]  Agent mode — tools, skills, memory, persistent sessions[/]"""


def run_agent_chat(fast: bool = False, model: Optional[str] = None):
    """Interactive agent chat with tools and persistent KV cache."""
    from kandiga.agents.pipeline import AgentPipeline
    from kandiga.agents.memory import Memory
    from kandiga.agents.skills import SkillEngine

    console.print(_LOGO)
    console.print()

    # Load dual engine (4B fast + 35B brain)
    try:
        from kandiga.agents.dual_engine import DualEngine
        console.print("[dim]Loading dual engine (4B writer + 35B brain)...[/]")
        t0 = time.time()
        engine = DualEngine(fast_mode=fast)
        engine.load()
        load_s = time.time() - t0
        stats = engine.stats
        console.print(
            f"[green]Ready.[/] [dim]({load_s:.1f}s, "
            f"RSS {stats['rss_mb']:.0f}MB, GPU {stats['gpu_active_mb']:.0f}MB)[/]"
        )
        console.print(f"  [dim]4B: {stats.get('struct_model','?').split('/')[-1]} — tool decisions[/]")
        console.print(f"  [dim]35B K={stats.get('k_fast',2)}: ~9-10 tok/s (writing)[/]")
        console.print(f"  [dim]35B K={stats.get('k_precise',4)}: ~5 tok/s (verification)[/]")
    except Exception as e:
        # Fallback to single engine
        from kandiga.engine import KandigaEngine
        console.print(f"[dim]Dual engine failed ({e}), using single 35B...[/]")
        model = model or KandigaEngine.DEFAULT_MODEL
        console.print("[dim]Loading model...[/]")
        t0 = time.time()
        engine = KandigaEngine(model_path=model, fast_mode=fast)
        engine.load()
        load_s = time.time() - t0
        stats = engine.stats
        console.print(
            f"[green]Ready.[/] [dim]({load_s:.1f}s, "
            f"RSS {stats['rss_mb']:.0f}MB, GPU {stats['gpu_active_mb']:.0f}MB)[/]"
        )

    # Set up ALL tools
    from kandiga.agents import full_registry
    registry = full_registry()
    console.print(f"[dim]Tools: {len(registry.tool_names)}[/]")

    # Memory
    memory = Memory()

    # Skills
    skills = SkillEngine()
    skill_count = skills.load_all()
    if skill_count > 0:
        console.print(f"[dim]Skills: {skill_count}[/]")

    # State store + pattern tracker
    from kandiga.agents.state import StateStore
    from kandiga.agents.auto_skills import PatternTracker
    state_store = StateStore()
    pattern_tracker = PatternTracker(skill_engine=skills)
    db_stats = state_store.stats()
    if db_stats["sessions"] > 0:
        console.print(f"[dim]History: {db_stats['sessions']} sessions, {db_stats['messages']} messages[/]")

    # Scheduler
    from kandiga.agents.scheduler import Scheduler, parse_natural_schedule
    scheduler = Scheduler()
    sched_count = scheduler.load_tasks()
    if sched_count > 0:
        console.print(f"[dim]Schedules: {sched_count}[/]")

    # MCP
    try:
        from kandiga.agents.mcp_client import MCPManager
        mcp = MCPManager()
        mcp_count = mcp.load_config()
        if mcp_count > 0:
            mcp_tools = mcp.register_all_tools(registry)
            console.print(f"[dim]MCP: {mcp_count} servers, {mcp_tools} tools[/]")
    except Exception:
        pass

    console.print()
    console.print("  [dim]Commands:[/] /quit /save /load /skills /memory /tools /schedules /history /stats /clear")
    console.print("  [dim]         Say 'every 6h check the news' to create scheduled tasks[/]")
    console.print("  [dim]         Persistent KV cache + SQLite state — everything is saved[/]")
    console.print()

    # Agent loop (Hermes-style: model drives the loop)
    from kandiga.agents.agent_loop import AgentLoop

    def on_stage(stage, detail):
        console.print(f"  [dim][{stage}][/] [dim]{detail}[/]")

    agent = AgentLoop(engine, registry=registry, on_stage=on_stage)

    # Keep old pipeline for compatibility (sessions, state, etc)
    pipeline = AgentPipeline(
        engine, registry=registry, on_stage=on_stage,
        state_store=state_store,
        pattern_tracker=pattern_tracker,
        skill_engine=skills,
    )

    # Wire scheduler to pipeline
    scheduler._run_fn = lambda q: pipeline.run(q)
    scheduler.start()

    # Start persistent KV cache session on both agent loop and pipeline
    agent.start_session()
    pipeline.start_session()
    turn = 0

    while True:
        try:
            user_input = console.input("[bold cyan]\u203a[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        # Commands
        if stripped in ("/quit", "/exit", "/q"):
            pipeline.end_session()
            console.print("[dim]Session ended. Bye.[/]")
            break

        if stripped == "/save":
            import os
            path = os.path.expanduser("~/.kandiga/sessions/last_session.npz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                pipeline.save_session(path)
                console.print(f"  [green]Session saved[/] [dim]({path})[/]")
            except Exception as e:
                console.print(f"  [red]Save failed: {e}[/]")
            continue

        if stripped == "/load":
            import os
            path = os.path.expanduser("~/.kandiga/sessions/last_session.npz")
            try:
                pipeline.load_session(path)
                console.print(f"  [green]Session loaded[/] [dim]({path})[/]")
            except Exception as e:
                console.print(f"  [red]Load failed: {e}[/]")
            continue

        if stripped == "/skills":
            all_skills = skills.list_all()
            if all_skills:
                for s in all_skills:
                    tags = f" [{', '.join(s.tags)}]" if s.tags else ""
                    console.print(f"  [cyan]{s.name}[/]: {s.description}{tags}")
            else:
                console.print("  [dim]No skills loaded. Create one with: /skill create <name>[/]")
            continue

        if stripped == "/memory":
            text = memory.read_memory()
            if text:
                console.print(Panel(text[-1000:], title="Memory", border_style="dim"))
            else:
                console.print("  [dim]No memories saved yet.[/]")
            continue

        if stripped == "/tools":
            console.print(f"  [dim]{registry.describe_tools()}[/]")
            continue

        if stripped == "/stats":
            s = engine.stats
            console.print(
                f"  [dim]RSS: {s['rss_mb']:.0f}MB | GPU: {s['gpu_active_mb']:.0f}MB | "
                f"Mode: {s['mode']} | Turn: {turn}[/]"
            )
            console.print(f"  [dim]Memory: {memory.stats}[/]")
            console.print(f"  [dim]DB: {state_store.stats()}[/]")
            suggestions = pattern_tracker.get_suggestions()
            if suggestions:
                console.print(f"  [dim]Skill suggestions: {len(suggestions)} patterns detected[/]")
            continue

        if stripped == "/history":
            sessions = state_store.list_sessions(limit=10)
            if sessions:
                for sess in sessions:
                    console.print(
                        f"  [cyan]{sess['id']}[/] {sess['started_at'][:16]} "
                        f"— {sess['turn_count']} turns, {sess['model'].split('/')[-1] if sess['model'] else '?'}"
                    )
            else:
                console.print("  [dim]No session history yet.[/]")
            continue

        if stripped.startswith("/search "):
            query_text = stripped[8:].strip()
            results = state_store.search(query_text, limit=5)
            if results:
                for r in results:
                    role = r['role']
                    content = r['content'][:150]
                    console.print(f"  [{r['created_at'][:16]}] [dim]{role}:[/] {content}")
            else:
                console.print(f"  [dim]No results for: {query_text}[/]")
            continue

        if stripped == "/suggest":
            suggestions = pattern_tracker.get_suggestions()
            if suggestions:
                for s in suggestions:
                    console.print(f"  [yellow]Pattern ({s['count']}x):[/] {s['pattern']}")
                    console.print(f"  [dim]Examples: {s['examples'][:2]}[/]")
                console.print(f"  [dim]Use /autoskill to create skills from these patterns[/]")
            else:
                console.print("  [dim]No patterns detected yet. Keep using the agent![/]")
            continue

        if stripped == "/autoskill":
            from kandiga.agents.auto_skills import generate_skill_from_suggestion
            suggestions = pattern_tracker.get_suggestions()
            if not suggestions:
                console.print("  [dim]No patterns to create skills from.[/]")
                continue
            for s in suggestions:
                skill_def = generate_skill_from_suggestion(s, engine=engine)
                path = pattern_tracker.create_skill_from_pattern(
                    s['pattern'], skill_def['name'], skill_def['description'], skill_def['instructions'],
                )
                if path:
                    console.print(f"  [green]Created skill:[/] {skill_def['name']} [dim]({path})[/]")
            skills.load_all()
            continue

        if stripped == "/clear":
            pipeline.end_session()
            pipeline.start_session()
            turn = 0
            console.clear()
            console.print(_LOGO)
            console.print("\n  [dim]Session reset. Fresh start.[/]\n")
            continue

        if stripped == "/schedules" or stripped == "/cron":
            tasks = scheduler.list_tasks()
            if tasks:
                for t in tasks:
                    status = "[green]ON[/]" if t.enabled else "[red]OFF[/]"
                    last = t.last_run[:16] if t.last_run else "never"
                    console.print(f"  {status} [cyan]{t.id}[/] {t.schedule} — {t.name}")
                    console.print(f"       [dim]Last: {last} | Runs: {t.run_count}[/]")
            else:
                console.print("  [dim]No scheduled tasks. Try: 'every 6 hours check the news'[/]")
            continue

        if stripped.startswith("/cancel "):
            task_id = stripped.split()[1]
            if scheduler.remove_task(task_id):
                console.print(f"  [green]Removed task {task_id}[/]")
            else:
                console.print(f"  [red]Task {task_id} not found[/]")
            continue

        if stripped.startswith("/remember "):
            content = stripped[10:].strip()
            memory.add_memory(content, category="user")
            console.print(f"  [green]Remembered.[/]")
            continue

        if stripped.startswith("/"):
            console.print(f"  [dim]Unknown command: {stripped}[/]")
            continue

        # Detect natural language schedule: "every X do Y"
        if stripped.lower().startswith("every "):
            parsed = parse_natural_schedule(stripped)
            if parsed:
                task = scheduler.add_task(**parsed)
                console.print(f"  [green]Scheduled![/] [cyan]{task.id}[/]")
                console.print(f"  [dim]Schedule: {task.schedule}[/]")
                console.print(f"  [dim]Query: {task.query}[/]")
                if task.condition:
                    console.print(f"  [dim]Condition: {task.condition}[/]")
                if task.action_on_match:
                    console.print(f"  [dim]Action: {task.action_on_match}[/]")
                console.print(f"  [dim]Use /schedules to see all, /cancel {task.id} to remove[/]")
                continue

        # Build context with memory
        context = ""
        mem_ctx = memory.build_context(stripped, max_chars=1000)
        if mem_ctx:
            context = f"Agent memory:\n{mem_ctx}"

        # Check for matching skills
        matched_skills = skills.match(stripped)
        if matched_skills:
            skill = matched_skills[0]
            context += f"\n\nActive skill [{skill.name}]:\n{skill.instructions}"
            console.print(f"  [dim]Using skill: {skill.name}[/]")

        # Run agent loop (model drives the loop, like Hermes)
        console.print()
        t0 = time.time()

        try:
            result = agent.run(stripped, context=context)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")
            continue

        elapsed = time.time() - t0
        turn += 1

        # Display response
        try:
            md = Markdown(result.content)
            console.print(md)
        except Exception:
            console.print(result.content)

        # Status line
        parts = [
            f"Turn {turn}",
            f"{elapsed:.1f}s",
            f"route={result.route}",
            f"conf={result.confidence:.2f}",
        ]
        if result.tool_results:
            tools_ok = sum(1 for tr in result.tool_results if tr.success)
            tools_total = len(result.tool_results)
            parts.append(f"tools={tools_ok}/{tools_total}")
        if result.verified:
            parts.append("[green]verified[/]")

        console.print(f"\n[dim]{' · '.join(parts)}[/]\n")

        # Log to daily memory
        memory.log_daily(f"Q: {stripped[:100]}\nA: {result.content[:200]}\nRoute: {result.route} Conf: {result.confidence:.2f}")
