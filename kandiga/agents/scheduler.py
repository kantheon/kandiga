"""Task scheduler — cron jobs for the agent.

"Every 6 hours check the news and notify me"
"Every morning at 9am summarize my calendar"
"Every hour check if /tmp/deploy.log has errors and alert me"

Tasks persist in ~/.kandiga/schedules/ as JSON. A background thread
runs them on schedule. Results are stored and can trigger actions.

Schedule formats:
  "every 6h"          — every 6 hours
  "every 30m"         — every 30 minutes
  "every day at 9:00" — daily at 9:00 AM
  "every monday at 8:00" — weekly
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional


SCHEDULES_DIR = os.path.expanduser("~/.kandiga/schedules")


@dataclass
class ScheduledTask:
    id: str
    name: str
    query: str
    schedule: str  # "every 6h", "every day at 9:00", etc.
    condition: str = ""  # optional: "if errors" / "if changed"
    action_on_match: str = ""  # what to do when condition matches
    enabled: bool = True
    created_at: str = ""
    last_run: str = ""
    last_result: str = ""
    last_success: bool = True
    run_count: int = 0
    _next_run: float = 0.0  # internal: epoch timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "query": self.query,
            "schedule": self.schedule,
            "condition": self.condition,
            "action_on_match": self.action_on_match,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "last_result": self.last_result[:500],
            "last_success": self.last_success,
            "run_count": self.run_count,
        }


def parse_schedule(schedule: str) -> Optional[float]:
    """Parse a schedule string into interval in seconds.

    Returns None for time-of-day schedules (handled separately).
    """
    s = schedule.lower().strip()

    # "every Xm" or "every X minutes"
    m = re.match(r"every\s+(\d+)\s*m(?:in(?:ute)?s?)?$", s)
    if m:
        return int(m.group(1)) * 60

    # "every Xh" or "every X hours"
    m = re.match(r"every\s+(\d+)\s*h(?:ours?)?$", s)
    if m:
        return int(m.group(1)) * 3600

    # "every Xd" or "every X days"
    m = re.match(r"every\s+(\d+)\s*d(?:ays?)?$", s)
    if m:
        return int(m.group(1)) * 86400

    # "every day at HH:MM"
    m = re.match(r"every\s+day\s+at\s+(\d{1,2}):(\d{2})$", s)
    if m:
        return -1  # special: daily at time

    # "every hour"
    if s in ("every hour", "hourly"):
        return 3600

    # "every day" / "daily"
    if s in ("every day", "daily"):
        return 86400

    return None


def next_run_time(task: ScheduledTask) -> float:
    """Calculate the next run time for a task."""
    interval = parse_schedule(task.schedule)

    if interval is None:
        return time.time() + 3600  # fallback: 1 hour

    if interval == -1:
        # Daily at specific time
        m = re.match(r"every\s+day\s+at\s+(\d{1,2}):(\d{2})", task.schedule.lower())
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target.timestamp()
        return time.time() + 86400

    if interval > 0:
        if task.last_run:
            try:
                last = datetime.fromisoformat(task.last_run).timestamp()
                return last + interval
            except (ValueError, TypeError):
                pass
        return time.time() + interval

    return time.time() + 3600


class Scheduler:
    """Background task scheduler."""

    def __init__(self, run_task_fn: Optional[Callable] = None):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._run_fn = run_task_fn  # fn(query: str) -> AgentResult
        self._thread: Optional[threading.Thread] = None
        self._running = False
        os.makedirs(SCHEDULES_DIR, exist_ok=True)

    def load_tasks(self) -> int:
        """Load all saved tasks from disk."""
        count = 0
        if not os.path.isdir(SCHEDULES_DIR):
            return 0
        for fname in os.listdir(SCHEDULES_DIR):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(SCHEDULES_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                task = ScheduledTask(**{k: v for k, v in data.items() if not k.startswith('_')})
                task._next_run = next_run_time(task)
                self._tasks[task.id] = task
                count += 1
            except Exception:
                pass
        return count

    def add_task(
        self,
        name: str,
        query: str,
        schedule: str,
        condition: str = "",
        action_on_match: str = "",
    ) -> ScheduledTask:
        """Create and save a new scheduled task."""
        task = ScheduledTask(
            id=uuid.uuid4().hex[:8],
            name=name,
            query=query,
            schedule=schedule,
            condition=condition,
            action_on_match=action_on_match,
            created_at=datetime.now().isoformat(),
        )
        task._next_run = next_run_time(task)
        self._tasks[task.id] = task
        self._save_task(task)
        return task

    def remove_task(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            path = os.path.join(SCHEDULES_DIR, f"{task_id}.json")
            if os.path.isfile(path):
                os.remove(path)
            return True
        return False

    def list_tasks(self) -> List[ScheduledTask]:
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        return self._tasks.get(task_id)

    def _save_task(self, task: ScheduledTask) -> None:
        path = os.path.join(SCHEDULES_DIR, f"{task.id}.json")
        with open(path, "w") as f:
            json.dump(task.to_dict(), f, indent=2)

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        """Main scheduler loop — checks every 30 seconds."""
        while self._running:
            now = time.time()
            for task in list(self._tasks.values()):
                if not task.enabled:
                    continue
                if now >= task._next_run:
                    self._execute_task(task)
            time.sleep(30)

    def set_delivery(self, deliver_fn: Optional[Callable] = None):
        """Set a delivery function: fn(task, result_text) -> None."""
        self._deliver_fn = deliver_fn

    def _execute_task(self, task: ScheduledTask) -> None:
        """Run a scheduled task."""
        try:
            if self._run_fn:
                result = self._run_fn(task.query)
                result_text = result.content if hasattr(result, 'content') else str(result)

                # Check condition
                if task.condition:
                    condition_met = self._check_condition(task.condition, result_text)
                    if not condition_met:
                        # Condition not met — skip action but still log
                        task.last_run = datetime.now().isoformat()
                        task.last_result = f"[condition not met] {result_text[:200]}"
                        task.last_success = True
                        task.run_count += 1
                        task._next_run = next_run_time(task)
                        self._save_task(task)
                        return

                    # Condition met — run the action
                    if task.action_on_match:
                        action_result = self._run_fn(task.action_on_match)
                        result_text += f"\n[action: {task.action_on_match}]\n{action_result.content if hasattr(action_result, 'content') else str(action_result)}"

                task.last_run = datetime.now().isoformat()
                task.last_result = result_text[:1000]
                task.last_success = True
                task.run_count += 1
            else:
                task.last_run = datetime.now().isoformat()
                task.last_result = "No runner configured"
                task.last_success = False

        except Exception as e:
            task.last_run = datetime.now().isoformat()
            task.last_result = f"Error: {e}"
            task.last_success = False

        task._next_run = next_run_time(task)
        self._save_task(task)

    def _check_condition(self, condition: str, result: str) -> bool:
        """Check if a condition is met against the result."""
        c = condition.lower()
        r = result.lower()

        # "if errors" / "if error"
        if "error" in c:
            return "error" in r

        # "if changed"
        if "changed" in c or "different" in c:
            return True  # always true for now (need diff tracking)

        # "if contains X"
        m = re.search(r"if\s+contains?\s+['\"]?(.+?)['\"]?\s*$", c)
        if m:
            return m.group(1).lower() in r

        # "if not empty"
        if "not empty" in c or "notempty" in c:
            return len(r.strip()) > 0

        # "if > N" (numeric check)
        m = re.search(r"if\s*>\s*(\d+)", c)
        if m:
            numbers = re.findall(r"\d+", r)
            if numbers:
                return any(int(n) > int(m.group(1)) for n in numbers)

        return True  # default: condition met


def parse_natural_schedule(text: str) -> Optional[Dict[str, str]]:
    """Parse natural language into a scheduled task definition.

    Examples:
        "every 6 hours check the news and notify me"
        "every morning at 9am summarize my calendar"
        "every hour check /tmp/deploy.log for errors and alert me"
    """
    t = text.lower().strip()

    # Extract schedule
    schedule = ""
    m = re.match(r"(every\s+\d+\s*(?:h(?:ours?)?|m(?:in(?:ute)?s?)?|d(?:ays?)?))", t)
    if m:
        schedule = m.group(1)
    m = re.match(r"(every\s+(?:day|morning|evening|night|hour|monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+at\s+\d{1,2}(?::\d{2})?(?:\s*(?:am|pm))?)?)", t)
    if m:
        sched = m.group(1)
        if "morning" in sched:
            schedule = "every day at 9:00"
        elif "evening" in sched:
            schedule = "every day at 18:00"
        elif "night" in sched:
            schedule = "every day at 22:00"
        else:
            schedule = sched
    if "hourly" in t:
        schedule = "every 1h"

    if not schedule:
        return None

    # Extract query (everything after the schedule, before "and")
    rest = t[len(schedule):].strip()

    # Look for conditional
    condition = ""
    action = ""
    if " and " in rest:
        parts = rest.split(" and ", 1)
        query_part = parts[0].strip()
        action_part = parts[1].strip()

        # Check for "if" condition
        if "if " in query_part:
            idx = query_part.index("if ")
            condition = query_part[idx:].strip()
            query_part = query_part[:idx].strip()

        action = action_part
    else:
        query_part = rest

    # Clean up query
    query = query_part.strip()
    if query.startswith("check ") or query.startswith("run "):
        pass  # keep as-is
    elif not query:
        query = text  # fallback to full text

    return {
        "schedule": schedule,
        "query": query,
        "condition": condition,
        "action_on_match": action,
        "name": query[:50],
    }
