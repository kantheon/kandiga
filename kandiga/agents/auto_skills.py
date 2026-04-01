"""Skill auto-creation — learns from your patterns.

When the agent detects you've done similar tasks N times,
it offers to create a reusable skill. The skill is a SKILL.md
file that teaches the agent the pattern.

This is what makes OpenClaw's ClawHub work — skills that write themselves.
But ours runs locally with no API cost.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional

from kandiga.agents.skills import SkillEngine


PATTERN_FILE = os.path.expanduser("~/.kandiga/patterns.json")
PATTERN_THRESHOLD = 3  # create skill after seeing pattern N times


class PatternTracker:
    """Tracks query patterns and suggests skill creation."""

    def __init__(self, skill_engine: Optional[SkillEngine] = None):
        self.skill_engine = skill_engine
        self._patterns: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if os.path.isfile(PATTERN_FILE):
            try:
                with open(PATTERN_FILE) as f:
                    self._patterns = json.load(f)
            except Exception:
                self._patterns = {}

    def _save(self):
        os.makedirs(os.path.dirname(PATTERN_FILE), exist_ok=True)
        with open(PATTERN_FILE, "w") as f:
            json.dump(self._patterns, f, indent=2)

    def track(self, query: str, route: str, tools_used: List[str],
              success: bool) -> Optional[Dict]:
        """Track a query pattern. Returns skill suggestion if threshold met."""
        # Normalize query to a pattern
        pattern = self._normalize(query)
        if not pattern:
            return None

        key = f"{pattern}|{route}|{','.join(sorted(tools_used))}"

        if key not in self._patterns:
            self._patterns[key] = {
                "pattern": pattern,
                "route": route,
                "tools": tools_used,
                "count": 0,
                "examples": [],
                "last_seen": "",
                "skill_created": False,
            }

        entry = self._patterns[key]
        entry["count"] += 1
        entry["last_seen"] = time.strftime("%Y-%m-%d %H:%M")
        if len(entry["examples"]) < 5:
            entry["examples"].append(query[:200])

        self._save()

        # Suggest skill if threshold met and not already created
        if entry["count"] >= PATTERN_THRESHOLD and not entry["skill_created"]:
            return {
                "pattern": pattern,
                "count": entry["count"],
                "examples": entry["examples"],
                "route": route,
                "tools": tools_used,
            }

        return None

    def create_skill_from_pattern(
        self,
        pattern: str,
        name: str,
        description: str,
        instructions: str,
    ) -> Optional[str]:
        """Create a skill from a detected pattern."""
        if not self.skill_engine:
            return None

        path = self.skill_engine.create_skill(
            name=name,
            description=description,
            instructions=instructions,
            tags=["auto-created"],
            author="kandiga-auto",
        )

        # Mark pattern as having a skill
        for key, entry in self._patterns.items():
            if entry["pattern"] == pattern:
                entry["skill_created"] = True
        self._save()

        return path

    def get_suggestions(self) -> List[Dict]:
        """Get all patterns that could become skills."""
        suggestions = []
        for key, entry in self._patterns.items():
            if entry["count"] >= PATTERN_THRESHOLD and not entry["skill_created"]:
                suggestions.append({
                    "pattern": entry["pattern"],
                    "count": entry["count"],
                    "examples": entry["examples"],
                    "route": entry["route"],
                    "tools": entry["tools"],
                })
        return sorted(suggestions, key=lambda x: -x["count"])

    def _normalize(self, query: str) -> str:
        """Normalize a query into a pattern by removing specifics."""
        q = query.lower().strip()

        # Remove file paths
        q = re.sub(r'/[\w/.-]+', '<PATH>', q)
        # Remove quoted strings
        q = re.sub(r'"[^"]*"', '<TEXT>', q)
        q = re.sub(r"'[^']*'", '<TEXT>', q)
        # Remove numbers
        q = re.sub(r'\b\d+\b', '<N>', q)
        # Remove URLs
        q = re.sub(r'https?://\S+', '<URL>', q)
        # Collapse whitespace
        q = re.sub(r'\s+', ' ', q).strip()

        if len(q) < 5:
            return ""
        return q


def generate_skill_from_suggestion(
    suggestion: Dict,
    engine=None,
) -> Dict[str, str]:
    """Generate a SKILL.md from a pattern suggestion.

    If engine is provided, uses the 35B to write the instructions.
    Otherwise generates a template.
    """
    pattern = suggestion["pattern"]
    examples = suggestion.get("examples", [])
    tools = suggestion.get("tools", [])

    # Generate name from pattern
    name = re.sub(r'[<>]', '', pattern)
    name = re.sub(r'\s+', '-', name.strip())[:30]
    name = name.strip('-') or "auto-skill"

    description = f"Auto-detected pattern ({suggestion['count']} occurrences)"

    if engine and hasattr(engine, 'generate'):
        # Use the model to write good instructions
        prompt = (
            f"Write concise instructions for an AI agent skill.\n"
            f"Pattern: {pattern}\n"
            f"Examples of user queries:\n"
            + "\n".join(f"- {e}" for e in examples[:3])
            + f"\nTools used: {', '.join(tools)}\n\n"
            f"Write 3-5 clear instruction steps. No preamble."
        )
        try:
            instructions = engine.generate(prompt, max_tokens=300, temp=0.0, stream=False)
        except Exception:
            instructions = _template_instructions(pattern, examples, tools)
    else:
        instructions = _template_instructions(pattern, examples, tools)

    return {
        "name": name,
        "description": description,
        "instructions": instructions,
    }


def _template_instructions(pattern: str, examples: List[str], tools: List[str]) -> str:
    lines = [f"This skill handles queries matching: {pattern}\n"]
    if tools:
        lines.append(f"Tools to use: {', '.join(tools)}\n")
    lines.append("Steps:")
    lines.append("1. Identify the specific target from the user's query")
    if "web_search" in tools:
        lines.append("2. Search the web for relevant information")
    if "read_file" in tools:
        lines.append("2. Read the specified file")
    if "write_file" in tools:
        lines.append("3. Generate the requested content")
        lines.append("4. Write to the specified location")
    lines.append(f"\nExample queries:\n" + "\n".join(f"- {e}" for e in examples[:3]))
    return "\n".join(lines)
