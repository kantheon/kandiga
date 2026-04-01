"""MCP client — connect to any MCP server and use its tools.

Extends the agent's capabilities with the entire MCP ecosystem.
Any MCP server (filesystem, database, GitHub, Slack, etc.) can be
added and its tools become available to the agent.

Config: ~/.kandiga/mcp.json
{
    "servers": {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": "..."}
        }
    }
}
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from kandiga.agents.tools import ToolRegistry
from kandiga.agents.protocol import ToolCall, ToolResult


MCP_CONFIG = os.path.expanduser("~/.kandiga/mcp.json")


class MCPConnection:
    """Connection to a single MCP server via stdio."""

    def __init__(self, name: str, command: str, args: List[str],
                 env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self._process: Optional[subprocess.Popen] = None
        self._tools: List[Dict] = []
        self._request_id = 0

    def connect(self) -> bool:
        """Start the MCP server process."""
        try:
            full_env = dict(os.environ)
            full_env.update(self.env)
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=full_env,
                text=True,
            )
            # Initialize
            resp = self._send({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05",
                           "capabilities": {},
                           "clientInfo": {"name": "kandiga", "version": "0.1.0"}},
            })
            if not resp or "error" in resp:
                return False

            # Send initialized notification
            self._send_notification({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            })

            # List tools
            resp = self._send({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            })
            if resp and "result" in resp:
                self._tools = resp["result"].get("tools", [])

            return True
        except Exception:
            return False

    def disconnect(self):
        if self._process:
            self._process.terminate()
            self._process = None

    @property
    def tools(self) -> List[Dict]:
        return self._tools

    def call_tool(self, tool_name: str, arguments: Dict) -> str:
        """Call a tool on the MCP server."""
        resp = self._send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        if not resp:
            return "Error: no response from MCP server"
        if "error" in resp:
            return f"Error: {resp['error'].get('message', 'unknown')}"
        result = resp.get("result", {})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else json.dumps(result)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, message: Dict) -> Optional[Dict]:
        if not self._process or not self._process.stdin:
            return None
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if line:
                return json.loads(line.strip())
        except Exception:
            pass
        return None

    def _send_notification(self, message: Dict):
        if not self._process or not self._process.stdin:
            return
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except Exception:
            pass


class MCPManager:
    """Manages multiple MCP server connections."""

    def __init__(self, config_path: str = MCP_CONFIG):
        self.config_path = config_path
        self._connections: Dict[str, MCPConnection] = {}

    def load_config(self) -> int:
        """Load MCP config and connect to servers."""
        if not os.path.isfile(self.config_path):
            return 0

        try:
            with open(self.config_path) as f:
                config = json.load(f)
        except Exception:
            return 0

        count = 0
        for name, server in config.get("servers", {}).items():
            conn = MCPConnection(
                name=name,
                command=server.get("command", ""),
                args=server.get("args", []),
                env=server.get("env", {}),
            )
            if conn.connect():
                self._connections[name] = conn
                count += 1

        return count

    def all_tools(self) -> List[Dict]:
        """Get all tools from all connected MCP servers."""
        tools = []
        for name, conn in self._connections.items():
            for tool in conn.tools:
                tool["_mcp_server"] = name
                tools.append(tool)
        return tools

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> str:
        conn = self._connections.get(server_name)
        if not conn:
            return f"Error: MCP server '{server_name}' not connected"
        return conn.call_tool(tool_name, arguments)

    def register_all_tools(self, registry: ToolRegistry) -> int:
        """Register all MCP tools into the agent's tool registry."""
        count = 0
        for name, conn in self._connections.items():
            for tool in conn.tools:
                tool_name = f"mcp_{name}_{tool['name']}"
                desc = tool.get("description", "MCP tool")
                schema = {}
                input_schema = tool.get("inputSchema", {})
                for prop, info in input_schema.get("properties", {}).items():
                    schema[prop] = info.get("type", "str")

                # Create closure for this specific tool
                _server = name
                _tool = tool["name"]

                def make_caller(s, t):
                    def caller(**kwargs):
                        return self.call_tool(s, t, kwargs)
                    return caller

                registry.register(tool_name, f"[MCP:{name}] {desc}", schema, make_caller(_server, _tool))
                count += 1

        return count

    def disconnect_all(self):
        for conn in self._connections.values():
            conn.disconnect()
        self._connections.clear()
