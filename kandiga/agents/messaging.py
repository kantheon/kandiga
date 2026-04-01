"""Messaging gateway — iMessage + Telegram.

iMessage: native macOS via AppleScript (zero dependencies)
Telegram: via python-telegram-bot (optional)

The agent can receive messages, process them through the pipeline,
and respond — on your phone, while you're away from the computer.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import json
from typing import Callable, Optional

from kandiga.agents.tools import ToolRegistry


# ===========================================================================
# iMessage (native macOS, no dependencies)
# ===========================================================================

def imessage_send(to: str, message: str) -> str:
    """Send an iMessage to a phone number or email."""
    # Sanitize for AppleScript
    message_escaped = message.replace('"', '\\"').replace("'", "\\'")
    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{to}" of targetService
        send "{message_escaped}" to targetBuddy
    end tell
    '''
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return f"Error sending iMessage: {r.stderr.strip()}"
        return f"Sent iMessage to {to}: {message[:100]}"
    except Exception as e:
        return f"Error: {e}"


def imessage_read(count: int = 5) -> str:
    """Read recent iMessages."""
    script = f'''
    tell application "Messages"
        set output to ""
        set recentChats to chats
        repeat with i from 1 to (count of recentChats)
            if i > {count} then exit repeat
            set c to item i of recentChats
            set chatName to name of c
            try
                set lastMsg to body of last item of messages of c
                set output to output & chatName & ": " & lastMsg & linefeed
            end try
        end repeat
        return output
    end tell
    '''
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return f"Error reading messages: {r.stderr.strip()}"
        return r.stdout.strip() or "(no recent messages)"
    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# Telegram (requires python-telegram-bot)
# ===========================================================================

class TelegramGateway:
    """Telegram bot that forwards messages to the agent pipeline.

    Setup:
        1. Create bot via @BotFather
        2. Set KANDIGA_TELEGRAM_TOKEN env var
        3. Start gateway: gateway.start()
    """

    def __init__(self, token: Optional[str] = None,
                 on_message: Optional[Callable] = None):
        self.token = token or os.environ.get("KANDIGA_TELEGRAM_TOKEN", "")
        self.on_message = on_message  # fn(text: str, chat_id: int) -> str
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start the Telegram bot in a background thread."""
        if not self.token:
            return False
        try:
            import telegram
        except ImportError:
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run_polling, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False

    def send(self, chat_id: int, text: str) -> bool:
        """Send a message to a Telegram chat."""
        if not self.token:
            return False
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            # Split long messages
            for i in range(0, len(text), 4000):
                chunk = text[i:i+4000]
                requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=10)
            return True
        except Exception:
            return False

    def _run_polling(self):
        """Poll for Telegram updates."""
        import requests
        url = f"https://api.telegram.org/bot{self.token}"
        offset = 0

        while self._running:
            try:
                resp = requests.get(
                    f"{url}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                    timeout=35,
                )
                data = resp.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = msg.get("chat", {}).get("id")

                    if text and chat_id and self.on_message:
                        try:
                            response = self.on_message(text, chat_id)
                            self.send(chat_id, response)
                        except Exception as e:
                            self.send(chat_id, f"Error: {e}")
            except Exception:
                time.sleep(5)


# ===========================================================================
# Tool registration
# ===========================================================================

def register_messaging_tools(registry: ToolRegistry) -> int:
    tools = [
        ("imessage_send", "Send an iMessage", {"to": "str", "message": "str"}, imessage_send),
        ("imessage_read", "Read recent iMessages", {"count": "int"}, imessage_read),
    ]
    for name, desc, schema, func in tools:
        registry.register(name, desc, schema, func)
    return len(tools)
