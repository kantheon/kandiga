"""Kandiga CLI — Run frontier MoE models on consumer hardware."""

from __future__ import annotations

import argparse
import sys
import threading


def _check_update_background():
    """Non-blocking update check on startup."""
    try:
        from kandiga._update import check_for_update, print_update_notice
        latest = check_for_update()
        if latest:
            print_update_notice(latest)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        prog="kandiga",
        description="Kandiga \u2014 Run frontier MoE models on consumer hardware",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit"
    )
    subparsers = parser.add_subparsers(dest="command")

    # kandiga setup
    setup_parser = subparsers.add_parser(
        "setup", help="Download model and prepare expert files"
    )
    setup_parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-35B-A3B-4bit",
        help="HuggingFace model ID (default: Qwen3.5-35B-A3B-4bit)",
    )

    # kandiga chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--fast", action="store_true", help="K=4 mode (~6.5 tok/s, slightly lower quality)"
    )
    chat_parser.add_argument(
        "--tools", action="store_true", help="Enable web search + file access"
    )
    chat_parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-35B-A3B-4bit",
        help="HuggingFace model ID",
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per response (default: 2048)",
    )

    # kandiga serve
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible HTTP API")
    serve_parser.add_argument(
        "--port", type=int, default=8340, help="Port to listen on (default: 8340)"
    )
    serve_parser.add_argument(
        "--fast", action="store_true", help="K=4 mode"
    )
    serve_parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-35B-A3B-4bit",
        help="HuggingFace model ID",
    )

    # kandiga bench
    subparsers.add_parser("bench", help="Run inference benchmarks")

    # kandiga update
    subparsers.add_parser("update", help="Update kandiga to the latest version")

    # kandiga changelog
    subparsers.add_parser("changelog", help="Show version history")

    # Top-level flags for default (chat) mode
    parser.add_argument(
        "--fast", action="store_true", help="K=4 mode"
    )

    # Check if any arg is a known subcommand
    known_commands = {"setup", "chat", "serve", "bench", "update", "changelog"}
    has_subcommand = any(a in known_commands for a in sys.argv[1:])

    if not has_subcommand and len(sys.argv) > 1 and not all(a.startswith("-") for a in sys.argv[1:]):
        # One-shot mode: treat non-flag args as prompt
        fast = "--fast" in sys.argv
        prompt_words = [a for a in sys.argv[1:] if not a.startswith("-")]
        args = argparse.Namespace(
            command=None, version=False, fast=fast, prompt=prompt_words
        )
    else:
        parser.add_argument(
            "prompt", nargs="*", help="One-shot prompt"
        )
        args = parser.parse_args()

    # --version
    if args.version:
        from kandiga import __version__
        print(f"kandiga {__version__}")
        return

    # Start background update check (doesn't block startup)
    if args.command not in ("update", "changelog"):
        t = threading.Thread(target=_check_update_background, daemon=True)
        t.start()

    if args.command == "setup":
        from kandiga.setup import run_setup

        run_setup(args.model)
    elif args.command == "chat":
        from kandiga.chat import run_chat

        run_chat(
            fast=args.fast,
            tools=args.tools,
            model=args.model,
            max_tokens=args.max_tokens,
        )
    elif args.command == "serve":
        from kandiga.serve import run_serve

        run_serve(port=args.port, fast=args.fast, model=args.model)
    elif args.command == "bench":
        from kandiga.bench import run_bench

        run_bench()
    elif args.command == "update":
        from kandiga._update import run_update

        run_update()
    elif args.command == "changelog":
        from kandiga._update import run_changelog

        run_changelog()
    else:
        # Default: if prompt given, one-shot. Otherwise, interactive chat.
        if args.prompt:
            from kandiga.chat import one_shot

            one_shot(" ".join(args.prompt), fast=args.fast)
        else:
            from kandiga.chat import run_chat

            run_chat(fast=args.fast)


if __name__ == "__main__":
    main()
