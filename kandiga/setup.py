"""One-time setup: download model, split experts, pack binary format, build dylib."""

from __future__ import annotations

import os
import sys
import time

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

console = Console()

EXPERT_CACHE_DIR = os.path.expanduser("~/.kandiga/experts")

MODELS = [
    {
        "id": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "name": "Qwen3.5-35B",
        "params": "35B (3B active)",
        "experts": "256 total, 8 active",
        "disk": "20 GB",
        "ram": "~2 GB",
        "min_mac": "8 GB",
        "speed": "3.4–6.5 tok/s",
    },
    {
        "id": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "name": "Qwen3.5-122B",
        "params": "122B (10B active)",
        "experts": "256 total, 8 active",
        "disk": "70 GB",
        "ram": "~4 GB",
        "min_mac": "16 GB",
        "speed": "~2 tok/s",
    },
    {
        "id": "mlx-community/Qwen3.5-397B-A17B-4bit",
        "name": "Qwen3.5-397B",
        "params": "397B (17B active)",
        "experts": "512 total, 10 active",
        "disk": "224 GB",
        "ram": "~8 GB",
        "min_mac": "24 GB",
        "speed": "~1 tok/s",
    },
]


def _get_system_ram_gb() -> int:
    """Get total system RAM in GB."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        return int(result.stdout.strip()) // (1024 ** 3)
    except Exception:
        return 0


def _pick_model() -> str:
    """Interactive model picker."""
    ram_gb = _get_system_ram_gb()

    console.print("[bold cyan]Available Models[/]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Model", style="bold")
    table.add_column("Parameters")
    table.add_column("Disk")
    table.add_column("RAM (Kandiga)")
    table.add_column("Min. Mac")
    table.add_column("Speed")

    for i, m in enumerate(MODELS):
        min_ram = int(m["min_mac"].split()[0])
        fits = ram_gb >= min_ram if ram_gb > 0 else True
        marker = " [green]✓[/]" if fits else " [red]✗[/]"
        table.add_row(
            str(i + 1),
            m["name"],
            m["params"],
            m["disk"],
            m["ram"],
            m["min_mac"] + marker,
            m["speed"],
        )

    console.print(table)

    if ram_gb > 0:
        console.print(f"\n  [dim]Your Mac: {ram_gb} GB RAM[/]")

    console.print()
    choice = Prompt.ask(
        "  Choose a model",
        choices=[str(i + 1) for i in range(len(MODELS))],
        default="1",
    )

    model = MODELS[int(choice) - 1]
    console.print(f"\n  Selected: [bold cyan]{model['name']}[/] ({model['disk']} download)\n")
    return model["id"]


def run_setup(model_path: str | None = None):
    """Run the full setup pipeline."""
    console.print()
    console.print("[bold cyan]Kandiga Setup[/]")
    console.print()

    # If no model specified, show picker
    if model_path is None or model_path == "mlx-community/Qwen3.5-35B-A3B-4bit":
        # Check if user passed --model explicitly
        if model_path is None:
            model_path = _pick_model()
        # else use the default

    model_name = model_path.split("/")[-1]
    model_cache_dir = os.path.join(EXPERT_CACHE_DIR, model_name)
    packed_dir = os.path.join(model_cache_dir, "packed")

    # -----------------------------------------------------------------------
    # Step 1: Download model
    # -----------------------------------------------------------------------
    console.print("[bold]Step 1/4:[/] Downloading model from HuggingFace...")
    console.print(f"  [dim]{model_path}[/]")

    try:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_path)
        console.print(f"  [green]\u2713[/] Model ready at {model_dir}")
    except Exception as e:
        console.print(f"  [red]\u2717[/] Download failed: {e}")
        console.print("  [dim]Make sure you have huggingface_hub installed and internet access.[/]")
        sys.exit(1)

    console.print()

    # -----------------------------------------------------------------------
    # Step 2: Split expert weights
    # -----------------------------------------------------------------------
    console.print("[bold]Step 2/4:[/] Splitting expert weights (one-time, ~45s)...")
    console.print(f"  [dim]40 layers x 256 experts = 10,240 files[/]")

    if os.path.isdir(os.path.join(model_cache_dir, "layer_00")):
        console.print(f"  [yellow]\u2713[/] Already split, skipping.")
    else:
        try:
            from kandiga._split_experts import split_experts
            split_experts(model_dir, model_cache_dir)
            console.print(f"  [green]\u2713[/] Experts split to {model_cache_dir}")
        except Exception as e:
            console.print(f"  [red]\u2717[/] Split failed: {e}")
            sys.exit(1)

    console.print()

    # -----------------------------------------------------------------------
    # Step 3: Pack binary format
    # -----------------------------------------------------------------------
    console.print("[bold]Step 3/4:[/] Packing binary format (one-time, ~60s)...")
    console.print(f"  [dim]40 layer files, ~441MB each[/]")

    if os.path.isdir(packed_dir) and any(f.endswith(".bin") for f in os.listdir(packed_dir)):
        console.print(f"  [yellow]\u2713[/] Already packed, skipping.")
    else:
        try:
            from kandiga._pack_experts import pack_experts
            pack_experts(model_cache_dir, packed_dir)
            console.print(f"  [green]\u2713[/] Binary format ready at {packed_dir}")
        except Exception as e:
            console.print(f"  [red]\u2717[/] Pack failed: {e}")
            sys.exit(1)

    console.print()

    # -----------------------------------------------------------------------
    # Step 4: Build CPU expert dylib
    # -----------------------------------------------------------------------
    console.print("[bold]Step 4/4:[/] Building CPU expert library...")

    try:
        from kandiga._build import build_cpu_expert_dylib
        dylib_path = build_cpu_expert_dylib()
        console.print(f"  [green]\u2713[/] Library built at {dylib_path}")
    except Exception as e:
        console.print(f"  [red]\u2717[/] Build failed: {e}")
        console.print("  [dim]Make sure Xcode command line tools are installed.[/]")
        sys.exit(1)

    console.print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    # Calculate total size
    total_bytes = 0
    if os.path.isdir(packed_dir):
        for f in os.listdir(packed_dir):
            fp = os.path.join(packed_dir, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)

    console.print("[bold green]Setup complete![/]")
    console.print()
    console.print(f"  Expert files: {packed_dir}")
    if total_bytes > 0:
        console.print(f"  Disk usage:   {total_bytes / 1e9:.1f}GB (packed binary)")
    console.print()
    console.print("  Run [bold cyan]kandiga chat[/] to start chatting.")
    console.print("  Run [bold cyan]kandiga chat --fast[/] for ~2x speed.")
    console.print()
