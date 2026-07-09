"""First-run setup wizard, shown when `opbdh` is invoked bare with no config anywhere."""

from __future__ import annotations

import os

from .config import OpbdhConfig, discover_local_config, global_config_path, save_config

_PROVIDER_TOKENS = {
    "runpod": ("RUNPOD_API_TOKEN", "RUNPOD_API_KEY"),
    "primeintellect": ("PRIME_INTELLECT_API_KEY", "PRIME_API_KEY"),
}


def is_configured() -> bool:
    return global_config_path().exists() or discover_local_config() is not None


def _missing_token_vars(provider: str) -> tuple[str, ...]:
    names = _PROVIDER_TOKENS[provider]
    if any(os.environ.get(name, "").strip() for name in names):
        return ()
    return names


def run_setup_wizard() -> int:
    try:
        import clypi
        import clypi.parsers as cp
    except ImportError:
        print("clypi is required for the setup wizard; run `opbdh config wizard` instead.")
        return 1
    try:
        return _wizard(clypi, cp)
    except (KeyboardInterrupt, EOFError):
        clypi.cprint("\n\nThis mission is too important for me to allow you to jeopardize it. Bye!", fg="red")
        return 1


def _banner(clypi) -> None:
    frame = clypi.Styler(fg="cyan")
    eye = clypi.Styler(fg="red", bold=True)
    title = clypi.Styler(bold=True)
    print()
    print(frame("  ╭──────────────────────────────────╮"))
    print(frame("  │") + "            " + eye("◉") + "  " + title("OPBDH") + "             " + frame("│"))
    print(frame("  │") + "   Open the Pod Bay Door, Hal.    " + frame("│"))
    print(frame("  ╰──────────────────────────────────╯"))
    print()


def _wizard(clypi, cp) -> int:
    _banner(clypi)
    clypi.cprint("Good afternoon. I couldn't find any configuration, so let's set up your first mission.", bold=True)
    clypi.cprint("Everything you choose here becomes a default you can override per run.\n", fg="cyan")

    provider = clypi.prompt(
        "Compute provider (runpod / primeintellect)",
        default="runpod",
        parser=cp.Literal("runpod", "primeintellect"),
    )
    missing = _missing_token_vars(provider)
    if missing:
        clypi.cprint(f"\n  No API key found in the environment ({' or '.join(missing)}).", fg="yellow")
        clypi.cprint(f'  Before launching, run:  export {missing[0]}="..."\n', fg="yellow")
    else:
        clypi.cprint("\n  ✔ API key found in the environment.\n", fg="green")

    model = clypi.prompt("Default Hugging Face model id (blank to choose per run)", default="", parser=cp.Str())
    code = clypi.prompt("Default code path", default="{cwd}/run.py", parser=cp.Str())
    vram_gb = clypi.prompt("Minimum GPU VRAM (GB)", default=24, parser=cp.Int(gte=1))
    hourly = clypi.prompt("Max dollars per hour (0 for no cap)", default=0.0, parser=cp.Float(gte=0))
    spend = clypi.prompt("Max spend per run (dollars)", default=5.0, parser=cp.Float(gt=0))

    auto_volume = False
    data_center = ""
    if provider == "runpod":
        auto_volume = clypi.confirm(
            "Create a persistent network volume for the model cache when needed?", default=False
        )
        if auto_volume:
            data_center = clypi.prompt("RunPod data center id for the volume", default="EU-RO-1", parser=cp.Str())

    scope = clypi.prompt(
        "Save config as (global: all projects / local: this directory)",
        default="global",
        parser=cp.Literal("global", "local"),
    )

    from pathlib import Path

    target = global_config_path() if scope == "global" else Path.cwd() / "opbdh.json"
    cfg = OpbdhConfig(
        model_id=model,
        code=code,
        provider=provider,
        vram_gb=vram_gb,
        max_dollars_per_hour=hourly or None,
        max_spend_dollars=spend,
        auto_network_volume=auto_volume,
        network_volume_data_center_id=data_center,
    )
    save_config(cfg, target)

    clypi.cprint(f"\n  ✔ Config saved to {target}", fg="green", bold=True)
    clypi.cprint("\nNext steps:", bold=True)
    if missing:
        clypi.cprint(f'  export {missing[0]}="..."', fg="yellow")
    print("  opbdh launch ./run.py --dry-run    # verify and price a run without spending")
    print("  opbdh launch ./run.py              # open the pod bay door")
    print()
    return 0
