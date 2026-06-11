from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .config import OpbdhConfig, global_config_path, load_config, save_config
from .gpu import candidate_gpus
from .hf import estimate_model_size_gb, suggested_network_volume_gb
from .runpod import make_plan, plan_summary, run_plan
from .verify import verify_code


app = typer.Typer(help="OPBDH: Open the Pod Bay Door, Hal. Run model scripts on RunPod.")
run_app = typer.Typer(help="Plan, launch, or interactively build a RunPod run.")
config_app = typer.Typer(help="Inspect and build OPBDH config.")
models_app = typer.Typer(help="Hugging Face model helpers.")
app.add_typer(run_app, name="run")
app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")
console = Console()


def _overrides(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None and value != ""}


def _print_plan(payload: dict[str, Any]) -> None:
    table = Table(title="OPBDH plan")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, value in payload.items():
        if isinstance(value, list):
            display = "\n".join(str(item) for item in value) or "[]"
        else:
            display = str(value)
        table.add_row(key, display)
    console.print(table)


def _confirm_launch(plan_payload: dict[str, Any]) -> bool:
    hourly = plan_payload.get("estimated_hourly_dollars")
    max_spend = plan_payload.get("max_spend_dollars")
    console.print(
        f"[yellow]This can launch billable RunPod compute.[/] "
        f"Estimated first GPU candidate: ${hourly}/hr, max spend guard: ${max_spend}."
    )
    return typer.confirm("Launch now?", default=False)


@app.command()
def plan(
    code: Path | None = typer.Argument(None, help="Code file or directory to upload."),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Local OPBDH JSON config."),
    model: str | None = typer.Option(None, "--model", "-m", help="Hugging Face model id."),
    command: str | None = typer.Option(None, "--command", help="Remote shell command. Defaults from code path."),
    vram_gb: int | None = typer.Option(None, "--vram-gb", help="Minimum GPU VRAM."),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", help="Estimated hourly cap."),
    max_spend: float | None = typer.Option(None, "--max-spend", help="Spend guard for this run."),
) -> None:
    cfg = load_config(
        local_config=config_file,
        overrides=_overrides(
            model_id=model,
            code=str(code) if code else None,
            command=command,
            vram_gb=vram_gb,
            max_dollars_per_hour=max_dollars_per_hour,
            max_spend_dollars=max_spend,
        ),
    )
    if not cfg.code:
        raise typer.BadParameter("Code path is required, either as an argument or config.code.")
    opbdh_plan = make_plan(cfg, code_path=Path(cfg.code))
    _print_plan(plan_summary(opbdh_plan))


@app.command()
def verify(
    code: Path = typer.Argument(..., help="Code file or directory to statically verify."),
    command: str = typer.Option("", "--command", help="Remote command, if the path needs one."),
) -> None:
    result = verify_code(code, command=command)
    if result.ok:
        console.print(f"[green]OK[/] checked {len(result.checked)} file(s).")
        return
    for error in result.errors:
        console.print(f"[red]{error}[/]")
    raise typer.Exit(1)


def _load_run_config(
    *,
    code: Path | None,
    config_file: Path | None,
    model: str | None,
    command: str | None,
    vram_gb: int | None,
    max_dollars_per_hour: float | None,
    max_spend: float | None,
    network_volume_id: str | None,
    auto_network_volume: bool | None,
    network_volume_data_center_id: str | None,
) -> OpbdhConfig:
    return load_config(
        local_config=config_file,
        overrides=_overrides(
            model_id=model,
            code=str(code) if code else None,
            command=command,
            vram_gb=vram_gb,
            max_dollars_per_hour=max_dollars_per_hour,
            max_spend_dollars=max_spend,
            network_volume_id=network_volume_id,
            auto_network_volume=auto_network_volume,
            network_volume_data_center_id=network_volume_data_center_id,
        ),
    )


def _execute_run(config: OpbdhConfig, *, dry_run: bool, yes: bool) -> None:
    if not config.code:
        raise typer.BadParameter("Code path is required, either as an argument or config.code.")
    opbdh_plan = make_plan(config, code_path=Path(config.code))
    payload = plan_summary(opbdh_plan)
    _print_plan(payload)
    if dry_run:
        run_plan(opbdh_plan, dry_run=True)
        console.print(f"[green]Dry run written to[/] {opbdh_plan.results_dir}")
        return
    if not yes and not _confirm_launch(payload):
        raise typer.Exit(1)
    result = run_plan(opbdh_plan)
    if result:
        console.print(f"[green]Run complete[/] {result.results_dir}")


@run_app.command("now")
def run_now(
    code: Path | None = typer.Argument(None, help="Code file or directory to upload."),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Local OPBDH JSON config."),
    model: str | None = typer.Option(None, "--model", "-m", help="Hugging Face model id."),
    command: str | None = typer.Option(None, "--command", help="Remote shell command. Defaults from code path."),
    vram_gb: int | None = typer.Option(None, "--vram-gb", help="Minimum GPU VRAM."),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", help="Estimated hourly cap."),
    max_spend: float | None = typer.Option(None, "--max-spend", help="Spend guard for this run."),
    network_volume_id: str | None = typer.Option(None, "--network-volume-id", help="Existing RunPod network volume id."),
    auto_network_volume: bool | None = typer.Option(
        None,
        "--auto-network-volume/--no-auto-network-volume",
        help="Create a network volume if none is configured.",
    ),
    network_volume_data_center_id: str | None = typer.Option(
        None,
        "--network-volume-data-center-id",
        help="RunPod data center id for auto-created volumes, for example EU-RO-1.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Verify and print the plan without contacting RunPod."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip billable-compute confirmation."),
) -> None:
    cfg = _load_run_config(
        code=code,
        config_file=config_file,
        model=model,
        command=command,
        vram_gb=vram_gb,
        max_dollars_per_hour=max_dollars_per_hour,
        max_spend=max_spend,
        network_volume_id=network_volume_id,
        auto_network_volume=auto_network_volume,
        network_volume_data_center_id=network_volume_data_center_id,
    )
    _execute_run(cfg, dry_run=dry_run, yes=yes)


@app.command("launch")
def launch(
    code: Path | None = typer.Argument(None, help="Code file or directory to upload."),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Local OPBDH JSON config."),
    model: str | None = typer.Option(None, "--model", "-m", help="Hugging Face model id."),
    command: str | None = typer.Option(None, "--command", help="Remote shell command. Defaults from code path."),
    vram_gb: int | None = typer.Option(None, "--vram-gb", help="Minimum GPU VRAM."),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", help="Estimated hourly cap."),
    max_spend: float | None = typer.Option(None, "--max-spend", help="Spend guard for this run."),
    network_volume_id: str | None = typer.Option(None, "--network-volume-id", help="Existing RunPod network volume id."),
    auto_network_volume: bool | None = typer.Option(None, "--auto-network-volume/--no-auto-network-volume"),
    network_volume_data_center_id: str | None = typer.Option(None, "--network-volume-data-center-id"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Verify and print the plan without contacting RunPod."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip billable-compute confirmation."),
) -> None:
    """Shortcut for `opbdh run now`."""
    cfg = _load_run_config(
        code=code,
        config_file=config_file,
        model=model,
        command=command,
        vram_gb=vram_gb,
        max_dollars_per_hour=max_dollars_per_hour,
        max_spend=max_spend,
        network_volume_id=network_volume_id,
        auto_network_volume=auto_network_volume,
        network_volume_data_center_id=network_volume_data_center_id,
    )
    _execute_run(cfg, dry_run=dry_run, yes=yes)


@run_app.command("wizard")
def run_wizard(
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Local OPBDH JSON config."),
) -> None:
    try:
        import questionary
    except Exception as exc:
        raise typer.BadParameter("questionary is required for the run wizard.") from exc

    base = load_config(local_config=config_file)
    model = _questionary_model(questionary, default=base.model_id or "Qwen")
    model_estimate = estimate_model_size_gb(model)
    code = questionary.path("Code file or directory", default=base.code or str(Path.cwd() / "run.py")).ask() or base.code
    command = questionary.text("Remote command override", default=base.command).ask() or ""
    vram_gb = int(questionary.text("Minimum VRAM GB", default=str(base.vram_gb)).ask() or str(base.vram_gb))
    hourly_default = "" if base.max_dollars_per_hour is None else str(base.max_dollars_per_hour)
    hourly_text = questionary.text("Max dollars/hour estimate (blank for no cap)", default=hourly_default).ask() or ""
    spend = float(questionary.text("Max spend dollars", default=str(base.max_spend_dollars)).ask() or str(base.max_spend_dollars))
    network_volume_id = questionary.text("Existing network volume id (blank for none)", default=base.network_volume_id).ask() or ""
    auto_volume = False
    data_center = base.network_volume_data_center_id
    if not network_volume_id:
        suggested_volume = suggested_network_volume_gb(model_estimate, fallback_gb=base.pod_volume_gb)
        auto_volume = bool(questionary.confirm(f"Create a network volume if needed? Suggested size: {suggested_volume} GB", default=base.auto_network_volume).ask())
        if auto_volume:
            data_center = questionary.text("RunPod data center id", default=data_center or "EU-RO-1").ask() or ""
            base.network_volume_size_gb = suggested_volume
    dry_run = bool(questionary.confirm("Dry run first?", default=True).ask())
    yes = bool(questionary.confirm("Skip launch confirmation?", default=False).ask()) if not dry_run else True
    cfg = load_config(
        local_config=config_file,
        overrides=_overrides(
            model_id=model,
            code=code,
            command=command,
            vram_gb=vram_gb,
            max_dollars_per_hour=float(hourly_text) if hourly_text else None,
            max_spend_dollars=spend,
            network_volume_id=network_volume_id,
            auto_network_volume=auto_volume,
            network_volume_data_center_id=data_center,
            network_volume_size_gb=base.network_volume_size_gb,
        ),
    )
    _execute_run(cfg, dry_run=dry_run, yes=yes)


@config_app.command("show")
def config_show(
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Local OPBDH JSON config."),
) -> None:
    cfg = load_config(local_config=config_file)
    console.print_json(json.dumps(asdict(cfg), indent=2, sort_keys=True))


@config_app.command("write")
def config_write(
    output: Path | None = typer.Option(None, "--output", "-o", help="Config path. Defaults to global config."),
    model: str = typer.Option(..., "--model", "-m", help="Hugging Face model id."),
    code: str = typer.Option("", "--code", help="Default code path. Supports {cwd}, {model_slug}, and env vars."),
    command: str = typer.Option("", "--command", help="Default remote command."),
    vram_gb: int = typer.Option(24, "--vram-gb"),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour"),
    max_spend: float = typer.Option(5.0, "--max-spend"),
    auto_network_volume: bool = typer.Option(False, "--auto-network-volume/--no-auto-network-volume"),
    network_volume_data_center_id: str = typer.Option("", "--network-volume-data-center-id"),
) -> None:
    cfg = OpbdhConfig(
        model_id=model,
        code=code,
        command=command,
        vram_gb=vram_gb,
        max_dollars_per_hour=max_dollars_per_hour,
        max_spend_dollars=max_spend,
        auto_network_volume=auto_network_volume,
        network_volume_data_center_id=network_volume_data_center_id,
    )
    path = save_config(cfg, output or global_config_path())
    console.print(f"[green]Wrote[/] {path}")


@config_app.command("wizard")
def config_wizard(
    scope: str = typer.Option("global", "--scope", help="global or local"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    try:
        import questionary
    except Exception as exc:
        raise typer.BadParameter("questionary is required for the wizard; use `opbdh config write` instead.") from exc

    model = _questionary_model(questionary)
    model_estimate = estimate_model_size_gb(model)
    suggested_volume = suggested_network_volume_gb(model_estimate)
    code = questionary.text("Default local code path", default="{cwd}/run.py").ask() or ""
    command = questionary.text("Remote command override", default="").ask() or ""
    vram_gb = int(questionary.text("Minimum VRAM GB", default="24").ask() or "24")
    hourly_text = questionary.text("Max dollars/hour estimate (blank for no cap)", default="").ask() or ""
    spend = float(questionary.text("Max spend dollars", default="5").ask() or "5")
    auto_volume = bool(questionary.confirm("Create a RunPod network volume when none is configured?", default=False).ask())
    data_center = ""
    if auto_volume:
        data_center = questionary.text("RunPod data center id for the volume", default="EU-RO-1").ask() or ""
        console.print(f"Suggested volume size for {model}: {suggested_volume} GB")
    cfg = OpbdhConfig(
        model_id=model,
        code=code,
        command=command,
        vram_gb=vram_gb,
        max_dollars_per_hour=float(hourly_text) if hourly_text else None,
        max_spend_dollars=spend,
        auto_network_volume=auto_volume,
        network_volume_data_center_id=data_center,
        network_volume_size_gb=suggested_volume if auto_volume else None,
    )
    if output:
        target = output
    elif scope == "local":
        target = Path.cwd() / "opbdh.json"
    elif scope == "global":
        target = global_config_path()
    else:
        raise typer.BadParameter("--scope must be global or local")
    save_config(cfg, target)
    console.print(f"[green]Wrote[/] {target}")


def _questionary_model(questionary: Any, *, default: str = "Qwen") -> str:
    query = questionary.text("Search Hugging Face models", default=default).ask() or ""
    choices: list[str] = []
    if query.strip():
        try:
            from huggingface_hub import HfApi

            choices = [model.modelId for model in HfApi().list_models(search=query, limit=25) if model.modelId]
        except Exception:
            choices = []
    if choices:
        selected = questionary.autocomplete("Model", choices=choices, default=choices[0]).ask()
        if selected:
            return str(selected)
    return questionary.text("Model id", default=query).ask() or query


@models_app.command("search")
def models_search(query: str, limit: int = typer.Option(10, "--limit", "-n")) -> None:
    from huggingface_hub import HfApi

    table = Table(title=f"Hugging Face models: {query}")
    table.add_column("Model")
    table.add_column("Downloads", justify="right")
    for model in HfApi().list_models(search=query, limit=limit):
        table.add_row(str(model.modelId), str(getattr(model, "downloads", "") or ""))
    console.print(table)


@models_app.command("size")
def models_size(model: str) -> None:
    estimate = estimate_model_size_gb(model)
    console.print_json(json.dumps({
        "model": model,
        "size_gb": estimate.size_gb,
        "source": estimate.source,
        "suggested_network_volume_gb": suggested_network_volume_gb(estimate),
    }))


@app.command("gpus")
def gpus(
    vram_gb: int = typer.Option(24, "--vram-gb"),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour"),
    cloud_type: str = typer.Option("SECURE", "--cloud-type"),
) -> None:
    table = Table(title="OPBDH GPU candidates")
    table.add_column("RunPod GPU id")
    table.add_column("VRAM", justify="right")
    table.add_column("$/hr estimate", justify="right")
    for gpu in candidate_gpus(vram_gb, max_dollars_per_hour, cloud_type):
        table.add_row(gpu.id, str(gpu.memory_gb), f"{gpu.hourly(cloud_type):.2f}")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
