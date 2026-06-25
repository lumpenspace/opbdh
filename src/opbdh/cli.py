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
from .hal import QUOTE_OVERSPEND, QUOTE_REFUSAL, QUOTE_SUCCESS, hal_says
from .hf import estimate_model_size_gb, suggested_network_volume_gb
from .runpod import MaxSpendReached, make_plan, plan_summary, run_plan
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
    hal_says(QUOTE_REFUSAL)
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

def _valid_datacenters_for_gpus(gpu_types: list[str]) -> list[str]:
    # RunPod's GraphQL API no longer cleanly maps GPUs to datacenters.
    # We fetch the generic list of datacenters instead so the user can choose from valid options.
    import urllib.request
    import json
    from .remote import runpod_api_token
    query = """
    query {
      dataCenters {
        id
      }
    }
    """
    try:
        token = runpod_api_token()
        request = urllib.request.Request(
            f"https://api.runpod.io/graphql?api_key={token}",
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "opbdh/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return sorted([dc["id"] for dc in data.get("data", {}).get("dataCenters", [])])
    except Exception:
        return []


def _offer_save_config(config: OpbdhConfig) -> None:
    try:
        import questionary
        from .config import discover_local_config, save_config
        local_path = discover_local_config()
        if local_path:
            save = questionary.confirm(f"\nSave these updated requirements back to {local_path.name}?", default=True).ask()
            if save:
                save_config(config, local_path)
                console.print(f"[green]Saved updated config to {local_path}[/]")
    except Exception:
        pass


def _prompt_existing_network_volume(opbdh_plan, yes: bool, dry_run: bool) -> None:
    if opbdh_plan.config.auto_network_volume and not opbdh_plan.network_volume_id and not dry_run:
        from .runpod import find_network_volume, model_slug
        volume_name = opbdh_plan.config.network_volume_name or f"opbdh-{model_slug(opbdh_plan.config.model_id)}"
        dc_id = opbdh_plan.config.network_volume_data_center_id.strip()
        if dc_id:
            existing = find_network_volume(
                name=volume_name,
                data_center_id=dc_id,
                search_from=opbdh_plan.code_path.parent,
            )
            if existing and not yes:
                try:
                    import questionary
                except ImportError:
                    pass
                else:
                    while True:
                        choice = questionary.select(
                            f"\nFound existing network volume '{volume_name}' ({existing['id']}) in {dc_id}.",
                            choices=[
                                "Keep existing volume (reuse data)",
                                "Create new volume (requires different name)",
                                "Cancel launch"
                            ]
                        ).ask()
                        
                        if choice == "Keep existing volume (reuse data)":
                            opbdh_plan.network_volume_id = str(existing["id"])
                            break
                        elif choice == "Create new volume (requires different name)":
                            import typer
                            new_name = questionary.text("Enter new volume name:", instruction="(Ctrl-C or empty to go back)", default=f"{volume_name}-new").ask()
                            if not new_name or not new_name.strip():
                                continue
                            
                            new_size = questionary.text(
                                f"Enter volume size in GB (default: {opbdh_plan.config.pod_volume_gb}):", 
                                default=str(opbdh_plan.config.pod_volume_gb)
                            ).ask()
                            if new_size is None:
                                continue
                                
                            try:
                                opbdh_plan.config.network_volume_size_gb = int(new_size.strip() or opbdh_plan.config.pod_volume_gb)
                            except ValueError:
                                console.print("[red]Invalid size, using default.[/]")
                                opbdh_plan.config.network_volume_size_gb = opbdh_plan.config.pod_volume_gb

                            opbdh_plan.config.network_volume_name = new_name.strip()
                            break
                        else:
                            import typer
                            raise typer.Exit(1)

def _execute_run(config: OpbdhConfig, *, dry_run: bool, yes: bool) -> None:
    if not config.code:
        raise typer.BadParameter("Code path is required, either as an argument or config.code.")
    opbdh_plan = make_plan(config, code_path=Path(config.code))
    
    _prompt_existing_network_volume(opbdh_plan, yes, dry_run)

    payload = plan_summary(opbdh_plan)
    _print_plan(payload)
    if dry_run:
        run_plan(opbdh_plan, dry_run=True)
        console.print(f"[green]Dry run written to[/] {opbdh_plan.results_dir}")
        return

    if not yes and not _confirm_launch(payload):
        hal_says(QUOTE_REFUSAL)
        raise typer.Exit(1)
    while True:
        try:
            result = run_plan(opbdh_plan)
            break
        except MaxSpendReached as exc:
            hal_says(QUOTE_OVERSPEND)
            console.print(f"[red]{exc}[/] Results synced so far are in {opbdh_plan.results_dir}.")
            raise typer.Exit(1) from exc
        except RuntimeError as exc:
            msg = str(exc)
            
            is_volume_err = "create network volume:" in msg and "not found or does not support network volumes" in msg
            is_pod_err = "create pod: could not find any pods with required specifications" in msg
            
            is_balance_err = "balance" in msg.lower() or "funds" in msg.lower() or "payment" in msg.lower()
            is_remote_job_err = "remote job failed with exit code" in msg
            
            if is_balance_err:
                console.print(f"\n[red]Insufficient RunPod Balance:[/] Your account does not have enough funds to launch this pod.")
                console.print(f"[dim]RunPod API details: {msg}[/]")
                console.print("[yellow]Please add funds to your RunPod account to continue.[/]")
                raise typer.Exit(1)
                
            if is_remote_job_err:
                console.print(f"\n[red]Execution Error:[/] {msg}")
                stderr_path = opbdh_plan.results_dir / "logs" / "stderr.log"
                if stderr_path.exists():
                    stderr_content = stderr_path.read_text(encoding="utf-8").strip()
                    if stderr_content:
                        console.print(f"\n[red]Remote Standard Error:[/]\n{stderr_content}")
                
                stdout_path = opbdh_plan.results_dir / "logs" / "stdout.log"
                if stdout_path.exists():
                    stdout_content = stdout_path.read_text(encoding="utf-8").strip()
                    if stdout_content:
                        console.print(f"\n[dim]Remote Standard Output:[/]\n{stdout_content}")
                raise typer.Exit(1)
                
            if is_volume_err or is_pod_err:
                import re
                try:
                    import questionary
                except ImportError:
                    raise exc
                
                
                if is_pod_err:
                    dc_locked = opbdh_plan.config.network_volume_data_center_id
                    
                    if dc_locked:
                        console.print(f"\n[yellow]Out of Stock Error:[/] The datacenter [bold]{dc_locked}[/] does not currently have any of the requested GPUs available.")
                    else:
                        console.print("\n[yellow]Global Out of Stock Error:[/] RunPod does not currently have any of the requested GPUs available across all datacenters.")
                        
                    console.print(f"[blue]You were searching for:[/] {', '.join(opbdh_plan.gpu_type_ids)}")
                    
                    if dc_locked and opbdh_plan.network_volume_id:
                        console.print("[dim]Note: Because you are reusing an existing network volume, the search was restricted to its datacenter. If you select a new datacenter below, a new network volume will be created there.[/]")
                else:
                    console.print(f"\n[yellow]RunPod Configuration Error:[/] {msg}")
                
                valid_dcs = []
                
                # First try to extract from the error message if it's the volume error
                match = re.search(r"Available data centers:\s*(.*?)(?:\"|\.|$)", msg)
                if match:
                    valid_dcs = [d.strip() for d in match.group(1).split(",")]
                
                # Fallback to querying general datacenters if not available
                if not valid_dcs:
                    console.print("\n[dim]Fetching generic data centers from RunPod...[/]")
                    valid_dcs = _valid_datacenters_for_gpus(opbdh_plan.gpu_type_ids)
                
                while True:
                    choices = [
                        "Try another data center",
                        "Change GPU requirements",
                        "Try without a network volume (ephemeral disk only)",
                        "Cancel launch"
                    ]
                    
                    choice = questionary.select(
                        "How would you like to proceed?",
                        choices=choices
                    ).ask()
                    
                    if choice == "Try another data center":
                        if valid_dcs:
                            new_dc = questionary.select("Select a data center:", choices=["< Back"] + valid_dcs + ["Enter manually..."]).ask()
                            if new_dc == "< Back":
                                continue
                            if new_dc == "Enter manually...":
                                new_dc = questionary.text("Enter data center id:", instruction="(or empty to go back)").ask()
                                if not new_dc or not new_dc.strip():
                                    continue
                        else:
                            new_dc = questionary.text("Enter data center id:", instruction="(or empty to go back)").ask()
                            if not new_dc or not new_dc.strip():
                                continue
                                
                        if new_dc and new_dc.strip() and new_dc != "Enter manually...":
                            old_dc = opbdh_plan.config.network_volume_data_center_id
                            old_vol_id = opbdh_plan.network_volume_id
                            
                            opbdh_plan.config.network_volume_data_center_id = new_dc.strip()
                            opbdh_plan.network_volume_id = ""
                            
                            if old_vol_id and old_dc and old_dc != new_dc.strip():
                                if questionary.confirm(f"\nDo you want to automatically delete your newly orphaned network volume ({old_vol_id}) in {old_dc}?", default=False).ask():
                                    try:
                                        from .remote import _runpod_rest
                                        _runpod_rest("DELETE", f"/networkvolumes/{old_vol_id}")
                                        console.print(f"[green]Deleted old network volume {old_vol_id}[/]")
                                    except Exception as e:
                                        console.print(f"[red]Failed to delete volume:[/] {e}")
                                        
                            _offer_save_config(opbdh_plan.config)
                            _prompt_existing_network_volume(opbdh_plan, yes, dry_run)
                            break
                    elif choice == "Change GPU requirements":
                        new_vram = questionary.text(
                            "Enter new minimum VRAM in GB:",
                            instruction="(Ctrl-C or empty to go back)",
                            default=str(config.vram_gb)
                        ).ask()
                        
                        if not new_vram or not new_vram.strip():
                            continue
                            
                        new_price = questionary.text(
                            "Enter new max dollars per hour:",
                            instruction="(leave blank for no max, Ctrl-C to go back)",
                            default=str(config.max_dollars_per_hour) if config.max_dollars_per_hour is not None else ""
                        ).ask()
                        
                        if new_price is None:
                            continue
                            
                        try:
                            config.vram_gb = int(new_vram)
                            config.max_dollars_per_hour = float(new_price) if new_price.strip() else None
                            opbdh_plan = make_plan(config, code_path=Path(config.code))
                            console.print(f"[green]Updated requirements! Now searching for:[/] {', '.join(opbdh_plan.gpu_type_ids)}")
                            _offer_save_config(config)
                            _prompt_existing_network_volume(opbdh_plan, yes, dry_run)
                        except ValueError as e:
                            console.print(f"[red]Error updating config:[/] {e}")
                            continue
                        break
                    elif choice == "Try without a network volume (ephemeral disk only)":
                        opbdh_plan.config.auto_network_volume = False
                        opbdh_plan.network_volume_id = ""
                        break
                    
                    raise typer.Exit(1) from exc
                continue
            raise
    if result:
        hal_says(QUOTE_SUCCESS)
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
