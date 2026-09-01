from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import hx
from .api import _huggingface_model_options
from .config import OpbdhConfig, global_config_path, load_config, save_config
from .estimate import GOALS, estimate_for_model
from .finetune import (
    DATA_FORMATS,
    FINETUNE_METHODS,
    MODEL_TYPES,
    FineTuneDataError,
    FineTuneExample,
    FineTuneProject,
    activate_finetune_recipe,
    add_project_examples,
    available_tags,
    build_finetune_run_config,
    create_finetune_recipe,
    delete_finetune_recipe,
    estimate_finetune_resources,
    expand_data_paths,
    find_finetune_root,
    infer_model_type,
    load_finetune_project,
    normalize_tags,
    prepare_finetune_job,
    project_data_path,
    read_examples,
    read_project_examples,
    recipe_names,
    rename_finetune_recipe,
    replace_project_examples,
    require_finetune_extra,
    save_finetune_project,
    set_finetune_method,
    sync_active_recipe,
    validate_examples_for_model,
    validate_recipe_name,
    write_examples,
)
from .gpu import candidate_gpus
from .hal import QUOTE_OVERSPEND, QUOTE_REFUSAL, QUOTE_SUCCESS, hal_says
from .hf import estimate_model_size_gb, suggested_network_volume_gb
from .remote import InsufficientCreditsError
from .runpod import MaxSpendReached, RunEvent, make_plan, plan_summary, run_plan
from .verify import verify_code


app = typer.Typer(help="OPBDH: Open the Pod Bay Door, Hal. Run model scripts on RunPod.")
run_app = typer.Typer(help="Plan, launch, or interactively build a RunPod run.")
config_app = typer.Typer(help="Inspect and build OPBDH config.")
models_app = typer.Typer(help="Hugging Face model helpers.")
app.add_typer(run_app, name="run")
app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")
console = Console()


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    from . import setup_wizard

    if not setup_wizard.is_configured() and _stdin_is_tty():
        raise typer.Exit(setup_wizard.run_setup_wizard())
    console.print(ctx.get_help())


def _overrides(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None and value != ""}


def _print_plan(payload: dict[str, Any]) -> None:
    table = Table(title="OPBDH plan", title_style="bold #ef4444", border_style="grey37")
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
    gpu_count = plan_payload.get("gpu_count", 1)
    provider_id = str(plan_payload.get("provider", "compute"))
    provider = {"runpod": "RunPod", "primeintellect": "Prime Intellect"}.get(
        provider_id, provider_id
    )
    cloud_type = plan_payload.get("cloud_type")
    location = f" {cloud_type}" if cloud_type else ""
    console.print(
        f"[yellow]This can launch billable {provider}{location} compute.[/] "
        f"Estimated {gpu_count}-GPU pod: ${hourly}/hr, max spend guard: ${max_spend}."
    )
    return typer.confirm("Launch now?", default=False)


@app.command()
def plan(
    code: Path | None = typer.Argument(None, help="Code file or directory to upload."),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Local OPBDH JSON config."),
    model: str | None = typer.Option(None, "--model", "-m", help="Hugging Face model id."),
    command: str | None = typer.Option(None, "--command", "-x", help="Remote shell command. Defaults from code path."),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Compute provider: runpod or primeintellect."),
    vram_gb: int | None = typer.Option(None, "--vram-gb", "-v", help="Minimum GPU VRAM."),
    gpu_count: int | None = typer.Option(None, "--gpu-count", "-g", help="GPUs per pod."),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", "-d", help="Estimated hourly cap."),
    max_spend: float | None = typer.Option(None, "--max-spend", "-s", help="Spend guard for this run."),
    min_vcpu_per_gpu: int | None = typer.Option(None, "--min-vcpu-per-gpu", "-u", help="Minimum host vCPUs per GPU."),
    min_ram_per_gpu: int | None = typer.Option(None, "--min-ram-per-gpu", "-r", help="Minimum host RAM per GPU, in GB."),
) -> None:
    cfg = load_config(
        local_config=config_file,
        overrides=_overrides(
            model_id=model,
            code=str(code) if code else None,
            command=command,
            provider=provider,
            vram_gb=vram_gb,
            gpu_count=gpu_count,
            max_dollars_per_hour=max_dollars_per_hour,
            max_spend_dollars=max_spend,
            min_vcpu_per_gpu=min_vcpu_per_gpu,
            min_ram_per_gpu_gb=min_ram_per_gpu,
        ),
    )
    if not cfg.code:
        raise typer.BadParameter("Code path is required, either as an argument or config.code.")
    opbdh_plan = make_plan(cfg, code_path=Path(cfg.code))
    _warn_ignored_volume_options(cfg)
    _print_plan(plan_summary(opbdh_plan))


@app.command()
def verify(
    code: Path = typer.Argument(..., help="Code file or directory to statically verify."),
    command: str = typer.Option("", "--command", "-x", help="Remote command, if the path needs one."),
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
    min_vcpu_per_gpu: int | None = None,
    min_ram_per_gpu: int | None = None,
    provider: str | None = None,
    gpu_count: int | None = None,
) -> OpbdhConfig:
    return load_config(
        local_config=config_file,
        overrides=_overrides(
            model_id=model,
            code=str(code) if code else None,
            command=command,
            provider=provider,
            vram_gb=vram_gb,
            gpu_count=gpu_count,
            max_dollars_per_hour=max_dollars_per_hour,
            max_spend_dollars=max_spend,
            network_volume_id=network_volume_id,
            auto_network_volume=auto_network_volume,
            network_volume_data_center_id=network_volume_data_center_id,
            min_vcpu_per_gpu=min_vcpu_per_gpu,
            min_ram_per_gpu_gb=min_ram_per_gpu,
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
            save = questionary.confirm(f"\nSave these updated requirements back to {local_path.name}?", default=True, style=hx.questionary_style()).ask()
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
                            ],
                            style=hx.questionary_style(),
                        ).ask()
                        
                        if choice == "Keep existing volume (reuse data)":
                            opbdh_plan.network_volume_id = str(existing["id"])
                            break
                        elif choice == "Create new volume (requires different name)":
                            import typer
                            new_name = questionary.text("Enter new volume name:", instruction="(Ctrl-C or empty to go back)", default=f"{volume_name}-new", style=hx.questionary_style()).ask()
                            if not new_name or not new_name.strip():
                                continue
                            
                            new_size = questionary.text(
                                f"Enter volume size in GB (default: {opbdh_plan.config.pod_volume_gb}):", 
                                default=str(opbdh_plan.config.pod_volume_gb),
                                style=hx.questionary_style(),
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

def _warn_ignored_volume_options(config: OpbdhConfig) -> None:
    if (config.provider.strip().lower() or "runpod") != "runpod" and (config.auto_network_volume or config.network_volume_id.strip()):
        console.print(
            "[yellow]Note:[/] network volumes are RunPod-only; ignoring the configured "
            "network volume options for this provider. The model will download to the pod's own disk."
        )


def _print_run_event(event: RunEvent) -> None:
    if event.kind == "billing":
        console.print(event.message)


def _execute_run(config: OpbdhConfig, *, dry_run: bool, yes: bool) -> None:
    if not config.code:
        raise typer.BadParameter("Code path is required, either as an argument or config.code.")
    try:
        opbdh_plan = make_plan(config, code_path=Path(config.code))
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    _warn_ignored_volume_options(config)

    if config.provider.strip().lower() == "runpod":
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
            result = run_plan(opbdh_plan, on_event=_print_run_event)
            break
        except MaxSpendReached as exc:
            hal_says(QUOTE_OVERSPEND)
            console.print(f"[red]{exc}[/] Results synced so far are in {opbdh_plan.results_dir}.")
            raise typer.Exit(1) from exc
        except InsufficientCreditsError as exc:
            console.print(f"\n[red]{exc}[/]")
            raise typer.Exit(1) from exc
        except RuntimeError as exc:
            msg = str(exc)
            
            is_volume_err = "create network volume:" in msg and "not found or does not support network volumes" in msg
            is_pod_err = "create pod: could not find any pods with required specifications" in msg
            
            is_balance_err = "balance" in msg.lower() or "funds" in msg.lower() or "payment" in msg.lower()
            is_remote_job_err = "remote job failed with exit code" in msg
            
            if is_balance_err:
                provider_name = "Prime Intellect" if config.provider.strip().lower() == "primeintellect" else "RunPod"
                console.print(f"\n[red]Insufficient {provider_name} Balance:[/] Your account does not have enough funds to launch this pod.")
                console.print(f"[dim]{provider_name} API details: {msg}[/]")
                console.print(f"[yellow]Please add funds to your {provider_name} account to continue.[/]")
                raise typer.Exit(1)
                
            if is_remote_job_err:
                console.print(f"\n[red]Execution Error:[/] {msg}")
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
                        choices=choices,
                        style=hx.questionary_style(),
                    ).ask()
                    
                    if choice == "Try another data center":
                        if valid_dcs:
                            new_dc = questionary.select("Select a data center:", choices=["< Back"] + valid_dcs + ["Enter manually..."], style=hx.questionary_style()).ask()
                            if new_dc == "< Back":
                                continue
                            if new_dc == "Enter manually...":
                                new_dc = questionary.text("Enter data center id:", instruction="(or empty to go back)", style=hx.questionary_style()).ask()
                                if not new_dc or not new_dc.strip():
                                    continue
                        else:
                            new_dc = questionary.text("Enter data center id:", instruction="(or empty to go back)", style=hx.questionary_style()).ask()
                            if not new_dc or not new_dc.strip():
                                continue
                                
                        if new_dc and new_dc.strip() and new_dc != "Enter manually...":
                            old_dc = opbdh_plan.config.network_volume_data_center_id
                            old_vol_id = opbdh_plan.network_volume_id
                            
                            opbdh_plan.config.network_volume_data_center_id = new_dc.strip()
                            opbdh_plan.network_volume_id = ""
                            
                            if old_vol_id and old_dc and old_dc != new_dc.strip():
                                if questionary.confirm(f"\nDo you want to automatically delete your newly orphaned network volume ({old_vol_id}) in {old_dc}?", default=False, style=hx.questionary_style()).ask():
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
                            default=str(config.vram_gb),
                            style=hx.questionary_style(),
                        ).ask()
                        
                        if not new_vram or not new_vram.strip():
                            continue
                            
                        new_price = questionary.text(
                            "Enter new max dollars per hour:",
                            instruction="(leave blank for no max, Ctrl-C to go back)",
                            default=str(config.max_dollars_per_hour) if config.max_dollars_per_hour is not None else "",
                            style=hx.questionary_style(),
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
    command: str | None = typer.Option(None, "--command", "-x", help="Remote shell command. Defaults from code path."),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Compute provider: runpod or primeintellect."),
    vram_gb: int | None = typer.Option(None, "--vram-gb", "-v", help="Minimum GPU VRAM."),
    gpu_count: int | None = typer.Option(None, "--gpu-count", "-g", help="GPUs per pod."),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", "-d", help="Estimated hourly cap."),
    max_spend: float | None = typer.Option(None, "--max-spend", "-s", help="Spend guard for this run."),
    network_volume_id: str | None = typer.Option(None, "--network-volume-id", "-V", help="Existing RunPod network volume id."),
    auto_network_volume: bool | None = typer.Option(
        None,
        "--auto-network-volume/--no-auto-network-volume", "-a/-A",
        help="Create a network volume if none is configured.",
    ),
    network_volume_data_center_id: str | None = typer.Option(
        None,
        "--network-volume-data-center-id", "-D",
        help="RunPod data center id for auto-created volumes, for example EU-RO-1.",
    ),
    min_vcpu_per_gpu: int | None = typer.Option(None, "--min-vcpu-per-gpu", "-u", help="Minimum host vCPUs per GPU."),
    min_ram_per_gpu: int | None = typer.Option(None, "--min-ram-per-gpu", "-r", help="Minimum host RAM per GPU, in GB."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Verify and print the plan without contacting RunPod."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip billable-compute confirmation."),
) -> None:
    cfg = _load_run_config(
        code=code,
        config_file=config_file,
        model=model,
        command=command,
        vram_gb=vram_gb,
        gpu_count=gpu_count,
        max_dollars_per_hour=max_dollars_per_hour,
        max_spend=max_spend,
        network_volume_id=network_volume_id,
        auto_network_volume=auto_network_volume,
        network_volume_data_center_id=network_volume_data_center_id,
        min_vcpu_per_gpu=min_vcpu_per_gpu,
        min_ram_per_gpu=min_ram_per_gpu,
        provider=provider,
    )
    _execute_run(cfg, dry_run=dry_run, yes=yes)


@app.command("launch")
def launch(
    code: Path | None = typer.Argument(None, help="Code file or directory to upload."),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Local OPBDH JSON config."),
    model: str | None = typer.Option(None, "--model", "-m", help="Hugging Face model id."),
    command: str | None = typer.Option(None, "--command", "-x", help="Remote shell command. Defaults from code path."),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Compute provider: runpod or primeintellect."),
    vram_gb: int | None = typer.Option(None, "--vram-gb", "-v", help="Minimum GPU VRAM."),
    gpu_count: int | None = typer.Option(None, "--gpu-count", "-g", help="GPUs per pod."),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", "-d", help="Estimated hourly cap."),
    max_spend: float | None = typer.Option(None, "--max-spend", "-s", help="Spend guard for this run."),
    network_volume_id: str | None = typer.Option(None, "--network-volume-id", "-V", help="Existing RunPod network volume id."),
    auto_network_volume: bool | None = typer.Option(None, "--auto-network-volume/--no-auto-network-volume", "-a/-A"),
    network_volume_data_center_id: str | None = typer.Option(None, "--network-volume-data-center-id", "-D"),
    min_vcpu_per_gpu: int | None = typer.Option(None, "--min-vcpu-per-gpu", "-u", help="Minimum host vCPUs per GPU."),
    min_ram_per_gpu: int | None = typer.Option(None, "--min-ram-per-gpu", "-r", help="Minimum host RAM per GPU, in GB."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Verify and print the plan without contacting RunPod."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip billable-compute confirmation."),
) -> None:
    """Shortcut for `opbdh run now`."""
    cfg = _load_run_config(
        code=code,
        config_file=config_file,
        model=model,
        command=command,
        vram_gb=vram_gb,
        gpu_count=gpu_count,
        max_dollars_per_hour=max_dollars_per_hour,
        max_spend=max_spend,
        network_volume_id=network_volume_id,
        auto_network_volume=auto_network_volume,
        network_volume_data_center_id=network_volume_data_center_id,
        min_vcpu_per_gpu=min_vcpu_per_gpu,
        min_ram_per_gpu=min_ram_per_gpu,
        provider=provider,
    )
    _execute_run(cfg, dry_run=dry_run, yes=yes)


@run_app.command("wizard")
def run_wizard(
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Local OPBDH JSON config."),
) -> None:
    hx.banner("plan a run interactively")
    try:
        import questionary
    except Exception as exc:
        raise typer.BadParameter("questionary is required for the run wizard.") from exc

    base = load_config(local_config=config_file)
    model = _questionary_model(questionary, default=base.model_id or "Qwen")
    model_estimate = estimate_model_size_gb(model)
    code = questionary.path("Code file or directory", default=base.code or str(Path.cwd() / "run.py"), style=hx.questionary_style()).ask() or base.code
    command = questionary.text("Remote command override", default=base.command, style=hx.questionary_style()).ask() or ""
    vram_gb = int(questionary.text("Minimum VRAM GB", default=str(base.vram_gb), style=hx.questionary_style()).ask() or str(base.vram_gb))
    gpu_count = int(questionary.text("GPUs per pod", default=str(base.gpu_count), style=hx.questionary_style()).ask() or str(base.gpu_count))
    hourly_default = "" if base.max_dollars_per_hour is None else str(base.max_dollars_per_hour)
    hourly_text = questionary.text("Max dollars/hour estimate (blank for no cap)", default=hourly_default, style=hx.questionary_style()).ask() or ""
    spend = float(questionary.text("Max spend dollars", default=str(base.max_spend_dollars), style=hx.questionary_style()).ask() or str(base.max_spend_dollars))
    network_volume_id = questionary.text("Existing network volume id (blank for none)", default=base.network_volume_id, style=hx.questionary_style()).ask() or ""
    auto_volume = False
    data_center = base.network_volume_data_center_id
    if not network_volume_id:
        suggested_volume = suggested_network_volume_gb(model_estimate, fallback_gb=base.pod_volume_gb)
        auto_volume = bool(questionary.confirm(f"Create a network volume if needed? Suggested size: {suggested_volume} GB", default=base.auto_network_volume, style=hx.questionary_style()).ask())
        if auto_volume:
            data_center = questionary.text("RunPod data center id", default=data_center or "EU-RO-1", style=hx.questionary_style()).ask() or ""
            base.network_volume_size_gb = suggested_volume
    dry_run = bool(questionary.confirm("Dry run first?", default=True, style=hx.questionary_style()).ask())
    yes = bool(questionary.confirm("Skip launch confirmation?", default=False, style=hx.questionary_style()).ask()) if not dry_run else True
    cfg = load_config(
        local_config=config_file,
        overrides=_overrides(
            model_id=model,
            code=code,
            command=command,
            vram_gb=vram_gb,
            gpu_count=gpu_count,
            max_dollars_per_hour=float(hourly_text) if hourly_text else None,
            max_spend_dollars=spend,
            network_volume_id=network_volume_id,
            auto_network_volume=auto_volume,
            network_volume_data_center_id=data_center,
            network_volume_size_gb=base.network_volume_size_gb,
        ),
    )
    _execute_run(cfg, dry_run=dry_run, yes=yes)


def _tag_values(values: list[str] | None) -> list[str]:
    split = [part for value in (values or []) for part in value.split(",")]
    return list(normalize_tags(split))


def _print_finetune_project(root: Path, project: FineTuneProject) -> None:
    table = Table(title="OPBDH fine-tune", title_style="bold #ef4444", border_style="grey37")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("recipe", project.active_recipe)
    table.add_row("model", project.model_id or "not selected")
    table.add_row("model type", project.model_type or "not selected")
    table.add_row("method", project.method)
    table.add_row("examples", str(project.example_count))
    table.add_row("tag groups", ", ".join(project.selected_tags) if project.selected_tags else "all examples")
    table.add_row("GPUs", str(project.gpu_count))
    table.add_row("dataset", str(project_data_path(root, project)))
    console.print(table)


def _fine_tune_model_type(questionary: Any, current: str = "") -> str:
    labels = {
        "Chat model (input is a message list)": "chat",
        "Base model (input is text)": "base",
    }
    choices = list(labels)
    if current in MODEL_TYPES:
        current_label = next(label for label, value in labels.items() if value == current)
        choices.remove(current_label)
        choices.insert(0, current_label)
    selected = questionary.select("Model type", choices=choices, style=hx.questionary_style()).ask()
    if selected is None:
        raise typer.Exit(1)
    return labels[str(selected)]


_FINE_TUNE_METHOD_LABELS = {
    "LoRA (recommended default)": "lora",
    "QLoRA (4-bit, lower VRAM)": "qlora",
    "Full fine-tune": "full",
}


def _fine_tune_method_choice(questionary: Any, current: str = "lora") -> str:
    choices = list(_FINE_TUNE_METHOD_LABELS)
    current_label = next(
        label for label, value in _FINE_TUNE_METHOD_LABELS.items() if value == current
    )
    choices.remove(current_label)
    choices.insert(0, current_label)
    selected = questionary.select(
        "Fine-tuning method",
        choices=choices,
        style=hx.questionary_style(),
    ).ask()
    if selected is None:
        raise typer.Exit(1)
    return _FINE_TUNE_METHOD_LABELS[str(selected)]


def _fine_tune_settings(
    questionary: Any,
    project: FineTuneProject,
    *,
    advanced: bool,
    choose_method: bool = True,
) -> None:
    if choose_method:
        set_finetune_method(project, _fine_tune_method_choice(questionary, project.method))

    gpu_choices = [str(project.gpu_count), "1", "2", "4", "8", "Custom..."]
    gpu_choices = list(dict.fromkeys(gpu_choices))
    selected_gpus = questionary.select(
        "GPUs in the pod",
        choices=gpu_choices,
        style=hx.questionary_style(),
    ).ask()
    if selected_gpus is None:
        raise typer.Exit(1)
    if selected_gpus == "Custom...":
        selected_gpus = questionary.text(
            "GPU count",
            default=str(project.gpu_count),
            style=hx.questionary_style(),
        ).ask()
    project.gpu_count = max(1, int(str(selected_gpus)))

    if not advanced:
        return
    project.epochs = float(
        questionary.text("Epochs", default=str(project.epochs), style=hx.questionary_style()).ask()
        or project.epochs
    )
    project.learning_rate = float(
        questionary.text(
            "Learning rate",
            default=str(project.learning_rate),
            style=hx.questionary_style(),
        ).ask()
        or project.learning_rate
    )
    project.max_length = int(
        questionary.text(
            "Maximum sequence length",
            default=str(project.max_length),
            style=hx.questionary_style(),
        ).ask()
        or project.max_length
    )
    project.per_device_batch_size = int(
        questionary.text(
            "Batch size per GPU",
            default=str(project.per_device_batch_size),
            style=hx.questionary_style(),
        ).ask()
        or project.per_device_batch_size
    )
    project.gradient_accumulation_steps = int(
        questionary.text(
            "Gradient accumulation steps",
            default=str(project.gradient_accumulation_steps),
            style=hx.questionary_style(),
        ).ask()
        or project.gradient_accumulation_steps
    )
    if project.method in {"lora", "qlora"}:
        project.lora_r = int(
            questionary.text(
                "LoRA rank",
                default=str(project.lora_r),
                style=hx.questionary_style(),
            ).ask()
            or project.lora_r
        )
        project.lora_alpha = int(
            questionary.text(
                "LoRA alpha",
                default=str(project.lora_alpha),
                style=hx.questionary_style(),
            ).ask()
            or project.lora_alpha
        )
        project.lora_dropout = float(
            questionary.text(
                "LoRA dropout",
                default=str(project.lora_dropout),
                style=hx.questionary_style(),
            ).ask()
            or project.lora_dropout
        )
    packing = questionary.confirm(
        "Pack short examples into full sequences?",
        default=project.packing,
        style=hx.questionary_style(),
    ).ask()
    if packing is not None:
        project.packing = bool(packing)
    project.seed = int(
        questionary.text(
            "Random seed",
            default=str(project.seed),
            style=hx.questionary_style(),
        ).ask()
        or project.seed
    )
    project.max_spend_dollars = float(
        questionary.text(
            "Maximum spend in dollars",
            default=str(project.max_spend_dollars),
            style=hx.questionary_style(),
        ).ask()
        or project.max_spend_dollars
    )


def _fine_tune_recipe_labels(project: FineTuneProject) -> dict[str, str]:
    sync_active_recipe(project)
    labels: dict[str, str] = {}
    for name in recipe_names(project):
        recipe = project.recipes[name]
        marker = " (active)" if name == project.active_recipe else ""
        model = recipe.get("model_id") or "no model"
        labels[f"{name}{marker} — {recipe['method']}, {model}, {recipe['gpu_count']} GPU(s)"] = name
    return labels


def _fine_tune_manage_recipes(questionary: Any, root: Path, project: FineTuneProject) -> None:
    while True:
        names = recipe_names(project)
        actions = []
        if len(names) > 1:
            actions.append("Switch recipe")
        actions.extend(
            [
                "Create recipe with defaults",
                "Clone current recipe",
                "Rename current recipe",
                "Reset current recipe parameters to defaults",
            ]
        )
        if len(names) > 1:
            actions.append("Delete a recipe")
        actions.append("Done")
        action = questionary.select(
            "Training recipes",
            choices=actions,
            style=hx.questionary_style(),
        ).ask()
        if action in {None, "Done"}:
            return

        if action == "Switch recipe":
            labels = _fine_tune_recipe_labels(project)
            selected = questionary.select(
                "Recipe",
                choices=list(labels),
                style=hx.questionary_style(),
            ).ask()
            if selected is None:
                continue
            activate_finetune_recipe(project, labels[str(selected)])
        elif action == "Create recipe with defaults":
            name = questionary.text(
                "Recipe name",
                default=f"{project.method}-experiment",
                style=hx.questionary_style(),
            ).ask()
            if name is None:
                continue
            method = _fine_tune_method_choice(questionary)
            create_finetune_recipe(project, str(name), method=method)
            if questionary.confirm(
                "Edit this recipe's parameters now?",
                default=False,
                style=hx.questionary_style(),
            ).ask():
                _fine_tune_settings(questionary, project, advanced=True, choose_method=False)
        elif action == "Clone current recipe":
            name = questionary.text(
                "Name for the clone",
                default=f"{project.active_recipe}-copy",
                style=hx.questionary_style(),
            ).ask()
            if name is None:
                continue
            create_finetune_recipe(project, str(name), clone_current=True)
        elif action == "Rename current recipe":
            name = questionary.text(
                "New recipe name",
                default=project.active_recipe,
                style=hx.questionary_style(),
            ).ask()
            if name is None:
                continue
            rename_finetune_recipe(project, str(name))
        elif action == "Reset current recipe parameters to defaults":
            if not questionary.confirm(
                f"Reset {project.active_recipe!r} {project.method} parameters?",
                default=False,
                style=hx.questionary_style(),
            ).ask():
                continue
            set_finetune_method(project, project.method, reset_defaults=True)
        elif action == "Delete a recipe":
            labels = _fine_tune_recipe_labels(project)
            selected = questionary.select(
                "Recipe to delete",
                choices=list(labels),
                style=hx.questionary_style(),
            ).ask()
            if selected is None:
                continue
            name = labels[str(selected)]
            if not questionary.confirm(
                f"Delete recipe {name!r}? The shared dataset will be kept.",
                default=False,
                style=hx.questionary_style(),
            ).ask():
                continue
            delete_finetune_recipe(project, name)

        save_finetune_project(root, project)
        console.print(f"[green]Active recipe:[/] {project.active_recipe}")


def _fine_tune_files(questionary: Any) -> tuple[list[FineTuneExample], list[Path]]:
    paths: list[Path] = []
    while True:
        raw = questionary.path(
            "TOML, JSON, JSONL file or directory",
            style=hx.questionary_style(),
        ).ask()
        if not raw:
            if paths:
                break
            raise typer.Exit(1)
        paths.append(Path(str(raw)))
        if not questionary.confirm("Add another path?", default=False, style=hx.questionary_style()).ask():
            break
    format_labels = {
        "Detect automatically": "auto",
        "OPBDH input/output": "opbdh",
        "OpenAI": "openai",
        "Anthropic": "anthropic",
    }
    selected_format = questionary.select(
        "Input format",
        choices=list(format_labels),
        style=hx.questionary_style(),
    ).ask()
    if selected_format is None:
        raise typer.Exit(1)
    tag_text = questionary.text(
        "Tags to add to every imported example (comma-separated, optional)",
        default="",
        style=hx.questionary_style(),
    ).ask() or ""
    expanded = expand_data_paths(paths)
    return (
        read_examples(expanded, source_format=format_labels[str(selected_format)], extra_tags=_tag_values([tag_text])),
        expanded,
    )


def _fine_tune_created_examples(questionary: Any, model_type: str) -> list[FineTuneExample]:
    examples: list[FineTuneExample] = []
    while True:
        if model_type == "chat":
            messages: list[dict[str, str]] = []
            while True:
                choices = ["user", "system", "developer", "assistant", "tool"]
                if messages:
                    choices.append("Done adding input messages")
                role = questionary.select(
                    "Input message role",
                    choices=choices,
                    style=hx.questionary_style(),
                ).ask()
                if role is None:
                    raise typer.Exit(1)
                if role == "Done adding input messages":
                    break
                content = questionary.text(
                    f"{role} message",
                    style=hx.questionary_style(),
                ).ask()
                if content is None:
                    raise typer.Exit(1)
                if str(content).strip():
                    messages.append({"role": str(role), "content": str(content)})
            input_value: str | list[dict[str, str]] = messages
        else:
            raw_input = questionary.text("Input", style=hx.questionary_style()).ask()
            if raw_input is None:
                raise typer.Exit(1)
            input_value = str(raw_input)

        output = questionary.text("Output", style=hx.questionary_style()).ask()
        if output is None:
            raise typer.Exit(1)
        tags = questionary.text(
            "Tags (comma-separated, optional)",
            default="",
            style=hx.questionary_style(),
        ).ask() or ""
        example = FineTuneExample(input_value, str(output), tuple(_tag_values([str(tags)])))
        # Reuse the canonical validator so interactive examples obey exactly
        # the same contract as hand-edited files.
        validated = read_examples_from_values([example])
        examples.extend(validated)
        if not questionary.confirm("Create another example?", default=True, style=hx.questionary_style()).ask():
            break
    return examples


def read_examples_from_values(examples: list[FineTuneExample]) -> list[FineTuneExample]:
    """Validate already-structured examples through the public data contract."""

    validated: list[FineTuneExample] = []
    for example in examples:
        payload = example.to_dict()
        # This import is intentionally local to keep the CLI's public imports
        # focused on operations rather than parser internals.
        from .finetune import example_from_record

        validated.append(example_from_record(payload, source_format="opbdh"))
    return validated


def _fine_tune_choose_tags(questionary: Any, project: FineTuneProject, examples: list[FineTuneExample]) -> None:
    tags = available_tags(examples)
    if not tags:
        console.print("[yellow]No examples have tags yet.[/]")
        project.selected_tags = []
        return
    mode = questionary.select(
        "Examples to train on",
        choices=["All examples", "Only selected tag groups"],
        style=hx.questionary_style(),
    ).ask()
    if mode != "Only selected tag groups":
        project.selected_tags = []
        return
    choices = [
        questionary.Choice(tag, checked=(not project.selected_tags or tag in project.selected_tags))
        for tag in tags
    ]
    selected = questionary.checkbox(
        "Tag groups (an example matching any selected tag is included)",
        choices=choices,
        style=hx.questionary_style(),
    ).ask()
    if not selected:
        console.print("[yellow]No tag selected; keeping all examples.[/]")
        project.selected_tags = []
    else:
        project.selected_tags = list(normalize_tags(selected))


def _fine_tune_example_label(index: int, example: FineTuneExample) -> str:
    if isinstance(example.input, str):
        preview = example.input
    else:
        preview = next(
            (message["content"] for message in reversed(example.input) if message["role"] == "user"),
            example.input[-1]["content"],
        )
    preview = " ".join(preview.split())
    if len(preview) > 60:
        preview = preview[:57] + "..."
    tags = ", ".join(example.tags) if example.tags else "untagged"
    return f"{index + 1}. {preview} [{tags}]"


def _fine_tune_tag_targets(questionary: Any, examples: list[FineTuneExample]) -> set[int]:
    tags = available_tags(examples)
    choices = ["All examples", *[f"Examples tagged: {tag}" for tag in tags]]
    selected = questionary.select(
        "Apply to",
        choices=choices,
        style=hx.questionary_style(),
    ).ask()
    if not selected or selected == "All examples":
        return set(range(len(examples)))
    tag = str(selected).removeprefix("Examples tagged: ")
    return {index for index, example in enumerate(examples) if tag in example.tags}


def _fine_tune_manage_tags(
    questionary: Any,
    root: Path,
    project: FineTuneProject,
    examples: list[FineTuneExample],
) -> list[FineTuneExample]:
    while True:
        tags = available_tags(examples)
        action = questionary.select(
            "Manage tags",
            choices=[
                "Add tags in bulk",
                "Remove tags in bulk",
                "Rename a tag",
                "Edit one example's tags",
                "Clear every tag",
                "Done",
            ],
            style=hx.questionary_style(),
        ).ask()
        if action in {None, "Done"}:
            return examples

        updated = list(examples)
        if action == "Add tags in bulk":
            targets = _fine_tune_tag_targets(questionary, examples)
            raw = questionary.text(
                "Tags to add (comma-separated)",
                style=hx.questionary_style(),
            ).ask() or ""
            additions = _tag_values([str(raw)])
            if not additions:
                continue
            for index in targets:
                example = updated[index]
                new_tags = normalize_tags([*example.tags, *additions])
                updated[index] = FineTuneExample(example.input, example.output, new_tags)
        elif action == "Remove tags in bulk":
            if not tags:
                console.print("[yellow]There are no tags to remove.[/]")
                continue
            targets = _fine_tune_tag_targets(questionary, examples)
            removals = questionary.checkbox(
                "Tags to remove",
                choices=tags,
                style=hx.questionary_style(),
            ).ask() or []
            removal_set = set(removals)
            for index in targets:
                example = updated[index]
                updated[index] = FineTuneExample(
                    example.input,
                    example.output,
                    tuple(tag for tag in example.tags if tag not in removal_set),
                )
        elif action == "Rename a tag":
            if not tags:
                console.print("[yellow]There are no tags to rename.[/]")
                continue
            old_tag = questionary.select("Tag to rename", choices=tags, style=hx.questionary_style()).ask()
            new_tag = questionary.text("New tag", style=hx.questionary_style()).ask()
            if not old_tag or not new_tag or not str(new_tag).strip():
                continue
            for index, example in enumerate(updated):
                replacement = [str(new_tag).strip() if tag == old_tag else tag for tag in example.tags]
                updated[index] = FineTuneExample(example.input, example.output, normalize_tags(replacement))
        elif action == "Edit one example's tags":
            labels = [_fine_tune_example_label(index, example) for index, example in enumerate(examples)]
            selected = questionary.select(
                "Example",
                choices=labels,
                style=hx.questionary_style(),
            ).ask()
            if selected is None:
                continue
            index = labels.index(str(selected))
            example = updated[index]
            raw = questionary.text(
                "Tags (comma-separated; blank clears them)",
                default=", ".join(example.tags),
                style=hx.questionary_style(),
            ).ask()
            if raw is None:
                continue
            updated[index] = FineTuneExample(example.input, example.output, tuple(_tag_values([str(raw)])))
        elif action == "Clear every tag":
            confirmed = questionary.confirm(
                "Clear tags from every example?",
                default=False,
                style=hx.questionary_style(),
            ).ask()
            if not confirmed:
                continue
            updated = [FineTuneExample(example.input, example.output) for example in examples]

        examples = replace_project_examples(root, project, updated)
        console.print(
            f"[green]Updated tags[/] ({len(available_tags(examples))} tag groups across {len(examples)} examples)"
        )


def _interactive_fine_tune(root: Path, project: FineTuneProject) -> bool:
    hx.banner("prepare a supervised fine-tune")
    try:
        import questionary
    except Exception as exc:
        raise typer.BadParameter("questionary is required for interactive fine-tuning.") from exc

    if not project.model_id:
        project.model_id = _questionary_model(questionary, default="Qwen")
    if not project.model_type:
        project.model_type = _fine_tune_model_type(questionary)

    examples = read_project_examples(root, project)
    while not examples:
        try:
            source_choice = questionary.select(
                "How would you like to provide examples?",
                choices=["Point OPBDH at files", "Create examples together"],
                style=hx.questionary_style(),
            ).ask()
            if source_choice is None:
                return False
            if source_choice == "Point OPBDH at files":
                incoming, sources = _fine_tune_files(questionary)
            else:
                incoming, sources = _fine_tune_created_examples(questionary, project.model_type), []
            validate_examples_for_model(incoming, project.model_type)
            examples = add_project_examples(root, project, incoming, source_files=sources)
        except (FineTuneDataError, OSError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            continue
        if questionary.confirm("Manage tags now?", default=True, style=hx.questionary_style()).ask():
            examples = _fine_tune_manage_tags(questionary, root, project, examples)
        if available_tags(examples):
            _fine_tune_choose_tags(questionary, project, examples)
        _fine_tune_settings(questionary, project, advanced=False)
        save_finetune_project(root, project)
        console.print(f"[green]Saved editable dataset to[/] {project_data_path(root, project)}")
        return True

    while True:
        project.example_count = len(examples)
        _print_finetune_project(root, project)
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Launch fine-tune",
                "Add/import files",
                "Create examples",
                "Manage example tags",
                "Choose tag groups",
                "Manage training recipes",
                "Edit current recipe",
                "Cancel",
            ],
            style=hx.questionary_style(),
        ).ask()
        try:
            if action == "Launch fine-tune":
                save_finetune_project(root, project)
                return True
            if action == "Add/import files":
                incoming, sources = _fine_tune_files(questionary)
                validate_examples_for_model(incoming, project.model_type)
                examples = add_project_examples(root, project, incoming, source_files=sources)
            elif action == "Create examples":
                incoming = _fine_tune_created_examples(questionary, project.model_type)
                examples = add_project_examples(root, project, incoming)
            elif action == "Manage example tags":
                examples = _fine_tune_manage_tags(questionary, root, project, examples)
            elif action == "Choose tag groups":
                _fine_tune_choose_tags(questionary, project, examples)
                save_finetune_project(root, project)
            elif action == "Manage training recipes":
                _fine_tune_manage_recipes(questionary, root, project)
            elif action == "Edit current recipe":
                previous_type = project.model_type
                project.model_id = _questionary_model(questionary, default=project.model_id)
                project.model_type = _fine_tune_model_type(questionary, current=project.model_type)
                try:
                    validate_examples_for_model(examples, project.model_type)
                except FineTuneDataError:
                    project.model_type = previous_type
                    raise
                _fine_tune_settings(questionary, project, advanced=True)
                save_finetune_project(root, project)
            else:
                return False
        except (FineTuneDataError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")


def _default_finetune_import_output(sources: list[Path]) -> Path:
    if len(sources) == 1 and sources[0].expanduser().is_file():
        source = sources[0].expanduser()
        return source.with_name(f"{source.stem}.opbdh.toml")
    return Path.cwd() / "opbdh-data.toml"


@app.command("ft:import")
def fine_tune_import(
    sources: list[Path] = typer.Argument(..., help="OpenAI, Anthropic, or OPBDH files/directories."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output .toml, .json, or .jsonl path."),
    source_format: str = typer.Option("auto", "--format", "-f", help=f"One of: {', '.join(DATA_FORMATS)}."),
    tags: list[str] | None = typer.Option(None, "--tag", "-t", help="Tag to add; repeat or comma-separate."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing output file."),
) -> None:
    """Convert OpenAI/Anthropic data into OPBDH's input/output format."""

    try:
        require_finetune_extra()
        expanded = expand_data_paths(sources)
        examples = read_examples(expanded, source_format=source_format, extra_tags=_tag_values(tags))
        target = output or _default_finetune_import_output(sources)
        write_examples(target, examples, overwrite=force)
    except (FineTuneDataError, FileExistsError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    tag_names = available_tags(examples)
    suffix = f"; tags: {', '.join(tag_names)}" if tag_names else ""
    console.print(f"[green]Wrote {len(examples)} example(s)[/] to {target.expanduser().resolve()}{suffix}")


@app.command("ft")
def fine_tune(
    data: list[Path] | None = typer.Option(None, "--data", "-d", help="Data file/directory; repeat as needed."),
    source_format: str = typer.Option("auto", "--format", "-f", help=f"One of: {', '.join(DATA_FORMATS)}."),
    model: str | None = typer.Option(None, "--model", "-m", help="Hugging Face base or chat model id."),
    model_type: str | None = typer.Option(None, "--model-type", help="base or chat; inferred from new data if omitted."),
    recipe: str | None = typer.Option(None, "--recipe", "-r", help="Saved training recipe to use or create."),
    method: str | None = typer.Option(None, "--method", help=f"One of: {', '.join(FINETUNE_METHODS)}."),
    tags: list[str] | None = typer.Option(None, "--tag", "-t", help="Train on matching tag group; repeat as needed."),
    gpu_count: int | None = typer.Option(None, "--gpu-count", "-g", min=1, help="GPUs in the training pod."),
    vram_gb: int | None = typer.Option(None, "--vram-gb", "-v", min=1, help="Override automatic VRAM sizing."),
    provider: str | None = typer.Option(None, "--provider", "-p", help="runpod or primeintellect."),
    epochs: float | None = typer.Option(None, "--epochs", min=0.01),
    learning_rate: float | None = typer.Option(None, "--learning-rate", min=0.0),
    max_length: int | None = typer.Option(None, "--max-length", min=1),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1, help="Examples per GPU per step."),
    gradient_accumulation: int | None = typer.Option(None, "--gradient-accumulation", min=1),
    lora_r: int | None = typer.Option(None, "--lora-r", min=1, help="LoRA/QLoRA adapter rank."),
    lora_alpha: int | None = typer.Option(None, "--lora-alpha", min=1),
    lora_dropout: float | None = typer.Option(None, "--lora-dropout", min=0.0, max=0.999999),
    packing: bool | None = typer.Option(None, "--packing/--no-packing", help="Pack short examples."),
    seed: int | None = typer.Option(None, "--seed", min=0),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", min=0.0),
    max_spend: float | None = typer.Option(None, "--max-spend", min=0.0),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Build and verify the job without renting a pod."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the billable-compute confirmation."),
) -> None:
    """Gather examples, size a pod, and launch a supervised fine-tune."""

    try:
        require_finetune_extra()
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    discovered_root = find_finetune_root()
    root = discovered_root or Path.cwd().resolve()
    base = load_config(cwd=root)
    project = load_finetune_project(root)
    project_exists = project is not None
    if project is None:
        project = FineTuneProject(
            model_id=base.model_id,
            gpu_count=base.gpu_count,
            provider=base.provider,
            max_dollars_per_hour=base.max_dollars_per_hour,
            max_spend_dollars=base.max_spend_dollars,
        )

    requested_method = method.strip().lower() if method else None
    requested_type = model_type.strip().lower() if model_type else None
    try:
        requested_recipe = validate_recipe_name(recipe) if recipe else None
    except FineTuneDataError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if requested_method and requested_method not in FINETUNE_METHODS:
        raise typer.BadParameter(f"--method must be one of: {', '.join(FINETUNE_METHODS)}")
    if requested_type and requested_type not in MODEL_TYPES:
        raise typer.BadParameter(f"--model-type must be one of: {', '.join(MODEL_TYPES)}")

    reset_method_defaults = not project_exists
    if requested_recipe:
        if not project.recipes:
            project.active_recipe = requested_recipe
        elif requested_recipe in project.recipes:
            activate_finetune_recipe(project, requested_recipe)
        else:
            create_finetune_recipe(project, requested_recipe, method=requested_method or "lora")
            reset_method_defaults = True

    if model:
        project.model_id = model
    if requested_type:
        project.model_type = requested_type
    if requested_method:
        set_finetune_method(project, requested_method, reset_defaults=reset_method_defaults)
    if gpu_count is not None:
        project.gpu_count = gpu_count
    if vram_gb is not None:
        project.vram_gb = vram_gb
    if provider:
        project.provider = provider
    if epochs is not None:
        project.epochs = epochs
    if learning_rate is not None:
        project.learning_rate = learning_rate
    if max_length is not None:
        project.max_length = max_length
    if batch_size is not None:
        project.per_device_batch_size = batch_size
    if gradient_accumulation is not None:
        project.gradient_accumulation_steps = gradient_accumulation
    if lora_r is not None:
        project.lora_r = lora_r
    if lora_alpha is not None:
        project.lora_alpha = lora_alpha
    if lora_dropout is not None:
        project.lora_dropout = lora_dropout
    if packing is not None:
        project.packing = packing
    if seed is not None:
        project.seed = seed
    if max_dollars_per_hour is not None:
        project.max_dollars_per_hour = max_dollars_per_hour
    if max_spend is not None:
        project.max_spend_dollars = max_spend
    if tags is not None:
        project.selected_tags = _tag_values(tags)

    try:
        if data:
            expanded = expand_data_paths(data)
            incoming = read_examples(expanded, source_format=source_format)
            if not project.model_type:
                project.model_type = infer_model_type(incoming)
            validate_examples_for_model(incoming, project.model_type)
            add_project_examples(root, project, incoming, source_files=expanded)

        direct_requested = any(
            value is not None
            for value in (
                data,
                model,
                model_type,
                recipe,
                method,
                tags,
                gpu_count,
                vram_gb,
                provider,
                epochs,
                learning_rate,
                max_length,
                batch_size,
                gradient_accumulation,
                lora_r,
                lora_alpha,
                lora_dropout,
                packing,
                seed,
                max_dollars_per_hour,
                max_spend,
            )
        ) or dry_run or yes
        examples = read_project_examples(root, project)
        needs_input = not project.model_id or not examples
        if _stdin_is_tty() and (not direct_requested or needs_input):
            if not _interactive_fine_tune(root, project):
                raise typer.Exit(1)
            examples = read_project_examples(root, project)
        elif needs_input:
            missing = "a model id" if not project.model_id else "training examples"
            raise FineTuneDataError(f"fine-tuning needs {missing}; run `opbdh ft` in a terminal or pass flags")

        if not project.model_type:
            project.model_type = infer_model_type(examples)
        validate_examples_for_model(examples, project.model_type)
        project.example_count = len(examples)
        save_finetune_project(root, project)
        job = prepare_finetune_job(root, project)
        console.print(f"[dim]Sizing {project.model_id} for {project.method}...[/]")
        resources = estimate_finetune_resources(project)
        config = build_finetune_run_config(base, root=root, project=project, job=job, resources=resources)
    except (FineTuneDataError, FileExistsError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_finetune_project(root, project)
    console.print(
        f"[cyan]Automatic resources:[/] {resources.vram_per_gpu_gb} GB VRAM/GPU, "
        f"{resources.host_ram_per_gpu_gb} GB RAM/GPU, {resources.disk_gb} GB disk; "
        f"effective batch {project.per_device_batch_size * project.gradient_accumulation_steps * project.gpu_count}."
    )
    if not project_exists:
        console.print(f"[green]Saved reusable fine-tune project to[/] {root / '.opbdh' / 'finetune.json'}")
    _execute_run(config, dry_run=dry_run, yes=yes)


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
    code: str = typer.Option("", "--code", "-f", help="Default code path. Supports {cwd}, {model_slug}, and env vars."),
    command: str = typer.Option("", "--command", "-x", help="Default remote command."),
    vram_gb: int = typer.Option(24, "--vram-gb", "-v"),
    gpu_count: int = typer.Option(1, "--gpu-count", "-g", min=1),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", "-d"),
    max_spend: float = typer.Option(5.0, "--max-spend", "-s"),
    auto_network_volume: bool = typer.Option(False, "--auto-network-volume/--no-auto-network-volume", "-a/-A"),
    network_volume_data_center_id: str = typer.Option("", "--network-volume-data-center-id", "-D"),
) -> None:
    cfg = OpbdhConfig(
        model_id=model,
        code=code,
        command=command,
        vram_gb=vram_gb,
        gpu_count=gpu_count,
        max_dollars_per_hour=max_dollars_per_hour,
        max_spend_dollars=max_spend,
        auto_network_volume=auto_network_volume,
        network_volume_data_center_id=network_volume_data_center_id,
    )
    path = save_config(cfg, output or global_config_path())
    console.print(f"[green]Wrote[/] {path}")


@config_app.command("wizard")
def config_wizard(
    scope: str = typer.Option("global", "--scope", "-s", help="global or local"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    hx.banner("configure your pod launcher")
    try:
        import questionary
    except Exception as exc:
        raise typer.BadParameter("questionary is required for the wizard; use `opbdh config write` instead.") from exc

    model = _questionary_model(questionary)
    model_estimate = estimate_model_size_gb(model)
    suggested_volume = suggested_network_volume_gb(model_estimate)
    code = questionary.text("Default local code path", default="{cwd}/run.py", style=hx.questionary_style()).ask() or ""
    command = questionary.text("Remote command override", default="", style=hx.questionary_style()).ask() or ""
    vram_gb = int(questionary.text("Minimum VRAM GB", default="24", style=hx.questionary_style()).ask() or "24")
    gpu_count = int(questionary.text("GPUs per pod", default="1", style=hx.questionary_style()).ask() or "1")
    hourly_text = questionary.text("Max dollars/hour estimate (blank for no cap)", default="", style=hx.questionary_style()).ask() or ""
    spend = float(questionary.text("Max spend dollars", default="5", style=hx.questionary_style()).ask() or "5")
    auto_volume = bool(questionary.confirm("Create a RunPod network volume when none is configured?", default=False, style=hx.questionary_style()).ask())
    data_center = ""
    if auto_volume:
        data_center = questionary.text("RunPod data center id for the volume", default="EU-RO-1", style=hx.questionary_style()).ask() or ""
        console.print(f"Suggested volume size for {model}: {suggested_volume} GB")
    cfg = OpbdhConfig(
        model_id=model,
        code=code,
        command=command,
        vram_gb=vram_gb,
        gpu_count=gpu_count,
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
    query = questionary.text("Search Hugging Face models", default=default, style=hx.questionary_style()).ask() or ""
    choices: list[str] = []
    if query.strip():
        try:
            choices = [
                option.model
                for option in _huggingface_model_options(query, limit=25)
                if option.model
            ]
        except Exception:
            choices = []
    if choices:
        selected = questionary.autocomplete("Model", choices=choices, default=choices[0], style=hx.questionary_style()).ask()
        if selected:
            return str(selected)
    return questionary.text("Model id", default=query, style=hx.questionary_style()).ask() or query


@models_app.command("search")
def models_search(query: str, limit: int = typer.Option(10, "--limit", "-n")) -> None:
    table = Table(title=f"Hugging Face models: {query}", title_style="bold #ef4444", border_style="grey37")
    table.add_column("Model")
    table.add_column("Details")
    for option in _huggingface_model_options(query, limit=limit):
        table.add_row(option.model, option.detail)
    console.print(table)


@models_app.command("estimate")
def models_estimate(
    model: str = typer.Argument(..., help="Hugging Face model id."),
    goal: str = typer.Option("inference", "--goal", "-g", help=f"One of: {', '.join(GOALS)}."),
    context: int | None = typer.Option(None, "--context", "-C", help="Context length. Defaults per goal."),
    batch: int = typer.Option(1, "--batch", "-b", help="Batch size (concurrent sequences)."),
    cloud_type: str = typer.Option("COMMUNITY", "--cloud-type", "-t", help="COMMUNITY or SECURE, for $/hr estimates."),
    json_out: bool = typer.Option(False, "--json", "-j", help="Emit machine-readable JSON instead of tables."),
) -> None:
    """Estimate VRAM, host RAM, and disk needed to run MODEL for a given goal."""
    try:
        estimate = estimate_for_model(model, goal, context_len=context, batch_size=batch)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    gpu_splits = []
    for num_gpus in (1, 2, 4, 8):
        needed = estimate.min_vram_gb(num_gpus)
        fits = candidate_gpus(needed, None, cloud_type)
        cheapest = min(fits, key=lambda gpu: gpu.hourly(cloud_type)) if fits else None
        gpu_splits.append(
            {
                "num_gpus": num_gpus,
                "min_vram_per_gpu_gb": needed,
                "cheapest_gpu": cheapest.id if cheapest else None,
                "estimated_dollars_per_hour": round(cheapest.hourly(cloud_type) * num_gpus, 2) if cheapest else None,
            }
        )

    if json_out:
        console.print_json(json.dumps({**asdict(estimate), "notes": list(estimate.notes), "gpu_splits": gpu_splits}))
        return

    table = Table(
        title=f"{model} — {estimate.goal} (context {estimate.context_len}, batch {estimate.batch_size})",
        title_style="bold #ef4444",
        border_style="grey37",
    )
    table.add_column("Component", style="cyan")
    table.add_column("Estimate", justify="right")
    table.add_row("Parameters", f"{estimate.param_count / 1e9:.2f}B")
    table.add_row("Weights", f"{estimate.weights_gb:.1f} GB")
    if estimate.kv_cache_gb:
        table.add_row("KV cache", f"{estimate.kv_cache_gb:.1f} GB")
    if estimate.activations_gb:
        table.add_row("Activations", f"{estimate.activations_gb:.1f} GB")
    if estimate.optimizer_gb:
        table.add_row("Grads + optimizer", f"{estimate.optimizer_gb:.1f} GB")
    table.add_row("[bold]Total VRAM[/]", f"[bold]{estimate.total_vram_gb:.1f} GB[/]")
    table.add_row("Host RAM", f"{estimate.host_ram_gb} GB")
    table.add_row("Disk / volume", f"{estimate.disk_gb} GB")
    console.print(table)

    fit_table = Table(
        title=f"GPU fit ({cloud_type.lower()} $/hr estimates)",
        title_style="bold #ef4444",
        border_style="grey37",
    )
    fit_table.add_column("GPUs", justify="right")
    fit_table.add_column("VRAM/GPU needed", justify="right")
    fit_table.add_column("Cheapest fit")
    fit_table.add_column("$/hr total", justify="right")
    for split in gpu_splits:
        fit_table.add_row(
            str(split["num_gpus"]),
            f"{split['min_vram_per_gpu_gb']} GB",
            split["cheapest_gpu"] or "[red]none in catalog[/]",
            f"{split['estimated_dollars_per_hour']:.2f}" if split["estimated_dollars_per_hour"] else "-",
        )
    console.print(fit_table)
    for note in estimate.notes:
        console.print(f"[dim]note: {note}[/]")


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
    vram_gb: int = typer.Option(24, "--vram-gb", "-v"),
    gpu_count: int = typer.Option(1, "--gpu-count", "-g", min=1),
    max_dollars_per_hour: float | None = typer.Option(None, "--max-dollars-per-hour", "-d"),
    cloud_type: str = typer.Option("SECURE", "--cloud-type", "-t"),
    provider: str = typer.Option("runpod", "--provider", "-p", help="Compute provider: runpod or primeintellect."),
) -> None:
    per_gpu_cap = (
        max_dollars_per_hour / gpu_count
        if max_dollars_per_hour is not None and max_dollars_per_hour > 0
        else max_dollars_per_hour
    )
    if provider.strip().lower() == "primeintellect":
        from .primeintellect import find_pi_offers, offer_hourly

        offers = find_pi_offers(
            min_vram_gb=vram_gb,
            max_dollars_per_hour=per_gpu_cap,
            cloud_type=cloud_type,
            gpu_count=gpu_count,
        )
        table = Table(title="Prime Intellect GPU offers (live)", title_style="bold #ef4444", border_style="grey37")
        table.add_column("GPU type")
        table.add_column("Provider")
        table.add_column("Region")
        table.add_column("VRAM", justify="right")
        table.add_column("$/hr", justify="right")
        table.add_column("Stock")
        for offer in offers:
            hourly = offer_hourly(offer)
            table.add_row(
                str(offer.get("gpuType", "?")),
                str(offer.get("provider", "?")),
                str(offer.get("dataCenter") or offer.get("region") or "?"),
                str(offer.get("gpuMemory", "?")),
                f"{hourly * gpu_count:.2f}" if hourly is not None else "?",
                str(offer.get("stockStatus", "?")),
            )
        console.print(table)
        return
    table = Table(title="OPBDH GPU candidates", title_style="bold #ef4444", border_style="grey37")
    table.add_column("RunPod GPU id")
    table.add_column("VRAM", justify="right")
    table.add_column("$/hr estimate", justify="right")
    for gpu in candidate_gpus(vram_gb, per_gpu_cap, cloud_type):
        table.add_row(gpu.id, str(gpu.memory_gb), f"{gpu.hourly(cloud_type) * gpu_count:.2f}")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
