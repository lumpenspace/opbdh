"""Programmatic API for opbdh.

The CLI is one front-end for this; here is the other:

    import opbdh

    result = opbdh.launch(
        "./train",
        model="Qwen/Qwen2.5-7B-Instruct",
        vram_gb=48,
        max_spend=5,
        on_event=lambda event: print(event.message),
    )
    print(result.outputs_dir)

Configuration layers exactly as it does for the CLI: keyword overrides beat
a local ``opbdh.json``, which beats ``~/.config/opbdh/config.json``. Any
:class:`~opbdh.config.OpbdhConfig` field may be passed as a keyword.

Unlike the CLI, these functions never prompt: a failed run cleans its pod up
rather than asking whether to keep it alive, and there is no launch
confirmation, so treat :func:`launch` as "yes, spend the money".
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from modelchoice import HuggingFaceSource, ModelCatalog, ModelOption

from .config import OpbdhConfig, load_config
from .estimate import GOALS, MemoryEstimate, estimate_for_model
from .gpu import GpuOffer, candidate_gpus
from .hf import ModelSizeEstimate, estimate_model_size_gb, suggested_network_volume_gb
from .runpod import (
    MaxSpendReached,
    OpbdhPlan,
    OpbdhRunResult,
    RunEvent,
    make_plan,
    plan_summary,
    run_plan,
)
from .verify import VerificationResult, verify_code

__all__ = [
    "GOALS",
    "MaxSpendReached",
    "OpbdhConfig",
    "OpbdhPlan",
    "OpbdhRunResult",
    "RunEvent",
    "collect_events",
    "configure",
    "estimate_memory",
    "estimate_model_size",
    "event_messages",
    "gpu_options",
    "launch",
    "plan",
    "search_models",
    "suggest_volume_gb",
    "summarize",
    "verify",
]

# `code` is accepted as a friendly alias for the config's `code` field, and
# these spellings mirror the CLI flags rather than the config field names.
_OVERRIDE_ALIASES = {
    "model": "model_id",
    "max_spend": "max_spend_dollars",
    "min_ram_per_gpu": "min_ram_per_gpu_gb",
}


def _overrides(code: str | Path | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    overrides = {_OVERRIDE_ALIASES.get(key, key): value for key, value in kwargs.items()}
    if code is not None:
        overrides["code"] = str(code)
    return {key: value for key, value in overrides.items() if value is not None and value != ""}


def configure(code: str | Path | None = None, *, config_file: str | Path | None = None, **overrides: Any) -> OpbdhConfig:
    """Resolve a config from the global file, a local ``opbdh.json``, and overrides.

    Keyword overrides are :class:`OpbdhConfig` field names, plus the CLI-style
    aliases ``model``, ``max_spend``, and ``min_ram_per_gpu``.
    """
    return load_config(
        local_config=Path(config_file).expanduser() if config_file else None,
        overrides=_overrides(code, overrides),
    )


def plan(
    code: str | Path | None = None,
    *,
    config_file: str | Path | None = None,
    run_id: str | None = None,
    **overrides: Any,
) -> OpbdhPlan:
    """Build a run plan without renting anything.

    Statically verifies the code, picks GPU candidates, and sizes any network
    volume. This may still call out to Hugging Face and the provider's
    catalogue, but it never creates a pod.
    """
    config = configure(code, config_file=config_file, **overrides)
    if not config.code:
        raise ValueError("a code path is required, either as an argument or config.code")
    return make_plan(config, code_path=Path(config.code), run_id=run_id)


def summarize(opbdh_plan: OpbdhPlan) -> dict[str, Any]:
    """A JSON-friendly summary of a plan, as shown by ``opbdh plan``."""
    return plan_summary(opbdh_plan)


def launch(
    code: str | Path | None = None,
    *,
    config_file: str | Path | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    progress: bool = False,
    on_event: Callable[[RunEvent], None] | None = None,
    **overrides: Any,
) -> OpbdhRunResult | None:
    """Plan and run in one call: rent a GPU pod, run the code, sync results home.

    This spends money without confirming. Returns None for a dry run,
    otherwise an :class:`OpbdhRunResult` whose ``outputs_dir`` holds whatever
    the remote job wrote to ``$OPBDH_RESULTS_DIR``.

    Raises :class:`MaxSpendReached` if the spend guard trips mid-run, and
    RuntimeError if the remote job exits non-zero (results synced so far are
    still on disk). The pod is always cleaned up.
    """
    return run_plan(
        plan(code, config_file=config_file, run_id=run_id, **overrides),
        dry_run=dry_run,
        progress=progress,
        interactive=False,
        on_event=on_event,
    )


def verify(code: str | Path, *, command: str = "") -> VerificationResult:
    """Statically check that `code` is runnable on a pod, without renting one."""
    return verify_code(Path(code).expanduser(), command=command)


def estimate_model_size(model_id: str, *, token: str | None = None) -> ModelSizeEstimate:
    """Total size of a Hugging Face model's weight files, in GB."""
    return estimate_model_size_gb(model_id, token=token)


def suggest_volume_gb(model_id_or_estimate: str | ModelSizeEstimate, **kwargs: Any) -> int:
    """Suggest a network volume size big enough to cache a model."""
    estimate = (
        model_id_or_estimate
        if isinstance(model_id_or_estimate, ModelSizeEstimate)
        else estimate_model_size_gb(model_id_or_estimate)
    )
    return suggested_network_volume_gb(estimate, **kwargs)


def estimate_memory(
    model_id: str,
    goal: str = "inference",
    *,
    context_len: int | None = None,
    batch_size: int = 1,
) -> MemoryEstimate:
    """Estimate VRAM/RAM/disk to run a model for a goal (see :data:`GOALS`)."""
    return estimate_for_model(model_id, goal, context_len=context_len, batch_size=batch_size)


def _huggingface_model_options(
    query: str, *, limit: int = 10, token: str | None = None
) -> list[ModelOption]:
    """Shared-catalog rows used by both the Python API and interactive CLI."""

    env = dict(os.environ)
    if token and token.strip():
        env["HF_TOKEN"] = token.strip()
    catalog = ModelCatalog([HuggingFaceSource(env=env)], env=env)
    return catalog.search_huggingface(query, limit=limit)


def search_models(query: str, *, limit: int = 10, token: str | None = None) -> list[str]:
    """Search Hugging Face for model ids matching `query`."""

    return [
        option.model
        for option in _huggingface_model_options(query, limit=limit, token=token)
        if option.model
    ]


def gpu_options(
    *,
    vram_gb: int = 24,
    max_dollars_per_hour: float | None = None,
    cloud_type: str = "SECURE",
) -> list[GpuOffer]:
    """GPUs in the catalogue meeting a VRAM floor and hourly price ceiling."""
    return candidate_gpus(vram_gb, max_dollars_per_hour, cloud_type)


def collect_events() -> tuple[list[RunEvent], Callable[[RunEvent], None]]:
    """A ready-made ``on_event`` sink plus the list it appends to.

        events, sink = opbdh.collect_events()
        opbdh.launch("./train", model=..., on_event=sink)
    """
    events: list[RunEvent] = []
    return events, events.append


def event_messages(events: Iterable[RunEvent], *, kind: str | None = None) -> list[str]:
    """Pull the messages out of collected events, optionally filtered by kind."""
    return [event.message for event in events if kind is None or event.kind == kind]
