"""OPBDH: a small RunPod launcher for model-backed scripts.

Library use:

    import opbdh

    result = opbdh.launch("./train", model="Qwen/Qwen2.5-7B-Instruct",
                          vram_gb=48, max_spend=5)

See :func:`opbdh.launch`, :func:`opbdh.plan`, and :mod:`opbdh.api`.
"""

from .execution import ExecutionTarget, launch_local, local_accelerators, plan_execution, require_local_capacity
from .api import (
    GOALS,
    InsufficientCreditsError,
    MaxSpendReached,
    OpbdhConfig,
    OpbdhPlan,
    OpbdhRunResult,
    RunEvent,
    RunpodBalance,
    collect_events,
    configure,
    estimate_memory,
    estimate_model_size,
    event_messages,
    gpu_options,
    launch,
    plan,
    runpod_balance,
    search_models,
    suggest_volume_gb,
    summarize,
    verify,
)

__all__ = [
    "__version__",
    # Running
    "launch",
    "launch_local",
    "plan_execution",
    "local_accelerators",
    "require_local_capacity",
    "ExecutionTarget",
    "plan",
    "summarize",
    "configure",
    "verify",
    "runpod_balance",
    # Sizing helpers
    "estimate_model_size",
    "estimate_memory",
    "suggest_volume_gb",
    "search_models",
    "gpu_options",
    # Types
    "OpbdhConfig",
    "OpbdhPlan",
    "OpbdhRunResult",
    "RunEvent",
    "RunpodBalance",
    "MaxSpendReached",
    "InsufficientCreditsError",
    "GOALS",
    # Event helpers
    "collect_events",
    "event_messages",
]

__version__ = "1.9.0"
