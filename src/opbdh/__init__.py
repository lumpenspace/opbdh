"""OPBDH: a small RunPod launcher for model-backed scripts.

Library use:

    import opbdh

    result = opbdh.launch("./train", model="Qwen/Qwen2.5-7B-Instruct",
                          vram_gb=48, max_spend=5)

See :func:`opbdh.launch`, :func:`opbdh.plan`, and :mod:`opbdh.api`.
"""

from .api import (
    GOALS,
    MaxSpendReached,
    OpbdhConfig,
    OpbdhPlan,
    OpbdhRunResult,
    RunEvent,
    collect_events,
    configure,
    estimate_memory,
    estimate_model_size,
    event_messages,
    gpu_options,
    launch,
    plan,
    search_models,
    suggest_volume_gb,
    summarize,
    verify,
)

__all__ = [
    "__version__",
    # Running
    "launch",
    "plan",
    "summarize",
    "configure",
    "verify",
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
    "MaxSpendReached",
    "GOALS",
    # Event helpers
    "collect_events",
    "event_messages",
]

__version__ = "1.4.0"
