"""Compute placement shared by clients running interactive model workloads.

Local accelerators are opt-in, capacity-checked and run in the caller's Python
environment. Cloud execution continues to use opbdh.launch and its spend cap.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class ExecutionTarget:
    target: str
    device: str
    available_gb: float
    required_gb: float
    sufficient: bool


def local_accelerators(*, torch_module=None) -> dict[str, float]:
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return {}
    result = {}
    if torch_module.cuda.is_available():
        # A single-model device map cannot pool independent GPUs' memory.
        result["cuda"] = float(torch_module.cuda.mem_get_info(0)[0]) / 2**30
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        mps = torch_module.mps
        if hasattr(mps, "recommended_max_memory"):
            result["mps"] = max(0., float(mps.recommended_max_memory() - mps.driver_allocated_memory()) / 2**30)
    return result


def plan_execution(target: str = "runpod", *, required_gb: float = 80, capacities=None) -> ExecutionTarget:
    if not math.isfinite(required_gb) or required_gb <= 0:
        raise ValueError("required memory must be finite and positive")
    if target == "runpod":
        return ExecutionTarget(target, "cuda", required_gb, required_gb, True)
    if target not in {"cuda", "mps"}:
        raise ValueError("target must be runpod, cuda or mps")
    capacities = local_accelerators() if capacities is None else capacities
    available = float(capacities.get(target, 0))
    return ExecutionTarget(target, target, available, required_gb, available >= required_gb)


def require_local_capacity(target: str, *, required_gb: float, capacities=None) -> ExecutionTarget:
    if target == "runpod":
        raise ValueError("use opbdh.launch for cloud execution")
    plan = plan_execution(target, required_gb=required_gb, capacities=capacities)
    if not plan.sufficient:
        raise ValueError(f"{target} has {plan.available_gb:.1f} GiB available; this workload requires {required_gb:.1f} GiB")
    return plan


def launch_local(argv: list[str], *, target: str, required_gb: float, cwd: str | Path = ".", env=None):
    """Run an explicit argv without a shell after checking local capacity."""
    require_local_capacity(target, required_gb=required_gb)
    if not argv or any(not isinstance(arg, str) for arg in argv):
        raise ValueError("argv must be a nonempty string list")
    environment = {**os.environ, **(env or {}), "OPBDH_DEVICE": target}
    return subprocess.run(argv, cwd=cwd, env=environment, check=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run a local accelerator workload with an opbdh capacity check")
    parser.add_argument("--target", choices=("cuda", "mps"), required=True)
    parser.add_argument("--required-gb", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    launch_local(command or [sys.executable, "--version"], target=args.target, required_gb=args.required_gb)


if __name__ == "__main__":
    main()
