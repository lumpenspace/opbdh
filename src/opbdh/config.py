from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_RUNPOD_CONTAINER_DISK_GB = 120
DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_RUNPOD_VOLUME_GB = 160
DEFAULT_RUNPOD_MIN_VCPU_PER_GPU = 4
DEFAULT_RUNPOD_MIN_RAM_PER_GPU_GB = 24
LOCAL_CONFIG_NAMES = ("opbdh.json", ".opbdh.json")


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


SUPPORTED_PROVIDERS = ("runpod", "primeintellect")


@dataclass(slots=True)
class OpbdhConfig:
    model_id: str = ""
    code: str = ""
    command: str = ""
    provider: str = "runpod"
    image: str = DEFAULT_RUNPOD_IMAGE
    cloud_type: str = "SECURE"
    vram_gb: int = 24
    max_dollars_per_hour: float | None = None
    max_spend_dollars: float = 5.0
    container_disk_gb: int = DEFAULT_RUNPOD_CONTAINER_DISK_GB
    pod_volume_gb: int = DEFAULT_RUNPOD_VOLUME_GB
    min_vcpu_per_gpu: int = DEFAULT_RUNPOD_MIN_VCPU_PER_GPU
    min_ram_per_gpu_gb: int = DEFAULT_RUNPOD_MIN_RAM_PER_GPU_GB
    network_volume_id: str = ""
    auto_network_volume: bool = False
    network_volume_data_center_id: str = ""
    network_volume_name: str = "opbdh-{model_slug}"
    network_volume_size_gb: int | None = None
    pre_download_model: bool = True
    results_dir: str = "runpod_results"
    poll_seconds: int = 20
    failure_keepalive_seconds: int = 120
    keep_pod_on_success: bool = False
    # Empty means auto: discover a standard ~/.ssh key, or generate a dedicated
    # opbdh keypair under the config dir when none exists.
    ssh_key: str = ""
    ssh_public_key: str = ""


def normalized_provider(config: OpbdhConfig) -> str:
    provider = config.provider.strip().lower() or "runpod"
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported provider {config.provider!r}; expected one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return provider


def config_dir() -> Path:
    explicit = os.environ.get("OPBDH_CONFIG_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser() / "opbdh").resolve()
    return (Path.home() / ".config" / "opbdh").resolve()


def global_config_path() -> Path:
    explicit = os.environ.get("OPBDH_CONFIG", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return config_dir() / "config.json"


def discover_local_config(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for root in [current, *current.parents]:
        for name in LOCAL_CONFIG_NAMES:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _known_fields() -> set[str]:
    return {field.name for field in fields(OpbdhConfig)}


def _coerce_config(data: dict[str, Any]) -> OpbdhConfig:
    known = _known_fields()
    payload = {key: value for key, value in data.items() if key in known and value is not None}
    return OpbdhConfig(**payload)


def merge_config(*layers: dict[str, Any]) -> OpbdhConfig:
    merged: dict[str, Any] = asdict(OpbdhConfig())
    known = _known_fields()
    for layer in layers:
        for key, value in layer.items():
            if key in known and value is not None:
                merged[key] = value
    return _coerce_config(merged)


def save_config(config: OpbdhConfig, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def model_slug(model_id: str) -> str:
    slug = model_id.strip().lower().replace("/", "-")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in slug).strip("-") or "model"


def interpolation_context(config: OpbdhConfig, *, cwd: Path | None = None, run_id: str | None = None) -> dict[str, str]:
    now = datetime.now(UTC)
    effective_run_id = run_id or now.strftime("%Y%m%d-%H%M%S")
    effective_cwd = (cwd or Path.cwd()).expanduser().resolve()
    return {
        "config_dir": str(config_dir()),
        "cwd": str(effective_cwd),
        "model_id": config.model_id,
        "model_slug": model_slug(config.model_id),
        "run_id": effective_run_id,
        "timestamp": effective_run_id,
    }


def interpolate_value(value: str, context: dict[str, str]) -> str:
    return os.path.expandvars(value).format_map(_SafeFormatDict(context))


def interpolate_config(config: OpbdhConfig, *, cwd: Path | None = None, run_id: str | None = None) -> OpbdhConfig:
    context = interpolation_context(config, cwd=cwd, run_id=run_id)
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, str):
            data[key] = interpolate_value(value, context)
    return _coerce_config(data)


def load_config(
    *,
    local_config: Path | None = None,
    overrides: dict[str, Any] | None = None,
    cwd: Path | None = None,
    run_id: str | None = None,
) -> OpbdhConfig:
    discovered = local_config if local_config is not None else discover_local_config(cwd)
    config = merge_config(
        _read_json_object(global_config_path()),
        _read_json_object(discovered),
        overrides or {},
    )
    return interpolate_config(config, cwd=cwd, run_id=run_id)
