from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth")


@dataclass(frozen=True, slots=True)
class ModelSizeEstimate:
    model_id: str
    size_gb: float | None
    source: str


def _file_size_bytes(sibling: Any) -> int:
    value = getattr(sibling, "size", None)
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(sibling, dict):
        dict_value = sibling.get("size")
        if isinstance(dict_value, int) and dict_value > 0:
            return dict_value
    return 0


def _file_name(sibling: Any) -> str:
    value = getattr(sibling, "rfilename", None) or getattr(sibling, "filename", None)
    if isinstance(value, str):
        return value
    if isinstance(sibling, dict):
        for key in ("rfilename", "filename", "path"):
            dict_value = sibling.get(key)
            if isinstance(dict_value, str):
                return dict_value
    return ""


def estimate_model_size_gb(model_id: str, *, token: str | None = None, siblings: Iterable[Any] | None = None) -> ModelSizeEstimate:
    if siblings is None:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(model_id, token=token, files_metadata=True)
            siblings = getattr(info, "siblings", None) or []
        except Exception:
            return ModelSizeEstimate(model_id=model_id, size_gb=None, source="unavailable")

    total = 0
    for sibling in siblings:
        name = _file_name(sibling).lower()
        if name.endswith(MODEL_WEIGHT_SUFFIXES) and "optimizer" not in name:
            total += _file_size_bytes(sibling)

    if total <= 0:
        return ModelSizeEstimate(model_id=model_id, size_gb=None, source="metadata")
    return ModelSizeEstimate(model_id=model_id, size_gb=total / (1024**3), source="metadata")


def suggested_network_volume_gb(
    estimate: ModelSizeEstimate,
    *,
    multiplier: float = 2.5,
    minimum_gb: int = 50,
    fallback_gb: int = 160,
) -> int:
    if estimate.size_gb is None:
        return max(minimum_gb, fallback_gb)
    return max(minimum_gb, int(math.ceil(estimate.size_gb * multiplier)))
