"""VRAM / host RAM estimation from Hugging Face metadata, without downloading weights.

Parameter counts come from the safetensors headers (fetched over HTTP by
huggingface_hub), architecture details from config.json. Everything else is
arithmetic over well-known memory formulas; results are estimates, good to
roughly +/-15%, and intentionally lean conservative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .hf import estimate_model_size_gb

GIB = 1024**3

GOALS = ("inference", "lora", "qlora", "full")

# Bytes per element for safetensors dtype tags.
DTYPE_BYTES: dict[str, float] = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
    "F4": 0.5, "I4": 0.5, "U4": 0.5,
}

# Mixed-precision Adam full fine-tune: 2 weights + 2 grads + 8 optimizer states + 4 fp32 master copy.
FULL_FINETUNE_BYTES_PER_PARAM = 16
# LoRA trainable fraction (rank ~16 adapters over linear layers), each trained param costing the full 16 bytes.
LORA_TRAINABLE_FRACTION = 0.005
# NF4 weights plus quantization constants.
QLORA_BYTES_PER_PARAM = 0.55
# CUDA context, cublas workspaces, allocator fragmentation, NCCL buffers.
PER_GPU_OVERHEAD_GB = 2.0

DEFAULT_INFERENCE_CONTEXT = 8192
DEFAULT_TRAINING_CONTEXT = 2048


@dataclass(frozen=True, slots=True)
class ModelSpecs:
    model_id: str
    param_count: int
    weight_bytes: int
    hidden_size: int | None = None
    num_layers: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    max_context: int | None = None
    source: str = "safetensors"

    @property
    def stored_bytes_per_param(self) -> float:
        if self.param_count <= 0:
            return 2.0
        return self.weight_bytes / self.param_count

    @property
    def has_architecture(self) -> bool:
        return None not in (self.num_layers, self.num_kv_heads, self.head_dim)


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    model_id: str
    goal: str
    param_count: int
    context_len: int
    batch_size: int
    weights_gb: float
    kv_cache_gb: float
    activations_gb: float
    optimizer_gb: float
    total_vram_gb: float
    host_ram_gb: int
    disk_gb: int
    notes: tuple[str, ...] = ()

    def per_gpu_vram_gb(self, num_gpus: int) -> float:
        """Per-GPU need assuming near-even sharding (tensor parallel / FSDP)."""
        return self.total_vram_gb / max(1, num_gpus) + PER_GPU_OVERHEAD_GB

    def min_vram_gb(self, num_gpus: int = 1) -> int:
        return int(math.ceil(self.per_gpu_vram_gb(num_gpus)))


def _dig(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def specs_from_config(
    model_id: str,
    config: dict[str, Any] | None,
    *,
    param_count: int,
    weight_bytes: int,
    source: str = "safetensors",
) -> ModelSpecs:
    config = dict(config or {})
    # Multimodal repos nest the language model under text_config; that is the part that dominates memory.
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        config = {**config, **text_config}

    hidden = _dig(config, "hidden_size", "n_embd", "d_model")
    layers = _dig(config, "num_hidden_layers", "n_layer", "num_layers")
    attn_heads = _dig(config, "num_attention_heads", "n_head")
    kv_heads = _dig(config, "num_key_value_heads") or attn_heads
    head_dim = _dig(config, "head_dim")
    if head_dim is None and hidden and attn_heads:
        head_dim = hidden // attn_heads
    max_context = _dig(config, "max_position_embeddings", "n_positions", "max_seq_len")

    return ModelSpecs(
        model_id=model_id,
        param_count=param_count,
        weight_bytes=weight_bytes,
        hidden_size=hidden,
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        max_context=max_context,
        source=source,
    )


def _params_from_safetensors(model_id: str, token: str | None) -> tuple[int, int] | None:
    try:
        from huggingface_hub import get_safetensors_metadata
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        meta = get_safetensors_metadata(model_id, token=token)
    except Exception:
        return None
    counts = getattr(meta, "parameter_count", None) or {}
    params = sum(counts.values())
    if params <= 0:
        return None
    weight_bytes = int(sum(count * DTYPE_BYTES.get(dtype.upper(), 2) for dtype, count in counts.items()))
    return params, weight_bytes


def _fetch_config_json(model_id: str, token: str | None) -> dict[str, Any] | None:
    try:
        import json

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, "config.json", token=token)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def fetch_model_specs(model_id: str, *, token: str | None = None) -> ModelSpecs:
    """Resolve param count and architecture from the Hub. Never downloads weights.

    Falls back to total weight-file size (assuming 16-bit storage) when the
    repo has no parseable safetensors headers.
    """
    resolved = _params_from_safetensors(model_id, token)
    if resolved is not None:
        params, weight_bytes = resolved
        source = "safetensors"
    else:
        size = estimate_model_size_gb(model_id, token=token)
        if size.size_gb is None:
            raise RuntimeError(
                f"could not determine size of {model_id!r}: no safetensors metadata and no weight files found"
            )
        weight_bytes = int(size.size_gb * GIB)
        params = weight_bytes // 2
        source = "file-size"
    return specs_from_config(
        model_id,
        _fetch_config_json(model_id, token),
        param_count=params,
        weight_bytes=weight_bytes,
        source=source,
    )


def _kv_cache_gb(specs: ModelSpecs, context_len: int, batch_size: int) -> float:
    if not specs.has_architecture:
        return 0.0
    kv_bytes = 2  # K and V cached in 16-bit
    total = 2 * specs.num_layers * specs.num_kv_heads * specs.head_dim * context_len * batch_size * kv_bytes
    return total / GIB


def _activations_gb(specs: ModelSpecs, context_len: int, batch_size: int) -> float:
    """Rough training-activation peak with gradient checkpointing enabled."""
    if specs.hidden_size is None or specs.num_layers is None:
        # No architecture info: scale with params instead, ~2 bytes/param is a mid-range guess.
        return specs.param_count * 2 / GIB * 0.15
    per_token = specs.hidden_size * specs.num_layers * 2 * 4  # 16-bit, ~4 tensors per layer live at peak
    return per_token * context_len * batch_size / GIB


def estimate_memory(
    specs: ModelSpecs,
    goal: str = "inference",
    *,
    context_len: int | None = None,
    batch_size: int = 1,
) -> MemoryEstimate:
    goal = goal.strip().lower()
    if goal not in GOALS:
        raise ValueError(f"unsupported goal {goal!r}; expected one of: {', '.join(GOALS)}")

    if context_len is None:
        context_len = DEFAULT_INFERENCE_CONTEXT if goal == "inference" else DEFAULT_TRAINING_CONTEXT
        if specs.max_context:
            context_len = min(context_len, specs.max_context)

    params = specs.param_count
    checkpoint_gb = specs.weight_bytes / GIB
    # Training casts fp32 checkpoints down to 16-bit before touching the GPU.
    compute_bytes_per_param = min(specs.stored_bytes_per_param, 2.0)
    notes: list[str] = []
    if specs.source != "safetensors":
        notes.append("parameter count inferred from weight-file size (no safetensors metadata)")
    if not specs.has_architecture:
        notes.append("architecture unknown: KV cache and activations are rough scalar guesses")

    kv_gb = 0.0
    activations_gb = 0.0
    optimizer_gb = 0.0

    if goal == "inference":
        weights_gb = checkpoint_gb
        kv_gb = _kv_cache_gb(specs, context_len, batch_size)
        overhead_gb = max(1.0, 0.1 * weights_gb)
    elif goal == "full":
        weights_gb = params * compute_bytes_per_param / GIB
        optimizer_gb = params * (FULL_FINETUNE_BYTES_PER_PARAM - compute_bytes_per_param) / GIB
        activations_gb = _activations_gb(specs, context_len, batch_size)
        overhead_gb = 0.1 * (weights_gb + optimizer_gb)
    elif goal == "lora":
        weights_gb = params * compute_bytes_per_param / GIB
        optimizer_gb = params * LORA_TRAINABLE_FRACTION * FULL_FINETUNE_BYTES_PER_PARAM / GIB
        activations_gb = _activations_gb(specs, context_len, batch_size)
        overhead_gb = max(1.0, 0.1 * weights_gb)
    else:  # qlora
        weights_gb = params * QLORA_BYTES_PER_PARAM / GIB
        optimizer_gb = params * LORA_TRAINABLE_FRACTION * FULL_FINETUNE_BYTES_PER_PARAM / GIB
        activations_gb = _activations_gb(specs, context_len, batch_size)
        overhead_gb = max(1.0, 0.15 * weights_gb)
        notes.append("assumes 4-bit NF4 quantization at load time (bitsandbytes)")

    total = weights_gb + kv_gb + activations_gb + optimizer_gb + overhead_gb

    # Host RAM mostly matters for loading/converting the checkpoint; training
    # additionally stages optimizer state during save.
    ram_factor = 2.0 if goal == "full" else 1.5
    host_ram_gb = max(16, int(math.ceil(checkpoint_gb * ram_factor)))

    # Disk: weights + tokenizer + HF cache double-write; training runs also write checkpoints.
    disk_factor = 4.0 if goal in ("full", "lora") else 2.5
    disk_gb = max(50, int(math.ceil(checkpoint_gb * disk_factor)))

    return MemoryEstimate(
        model_id=specs.model_id,
        goal=goal,
        param_count=params,
        context_len=context_len,
        batch_size=batch_size,
        weights_gb=round(weights_gb, 2),
        kv_cache_gb=round(kv_gb, 2),
        activations_gb=round(activations_gb, 2),
        optimizer_gb=round(optimizer_gb, 2),
        total_vram_gb=round(total, 2),
        host_ram_gb=host_ram_gb,
        disk_gb=disk_gb,
        notes=tuple(notes),
    )


def estimate_for_model(
    model_id: str,
    goal: str = "inference",
    *,
    context_len: int | None = None,
    batch_size: int = 1,
    token: str | None = None,
) -> MemoryEstimate:
    """One-call entry point for the orchestrator: model id + goal -> memory needs."""
    specs = fetch_model_specs(model_id, token=token)
    return estimate_memory(specs, goal, context_len=context_len, batch_size=batch_size)
