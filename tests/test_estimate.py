from __future__ import annotations

import pytest

from opbdh.estimate import (
    GIB,
    GOALS,
    ModelSpecs,
    estimate_memory,
    specs_from_config,
)


def _specs(**overrides: object) -> ModelSpecs:
    base = dict(
        model_id="Org/Test-1B",
        param_count=1_000_000_000,
        weight_bytes=2_000_000_000,
        hidden_size=2048,
        num_layers=24,
        num_kv_heads=8,
        head_dim=128,
        max_context=32768,
    )
    base.update(overrides)
    return ModelSpecs(**base)  # type: ignore[arg-type]


def test_specs_from_config_parses_architecture_and_derives_head_dim() -> None:
    specs = specs_from_config(
        "Org/Model",
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
        },
        param_count=7_000_000_000,
        weight_bytes=14_000_000_000,
    )

    assert specs.head_dim == 128
    assert specs.num_kv_heads == 8
    assert specs.max_context == 131072
    assert specs.has_architecture


def test_specs_from_config_falls_back_to_attention_heads_and_text_config() -> None:
    specs = specs_from_config(
        "Org/Multimodal",
        {
            "vision_config": {"hidden_size": 1024},
            "text_config": {"hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 16},
        },
        param_count=1,
        weight_bytes=2,
    )

    assert specs.hidden_size == 2048
    assert specs.num_kv_heads == 16  # no num_key_value_heads: fall back to attention heads
    assert specs.head_dim == 128


def test_inference_estimate_counts_weights_and_kv_cache() -> None:
    estimate = estimate_memory(_specs(), "inference", context_len=8192, batch_size=1)

    # 2 (K and V) x 24 layers x 8 kv heads x 128 head dim x 8192 ctx x 2 bytes = 0.75 GiB exactly.
    assert estimate.kv_cache_gb == 0.75
    assert estimate.weights_gb == round(2_000_000_000 / GIB, 2)
    assert estimate.optimizer_gb == 0
    assert estimate.total_vram_gb > estimate.weights_gb + estimate.kv_cache_gb


def test_goal_memory_ordering() -> None:
    specs = _specs()
    totals = {goal: estimate_memory(specs, goal, context_len=2048).total_vram_gb for goal in GOALS}

    assert totals["full"] > totals["lora"] > totals["qlora"]
    assert totals["full"] > totals["inference"]
    # Full fine-tune with Adam costs ~16 bytes/param; sanity-check the scale.
    assert totals["full"] > 16_000_000_000 / GIB * 0.9


def test_default_context_is_clamped_to_model_max() -> None:
    estimate = estimate_memory(_specs(max_context=1024), "inference")

    assert estimate.context_len == 1024


def test_per_gpu_split_shrinks_with_more_gpus() -> None:
    estimate = estimate_memory(_specs(), "full")

    assert estimate.min_vram_gb(1) > estimate.min_vram_gb(2) > estimate.min_vram_gb(8)
    # Sharding still pays a fixed per-GPU overhead, so 2 GPUs need more than half.
    assert estimate.per_gpu_vram_gb(2) > estimate.total_vram_gb / 2


def test_unknown_architecture_yields_note_and_no_kv() -> None:
    specs = _specs(hidden_size=None, num_layers=None, num_kv_heads=None, head_dim=None, source="file-size")
    estimate = estimate_memory(specs, "inference")

    assert estimate.kv_cache_gb == 0
    assert any("architecture unknown" in note for note in estimate.notes)
    assert any("file size" in note.replace("-", " ") for note in estimate.notes)


def test_fp32_checkpoint_is_cast_down_for_training() -> None:
    fp32 = estimate_memory(_specs(param_count=7_000_000_000, weight_bytes=28_000_000_000), "full", context_len=2048)
    bf16 = estimate_memory(_specs(param_count=7_000_000_000, weight_bytes=14_000_000_000), "full", context_len=2048)

    # Same param count means the same training footprint regardless of stored dtype.
    assert fp32.weights_gb == bf16.weights_gb
    assert fp32.optimizer_gb == bf16.optimizer_gb
    # But the fp32 checkpoint is bigger on disk and in host RAM.
    assert fp32.host_ram_gb > bf16.host_ram_gb


def test_rejects_unknown_goal() -> None:
    with pytest.raises(ValueError, match="unsupported goal"):
        estimate_memory(_specs(), "speedrun")
