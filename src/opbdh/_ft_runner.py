"""Remote SFT entry point copied into an OPBDH fine-tuning bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no training examples")
    return rows


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    here = Path(__file__).resolve().parent
    config = json.loads((here / "config.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(here / "dataset.jsonl")

    results_root = Path(os.environ.get("OPBDH_RESULTS_DIR", str(here / "results"))).resolve()
    checkpoints_dir = results_root / "checkpoints"
    model_dir = results_root / "model"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(has_cuda and not use_bf16)
    use_tf32 = bool(has_cuda and torch.cuda.get_device_capability()[0] >= 8)
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    method = str(config["method"])
    if method == "qlora" and not has_cuda:
        raise RuntimeError("QLoRA requires a CUDA GPU")

    peft_config = None
    if method in {"lora", "qlora"}:
        peft_config = LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    quantization_config = None
    if method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": bool(config["trust_remote_code"]),
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    model = AutoModelForCausalLM.from_pretrained(str(config["model_id"]), **model_kwargs)
    processing_class = AutoTokenizer.from_pretrained(
        str(config["model_id"]),
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token
    if method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    training_args = SFTConfig(
        output_dir=str(checkpoints_dir),
        num_train_epochs=float(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        max_length=int(config["max_length"]),
        completion_only_loss=True,
        packing=bool(config["packing"]),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_tf32,
        report_to="none",
        logging_steps=1,
        save_strategy="no",
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        ddp_find_unused_parameters=False if world_size > 1 else None,
        chat_template_path=str(config.get("chat_template") or "") or None,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(rows),
        processing_class=processing_class,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(model_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(model_dir)
    trainer.save_metrics("train", result.metrics)

    if trainer.is_world_process_zero():
        summary = {
            "recipe": config.get("recipe", "default"),
            "model_id": config["model_id"],
            "model_type": config["model_type"],
            "method": method,
            "example_count": len(rows),
            "selected_tags": config.get("selected_tags", []),
            "world_size": world_size,
            "effective_batch_size": (
                int(config["per_device_batch_size"])
                * int(config["gradient_accumulation_steps"])
                * world_size
            ),
            "metrics": _json_value(result.metrics),
        }
        (results_root / "training.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
