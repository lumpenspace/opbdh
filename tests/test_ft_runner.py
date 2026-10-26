"""Execute the shipped runner offline; no model downloads or GPU launches."""

import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opbdh.finetune import FineTuneExample, FineTuneProject, add_project_examples, prepare_finetune_job


@pytest.fixture
def run_job(tmp_path, monkeypatch):
    def run(*, method="lora", targets=None, checkpointing=True, cuda=False):
        project = FineTuneProject(model_id="offline/model", model_type="base", method=method)
        add_project_examples(tmp_path, project, [FineTuneExample("input", "output")])
        job = prepare_finetune_job(tmp_path, project)
        config_path = job.directory / "config.json"
        config = json.loads(config_path.read_text())
        if targets is None:  # Bundles generated before the field existed.
            config.pop("lora_target_modules")
        else:
            config["lora_target_modules"] = targets
        config_path.write_text(json.dumps(config))
        model = SimpleNamespace()
        if checkpointing is not None:
            model.supports_gradient_checkpointing = checkpointing
        tokenizer = SimpleNamespace(pad_token=None, eos_token="eos", save_pretrained=Mock())
        trainer = Mock(processing_class=tokenizer)
        trainer.train.return_value = SimpleNamespace(metrics={"loss": 1.0})
        trainer.is_world_process_zero.return_value = True
        peft = SimpleNamespace(LoraConfig=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
                               prepare_model_for_kbit_training=Mock(return_value=model))
        trl = SimpleNamespace(SFTConfig=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
                              SFTTrainer=Mock(return_value=trainer))
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda,
                is_bf16_supported=lambda: True, get_device_capability=lambda: (8, 0)),
                bfloat16="bf16", float16="fp16", float32="fp32"),
            "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
            "peft": peft,
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                BitsAndBytesConfig=Mock()),
            "trl": trl,
        }
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setenv("OPBDH_RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.delenv("OPBDH_PUSH_TO_HUB", raising=False)
        runpy.run_path(str(job.directory / "run.py"), run_name="__main__")
        trainer.train.assert_called_once()
        return peft, trl
    return run


@pytest.mark.parametrize("targets,expected", [
    (None, "all-linear"), ("", "all-linear"), (" ,  , ", "all-linear"),
    ("attn_query", ["attn_query"]),
    (" attn_query, attn_key, ,mlp_gate, ", ["attn_query", "attn_key", "mlp_gate"]),
])
def test_runner_passes_targets_to_peft(run_job, targets, expected):
    peft, trl = run_job(targets=targets)
    config = trl.SFTTrainer.call_args.kwargs["peft_config"]
    assert config.target_modules == expected
    assert (config.r, config.lora_alpha, config.lora_dropout) == (16, 32, 0.05)
    peft.LoraConfig.assert_called_once()


@pytest.mark.parametrize("method", ["lora", "qlora", "full"])
@pytest.mark.parametrize("support", [True, False, None])
def test_runner_checkpointing_follows_model_support(run_job, capsys, method, support):
    peft, trl = run_job(method=method, checkpointing=support, cuda=method == "qlora")
    args = trl.SFTTrainer.call_args.kwargs["args"]
    assert args.gradient_checkpointing is bool(support)
    assert args.gradient_checkpointing_kwargs == ({"use_reentrant": False} if support else None)
    assert ("expect higher memory use" in capsys.readouterr().out) is (not bool(support))
    if method == "qlora":
        assert peft.prepare_model_for_kbit_training.call_args.kwargs == {
            "use_gradient_checkpointing": bool(support),
        }
    else:
        peft.prepare_model_for_kbit_training.assert_not_called()
    if method == "full":
        peft.LoraConfig.assert_not_called()
        assert trl.SFTTrainer.call_args.kwargs["peft_config"] is None


def test_qlora_still_rejects_cpu(run_job):
    with pytest.raises(RuntimeError, match="QLoRA requires a CUDA GPU"):
        run_job(method="qlora", cuda=False)
