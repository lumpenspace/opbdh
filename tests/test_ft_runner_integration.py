"""CPU-only regression with the real training stack (optional outside CI).

Install the generated job's requirements plus torch to run these tests.
Models and tokenizer are tiny, randomly initialized, and entirely local.
"""

import json
import runpy

import pytest

from opbdh.finetune import FineTuneExample, FineTuneProject, add_project_examples, prepare_finetune_job

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")
tokenizers = pytest.importorskip("tokenizers")


class ParameterHeadModel(transformers.PreTrainedModel, transformers.GenerationMixin):
    """Talkie-like output head: a Parameter absent from named_modules()."""

    config_class = transformers.PreTrainedConfig
    supports_gradient_checkpointing = False

    def __init__(self):
        super().__init__(self.config_class(vocab_size=8, hidden_size=8, tie_word_embeddings=False))
        self.embed_tokens = torch.nn.Embedding(8, 8)
        self.attn_query = torch.nn.Linear(8, 8)
        self.attn_key = torch.nn.Linear(8, 8)
        self.mlp_gate = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Parameter(torch.randn(8, 8))

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = self.mlp_gate(self.attn_query(hidden) + self.attn_key(hidden))
        logits = torch.nn.functional.linear(hidden, self.lm_head)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1), ignore_index=-100,
            )
        return transformers.modeling_outputs.CausalLMOutput(loss=loss, logits=logits)


def test_parameter_output_head_reproduces_all_linear_failure():
    with pytest.raises(StopIteration):
        peft.get_peft_model(ParameterHeadModel(), peft.LoraConfig(target_modules="all-linear", task_type="CAUSAL_LM"))


@pytest.mark.parametrize("custom", [True, False])
def test_generated_runner_trains_and_saves_real_adapter_on_cpu(tmp_path, monkeypatch, custom):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OPBDH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("OPBDH_PUSH_TO_HUB", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = ParameterHeadModel() if custom else transformers.LlamaForCausalLM(
        transformers.LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            pad_token_id=0, eos_token_id=1, tie_word_embeddings=False)
    )
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "input": 3, "output": 4}, unk_token="[UNK]",
    ))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
    )
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    original_config = trl.SFTConfig
    original_trainer = trl.SFTTrainer
    seen = {}

    def cpu_config(**kwargs):
        seen["args"] = original_config(**kwargs, use_cpu=True, max_steps=1, disable_tqdm=True)
        return seen["args"]

    def trainer(**kwargs):
        seen["peft_config"] = kwargs["peft_config"]
        seen["trainer"] = original_trainer(**kwargs)
        return seen["trainer"]

    monkeypatch.setattr(trl, "SFTConfig", cpu_config)
    monkeypatch.setattr(trl, "SFTTrainer", trainer)
    project = FineTuneProject(model_id="offline/model", model_type="base", max_length=16,
        gradient_accumulation_steps=1, lora_target_modules=" attn_query, attn_key, " if custom else "")
    add_project_examples(tmp_path, project, [FineTuneExample("input ", "output")])
    job = prepare_finetune_job(tmp_path, project)
    if not custom:  # Old bundles must retain all-linear and checkpointing.
        path = job.directory / "config.json"
        config = json.loads(path.read_text())
        config.pop("lora_target_modules")
        path.write_text(json.dumps(config))
    runpy.run_path(str(job.directory / "run.py"), run_name="__main__")
    assert seen["args"].gradient_checkpointing is (not custom)
    assert seen["trainer"].state.global_step == 1
    trained = seen["trainer"].model
    if custom:
        assert hasattr(model.attn_query, "lora_A") and hasattr(model.attn_key, "lora_A")
        assert not hasattr(model.mlp_gate, "lora_A")
        assert not model.lm_head.requires_grad
    else:
        assert hasattr(model.model.layers[0].self_attn.q_proj, "lora_A")
        assert not hasattr(model.lm_head, "lora_A")
    assert any("lora_" in name and param.requires_grad for name, param in trained.named_parameters())
    assert (tmp_path / "results/model/adapter_config.json").is_file()
    summary = json.loads((tmp_path / "results/training.json").read_text())
    assert summary["metrics"]["train_loss"] > 0


def test_invalid_explicit_target_is_not_silently_replaced():
    with pytest.raises(ValueError, match="not found"):
        peft.get_peft_model(ParameterHeadModel(), peft.LoraConfig(target_modules=["missing_projection"]))
