from __future__ import annotations

import ast
import json
import io
import py_compile
import tarfile
from pathlib import Path

import pytest

from opbdh import finetune
from opbdh.config import OpbdhConfig
from opbdh.estimate import MemoryEstimate
from opbdh.finetune import (
    FineTuneDataError,
    FineTuneExample,
    FineTuneProject,
    activate_finetune_recipe,
    add_project_examples,
    available_tags,
    build_finetune_run_config,
    create_finetune_recipe,
    estimate_finetune_resources,
    infer_model_type,
    load_finetune_project,
    prepare_finetune_job,
    read_examples,
    read_project_examples,
    recipe_names,
    replace_project_examples,
    save_finetune_project,
    select_examples,
    write_examples,
)
from opbdh.runpod import build_bundle


def _estimate() -> MemoryEstimate:
    return MemoryEstimate(
        model_id="Org/Model",
        goal="lora",
        param_count=7_000_000_000,
        context_len=2048,
        batch_size=1,
        weights_gb=14.0,
        kv_cache_gb=0.0,
        activations_gb=5.0,
        optimizer_gb=1.0,
        total_vram_gb=22.0,
        host_ram_gb=48,
        disk_gb=60,
    )


def test_reads_one_base_example_from_simple_toml(tmp_path: Path) -> None:
    path = tmp_path / "example.toml"
    path.write_text(
        '''input = "Translate hello to Italian"
output = "Ciao"
tags = ["translation", "short"]
''',
        encoding="utf-8",
    )

    examples = read_examples([path])

    assert examples == [FineTuneExample("Translate hello to Italian", "Ciao", ("translation", "short"))]
    assert infer_model_type(examples) == "base"


def test_reads_one_chat_example_from_simple_toml(tmp_path: Path) -> None:
    path = tmp_path / "chat.toml"
    path.write_text(
        '''output = "Ciao!"
tags = ["greeting"]

[[input]]
role = "system"
content = "Reply in Italian."

[[input]]
role = "user"
content = "Hello"
''',
        encoding="utf-8",
    )

    examples = read_examples([path])

    assert examples[0].input == [
        {"role": "system", "content": "Reply in Italian."},
        {"role": "user", "content": "Hello"},
    ]
    assert examples[0].output == "Ciao!"
    assert infer_model_type(examples) == "chat"


@pytest.mark.parametrize(
    "examples",
    [
        [
            FineTuneExample("one\nline two", "first", ("a",)),
            FineTuneExample("second", "two", ("b",)),
        ],
        [
            FineTuneExample([{"role": "user", "content": "one"}], "first", ("a",)),
            FineTuneExample([{"role": "user", "content": "two"}], "second", ("b",)),
        ],
    ],
)
def test_toml_writer_round_trips_multiple_examples(tmp_path: Path, examples: list[FineTuneExample]) -> None:
    path = write_examples(tmp_path / "dataset.toml", examples)

    assert "[[examples]]" in path.read_text(encoding="utf-8")
    assert read_examples([path], source_format="opbdh") == examples


def test_reads_objects_arrays_jsonl_and_multiple_files(tmp_path: Path) -> None:
    one = tmp_path / "one.json"
    many = tmp_path / "many.json"
    lines = tmp_path / "lines.jsonl"
    one.write_text(json.dumps({"input": "one", "output": "1"}), encoding="utf-8")
    many.write_text(
        json.dumps([{"input": "two", "output": "2"}, {"input": "three", "output": "3"}]),
        encoding="utf-8",
    )
    lines.write_text(
        json.dumps({"input": "four", "output": "4"})
        + "\n"
        + json.dumps({"input": "five", "output": "5"})
        + "\n",
        encoding="utf-8",
    )

    examples = read_examples([one, many, lines])

    assert [example.output for example in examples] == ["1", "2", "3", "4", "5"]


def test_imports_openai_chat_and_prompt_completion(tmp_path: Path) -> None:
    path = tmp_path / "openai.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        )
        + "\n"
        + json.dumps({"prompt": "Question: 2+2?", "completion": "4"})
        + "\n",
        encoding="utf-8",
    )

    examples = read_examples([path], source_format="openai", extra_tags=["imported"])

    assert examples[0].input == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    assert examples[0].output == "Hello"
    assert examples[1].input == "Question: 2+2?"
    assert all(example.tags == ("imported",) for example in examples)


def test_imports_anthropic_messages_content_blocks_and_legacy_prompt(tmp_path: Path) -> None:
    modern = tmp_path / "modern.json"
    legacy = tmp_path / "legacy.json"
    modern.write_text(
        json.dumps(
            {
                "system": "Be useful.",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
                "output": [{"type": "text", "text": "Hello"}],
            }
        ),
        encoding="utf-8",
    )
    legacy.write_text(
        json.dumps({"prompt": "\n\nHuman: Hi\n\nAssistant:", "completion": "Hello"}),
        encoding="utf-8",
    )

    examples = read_examples([modern, legacy], source_format="anthropic")

    assert examples[0].input[0] == {"role": "system", "content": "Be useful."}  # type: ignore[index]
    assert examples[0].output == "Hello"
    assert examples[1].input == [{"role": "user", "content": "Hi"}]


def test_tag_groups_select_examples_with_any_matching_tag() -> None:
    examples = [
        FineTuneExample("a", "A", ("alpha",)),
        FineTuneExample("b", "B", ("beta", "shared")),
        FineTuneExample("c", "C"),
    ]

    assert available_tags(examples) == ["alpha", "beta", "shared"]
    assert [example.output for example in select_examples(examples, ["shared", "alpha"])] == ["A", "B"]
    assert select_examples(examples, []) == examples


def test_retagging_replaces_managed_examples_and_keeps_valid_selected_groups(tmp_path: Path) -> None:
    project = FineTuneProject(model_id="Org/Base", model_type="base", selected_tags=["keep", "removed"])
    add_project_examples(
        tmp_path,
        project,
        [FineTuneExample("one", "1", ("keep",)), FineTuneExample("two", "2", ("removed",))],
    )
    create_finetune_recipe(project, "second", clone_current=True)

    updated = replace_project_examples(
        tmp_path,
        project,
        [FineTuneExample("one", "1", ("keep", "new")), FineTuneExample("two", "2")],
    )

    assert read_project_examples(tmp_path, project) == updated
    assert project.selected_tags == ["keep"]
    assert all(recipe["selected_tags"] == ["keep"] for recipe in project.recipes.values())
    assert available_tags(updated) == ["keep", "new"]


def test_same_dataset_supports_multiple_method_specific_recipes(tmp_path: Path) -> None:
    project = FineTuneProject(
        model_id="Org/Base",
        model_type="base",
        selected_tags=["train"],
    )
    examples = [FineTuneExample("one", "1", ("train",)), FineTuneExample("two", "2")]
    add_project_examples(tmp_path, project, examples)

    create_finetune_recipe(project, "full-2gpu", method="full")
    project.gpu_count = 2
    project.epochs = 1.5
    save_finetune_project(tmp_path, project)

    activate_finetune_recipe(project, "default")
    project.learning_rate = 1e-4
    save_finetune_project(tmp_path, project)

    loaded = load_finetune_project(tmp_path)
    assert loaded is not None
    assert recipe_names(loaded) == ["default", "full-2gpu"]
    assert loaded.active_recipe == "default"
    assert loaded.method == "lora"
    assert loaded.learning_rate == 1e-4
    assert loaded.selected_tags == ["train"]
    assert read_project_examples(tmp_path, loaded) == examples

    activate_finetune_recipe(loaded, "full-2gpu")
    assert loaded.method == "full"
    assert loaded.learning_rate == 2e-5
    assert loaded.gpu_count == 2
    assert loaded.epochs == 1.5
    assert loaded.selected_tags == []
    job = prepare_finetune_job(tmp_path, loaded)
    assert job.directory.name == "full-2gpu"
    assert json.loads((job.directory / "config.json").read_text())["recipe"] == "full-2gpu"


def test_lora_target_modules_reach_the_runner_and_survive_a_reload(tmp_path: Path) -> None:
    # An architecture peft's "all-linear" shorthand cannot walk (its lm_head is
    # a bare nn.Parameter) needs its LoRA targets named outright.
    project = FineTuneProject(
        model_id="Org/Custom",
        model_type="chat",
        lora_target_modules="attn_query,attn_key,attn_value,mlp_gate",
    )
    add_project_examples(tmp_path, project, [FineTuneExample([{"role": "user", "content": "A"}], "one")])
    save_finetune_project(tmp_path, project)

    reloaded = load_finetune_project(tmp_path)
    assert reloaded is not None
    assert reloaded.lora_target_modules == "attn_query,attn_key,attn_value,mlp_gate"

    job = prepare_finetune_job(tmp_path, reloaded)
    config = json.loads((job.directory / "config.json").read_text())
    assert config["lora_target_modules"] == "attn_query,attn_key,attn_value,mlp_gate"


def test_lora_target_modules_default_to_the_all_linear_shorthand(tmp_path: Path) -> None:
    project = FineTuneProject(model_id="Org/Chat", model_type="chat")
    add_project_examples(tmp_path, project, [FineTuneExample([{"role": "user", "content": "A"}], "one")])
    job = prepare_finetune_job(tmp_path, project)
    # Empty in the config; the runner turns that into "all-linear", as before.
    assert json.loads((job.directory / "config.json").read_text())["lora_target_modules"] == ""


def test_version_one_project_is_promoted_to_default_recipe(tmp_path: Path) -> None:
    metadata = tmp_path / ".opbdh/finetune.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": "Org/Legacy",
                "model_type": "base",
                "method": "qlora",
                "gpu_count": 4,
                "learning_rate": 7e-5,
            }
        ),
        encoding="utf-8",
    )

    project = load_finetune_project(tmp_path)

    assert project is not None
    assert project.version == 2
    assert project.active_recipe == "default"
    assert recipe_names(project) == ["default"]
    assert project.model_id == "Org/Legacy"
    assert project.method == "qlora"
    assert project.gpu_count == 4
    assert project.learning_rate == 7e-5


def test_project_persists_editable_toml_and_prepares_filtered_multi_gpu_job(tmp_path: Path) -> None:
    project = FineTuneProject(
        model_id="Org/Chat",
        model_type="chat",
        gpu_count=4,
        selected_tags=["product"],
    )
    examples = [
        FineTuneExample([{"role": "user", "content": "A"}], "one", ("product",)),
        FineTuneExample([{"role": "user", "content": "B"}], "two", ("other",)),
    ]

    add_project_examples(tmp_path, project, examples, source_files=[tmp_path / "source.jsonl"])
    loaded_project = load_finetune_project(tmp_path)
    assert loaded_project is not None
    assert loaded_project.example_count == 2
    assert loaded_project.data_file.endswith(".toml")
    assert read_project_examples(tmp_path, loaded_project) == examples

    job = prepare_finetune_job(tmp_path, loaded_project)

    assert job.example_count == 1
    assert "--num_processes 4" in job.command
    rows = [json.loads(line) for line in (job.directory / "dataset.jsonl").read_text().splitlines()]
    assert rows == [
        {
            "prompt": [{"role": "user", "content": "A"}],
            "completion": [{"role": "assistant", "content": "one"}],
        }
    ]
    py_compile.compile(str(job.directory / "run.py"), doraise=True)
    bundle = build_bundle(
        OpbdhConfig(model_id=loaded_project.model_id, gpu_count=4),
        code_path=job.directory,
        command=job.command,
        run_id="ft-test",
    )
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert {
        "user/run.py",
        "user/config.json",
        "user/dataset.jsonl",
        "user/requirements.txt",
        "job.sh",
    }.issubset(names)


def test_generated_runner_matches_pinned_trl_api_contract(tmp_path: Path) -> None:
    project = FineTuneProject(model_id="Org/Base", model_type="base")
    add_project_examples(tmp_path, project, [FineTuneExample("input", "output")])
    job = prepare_finetune_job(tmp_path, project)
    source = (job.directory / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    calls = {
        call.func.id: call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    config_keywords = {keyword.arg for keyword in calls["SFTConfig"].keywords}
    trainer_keywords = {keyword.arg for keyword in calls["SFTTrainer"].keywords}
    assert "trust_remote_code" not in config_keywords
    assert "quantization_config" not in trainer_keywords
    assert {"model", "processing_class", "peft_config"}.issubset(trainer_keywords)
    assert "AutoModelForCausalLM.from_pretrained" in source
    assert "AutoTokenizer.from_pretrained" in source

    requirements = (job.directory / "requirements.txt").read_text(encoding="utf-8")
    assert "trl>=0.29.1,<0.30" in requirements
    assert "transformers>=5,<6" in requirements


def test_gradient_checkpointing_follows_the_model_not_a_hardcoded_true(tmp_path: Path) -> None:
    project = FineTuneProject(model_id="Org/Base", model_type="base")
    add_project_examples(tmp_path, project, [FineTuneExample("input", "output")])
    job = prepare_finetune_job(tmp_path, project)
    source = (job.directory / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    config_call = next(
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "SFTConfig"
    )
    checkpointing = next(k.value for k in config_call.keywords if k.arg == "gradient_checkpointing")
    # Asking an architecture that does not implement checkpointing for it makes
    # transformers refuse to train at all, so this is read off the loaded model
    # rather than assumed.
    assert not isinstance(checkpointing, ast.Constant)
    assert "supports_gradient_checkpointing" in source


def test_resource_estimate_keeps_full_replica_on_each_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(finetune, "estimate_for_model", lambda *args, **kwargs: _estimate())
    project = FineTuneProject(
        model_id="Org/Model",
        model_type="base",
        method="lora",
        gpu_count=4,
    )

    resources = estimate_finetune_resources(project)

    assert resources.vram_per_gpu_gb == _estimate().min_vram_gb(1)
    assert resources.vram_per_gpu_gb != _estimate().min_vram_gb(4)
    assert resources.host_ram_per_gpu_gb == 16
    assert resources.disk_gb == 80


def test_build_run_config_uses_generated_job_and_sized_pod(tmp_path: Path) -> None:
    project = FineTuneProject(model_id="Org/Model", model_type="base", gpu_count=2, provider="runpod")
    add_project_examples(tmp_path, project, [FineTuneExample("input", "output")])
    job = prepare_finetune_job(tmp_path, project)
    estimate = _estimate()
    resources = finetune.FineTuneResources(estimate, 28, 24, 80)

    config = build_finetune_run_config(
        OpbdhConfig(container_disk_gb=40, pod_volume_gb=50),
        root=tmp_path,
        project=project,
        job=job,
        resources=resources,
    )

    assert config.model_id == "Org/Model"
    assert config.code == str(job.directory)
    assert config.command == job.command
    assert config.gpu_count == 2
    assert config.vram_gb == 28
    assert config.container_disk_gb == 80
    assert config.results_dir == str(tmp_path / ".opbdh/finetune/results/default")


def test_mixed_base_and_chat_inputs_are_rejected() -> None:
    with pytest.raises(FineTuneDataError, match="mixes"):
        infer_model_type(
            [
                FineTuneExample("plain", "out"),
                FineTuneExample([{"role": "user", "content": "chat"}], "out"),
            ]
        )
