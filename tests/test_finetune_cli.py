from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from opbdh import cli
from opbdh.estimate import MemoryEstimate
from opbdh.finetune import (
    FineTuneExample,
    FineTuneProject,
    FineTuneResources,
    activate_finetune_recipe,
    add_project_examples,
    load_finetune_project,
    read_examples,
    read_project_examples,
    recipe_names,
)


runner = CliRunner()


class _Answer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


class _QuestionaryAnswers:
    def __init__(self, *answers):
        self.answers = iter(answers)

    def select(self, *args, **kwargs):
        return _Answer(next(self.answers))

    def text(self, *args, **kwargs):
        return _Answer(next(self.answers))

    def confirm(self, *args, **kwargs):
        return _Answer(next(self.answers))


def _resources() -> FineTuneResources:
    estimate = MemoryEstimate(
        model_id="Org/Base",
        goal="lora",
        param_count=1_000_000,
        context_len=512,
        batch_size=1,
        weights_gb=2.0,
        kv_cache_gb=0.0,
        activations_gb=1.0,
        optimizer_gb=1.0,
        total_vram_gb=8.0,
        host_ram_gb=16,
        disk_gb=20,
    )
    return FineTuneResources(estimate, 12, 16, 40)


def test_ft_import_defaults_to_human_editable_toml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "openai.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["ft:import", str(source), "--tag", "greeting"])

    assert result.exit_code == 0, result.output
    output = tmp_path / "openai.opbdh.toml"
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "input = [" in text
    assert read_examples([output])[0].tags == ("greeting",)


def test_interactive_tag_manager_bulk_adds_tags_to_loaded_examples(tmp_path: Path) -> None:
    project = FineTuneProject(model_id="Org/Base", model_type="base")
    examples = add_project_examples(
        tmp_path,
        project,
        [FineTuneExample("one", "1", ("old",)), FineTuneExample("two", "2")],
    )
    prompts = _QuestionaryAnswers("Add tags in bulk", "All examples", "new", "Done")

    updated = cli._fine_tune_manage_tags(prompts, tmp_path, project, examples)

    assert [example.tags for example in updated] == [("old", "new"), ("new",)]
    assert read_project_examples(tmp_path, project) == updated


def test_interactive_recipe_manager_creates_method_default_recipe(tmp_path: Path) -> None:
    project = FineTuneProject(model_id="Org/Base", model_type="base")
    add_project_examples(tmp_path, project, [FineTuneExample("one", "1")])
    prompts = _QuestionaryAnswers(
        "Create recipe with defaults",
        "full-2gpu",
        "Full fine-tune",
        False,
        "Done",
    )

    cli._fine_tune_manage_recipes(prompts, tmp_path, project)

    loaded = load_finetune_project(tmp_path)
    assert loaded is not None
    assert loaded.active_recipe == "full-2gpu"
    assert loaded.method == "full"
    assert loaded.learning_rate == 2e-5
    assert recipe_names(loaded) == ["default", "full-2gpu"]


def test_ft_builds_and_persists_a_two_gpu_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPBDH_CONFIG_DIR", str(tmp_path / "global-config"))
    data = tmp_path / "example.toml"
    data.write_text('input = "Question"\noutput = "Answer"\ntags = ["qa"]\n', encoding="utf-8")
    captured = {}
    monkeypatch.setattr(cli, "estimate_finetune_resources", lambda project: _resources())
    monkeypatch.setattr(
        cli,
        "_execute_run",
        lambda config, *, dry_run, yes: captured.update(config=config, dry_run=dry_run, yes=yes),
    )

    result = runner.invoke(
        cli.app,
        [
            "ft",
            "--data",
            str(data),
            "--model",
            "Org/Base",
            "--model-type",
            "base",
            "--gpu-count",
            "2",
            "--tag",
            "qa",
            "--dry-run",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    project = load_finetune_project(tmp_path)
    assert project is not None
    assert project.gpu_count == 2
    assert project.selected_tags == ["qa"]
    assert len(read_project_examples(tmp_path, project)) == 1
    assert "--num_processes 2" in captured["config"].command
    assert captured["config"].vram_gb == 12
    assert captured["dry_run"] is True
    assert captured["yes"] is True


def test_ft_reuses_saved_project_without_data_flags(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPBDH_CONFIG_DIR", str(tmp_path / "global-config"))
    data = tmp_path / "example.toml"
    data.write_text('input = "Question"\noutput = "Answer"\n', encoding="utf-8")
    monkeypatch.setattr(cli, "estimate_finetune_resources", lambda project: _resources())
    launches = []
    monkeypatch.setattr(cli, "_execute_run", lambda config, *, dry_run, yes: launches.append(config))

    first = runner.invoke(
        cli.app,
        ["ft", "--data", str(data), "--model", "Org/Base", "--model-type", "base", "--dry-run", "--yes"],
    )
    second = runner.invoke(cli.app, ["ft", "--dry-run", "--yes"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert len(launches) == 2
    assert launches[1].model_id == "Org/Base"


def test_ft_creates_independent_named_recipes_for_one_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPBDH_CONFIG_DIR", str(tmp_path / "global-config"))
    data = tmp_path / "example.toml"
    data.write_text('input = "Question"\noutput = "Answer"\n', encoding="utf-8")
    monkeypatch.setattr(cli, "estimate_finetune_resources", lambda project: _resources())
    launches = []
    monkeypatch.setattr(cli, "_execute_run", lambda config, *, dry_run, yes: launches.append(config))

    first = runner.invoke(
        cli.app,
        [
            "ft",
            "--data",
            str(data),
            "--model",
            "Org/Base",
            "--model-type",
            "base",
            "--recipe",
            "lora-fast",
            "--method",
            "lora",
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
            "--lora-dropout",
            "0.1",
            "--packing",
            "--seed",
            "7",
            "--dry-run",
            "--yes",
        ],
    )
    second = runner.invoke(
        cli.app,
        [
            "ft",
            "--recipe",
            "full-2gpu",
            "--method",
            "full",
            "--gpu-count",
            "2",
            "--epochs",
            "1.5",
            "--dry-run",
            "--yes",
        ],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    project = load_finetune_project(tmp_path)
    assert project is not None
    assert recipe_names(project) == ["full-2gpu", "lora-fast"]
    assert project.active_recipe == "full-2gpu"
    assert project.method == "full"
    assert project.learning_rate == 2e-5
    assert project.gpu_count == 2
    assert project.epochs == 1.5
    assert len(read_project_examples(tmp_path, project)) == 1
    assert json.loads(
        (tmp_path / ".opbdh/finetune/jobs/full-2gpu/config.json").read_text()
    )["method"] == "full"

    activate_finetune_recipe(project, "lora-fast")
    assert project.method == "lora"
    assert project.learning_rate == 2e-4
    assert project.gpu_count == 1
    assert project.lora_r == 8
    assert project.lora_alpha == 16
    assert project.lora_dropout == 0.1
    assert project.packing is True
    assert project.seed == 7
    assert len(launches) == 2
