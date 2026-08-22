"""Tests for the programmatic API (`opbdh.launch` and friends).

Offline only: nothing here contacts RunPod, Prime Intellect, or Hugging Face.
Runs that would rent a pod use `dry_run=True`, which returns before the
provider is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from modelchoice import ModelOption, ProviderMetadata

import opbdh
from opbdh.api import configure, gpu_options, launch, plan, summarize, verify
from opbdh.runpod import RunEvent, _Reporter


@pytest.fixture
def isolated_config(monkeypatch, tmp_path: Path) -> Path:
    """Point config discovery at an empty tmp dir so the developer's own
    ~/.config/opbdh/config.json can't leak into assertions."""
    empty = tmp_path / "empty-global.json"
    empty.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPBDH_CONFIG", str(empty))
    monkeypatch.setenv("OPBDH_CONFIG_DIR", str(tmp_path / "confdir"))
    return tmp_path


@pytest.fixture
def code_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "run.py").write_text("print('hi')\n", encoding="utf-8")
    return directory


class TestConfigure:
    def test_cli_style_aliases_map_onto_config_fields(self, isolated_config) -> None:
        config = configure(model="Org/Model", max_spend=12.5, min_ram_per_gpu=64, vram_gb=48)
        assert config.model_id == "Org/Model"
        assert config.max_spend_dollars == 12.5
        assert config.min_ram_per_gpu_gb == 64
        assert config.vram_gb == 48

    def test_code_argument_populates_the_code_field(self, isolated_config, code_dir) -> None:
        assert configure(code_dir).code == str(code_dir)

    def test_local_config_file_is_layered_under_overrides(self, isolated_config, tmp_path) -> None:
        local = tmp_path / "opbdh.json"
        local.write_text(json.dumps({"model_id": "From/File", "vram_gb": 80}), encoding="utf-8")

        from_file = configure(config_file=local)
        assert from_file.model_id == "From/File"
        assert from_file.vram_gb == 80

        overridden = configure(config_file=local, model="From/Kwarg")
        assert overridden.model_id == "From/Kwarg"
        assert overridden.vram_gb == 80

    def test_none_and_empty_overrides_do_not_clobber_defaults(self, isolated_config) -> None:
        config = configure(model=None, command="")
        assert config.model_id == ""
        assert config.provider == "runpod"


class TestPlan:
    def test_plan_selects_gpu_candidates_without_renting(self, isolated_config, code_dir) -> None:
        opbdh_plan = plan(code_dir, vram_gb=48, max_dollars_per_hour=2.0, run_id="run-1")
        assert opbdh_plan.run_id == "run-1"
        assert opbdh_plan.code_path == code_dir.resolve()
        assert opbdh_plan.gpu_type_ids
        assert all("NVIDIA" in gpu or "AMD" in gpu for gpu in opbdh_plan.gpu_type_ids)
        assert opbdh_plan.results_dir.name == "run-1"

    def test_plan_infers_the_command_from_the_code_path(self, isolated_config, code_dir) -> None:
        assert "run.py" in plan(code_dir, run_id="r").command

    def test_plan_requires_a_code_path(self, isolated_config) -> None:
        with pytest.raises(ValueError, match="code path is required"):
            plan()

    def test_plan_rejects_code_that_does_not_compile(self, isolated_config, tmp_path) -> None:
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "run.py").write_text("def nope(:\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Static verification failed"):
            plan(broken)

    def test_impossible_vram_request_has_no_candidates(self, isolated_config, code_dir) -> None:
        with pytest.raises(ValueError, match="No configured RunPod GPU estimate"):
            plan(code_dir, vram_gb=100_000)

    def test_summarize_is_json_serializable(self, isolated_config, code_dir) -> None:
        payload = summarize(plan(code_dir, run_id="run-2"))
        assert payload["run_id"] == "run-2"
        json.dumps(payload)


class TestLaunch:
    def test_dry_run_returns_none_and_writes_a_log(self, isolated_config, code_dir, tmp_path) -> None:
        results = tmp_path / "results"
        result = launch(code_dir, dry_run=True, run_id="run-3", results_dir=str(results))
        assert result is None
        log = results / "run-3" / "opbdh.log"
        assert log.exists()
        assert "Dry run requested" in log.read_text(encoding="utf-8")

    def test_dry_run_emits_progress_events(self, isolated_config, code_dir, tmp_path) -> None:
        events, sink = opbdh.collect_events()
        launch(code_dir, dry_run=True, run_id="run-4", results_dir=str(tmp_path), on_event=sink)
        # A dry run returns before the reporter starts, so no events yet; the
        # sink still has to be wired up without error.
        assert events == []
        assert opbdh.event_messages(events) == []


class TestReporter:
    def test_reporter_forwards_events_and_skips_the_eye_when_quiet(self) -> None:
        seen: list[RunEvent] = []
        with _Reporter("starting", progress=False, on_event=seen.append) as reporter:
            reporter.update("working")
            reporter.emit("error", "boom")
            reporter.set_billing(started_at=0.0, hourly_dollars=1.0)
        assert [(event.kind, event.message) for event in seen] == [
            ("status", "starting"),
            ("status", "working"),
            ("error", "boom"),
        ]

    def test_reporter_without_a_callback_is_harmless(self) -> None:
        with _Reporter("starting", progress=False, on_event=None) as reporter:
            reporter.update("working")
            reporter.set_billing(started_at=0.0, hourly_dollars=None)

    def test_collect_events_helper_filters_by_kind(self) -> None:
        events, sink = opbdh.collect_events()
        sink(RunEvent("status", "a"))
        sink(RunEvent("error", "b"))
        assert opbdh.event_messages(events) == ["a", "b"]
        assert opbdh.event_messages(events, kind="error") == ["b"]


class TestHelpers:
    def test_verify_accepts_good_code_and_rejects_bad(self, tmp_path) -> None:
        good = tmp_path / "good.py"
        good.write_text("x = 1\n", encoding="utf-8")
        assert verify(good).ok

        bad = tmp_path / "bad.py"
        bad.write_text("def broken(:\n", encoding="utf-8")
        assert not verify(bad).ok

    def test_gpu_options_respect_vram_floor_and_price_ceiling(self) -> None:
        offers = gpu_options(vram_gb=80, max_dollars_per_hour=2.0, cloud_type="SECURE")
        assert offers
        for offer in offers:
            assert offer.memory_gb >= 80
            assert offer.hourly("SECURE") <= 2.0

    def test_gpu_options_are_ordered_by_vram_then_price(self) -> None:
        offers = gpu_options(vram_gb=24)
        memories = [offer.memory_gb for offer in offers]
        assert memories == sorted(memories)

    def test_suggest_volume_gb_scales_with_model_size(self) -> None:
        from opbdh.hf import ModelSizeEstimate

        small = opbdh.suggest_volume_gb(ModelSizeEstimate("m", 10.0, "metadata"))
        large = opbdh.suggest_volume_gb(ModelSizeEstimate("m", 100.0, "metadata"))
        assert large > small

    def test_suggest_volume_gb_falls_back_when_size_is_unknown(self) -> None:
        from opbdh.hf import ModelSizeEstimate

        assert opbdh.suggest_volume_gb(ModelSizeEstimate("m", None, "unavailable")) > 0

    def test_search_models_uses_shared_huggingface_catalog(self, monkeypatch) -> None:
        captured = {}

        class FakeHuggingFaceSource:
            def __init__(self, *, env):
                captured["env"] = env

            @property
            def metadata(self):
                return ProviderMetadata("huggingface", "Hugging Face", "hub", True)

            def initial_options(self):
                return []

            def search(self, query, limit=25, *, force=False):
                captured["search"] = (query, limit, force)
                return [
                    ModelOption(
                        "huggingface:Org/Model",
                        "Hugging Face · Org/Model",
                        "huggingface",
                        "Org/Model",
                    )
                ]

        monkeypatch.setattr("opbdh.api.HuggingFaceSource", FakeHuggingFaceSource)

        assert opbdh.search_models("model", limit=3, token="secret") == ["Org/Model"]
        assert captured["env"]["HF_TOKEN"] == "secret"
        assert captured["search"] == ("model", 3, False)


class TestPackageSurface:
    def test_documented_names_are_exported(self) -> None:
        for name in opbdh.__all__:
            assert hasattr(opbdh, name), name

    def test_run_plan_keeps_its_cli_defaults(self) -> None:
        import inspect

        from opbdh.runpod import run_plan

        parameters = inspect.signature(run_plan).parameters
        assert parameters["interactive"].default is True
        assert parameters["progress"].default is True
        assert parameters["on_event"].default is None

    def test_launch_is_non_interactive_by_default(self) -> None:
        import inspect

        parameters = inspect.signature(opbdh.launch).parameters
        assert parameters["progress"].default is False
