from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from opbdh.config import OpbdhConfig, load_config
from opbdh.hal import HalEye, hal_enabled, hal_says
from opbdh.hf import estimate_model_size_gb, suggested_network_volume_gb
from opbdh.runpod import build_bundle, ensure_network_volume, make_plan
from opbdh.verify import default_command_for_code, verify_code


def test_config_loads_global_and_interpolates_paths(monkeypatch, tmp_path: Path) -> None:
    global_config = tmp_path / "config.json"
    global_config.write_text(
        json.dumps(
            {
                "model_id": "Org/Test Model",
                "code": "{cwd}/run.py",
                "results_dir": "${OPBDH_TEST_RESULTS}/{model_slug}/{run_id}",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPBDH_CONFIG", str(global_config))
    monkeypatch.setenv("OPBDH_TEST_RESULTS", str(tmp_path / "results"))

    config = load_config(cwd=tmp_path, run_id="run-123")

    assert config.code == str(tmp_path / "run.py")
    assert config.results_dir == str(tmp_path / "results" / "org-test-model" / "run-123")


def test_verify_code_compiles_python_and_reports_syntax_errors(tmp_path: Path) -> None:
    ok_file = tmp_path / "ok.py"
    bad_file = tmp_path / "bad.py"
    ok_file.write_text("print('hello')\n", encoding="utf-8")
    bad_file.write_text("def broken(:\n", encoding="utf-8")

    assert verify_code(ok_file).ok

    result = verify_code(bad_file)

    assert not result.ok
    assert "bad.py" in result.errors[0]


def test_default_command_and_bundle_include_user_code(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text("print('open the pod bay door')\n", encoding="utf-8")
    command = default_command_for_code(script)

    bundle = build_bundle(OpbdhConfig(model_id="Qwen/Qwen2.5-0.5B-Instruct"), code_path=script, command=command, run_id="abc")

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = set(archive.getnames())
        manifest = json.loads(archive.extractfile("opbdh_manifest.json").read().decode("utf-8"))  # type: ignore[union-attr]

    assert command == "python /opbdh-run/user/run.py"
    assert "user/run.py" in names
    assert "job.sh" in names
    assert manifest["run_id"] == "abc"


def test_bundle_excludes_dotenv_files(tmp_path: Path) -> None:
    code_dir = tmp_path / "project"
    code_dir.mkdir()
    (code_dir / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (code_dir / ".env").write_text("SECRET=topsecret\n", encoding="utf-8")
    (code_dir / ".env.production").write_text("SECRET=topsecret\n", encoding="utf-8")

    bundle = build_bundle(OpbdhConfig(), code_path=code_dir, command="python /opbdh-run/user/run.py", run_id="abc")

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = set(archive.getnames())

    assert "user/run.py" in names
    assert not any(".env" in name for name in names)


def test_job_script_does_not_embed_hf_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret")
    script = tmp_path / "run.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    bundle = build_bundle(OpbdhConfig(model_id="Org/Model"), code_path=script, command="python /opbdh-run/user/run.py", run_id="abc")

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        job_script = archive.extractfile("job.sh").read().decode("utf-8")  # type: ignore[union-attr]

    assert "hf_supersecret" not in job_script


def test_make_plan_does_not_call_hf_size_lookup_without_auto_volume(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def fail_lookup(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("HF lookup should not be needed")

    monkeypatch.setattr("opbdh.runpod.estimate_model_size_gb", fail_lookup)

    plan = make_plan(
        OpbdhConfig(model_id="Org/Model", vram_gb=24, max_dollars_per_hour=1.0),
        code_path=script,
        run_id="run-1",
    )

    assert plan.model_size_gb is None
    assert plan.gpu_type_ids


def test_auto_network_volume_uses_model_size_multiplier(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "opbdh.runpod.estimate_model_size_gb",
        lambda model_id: estimate_model_size_gb(model_id, siblings=[{"rfilename": "model.safetensors", "size": 30 * 1024**3}]),
    )

    def fake_create_network_volume(**kwargs):
        captured.update(kwargs)
        return {"id": "volume-123"}

    monkeypatch.setattr("opbdh.runpod.create_network_volume", fake_create_network_volume)
    plan = make_plan(
        OpbdhConfig(
            model_id="Org/Model",
            auto_network_volume=True,
            network_volume_data_center_id="EU-RO-1",
        ),
        code_path=script,
        run_id="run-2",
    )

    volume_id = ensure_network_volume(plan)

    assert volume_id == "volume-123"
    assert captured["size_gb"] == 75
    assert captured["data_center_id"] == "EU-RO-1"


def test_hal_stays_silent_outside_a_tty(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPBDH_NO_HAL", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("opbdh.hal._stdout_isatty", lambda: False)

    assert not hal_enabled()
    hal_says("I'm sorry, Dave.")
    with HalEye("waiting") as eye:
        eye.update("still waiting")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_hal_opt_outs_beat_a_tty(monkeypatch) -> None:
    monkeypatch.setattr("opbdh.hal._stdout_isatty", lambda: True)

    monkeypatch.setenv("OPBDH_NO_HAL", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert not hal_enabled()

    monkeypatch.delenv("OPBDH_NO_HAL")
    monkeypatch.setenv("NO_COLOR", "1")
    assert not hal_enabled()

    monkeypatch.delenv("NO_COLOR")
    assert hal_enabled()


def test_hf_model_size_estimate_and_volume_suggestion() -> None:
    estimate = estimate_model_size_gb(
        "Org/Model",
        siblings=[
            {"rfilename": "model-00001-of-00002.safetensors", "size": 10 * 1024**3},
            {"rfilename": "model-00002-of-00002.safetensors", "size": 12 * 1024**3},
            {"rfilename": "README.md", "size": 999},
        ],
    )

    assert estimate.size_gb == 22
    assert suggested_network_volume_gb(estimate) == 55
