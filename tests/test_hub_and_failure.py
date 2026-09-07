import io
import tarfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from opbdh.config import OpbdhConfig
from opbdh.runpod import (
    OpbdhPlan,
    _extract_tar_archive_safely,
    build_job_script,
    plan_summary,
    run_plan,
)


def test_build_job_script_with_push_to_hub():
    config = OpbdhConfig(
        model_id="test-model",
        push_to_hub="lumpenspace/my-model",
        push_to_hub_private=True,
    )
    script = build_job_script(config, command="python train.py", network_volume_id="")
    assert "export OPBDH_PUSH_TO_HUB=lumpenspace/my-model" in script
    assert "export OPBDH_PUSH_TO_HUB_PRIVATE=1" in script
    assert "uploading results to hugging face hub: lumpenspace/my-model" in script
    assert "HfApi" in script
    assert "api.upload_folder" in script


def test_build_job_script_without_push_to_hub():
    config = OpbdhConfig(
        model_id="test-model",
        push_to_hub="",
    )
    script = build_job_script(config, command="python train.py", network_volume_id="")
    assert "export OPBDH_PUSH_TO_HUB=''" in script
    assert "uploading results to hugging face hub" not in script


def test_safe_tar_extraction_handles_filters(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        data = b"hello world"
        info = tarfile.TarInfo(name="test.txt")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    buf.seek(0)
    dest = tmp_path / "extracted"
    dest.mkdir()
    with tarfile.open(fileobj=buf, mode="r:gz") as archive:
        _extract_tar_archive_safely(archive, dest)

    assert (dest / "test.txt").read_text() == "hello world"


def test_plan_summary_includes_push_to_hub(tmp_path):
    config = OpbdhConfig(
        push_to_hub="lumpenspace/my-model",
    )
    plan = OpbdhPlan(
        run_id="run-1",
        config=config,
        code_path=Path("run.py"),
        command="python run.py",
        gpu_type_ids=["NVIDIA A100"],
        estimated_hourly_dollars=2.0,
        model_size_gb=None,
        network_volume_id="",
        network_volume_size_gb=None,
        results_dir=tmp_path,
        verification_checked=set(),
    )
    summary = plan_summary(plan)
    assert summary["push_to_hub"] == "lumpenspace/my-model"


@patch("opbdh.runpod.resolve_ssh_key_paths")
@patch("opbdh.runpod.ensure_network_volume")
@patch("opbdh.runpod.create_runpod_pod")
@patch("opbdh.runpod.wait_for_runpod_pod")
@patch("opbdh.runpod.extract_runpod_ssh_target")
@patch("opbdh.runpod.wait_for_ssh")
@patch("opbdh.runpod._upload_bundle")
@patch("opbdh.runpod._start_remote_job")
@patch("opbdh.runpod._remote_status")
@patch("opbdh.runpod.sync_results_from_pod")
@patch("opbdh.runpod.delete_runpod_pod")
@patch("opbdh.runpod._send_failure_alert")
@patch("rich.console.Console.print")
def test_script_failure_keeps_pod_and_alerts(
    mock_print,
    mock_alert,
    mock_delete,
    mock_sync,
    mock_status,
    mock_start,
    mock_upload,
    mock_wait_ssh,
    mock_extract,
    mock_wait_pod,
    mock_create,
    mock_ensure_volume,
    mock_resolve_keys,
    tmp_path,
):
    mock_ensure_volume.return_value = "mock-vol-id"
    mock_resolve_keys.return_value = (MagicMock(), MagicMock())
    mock_create.return_value = ("pod-failure-123", "ssh-hint", "NVIDIA A100")
    mock_extract.return_value = MagicMock(host="127.0.0.1", port=22)
    mock_status.return_value = ("done", 1)  # Script fails with exit code 1

    config = OpbdhConfig(
        code="run.py",
        keep_pod_on_failure=True,
    )
    plan = OpbdhPlan(
        run_id="run-fail",
        config=config,
        code_path=Path("run.py"),
        command="python run.py",
        gpu_type_ids=["NVIDIA A100"],
        estimated_hourly_dollars=2.0,
        model_size_gb=None,
        network_volume_id="",
        network_volume_size_gb=None,
        results_dir=tmp_path / "results",
        verification_checked=set(),
    )

    with pytest.raises(RuntimeError, match="remote job failed with exit code 1"):
        run_plan(plan, progress=False, interactive=False)

    # Pod must NOT be deleted when script fails and keep_pod_on_failure=True
    mock_delete.assert_not_called()
    # Log should show pod preserved
    local_log = (tmp_path / "results" / "opbdh.log").read_text()
    assert "preserved (not deleted)" in local_log
