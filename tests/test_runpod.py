import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from opbdh.runpod import run_plan, OpbdhPlan
from opbdh.config import OpbdhConfig

@pytest.fixture
def mock_config():
    return OpbdhConfig(
        code="fake_path.py",
        model_id="fake-model",
        network_volume_data_center_id="US-MD-1",
        failure_keepalive_seconds=0
    )

@pytest.fixture
def mock_plan(mock_config, tmp_path):
    return OpbdhPlan(
        run_id="test_run",
        config=mock_config,
        code_path=Path("fake_path.py"),
        command="python fake_path.py",
        gpu_type_ids=["NVIDIA A100"],
        estimated_hourly_dollars=2.0,
        model_size_gb=5.0,
        network_volume_id="vol-123",
        network_volume_size_gb=100,
        results_dir=tmp_path / "fake_results",
        verification_checked=set()
    )

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
@patch("opbdh.runpod._timed_yes_no", return_value=False)
@patch("opbdh.runpod.delete_runpod_pod")
@patch("rich.console.Console.print")
def test_run_plan_prints_remote_logs_on_failure(
    mock_print, mock_delete, mock_yesno, mock_sync, mock_status, mock_start, mock_upload, mock_wait_ssh,
    mock_extract, mock_wait_pod, mock_create, mock_ensure_volume, mock_resolve_keys, mock_plan
):
    mock_ensure_volume.return_value = "mock-vol-id"
    mock_resolve_keys.return_value = (MagicMock(), MagicMock())
    mock_create.return_value = ("pod-123", "ssh-hint", "NVIDIA A100")
    mock_extract.return_value = MagicMock(host="127.0.0.1", port=22)
    mock_status.return_value = ("done", 1)  # Simulate failure
    
    logs_dir = mock_plan.results_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "stderr.log").write_text("Traceback: critical remote failure", encoding="utf-8")
    
    with pytest.raises(RuntimeError, match="remote job failed with exit code 1"):
        run_plan(mock_plan, dry_run=False)
        
    printed_texts = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert any("Traceback: critical remote failure" in str(text) for text in printed_texts)
