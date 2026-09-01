import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import opbdh.runpod as runpod_module
from opbdh.remote import InsufficientCreditsError, RunpodBalance
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


@pytest.fixture
def successful_runpod_lifecycle(monkeypatch):
    private_key = MagicMock()
    public_key = MagicMock()
    public_key.read_text.return_value = "ssh-ed25519 test"
    mocks = {
        "balance": MagicMock(return_value=None),
        "build_bundle": MagicMock(return_value=b"bundle"),
        "create": MagicMock(return_value=("pod-123", "ssh-hint", "NVIDIA A100")),
        "delete": MagicMock(),
        "ensure_volume": MagicMock(return_value="mock-vol-id"),
        "estimated_hourly": MagicMock(return_value=1.69),
        "extract": MagicMock(return_value=MagicMock(host="127.0.0.1", port=22)),
        "resolve_keys": MagicMock(return_value=(private_key, public_key)),
        "start": MagicMock(return_value="1234"),
        "status": MagicMock(return_value=("done", 0)),
        "sync": MagicMock(),
        "upload": MagicMock(),
        "wait_pod": MagicMock(return_value={}),
        "wait_ssh": MagicMock(),
    }
    for name, mock in {
        "runpod_balance": mocks["balance"],
        "build_bundle": mocks["build_bundle"],
        "create_runpod_pod": mocks["create"],
        "delete_runpod_pod": mocks["delete"],
        "ensure_network_volume": mocks["ensure_volume"],
        "estimated_hourly": mocks["estimated_hourly"],
        "extract_runpod_ssh_target": mocks["extract"],
        "resolve_ssh_key_paths": mocks["resolve_keys"],
        "_start_remote_job": mocks["start"],
        "_remote_status": mocks["status"],
        "sync_results_from_pod": mocks["sync"],
        "_upload_bundle": mocks["upload"],
        "wait_for_runpod_pod": mocks["wait_pod"],
        "wait_for_ssh": mocks["wait_ssh"],
    }.items():
        monkeypatch.setattr(runpod_module, name, mock)
    return mocks


def test_run_plan_emits_runpod_balance_before_and_after_run(
    mock_plan, successful_runpod_lifecycle
):
    successful_runpod_lifecycle["balance"].side_effect = [
        RunpodBalance(client_balance=39.70, current_spend_per_hour=1.69),
        RunpodBalance(client_balance=38.92, current_spend_per_hour=0.0),
    ]
    events = []

    result = run_plan(mock_plan, progress=False, interactive=False, on_event=events.append)

    assert result is not None
    assert [(event.kind, event.message) for event in events if event.kind == "billing"] == [
        ("billing", "$39.70 left · this pod ~$1.69/hr · ~23h"),
        ("billing", "$38.92 left after run"),
    ]
    successful_runpod_lifecycle["delete"].assert_called_once_with(
        "pod-123", search_from=mock_plan.code_path.parent
    )


def test_run_plan_omits_balance_events_when_lookup_fails(
    mock_plan, successful_runpod_lifecycle
):
    successful_runpod_lifecycle["balance"].side_effect = RuntimeError("account lookup failed")
    events = []

    result = run_plan(mock_plan, progress=False, interactive=False, on_event=events.append)

    assert result is not None
    assert [event for event in events if event.kind == "billing"] == []


def test_run_plan_emits_zero_balance_without_dividing_by_zero(
    mock_plan, successful_runpod_lifecycle
):
    mock_plan.config.gpu_count = 2
    successful_runpod_lifecycle["balance"].side_effect = [
        RunpodBalance(client_balance=0.0, current_spend_per_hour=0.0),
        None,
    ]
    events = []

    run_plan(mock_plan, progress=False, interactive=False, on_event=events.append)

    billing_messages = [event.message for event in events if event.kind == "billing"]
    assert billing_messages == ["$0.00 left · this pod ~$3.38/hr · ~0h"]


def test_run_plan_surfaces_typed_insufficient_credit_error(
    mock_plan, successful_runpod_lifecycle
):
    successful_runpod_lifecycle["create"].side_effect = InsufficientCreditsError(
        "RunPod refused the pod"
    )
    successful_runpod_lifecycle["balance"].return_value = RunpodBalance(
        client_balance=1.25,
        current_spend_per_hour=0.0,
    )

    with pytest.raises(InsufficientCreditsError) as exc_info:
        run_plan(mock_plan, progress=False, interactive=False)

    message = str(exc_info.value)
    assert "$1.25 left" in message
    assert "$1.69/hr" in message
    assert "Add funds to your RunPod account and try again" in message
    successful_runpod_lifecycle["balance"].assert_called_once_with()
    successful_runpod_lifecycle["delete"].assert_not_called()


def test_run_plan_credit_error_does_not_invent_a_zero_balance(
    mock_plan, successful_runpod_lifecycle
):
    successful_runpod_lifecycle["create"].side_effect = InsufficientCreditsError(
        "RunPod refused the pod"
    )
    successful_runpod_lifecycle["balance"].return_value = None

    with pytest.raises(InsufficientCreditsError) as exc_info:
        run_plan(mock_plan, progress=False, interactive=False)

    message = str(exc_info.value)
    assert "current balance unavailable" in message
    assert "$0.00 left" not in message


def test_run_plan_credit_error_prices_the_effective_gpu_override(
    monkeypatch, mock_plan, successful_runpod_lifecycle
):
    mock_plan.gpu_type_ids = ["GPU cheap", "GPU pinned"]
    mock_plan.config.gpu_count = 2
    monkeypatch.setenv("OPBDH_RUNPOD_GPU_TYPES", "GPU pinned")
    successful_runpod_lifecycle["estimated_hourly"].side_effect = (
        lambda gpu_type, cloud_type: 2.25 if gpu_type == "GPU pinned" else 0.50
    )
    successful_runpod_lifecycle["create"].side_effect = InsufficientCreditsError(
        "RunPod refused the pod"
    )
    successful_runpod_lifecycle["balance"].return_value = RunpodBalance(1.25, 0.0)

    with pytest.raises(InsufficientCreditsError, match=r"\$4\.50/hr"):
        run_plan(mock_plan, progress=False, interactive=False)

    assert successful_runpod_lifecycle["create"].call_args.kwargs["gpu_types"] == [
        "GPU pinned"
    ]

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
@patch("opbdh.runpod.runpod_balance", return_value=None)
@patch("rich.console.Console.print")
def test_run_plan_prints_remote_logs_on_failure(
    mock_print, mock_balance, mock_delete, mock_yesno, mock_sync, mock_status, mock_start, mock_upload, mock_wait_ssh,
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


def test_credit_wording_is_not_reported_as_insufficient_when_funded(
    monkeypatch, mock_plan, successful_runpod_lifecycle
):
    """Providers say "check your credit balance" in capacity errors too. A
    non-402 classification is a suspicion, so a visibly funded account must
    get the real error rather than being told it is out of money."""
    successful_runpod_lifecycle["create"].side_effect = InsufficientCreditsError(
        "RunPod API POST /pods failed with HTTP 500: no capacity; check your credit balance",
        status_code=500,
    )
    successful_runpod_lifecycle["balance"].return_value = RunpodBalance(35.96, 0.12)

    with pytest.raises(RuntimeError) as excinfo:
        run_plan(mock_plan, progress=False, interactive=False)

    assert not isinstance(excinfo.value, InsufficientCreditsError)
    assert "no capacity" in str(excinfo.value)


def test_http_402_is_still_definitive_even_when_the_balance_looks_fine(
    monkeypatch, mock_plan, successful_runpod_lifecycle
):
    """402 is the provider stating a payment problem; a stale-looking balance
    must not override it."""
    successful_runpod_lifecycle["create"].side_effect = InsufficientCreditsError(
        "RunPod API POST /pods failed with HTTP 402: Payment Required",
        status_code=402,
    )
    successful_runpod_lifecycle["balance"].return_value = RunpodBalance(35.96, 0.12)

    with pytest.raises(InsufficientCreditsError):
        run_plan(mock_plan, progress=False, interactive=False)
