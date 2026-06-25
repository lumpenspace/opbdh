import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import typer

from opbdh.cli import _execute_run
from opbdh.config import OpbdhConfig
from opbdh.runpod import OpbdhPlan

@pytest.fixture
def mock_config():
    return OpbdhConfig(
        code="fake_path.py",
        model_id="fake-model",
        network_volume_data_center_id="US-MD-1"
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

@patch("opbdh.cli.run_plan")
@patch("opbdh.cli.make_plan")
@patch("opbdh.cli.console.print")
def test_insufficient_balance_error(mock_print, mock_make_plan, mock_run_plan, mock_config, mock_plan):
    mock_make_plan.return_value = mock_plan
    mock_run_plan.side_effect = RuntimeError("RunPod API failed: not enough balance for this action.")
    
    with pytest.raises(typer.Exit) as exc_info:
        _execute_run(mock_config, dry_run=False, yes=True)
        
    assert exc_info.value.exit_code == 1
    printed_texts = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert any("Insufficient RunPod Balance" in str(text) for text in printed_texts)


@patch("opbdh.cli.run_plan")
@patch("opbdh.cli.make_plan")
@patch("opbdh.cli.console.print")
def test_remote_job_error_prints_stderr(mock_print, mock_make_plan, mock_run_plan, mock_config, mock_plan):
    mock_make_plan.return_value = mock_plan
    mock_run_plan.side_effect = RuntimeError("remote job failed with exit code 1; see fake_results/logs")
    
    logs_dir = mock_plan.results_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "stderr.log").write_text("Traceback: this is a remote crash", encoding="utf-8")
    
    with pytest.raises(typer.Exit) as exc_info:
        _execute_run(mock_config, dry_run=False, yes=True)
        
    assert exc_info.value.exit_code == 1
    printed_texts = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert any("Execution Error" in str(text) for text in printed_texts)
    assert any("Traceback: this is a remote crash" in str(text) for text in printed_texts)


@patch("opbdh.cli.run_plan")
@patch("opbdh.cli.make_plan")
@patch("questionary.select")
@patch("opbdh.cli.console.print")
def test_out_of_stock_triggers_menu(mock_print, mock_select, mock_make_plan, mock_run_plan, mock_config, mock_plan):
    mock_make_plan.return_value = mock_plan
    mock_run_plan.side_effect = [
        RuntimeError("create pod: could not find any pods with required specifications"),
        None  # Succeeds on the second attempt
    ]
    
    mock_ask = MagicMock(return_value="Try without a network volume (ephemeral disk only)")
    mock_select.return_value.ask = mock_ask
    
    # We must patch _prompt_existing_network_volume so it doesn't try to invoke questionary internally
    with patch("opbdh.cli._prompt_existing_network_volume"):
        _execute_run(mock_config, dry_run=False, yes=True)
    
    mock_select.assert_called_once()
    assert mock_plan.config.auto_network_volume is False
    assert mock_plan.network_volume_id == ""
