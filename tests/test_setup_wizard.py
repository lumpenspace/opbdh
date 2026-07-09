import json
from unittest.mock import patch

from typer.testing import CliRunner

from opbdh.cli import app
from opbdh.setup_wizard import is_configured, run_setup_wizard

runner = CliRunner()


def test_is_configured_false_without_any_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPBDH_CONFIG", str(tmp_path / "nope.json"))
    assert not is_configured()


def test_is_configured_true_with_global_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPBDH_CONFIG", str(config_path))
    assert is_configured()


def test_is_configured_true_with_local_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPBDH_CONFIG", str(tmp_path / "nope.json"))
    (tmp_path / "opbdh.json").write_text("{}", encoding="utf-8")
    assert is_configured()


@patch("opbdh.cli._stdin_is_tty", return_value=True)
@patch("opbdh.setup_wizard.run_setup_wizard", return_value=0)
@patch("opbdh.setup_wizard.is_configured", return_value=False)
def test_bare_invocation_runs_wizard_when_unconfigured(mock_configured, mock_wizard, mock_tty):
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    mock_wizard.assert_called_once()


@patch("opbdh.cli._stdin_is_tty", return_value=True)
@patch("opbdh.setup_wizard.run_setup_wizard", return_value=0)
@patch("opbdh.setup_wizard.is_configured", return_value=True)
def test_bare_invocation_shows_help_when_configured(mock_configured, mock_wizard, mock_tty):
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    mock_wizard.assert_not_called()
    assert "Usage" in result.output


@patch("opbdh.cli._stdin_is_tty", return_value=False)
@patch("opbdh.setup_wizard.run_setup_wizard", return_value=0)
@patch("opbdh.setup_wizard.is_configured", return_value=False)
def test_bare_invocation_never_runs_wizard_without_tty(mock_configured, mock_wizard, mock_tty):
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    mock_wizard.assert_not_called()


def test_wizard_writes_config(tmp_path, monkeypatch):
    target = tmp_path / "config" / "config.json"
    monkeypatch.setenv("OPBDH_CONFIG", str(target))
    monkeypatch.delenv("PRIME_INTELLECT_API_KEY", raising=False)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)

    answers = iter([
        "primeintellect",        # provider
        "Qwen/Qwen2.5-0.5B",     # model
        "{cwd}/run.py",          # code
        48,                      # vram
        2.0,                     # max $/hr
        3.0,                     # max spend
        "global",                # scope
    ])
    with (
        patch("clypi.prompt", side_effect=lambda *a, **k: next(answers)),
        patch("clypi.confirm", return_value=False),
    ):
        assert run_setup_wizard() == 0

    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["provider"] == "primeintellect"
    assert saved["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert saved["vram_gb"] == 48
    assert saved["max_dollars_per_hour"] == 2.0
    assert saved["max_spend_dollars"] == 3.0
    assert saved["auto_network_volume"] is False


def test_wizard_handles_interrupt(monkeypatch):
    with patch("clypi.prompt", side_effect=KeyboardInterrupt):
        assert run_setup_wizard() == 1
