from unittest.mock import Mock

from modelchoice import ModelOption
from typer.testing import CliRunner

from opbdh import cli


runner = CliRunner()


def _options(query: str, *, limit: int = 25, token=None):
    del token
    return [
        ModelOption(
            f"huggingface:Org/{query}",
            f"Hugging Face · Org/{query}",
            "huggingface",
            f"Org/{query}",
            "huggingface-live",
            "text-generation · 1,234 downloads",
        )
    ][:limit]


def test_questionary_picker_uses_shared_huggingface_options(monkeypatch):
    monkeypatch.setattr(cli, "_huggingface_model_options", _options)
    questionary = Mock()
    questionary.text.return_value.ask.return_value = "Qwen"
    questionary.autocomplete.return_value.ask.return_value = "Org/Qwen"

    assert cli._questionary_model(questionary) == "Org/Qwen"
    assert questionary.autocomplete.call_args.kwargs["choices"] == ["Org/Qwen"]


def test_models_search_renders_shared_catalog_details(monkeypatch):
    monkeypatch.setattr(cli, "_huggingface_model_options", _options)

    result = runner.invoke(cli.app, ["models", "search", "Qwen", "--limit", "5"])

    assert result.exit_code == 0
    assert "Org/Qwen" in result.output
    assert "1,234 downloads" in result.output
