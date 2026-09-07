from unittest.mock import MagicMock, Mock, patch

from typer.testing import CliRunner

from opbdh import cli
from opbdh.serve import normalize_endpoint_name


runner = CliRunner()


def _mock_endpoint(
    name: str = "test-model-ep",
    repo: str = "org/test-model",
    state: str = "running",
    url: str = "https://test-model-ep.endpoints.huggingface.cloud",
) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.repository = repo
    ep.status = state
    ep.url = url
    ep.raw = {
        "name": name,
        "model": {"repository": repo, "framework": "custom", "revision": "main", "task": "text-generation"},
        "status": {"state": state, "url": url, "createdAt": "2026-09-07T12:00:00Z", "updatedAt": "2026-09-07T12:05:00Z"},
        "compute": {
            "accelerator": "gpu",
            "instanceType": "nvidia-a10g",
            "instanceSize": "x1",
            "scaling": {"minReplica": 0, "maxReplica": 1, "scaleToZeroTimeout": 15},
        },
        "provider": {"vendor": "aws", "region": "us-east-1"},
    }
    return ep


def test_normalize_endpoint_name():
    assert normalize_endpoint_name("lumpenspace/reword-grpo-scaled") == "reword-grpo-scaled-ep"
    assert normalize_endpoint_name("Qwen/Qwen2.5-32B", "custom-ep") == "custom-ep"
    assert normalize_endpoint_name("user/model_with_underscores") == "model-with-underscores-ep"
    # Max length 32 chars
    long_name = "a" * 50
    assert len(normalize_endpoint_name("test", long_name)) <= 32


def test_serve_list_renders_table(monkeypatch):
    mock_api = Mock()
    mock_api.list_inference_endpoints.return_value = [
        _mock_endpoint("my-model-ep", "lumpenspace/my-model", "running"),
        _mock_endpoint("second-ep", "org/other", "paused", url=None),
    ]

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        result = runner.invoke(cli.app, ["serve", "list"])

    assert result.exit_code == 0
    assert "Hugging Face Inference Endpoints" in result.output
    assert "my-model-ep" in result.output
    assert "lumpenspace" in result.output
    assert "my-model" in result.output
    assert "second-ep" in result.output
    assert "org/other" in result.output
    assert "running" in result.output
    assert "paused" in result.output


def test_serve_list_json(monkeypatch):
    mock_api = Mock()
    mock_api.list_inference_endpoints.return_value = [
        _mock_endpoint("json-ep", "org/json-model", "running"),
    ]

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        result = runner.invoke(cli.app, ["serve", "list", "--json"])

    assert result.exit_code == 0
    assert '"name": "json-ep"' in result.output
    assert '"repository": "org/json-model"' in result.output


def test_serve_create_calls_api_and_prints_plan(monkeypatch):
    mock_api = Mock()
    created_ep = _mock_endpoint("reword-ep", "lumpenspace/reword-grpo-scaled", "running")
    mock_api.create_inference_endpoint.return_value = created_ep

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        result = runner.invoke(
            cli.app,
            [
                "serve",
                "create",
                "lumpenspace/reword-grpo-scaled",
                "--name",
                "reword-ep",
                "--instance-type",
                "nvidia-a10g",
                "--scale-to-zero",
                "15",
                "--no-wait",
                "-y",
            ],
        )

    assert result.exit_code == 0
    assert "reword-ep" in result.output
    assert "created successfully" in result.output
    mock_api.create_inference_endpoint.assert_called_once()
    kwargs = mock_api.create_inference_endpoint.call_args.kwargs
    assert kwargs["name"] == "reword-ep"
    assert kwargs["repository"] == "lumpenspace/reword-grpo-scaled"
    assert kwargs["instance_type"] == "nvidia-a10g"
    assert kwargs["scale_to_zero_timeout"] == 15
    assert kwargs["min_replica"] == 0  # Enabled scale-to-zero


def test_serve_status(monkeypatch):
    mock_api = Mock()
    mock_api.get_inference_endpoint.return_value = _mock_endpoint("status-ep", "org/status-model", "running")

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        result = runner.invoke(cli.app, ["serve", "status", "status-ep"])

    assert result.exit_code == 0
    assert "Endpoint: status-ep" in result.output
    assert "org/status-model" in result.output
    assert "running" in result.output


def test_serve_pause_and_resume(monkeypatch):
    mock_ep = _mock_endpoint("lifecycle-ep", "org/model", "running")
    mock_api = Mock()
    mock_api.get_inference_endpoint.return_value = mock_ep

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        pause_res = runner.invoke(cli.app, ["serve", "pause", "lifecycle-ep"])
        assert pause_res.exit_code == 0
        assert "paused" in pause_res.output
        mock_ep.pause.assert_called_once()

        resume_res = runner.invoke(cli.app, ["serve", "resume", "lifecycle-ep"])
        assert resume_res.exit_code == 0
        assert "resumed" in resume_res.output
        mock_ep.resume.assert_called_once()


def test_serve_delete(monkeypatch):
    mock_api = Mock()

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        result = runner.invoke(cli.app, ["serve", "delete", "target-ep", "-y"])

    assert result.exit_code == 0
    assert "target-ep' deleted" in result.output
    mock_api.delete_inference_endpoint.assert_called_once_with(name="target-ep", namespace=None)


def test_serve_test_inference(monkeypatch):
    mock_client = Mock()
    mock_client.text_generation.return_value = "Generated text from cloud model."

    with (
        patch("huggingface_hub.InferenceClient", return_value=mock_client),
        patch("opbdh.serve.list_endpoints", return_value=[]),
    ):
        result = runner.invoke(
            cli.app,
            ["serve", "test", "https://endpoint-url.cloud", "--prompt", "Hello model"],
        )

    assert result.exit_code == 0
    assert "Generated text from cloud model." in result.output
    mock_client.text_generation.assert_called_once()
    assert mock_client.text_generation.call_args.args[0] == "Hello model"


def test_permission_error_displays_actionable_advice(monkeypatch):
    mock_api = Mock()
    mock_api.list_inference_endpoints.side_effect = Exception("403 Forbidden: missing permissions: inference.endpoints.read")

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        result = runner.invoke(cli.app, ["serve", "list"])

    assert result.exit_code == 1
    assert "lacks permissions to manage Inference Endpoints" in result.output
    assert "https://huggingface.co/settings/tokens" in result.output
