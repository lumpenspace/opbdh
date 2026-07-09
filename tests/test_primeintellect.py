from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from opbdh.config import OpbdhConfig
from opbdh.primeintellect import (
    _select_image,
    ensure_pi_ssh_key,
    extract_pi_ssh_target,
    find_pi_offers,
)
from opbdh.runpod import OpbdhPlan, make_plan, run_plan


def _offer(**overrides):
    offer = {
        "cloudId": "n3-A100x1",
        "gpuType": "A100_80GB",
        "socket": "SXM4",
        "provider": "hyperstack",
        "gpuMemory": 80,
        "stockStatus": "High",
        "isSpot": False,
        "prices": {"onDemand": 1.5},
        "images": ["ubuntu_22_cuda_12", "cuda_12_4_pytorch_2_5"],
        "disk": {"min": 40, "max": 500},
    }
    offer.update(overrides)
    return offer


def test_extract_pi_ssh_target_from_connection_string():
    target = extract_pi_ssh_target({"sshConnection": "root@135.23.125.123 -p 2222"})
    assert target is not None
    assert target.host == "135.23.125.123"
    assert target.port == 2222


def test_extract_pi_ssh_target_defaults_to_port_22():
    target = extract_pi_ssh_target({"sshConnection": ["ubuntu@gpu-77.example.com"]})
    assert target is not None
    assert target.host == "gpu-77.example.com"
    assert target.port == 22


def test_extract_pi_ssh_target_falls_back_to_ip_and_port_mapping():
    pod = {
        "sshConnection": None,
        "ip": ["10.0.0.5"],
        "primePortMapping": [
            {"internal": "8888", "external": "40001", "protocol": "TCP", "usedBy": "JUPYTER_NOTEBOOK"},
            {"internal": "22", "external": "40002", "protocol": "TCP", "usedBy": "SSH"},
        ],
    }
    target = extract_pi_ssh_target(pod)
    assert target is not None
    assert target.host == "10.0.0.5"
    assert target.port == 40002


def test_extract_pi_ssh_target_returns_none_without_endpoint():
    assert extract_pi_ssh_target({"sshConnection": None, "ip": None}) is None


@patch("opbdh.primeintellect._pi_rest")
def test_find_pi_offers_filters_and_sorts(mock_rest):
    mock_rest.return_value = {
        "items": [
            _offer(gpuType="H100_80GB", prices={"onDemand": 2.5}),
            _offer(gpuType="A100_80GB", prices={"onDemand": 1.5}),
            _offer(gpuType="RTX4090_24GB", gpuMemory=24),
            _offer(gpuType="B200_180GB", gpuMemory=180, prices={"onDemand": 9.0}),
            _offer(gpuType="H100_80GB", isSpot=True, prices={"onDemand": 0.9}),
            _offer(gpuType="H100_80GB", stockStatus="Unavailable"),
            _offer(gpuType="H200_141GB", gpuMemory=141, prices={}),
        ]
    }
    offers = find_pi_offers(min_vram_gb=48, max_dollars_per_hour=5.0, cloud_type="SECURE")
    assert [o["gpuType"] for o in offers] == ["A100_80GB", "H100_80GB"]
    assert "security=secure_cloud" in mock_rest.call_args.args[1]


@patch("opbdh.primeintellect._pi_rest")
def test_ensure_pi_ssh_key_reuses_matching_key(mock_rest):
    mock_rest.return_value = {
        "data": [{"id": "key-1", "publicKey": "ssh-ed25519 AAAAC3Nz some-other-comment"}]
    }
    key_id = ensure_pi_ssh_key("ssh-ed25519 AAAAC3Nz opbdh")
    assert key_id == "key-1"
    assert mock_rest.call_count == 1


@patch("opbdh.primeintellect._pi_rest")
def test_ensure_pi_ssh_key_uploads_when_missing(mock_rest):
    mock_rest.side_effect = [
        {"data": []},
        {"id": "key-9"},
    ]
    key_id = ensure_pi_ssh_key("ssh-ed25519 AAAAC3Nz opbdh")
    assert key_id == "key-9"
    method, path = mock_rest.call_args.args[:2]
    assert (method, path) == ("POST", "/ssh_keys/")


def test_select_image_ignores_docker_tags_and_prefers_pytorch():
    offer = _offer()
    assert _select_image(offer, "runpod/pytorch:2.8.0-py3.11") == "cuda_12_4_pytorch_2_5"
    assert _select_image(offer, "ubuntu_26") == "ubuntu_26"


def _write_code(tmp_path: Path) -> Path:
    code = tmp_path / "train.py"
    code.write_text("print('hi')\n", encoding="utf-8")
    return code


@patch("opbdh.runpod.find_pi_offers")
def test_make_plan_ignores_network_volumes_for_primeintellect(mock_offers, tmp_path):
    mock_offers.return_value = [_offer()]
    config = OpbdhConfig(
        provider="primeintellect",
        auto_network_volume=True,
        network_volume_id="vol-123",
        network_volume_data_center_id="EU-RO-1",
        results_dir=str(tmp_path / "results"),
    )
    plan = make_plan(config, code_path=_write_code(tmp_path))
    assert plan.network_volume_id == ""
    assert plan.network_volume_size_gb is None


def test_make_plan_rejects_unknown_provider(tmp_path):
    config = OpbdhConfig(provider="vastai")
    with pytest.raises(ValueError, match="unsupported provider"):
        make_plan(config, code_path=_write_code(tmp_path))


@patch("opbdh.runpod.find_pi_offers")
def test_make_plan_uses_live_offers_for_primeintellect(mock_offers, tmp_path):
    mock_offers.return_value = [_offer(), _offer(gpuType="H100_80GB", prices={"onDemand": 2.5})]
    config = OpbdhConfig(provider="primeintellect", vram_gb=48, results_dir=str(tmp_path / "results"))
    plan = make_plan(config, code_path=_write_code(tmp_path))
    assert plan.gpu_type_ids == ["A100_80GB (hyperstack/n3-A100x1)", "H100_80GB (hyperstack/n3-A100x1)"]
    assert plan.estimated_hourly_dollars == 1.5


@pytest.fixture
def pi_plan(tmp_path):
    config = OpbdhConfig(
        code="fake_path.py",
        model_id="fake-model",
        provider="primeintellect",
        failure_keepalive_seconds=0,
    )
    return OpbdhPlan(
        run_id="test_run",
        config=config,
        code_path=Path("fake_path.py"),
        command="python fake_path.py",
        gpu_type_ids=["A100_80GB (hyperstack/n3-A100x1)"],
        estimated_hourly_dollars=1.5,
        model_size_gb=5.0,
        network_volume_id="",
        network_volume_size_gb=None,
        results_dir=tmp_path / "fake_results",
        verification_checked=set(),
    )


@patch("opbdh.runpod.resolve_ssh_key_paths")
@patch("opbdh.runpod.ensure_pi_ssh_key", return_value="key-1")
@patch("opbdh.runpod.find_pi_offers")
@patch("opbdh.runpod.create_pi_pod", return_value=("pod-42", "A100_80GB (hyperstack/n3-A100x1)", 1.5))
@patch("opbdh.runpod.wait_for_pi_pod")
@patch("opbdh.runpod.extract_pi_ssh_target")
@patch("opbdh.runpod.wait_for_ssh")
@patch("opbdh.runpod._upload_bundle")
@patch("opbdh.runpod._start_remote_job")
@patch("opbdh.runpod._remote_status", return_value=("done", 0))
@patch("opbdh.runpod.sync_results_from_pod")
@patch("opbdh.runpod.delete_pi_pod")
def test_run_plan_uses_primeintellect_lifecycle(
    mock_delete, mock_sync, mock_status, mock_start, mock_upload, mock_wait_ssh,
    mock_extract, mock_wait_pod, mock_create, mock_offers, mock_ensure_key, mock_resolve_keys, pi_plan
):
    mock_resolve_keys.return_value = (MagicMock(), MagicMock())
    mock_offers.return_value = [_offer()]
    mock_extract.return_value = MagicMock(host="10.0.0.5", port=40002)

    result = run_plan(pi_plan, dry_run=False)

    assert result is not None
    assert result.pod_id == "pod-42"
    assert result.returncode == 0
    mock_delete.assert_called_once_with("pod-42")
    mock_create.assert_called_once()
    assert mock_create.call_args.kwargs["ssh_key_id"] == "key-1"
