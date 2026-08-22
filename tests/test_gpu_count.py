"""gpu_count plumbing: config field, plan arithmetic, pod payload."""

from opbdh import primeintellect
from opbdh import runpod as runpod_module
from opbdh.config import OpbdhConfig
from opbdh.runpod import make_plan


def _pi_offer(hourly: float = 2.0, memory: int = 80) -> dict:
    return {
        "cloudId": "c1",
        "gpuType": "H100_80GB",
        "socket": "PCIe",
        "provider": "primecompute",
        "gpuMemory": memory,
        "prices": {"onDemand": hourly},
        "stockStatus": "Available",
    }


def test_config_default_gpu_count_is_one():
    assert OpbdhConfig().gpu_count == 1


def test_plan_estimate_scales_with_gpu_count(tmp_path):
    code = tmp_path / "run.py"
    code.write_text("print('hi')\n")
    single = make_plan(
        OpbdhConfig(code=str(code), vram_gb=24, gpu_count=1, pre_download_model=False),
        code_path=code,
    )
    double = make_plan(
        OpbdhConfig(code=str(code), vram_gb=24, gpu_count=2, pre_download_model=False),
        code_path=code,
    )
    assert double.estimated_hourly_dollars == 2 * single.estimated_hourly_dollars


def test_plan_cap_applies_to_whole_pod(tmp_path):
    code = tmp_path / "run.py"
    code.write_text("print('hi')\n")
    # A cap that admits one cheap GPU must exclude two of them.
    single = make_plan(
        OpbdhConfig(code=str(code), vram_gb=24, gpu_count=1, max_dollars_per_hour=1.0, pre_download_model=False),
        code_path=code,
    )
    assert single.gpu_type_ids
    try:
        make_plan(
            OpbdhConfig(code=str(code), vram_gb=24, gpu_count=4, max_dollars_per_hour=1.0, pre_download_model=False),
            code_path=code,
        )
    except ValueError as exc:
        assert "x4" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("4-GPU pod should not fit under a $1/hr cap")


def test_pi_plan_threads_gpu_count_and_scales_estimate(tmp_path, monkeypatch):
    calls = {}

    def fake_find(*, min_vram_gb, max_dollars_per_hour, cloud_type, gpu_count=1):
        calls["gpu_count"] = gpu_count
        calls["per_gpu_cap"] = max_dollars_per_hour
        return [_pi_offer(hourly=2.0)]

    monkeypatch.setattr(runpod_module, "find_pi_offers", fake_find)
    code = tmp_path / "run.py"
    code.write_text("print('hi')\n")
    plan = make_plan(
        OpbdhConfig(
            code=str(code),
            provider="primeintellect",
            vram_gb=24,
            gpu_count=4,
            max_dollars_per_hour=12.0,
            pre_download_model=False,
        ),
        code_path=code,
    )
    # offers are queried per 4-GPU configuration, capped per GPU
    assert calls["gpu_count"] == 4
    assert calls["per_gpu_cap"] == 3.0
    # per-GPU offer price scales to the whole pod
    assert plan.estimated_hourly_dollars == 8.0


def test_pi_pod_payload_carries_gpu_count(monkeypatch):
    bodies = []

    def fake_rest(method, path, *, api_token=None, body=None, timeout=60):
        bodies.append((method, path, body))
        return {"id": "pod-1", "priceHr": 7.5}

    monkeypatch.setattr(primeintellect, "_pi_rest", fake_rest)
    pod_id, label, hourly = primeintellect.create_pi_pod(
        name="opbdh-test", offers=[_pi_offer()], ssh_key_id="k1", gpu_count=4
    )
    assert pod_id == "pod-1"
    assert bodies[0][2]["pod"]["gpuCount"] == 4
    assert hourly == 7.5  # the API's whole-pod price wins when present


def test_pi_pod_hourly_fallback_scales_with_gpu_count(monkeypatch):
    def fake_rest(method, path, *, api_token=None, body=None, timeout=60):
        return {"id": "pod-1"}  # no priceHr: fall back to per-GPU price x count

    monkeypatch.setattr(primeintellect, "_pi_rest", fake_rest)
    _, _, hourly = primeintellect.create_pi_pod(
        name="opbdh-test", offers=[_pi_offer(hourly=2.0)], ssh_key_id="k1", gpu_count=4
    )
    assert hourly == 8.0
