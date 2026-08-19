"""gpu_count plumbing: config field, plan arithmetic, pod payload."""

from opbdh.config import OpbdhConfig
from opbdh.runpod import make_plan


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
