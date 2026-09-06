from types import SimpleNamespace
import pytest
from opbdh.execution import local_accelerators, plan_execution, require_local_capacity, launch_local


def test_cloud_default_and_local_capacity_gate():
    remote = plan_execution()
    assert remote.target == 'runpod'
    assert remote.required_gb == 80
    assert require_local_capacity('mps', required_gb=80, capacities={'mps': 90}).sufficient
    with pytest.raises(ValueError):
        require_local_capacity('cuda', required_gb=80, capacities={'cuda': 24})
    with pytest.raises(ValueError):
        require_local_capacity('runpod', required_gb=80)
    for invalid in [0, -1, float('nan')]:
        with pytest.raises(ValueError):
            plan_execution(required_gb=invalid)


def test_inventory_uses_free_memory_not_combined_gpus():
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=lambda i: (20*2**30, 80*2**30)),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                           mps=SimpleNamespace(recommended_max_memory=lambda: 100*2**30, driver_allocated_memory=lambda: 15*2**30))
    assert local_accelerators(torch_module=fake) == {'cuda': 20, 'mps': 85}


def test_local_launcher_preserves_argv_and_environment(monkeypatch, tmp_path):
    import opbdh.execution as module
    monkeypatch.setattr(module, 'require_local_capacity', lambda *a, **kw: None)
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda argv, **kwargs: calls.append((argv, kwargs)))
    launch_local(['python', 'job.py', 'literal;value'], target='mps', required_gb=80, cwd=tmp_path, env={'CUSTOM': 'value'})
    args, kwargs = calls[0]
    assert args[-1] == 'literal;value'
    assert kwargs['env']['OPBDH_DEVICE'] == 'mps'
    assert kwargs['env']['CUSTOM'] == 'value'
    assert 'shell' not in kwargs
