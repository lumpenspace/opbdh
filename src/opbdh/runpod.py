from __future__ import annotations

import io
import json
import os
import select
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .remote import (
    RUNPOD_CACHE_ROOT,
    RUNPOD_NETWORK_CACHE_ROOT,
    RunpodSshTarget,
    _runpod_rest,
    create_runpod_pod,
    delete_runpod_pod,
    extract_runpod_ssh_target,
    remote_bash_command,
    scp_base,
    ssh_base,
    wait_for_runpod_pod,
    wait_for_ssh,
)

from .config import OpbdhConfig, model_slug
from .gpu import candidate_gpus, estimated_hourly
from .hf import estimate_model_size_gb, suggested_network_volume_gb
from .verify import default_command_for_code, verify_code


RUN_ROOT = "/opbdh-run"
EXCLUDED_NAMES = {
    ".DS_Store",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "runpod_results",
}


@dataclass(slots=True)
class OpbdhPlan:
    run_id: str
    config: OpbdhConfig
    code_path: Path
    command: str
    gpu_type_ids: list[str]
    estimated_hourly_dollars: float | None
    model_size_gb: float | None
    network_volume_id: str
    network_volume_size_gb: int | None
    results_dir: Path
    verification_checked: list[Path]


@dataclass(slots=True)
class OpbdhRunResult:
    run_id: str
    pod_id: str
    gpu_type_id: str
    results_dir: Path
    returncode: int
    kept_pod: bool = False


def _should_include(relative: Path) -> bool:
    return not any(part in EXCLUDED_NAMES for part in relative.parts)


def _reset_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if not _should_include(Path(info.name)):
        return None
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _relative_user_path(path: Path, root: Path) -> Path:
    return Path("user") / path.relative_to(root)


def _add_code_to_tar(archive: tarfile.TarFile, code_path: Path) -> None:
    code_path = code_path.expanduser().resolve()
    if code_path.is_file():
        archive.add(code_path, arcname=str(Path("user") / code_path.name), filter=_reset_tar_info)
        return
    for path in sorted(code_path.rglob("*")):
        relative = path.relative_to(code_path)
        if not _should_include(relative):
            continue
        archive.add(path, arcname=str(_relative_user_path(path, code_path)), filter=_reset_tar_info)


def build_job_script(config: OpbdhConfig, *, command: str, network_volume_id: str) -> str:
    cache_root = RUNPOD_NETWORK_CACHE_ROOT if network_volume_id else RUNPOD_CACHE_ROOT
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    download_block = ""
    if config.pre_download_model and config.model_id:
        download_block = f"""
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={config.model_id!r},
    cache_dir={str(Path(cache_root) / "huggingface" / "hub")!r},
)
PY
echo "model cache ready" > logs/model_downloaded
""".strip()

    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
cd {RUN_ROOT}
mkdir -p logs results {shlex.quote(cache_root)}
on_exit() {{
  code=$?
  echo "$code" > logs/exit_code
  date -Is > logs/finished_at
  exit "$code"
}}
trap on_exit EXIT
date -Is > logs/started_at
python3 -m pip install --upgrade pip
python3 -m pip install "huggingface-hub>=1.3,<2" "transformers>=5.2,<5.9"
if [ -f user/requirements.txt ]; then
  python3 -m pip install -r user/requirements.txt
fi
export HF_HOME={shlex.quote(str(Path(cache_root) / "huggingface"))}
export HUGGINGFACE_HUB_CACHE={shlex.quote(str(Path(cache_root) / "huggingface" / "hub"))}
export HF_XET_CACHE={shlex.quote(str(Path(cache_root) / "huggingface" / "xet"))}
export TRANSFORMERS_CACHE={shlex.quote(str(Path(cache_root) / "huggingface" / "transformers"))}
export XDG_CACHE_HOME={shlex.quote(str(Path(cache_root) / "xdg"))}
export PIP_CACHE_DIR={shlex.quote(str(Path(cache_root) / "pip"))}
export HF_TOKEN={shlex.quote(hf_token)}
export HUGGING_FACE_HUB_TOKEN={shlex.quote(hf_token)}
export OPBDH_MODEL_ID={shlex.quote(config.model_id)}
export OPBDH_RESULTS_DIR={RUN_ROOT}/results
export PYTHONUNBUFFERED=1
{download_block}
echo "running user command" > logs/user_command_started
{command}
echo "user command complete" > logs/user_command_completed
""".strip() + "\n"


def build_bundle(config: OpbdhConfig, *, code_path: Path, command: str, run_id: str, network_volume_id: str = "") -> bytes:
    buffer = io.BytesIO()
    manifest = {
        "run_id": run_id,
        "config": asdict(config),
        "command": command,
        "network_volume_id": network_volume_id,
    }
    script = build_job_script(config, command=command, network_volume_id=network_volume_id)
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        _add_code_to_tar(archive, code_path)
        for name, payload in {
            "opbdh_manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            "job.sh": script,
        }.items():
            encoded = payload.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(encoded)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            info.uid = 0
            info.gid = 0
            archive.addfile(info, io.BytesIO(encoded))
    return buffer.getvalue()


def create_network_volume(*, name: str, data_center_id: str, size_gb: int, search_from: Path | None = None) -> dict[str, Any]:
    payload = _runpod_rest(
        "POST",
        "/networkvolumes",
        body={"dataCenterId": data_center_id, "name": name[:191], "size": int(size_gb)},
        search_from=search_from,
    )
    if not isinstance(payload, dict) or not payload.get("id"):
        raise RuntimeError(f"unexpected RunPod network volume response: {payload!r}")
    return payload


def make_plan(config: OpbdhConfig, *, code_path: Path, run_id: str | None = None) -> OpbdhPlan:
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    code_path = code_path.expanduser().resolve()
    verification = verify_code(code_path, command=config.command)
    if not verification.ok:
        raise ValueError("Static verification failed:\n" + "\n".join(verification.errors))
    command = config.command.strip() or default_command_for_code(code_path)
    candidates = candidate_gpus(config.vram_gb, config.max_dollars_per_hour, config.cloud_type)
    if not candidates:
        raise ValueError(
            f"No configured RunPod GPU estimate satisfies {config.vram_gb} GB VRAM"
            f" under {config.max_dollars_per_hour}/hr."
        )
    network_volume_id = config.network_volume_id.strip()
    network_volume_size_gb = config.network_volume_size_gb
    model_estimate = estimate_model_size_gb(config.model_id) if config.model_id and config.auto_network_volume and not network_volume_size_gb else None
    if not network_volume_id and config.auto_network_volume:
        network_volume_size_gb = network_volume_size_gb or suggested_network_volume_gb(
            model_estimate,
            fallback_gb=config.pod_volume_gb,
        )
    first = candidates[0]
    results_dir = Path(config.results_dir).expanduser().resolve() / run_id
    return OpbdhPlan(
        run_id=run_id,
        config=config,
        code_path=code_path,
        command=command,
        gpu_type_ids=[gpu.id for gpu in candidates],
        estimated_hourly_dollars=first.hourly(config.cloud_type),
        model_size_gb=model_estimate.size_gb if model_estimate else None,
        network_volume_id=network_volume_id,
        network_volume_size_gb=network_volume_size_gb,
        results_dir=results_dir,
        verification_checked=verification.checked,
    )


def _ssh_key_paths(config: OpbdhConfig) -> tuple[Path, Path]:
    private_key = Path(config.ssh_key).expanduser()
    public_key = Path(config.ssh_public_key).expanduser()
    if not private_key.exists():
        raise FileNotFoundError(f"RunPod SSH private key not found: {private_key}")
    if not public_key.exists():
        raise FileNotFoundError(f"RunPod SSH public key not found: {public_key}")
    return private_key, public_key


def _upload_bundle(ssh_target: RunpodSshTarget, key_path: Path, bundle: bytes) -> None:
    remote_archive = "/tmp/opbdh-bundle.tar.gz"
    with tempfile.NamedTemporaryFile(prefix="opbdh-", suffix=".tar.gz", delete=False) as handle:
        handle.write(bundle)
        local_archive = Path(handle.name)
    try:
        upload = subprocess.run(
            scp_base(ssh_target, key_path) + [str(local_archive), f"root@{ssh_target.host}:{remote_archive}"],
            capture_output=True,
        )
        if upload.returncode != 0:
            raise RuntimeError(_completed_output(upload))
        extract = subprocess.run(
            ssh_base(ssh_target, key_path)
            + [
                remote_bash_command(
                    f"rm -rf {RUN_ROOT} && mkdir -p {RUN_ROOT} && "
                    f"tar --no-same-owner -xzf {shlex.quote(remote_archive)} -C {RUN_ROOT} && "
                    f"rm -f {shlex.quote(remote_archive)}"
                )
            ],
            capture_output=True,
        )
        if extract.returncode != 0:
            raise RuntimeError(_completed_output(extract))
    finally:
        local_archive.unlink(missing_ok=True)


def _completed_output(completed: subprocess.CompletedProcess[bytes]) -> str:
    stdout = completed.stdout.decode("utf-8", errors="ignore") if isinstance(completed.stdout, bytes) else str(completed.stdout or "")
    stderr = completed.stderr.decode("utf-8", errors="ignore") if isinstance(completed.stderr, bytes) else str(completed.stderr or "")
    return "\n".join(part for part in (stdout, stderr) if part.strip())


def _remote_text(ssh_target: RunpodSshTarget, key_path: Path, script: str) -> str:
    completed = subprocess.run(
        ssh_base(ssh_target, key_path) + [remote_bash_command(script)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout


def _start_remote_job(ssh_target: RunpodSshTarget, key_path: Path) -> str:
    return _remote_text(
        ssh_target,
        key_path,
        f"cd {RUN_ROOT} && mkdir -p logs results && "
        "(nohup bash job.sh > logs/stdout.log 2> logs/stderr.log & echo $! > logs/job.pid) && "
        "cat logs/job.pid",
    ).strip()


def _remote_status(ssh_target: RunpodSshTarget, key_path: Path) -> tuple[str, int | None]:
    output = _remote_text(
        ssh_target,
        key_path,
        f"cd {RUN_ROOT} && "
        "if [ -f logs/exit_code ]; then echo done:$(cat logs/exit_code); "
        "elif [ -f logs/job.pid ] && kill -0 $(cat logs/job.pid) 2>/dev/null; then echo running; "
        "else echo missing; fi",
    ).strip()
    if output.startswith("done:"):
        try:
            return "done", int(output.split(":", 1)[1])
        except ValueError:
            return "done", 1
    return output, None


def _stop_remote_job(ssh_target: RunpodSshTarget, key_path: Path) -> None:
    _remote_text(
        ssh_target,
        key_path,
        f"cd {RUN_ROOT} && if [ -f logs/job.pid ]; then kill $(cat logs/job.pid) 2>/dev/null || true; fi",
    )


def sync_results_from_pod(ssh_target: RunpodSshTarget, key_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ssh_base(ssh_target, key_path)
        + [remote_bash_command(f"cd {RUN_ROOT} && tar czf - logs results 2>/dev/null || true")],
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(_completed_output(completed))
    if not completed.stdout:
        return
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:gz") as archive:
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            archive.extractall(destination)


def _timed_yes_no(prompt: str, *, timeout_seconds: int, default: bool = False) -> bool:
    if not sys.stdin.isatty():
        return default
    sys.stdout.write(f"{prompt} [{'Y/n' if default else 'y/N'}] ")
    sys.stdout.flush()
    readable, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not readable:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return default
    answer = sys.stdin.readline().strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _append_local_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message.rstrip()}\n")


def ensure_network_volume(plan: OpbdhPlan) -> str:
    if plan.network_volume_id:
        return plan.network_volume_id
    if not plan.config.auto_network_volume:
        return ""
    if not plan.config.network_volume_data_center_id.strip():
        raise ValueError("--network-volume-data-center-id is required when --auto-network-volume creates a disk.")
    size_gb = plan.network_volume_size_gb or plan.config.pod_volume_gb
    volume_name = plan.config.network_volume_name or f"opbdh-{model_slug(plan.config.model_id)}"
    created = create_network_volume(
        name=volume_name,
        data_center_id=plan.config.network_volume_data_center_id,
        size_gb=size_gb,
        search_from=plan.code_path.parent,
    )
    return str(created["id"])


def run_plan(plan: OpbdhPlan, *, dry_run: bool = False) -> OpbdhRunResult | None:
    plan.results_dir.mkdir(parents=True, exist_ok=True)
    local_log = plan.results_dir / "opbdh.log"
    _append_local_log(local_log, f"OPBDH run {plan.run_id}")
    _append_local_log(local_log, f"Code: {plan.code_path}")
    _append_local_log(local_log, f"Command: {plan.command}")
    _append_local_log(local_log, f"GPU candidates: {', '.join(plan.gpu_type_ids)}")
    if dry_run:
        _append_local_log(local_log, "Dry run requested; not contacting RunPod.")
        return None

    private_key, public_key = _ssh_key_paths(plan.config)
    public_key_text = public_key.read_text(encoding="utf-8").strip()
    pod_id = ""
    ssh_target: RunpodSshTarget | None = None
    network_volume_id = ""
    selected_gpu_type = ""
    delete_pod = True
    try:
        network_volume_id = ensure_network_volume(plan)
        bundle = build_bundle(
            plan.config,
            code_path=plan.code_path,
            command=plan.command,
            run_id=plan.run_id,
            network_volume_id=network_volume_id,
        )
        _append_local_log(local_log, "Requesting RunPod pod.")
        pod_id, ssh_label, selected_gpu_type = create_runpod_pod(
            name=f"opbdh-{plan.run_id}",
            cloud_type=plan.config.cloud_type,
            public_key=public_key_text,
            gpu_types=plan.gpu_type_ids,
            image=plan.config.image,
            volume_gb=plan.config.pod_volume_gb,
            container_disk_gb=plan.config.container_disk_gb,
            network_volume_id=network_volume_id,
            search_from=plan.code_path.parent,
        )
        _append_local_log(local_log, f"Pod {pod_id} requested on {selected_gpu_type}; SSH hint: {ssh_label}.")
        pod = wait_for_runpod_pod(pod_id, search_from=plan.code_path.parent)
        ssh_target = extract_runpod_ssh_target(pod)
        if ssh_target is None:
            raise RuntimeError(f"RunPod pod {pod_id} is running but public SSH mapping was not returned")
        _append_local_log(local_log, f"Waiting for SSH at root@{ssh_target.host}:{ssh_target.port}.")
        wait_for_ssh(ssh_target, private_key)
        _append_local_log(local_log, "Uploading code bundle.")
        _upload_bundle(ssh_target, private_key, bundle)
        _append_local_log(local_log, "Bundle uploaded. Starting remote job.")
        remote_pid = _start_remote_job(ssh_target, private_key)
        _append_local_log(local_log, f"Remote PID {remote_pid}.")
        start = time.time()
        hourly = estimated_hourly(selected_gpu_type, plan.config.cloud_type) or plan.estimated_hourly_dollars or 0.0
        while True:
            status, returncode = _remote_status(ssh_target, private_key)
            sync_results_from_pod(ssh_target, private_key, plan.results_dir)
            if status == "done":
                final_code = returncode if returncode is not None else 1
                _append_local_log(local_log, f"Remote job finished with exit code {final_code}.")
                if final_code != 0:
                    raise RuntimeError(f"remote job failed with exit code {final_code}; see {plan.results_dir / 'logs'}")
                delete_pod = not plan.config.keep_pod_on_success
                return OpbdhRunResult(
                    run_id=plan.run_id,
                    pod_id=pod_id,
                    gpu_type_id=selected_gpu_type,
                    results_dir=plan.results_dir,
                    returncode=final_code,
                    kept_pod=not delete_pod,
                )
            if status != "running":
                raise RuntimeError(f"remote job status became {status!r}")
            if hourly > 0 and plan.config.max_spend_dollars > 0:
                spent = ((time.time() - start) / 3600) * hourly
                if spent >= plan.config.max_spend_dollars:
                    _stop_remote_job(ssh_target, private_key)
                    raise RuntimeError(f"max spend reached (${spent:.2f} >= ${plan.config.max_spend_dollars:.2f})")
            time.sleep(max(5, int(plan.config.poll_seconds)))
    except Exception as exc:
        _append_local_log(local_log, f"Failure: {exc}")
        if ssh_target is not None:
            try:
                sync_results_from_pod(ssh_target, private_key, plan.results_dir)
            except Exception as sync_exc:
                _append_local_log(local_log, f"Failure sync failed: {sync_exc}")
        if pod_id and ssh_target is not None:
            keep = _timed_yes_no(
                f"Run failed. Keep RunPod pod {pod_id} running for debugging?",
                timeout_seconds=max(1, int(plan.config.failure_keepalive_seconds)),
                default=False,
            )
            delete_pod = not keep
            if keep:
                _append_local_log(local_log, f"Keeping failed pod {pod_id} running by user request.")
        raise
    finally:
        if pod_id and delete_pod:
            try:
                _append_local_log(local_log, f"Deleting RunPod pod {pod_id}.")
                delete_runpod_pod(pod_id, search_from=plan.code_path.parent)
                _append_local_log(local_log, f"RunPod pod {pod_id} deleted.")
            except Exception as exc:
                _append_local_log(local_log, f"Pod deletion failed: {exc}")


def plan_summary(plan: OpbdhPlan) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "model_id": plan.config.model_id,
        "code_path": str(plan.code_path),
        "command": plan.command,
        "gpu_candidates": plan.gpu_type_ids,
        "estimated_hourly_dollars": plan.estimated_hourly_dollars,
        "max_spend_dollars": plan.config.max_spend_dollars,
        "model_size_gb": plan.model_size_gb,
        "network_volume_id": plan.network_volume_id,
        "network_volume_size_gb": plan.network_volume_size_gb,
        "results_dir": str(plan.results_dir),
        "verified_files": [str(path) for path in plan.verification_checked],
    }
