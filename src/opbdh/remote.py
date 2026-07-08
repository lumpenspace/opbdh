from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_RUNPOD_CONTAINER_DISK_GB,
    DEFAULT_RUNPOD_IMAGE,
    DEFAULT_RUNPOD_MIN_RAM_PER_GPU_GB,
    DEFAULT_RUNPOD_MIN_VCPU_PER_GPU,
    DEFAULT_RUNPOD_VOLUME_GB,
)


DEFAULT_RUNPOD_GPU_TYPES = (
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 NVL",
    "NVIDIA H100 80GB HBM3",
)
RUNPOD_CACHE_ROOT = "/root/.cache/opbdh"
RUNPOD_NETWORK_CACHE_ROOT = "/workspace/opbdh-cache"


@dataclass(frozen=True, slots=True)
class RunpodSshTarget:
    host: str
    port: int

    def label(self) -> str:
        return f"{self.host}:{self.port}"


def runpod_api_token(api_token: str | None = None) -> str:
    token = (api_token or os.environ.get("RUNPOD_API_TOKEN") or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not token:
        raise ValueError("RUNPOD_API_TOKEN or RUNPOD_API_KEY is required")
    return token


def _runpod_rest(
    method: str,
    path: str,
    *,
    api_token: str | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 60,
    search_from: Path | None = None,
) -> dict[str, Any] | list[Any] | None:
    del search_from
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1{path}",
        data=(json.dumps(body).encode("utf-8") if body is not None else None),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {runpod_api_token(api_token)}",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"RunPod API {method} {path} failed with HTTP {exc.code}: {detail or exc.reason}") from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def runpod_gpu_types() -> list[str]:
    configured = os.environ.get("OPBDH_RUNPOD_GPU_TYPES", "").strip()
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return list(DEFAULT_RUNPOD_GPU_TYPES)


def create_runpod_pod(
    *,
    name: str,
    cloud_type: str,
    public_key: str,
    gpu_types: list[str] | None = None,
    image: str | None = None,
    volume_gb: int | None = None,
    container_disk_gb: int | None = None,
    min_vcpu_per_gpu: int | None = None,
    min_ram_per_gpu_gb: int | None = None,
    network_volume_id: str | None = None,
    search_from: Path | None = None,
) -> tuple[str, str, str]:
    del search_from
    last_error: Exception | None = None
    configured_cloud = (cloud_type or "SECURE").strip().upper()
    cloud_options = ["SECURE", "COMMUNITY"] if configured_cloud == "ALL" else [configured_cloud]
    for gpu_type in gpu_types or runpod_gpu_types():
        for effective_cloud in cloud_options:
            body: dict[str, Any] = {
                "cloudType": effective_cloud,
                "computeType": "GPU",
                "gpuCount": 1,
                "gpuTypeIds": [gpu_type],
                "gpuTypePriority": "availability",
                "containerDiskInGb": int(container_disk_gb) if container_disk_gb is not None else DEFAULT_RUNPOD_CONTAINER_DISK_GB,
                "minVCPUPerGPU": int(min_vcpu_per_gpu) if min_vcpu_per_gpu is not None else DEFAULT_RUNPOD_MIN_VCPU_PER_GPU,
                "minRAMPerGPU": int(min_ram_per_gpu_gb) if min_ram_per_gpu_gb is not None else DEFAULT_RUNPOD_MIN_RAM_PER_GPU_GB,
                "name": name[:190],
                "imageName": (image or "").strip() or DEFAULT_RUNPOD_IMAGE,
                "ports": ["22/tcp"],
                "supportPublicIp": True,
                "volumeMountPath": "/workspace",
                "env": {"SSH_PUBLIC_KEY": public_key, "PUBLIC_KEY": public_key},
            }
            if (network_volume_id or "").strip():
                body["networkVolumeId"] = str(network_volume_id).strip()
            else:
                body["volumeInGb"] = int(volume_gb) if volume_gb is not None else DEFAULT_RUNPOD_VOLUME_GB
            try:
                data = _runpod_rest("POST", "/pods", body=body)
                if not isinstance(data, dict):
                    raise RuntimeError(f"unexpected RunPod create response: {data!r}")
                target = extract_runpod_ssh_target(data)
                return str(data["id"]), target.label() if target else "", gpu_type
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"failed to create RunPod pod for configured GPU types: {last_error}")


def extract_runpod_ssh_target(pod: dict[str, Any]) -> RunpodSshTarget | None:
    public_ip = str(pod.get("publicIp") or "").strip()
    port_mappings = pod.get("portMappings")
    mapped_port: Any = None
    if isinstance(port_mappings, dict):
        mapped_port = port_mappings.get("22") or port_mappings.get(22)
    if not public_ip or mapped_port in {None, ""}:
        return None
    return RunpodSshTarget(host=public_ip, port=int(mapped_port))


def wait_for_runpod_pod(pod_id: str, *, search_from: Path | None = None, timeout_seconds: int = 1200) -> dict[str, Any]:
    del search_from
    deadline = time.time() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        pod = _runpod_rest("GET", f"/pods/{pod_id}?includeMachine=true")
        if isinstance(pod, dict):
            last = pod
            desired_status = str(pod.get("desiredStatus") or "").strip().upper()
            if desired_status == "RUNNING" and extract_runpod_ssh_target(pod):
                return pod
            if desired_status in {"EXITED", "TERMINATED"}:
                raise RuntimeError(f"RunPod pod {pod_id} stopped before SSH became available: {pod}")
        time.sleep(10)
    raise TimeoutError(f"RunPod pod {pod_id} did not expose publicIp and portMappings[22] before timeout: {last}")


def delete_runpod_pod(pod_id: str, *, search_from: Path | None = None) -> None:
    del search_from
    _runpod_rest("DELETE", f"/pods/{pod_id}")


# Pods are ephemeral and RunPod reuses host:port pairs, so pinning host keys in
# ~/.ssh/known_hosts would only cause spurious key-changed failures later.
_SSH_COMMON_OPTIONS = (
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ConnectTimeout=10",
)


def ssh_base(ssh_target: RunpodSshTarget, key_path: Path) -> list[str]:
    return [
        "ssh",
        *_SSH_COMMON_OPTIONS,
        "-p",
        str(ssh_target.port),
        "-i",
        str(key_path.expanduser()),
        f"root@{ssh_target.host}",
    ]


def scp_base(ssh_target: RunpodSshTarget, key_path: Path) -> list[str]:
    return [
        "scp",
        *_SSH_COMMON_OPTIONS,
        "-P",
        str(ssh_target.port),
        "-i",
        str(key_path.expanduser()),
    ]


def remote_bash_command(script: str) -> str:
    return "bash -lc " + shlex.quote(script)


def wait_for_ssh(ssh_target: RunpodSshTarget, key_path: Path, timeout_seconds: int = 1200) -> None:
    deadline = time.time() + timeout_seconds
    command = ssh_base(ssh_target, key_path) + ["echo", "ready"]
    while time.time() < deadline:
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0 and "ready" in completed.stdout:
            return
        time.sleep(10)
    raise TimeoutError(f"ssh to root@{ssh_target.host}:{ssh_target.port} did not become ready")
