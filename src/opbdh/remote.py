from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
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


class InsufficientCreditsError(RuntimeError):
    """RunPod refused to create a pod because the account lacks credit.

    ``status_code`` is the provider's HTTP status. 402 is definitive; anything
    else means the classification came from wording in the provider's message
    and should be checked against the real balance before being reported as a
    money problem.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class RunpodBalance:
    client_balance: float | None
    current_spend_per_hour: float | None


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


def runpod_balance(api_token: str | None = None, *, timeout: int = 5) -> RunpodBalance | None:
    """Return RunPod account credit and current hourly spend, if available.

    Account visibility is advisory and must never prevent a launch, so token,
    HTTP, and response-parsing failures all return ``None``.
    """
    try:
        token = runpod_api_token(api_token)
        query = "query { myself { clientBalance currentSpendPerHr } }"
        request = urllib.request.Request(
            "https://api.runpod.io/graphql?" + urllib.parse.urlencode({"api_key": token}),
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "opbdh/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") if isinstance(payload, dict) else None
        myself = data.get("myself") if isinstance(data, dict) else None
        if not isinstance(myself, dict):
            return None
        client_balance = myself.get("clientBalance")
        current_spend = myself.get("currentSpendPerHr")
        return RunpodBalance(
            client_balance=float(client_balance) if client_balance is not None else None,
            current_spend_per_hour=float(current_spend) if current_spend is not None else None,
        )
    except Exception:
        return None


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
        message = f"RunPod API {method} {path} failed with HTTP {exc.code}: {detail or exc.reason}"
        credit_words = ("credit", "balance", "funds", "payment")
        provider_detail = f"{detail} {exc.reason}".lower()
        is_pod_create = method.upper() == "POST" and path == "/pods"
        if is_pod_create and (exc.code == 402 or any(word in provider_detail for word in credit_words)):
            # Wording alone is a suspicion, not a verdict — providers mention
            # "check your credit balance" in capacity errors too. The status
            # travels so the caller can confirm against the real balance.
            raise InsufficientCreditsError(message, status_code=exc.code) from exc
        raise RuntimeError(message) from exc
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
    gpu_count: int | None = None,
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
                "gpuCount": max(1, int(gpu_count)) if gpu_count is not None else 1,
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
            except InsufficientCreditsError:
                raise
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
