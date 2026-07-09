from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .remote import RunpodSshTarget

PI_API_BASE = "https://api.primeintellect.ai/api/v1"

# Environment images offered per availability item; prefer a recent PyTorch
# stack when the offer supports one so user code finds torch preinstalled.
_PREFERRED_IMAGES = (
    "cuda_12_6_pytorch_2_7",
    "cuda_12_4_pytorch_2_6",
    "cuda_12_4_pytorch_2_5",
    "cuda_12_4_pytorch_2_4",
    "cuda_12_1_pytorch_2_4",
    "cuda_12_1_pytorch_2_3",
    "cuda_12_1_pytorch_2_2",
    "ubuntu_22_cuda_12",
)


def pi_api_token(api_token: str | None = None) -> str:
    token = (
        api_token
        or os.environ.get("PRIME_INTELLECT_API_KEY")
        or os.environ.get("PRIME_API_KEY")
        or ""
    ).strip()
    if not token:
        raise ValueError("PRIME_INTELLECT_API_KEY or PRIME_API_KEY is required")
    return token


def _pi_rest(
    method: str,
    path: str,
    *,
    api_token: str | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any] | list[Any] | None:
    request = urllib.request.Request(
        f"{PI_API_BASE}{path}",
        data=(json.dumps(body).encode("utf-8") if body is not None else None),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {pi_api_token(api_token)}",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            f"Prime Intellect API {method} {path} failed with HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def _security_filter(cloud_type: str) -> str | None:
    cloud = (cloud_type or "").strip().upper()
    if cloud == "COMMUNITY":
        return "community_cloud"
    if cloud == "ALL":
        return None
    return "secure_cloud"


def offer_hourly(offer: dict[str, Any]) -> float | None:
    prices = offer.get("prices") or {}
    for key in ("onDemand", "communityPrice"):
        value = prices.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def offer_label(offer: dict[str, Any]) -> str:
    return f"{offer.get('gpuType', '?')} ({offer.get('provider', '?')}/{offer.get('cloudId', '?')})"


def find_pi_offers(
    *,
    min_vram_gb: int,
    max_dollars_per_hour: float | None,
    cloud_type: str,
    gpu_count: int = 1,
) -> list[dict[str, Any]]:
    query = f"?gpu_count={int(gpu_count)}&page_size=100"
    security = _security_filter(cloud_type)
    if security:
        query += f"&security={security}"
    payload = _pi_rest("GET", f"/availability/gpus{query}")
    items = payload.get("items", []) if isinstance(payload, dict) else []
    offers: list[dict[str, Any]] = []
    for offer in items:
        if not isinstance(offer, dict):
            continue
        if offer.get("isSpot"):
            continue
        if (offer.get("stockStatus") or "").lower() == "unavailable":
            continue
        memory = offer.get("gpuMemory")
        if not isinstance(memory, (int, float)) or memory < min_vram_gb:
            continue
        hourly = offer_hourly(offer)
        if hourly is None:
            continue
        if max_dollars_per_hour is not None and max_dollars_per_hour > 0 and hourly > max_dollars_per_hour:
            continue
        offers.append(offer)
    return sorted(offers, key=lambda o: (o.get("gpuMemory", 0), offer_hourly(o) or 0.0, offer_label(o)))


def ensure_pi_ssh_key(public_key_text: str) -> str:
    key_body = " ".join(public_key_text.strip().split()[:2])
    payload = _pi_rest("GET", "/ssh_keys/")
    existing = payload.get("data", []) if isinstance(payload, dict) else []
    for key in existing:
        if isinstance(key, dict) and " ".join(str(key.get("publicKey", "")).split()[:2]) == key_body:
            return str(key["id"])
    created = _pi_rest("POST", "/ssh_keys/", body={"name": "opbdh", "publicKey": public_key_text.strip()})
    if not isinstance(created, dict) or not created.get("id"):
        raise RuntimeError(f"unexpected Prime Intellect SSH key response: {created!r}")
    return str(created["id"])


def _select_image(offer: dict[str, Any], configured_image: str) -> str:
    image = configured_image.strip()
    # config.image defaults to a RunPod docker tag; Prime Intellect wants an
    # environment enum, so only pass through values that look like one.
    if image and "/" not in image and ":" not in image:
        return image
    available = [str(i) for i in (offer.get("images") or [])]
    for candidate in _PREFERRED_IMAGES:
        if not available or candidate in available:
            return candidate
    return available[0]


def _clamped_disk_gb(offer: dict[str, Any], requested_gb: int) -> int | None:
    disk = offer.get("disk") or {}
    minimum = disk.get("min")
    maximum = disk.get("max")
    size = int(requested_gb)
    if isinstance(minimum, (int, float)):
        size = max(size, int(minimum))
    if isinstance(maximum, (int, float)):
        size = min(size, int(maximum))
    return size if size > 0 else None


def create_pi_pod(
    *,
    name: str,
    offers: list[dict[str, Any]],
    ssh_key_id: str,
    image: str = "",
    disk_gb: int | None = None,
    max_dollars_per_hour: float | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, str, float]:
    """Try each offer in order; return (pod_id, gpu_label, hourly_dollars)."""
    last_error: Exception | None = None
    for offer in offers:
        pod: dict[str, Any] = {
            "name": name[:190],
            "cloudId": offer["cloudId"],
            "gpuType": offer["gpuType"],
            "socket": offer["socket"],
            "gpuCount": 1,
            "image": _select_image(offer, image),
            "sshKeyId": ssh_key_id,
        }
        if disk_gb is not None:
            clamped = _clamped_disk_gb(offer, disk_gb)
            if clamped is not None:
                pod["diskSize"] = clamped
        if max_dollars_per_hour is not None and max_dollars_per_hour > 0:
            pod["maxPrice"] = float(max_dollars_per_hour)
        if offer.get("dataCenter"):
            pod["dataCenterId"] = offer["dataCenter"]
        if offer.get("country"):
            pod["country"] = offer["country"]
        if offer.get("security"):
            pod["security"] = offer["security"]
        if env:
            pod["envVars"] = [{"key": key, "value": value} for key, value in env.items()]
        body = {"pod": pod, "provider": {"type": offer.get("provider") or "primecompute"}}
        try:
            data = _pi_rest("POST", "/pods/", body=body)
            if not isinstance(data, dict) or not data.get("id"):
                raise RuntimeError(f"unexpected Prime Intellect create response: {data!r}")
            hourly = data.get("priceHr")
            if not isinstance(hourly, (int, float)) or hourly <= 0:
                hourly = offer_hourly(offer) or 0.0
            return str(data["id"]), offer_label(offer), float(hourly)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to create Prime Intellect pod for available offers: {last_error}")


def wait_for_pi_pod(pod_id: str, *, timeout_seconds: int = 1800, poll_seconds: int = 10) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_status = ""
    while time.time() < deadline:
        pod = _pi_rest("GET", f"/pods/{pod_id}")
        if isinstance(pod, dict):
            last_status = str(pod.get("status") or "")
            if last_status in {"ERROR", "TERMINATED", "DELETING"}:
                raise RuntimeError(f"Prime Intellect pod {pod_id} entered status {last_status}")
            if last_status == "ACTIVE" and extract_pi_ssh_target(pod) is not None:
                return pod
        time.sleep(poll_seconds)
    raise RuntimeError(f"Prime Intellect pod {pod_id} not ready after {timeout_seconds}s (status {last_status!r})")


def extract_pi_ssh_target(pod: dict[str, Any]) -> RunpodSshTarget | None:
    connections = pod.get("sshConnection")
    if isinstance(connections, str):
        connections = [connections]
    for connection in connections or []:
        match = re.match(r"^\s*(?:[\w.-]+@)?([\w.-]+)(?:\s+-p\s*(\d+))?\s*$", str(connection))
        if match:
            return RunpodSshTarget(host=match.group(1), port=int(match.group(2) or 22))
    ips = pod.get("ip")
    if isinstance(ips, str):
        ips = [ips]
    for ip in ips or []:
        if str(ip).strip():
            port = 22
            for mapping in pod.get("primePortMapping") or []:
                if isinstance(mapping, dict) and mapping.get("usedBy") == "SSH":
                    try:
                        port = int(mapping.get("external") or 22)
                    except (TypeError, ValueError):
                        port = 22
                    break
            return RunpodSshTarget(host=str(ip).strip(), port=port)
    return None


def delete_pi_pod(pod_id: str) -> None:
    _pi_rest("DELETE", f"/pods/{pod_id}")
