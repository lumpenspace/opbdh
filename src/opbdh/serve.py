from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .hal import HalEye


HF_ENDPOINTS_TOKEN_DOCS = "https://huggingface.co/settings/tokens"


class InferencePermissionError(Exception):
    """Raised when Hugging Face token lacks inference endpoints permissions."""


class InferenceEndpointNotFoundError(Exception):
    """Raised when requested inference endpoint is not found."""


@dataclass(slots=True)
class EndpointSummary:
    name: str
    repository: str
    status: str
    url: str | None = None
    accelerator: str | None = None
    instance_type: str | None = None
    instance_size: str | None = None
    vendor: str | None = None
    region: str | None = None
    scale_to_zero_timeout: int | None = None
    min_replica: int = 1
    max_replica: int = 1
    created_at: str | None = None
    updated_at: str | None = None
    raw: dict[str, Any] | None = None


def normalize_endpoint_name(model_id: str, name: str | None = None) -> str:
    if name and name.strip():
        base = name.strip()
    else:
        # e.g. "lumpenspace/reword-grpo-scaled" -> "reword-grpo-scaled"
        base = model_id.split("/")[-1] if "/" in model_id else model_id
        if not base.endswith("-ep"):
            base = f"{base}-ep"
    # Endpoints names: lowercase, numbers, hyphens, max 32 chars
    clean = re.sub(r"[^a-zA-Z0-9\-]", "-", base).lower().strip("-")
    clean = re.sub(r"-+", "-", clean)
    return clean[:32] or "opbdh-endpoint"


def _check_hf_permission_error(exc: Exception) -> None:
    msg = str(exc)
    if "403" in msg or "inference.endpoints" in msg or "Forbidden" in msg:
        raise InferencePermissionError(
            "Your Hugging Face token lacks permissions to manage Inference Endpoints.\n"
            f"Please visit {HF_ENDPOINTS_TOKEN_DOCS} to grant:\n"
            "  - 'Make calls to Inference Endpoints'\n"
            "  - 'Manage Inference Endpoints'"
        ) from exc


def endpoint_to_summary(endpoint: Any) -> EndpointSummary:
    raw = getattr(endpoint, "raw", None) or {}
    compute = raw.get("compute", {}) if isinstance(raw, dict) else {}
    scaling = compute.get("scaling", {}) if isinstance(compute, dict) else {}
    provider = raw.get("provider", {}) if isinstance(raw, dict) else {}
    status_dict = raw.get("status", {}) if isinstance(raw, dict) else {}

    name = getattr(endpoint, "name", "") or raw.get("name", "")
    repository = getattr(endpoint, "repository", "") or raw.get("model", {}).get("repository", "")
    status = getattr(endpoint, "status", "") or status_dict.get("state", "unknown")
    url = getattr(endpoint, "url", None) or status_dict.get("url")

    accelerator = compute.get("accelerator")
    instance_type = compute.get("instanceType")
    instance_size = compute.get("instanceSize")
    vendor = provider.get("vendor")
    region = provider.get("region")
    scale_to_zero_timeout = scaling.get("scaleToZeroTimeout")
    min_replica = scaling.get("minReplica", 1)
    max_replica = scaling.get("maxReplica", 1)

    created_at = str(getattr(endpoint, "created_at", None) or status_dict.get("createdAt", ""))
    updated_at = str(getattr(endpoint, "updated_at", None) or status_dict.get("updatedAt", ""))

    return EndpointSummary(
        name=name,
        repository=repository,
        status=status,
        url=url,
        accelerator=accelerator,
        instance_type=instance_type,
        instance_size=instance_size,
        vendor=vendor,
        region=region,
        scale_to_zero_timeout=scale_to_zero_timeout,
        min_replica=min_replica,
        max_replica=max_replica,
        created_at=created_at,
        updated_at=updated_at,
        raw=raw,
    )


def list_endpoints(
    *,
    namespace: str | None = None,
    token: str | None = None,
) -> list[EndpointSummary]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        endpoints = api.list_inference_endpoints(namespace=namespace)
    except Exception as exc:
        _check_hf_permission_error(exc)
        raise

    return [endpoint_to_summary(ep) for ep in endpoints]


def get_endpoint(
    name: str,
    *,
    namespace: str | None = None,
    token: str | None = None,
) -> EndpointSummary:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        ep = api.get_inference_endpoint(name=name, namespace=namespace)
    except Exception as exc:
        _check_hf_permission_error(exc)
        if "404" in str(exc) or "not found" in str(exc).lower():
            raise InferenceEndpointNotFoundError(f"Endpoint '{name}' was not found.") from exc
        raise

    return endpoint_to_summary(ep)


def create_endpoint(
    model_id: str,
    *,
    name: str | None = None,
    accelerator: str = "gpu",
    instance_type: str = "nvidia-a10g",
    instance_size: str = "x1",
    vendor: str = "aws",
    region: str = "us-east-1",
    scale_to_zero_timeout: int = 15,
    min_replica: int | None = None,
    max_replica: int = 1,
    framework: str = "custom",
    task: str = "text-generation",
    wait: bool = False,
    timeout: int = 600,
    namespace: str | None = None,
    token: str | None = None,
) -> EndpointSummary:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    ep_name = normalize_endpoint_name(model_id, name)

    effective_min_replica = min_replica if min_replica is not None else (0 if scale_to_zero_timeout > 0 else 1)

    try:
        endpoint = api.create_inference_endpoint(
            name=ep_name,
            repository=model_id,
            framework=framework,
            task=task,
            accelerator=accelerator,
            instance_type=instance_type,
            instance_size=instance_size,
            vendor=vendor,
            region=region,
            min_replica=effective_min_replica,
            max_replica=max_replica,
            scale_to_zero_timeout=scale_to_zero_timeout if scale_to_zero_timeout > 0 else None,
            type="authenticated",
            namespace=namespace,
        )
    except Exception as exc:
        _check_hf_permission_error(exc)
        raise

    if wait:
        with HalEye(f"Waiting for endpoint '{ep_name}' to become ready..."):
            endpoint.wait(timeout=timeout)

    return endpoint_to_summary(endpoint)


def pause_endpoint(
    name: str,
    *,
    namespace: str | None = None,
    token: str | None = None,
) -> EndpointSummary:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        ep = api.get_inference_endpoint(name=name, namespace=namespace)
        ep.pause()
    except Exception as exc:
        _check_hf_permission_error(exc)
        raise

    return endpoint_to_summary(ep)


def resume_endpoint(
    name: str,
    *,
    namespace: str | None = None,
    token: str | None = None,
) -> EndpointSummary:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        ep = api.get_inference_endpoint(name=name, namespace=namespace)
        ep.resume()
    except Exception as exc:
        _check_hf_permission_error(exc)
        raise

    return endpoint_to_summary(ep)


def delete_endpoint(
    name: str,
    *,
    namespace: str | None = None,
    token: str | None = None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.delete_inference_endpoint(name=name, namespace=namespace)
    except Exception as exc:
        _check_hf_permission_error(exc)
        raise


def test_inference(
    target: str,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    token: str | None = None,
    namespace: str | None = None,
) -> str:
    from huggingface_hub import InferenceClient

    client_target = target

    if not (target.startswith("http://") or target.startswith("https://")):
        try:
            endpoints = list_endpoints(namespace=namespace, token=token)
            for ep in endpoints:
                if ep.name == target and ep.url:
                    client_target = ep.url
                    break
        except Exception:
            pass

    client = InferenceClient(model=client_target, token=token)
    try:
        output = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        _check_hf_permission_error(exc)
        raise

    return str(output)
