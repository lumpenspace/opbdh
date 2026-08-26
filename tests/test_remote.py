from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse

import pytest

import opbdh
from opbdh import api
from opbdh import remote


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_runpod_balance_parses_account_values_and_request(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(request, *, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return _Response(
            {"data": {"myself": {"clientBalance": 39.70, "currentSpendPerHr": "1.571"}}}
        )

    monkeypatch.setattr(remote.urllib.request, "urlopen", fake_urlopen)

    result = remote.runpod_balance("secret/+token")

    assert result == remote.RunpodBalance(client_balance=39.70, current_spend_per_hour=1.571)
    request = seen["request"]
    assert request.get_method() == "POST"
    assert seen["timeout"] == 5
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query) == {
        "api_key": ["secret/+token"]
    }
    assert json.loads(request.data) == {
        "query": "query { myself { clientBalance currentSpendPerHr } }"
    }


@pytest.mark.parametrize(
    ("myself", "expected"),
    [
        ({"clientBalance": None, "currentSpendPerHr": None}, remote.RunpodBalance(None, None)),
        ({"clientBalance": 12.5}, remote.RunpodBalance(12.5, None)),
        ({"currentSpendPerHr": 0}, remote.RunpodBalance(None, 0.0)),
    ],
)
def test_runpod_balance_preserves_null_missing_and_zero_values(monkeypatch, myself, expected) -> None:
    monkeypatch.setattr(
        remote.urllib.request,
        "urlopen",
        lambda request, *, timeout: _Response({"data": {"myself": myself}}),
    )

    assert remote.runpod_balance("token") == expected


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": {}},
        {"data": {"myself": None}},
        {"data": {"myself": []}},
    ],
)
def test_runpod_balance_returns_none_for_invalid_myself(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        remote.urllib.request,
        "urlopen",
        lambda request, *, timeout: _Response(payload),
    )

    assert remote.runpod_balance("token") is None


def test_runpod_balance_fails_soft_without_a_token(monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    def unexpected_urlopen(*args, **kwargs):  # pragma: no cover - assertion is the point
        raise AssertionError("missing credentials must not make an HTTP request")

    monkeypatch.setattr(remote.urllib.request, "urlopen", unexpected_urlopen)

    assert remote.runpod_balance() is None


def test_runpod_balance_fails_soft_without_exposing_the_token(monkeypatch, capsys) -> None:
    secret = "never-print-this-token"
    error = urllib.error.HTTPError(
        f"https://api.runpod.io/graphql?api_key={secret}",
        500,
        "server error",
        {},
        io.BytesIO(b"failed"),
    )
    monkeypatch.setattr(remote.urllib.request, "urlopen", lambda request, *, timeout: (_ for _ in ()).throw(error))

    assert remote.runpod_balance(secret) is None
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (402, b'{"error":"request refused"}'),
        (400, b'{"error":"not enough funds to create this pod"}'),
    ],
)
def test_runpod_rest_classifies_pod_credit_failures(monkeypatch, status, detail) -> None:
    error = urllib.error.HTTPError(
        "https://rest.runpod.io/v1/pods",
        status,
        "request failed",
        {},
        io.BytesIO(detail),
    )
    monkeypatch.setattr(remote.urllib.request, "urlopen", lambda request, *, timeout: (_ for _ in ()).throw(error))

    with pytest.raises(remote.InsufficientCreditsError):
        remote._runpod_rest("POST", "/pods", api_token="token", body={})


def test_runpod_rest_does_not_classify_other_failures_as_credit_errors(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "https://rest.runpod.io/v1/networkvolumes",
        402,
        "payment required",
        {},
        io.BytesIO(b"payment required"),
    )
    monkeypatch.setattr(remote.urllib.request, "urlopen", lambda request, *, timeout: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError) as exc_info:
        remote._runpod_rest("POST", "/networkvolumes", api_token="token", body={})
    assert not isinstance(exc_info.value, remote.InsufficientCreditsError)


def test_runpod_rest_leaves_unrelated_pod_errors_generic(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "https://rest.runpod.io/v1/pods",
        500,
        "server error",
        {},
        io.BytesIO(b"temporarily unavailable"),
    )
    monkeypatch.setattr(remote.urllib.request, "urlopen", lambda request, *, timeout: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError) as exc_info:
        remote._runpod_rest("POST", "/pods", api_token="token", body={})
    assert not isinstance(exc_info.value, remote.InsufficientCreditsError)


def test_create_runpod_pod_does_not_retry_or_wrap_credit_failure(monkeypatch) -> None:
    calls = 0

    def insufficient(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise remote.InsufficientCreditsError("not enough balance")

    monkeypatch.setattr(remote, "_runpod_rest", insufficient)

    with pytest.raises(remote.InsufficientCreditsError, match="not enough balance"):
        remote.create_runpod_pod(
            name="test",
            cloud_type="ALL",
            public_key="key",
            gpu_types=["GPU one", "GPU two"],
        )
    assert calls == 1


def test_balance_api_is_exported_from_the_package() -> None:
    assert api.runpod_balance is remote.runpod_balance
    assert api.RunpodBalance is remote.RunpodBalance
    assert api.InsufficientCreditsError is remote.InsufficientCreditsError
    assert opbdh.runpod_balance is remote.runpod_balance
    assert opbdh.RunpodBalance is remote.RunpodBalance
    assert opbdh.InsufficientCreditsError is remote.InsufficientCreditsError
