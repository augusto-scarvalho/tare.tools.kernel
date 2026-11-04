import json
import urllib.request
from dataclasses import FrozenInstanceError, dataclass

import pytest

from compute_plane.live_prompt_cache import (
    BedrockMantleHttpTransport,
    BedrockPromptCacheAdapter,
    BedrockResponsesPromptCacheAdapter,
    LivePromptCacheRejected,
    LivePromptCacheRejectReason,
    OpenRouterHttpTransport,
    OpenRouterPromptCacheAdapter,
    TokenRates,
    TransportResponse,
)


RATES = TokenRates(
    input_usd="0.000003",
    cache_write_usd="0.00000375",
    cache_read_usd="0.0000003",
    output_usd="0.000015",
)

BEDROCK_LUNA_RATES = TokenRates(
    input_usd="0.00000044",
    cache_write_usd="0.00000055",
    cache_read_usd="0.000000044",
    output_usd="0.00000198",
)


@dataclass(frozen=True)
class ProjectionFixture:
    source_revision: int = 3
    source_snapshot_sha256: str = "a" * 64
    projected_bytes: bytes = (
        b'[{"content":"stable reference","role":"user"},'
        b'{"content":"reference accepted","role":"assistant"},'
        b'{"content":"reply CACHE_OK","role":"user"}]'
    )
    projection_generation: int = 1


class QueueOpenRouterTransport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def send(self, request_bytes):
        self.requests.append(request_bytes)
        return TransportResponse(
            body=json.dumps(self.payloads.pop(0)).encode(),
            status_code=200,
            elapsed_ms=25,
            request_id="or-request",
        )


class QueueBedrockTransport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def invoke(self, model_id, request_bytes):
        self.calls.append((model_id, request_bytes))
        return TransportResponse(
            body=json.dumps(self.payloads.pop(0)).encode(),
            status_code=200,
            elapsed_ms=30,
            request_id="aws-request",
        )


class QueueMantleTransport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def send(self, request_bytes):
        self.requests.append(request_bytes)
        return TransportResponse(
            body=json.dumps(self.payloads.pop(0)).encode(),
            status_code=200,
            elapsed_ms=35,
            request_id="mantle-request",
        )


def _openrouter_payload(*, cached, written, model="anthropic/claude-sonnet-4.6"):
    return {
        "id": "gen-test",
        "model": model,
        "choices": [{"message": {"content": "CACHE_OK"}}],
        "usage": {
            "prompt_tokens": 1100,
            "completion_tokens": 2,
            "cost": 0.0041175 if written else 0.000495,
            "prompt_tokens_details": {
                "cached_tokens": cached,
                "cache_write_tokens": written,
            },
        },
    }


def _bedrock_payload(*, cached, written, model=None):
    payload = {
        "id": "msg-test",
        "content": [{"type": "text", "text": "CACHE_OK"}],
        "usage": {
            "input_tokens": 50,
            "output_tokens": 2,
            "cache_creation_input_tokens": written,
            "cache_read_input_tokens": cached,
        },
    }
    if model is not None:
        payload["model"] = model
    return payload


def _mantle_payload(*, cached, written, model="openai.gpt-5.6-luna"):
    return {
        "id": "resp-test",
        "model": model,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "CACHE_OK"}],
            }
        ],
        "usage": {
            "input_tokens": 1100,
            "output_tokens": 2,
            "input_tokens_details": {
                "cached_tokens": cached,
                "cache_write_tokens": written,
            },
        },
    }


def _openrouter(transport, **overrides):
    options = {
        "transport": transport,
        "admit_projection": lambda projection: projection,
        "model": "anthropic/claude-sonnet-4.6",
        "system_prompt": "P3a qualification",
        "cached_message_count": 1,
        "token_rates": RATES,
    }
    options.update(overrides)
    return OpenRouterPromptCacheAdapter(**options)


def _bedrock(transport, **overrides):
    options = {
        "transport": transport,
        "admit_projection": lambda projection: projection,
        "model": "us.anthropic.claude-sonnet-4-6",
        "system_prompt": "P3a qualification",
        "cached_message_count": 1,
        "token_rates": RATES,
    }
    options.update(overrides)
    return BedrockPromptCacheAdapter(**options)


def _mantle(transport, **overrides):
    options = {
        "transport": transport,
        "admit_projection": lambda projection: projection,
        "model": "openai.gpt-5.6-luna",
        "system_prompt": "P3a qualification",
        "cached_message_count": 1,
        "token_rates": BEDROCK_LUNA_RATES,
    }
    options.update(overrides)
    return BedrockResponsesPromptCacheAdapter(**options)


def test_openrouter_exact_request_and_cold_warm_telemetry():
    transport = QueueOpenRouterTransport(
        [
            _openrouter_payload(cached=0, written=1050),
            _openrouter_payload(cached=1050, written=0),
        ]
    )
    adapter = _openrouter(transport)

    cold = adapter.execute(ProjectionFixture())
    warm = adapter.execute(ProjectionFixture())
    body = json.loads(transport.requests[0])

    assert body["provider"] == {
        "only": ["anthropic"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert body["messages"][1]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert body["messages"][-1]["content"][0]["text"] == "reply CACHE_OK"
    assert cold.request.request_bytes == warm.request.request_bytes
    assert cold.request.cache_plan == warm.request.cache_plan
    assert cold.telemetry.hit is False
    assert cold.telemetry.cache_write_tokens == 1050
    assert cold.telemetry.estimated_cost_usd == "0.0041175"
    assert cold.telemetry.estimated_savings_usd == "-0.0007875"
    assert warm.telemetry.hit is True
    assert warm.telemetry.cache_read_tokens == 1050
    assert warm.telemetry.estimated_cost_usd == "0.000495"
    assert warm.telemetry.estimated_savings_usd == "0.002835"
    assert warm.response_text == "CACHE_OK"


def test_bedrock_exact_request_model_pin_and_cold_warm_telemetry():
    transport = QueueBedrockTransport(
        [
            _bedrock_payload(cached=0, written=1050),
            _bedrock_payload(cached=1050, written=0),
        ]
    )
    adapter = _bedrock(transport)

    cold = adapter.execute(ProjectionFixture())
    warm = adapter.execute(ProjectionFixture())
    model_id, request_bytes = transport.calls[0]
    body = json.loads(request_bytes)

    assert model_id == "us.anthropic.claude-sonnet-4-6"
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert cold.request.request_bytes == warm.request.request_bytes
    assert cold.telemetry.hit is False
    assert warm.telemetry.hit is True
    assert warm.telemetry.estimated_cost_usd == "0.000495"
    assert warm.response_text == "CACHE_OK"


def test_bedrock_accepts_only_explicitly_pinned_served_model_alias():
    accepted = QueueBedrockTransport(
        [_bedrock_payload(cached=0, written=1050, model="claude-sonnet-4-6")]
    )
    result = _bedrock(
        accepted, served_model_aliases=("claude-sonnet-4-6",)
    ).execute(ProjectionFixture())
    assert result.telemetry.served_model == "claude-sonnet-4-6"

    rejected = QueueBedrockTransport(
        [_bedrock_payload(cached=0, written=1050, model="other-model")]
    )
    with pytest.raises(LivePromptCacheRejected) as mismatch:
        _bedrock(
            rejected, served_model_aliases=("claude-sonnet-4-6",)
        ).execute(ProjectionFixture())
    assert mismatch.value.reason is LivePromptCacheRejectReason.MODEL_MISMATCH


def test_bedrock_mantle_exact_request_and_cold_warm_telemetry():
    transport = QueueMantleTransport(
        [
            _mantle_payload(cached=0, written=1050),
            _mantle_payload(cached=1050, written=0),
        ]
    )
    adapter = _mantle(transport)

    cold = adapter.execute(ProjectionFixture())
    warm = adapter.execute(ProjectionFixture())
    body = json.loads(transport.requests[0])

    assert body["model"] == "openai.gpt-5.6-luna"
    assert body["store"] is False
    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert len(body["prompt_cache_key"]) == 64
    assert body["input"][1]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert body["input"][2]["role"] == "assistant"
    assert body["input"][2]["content"] == "reference accepted"
    assert body["input"][-1]["content"][0]["text"] == "reply CACHE_OK"
    assert cold.request.request_bytes == warm.request.request_bytes
    assert cold.request.cache_plan.ttl_seconds == 1800
    assert cold.telemetry.hit is False
    assert cold.telemetry.uncached_input_tokens == 50
    assert cold.telemetry.estimated_cost_usd == "0.00060346"
    assert warm.telemetry.hit is True
    assert warm.telemetry.uncached_input_tokens == 50
    assert warm.telemetry.estimated_cost_usd == "0.00007216"
    assert warm.response_text == "CACHE_OK"


def test_bedrock_mantle_model_mismatch_fails_closed():
    transport = QueueMantleTransport(
        [_mantle_payload(cached=0, written=1050, model="openai.other")]
    )
    with pytest.raises(LivePromptCacheRejected) as rejected:
        _mantle(transport).execute(ProjectionFixture())
    assert rejected.value.reason is LivePromptCacheRejectReason.MODEL_MISMATCH


def test_bedrock_mantle_rejects_breakpoint_on_assistant_before_transport():
    transport = QueueMantleTransport([])
    adapter = _mantle(transport, cached_message_count=2)

    with pytest.raises(LivePromptCacheRejected) as rejected:
        adapter.execute(ProjectionFixture())

    assert rejected.value.reason is LivePromptCacheRejectReason.INVALID_MESSAGES
    assert transport.requests == []


def test_provider_domains_have_distinct_prefix_digests():
    projection = ProjectionFixture()
    openrouter = _openrouter(QueueOpenRouterTransport([])).build_request(projection)
    bedrock = _bedrock(QueueBedrockTransport([])).build_request(projection)
    mantle = _mantle(QueueMantleTransport([])).build_request(projection)

    assert openrouter.request_bytes.startswith(openrouter.request_prefix_bytes)
    assert bedrock.request_bytes.startswith(bedrock.request_prefix_bytes)
    assert mantle.request_bytes.startswith(mantle.request_prefix_bytes)
    assert len(
        {
            openrouter.cache_plan.prefix_digest,
            bedrock.cache_plan.prefix_digest,
            mantle.cache_plan.prefix_digest,
        }
    ) == 3
    assert {
        openrouter.cache_plan.source_revision,
        bedrock.cache_plan.source_revision,
        mantle.cache_plan.source_revision,
    } == {3}


def test_stale_admission_fails_before_any_live_transport_call():
    transport = QueueOpenRouterTransport([])

    def reject(_projection):
        raise RuntimeError("STALE_PROJECTION")

    adapter = _openrouter(transport, admit_projection=reject)
    with pytest.raises(RuntimeError, match="STALE_PROJECTION"):
        adapter.execute(ProjectionFixture())
    assert transport.requests == []


def test_openrouter_served_model_mismatch_fails_closed():
    transport = QueueOpenRouterTransport(
        [_openrouter_payload(cached=0, written=1050, model="anthropic/other")]
    )
    with pytest.raises(LivePromptCacheRejected) as rejected:
        _openrouter(transport).execute(ProjectionFixture())
    assert rejected.value.reason is LivePromptCacheRejectReason.MODEL_MISMATCH


def test_invalid_projection_message_shape_fails_closed():
    projection = ProjectionFixture(
        projected_bytes=b'[{"content":"secret","extra":true,"role":"user"},{"content":"q","role":"user"}]'
    )
    with pytest.raises(LivePromptCacheRejected) as rejected:
        _openrouter(QueueOpenRouterTransport([])).build_request(projection)
    assert rejected.value.reason is LivePromptCacheRejectReason.INVALID_MESSAGES


def test_live_telemetry_is_frozen_and_non_authoritative():
    transport = QueueOpenRouterTransport(
        [_openrouter_payload(cached=0, written=1050)]
    )
    telemetry = _openrouter(transport).execute(ProjectionFixture()).telemetry

    with pytest.raises(FrozenInstanceError):
        telemetry.hit = True
    assert {"work", "authority", "evidence", "settlement"}.isdisjoint(vars(telemetry))


def test_openrouter_secret_never_enters_request_or_repr():
    secret = "secret-never-serialize"
    transport = OpenRouterHttpTransport(api_key=secret)
    request = _openrouter(transport).build_request(ProjectionFixture())

    assert secret.encode() not in request.request_bytes
    assert secret not in repr(transport)


def test_bedrock_mantle_secret_never_enters_request_or_repr():
    secret = "secret-never-serialize"
    transport = BedrockMantleHttpTransport(api_key=secret)
    request = _mantle(transport).build_request(ProjectionFixture())

    assert secret.encode() not in request.request_bytes
    assert secret not in repr(transport)


def test_bedrock_mantle_uses_gpt_56_responses_path(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        raise OSError("stop before network")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    transport = BedrockMantleHttpTransport(api_key="test-secret")

    with pytest.raises(LivePromptCacheRejected):
        transport.send(b"{}")

    assert captured["url"] == (
        "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
    )


def test_token_rates_reject_non_finite_or_negative_values():
    with pytest.raises(ValueError):
        TokenRates("NaN", "1", "1", "1")
    with pytest.raises(ValueError):
        TokenRates("-1", "1", "1", "1")
