"""Narrow live-provider prompt-cache adapters for the P3a qualification slice.

The adapters serialize exact provider request bytes and expose detached cache
telemetry. They do not own credentials, conversation truth, Work or settlement.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Protocol

from .prompt_cache import (
    PROMPT_CACHE_PLAN_VERSION,
    ProjectionAdmission,
    ProjectionTicket,
    PromptCacheLimits,
    PromptCachePlan,
    PromptCacheRejected,
    PromptCacheRejectReason,
    ProviderRequest,
    compute_provider_prefix_digest,
    detach_projection,
)


OPENROUTER_WIRE_VERSION = "openrouter-chat-completions-v1"
BEDROCK_ANTHROPIC_WIRE_VERSION = "bedrock-anthropic-messages-2023-05-31"
BEDROCK_OPENAI_RESPONSES_WIRE_VERSION = "bedrock-openai-responses-v1"
DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_BEDROCK_MANTLE_ENDPOINT = (
    "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
)


class LivePromptCacheRejectReason(str, Enum):
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    INVALID_MESSAGES = "INVALID_MESSAGES"
    TRANSPORT_FAILED = "TRANSPORT_FAILED"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    MODEL_MISMATCH = "MODEL_MISMATCH"


class LivePromptCacheRejected(RuntimeError):
    """Sanitized live-provider failure that never embeds request or secret bytes."""

    def __init__(self, reason: LivePromptCacheRejectReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class TokenRates:
    """USD-per-token rates pinned by the qualification receipt."""

    input_usd: str
    cache_write_usd: str
    cache_read_usd: str
    output_usd: str

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            try:
                parsed = Decimal(value)
            except (InvalidOperation, TypeError) as exc:
                raise ValueError(f"{name} must be a non-negative decimal string") from exc
            if not parsed.is_finite() or parsed < 0:
                raise ValueError(f"{name} must be a non-negative decimal string")

    def estimate(
        self,
        *,
        uncached_input_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
        output_tokens: int,
    ) -> str:
        cost = (
            Decimal(self.input_usd) * uncached_input_tokens
            + Decimal(self.cache_write_usd) * cache_write_tokens
            + Decimal(self.cache_read_usd) * cache_read_tokens
            + Decimal(self.output_usd) * output_tokens
        )
        return _decimal_text(cost)

    def savings(
        self,
        *,
        cache_write_tokens: int,
        cache_read_tokens: int,
    ) -> str:
        savings = (
            (Decimal(self.input_usd) - Decimal(self.cache_read_usd))
            * cache_read_tokens
            - (Decimal(self.cache_write_usd) - Decimal(self.input_usd))
            * cache_write_tokens
        )
        return _decimal_text(savings)


@dataclass(frozen=True)
class TransportResponse:
    body: bytes
    status_code: int
    elapsed_ms: int
    request_id: str | None = None


class OpenRouterTransport(Protocol):
    def send(self, request_bytes: bytes) -> TransportResponse: ...


class BedrockTransport(Protocol):
    def invoke(self, model_id: str, request_bytes: bytes) -> TransportResponse: ...


class BedrockMantleTransport(Protocol):
    def send(self, request_bytes: bytes) -> TransportResponse: ...


class OpenRouterHttpTransport:
    """Minimal real transport; the API key never enters request bodies or receipts."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = DEFAULT_OPENROUTER_ENDPOINT,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be non-empty")
        if not endpoint.startswith("https://"):
            raise ValueError("OpenRouter endpoint must use HTTPS")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def send(self, request_bytes: bytes) -> TransportResponse:
        request = urllib.request.Request(
            self._endpoint,
            data=request_bytes,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "tare.tools.kernel-prompt-cache-p3a/1",
            },
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read()
                status = response.status
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            request_id = exc.headers.get("x-request-id") if exc.headers else None
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.TRANSPORT_FAILED,
                f"OpenRouter HTTP {exc.code}; request_id={request_id or 'unavailable'}",
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.TRANSPORT_FAILED,
                f"OpenRouter transport failed: {type(exc).__name__}",
            ) from exc
        return TransportResponse(
            body=body,
            status_code=status,
            elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
            request_id=request_id,
        )


class BedrockInvokeModelTransport:
    """Thin wrapper around an injected boto3 bedrock-runtime client."""

    def __init__(self, client: Any) -> None:
        if not callable(getattr(client, "invoke_model", None)):
            raise ValueError("client must provide invoke_model")
        self._client = client

    def invoke(self, model_id: str, request_bytes: bytes) -> TransportResponse:
        started = time.monotonic()
        try:
            response = self._client.invoke_model(
                modelId=model_id,
                body=request_bytes,
                contentType="application/json",
                accept="application/json",
            )
            body = response["body"].read()
            metadata = response.get("ResponseMetadata") or {}
            status = int(metadata.get("HTTPStatusCode", 200))
            request_id = metadata.get("RequestId")
        except Exception as exc:
            response = getattr(exc, "response", {}) or {}
            error = response.get("Error", {}) if isinstance(response, dict) else {}
            metadata = response.get("ResponseMetadata", {}) if isinstance(response, dict) else {}
            code = error.get("Code") or type(exc).__name__
            request_id = metadata.get("RequestId")
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.TRANSPORT_FAILED,
                f"Bedrock transport failed: {code}; request_id={request_id or 'unavailable'}",
            ) from exc
        return TransportResponse(
            body=body,
            status_code=status,
            elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
            request_id=request_id,
        )


class BedrockMantleHttpTransport:
    """Narrow bearer transport for Bedrock's OpenAI-compatible Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = DEFAULT_BEDROCK_MANTLE_ENDPOINT,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be non-empty")
        if not endpoint.startswith("https://"):
            raise ValueError("Bedrock Mantle endpoint must use HTTPS")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def send(self, request_bytes: bytes) -> TransportResponse:
        request = urllib.request.Request(
            self._endpoint,
            data=request_bytes,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "tare.tools.kernel-prompt-cache-p3a/1",
            },
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read()
                status = response.status
                request_id = response.headers.get("x-amzn-requestid")
        except urllib.error.HTTPError as exc:
            request_id = exc.headers.get("x-amzn-requestid") if exc.headers else None
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.TRANSPORT_FAILED,
                f"Bedrock Mantle HTTP {exc.code}; request_id={request_id or 'unavailable'}",
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.TRANSPORT_FAILED,
                f"Bedrock Mantle transport failed: {type(exc).__name__}",
            ) from exc
        return TransportResponse(
            body=body,
            status_code=status,
            elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
            request_id=request_id,
        )


@dataclass(frozen=True)
class LivePromptCacheTelemetry:
    provider: str
    requested_model: str
    served_model: str
    source_revision: int
    prefix_digest: str
    request_sha256: str
    hit: bool
    uncached_input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    elapsed_ms: int
    reported_cost_usd: str | None
    estimated_cost_usd: str
    estimated_savings_usd: str
    request_id: str | None


@dataclass(frozen=True)
class LiveProviderResult:
    request: ProviderRequest
    response_text: str
    response_sha256: str
    telemetry: LivePromptCacheTelemetry


class _LiveCacheAdapterBase:
    def __init__(
        self,
        *,
        admit_projection: ProjectionAdmission,
        model: str,
        system_prompt: str | bytes,
        cached_message_count: int,
        token_rates: TokenRates,
        tool_schemas: Sequence[bytes] = (),
        max_tokens: int = 16,
        ttl_seconds: int = 300,
        limits: PromptCacheLimits | None = None,
    ) -> None:
        if not callable(admit_projection):
            raise ValueError("admit_projection must be callable")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be non-empty")
        if not isinstance(cached_message_count, int) or cached_message_count < 1:
            raise ValueError("cached_message_count must be positive")
        if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if ttl_seconds not in (300, 1800, 3600):
            raise ValueError("ttl_seconds must be 300, 1800 or 3600")
        self._admit_projection = admit_projection
        self._model = model
        self._system_prompt = _utf8_bytes(system_prompt)
        self._cached_message_count = cached_message_count
        self._token_rates = token_rates
        self._tool_schemas = tuple(_json_object_bytes(value) for value in tool_schemas)
        self._max_tokens = max_tokens
        self._ttl_seconds = ttl_seconds
        self._limits = limits or PromptCacheLimits()
        if len(self._system_prompt) > self._limits.max_system_prompt_bytes:
            raise ValueError("system_prompt exceeds max_system_prompt_bytes")
        if len(self._tool_schemas) > self._limits.max_tool_schemas:
            raise ValueError("tool_schemas exceeds max_tool_schemas")

    def _admit(self, projection: ProjectionTicket) -> tuple[Any, tuple[tuple[str, str], ...], bytes]:
        admitted = self._admit_projection(projection)
        detached = detach_projection(admitted, self._limits)
        raw_messages = json.loads(detached.projected_bytes)
        messages: list[tuple[str, str]] = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise LivePromptCacheRejected(
                    LivePromptCacheRejectReason.INVALID_MESSAGES,
                    f"projection message {index} must contain only role and content",
                )
            role = message["role"]
            content = message["content"]
            if role not in ("user", "assistant") or not isinstance(content, str):
                raise LivePromptCacheRejected(
                    LivePromptCacheRejectReason.INVALID_MESSAGES,
                    f"projection message {index} has unsupported role or content",
                )
            content.encode("utf-8", errors="strict")
            messages.append((role, content))
        if self._cached_message_count >= len(messages):
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.INVALID_MESSAGES,
                "at least one projected message must remain after the cache point",
            )
        prefix_value = [
            {"content": content, "role": role}
            for role, content in messages[: self._cached_message_count]
        ]
        projected_prefix_bytes = _canonical_json_bytes(prefix_value)
        return detached, tuple(messages), projected_prefix_bytes

    def _plan(
        self,
        *,
        provider: str,
        wire_version: str,
        detached: Any,
        projected_prefix_bytes: bytes,
        request_prefix_bytes: bytes,
    ) -> PromptCachePlan:
        cache_point_end = len(request_prefix_bytes)
        digest = compute_provider_prefix_digest(
            provider=provider,
            model=self._model,
            wire_version=wire_version,
            system_prompt_bytes=self._system_prompt,
            tool_schema_bytes=self._tool_schemas,
            projected_message_prefix_bytes=projected_prefix_bytes,
            projection_generation=detached.projection_generation,
            cache_plan_version=PROMPT_CACHE_PLAN_VERSION,
            cache_point_end=cache_point_end,
            ttl_seconds=self._ttl_seconds,
            request_prefix_bytes=request_prefix_bytes,
        )
        return PromptCachePlan(
            provider=provider,
            model=self._model,
            wire_version=wire_version,
            cache_plan_version=PROMPT_CACHE_PLAN_VERSION,
            source_revision=detached.source_revision,
            source_snapshot_sha256=detached.source_snapshot_sha256,
            projection_generation=detached.projection_generation,
            prefix_digest=digest,
            cache_point_end=cache_point_end,
            ttl_seconds=self._ttl_seconds,
        )

    def _request(self, request_bytes: bytes, request_prefix_bytes: bytes, plan: PromptCachePlan) -> ProviderRequest:
        if len(request_bytes) > self._limits.max_request_bytes:
            raise PromptCacheRejected(
                PromptCacheRejectReason.REQUEST_TOO_LARGE,
                "serialized provider request exceeds max_request_bytes",
            )
        return ProviderRequest(
            request_bytes=request_bytes,
            request_prefix_bytes=request_prefix_bytes,
            cache_plan=plan,
        )

    def _telemetry(
        self,
        *,
        provider: str,
        served_model: str,
        request: ProviderRequest,
        transport: TransportResponse,
        uncached_input_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
        output_tokens: int,
        reported_cost_usd: str | None,
    ) -> LivePromptCacheTelemetry:
        counts = (
            uncached_input_tokens,
            cache_write_tokens,
            cache_read_tokens,
            output_tokens,
        )
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.INVALID_RESPONSE,
                "provider returned invalid token telemetry",
            )
        return LivePromptCacheTelemetry(
            provider=provider,
            requested_model=self._model,
            served_model=served_model,
            source_revision=request.cache_plan.source_revision,
            prefix_digest=request.cache_plan.prefix_digest,
            request_sha256=hashlib.sha256(request.request_bytes).hexdigest(),
            hit=cache_read_tokens > 0,
            uncached_input_tokens=uncached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            output_tokens=output_tokens,
            elapsed_ms=transport.elapsed_ms,
            reported_cost_usd=reported_cost_usd,
            estimated_cost_usd=self._token_rates.estimate(
                uncached_input_tokens=uncached_input_tokens,
                cache_write_tokens=cache_write_tokens,
                cache_read_tokens=cache_read_tokens,
                output_tokens=output_tokens,
            ),
            estimated_savings_usd=self._token_rates.savings(
                cache_write_tokens=cache_write_tokens,
                cache_read_tokens=cache_read_tokens,
            ),
            request_id=transport.request_id,
        )


class _AnthropicCacheAdapterBase(_LiveCacheAdapterBase):
    def __init__(self, **kwargs: Any) -> None:
        ttl_seconds = kwargs.get("ttl_seconds", 300)
        if ttl_seconds not in (300, 3600):
            raise ValueError("Anthropic ttl_seconds must be 300 or 3600")
        super().__init__(**kwargs)

    def _message_bytes(self, role: str, content: str, *, cache_point: bool) -> bytes:
        block = b'{"type":"text","text":' + _json_string_bytes(content)
        if cache_point:
            block += b',"cache_control":' + self._cache_control_bytes()
        block += b"}"
        return b'{"role":' + _json_string_bytes(role) + b',"content":[' + block + b"]}"

    def _cache_control_bytes(self) -> bytes:
        if self._ttl_seconds == 3600:
            return b'{"type":"ephemeral","ttl":"1h"}'
        return b'{"type":"ephemeral"}'

    def _messages_prefix_and_suffix(
        self,
        messages: tuple[tuple[str, str], ...],
    ) -> tuple[bytes, bytes]:
        serialized = [
            self._message_bytes(
                role,
                content,
                cache_point=index + 1 == self._cached_message_count,
            )
            for index, (role, content) in enumerate(messages)
        ]
        prefix = b",".join(serialized[: self._cached_message_count])
        suffix = b"," + b",".join(serialized[self._cached_message_count :])
        return prefix, suffix


class OpenRouterPromptCacheAdapter(_AnthropicCacheAdapterBase):
    def __init__(
        self,
        *,
        transport: OpenRouterTransport,
        provider_slug: str = "anthropic",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not callable(getattr(transport, "send", None)):
            raise ValueError("transport must provide send")
        if not isinstance(provider_slug, str) or not provider_slug:
            raise ValueError("provider_slug must be non-empty")
        self._transport = transport
        self._provider_slug = provider_slug

    def build_request(self, projection: ProjectionTicket) -> ProviderRequest:
        detached, messages, projected_prefix = self._admit(projection)
        messages_prefix, messages_suffix = self._messages_prefix_and_suffix(messages)
        start = (
            b'{"model":'
            + _json_string_bytes(self._model)
            + b',"provider":{"only":['
            + _json_string_bytes(self._provider_slug)
            + b'],"allow_fallbacks":false,"require_parameters":true}'
            + b',"max_tokens":'
            + str(self._max_tokens).encode("ascii")
            + b',"temperature":0,"stream":false'
        )
        if self._tool_schemas:
            start += b',"tools":[' + b",".join(self._tool_schemas) + b"]"
        system_message = self._message_bytes(
            "system", self._system_prompt.decode("utf-8"), cache_point=False
        )
        request_prefix = start + b',"messages":[' + system_message + b"," + messages_prefix
        request_bytes = request_prefix + messages_suffix + b"]}"
        plan = self._plan(
            provider=f"openrouter/{self._provider_slug}",
            wire_version=OPENROUTER_WIRE_VERSION,
            detached=detached,
            projected_prefix_bytes=projected_prefix,
            request_prefix_bytes=request_prefix,
        )
        return self._request(request_bytes, request_prefix, plan)

    def execute(self, projection: ProjectionTicket) -> LiveProviderResult:
        request = self.build_request(projection)
        transport = self._transport.send(request.request_bytes)
        payload = _json_response(transport.body, "OpenRouter")
        served_model = _required_string(payload, "model", "OpenRouter")
        if served_model != self._model:
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.MODEL_MISMATCH,
                f"OpenRouter served {served_model!r} instead of the pinned model",
            )
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.INVALID_RESPONSE,
                "OpenRouter response is missing usage",
            )
        details = usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            details = {}
        prompt_tokens = _non_negative_int(usage.get("prompt_tokens", 0), "prompt_tokens")
        cache_read = _non_negative_int(details.get("cached_tokens", 0), "cached_tokens")
        cache_write = _non_negative_int(
            details.get("cache_write_tokens", 0), "cache_write_tokens"
        )
        uncached = max(0, prompt_tokens - cache_read - cache_write)
        output_tokens = _non_negative_int(
            usage.get("completion_tokens", 0), "completion_tokens"
        )
        reported_cost = usage.get("cost")
        reported_cost_text = str(reported_cost) if reported_cost is not None else None
        response_text = _openrouter_text(payload)
        telemetry = self._telemetry(
            provider="openrouter",
            served_model=served_model,
            request=request,
            transport=transport,
            uncached_input_tokens=uncached,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            output_tokens=output_tokens,
            reported_cost_usd=reported_cost_text,
        )
        return LiveProviderResult(
            request=request,
            response_text=response_text,
            response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            telemetry=telemetry,
        )


class BedrockPromptCacheAdapter(_AnthropicCacheAdapterBase):
    def __init__(
        self,
        *,
        transport: BedrockTransport,
        served_model_aliases: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not callable(getattr(transport, "invoke", None)):
            raise ValueError("transport must provide invoke")
        if not isinstance(served_model_aliases, Sequence) or isinstance(
            served_model_aliases, (str, bytes, bytearray)
        ):
            raise ValueError("served_model_aliases must be an ordered sequence")
        if any(not isinstance(alias, str) or not alias for alias in served_model_aliases):
            raise ValueError("served_model_aliases must contain non-empty strings")
        self._transport = transport
        self._served_model_aliases = tuple(served_model_aliases)

    def build_request(self, projection: ProjectionTicket) -> ProviderRequest:
        detached, messages, projected_prefix = self._admit(projection)
        messages_prefix, messages_suffix = self._messages_prefix_and_suffix(messages)
        start = (
            b'{"anthropic_version":"bedrock-2023-05-31","max_tokens":'
            + str(self._max_tokens).encode("ascii")
            + b',"temperature":0,"system":[{"type":"text","text":'
            + _json_string_bytes(self._system_prompt.decode("utf-8"))
            + b"}]"
        )
        if self._tool_schemas:
            start += b',"tools":[' + b",".join(self._tool_schemas) + b"]"
        request_prefix = start + b',"messages":[' + messages_prefix
        request_bytes = request_prefix + messages_suffix + b"]}"
        plan = self._plan(
            provider="amazon-bedrock",
            wire_version=BEDROCK_ANTHROPIC_WIRE_VERSION,
            detached=detached,
            projected_prefix_bytes=projected_prefix,
            request_prefix_bytes=request_prefix,
        )
        return self._request(request_bytes, request_prefix, plan)

    def execute(self, projection: ProjectionTicket) -> LiveProviderResult:
        request = self.build_request(projection)
        transport = self._transport.invoke(self._model, request.request_bytes)
        payload = _json_response(transport.body, "Bedrock")
        response_model = payload.get("model")
        served_model = response_model if isinstance(response_model, str) else self._model
        if served_model not in (self._model, *self._served_model_aliases):
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.MODEL_MISMATCH,
                f"Bedrock served {served_model!r} instead of the pinned model",
            )
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.INVALID_RESPONSE,
                "Bedrock response is missing usage",
            )
        uncached = _first_non_negative_int(
            usage, "input_tokens", "inputTokens", default=0
        )
        cache_write = _first_non_negative_int(
            usage,
            "cache_creation_input_tokens",
            "cache_write_input_tokens",
            "cacheWriteInputTokens",
            default=0,
        )
        cache_read = _first_non_negative_int(
            usage,
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cached_tokens",
            default=0,
        )
        output_tokens = _first_non_negative_int(
            usage, "output_tokens", "outputTokens", default=0
        )
        response_text = _bedrock_text(payload)
        telemetry = self._telemetry(
            provider="amazon-bedrock",
            served_model=served_model,
            request=request,
            transport=transport,
            uncached_input_tokens=uncached,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            output_tokens=output_tokens,
            reported_cost_usd=None,
        )
        return LiveProviderResult(
            request=request,
            response_text=response_text,
            response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            telemetry=telemetry,
        )


class BedrockResponsesPromptCacheAdapter(_LiveCacheAdapterBase):
    """Explicit GPT-5.6 cache breakpoints on Bedrock Mantle Responses."""

    def __init__(
        self,
        *,
        transport: BedrockMantleTransport,
        ttl_seconds: int = 1800,
        **kwargs: Any,
    ) -> None:
        if ttl_seconds != 1800:
            raise ValueError("Bedrock GPT-5.6 ttl_seconds must be 1800")
        super().__init__(ttl_seconds=ttl_seconds, **kwargs)
        if not callable(getattr(transport, "send", None)):
            raise ValueError("transport must provide send")
        self._transport = transport

    def _message_bytes(self, role: str, content: str, *, cache_point: bool) -> bytes:
        if role == "assistant":
            if cache_point:
                raise LivePromptCacheRejected(
                    LivePromptCacheRejectReason.INVALID_MESSAGES,
                    "Bedrock GPT-5.6 cannot place a cache breakpoint on an assistant message",
                )
            return (
                b'{"type":"message","role":"assistant","content":'
                + _json_string_bytes(content)
                + b"}"
            )
        block = b'{"type":"input_text","text":' + _json_string_bytes(content)
        if cache_point:
            block += b',"prompt_cache_breakpoint":{"mode":"explicit"}'
        block += b"}"
        return b'{"type":"message","role":' + _json_string_bytes(role) + b',"content":[' + block + b"]}"

    def _messages_prefix_and_suffix(
        self,
        messages: tuple[tuple[str, str], ...],
    ) -> tuple[bytes, bytes]:
        serialized = [
            self._message_bytes(
                role,
                content,
                cache_point=index + 1 == self._cached_message_count,
            )
            for index, (role, content) in enumerate(messages)
        ]
        prefix = b",".join(serialized[: self._cached_message_count])
        suffix = b"," + b",".join(serialized[self._cached_message_count :])
        return prefix, suffix

    def _cache_key(self, detached: Any, projected_prefix: bytes) -> str:
        material = _canonical_json_bytes(
            {
                "domain": "tare.tools.kernel/bedrock-gpt-5.6-cache-key/v1",
                "model": self._model,
                "projection_generation": detached.projection_generation,
                "projected_prefix_sha256": hashlib.sha256(projected_prefix).hexdigest(),
                "system_prompt_sha256": hashlib.sha256(self._system_prompt).hexdigest(),
                "tool_schema_sha256s": [
                    hashlib.sha256(schema).hexdigest() for schema in self._tool_schemas
                ],
            }
        )
        return hashlib.sha256(material).hexdigest()

    def build_request(self, projection: ProjectionTicket) -> ProviderRequest:
        detached, messages, projected_prefix = self._admit(projection)
        messages_prefix, messages_suffix = self._messages_prefix_and_suffix(messages)
        start = (
            b'{"model":'
            + _json_string_bytes(self._model)
            + b',"store":false,"max_output_tokens":'
            + str(self._max_tokens).encode("ascii")
            + b',"prompt_cache_key":'
            + _json_string_bytes(self._cache_key(detached, projected_prefix))
            + b',"prompt_cache_options":{"mode":"explicit","ttl":"30m"}'
        )
        if self._tool_schemas:
            start += b',"tools":[' + b",".join(self._tool_schemas) + b"]"
        developer = self._message_bytes(
            "developer", self._system_prompt.decode("utf-8"), cache_point=False
        )
        request_prefix = start + b',"input":[' + developer + b"," + messages_prefix
        request_bytes = request_prefix + messages_suffix + b"]}"
        plan = self._plan(
            provider="amazon-bedrock/openai",
            wire_version=BEDROCK_OPENAI_RESPONSES_WIRE_VERSION,
            detached=detached,
            projected_prefix_bytes=projected_prefix,
            request_prefix_bytes=request_prefix,
        )
        return self._request(request_bytes, request_prefix, plan)

    def execute(self, projection: ProjectionTicket) -> LiveProviderResult:
        request = self.build_request(projection)
        transport = self._transport.send(request.request_bytes)
        payload = _json_response(transport.body, "Bedrock Mantle")
        served_model = _required_string(payload, "model", "Bedrock Mantle")
        if served_model != self._model:
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.MODEL_MISMATCH,
                f"Bedrock Mantle served {served_model!r} instead of the pinned model",
            )
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise LivePromptCacheRejected(
                LivePromptCacheRejectReason.INVALID_RESPONSE,
                "Bedrock Mantle response is missing usage",
            )
        details = usage.get("input_tokens_details") or {}
        if not isinstance(details, dict):
            details = {}
        input_tokens = _non_negative_int(usage.get("input_tokens", 0), "input_tokens")
        cache_read = _non_negative_int(details.get("cached_tokens", 0), "cached_tokens")
        cache_write = _non_negative_int(
            details.get("cache_write_tokens", 0), "cache_write_tokens"
        )
        uncached = max(0, input_tokens - cache_read - cache_write)
        output_tokens = _non_negative_int(
            usage.get("output_tokens", 0), "output_tokens"
        )
        response_text = _responses_text(payload)
        telemetry = self._telemetry(
            provider="amazon-bedrock",
            served_model=served_model,
            request=request,
            transport=transport,
            uncached_input_tokens=uncached,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            output_tokens=output_tokens,
            reported_cost_usd=None,
        )
        return LiveProviderResult(
            request=request,
            response_text=response_text,
            response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            telemetry=telemetry,
        )


def _utf8_bytes(value: str | bytes) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8", errors="strict")
    if isinstance(value, bytes):
        value.decode("utf-8", errors="strict")
        return bytes(value)
    raise ValueError("system_prompt must be str or bytes")


def _json_object_bytes(value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise ValueError("tool schemas must be immutable bytes")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool schema must encode a JSON object")
    return bytes(value)


def _json_string_bytes(value: str) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_response(body: bytes, provider: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            f"{provider} returned invalid JSON",
        ) from exc
    if not isinstance(payload, dict):
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            f"{provider} response must be a JSON object",
        )
    return payload


def _required_string(payload: dict[str, Any], key: str, provider: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            f"{provider} response is missing {key}",
        )
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            f"provider returned invalid {field}",
        )
    return value


def _first_non_negative_int(
    values: dict[str, Any],
    *keys: str,
    default: int,
) -> int:
    for key in keys:
        if key in values:
            return _non_negative_int(values[key], key)
    return default


def _openrouter_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            "OpenRouter response has no choice",
        )
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            "OpenRouter response has no message",
        )
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [block.get("text") for block in content if isinstance(block, dict)]
        if texts and all(isinstance(text, str) for text in texts):
            return "".join(texts)
    raise LivePromptCacheRejected(
        LivePromptCacheRejectReason.INVALID_RESPONSE,
        "OpenRouter response has no text content",
    )


def _bedrock_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            "Bedrock response has no content",
        )
    texts = [block.get("text") for block in content if isinstance(block, dict)]
    if not texts or not all(isinstance(text, str) for text in texts):
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            "Bedrock response has no text content",
        )
    return "".join(texts)


def _responses_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            "Bedrock Mantle response has no output",
        )
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
    if not texts:
        raise LivePromptCacheRejected(
            LivePromptCacheRejectReason.INVALID_RESPONSE,
            "Bedrock Mantle response has no text output",
        )
    return "".join(texts)


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
