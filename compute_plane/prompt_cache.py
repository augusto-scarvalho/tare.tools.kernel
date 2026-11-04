"""Bounded fake-provider prompt-cache planning for the P2 laboratory.

Provider request serialization and cache decoration belong to the Kernel. The
canonical tactical conversation remains owned by ``tare.tools.agent-runtime``;
this module consumes only a detached, admitted projection ticket.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Protocol


PROMPT_CACHE_DIGEST_DOMAIN = b"tare-provider-prefix-digest-v1\x00"
PROMPT_CACHE_PLAN_VERSION = "tare-prompt-cache-plan-v1"
FAKE_PROVIDER_WIRE_VERSION = "tare-fake-provider-json-v1"


class ProjectionTicket(Protocol):
    """Narrow structural port implemented by Runtime ``ContextProjection``."""

    source_revision: int
    source_snapshot_sha256: str
    projected_bytes: bytes
    projection_generation: int


ProjectionAdmission = Callable[[ProjectionTicket], ProjectionTicket]


class PromptCacheRejectReason(str, Enum):
    INVALID_PROJECTION = "INVALID_PROJECTION"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"


class PromptCacheRejected(RuntimeError):
    """Fail-closed P2 rejection before fake-provider execution."""

    def __init__(self, reason: PromptCacheRejectReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PromptCacheLimits:
    max_system_prompt_bytes: int = 256 * 1024
    max_tool_schemas: int = 128
    max_tool_schema_bytes: int = 256 * 1024
    max_projection_bytes: int = 2 * 1024 * 1024
    max_request_bytes: int = 4 * 1024 * 1024
    max_cache_entries: int = 128

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PromptCachePlan:
    """Immutable, non-authoritative description of one provider cache point."""

    provider: str
    model: str
    wire_version: str
    cache_plan_version: str
    source_revision: int
    source_snapshot_sha256: str
    projection_generation: int
    prefix_digest: str
    cache_point_end: int
    ttl_seconds: int


@dataclass(frozen=True)
class ProviderRequest:
    """Exact fake-provider wire bytes plus their cache plan."""

    request_bytes: bytes
    request_prefix_bytes: bytes
    cache_plan: PromptCachePlan


@dataclass(frozen=True)
class PromptCacheTelemetry:
    """Detached cache observations; never Work, Authority, Evidence or settlement."""

    hit: bool
    cache_read_tokens: int
    cache_write_tokens: int
    prefix_digest: str


@dataclass(frozen=True)
class FakeProviderResult:
    request: ProviderRequest
    response_bytes: bytes
    telemetry: PromptCacheTelemetry


@dataclass(frozen=True)
class DetachedProjection:
    source_revision: int
    source_snapshot_sha256: str
    projected_bytes: bytes
    projection_generation: int


class FakePromptCachingAdapter:
    """Hermetic P2 adapter with a bounded, disposable in-memory fake cache.

    Admission is injected from the Runtime owner. It linearizes once before
    serialization. If the canonical log advances afterward, this adapter still
    serializes only the detached admitted ticket and attributes the output to its
    original source revision.
    """

    def __init__(
        self,
        *,
        admit_projection: ProjectionAdmission,
        provider: str,
        model: str,
        system_prompt: str | bytes,
        tool_schemas: Sequence[bytes] = (),
        wire_version: str = FAKE_PROVIDER_WIRE_VERSION,
        cache_plan_version: str = PROMPT_CACHE_PLAN_VERSION,
        ttl_seconds: int = 300,
        limits: PromptCacheLimits | None = None,
    ) -> None:
        if not callable(admit_projection):
            raise ValueError("admit_projection must be callable")
        self._limits = limits or PromptCacheLimits()
        self._admit_projection = admit_projection
        self._provider = _bounded_label("provider", provider)
        self._model = _bounded_label("model", model)
        self._wire_version = _bounded_label("wire_version", wire_version)
        self._cache_plan_version = _bounded_label(
            "cache_plan_version", cache_plan_version
        )
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds < 1:
            raise ValueError("ttl_seconds must be a positive integer")
        self._ttl_seconds = ttl_seconds
        self._system_prompt = _utf8_bytes("system_prompt", system_prompt)
        if len(self._system_prompt) > self._limits.max_system_prompt_bytes:
            raise ValueError("system_prompt exceeds max_system_prompt_bytes")
        if not isinstance(tool_schemas, Sequence) or isinstance(
            tool_schemas, (str, bytes, bytearray)
        ):
            raise ValueError("tool_schemas must be an ordered sequence of bytes")
        if len(tool_schemas) > self._limits.max_tool_schemas:
            raise ValueError("tool_schemas exceeds max_tool_schemas")
        self._tool_schemas = tuple(
            _validated_json_object_bytes(
                f"tool_schemas[{index}]",
                value,
                self._limits.max_tool_schema_bytes,
            )
            for index, value in enumerate(tool_schemas)
        )
        self._cache_lock = Lock()
        self._fake_cache: dict[str, bytes] = {}

    @property
    def cache_entries(self) -> int:
        with self._cache_lock:
            return len(self._fake_cache)

    def clear_cache(self) -> None:
        """Physically replace all disposable fake cache state."""

        with self._cache_lock:
            self._fake_cache = {}

    def build_request(self, projection: ProjectionTicket) -> ProviderRequest:
        admitted = self._admit_projection(projection)
        detached = detach_projection(admitted, self._limits)
        prefix = self._serialize_request_prefix(detached)
        cache_point_end = len(prefix)
        cache_decoration = (
            b',"cache_control":{"type":"ephemeral","ttl_seconds":'
            + str(self._ttl_seconds).encode("ascii")
            + b"}}"
        )
        request_bytes = prefix + cache_decoration
        if len(request_bytes) > self._limits.max_request_bytes:
            raise PromptCacheRejected(
                PromptCacheRejectReason.REQUEST_TOO_LARGE,
                "serialized provider request exceeds max_request_bytes",
            )
        digest = self._prefix_digest(
            detached=detached,
            request_prefix_bytes=prefix,
            cache_point_end=cache_point_end,
        )
        plan = PromptCachePlan(
            provider=self._provider,
            model=self._model,
            wire_version=self._wire_version,
            cache_plan_version=self._cache_plan_version,
            source_revision=detached.source_revision,
            source_snapshot_sha256=detached.source_snapshot_sha256,
            projection_generation=detached.projection_generation,
            prefix_digest=digest,
            cache_point_end=cache_point_end,
            ttl_seconds=self._ttl_seconds,
        )
        return ProviderRequest(
            request_bytes=request_bytes,
            request_prefix_bytes=prefix,
            cache_plan=plan,
        )

    def execute(self, projection: ProjectionTicket) -> FakeProviderResult:
        """Exercise hit/miss behavior without network or provider dependencies."""

        request = self.build_request(projection)
        response = _fake_response(request.request_bytes)
        digest = request.cache_plan.prefix_digest
        prefix_tokens = max(1, math.ceil(len(request.request_prefix_bytes) / 4))
        with self._cache_lock:
            hit = digest in self._fake_cache
            if not hit:
                if len(self._fake_cache) >= self._limits.max_cache_entries:
                    oldest_digest = next(iter(self._fake_cache))
                    del self._fake_cache[oldest_digest]
                self._fake_cache[digest] = response
        telemetry = PromptCacheTelemetry(
            hit=hit,
            cache_read_tokens=prefix_tokens if hit else 0,
            cache_write_tokens=0 if hit else prefix_tokens,
            prefix_digest=digest,
        )
        return FakeProviderResult(
            request=request,
            response_bytes=response,
            telemetry=telemetry,
        )

    def _serialize_request_prefix(self, projection: DetachedProjection) -> bytes:
        """Serialize the exact fake-provider prefix up to the cache point."""

        return (
            b'{"model":'
            + _json_string_bytes(self._model)
            + b',"system":'
            + _json_string_bytes(self._system_prompt.decode("utf-8"))
            + b',"tools":['
            + b",".join(self._tool_schemas)
            + b'],"messages":'
            + projection.projected_bytes
        )

    def _prefix_digest(
        self,
        *,
        detached: DetachedProjection,
        request_prefix_bytes: bytes,
        cache_point_end: int,
    ) -> str:
        return compute_provider_prefix_digest(
            provider=self._provider,
            model=self._model,
            wire_version=self._wire_version,
            system_prompt_bytes=self._system_prompt,
            tool_schema_bytes=self._tool_schemas,
            projected_message_prefix_bytes=detached.projected_bytes,
            projection_generation=detached.projection_generation,
            cache_plan_version=self._cache_plan_version,
            cache_point_end=cache_point_end,
            ttl_seconds=self._ttl_seconds,
            request_prefix_bytes=request_prefix_bytes,
        )


def detach_projection(
    projection: ProjectionTicket,
    limits: PromptCacheLimits,
) -> DetachedProjection:
    """Validate and copy an already admitted Runtime projection ticket."""

    try:
        source_revision = projection.source_revision
        snapshot_sha256 = projection.source_snapshot_sha256
        projected_bytes = projection.projected_bytes
        generation = projection.projection_generation
    except AttributeError as exc:
        raise PromptCacheRejected(
            PromptCacheRejectReason.INVALID_PROJECTION,
            f"projection ticket is incomplete: {exc}",
        ) from exc
    if (
        not isinstance(source_revision, int)
        or isinstance(source_revision, bool)
        or source_revision < 0
    ):
        raise PromptCacheRejected(
            PromptCacheRejectReason.INVALID_PROJECTION,
            "source_revision must be a non-negative integer",
        )
    if not _is_sha256(snapshot_sha256):
        raise PromptCacheRejected(
            PromptCacheRejectReason.INVALID_PROJECTION,
            "source_snapshot_sha256 must be a lowercase SHA-256 digest",
        )
    if not isinstance(projected_bytes, bytes):
        raise PromptCacheRejected(
            PromptCacheRejectReason.INVALID_PROJECTION,
            "projected_bytes must be immutable bytes",
        )
    if len(projected_bytes) > limits.max_projection_bytes:
        raise PromptCacheRejected(
            PromptCacheRejectReason.INVALID_PROJECTION,
            "projected_bytes exceeds max_projection_bytes",
        )
    try:
        messages = json.loads(projected_bytes)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PromptCacheRejected(
            PromptCacheRejectReason.INVALID_PROJECTION,
            f"projected_bytes is not valid UTF-8 JSON: {exc}",
        ) from exc
    if not isinstance(messages, list):
        raise PromptCacheRejected(
            PromptCacheRejectReason.INVALID_PROJECTION,
            "projected_bytes must encode a JSON array",
        )
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise PromptCacheRejected(
            PromptCacheRejectReason.INVALID_PROJECTION,
            "projection_generation must be a positive integer",
        )
    return DetachedProjection(
        source_revision=source_revision,
        source_snapshot_sha256=snapshot_sha256,
        projected_bytes=bytes(projected_bytes),
        projection_generation=generation,
    )


def compute_provider_prefix_digest(
    *,
    provider: str,
    model: str,
    wire_version: str,
    system_prompt_bytes: bytes,
    tool_schema_bytes: Sequence[bytes],
    projected_message_prefix_bytes: bytes,
    projection_generation: int,
    cache_plan_version: str,
    cache_point_end: int,
    ttl_seconds: int,
    request_prefix_bytes: bytes,
) -> str:
    """Hash complete provider bindings plus the exact serialized cache prefix."""

    digest = hashlib.sha256(PROMPT_CACHE_DIGEST_DOMAIN)
    fields = [
        (b"provider", provider.encode("utf-8")),
        (b"model", model.encode("utf-8")),
        (b"wire_version", wire_version.encode("utf-8")),
        (b"system_prompt", system_prompt_bytes),
        (b"projection_bytes", projected_message_prefix_bytes),
        (b"projection_generation", projection_generation.to_bytes(8, "big")),
        (b"cache_plan_version", cache_plan_version.encode("utf-8")),
        (b"cache_point_end", cache_point_end.to_bytes(8, "big")),
        (b"ttl_seconds", ttl_seconds.to_bytes(8, "big")),
    ]
    fields.extend(
        (f"tool_schema[{index}]".encode("ascii"), schema)
        for index, schema in enumerate(tool_schema_bytes)
    )
    fields.append((b"request_prefix_bytes", request_prefix_bytes))
    for name, value in fields:
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _bounded_label(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if len(encoded) > 256:
        raise ValueError(f"{name} cannot exceed 256 UTF-8 bytes")
    return value


def _utf8_bytes(name: str, value: str | bytes) -> bytes:
    if isinstance(value, str):
        try:
            return value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ValueError(f"{name} must be valid UTF-8") from exc
    if isinstance(value, bytes):
        try:
            value.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ValueError(f"{name} must be valid UTF-8") from exc
        return bytes(value)
    raise ValueError(f"{name} must be str or bytes")


def _validated_json_object_bytes(name: str, value: bytes, limit: int) -> bytes:
    if not isinstance(value, bytes):
        raise ValueError(f"{name} must be immutable bytes")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds max_tool_schema_bytes")
    try:
        decoded = json.loads(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return bytes(value)


def _json_string_bytes(value: str) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _fake_response(request_bytes: bytes) -> bytes:
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    return b'{"request_sha256":"' + request_sha256.encode("ascii") + b'"}'
