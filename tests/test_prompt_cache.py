import hashlib
import json
import threading
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from compute_plane.prompt_cache import (
    FakePromptCachingAdapter,
    PromptCacheLimits,
    PromptCacheRejected,
    PromptCacheRejectReason,
)


@dataclass(frozen=True)
class ProjectionFixture:
    source_revision: int = 2
    source_snapshot_sha256: str = "a" * 64
    projected_bytes: bytes = b'[{"content":"hello","role":"user"}]'
    projection_generation: int = 1
    source_event_hashes: tuple[str, ...] = ("b" * 64,)


def _adapter(**overrides):
    options = {
        "admit_projection": lambda projection: projection,
        "provider": "fake/provider",
        "model": "fake-model-1",
        "system_prompt": b"You are precise.",
        "tool_schemas": (
            b'{"name":"read","parameters":{"type":"object"}}',
            b'{"name":"write","parameters":{"type":"object"}}',
        ),
    }
    options.update(overrides)
    return FakePromptCachingAdapter(**options)


def test_identical_prefix_has_stable_plan_and_cold_warm_parity():
    projection = ProjectionFixture()
    adapter = _adapter()

    cold = adapter.execute(projection)
    warm = adapter.execute(projection)

    assert cold.request == warm.request
    assert cold.response_bytes == warm.response_bytes
    assert cold.telemetry.hit is False
    assert cold.telemetry.cache_write_tokens > 0
    assert cold.telemetry.cache_read_tokens == 0
    assert warm.telemetry.hit is True
    assert warm.telemetry.cache_read_tokens == cold.telemetry.cache_write_tokens
    assert warm.telemetry.cache_write_tokens == 0

    adapter.clear_cache()
    rebuilt = adapter.execute(projection)
    assert rebuilt.telemetry.hit is False
    assert rebuilt.request == cold.request
    assert rebuilt.response_bytes == cold.response_bytes


@pytest.mark.parametrize(
    ("adapter_change", "projection_change"),
    [
        ({"provider": "fake/other"}, {}),
        ({"model": "fake-model-2"}, {}),
        ({"wire_version": "tare-fake-provider-json-v2"}, {}),
        ({"system_prompt": b"You are precise!"}, {}),
        ({"tool_schemas": (b'{"name":"other"}',)}, {}),
        (
            {
                "tool_schemas": (
                    b'{"name":"write","parameters":{"type":"object"}}',
                    b'{"name":"read","parameters":{"type":"object"}}',
                )
            },
            {},
        ),
        ({"cache_plan_version": "tare-prompt-cache-plan-v2"}, {}),
        ({"ttl_seconds": 301}, {}),
        ({}, {"projection_generation": 2}),
        ({}, {"projected_bytes": b'[{"content":"hello!","role":"user"}]'}),
    ],
)
def test_complete_identity_mutation_matrix_misses(adapter_change, projection_change):
    baseline_projection = ProjectionFixture()
    baseline = _adapter().build_request(baseline_projection)
    changed_projection = replace(baseline_projection, **projection_change)
    changed = _adapter(**adapter_change).build_request(changed_projection)

    assert changed.cache_plan.prefix_digest != baseline.cache_plan.prefix_digest


def test_one_byte_wire_change_misses_without_changing_labels_or_event_hashes():
    class DriftedWireAdapter(FakePromptCachingAdapter):
        def _serialize_request_prefix(self, projection):
            return super()._serialize_request_prefix(projection) + b" "

    projection = ProjectionFixture()
    baseline = _adapter().build_request(projection)
    drifted = DriftedWireAdapter(
        admit_projection=lambda ticket: ticket,
        provider="fake/provider",
        model="fake-model-1",
        system_prompt=b"You are precise.",
        tool_schemas=(
            b'{"name":"read","parameters":{"type":"object"}}',
            b'{"name":"write","parameters":{"type":"object"}}',
        ),
    ).build_request(projection)

    assert drifted.request_prefix_bytes == baseline.request_prefix_bytes + b" "
    assert drifted.cache_plan.wire_version == baseline.cache_plan.wire_version
    assert projection.source_event_hashes == ("b" * 64,)
    assert drifted.cache_plan.prefix_digest != baseline.cache_plan.prefix_digest


def test_provider_decoration_never_enters_projection_or_event_identity():
    projection = ProjectionFixture()
    before_projection = projection.projected_bytes
    before_event_hashes = projection.source_event_hashes

    first = _adapter(provider="fake/one").build_request(projection)
    second = _adapter(provider="fake/two").build_request(projection)

    assert b"cache_control" in first.request_bytes
    assert b"cache_control" not in first.request_prefix_bytes
    assert b"cache_control" not in projection.projected_bytes
    assert projection.projected_bytes == before_projection
    assert projection.source_event_hashes == before_event_hashes
    assert first.cache_plan.prefix_digest != second.cache_plan.prefix_digest
    assert hashlib.sha256(projection.projected_bytes).hexdigest() != first.cache_plan.prefix_digest


def test_stale_projection_admission_fails_before_serialization_or_cache_write():
    def reject_stale(_projection):
        raise RuntimeError("STALE_PROJECTION")

    adapter = _adapter(admit_projection=reject_stale)
    with pytest.raises(RuntimeError, match="STALE_PROJECTION"):
        adapter.execute(ProjectionFixture())
    assert adapter.cache_entries == 0


def test_advance_after_admission_cannot_mix_new_projection_state():
    admitted = threading.Event()
    continue_serialization = threading.Event()
    current = {"revision": 1}
    original = ProjectionFixture(
        source_revision=1,
        projected_bytes=b'[{"content":"old","role":"user"}]',
    )

    def admit(ticket):
        if ticket.source_revision != current["revision"]:
            raise RuntimeError("STALE_PROJECTION")
        admitted.set()
        assert continue_serialization.wait(timeout=2)
        return ticket

    adapter = _adapter(admit_projection=admit)
    outcome = {}

    def build():
        outcome["request"] = adapter.build_request(original)

    worker = threading.Thread(target=build)
    worker.start()
    assert admitted.wait(timeout=2)
    current["revision"] = 2
    newer_projection = b'[{"content":"new","role":"user"}]'
    continue_serialization.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    request = outcome["request"]
    assert request.cache_plan.source_revision == 1
    assert original.projected_bytes in request.request_prefix_bytes
    assert newer_projection not in request.request_bytes


def test_request_bytes_are_exact_valid_fake_wire_and_plan_marks_prefix_end():
    request = _adapter().build_request(ProjectionFixture())

    assert request.request_bytes[: request.cache_plan.cache_point_end] == request.request_prefix_bytes
    decoded = json.loads(request.request_bytes)
    assert decoded["model"] == "fake-model-1"
    assert decoded["messages"] == [{"content": "hello", "role": "user"}]
    assert decoded["cache_control"] == {"type": "ephemeral", "ttl_seconds": 300}


def test_telemetry_and_plan_are_frozen_and_have_no_authority_fields():
    result = _adapter().execute(ProjectionFixture())

    with pytest.raises(FrozenInstanceError):
        result.telemetry.hit = True
    with pytest.raises(FrozenInstanceError):
        result.request.cache_plan.source_revision = 99
    forbidden = {"work", "authority", "evidence", "settlement", "claim_eligible"}
    assert forbidden.isdisjoint(vars(result.telemetry))
    assert forbidden.isdisjoint(vars(result.request.cache_plan))


def test_projection_and_request_bounds_fail_closed():
    projection_limited = _adapter(
        limits=PromptCacheLimits(max_projection_bytes=2),
        system_prompt=b"x",
        tool_schemas=(),
    )

    with pytest.raises(PromptCacheRejected) as rejected:
        projection_limited.build_request(ProjectionFixture(projected_bytes=b"[] "))
    assert rejected.value.reason is PromptCacheRejectReason.INVALID_PROJECTION

    request_limited = _adapter(
        limits=PromptCacheLimits(max_request_bytes=8),
        system_prompt=b"x",
        tool_schemas=(),
    )
    with pytest.raises(PromptCacheRejected) as rejected:
        request_limited.build_request(ProjectionFixture(projected_bytes=b"[]"))
    assert rejected.value.reason is PromptCacheRejectReason.REQUEST_TOO_LARGE
