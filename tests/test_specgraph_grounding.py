from __future__ import annotations

import subprocess
import json

import pytest

from compute_plane.specgraph_grounding import (
    SpecGraphGroundingRejected,
    SpecGraphGroundingRejectReason,
    SpecGraphScopeDecision,
    check_specgraph_scope,
    materialize_specgraph_grounding,
)


def _payload(
    *,
    status: str = "READY",
    reason_code: str = "OK",
    repository: str = "tare.tools.kernel",
    requested_ids: list[str] | None = None,
    scope: str = "a" * 64,
) -> bytes:
    data = {
        "byte_count": 0,
        "entries": [],
        "grounding_scope_sha256": scope,
        "manifest_sha256": "b" * 64,
        "manifest_version": "2.0.0",
        "matched_count": 0,
        "omitted_count": 0,
        "reason_code": reason_code,
        "repository": repository,
        "requested_ids": requested_ids or [],
        "returned_count": 0,
        "schema": "tare.tools/repository-grounding/1",
        "status": status,
    }
    for _ in range(8):
        encoded = json.dumps(
            data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        if data["byte_count"] == len(encoded):
            return encoded
        data["byte_count"] = len(encoded)
    raise AssertionError("byte_count did not converge")


def test_adapter_transports_exact_stdout_and_explicit_arguments() -> None:
    exact = _payload()
    observed: dict[str, object] = {}

    def runner(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, exact, b"")

    result = materialize_specgraph_grounding(
        repository="tare.tools.kernel",
        library_manifest="C:/library/LIBRARY_MANIFEST.json",
        command=("python", "-m", "specgraph.cli"),
        runner=runner,
    )

    assert result is exact
    assert observed["argv"] == [
        "python",
        "-m",
        "specgraph.cli",
        "ground",
        "--repo",
        "tare.tools.kernel",
        "--library-manifest",
        "C:/library/LIBRARY_MANIFEST.json",
        "--max-bytes",
        "8192",
        "--format",
        "json",
    ]
    assert observed["kwargs"] == {
        "check": False,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": 5.0,
    }


@pytest.mark.parametrize(
    ("completed", "reason"),
    [
        (
            subprocess.CompletedProcess(["specgraph"], 1, b"private", b"secret"),
            SpecGraphGroundingRejectReason.SOURCE_FAILED,
        ),
        (
            subprocess.CompletedProcess(["specgraph"], 0, b"", b""),
            SpecGraphGroundingRejectReason.INVALID_OUTPUT,
        ),
        (
            subprocess.CompletedProcess(["specgraph"], 0, b"x" * 1_025, b""),
            SpecGraphGroundingRejectReason.OUTPUT_TOO_LARGE,
        ),
    ],
)
def test_adapter_fails_closed_without_exposing_process_output(completed, reason) -> None:
    def runner(*_args, **_kwargs):
        return completed

    with pytest.raises(SpecGraphGroundingRejected) as rejected:
        materialize_specgraph_grounding(
            repository="tare.tools.kernel",
            library_manifest="manifest.json",
            max_bytes=1_024,
            runner=runner,
        )

    assert rejected.value.reason is reason
    assert "private" not in str(rejected.value)
    assert "secret" not in str(rejected.value)


def test_timeout_is_sanitized_and_fails_closed() -> None:
    def runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["specgraph"], 5)

    with pytest.raises(SpecGraphGroundingRejected) as rejected:
        materialize_specgraph_grounding(
            repository="tare.tools.kernel",
            library_manifest="manifest.json",
            runner=runner,
        )

    assert rejected.value.reason is SpecGraphGroundingRejectReason.SOURCE_TIMEOUT


def test_manifest_source_must_be_supplied_explicitly() -> None:
    with pytest.raises(ValueError, match="explicit"):
        materialize_specgraph_grounding(
            repository="tare.tools.kernel",
            library_manifest="",
        )


def test_work_ids_are_forwarded_in_stable_order_and_identity_checked() -> None:
    exact = _payload(requested_ids=["ADR-A", "SPEC-B"])
    observed: dict[str, object] = {}

    def runner(argv, **_kwargs):
        observed["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, exact, b"")

    result = materialize_specgraph_grounding(
        repository="tare.tools.kernel",
        library_manifest="manifest.json",
        include_ids=("SPEC-B", "ADR-A"),
        runner=runner,
    )

    assert result is exact
    assert observed["argv"][-4:] == [
        "--include-id",
        "ADR-A",
        "--include-id",
        "SPEC-B",
    ]


def test_scope_check_distinguishes_current_drift_and_unavailable() -> None:
    current_scope = "a" * 64

    def current_runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, _payload(scope=current_scope), b"")

    def drift_runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            1,
            _payload(
                status="DRIFT",
                reason_code="GROUNDING_SCOPE_CHANGED",
                scope="c" * 64,
            ),
            b"",
        )

    def unavailable_runner(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 5)

    current = check_specgraph_scope(
        repository="tare.tools.kernel",
        library_manifest="manifest.json",
        include_ids=(),
        expected_scope_sha256=current_scope,
        runner=current_runner,
    )
    drifted = check_specgraph_scope(
        repository="tare.tools.kernel",
        library_manifest="manifest.json",
        include_ids=(),
        expected_scope_sha256=current_scope,
        runner=drift_runner,
    )
    unavailable = check_specgraph_scope(
        repository="tare.tools.kernel",
        library_manifest="manifest.json",
        include_ids=(),
        expected_scope_sha256=current_scope,
        runner=unavailable_runner,
    )

    assert current.decision is SpecGraphScopeDecision.CURRENT
    assert drifted.decision is SpecGraphScopeDecision.DRIFTED
    assert drifted.observed_scope_sha256 == "c" * 64
    assert unavailable.decision is SpecGraphScopeDecision.UNAVAILABLE
