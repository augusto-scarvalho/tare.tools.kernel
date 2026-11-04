from __future__ import annotations

import json
import subprocess

import pytest

from compute_plane.backlog_work import (
    BacklogScopeDecision,
    BacklogWorkRejected,
    BacklogWorkRejectReason,
    check_backlog_scope,
    materialize_backlog_work,
)


def payload(status: str = "READY", scope: str = "c" * 64) -> bytes:
    value = {
        "byte_count": 0,
        "execution_scope_sha256": scope,
        "schema": "tare.tools/work-grounding/1",
        "status": status,
    }
    for _ in range(8):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        if value["byte_count"] == len(raw):
            return raw
        value["byte_count"] = len(raw)
    raise AssertionError("fixture did not converge")


def test_materializer_returns_exact_ready_stdout_and_explicit_command() -> None:
    exact = payload()
    seen: dict[str, object] = {}

    def runner(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, exact, b"")

    result = materialize_backlog_work(
        work_id="WORK-1",
        work_graph="C:/state/work-graph.json",
        command=("python", "-m", "graph_backlog.cli"),
        runner=runner,
    )

    assert result is exact
    assert seen["argv"] == [
        "python",
        "-m",
        "graph_backlog.cli",
        "--format",
        "json",
        "ground",
        "WORK-1",
        "--work-graph",
        "C:/state/work-graph.json",
        "--profile",
        "operational",
        "--max-bytes",
        "8192",
    ]


def test_scope_check_distinguishes_current_drift_and_unavailable() -> None:
    outputs = [
        subprocess.CompletedProcess([], 0, payload("READY"), b""),
        subprocess.CompletedProcess([], 1, payload("DRIFT", "d" * 64), b""),
        subprocess.CompletedProcess([], 1, payload("BLOCKED"), b""),
    ]

    def runner(*_args, **_kwargs):
        return outputs.pop(0)

    common = {
        "work_id": "WORK-1",
        "work_graph": "graph.json",
        "expected_scope_sha256": "c" * 64,
        "runner": runner,
    }
    current = check_backlog_scope(**common)
    drifted = check_backlog_scope(**common)
    unavailable = check_backlog_scope(**common)

    assert current.decision is BacklogScopeDecision.CURRENT
    assert drifted.decision is BacklogScopeDecision.DRIFTED
    assert drifted.observed_scope_sha256 == "d" * 64
    assert unavailable.decision is BacklogScopeDecision.UNAVAILABLE


def test_non_executable_initial_work_fails_closed() -> None:
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, payload("BLOCKED"), b"private")

    with pytest.raises(BacklogWorkRejected) as rejected:
        materialize_backlog_work(
            work_id="WORK-1",
            work_graph="graph.json",
            runner=runner,
        )

    assert rejected.value.reason is BacklogWorkRejectReason.WORK_NOT_EXECUTABLE
    assert "private" not in str(rejected.value)


def test_scope_timeout_becomes_unavailable() -> None:
    def runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["backlog"], 5)

    result = check_backlog_scope(
        work_id="WORK-1",
        work_graph="graph.json",
        expected_scope_sha256="c" * 64,
        runner=runner,
    )

    assert result.decision is BacklogScopeDecision.UNAVAILABLE
    assert result.reason_code == "SOURCE_TIMEOUT"


def test_noncanonical_or_incompatible_grounding_fails_closed() -> None:
    canonical = payload()
    noncanonical = canonical + b"\n"

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, noncanonical, b"")

    with pytest.raises(BacklogWorkRejected) as rejected:
        materialize_backlog_work(
            work_id="WORK-1",
            work_graph="graph.json",
            runner=runner,
        )

    assert rejected.value.reason is BacklogWorkRejectReason.INVALID_OUTPUT


def test_scope_check_rejects_invalid_expected_digest_before_transport() -> None:
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess([], 0, payload(), b"")

    with pytest.raises(ValueError, match="SHA-256"):
        check_backlog_scope(
            work_id="WORK-1",
            work_graph="graph.json",
            expected_scope_sha256="not-a-digest",
            runner=runner,
        )

    assert not called
