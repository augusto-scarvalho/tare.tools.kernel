"""Bounded subprocess transport and scope checks for ``backlog ground``."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence


MIN_WORK_BYTES = 1_024
MAX_WORK_BYTES = 65_536
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BacklogWorkRejectReason(str, Enum):
    SOURCE_FAILED = "SOURCE_FAILED"
    SOURCE_TIMEOUT = "SOURCE_TIMEOUT"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    OUTPUT_TOO_LARGE = "OUTPUT_TOO_LARGE"
    WORK_NOT_EXECUTABLE = "WORK_NOT_EXECUTABLE"


class BacklogWorkRejected(RuntimeError):
    def __init__(self, reason: BacklogWorkRejectReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class BacklogScopeDecision(str, Enum):
    CURRENT = "CURRENT"
    DRIFTED = "DRIFTED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class BacklogScopeCheck:
    decision: BacklogScopeDecision
    reason_code: str
    observed_scope_sha256: str | None = None


def materialize_backlog_work(
    *,
    work_id: str,
    work_graph: str | Path,
    command: Sequence[str] = ("backlog",),
    max_bytes: int = 8_192,
    timeout_seconds: float = 5.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> bytes:
    """Return exact stdout only when the work item is execution-eligible."""

    completed = _run_ground(
        work_id=work_id,
        work_graph=work_graph,
        command=command,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
        expected_scope_sha256=None,
        runner=runner,
    )
    payload = _validated_payload(completed.stdout, max_bytes)
    if completed.returncode != 0 or payload.get("status") != "READY":
        raise BacklogWorkRejected(
            BacklogWorkRejectReason.WORK_NOT_EXECUTABLE,
            "backlog work is not eligible for managed execution",
        )
    return completed.stdout


def check_backlog_scope(
    *,
    work_id: str,
    work_graph: str | Path,
    expected_scope_sha256: str,
    command: Sequence[str] = ("backlog",),
    max_bytes: int = 8_192,
    timeout_seconds: float = 5.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> BacklogScopeCheck:
    """Re-evaluate an existing work fence without mutating the graph."""

    try:
        completed = _run_ground(
            work_id=work_id,
            work_graph=work_graph,
            command=command,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
            expected_scope_sha256=expected_scope_sha256,
            runner=runner,
        )
        payload = _validated_payload(completed.stdout, max_bytes)
    except BacklogWorkRejected as exc:
        return BacklogScopeCheck(
            BacklogScopeDecision.UNAVAILABLE,
            exc.reason.value,
        )

    status = payload.get("status")
    observed = payload.get("execution_scope_sha256")
    if status == "READY" and completed.returncode == 0:
        if observed != expected_scope_sha256:
            return BacklogScopeCheck(
                BacklogScopeDecision.DRIFTED,
                "SCOPE_MISMATCH_ON_READY",
                observed if isinstance(observed, str) else None,
            )
        return BacklogScopeCheck(
            BacklogScopeDecision.CURRENT,
            "EXECUTION_SCOPE_CURRENT",
            observed,
        )
    if status == "DRIFT":
        return BacklogScopeCheck(
            BacklogScopeDecision.DRIFTED,
            "EXECUTION_SCOPE_CHANGED",
            observed if isinstance(observed, str) else None,
        )
    return BacklogScopeCheck(
        BacklogScopeDecision.UNAVAILABLE,
        f"WORK_STATUS_{status}" if isinstance(status, str) else "INVALID_STATUS",
        observed if isinstance(observed, str) else None,
    )


def _run_ground(
    *,
    work_id: str,
    work_graph: str | Path,
    command: Sequence[str],
    max_bytes: int,
    timeout_seconds: float,
    expected_scope_sha256: str | None,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> subprocess.CompletedProcess[bytes]:
    if not isinstance(work_id, str) or not work_id.strip():
        raise ValueError("work_id must be non-empty")
    graph_path = str(work_graph)
    if not graph_path.strip():
        raise ValueError("work_graph must be explicit")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must contain explicit non-empty arguments")
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or not MIN_WORK_BYTES <= max_bytes <= MAX_WORK_BYTES
    ):
        raise ValueError(f"max_bytes must be between {MIN_WORK_BYTES} and {MAX_WORK_BYTES}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    argv = [
        *command,
        "--format",
        "json",
        "ground",
        work_id,
        "--work-graph",
        graph_path,
        "--profile",
        "operational",
        "--max-bytes",
        str(max_bytes),
    ]
    if expected_scope_sha256 is not None:
        if not _SHA256.fullmatch(expected_scope_sha256):
            raise ValueError(
                "expected_scope_sha256 must be a lowercase SHA-256 digest"
            )
        argv.extend(["--expected-scope-sha256", expected_scope_sha256])
    try:
        return runner(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise BacklogWorkRejected(
            BacklogWorkRejectReason.SOURCE_TIMEOUT,
            "backlog grounding timed out",
        ) from exc
    except OSError as exc:
        raise BacklogWorkRejected(
            BacklogWorkRejectReason.SOURCE_FAILED,
            "backlog grounding could not start",
        ) from exc


def _validated_payload(stdout: bytes, max_bytes: int) -> dict[str, Any]:
    if not isinstance(stdout, bytes) or not stdout:
        raise BacklogWorkRejected(
            BacklogWorkRejectReason.INVALID_OUTPUT,
            "backlog grounding returned invalid stdout",
        )
    if len(stdout) > max_bytes:
        raise BacklogWorkRejected(
            BacklogWorkRejectReason.OUTPUT_TOO_LARGE,
            "backlog grounding exceeded the requested byte budget",
        )
    try:
        payload = json.loads(stdout.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BacklogWorkRejected(
            BacklogWorkRejectReason.INVALID_OUTPUT,
            "backlog grounding returned invalid JSON",
        ) from exc
    if not isinstance(payload, dict) or payload.get("byte_count") != len(stdout):
        raise BacklogWorkRejected(
            BacklogWorkRejectReason.INVALID_OUTPUT,
            "backlog grounding byte identity is inconsistent",
        )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if canonical != stdout or payload.get("schema") != "tare.tools/work-grounding/1":
        raise BacklogWorkRejected(
            BacklogWorkRejectReason.INVALID_OUTPUT,
            "backlog grounding contract is unsupported or non-canonical",
        )
    return payload
