"""Bounded subprocess adapter for the ``specgraph ground`` contract."""

from __future__ import annotations

import subprocess
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence


MIN_GROUNDING_BYTES = 1_024
MAX_GROUNDING_BYTES = 65_536
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SpecGraphGroundingRejectReason(str, Enum):
    SOURCE_FAILED = "SOURCE_FAILED"
    SOURCE_TIMEOUT = "SOURCE_TIMEOUT"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    OUTPUT_TOO_LARGE = "OUTPUT_TOO_LARGE"


class SpecGraphGroundingRejected(RuntimeError):
    def __init__(
        self,
        reason: SpecGraphGroundingRejectReason,
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason


class SpecGraphScopeDecision(str, Enum):
    CURRENT = "CURRENT"
    DRIFTED = "DRIFTED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SpecGraphScopeCheck:
    decision: SpecGraphScopeDecision
    reason_code: str
    observed_scope_sha256: str | None = None


def materialize_specgraph_grounding(
    *,
    repository: str,
    library_manifest: str | Path,
    command: Sequence[str] = ("specgraph",),
    max_bytes: int = 8_192,
    include_ids: Sequence[str] = (),
    timeout_seconds: float = 5.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> bytes:
    """Return exact canonical stdout for a usable grounding outcome."""

    normalized_ids = _include_ids(include_ids)
    completed = _run_ground(
        repository=repository,
        library_manifest=library_manifest,
        command=command,
        max_bytes=max_bytes,
        include_ids=normalized_ids,
        expected_scope_sha256=None,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    if completed.returncode != 0:
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.SOURCE_FAILED,
            "specgraph grounding returned a non-usable outcome",
        )
    payload = _validated_payload(
        completed.stdout, max_bytes, repository.strip(), normalized_ids
    )
    if payload.get("status") not in {
        "READY",
        "READY_TRUNCATED",
        "NOT_MAPPED",
    }:
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.SOURCE_FAILED,
            "specgraph grounding returned a non-usable outcome",
        )
    return completed.stdout


def check_specgraph_scope(
    *,
    repository: str,
    library_manifest: str | Path,
    include_ids: Sequence[str],
    expected_scope_sha256: str,
    command: Sequence[str] = ("specgraph",),
    max_bytes: int = 8_192,
    timeout_seconds: float = 5.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> SpecGraphScopeCheck:
    """Re-evaluate an existing Work-selected grounding fence read-only."""

    normalized_ids = _include_ids(include_ids)
    try:
        completed = _run_ground(
            repository=repository,
            library_manifest=library_manifest,
            command=command,
            max_bytes=max_bytes,
            include_ids=normalized_ids,
            expected_scope_sha256=expected_scope_sha256,
            timeout_seconds=timeout_seconds,
            runner=runner,
        )
        payload = _validated_payload(
            completed.stdout, max_bytes, repository.strip(), normalized_ids
        )
    except SpecGraphGroundingRejected as exc:
        return SpecGraphScopeCheck(
            SpecGraphScopeDecision.UNAVAILABLE,
            exc.reason.value,
        )

    status = payload.get("status")
    observed = payload.get("grounding_scope_sha256")
    if status in {"READY", "READY_TRUNCATED", "NOT_MAPPED"} and completed.returncode == 0:
        if observed != expected_scope_sha256:
            return SpecGraphScopeCheck(
                SpecGraphScopeDecision.DRIFTED,
                "SCOPE_MISMATCH_ON_READY",
                observed if isinstance(observed, str) else None,
            )
        return SpecGraphScopeCheck(
            SpecGraphScopeDecision.CURRENT,
            "GROUNDING_SCOPE_CURRENT",
            observed,
        )
    if status == "DRIFT":
        return SpecGraphScopeCheck(
            SpecGraphScopeDecision.DRIFTED,
            "GROUNDING_SCOPE_CHANGED",
            observed if isinstance(observed, str) else None,
        )
    return SpecGraphScopeCheck(
        SpecGraphScopeDecision.UNAVAILABLE,
        f"GROUNDING_STATUS_{status}" if isinstance(status, str) else "INVALID_STATUS",
        observed if isinstance(observed, str) else None,
    )


def _run_ground(
    *,
    repository: str,
    library_manifest: str | Path,
    command: Sequence[str],
    max_bytes: int,
    include_ids: tuple[str, ...],
    expected_scope_sha256: str | None,
    timeout_seconds: float,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> subprocess.CompletedProcess[bytes]:

    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("repository must be non-empty")
    repo = repository.strip()
    if len(repo.encode("utf-8")) > 256:
        raise ValueError("repository exceeds 256 bytes")
    manifest = str(library_manifest)
    if not manifest.strip():
        raise ValueError("library_manifest must be explicit")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must contain explicit non-empty arguments")
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or not MIN_GROUNDING_BYTES <= max_bytes <= MAX_GROUNDING_BYTES
    ):
        raise ValueError(
            f"max_bytes must be between {MIN_GROUNDING_BYTES} and {MAX_GROUNDING_BYTES}"
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    argv = [
        *command,
        "ground",
        "--repo",
        repo,
        "--library-manifest",
        manifest,
        "--max-bytes",
        str(max_bytes),
        "--format",
        "json",
    ]
    for include_id in include_ids:
        argv.extend(["--include-id", include_id])
    if expected_scope_sha256 is not None:
        if not _SHA256.fullmatch(expected_scope_sha256):
            raise ValueError(
                "expected_scope_sha256 must be a lowercase SHA-256 digest"
            )
        argv.extend(["--expected-scope-sha256", expected_scope_sha256])
    try:
        completed = runner(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.SOURCE_TIMEOUT,
            "specgraph grounding timed out",
        ) from exc
    except OSError as exc:
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.SOURCE_FAILED,
            "specgraph grounding could not start",
        ) from exc

    return completed


def _include_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or len(value) > 128:
        raise ValueError("include_ids must be a bounded string sequence")
    if any(not isinstance(item, str) for item in value):
        raise ValueError("include_ids must contain strings")
    normalized = tuple(item.strip() for item in value)
    if any(not item or len(item.encode("utf-8")) > 256 for item in normalized):
        raise ValueError("include_ids must contain bounded non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("include_ids must not contain duplicates")
    return tuple(sorted(normalized, key=lambda item: (item.casefold(), item)))


def _validated_payload(
    stdout: bytes,
    max_bytes: int,
    repository: str,
    include_ids: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(stdout, bytes) or not stdout:
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.INVALID_OUTPUT,
            "specgraph grounding returned invalid stdout",
        )
    if len(stdout) > max_bytes:
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.OUTPUT_TOO_LARGE,
            "specgraph grounding exceeded the requested byte budget",
        )
    try:
        payload = json.loads(stdout.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.INVALID_OUTPUT,
            "specgraph grounding returned invalid JSON",
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("byte_count") != len(stdout)
        or payload.get("schema") != "tare.tools/repository-grounding/1"
        or payload.get("repository") != repository.strip()
        or payload.get("requested_ids") != list(include_ids)
    ):
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.INVALID_OUTPUT,
            "specgraph grounding identity is inconsistent",
        )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if canonical != stdout:
        raise SpecGraphGroundingRejected(
            SpecGraphGroundingRejectReason.INVALID_OUTPUT,
            "specgraph grounding output is non-canonical",
        )
    return payload
