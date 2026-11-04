"""Bounded provider serialization and verification evidence for managed Work."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


MAX_MANAGED_PROMPT_BYTES = 256 * 1024
MAX_VERIFICATION_OUTPUT_BYTES = 1024 * 1024
MAX_TRACKED_FILES = 4_096
MAX_TRACKED_BYTES = 64 * 1024 * 1024
MAX_CHECKS = 64


class ManagedWorkRejected(RuntimeError):
    """Fail-closed rejection of an invalid or unbounded managed-work input."""


@dataclass(frozen=True)
class VerificationCommand:
    criterion: str
    argv: tuple[str, ...]
    timeout_seconds: float | None = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, str) or not self.criterion.strip():
            raise ValueError("criterion must be non-empty")
        if not self.argv or any(not isinstance(part, str) or not part for part in self.argv):
            raise ValueError("argv must contain explicit non-empty arguments")
        if self.timeout_seconds is not None and (self.timeout_seconds <= 0 or self.timeout_seconds > 300):
            raise ValueError("timeout_seconds must be between 0 and 300, or None")


@dataclass(frozen=True)
class ObservedWorkProgress:
    state_sha256: str
    evidence_sha256: str
    remaining_obligations: int
    verified_evidence_count: int
    # Exact owner-observed command receipts; absence on legacy instances is
    # explicit and cannot be treated as an execution attestation.
    evidence_bytes: bytes = b""


def serialize_managed_aider_prompt(
    instruction: str,
    work_bytes: bytes,
    *,
    grounding_bytes: bytes | None = None,
    grounding_bytes_set: Sequence[bytes] = (),
    max_bytes: int = MAX_MANAGED_PROMPT_BYTES,
) -> str:
    """Place immutable Work first and operator instruction last in exact framing."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ManagedWorkRejected("instruction must be non-empty")
    work_text = _utf8_section(work_bytes, "work")
    if grounding_bytes is not None and grounding_bytes_set:
        raise ManagedWorkRejected("use grounding_bytes or grounding_bytes_set, not both")
    if not isinstance(grounding_bytes_set, (tuple, list)):
        raise ManagedWorkRejected("grounding_bytes_set must be a bounded sequence")
    if len(grounding_bytes_set) > 16:
        raise ManagedWorkRejected("grounding_bytes_set exceeds 16 repositories")
    admitted_groundings = (
        tuple(grounding_bytes_set)
        if grounding_bytes is None
        else (grounding_bytes,)
    )
    grounding_texts = tuple(
        _utf8_section(value, f"grounding[{index}]")
        for index, value in enumerate(admitted_groundings, start=1)
    )
    parts = [
        "TARE_MANAGED_WORK/2\n",
        _section_header("WORK", work_bytes),
        work_text,
        "\n",
    ]
    for index, (value, text) in enumerate(
        zip(admitted_groundings, grounding_texts, strict=True), start=1
    ):
        parts.extend(
            [
                _section_header(f"REPOSITORY_GROUNDING_{index:04d}", value),
                text,
                "\n",
            ]
        )
    instruction_bytes = instruction.encode("utf-8", errors="strict")
    parts.extend(
        [
            _section_header("OPERATOR_INSTRUCTION", instruction_bytes),
            instruction,
        ]
    )
    rendered = "".join(parts)
    if len(rendered.encode("utf-8")) > max_bytes:
        raise ManagedWorkRejected("managed Aider prompt exceeds its byte budget")
    return rendered


def observe_work_progress(
    *,
    workspace: str | Path,
    exit_criteria: Sequence[str],
    checks: Sequence[VerificationCommand],
    tracked_paths: Sequence[str | Path],
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> ObservedWorkProgress:
    """Run explicit checks and hash bounded workspace/evidence state."""

    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ManagedWorkRejected("workspace must be an existing directory")
    criteria = _validate_plan(exit_criteria, checks)
    receipts: list[dict[str, object]] = []
    passed: set[str] = set()
    by_criterion = {check.criterion: check for check in checks}
    command_cache: dict[tuple[tuple[str, ...], float | None], dict[str, object]] = {}
    for criterion in criteria:
        check = by_criterion[criterion]
        cache_key = (check.argv, check.timeout_seconds)
        cached = command_cache.get(cache_key)
        if cached is None:
            receipt = _run_check(root, check, runner)
            command_cache[cache_key] = receipt
        else:
            receipt = {**cached, "criterion": criterion, "reused": True}
        receipts.append(receipt)
        if receipt["status"] == "PASS":
            passed.add(criterion)

    state_sha256 = _workspace_sha256(root, tracked_paths)
    evidence_bytes = json.dumps(
        {"schema": "tare.tools/verification-progress/1", "receipts": receipts},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ObservedWorkProgress(
        state_sha256=state_sha256,
        evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        remaining_obligations=len(criteria) - len(passed),
        verified_evidence_count=len(passed),
        evidence_bytes=evidence_bytes,
    )


def _validate_plan(
    exit_criteria: Sequence[str], checks: Sequence[VerificationCommand]
) -> tuple[str, ...]:
    if isinstance(exit_criteria, (str, bytes)) or not exit_criteria:
        raise ManagedWorkRejected("exit criteria must be a non-empty sequence")
    criteria = tuple(exit_criteria)
    if len(criteria) > MAX_CHECKS or any(
        not isinstance(item, str) or not item.strip() for item in criteria
    ):
        raise ManagedWorkRejected("exit criteria are invalid or unbounded")
    if len(set(criteria)) != len(criteria):
        raise ManagedWorkRejected("exit criteria must be unique")
    if not checks or len(checks) > MAX_CHECKS:
        raise ManagedWorkRejected("verification checks are missing or unbounded")
    check_names = tuple(check.criterion for check in checks)
    if len(set(check_names)) != len(check_names) or set(check_names) != set(criteria):
        raise ManagedWorkRejected("verification checks must exactly cover exit criteria")
    return criteria


def _run_check(
    root: Path,
    check: VerificationCommand,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> dict[str, object]:
    from .workspace import isolated_child_environment

    command_bytes = json.dumps(
        list(check.argv), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    base: dict[str, object] = {
        "command_sha256": hashlib.sha256(command_bytes).hexdigest(),
        "criterion": check.criterion,
    }
    try:
        completed = runner(
            list(check.argv),
            check=False,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=check.timeout_seconds,
            env=isolated_child_environment(),
        )
    except subprocess.TimeoutExpired:
        return {**base, "status": "TIMEOUT"}
    except OSError:
        return {**base, "status": "UNAVAILABLE"}

    stdout = _output_bytes(completed.stdout)
    stderr = _output_bytes(completed.stderr)
    if len(stdout) + len(stderr) > MAX_VERIFICATION_OUTPUT_BYTES:
        return {**base, "status": "OUTPUT_TOO_LARGE"}
    return {
        **base,
        "exit_code": completed.returncode,
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _workspace_sha256(root: Path, tracked_paths: Sequence[str | Path]) -> str:
    if not tracked_paths:
        raise ManagedWorkRejected("tracked_paths must be explicit and non-empty")
    records: dict[str, Mapping[str, object]] = {}
    total_bytes = 0
    for item in tracked_paths:
        candidate = Path(item)
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ManagedWorkRejected("tracked path escapes the workspace")
        if ".git" in resolved.relative_to(root).parts:
            raise ManagedWorkRejected("tracked paths cannot include Git internals")
        paths = [resolved]
        if resolved.is_dir():
            paths = sorted(path for path in resolved.rglob("*") if ".git" not in path.parts)
        if not resolved.exists():
            relative = resolved.relative_to(root).as_posix()
            records[relative] = {"status": "MISSING"}
            continue
        for path in paths:
            if path.is_symlink():
                raise ManagedWorkRejected("tracked paths cannot contain symlinks")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            total_bytes += len(content)
            if len(records) >= MAX_TRACKED_FILES or total_bytes > MAX_TRACKED_BYTES:
                raise ManagedWorkRejected("tracked workspace exceeds its observation budget")
            records[relative] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "status": "PRESENT",
            }
    encoded = json.dumps(
        {"schema": "tare.tools/workspace-state/1", "files": records},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utf8_section(value: bytes, name: str) -> str:
    if not isinstance(value, bytes) or not value:
        raise ManagedWorkRejected(f"{name} bytes must be non-empty")
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ManagedWorkRejected(f"{name} bytes must be UTF-8") from exc


def _section_header(name: str, value: bytes) -> str:
    return (
        f"--{name} bytes={len(value)} "
        f"sha256={hashlib.sha256(value).hexdigest()}--\n"
    )


def _output_bytes(value: object) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace")
    return repr(value).encode("utf-8", errors="replace")
