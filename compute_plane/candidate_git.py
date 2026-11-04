"""Isolated Git object operations for an OS-owned immutable candidate receipt."""

from __future__ import annotations

import hashlib
import subprocess
from .managed_process import host_executable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

from .workspace import (
    WorkspaceLease,
    WorkspaceQualificationRejected,
    _run_git,
    isolated_child_environment,
    qualify_ephemeral_git_worktree,
)


_REGULAR_MODES = {"000000", "100644", "100755"}


class CandidateGitRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class ChangedEntry:
    path: str
    old_mode: str
    new_mode: str
    status: str


@dataclass(frozen=True)
class CandidateGitIdentity:
    candidate_commit: str
    candidate_tree: str
    diff_sha256: str
    changed_paths: tuple[str, ...]
    changed_entries: tuple[ChangedEntry, ...]


def materialize_candidate_commit(
    lease: WorkspaceLease,
    *,
    tracked_paths: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[CandidateGitIdentity, WorkspaceLease]:
    """Capture all eligible bytes, validate them, then pin a clean detached HEAD."""

    identity = prepare_candidate_commit(lease, tracked_paths=tracked_paths, runner=runner)
    return pin_candidate_commit(lease, candidate_commit=identity.candidate_commit,
                                tracked_paths=tracked_paths, runner=runner)


def prepare_candidate_commit(
    lease: WorkspaceLease, *, tracked_paths: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CandidateGitIdentity:
    """Create a deterministic immutable object without moving HEAD.

    The host can durably record this identity before pinning. Orphan objects
    before that record are harmless; they carry no admission authority.
    """

    qualification = qualify_ephemeral_git_worktree(
        lease.root, require_clean=False, runner=runner
    )
    if qualification.head != lease.base_commit:
        raise CandidateGitRejected("BASE_COMMIT_MISMATCH")
    tracked = _tracked_paths(tracked_paths)
    _git(lease.root, "add", "-A", "--", ".", runner=runner)
    tree = _git(lease.root, "write-tree", runner=runner)
    base_tree = _git(lease.root, "rev-parse", f"{lease.base_commit}^{{tree}}", runner=runner)
    if tree == base_tree:
        raise CandidateGitRejected("NO_CANDIDATE_CHANGES")
    candidate = _git(
        lease.root,
        "commit-tree", tree, "-p", lease.base_commit,
        runner=runner,
        extra_env={
            "GIT_AUTHOR_NAME": "tare.tools",
            "GIT_AUTHOR_EMAIL": "candidate@tare.tools.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_NAME": "tare.tools",
            "GIT_COMMITTER_EMAIL": "candidate@tare.tools.invalid",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        },
        stdin="tare.tools immutable candidate\n",
    )
    return inspect_candidate_commit(
        lease.repository_root,
        base_commit=lease.base_commit,
        candidate_commit=candidate,
        tracked_paths=tracked,
        runner=runner,
    )


def pin_candidate_commit(
    lease: WorkspaceLease, *, candidate_commit: str, tracked_paths: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[CandidateGitIdentity, WorkspaceLease]:
    """Pin/reconcile only an explicitly recorded candidate, refusing changed bytes."""
    identity = inspect_candidate_commit(lease.repository_root, base_commit=lease.base_commit,
        candidate_commit=candidate_commit, tracked_paths=tracked_paths, runner=runner)
    current = qualify_ephemeral_git_worktree(lease.root, require_clean=False, runner=runner)
    if (current.git_common_dir != lease.git_common_dir or current.git_admin_dir != lease.git_admin_dir
            or current.head not in {lease.base_commit, candidate_commit}):
        raise CandidateGitRejected("CANDIDATE_WORKSPACE_MISMATCH")
    if current.head == candidate_commit:
        if not current.clean:
            raise CandidateGitRejected("CANDIDATE_BYTES_CHANGED")
        return identity, replace(lease, pinned_commit=candidate_commit)
    _git(lease.root, "add", "-A", "--", ".", runner=runner)
    if _git(lease.root, "write-tree", runner=runner) != identity.candidate_tree:
        raise CandidateGitRejected("CANDIDATE_BYTES_CHANGED")
    _git(lease.root, "checkout", "--detach", candidate_commit, runner=runner)
    pinned = qualify_ephemeral_git_worktree(lease.root, runner=runner)
    if pinned.head != candidate_commit or not pinned.clean:
        raise CandidateGitRejected("CANDIDATE_PIN_FAILED")
    return identity, replace(lease, pinned_commit=candidate_commit)


def inspect_candidate_commit(
    repository: str | Path,
    *,
    base_commit: str,
    candidate_commit: str,
    tracked_paths: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CandidateGitIdentity:
    """Recompute identity from immutable objects without reading a worktree."""

    root = Path(repository).resolve()
    tracked = _tracked_paths(tracked_paths)
    # One process for parents and tree: "commit <candidate> <parents...>" then the tree on its own line.
    header, _, tree = _git(
        root, "rev-list", "--parents", "-n", "1", "--format=%T", candidate_commit, runner=runner,
    ).partition("\n")
    if header.split() != ["commit", candidate_commit, base_commit]:
        raise CandidateGitRejected("CANDIDATE_PARENT_MISMATCH")
    raw = _git_bytes(
        root,
        "diff-tree", "--no-commit-id", "-r", "-z", "--raw", "--no-renames",
        "--no-ext-diff", "--no-textconv",
        base_commit, candidate_commit,
        runner=runner,
    )
    entries = _parse_raw_entries(raw)
    if not entries:
        raise CandidateGitRejected("NO_CANDIDATE_CHANGES")
    for entry in entries:
        if entry.old_mode not in _REGULAR_MODES or entry.new_mode not in _REGULAR_MODES:
            raise CandidateGitRejected("CANDIDATE_MODE_FORBIDDEN")
        if not _within_tracked(entry.path, tracked):
            raise CandidateGitRejected("CANDIDATE_PATH_OUTSIDE_SCOPE")
    entries = tuple(sorted(entries, key=lambda item: item.path.encode("utf-8")))
    changed_paths = tuple(entry.path for entry in entries)
    if len(set(changed_paths)) != len(changed_paths):
        raise CandidateGitRejected("CANDIDATE_PATH_DUPLICATED")
    patch = _git_bytes(
        root,
        "diff", "--full-index", "--no-renames", "--no-color",
        "--no-ext-diff", "--no-textconv",
        base_commit, candidate_commit, "--", *tracked,
        runner=runner,
    )
    return CandidateGitIdentity(
        candidate_commit=candidate_commit,
        candidate_tree=tree,
        diff_sha256=hashlib.sha256(patch).hexdigest(),
        changed_paths=changed_paths,
        changed_entries=entries,
    )


def land_candidate_ref(
    repository: str | Path,
    *,
    target_ref: str,
    base_commit: str,
    candidate_commit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Advance an existing target ref by compare-and-swap, with crash reconciliation."""

    root = Path(repository).resolve()
    current = _git(
        root, "show-ref", "--verify", "--hash", target_ref,
        runner=runner, ok=(0, 1, 128),
    )
    if not current:
        raise CandidateGitRejected("TARGET_REF_MISSING")
    if current == candidate_commit:
        return "ALREADY_LANDED"
    if current != base_commit:
        # Another commit may have followed a completed landing before its host
        # checkpoint. Observe ancestry; never rewind the newer target ref.
        if _git(root, 'merge-base', candidate_commit, current, runner=runner, ok=(0, 1)) == candidate_commit:
            return "ALREADY_LANDED"
        raise CandidateGitRejected("BASE_REF_DIVERGED")
    try:
        _git(root, "update-ref", target_ref, candidate_commit, base_commit, runner=runner)
    except CandidateGitRejected as exc:
        raise CandidateGitRejected("BASE_REF_DIVERGED") from exc
    return "LANDED"


def _parse_raw_entries(raw: bytes) -> tuple[ChangedEntry, ...]:
    parts = raw.split(b"\x00")
    if parts and not parts[-1]:
        parts.pop()
    if len(parts) % 2:
        raise CandidateGitRejected("CANDIDATE_RAW_DIFF_INVALID")
    entries: list[ChangedEntry] = []
    for index in range(0, len(parts), 2):
        try:
            header = parts[index].decode("ascii", errors="strict")
            path = parts[index + 1].decode("utf-8", errors="strict")
            fields = header.split()
            old_mode, new_mode = fields[0][1:], fields[1]
            status = fields[4]
        except (UnicodeDecodeError, IndexError) as exc:
            raise CandidateGitRejected("CANDIDATE_RAW_DIFF_INVALID") from exc
        if not status or status[0] not in {"A", "D", "M", "T"}:
            raise CandidateGitRejected("CANDIDATE_STATUS_INVALID")
        canonical = PurePosixPath(path).as_posix()
        if canonical != path or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise CandidateGitRejected("CANDIDATE_PATH_INVALID")
        entries.append(ChangedEntry(path, old_mode, new_mode, status[0]))
    return tuple(entries)


def _tracked_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if not paths or isinstance(paths, (str, bytes)):
        raise CandidateGitRejected("TRACKED_PATHS_INVALID")
    values = tuple(paths)
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise CandidateGitRejected("TRACKED_PATHS_NOT_CANONICAL")
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value or ":" in value:
            raise CandidateGitRejected("TRACKED_PATH_INVALID")
        path = PurePosixPath(value)
        if (
            value != path.as_posix()
            or value.startswith("/")
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(part.lower() == ".git" for part in path.parts)
        ):
            raise CandidateGitRejected("TRACKED_PATH_INVALID")
    return values


def _within_tracked(path: str, tracked: Sequence[str]) -> bool:
    return any(path == item or path.startswith(f"{item}/") for item in tracked)


def _git(
    cwd: Path,
    *args: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    ok: tuple[int, ...] = (0,),
    extra_env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> str:
    if extra_env is None and stdin is None:
        try:
            return _run_git(cwd, *args, runner=runner, ok=ok)
        except WorkspaceQualificationRejected as exc:
            raise CandidateGitRejected(str(exc)) from exc
    completed = runner(
        [host_executable("git", required=True), "-c", "core.hooksPath=", "-c",
         "interactive.diffFilter=", "-c", "core.pager=cat", "-c",
         "color.ui=false", "-C", str(cwd), *args],
        input=stdin,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=isolated_child_environment(extra_env),
    )
    if completed.returncode not in ok:
        raise CandidateGitRejected("Git candidate operation failed")
    return completed.stdout.strip()


def _git_bytes(
    cwd: Path,
    *args: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> bytes:
    completed = runner(
        [host_executable("git", required=True), "-c", "core.hooksPath=", "-c",
         "interactive.diffFilter=", "-c", "core.pager=cat", "-c",
         "color.ui=false", "-C", str(cwd), *args],
        capture_output=True,
        check=False,
        timeout=30,
        env=isolated_child_environment(),
    )
    if completed.returncode != 0:
        raise CandidateGitRejected("Git candidate inspection failed")
    output = completed.stdout
    return output if isinstance(output, bytes) else output.encode("utf-8")
