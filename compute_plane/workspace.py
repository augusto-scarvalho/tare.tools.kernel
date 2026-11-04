"""Exact provisioning and qualification for disposable linked Git workspaces."""

from __future__ import annotations

import os
import re
import subprocess
from .managed_process import host_executable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_WORKTREE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_GIT_CONFIG = (
    "-c", "core.hooksPath=",
    "-c", "interactive.diffFilter=",
    "-c", "core.pager=cat",
    "-c", "pager.status=false",
    "-c", "color.ui=false",
    "-c", f"core.autocrlf={'true' if os.name == 'nt' else 'false'}",
    # Windows: ephemeral worktrees sit under a deep custody path, so tracked
    # paths near MAX_PATH fail checkout without long-path support.
    *(("-c", "core.longpaths=true") if os.name == "nt" else ()),
)
_ENV_ALLOWLIST = (
    "APPDATA", "COMSPEC", "LANG", "LC_ALL", "LOCALAPPDATA", "PATH", "PATHEXT",
    "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "USERPROFILE",
    "WINDIR",
)


class WorkspaceQualificationRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceQualification:
    root: Path
    git_common_dir: Path
    git_admin_dir: Path
    head: str
    linked_worktree: bool
    clean: bool


@dataclass(frozen=True)
class WorkspaceLease:
    worktree_id: str
    repository_root: Path
    root: Path
    git_common_dir: Path
    git_admin_dir: Path
    base_commit: str
    pinned_commit: str | None = None


def isolated_child_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a child environment from an explicit OS allowlist, never a denylist."""

    result = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    result.update({
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    })
    if extra:
        for key, value in extra.items():
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise WorkspaceQualificationRejected("child environment override is invalid")
            result[key] = value
    return result


def qualify_ephemeral_git_worktree(
    workspace: str | Path,
    *,
    require_clean: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WorkspaceQualification:
    """Prove a clean, detached linked worktree without mutating it."""

    root = _resolved(workspace)
    if not root.is_dir():
        raise WorkspaceQualificationRejected("workspace must be an existing directory")
    git_file = root / ".git"
    if not git_file.is_file() or git_file.is_symlink():
        raise WorkspaceQualificationRejected("workspace must be a linked Git worktree")

    def git(*args: str) -> str:
        return _run_git(root, *args, runner=runner)

    # One process for the identity (each Git process costs ~30-50 ms on Windows): top-level, common and
    # administrative directories, HEAD, and HEAD's symbolic name ("HEAD" itself when detached).
    identity = git("rev-parse", "--show-toplevel", "--git-common-dir", "--absolute-git-dir", "HEAD",
                   "--symbolic-full-name", "HEAD").splitlines()
    if len(identity) != 5:
        raise WorkspaceQualificationRejected("workspace Git identity is ambiguous")
    top_raw, common_text, admin_text, head, symbolic = identity
    if _resolved(top_raw) != root:
        raise WorkspaceQualificationRejected("workspace root must equal Git top-level")
    common_raw = Path(common_text)
    common_dir = _resolved(common_raw) if common_raw.is_absolute() else _resolved(root / common_raw)
    admin_dir = _resolved(admin_text)
    if common_dir == root or common_dir.is_relative_to(root):
        raise WorkspaceQualificationRejected(
            "linked worktree common Git data must remain outside workspace"
        )
    if not admin_dir.is_relative_to(common_dir / "worktrees"):
        raise WorkspaceQualificationRejected("workspace Git identity is not linked")
    if not _COMMIT.fullmatch(head):
        raise WorkspaceQualificationRejected("workspace HEAD is not a full commit identity")
    if symbolic != "HEAD":
        raise WorkspaceQualificationRejected("workspace HEAD must be detached")
    dirty = bool(git("status", "--porcelain=v1", "--untracked-files=all"))
    if require_clean and dirty:
        raise WorkspaceQualificationRejected("workspace must be clean before strict execution")
    return WorkspaceQualification(root, common_dir, admin_dir, head, True, not dirty)


def provision_linked_worktree(
    repository: str | Path,
    destination: str | Path,
    *,
    base_commit: str,
    worktree_id: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WorkspaceLease:
    """Create exactly one detached worktree at an existing full commit."""

    if not isinstance(base_commit, str) or not _COMMIT.fullmatch(base_commit):
        raise WorkspaceQualificationRejected("base commit must be a full lowercase SHA-1")
    if not isinstance(worktree_id, str) or not _WORKTREE_ID.fullmatch(worktree_id):
        raise WorkspaceQualificationRejected("worktree identity is invalid")
    repository_root = _resolved(repository)
    if not repository_root.is_dir():
        raise WorkspaceQualificationRejected("repository must be an existing directory")
    top_level = _resolved(
        _run_git(repository_root, "rev-parse", "--show-toplevel", runner=runner)
    )
    if top_level != repository_root:
        raise WorkspaceQualificationRejected("repository must be its Git top-level")
    destination_root = _resolved(destination)
    if destination_root == repository_root:
        raise WorkspaceQualificationRejected("main checkout cannot be the strict workspace")
    if destination_root.exists():
        raise WorkspaceQualificationRejected("worktree destination already exists")
    if not destination_root.parent.is_dir():
        raise WorkspaceQualificationRejected("worktree destination parent must exist")
    _run_git(
        repository_root,
        "cat-file", "-e", f"{base_commit}^{{commit}}",
        runner=runner,
        unavailable_reason="BASE_COMMIT_UNAVAILABLE",
    )
    try:
        _run_git(
            repository_root,
            "worktree", "add", "--detach", "--no-checkout",
            str(destination_root), base_commit,
            runner=runner,
        )
        _run_git(destination_root, "checkout", "--detach", base_commit, runner=runner)
        qualification = qualify_ephemeral_git_worktree(destination_root, runner=runner)
        if qualification.head != base_commit:
            raise WorkspaceQualificationRejected("worktree revision differs from base commit")
    except Exception:
        if destination_root.exists():
            _run_git(
                repository_root,
                "worktree", "remove", "--force", str(destination_root),
                runner=runner,
                ok=(0, 1, 128),
            )
        raise
    return WorkspaceLease(
        worktree_id=worktree_id,
        repository_root=repository_root,
        root=qualification.root,
        git_common_dir=qualification.git_common_dir,
        git_admin_dir=qualification.git_admin_dir,
        base_commit=base_commit,
        pinned_commit=base_commit,
    )


def require_base_commit(
    repository: str | Path,
    base_commit: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    """Read-only proof that one full commit exists in an explicit repository."""

    if not isinstance(base_commit, str) or not _COMMIT.fullmatch(base_commit):
        raise WorkspaceQualificationRejected("BASE_COMMIT_UNAVAILABLE")
    root = _resolved(repository)
    if not root.is_dir():
        raise WorkspaceQualificationRejected("BASE_COMMIT_UNAVAILABLE")
    _run_git(
        root, "cat-file", "-e", f"{base_commit}^{{commit}}",
        runner=runner, unavailable_reason="BASE_COMMIT_UNAVAILABLE",
    )
    return root


def reconcile_linked_worktree(
    repository: str | Path,
    destination: str | Path,
    *,
    base_commit: str,
    worktree_id: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WorkspaceLease | None:
    """Recover only the exact intended path; absence is a clean pre-add crash."""

    destination_root = _resolved(destination)
    if not destination_root.exists():
        return None
    qualification = qualify_ephemeral_git_worktree(destination_root, runner=runner)
    repository_root = _resolved(repository)
    expected_common = Path(
        _run_git(repository_root, "rev-parse", "--git-common-dir", runner=runner)
    )
    expected_common = (
        _resolved(expected_common)
        if expected_common.is_absolute()
        else _resolved(repository_root / expected_common)
    )
    if qualification.git_common_dir != expected_common or qualification.head != base_commit:
        raise WorkspaceQualificationRejected("workspace intent does not match linked worktree")
    if not _WORKTREE_ID.fullmatch(worktree_id):
        raise WorkspaceQualificationRejected("worktree identity is invalid")
    return WorkspaceLease(
        worktree_id, repository_root, qualification.root,
        qualification.git_common_dir, qualification.git_admin_dir, base_commit,
        base_commit,
    )


def cleanup_linked_worktree(
    lease: WorkspaceLease,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Remove only the linked worktree whose exact administrative identity matches."""

    if not isinstance(lease, WorkspaceLease):
        raise WorkspaceQualificationRejected("cleanup requires an exact workspace lease")
    current = qualify_ephemeral_git_worktree(lease.root, runner=runner)
    if (
        current.root != lease.root
        or current.git_common_dir != lease.git_common_dir
        or current.git_admin_dir != lease.git_admin_dir
        or current.head != (lease.pinned_commit or lease.base_commit)
    ):
        raise WorkspaceQualificationRejected("workspace cleanup identity mismatch")
    _run_git(
        lease.repository_root,
        "worktree", "remove", "--force", str(lease.root),
        runner=runner,
    )
    if lease.root.exists():
        raise WorkspaceQualificationRejected("Git did not remove the exact worktree")


def _resolved(path: str | Path) -> Path:
    """Path.resolve(), plain even if something replaced realpath (see windows_paths): these paths reach Git as
    arguments, and Git cannot create directories under the long-path prefix."""
    from .windows_paths import plain
    return Path(plain(str(Path(path).resolve()), str(path)))


def _run_git(
    cwd: Path,
    *args: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    ok: tuple[int, ...] = (0,),
    unavailable_reason: str = "Git workspace verification failed",
) -> str:
    try:
        completed = runner(
            [host_executable("git", required=True), *_GIT_CONFIG, "-C", str(cwd), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
            env=isolated_child_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceQualificationRejected("Git workspace verification unavailable") from exc
    if completed.returncode not in ok:
        detail = (completed.stderr or "").strip().splitlines()[-1:] or [f"exit {completed.returncode}"]
        raise WorkspaceQualificationRejected(
            unavailable_reason if unavailable_reason.isupper()  # coded reasons stay exact
            else f"{unavailable_reason}: git {args[0]}: {detail[0][:300]}"
        )
    return completed.stdout.strip()
