from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from compute_plane.candidate_git import (
    CandidateGitRejected,
    inspect_candidate_commit,
    land_candidate_ref,
    materialize_candidate_commit,
)
from compute_plane.workspace import cleanup_linked_worktree, provision_linked_worktree


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git("init", cwd=repository)
    _git("config", "user.email", "test@example.invalid", cwd=repository)
    _git("config", "user.name", "Test", cwd=repository)
    (repository / "src").mkdir()
    (repository / "src" / "file.txt").write_text("base\n", encoding="utf-8")
    (repository / "outside.txt").write_text("outside\n", encoding="utf-8")
    _git("add", ".", cwd=repository)
    _git("commit", "-m", "base", cwd=repository)
    _git("branch", "-M", "main", cwd=repository)
    return repository, _git("rev-parse", "HEAD", cwd=repository)


def test_candidate_is_immutable_complete_and_pinned_through_gc(tmp_path: Path):
    repository, base = _repository(tmp_path)
    lease = provision_linked_worktree(
        repository,
        tmp_path / "worktree",
        base_commit=base,
        worktree_id="worktree-candidate",
    )
    (lease.root / "src" / "file.txt").write_text("changed\n", encoding="utf-8")
    (lease.root / "src" / "new.txt").write_text("new\n", encoding="utf-8")

    identity, pinned = materialize_candidate_commit(lease, tracked_paths=["src"])
    assert identity.changed_paths == ("src/file.txt", "src/new.txt")
    assert pinned.pinned_commit == identity.candidate_commit
    assert _git("status", "--porcelain", cwd=pinned.root) == ""
    assert inspect_candidate_commit(
        repository,
        base_commit=base,
        candidate_commit=identity.candidate_commit,
        tracked_paths=["src"],
    ) == identity

    _git("reflog", "expire", "--expire=now", "--all", cwd=repository)
    _git("gc", "--prune=now", cwd=repository)
    assert _git("cat-file", "-t", identity.candidate_commit, cwd=repository) == "commit"
    cleanup_linked_worktree(pinned)


def test_out_of_scope_change_is_rejected_before_candidate_pin(tmp_path: Path):
    repository, base = _repository(tmp_path)
    lease = provision_linked_worktree(
        repository,
        tmp_path / "worktree",
        base_commit=base,
        worktree_id="worktree-outside1",
    )
    with pytest.raises(CandidateGitRejected, match="TRACKED_PATH_INVALID"):
        materialize_candidate_commit(lease, tracked_paths=["../src"])
    (lease.root / "outside.txt").write_text("escaped\n", encoding="utf-8")

    with pytest.raises(CandidateGitRejected, match="OUTSIDE_SCOPE"):
        materialize_candidate_commit(lease, tracked_paths=["src"])
    assert _git("rev-parse", "HEAD", cwd=lease.root) == base
    _git("reset", "--hard", base, cwd=lease.root)
    cleanup_linked_worktree(lease)


def test_landing_is_compare_and_swap_and_reconciles_completed_cas(tmp_path: Path):
    repository, base = _repository(tmp_path)
    first = provision_linked_worktree(
        repository, tmp_path / "first", base_commit=base,
        worktree_id="worktree-first1",
    )
    second = provision_linked_worktree(
        repository, tmp_path / "second", base_commit=base,
        worktree_id="worktree-second",
    )
    (first.root / "src" / "file.txt").write_text("first\n", encoding="utf-8")
    (second.root / "src" / "file.txt").write_text("second\n", encoding="utf-8")
    first_identity, first = materialize_candidate_commit(first, tracked_paths=["src"])
    second_identity, second = materialize_candidate_commit(second, tracked_paths=["src"])

    assert land_candidate_ref(
        repository,
        target_ref="refs/heads/main",
        base_commit=base,
        candidate_commit=first_identity.candidate_commit,
    ) == "LANDED"
    assert land_candidate_ref(
        repository,
        target_ref="refs/heads/main",
        base_commit=base,
        candidate_commit=first_identity.candidate_commit,
    ) == "ALREADY_LANDED"
    with pytest.raises(CandidateGitRejected, match="BASE_REF_DIVERGED"):
        land_candidate_ref(
            repository,
            target_ref="refs/heads/main",
            base_commit=base,
            candidate_commit=second_identity.candidate_commit,
        )
    _git('commit', '--allow-empty', '-qm', 'following commit', cwd=repository)
    advanced = _git('rev-parse', 'HEAD', cwd=repository)
    assert land_candidate_ref(repository, target_ref='refs/heads/main', base_commit=base,
        candidate_commit=first_identity.candidate_commit) == 'ALREADY_LANDED'
    assert _git('rev-parse', 'HEAD', cwd=repository) == advanced
    cleanup_linked_worktree(first)
    cleanup_linked_worktree(second)


def test_missing_target_ref_never_creates_it(tmp_path: Path):
    repository, base = _repository(tmp_path)
    with pytest.raises(CandidateGitRejected, match="TARGET_REF_MISSING"):
        land_candidate_ref(
            repository,
            target_ref="refs/heads/missing",
            base_commit=base,
            candidate_commit=base,
        )
    assert subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", "refs/heads/missing"],
        cwd=repository,
        check=False,
    ).returncode == 1
