"""Git process budgets: each process costs ~30-50 ms on Windows, and these paths run several times per Work."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from compute_plane import audit_driver, audit_sources
from compute_plane.candidate_git import CandidateGitRejected, inspect_candidate_commit
from compute_plane.workspace import WorkspaceQualificationRejected, qualify_ephemeral_git_worktree


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "source"
    root.mkdir()
    _git("init", "-q", cwd=root)
    (root / "a.txt").write_bytes(b"base\n")
    _git("add", ".", cwd=root)
    _git("commit", "-qm", "base", cwd=root)
    base = _git("rev-parse", "HEAD", cwd=root)
    (root / "a.txt").write_bytes(b"changed\n")
    _git("commit", "-qam", "candidate", cwd=root)
    return root, base, _git("rev-parse", "HEAD", cwd=root)


def counting(calls: list[list[str]], real=subprocess.run):
    def run(argv, **kwargs):
        calls.append(argv)
        return real(argv, **kwargs)
    return run


def test_worktree_qualification_takes_two_processes_and_still_refuses_a_branch(repository, tmp_path):
    root, base, _ = repository
    detached, attached = tmp_path / "detached", tmp_path / "attached"
    _git("worktree", "add", "-q", "--detach", str(detached), base, cwd=root)
    _git("worktree", "add", "-q", "-b", "topic", str(attached), base, cwd=root)
    calls: list[list[str]] = []
    qualification = qualify_ephemeral_git_worktree(detached, runner=counting(calls))
    assert qualification.head == base and qualification.clean
    assert qualification.git_common_dir == (root / ".git").resolve()
    assert len(calls) == 2
    with pytest.raises(WorkspaceQualificationRejected, match="detached"):
        qualify_ephemeral_git_worktree(attached)


def test_candidate_inspection_takes_three_processes_and_still_checks_the_parent(repository):
    root, base, candidate = repository
    calls: list[list[str]] = []
    identity = inspect_candidate_commit(root, base_commit=base, candidate_commit=candidate,
                                        tracked_paths=["a.txt"], runner=counting(calls))
    assert identity.candidate_tree == _git("rev-parse", candidate + "^{tree}", cwd=root)
    assert identity.changed_paths == ("a.txt",) and len(calls) == 3
    with pytest.raises(CandidateGitRejected, match="CANDIDATE_PARENT_MISMATCH"):
        inspect_candidate_commit(root, base_commit=candidate, candidate_commit=candidate, tracked_paths=["a.txt"])


def test_revision_sources_read_size_from_the_tree_listing(repository, monkeypatch):
    root, base, candidate = repository
    calls: list[list[str]] = []
    monkeypatch.setattr(audit_sources.subprocess, "run", counting(calls))
    assert audit_sources.audit_source(root, candidate, "a.txt") == b"changed\n"
    assert len(calls) == 2  # ls-tree --long, cat-file blob
    assert audit_sources.audit_source(root, candidate, "missing.txt") is None
    with pytest.raises(ValueError, match="audit_source_over_budget"):
        audit_sources.audit_source(root, candidate, "a.txt", limit=4)
    assert audit_sources.audit_source(root, "INDEX", "a.txt") == b"changed\n"


def test_audit_candidate_identity_is_one_process_and_every_blob_one_more(repository, monkeypatch):
    root, _, candidate = repository
    tree = _git("rev-parse", candidate + "^{tree}", cwd=root)
    blob = _git("rev-parse", candidate + ":a.txt", cwd=root)
    import hashlib
    artifact = {"path": "a.txt", "git_blob": blob, "sha256": hashlib.sha256(b"changed\n").hexdigest(), "size_bytes": 8}
    calls: list[list[str]] = []
    monkeypatch.setattr(audit_driver.subprocess, "run", counting(calls))
    fingerprints = audit_driver.inspect_audit_candidate(root, {"commit": candidate, "tree": tree}, [artifact])
    assert fingerprints == {"a.txt": (artifact["sha256"], 8)}
    assert len(calls) == 4  # identity, status, ls-files, blobs
    with pytest.raises(ValueError, match="candidate_blob_mismatch"):
        audit_driver.inspect_audit_candidate(root, {"commit": candidate, "tree": tree}, [{**artifact, "git_blob": "0" * 40}])
    with pytest.raises(ValueError, match="candidate_mismatch_or_dirty"):
        audit_driver.inspect_audit_candidate(root, {"commit": candidate, "tree": "0" * 40}, [artifact])
