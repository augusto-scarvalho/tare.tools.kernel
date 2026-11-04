from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from compute_plane.workspace import (
    WorkspaceLease,
    WorkspaceQualificationRejected,
    cleanup_linked_worktree,
    isolated_child_environment,
    provision_linked_worktree,
    qualify_ephemeral_git_worktree,
    reconcile_linked_worktree,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_clean_linked_worktree_is_qualified_without_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    worktree = tmp_path / "worktree"
    source.mkdir()
    _git("init", cwd=source)
    _git("config", "user.email", "test@example.invalid", cwd=source)
    _git("config", "user.name", "Test", cwd=source)
    (source / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git("add", "tracked.txt", cwd=source)
    _git("commit", "-m", "baseline", cwd=source)
    _git("worktree", "add", "--detach", str(worktree), "HEAD", cwd=source)

    qualification = qualify_ephemeral_git_worktree(worktree)

    assert qualification.root == worktree.resolve()
    assert qualification.linked_worktree is True
    assert qualification.clean is True
    assert len(qualification.head) == 40
    assert qualification.git_common_dir == (source / ".git").resolve()


def test_plain_directory_and_dirty_worktree_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceQualificationRejected, match="linked"):
        qualify_ephemeral_git_worktree(tmp_path)

    source = tmp_path / "source"
    worktree = tmp_path / "worktree"
    source.mkdir()
    _git("init", cwd=source)
    _git("config", "user.email", "test@example.invalid", cwd=source)
    _git("config", "user.name", "Test", cwd=source)
    (source / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git("add", "tracked.txt", cwd=source)
    _git("commit", "-m", "baseline", cwd=source)
    _git("worktree", "add", "--detach", str(worktree), "HEAD", cwd=source)
    (worktree / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(WorkspaceQualificationRejected, match="clean"):
        qualify_ephemeral_git_worktree(worktree)


def _repository(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    _git("init", cwd=source)
    _git("config", "user.email", "test@example.invalid", cwd=source)
    _git("config", "user.name", "Test", cwd=source)
    (source / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git("add", "tracked.txt", cwd=source)
    _git("commit", "-m", "baseline", cwd=source)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return source, head


def test_provision_reconcile_and_exact_cleanup(tmp_path: Path) -> None:
    source, head = _repository(tmp_path)
    destination = tmp_path / "strict-worktree"
    lease = provision_linked_worktree(
        source,
        destination,
        base_commit=head,
        worktree_id="worktree-12345678",
    )

    assert isinstance(lease, WorkspaceLease)
    assert lease.root == destination.resolve()
    assert lease.base_commit == head
    assert reconcile_linked_worktree(
        source,
        destination,
        base_commit=head,
        worktree_id="worktree-12345678",
    ) == lease

    cleanup_linked_worktree(lease)
    assert not destination.exists()
    assert reconcile_linked_worktree(
        source,
        destination,
        base_commit=head,
        worktree_id="worktree-12345678",
    ) is None


def test_provision_rejects_existing_destination_and_missing_base_without_mutation(
    tmp_path: Path,
) -> None:
    source, head = _repository(tmp_path)
    destination = tmp_path / "occupied"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(WorkspaceQualificationRejected, match="already exists"):
        provision_linked_worktree(
            source,
            destination,
            base_commit=head,
            worktree_id="worktree-12345678",
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    with pytest.raises(WorkspaceQualificationRejected, match="BASE_COMMIT_UNAVAILABLE"):
        provision_linked_worktree(
            source,
            tmp_path / "missing-base",
            base_commit="f" * 40,
            worktree_id="worktree-12345678",
        )
    assert not (tmp_path / "missing-base").exists()


def test_cleanup_rejects_another_worktree_identity(tmp_path: Path) -> None:
    source, head = _repository(tmp_path)
    first = provision_linked_worktree(
        source, tmp_path / "first", base_commit=head, worktree_id="worktree-first1"
    )
    second = provision_linked_worktree(
        source, tmp_path / "second", base_commit=head, worktree_id="worktree-second"
    )

    forged = replace(first, root=second.root)
    with pytest.raises(WorkspaceQualificationRejected, match="identity mismatch"):
        cleanup_linked_worktree(forged)
    assert first.root.exists()
    assert second.root.exists()
    cleanup_linked_worktree(first)
    cleanup_linked_worktree(second)


def test_platform_data_directories_are_preserved_without_provider_secrets(monkeypatch) -> None:
    # Godot falls back to the project folder for its editor cache without APPDATA.
    expected = {"APPDATA": "C:/user/Roaming", "LOCALAPPDATA": "C:/user/Local",
                "USERPROFILE": "C:/user"}
    for key, value in expected.items():
        monkeypatch.setenv(key, value)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TARE_TEST_SECRET"):
        monkeypatch.setenv(key, "must-not-leak")
    child = isolated_child_environment()
    assert all(child[key] == value for key, value in expected.items())
    assert not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TARE_TEST_SECRET"} & child.keys()


def test_git_hooks_and_host_secrets_are_not_inherited(tmp_path: Path, monkeypatch) -> None:
    source, head = _repository(tmp_path)
    hook_marker = tmp_path / "hook-ran"
    hooks = source / ".git" / "hooks"
    hook = hooks / "post-checkout"
    hook.write_text(f"#!/bin/sh\necho ran > '{hook_marker.as_posix()}'\n", encoding="utf-8")
    hook.chmod(0o755)
    monkeypatch.setenv("TARE_TEST_SECRET", "must-not-leak")
    seen_envs: list[dict[str, str]] = []

    def recording_runner(*args, **kwargs):
        seen_envs.append(dict(kwargs["env"]))
        return subprocess.run(*args, **kwargs)

    lease = provision_linked_worktree(
        source,
        tmp_path / "isolated",
        base_commit=head,
        worktree_id="worktree-isolate",
        runner=recording_runner,
    )
    assert not hook_marker.exists()
    assert seen_envs
    assert all("TARE_TEST_SECRET" not in child for child in seen_envs)
    assert "TARE_TEST_SECRET" not in isolated_child_environment()
    cleanup_linked_worktree(lease, runner=recording_runner)


@pytest.mark.skipif(os.name != "nt", reason="Windows realpath keeps a \\\\?\\ prefix only on Windows")
def test_a_prefixed_resolution_still_hands_git_a_plain_path(tmp_path: Path, monkeypatch) -> None:
    """Windows' realpath may keep `\\\\?\\` on a path that does not exist yet; Git then failed with
    "could not create leading directories of '//?/C:/...': Invalid argument"."""
    source, head = _repository(tmp_path)
    destination = tmp_path / "prefixed"
    resolve = Path.resolve
    target = resolve(destination)
    def prefixed(self, strict=False):
        resolved = resolve(self, strict)
        return Path("\\\\?\\" + str(resolved)) if resolved == target else resolved
    monkeypatch.setattr(Path, "resolve", prefixed)
    calls: list[list[str]] = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.run(argv, **kwargs)
    lease = provision_linked_worktree(source, destination, base_commit=head, worktree_id="worktree-prefixed",
                                      runner=runner)
    assert lease.root == resolve(destination) and not str(lease.root).startswith("\\\\?\\")
    assert not [arg for argv in calls for arg in argv if arg.startswith("\\\\?\\")]
