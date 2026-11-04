from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from compute_plane.managed_work import (
    ManagedWorkRejected,
    ObservedWorkProgress,
    VerificationCommand,
    observe_work_progress,
    serialize_managed_aider_prompt,
)


def test_managed_prompt_preserves_stable_work_before_instruction() -> None:
    work = b'{"schema":"tare.tools/work-grounding/1"}'
    first = serialize_managed_aider_prompt("change one file", work)
    second = serialize_managed_aider_prompt("change another file", work)

    prefix = first.split("--OPERATOR_INSTRUCTION", 1)[0]
    assert second.startswith(prefix)
    assert work.decode() in first
    assert first.endswith("change one file")
    assert hashlib.sha256(work).hexdigest() in first


def test_managed_prompt_frames_multiple_groundings_without_merging() -> None:
    work = b'{"schema":"tare.tools/work-grounding/1"}'
    repo_a = b'{"repository":"repo-a"}'
    repo_b = b'{"repository":"repo-b"}'

    rendered = serialize_managed_aider_prompt(
        "change both", work, grounding_bytes_set=(repo_a, repo_b)
    )

    assert rendered.startswith("TARE_MANAGED_WORK/2\n")
    assert rendered.index(repo_a.decode()) < rendered.index(repo_b.decode())
    assert "--REPOSITORY_GROUNDING_0001" in rendered
    assert "--REPOSITORY_GROUNDING_0002" in rendered
    assert rendered.endswith("change both")


def test_managed_prompt_rejects_mixed_or_unbounded_groundings() -> None:
    with pytest.raises(ManagedWorkRejected, match="not both"):
        serialize_managed_aider_prompt(
            "change", b"{}", grounding_bytes=b"{}", grounding_bytes_set=(b"{}",)
        )
    with pytest.raises(ManagedWorkRejected, match="16"):
        serialize_managed_aider_prompt(
            "change", b"{}", grounding_bytes_set=tuple(b"{}" for _ in range(17))
        )


def test_progress_requires_exact_criterion_coverage(tmp_path) -> None:
    with pytest.raises(ManagedWorkRejected, match="exactly cover"):
        observe_work_progress(
            workspace=tmp_path,
            exit_criteria=("tests", "lint"),
            checks=(VerificationCommand("tests", ("verify",)),),
            tracked_paths=("src",),
        )


def test_progress_counts_only_passing_explicit_checks(tmp_path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("value = 1\n", encoding="utf-8")
    results = [
        subprocess.CompletedProcess([], 0, b"passed", b""),
        subprocess.CompletedProcess([], 1, b"", b"failed"),
    ]

    observed = observe_work_progress(
        workspace=tmp_path,
        exit_criteria=("tests", "lint"),
        checks=(
            VerificationCommand("tests", ("verify-tests",)),
            VerificationCommand("lint", ("verify-lint",)),
        ),
        tracked_paths=("src",),
        runner=lambda *_args, **_kwargs: results.pop(0),
    )

    assert observed.remaining_obligations == 1
    assert observed.verified_evidence_count == 1
    assert len(observed.state_sha256) == 64
    assert len(observed.evidence_sha256) == 64
    assert hashlib.sha256(observed.evidence_bytes).hexdigest() == observed.evidence_sha256
    evidence = json.loads(observed.evidence_bytes)
    assert evidence["schema"] == "tare.tools/verification-progress/1"
    assert [(receipt["criterion"], receipt["status"]) for receipt in evidence["receipts"]] == [
        ("tests", "PASS"), ("lint", "FAIL")]
    assert evidence["receipts"][0]["stdout_sha256"] == hashlib.sha256(b"passed").hexdigest()


def test_legacy_observed_progress_does_not_invent_receipt_bytes():
    legacy = ObservedWorkProgress("state", "evidence", 1, 0)
    assert legacy.evidence_bytes == b""


def test_state_identity_changes_without_counting_novelty_as_progress(tmp_path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("one", encoding="utf-8")
    check = VerificationCommand("tests", ("verify",))
    runner = lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", b"")

    first = observe_work_progress(
        workspace=tmp_path,
        exit_criteria=("tests",),
        checks=(check,),
        tracked_paths=("result.txt",),
        runner=runner,
    )
    target.write_text("two", encoding="utf-8")
    second = observe_work_progress(
        workspace=tmp_path,
        exit_criteria=("tests",),
        checks=(check,),
        tracked_paths=("result.txt",),
        runner=runner,
    )

    assert first.state_sha256 != second.state_sha256
    assert first.remaining_obligations == second.remaining_obligations == 1
    assert first.verified_evidence_count == second.verified_evidence_count == 0


def test_identical_verifier_is_executed_once_for_multiple_criteria(tmp_path) -> None:
    calls = 0

    def runner(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess([], 0, b"passed", b"")

    observed = observe_work_progress(
        workspace=tmp_path,
        exit_criteria=("unit tests", "integration tests"),
        checks=(
            VerificationCommand("unit tests", ("verify-all",)),
            VerificationCommand("integration tests", ("verify-all",)),
        ),
        tracked_paths=("result.txt",),
        runner=runner,
    )

    assert calls == 1
    assert observed.remaining_obligations == 0
    assert observed.verified_evidence_count == 2


def test_progress_rejects_workspace_escape(tmp_path) -> None:
    check = VerificationCommand("tests", ("verify",))
    runner = lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b"")
    with pytest.raises(ManagedWorkRejected, match="escapes"):
        observe_work_progress(
            workspace=tmp_path,
            exit_criteria=("tests",),
            checks=(check,),
            tracked_paths=(tmp_path.parent,),
            runner=runner,
        )
