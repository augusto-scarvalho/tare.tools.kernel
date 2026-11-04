"""Tests for scoped instruction discovery helper over actual filesystem."""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

import pytest

from compute_plane.mediated_tools import MediatedTools
from compute_plane.native_instructions import (
    ScopedInstructions,
    MAX_FILE_BYTES,
    MAX_AGGREGATE_BYTES,
    MAX_UNIQUE_RECORDS,
    MAX_CHAIN_DEPTH,
)
from compute_plane.native_journal import io_path


def make_tools(repo_root: Path) -> MediatedTools:
    return MediatedTools({
        "role": "explorer",
        "repositories": {"repo": str(repo_root)},
        "specgraph_edit_context": False,
    })


def test_ordering_inheritance_and_sibling_isolation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    # Root AGENTS.md must be excluded
    (repo / "AGENTS.md").write_text("root instructions")

    a = repo / "a"
    a.mkdir()
    (a / "AGENTS.md").write_text("instruction a")

    b = a / "b"
    b.mkdir()
    (b / "AGENTS.md").write_text("instruction b")

    c = b / "c"
    c.mkdir()
    (c / "AGENTS.md").write_text("instruction c")
    (c / "module.py").write_text("# c module")

    sibling = a / "sibling"
    sibling.mkdir()
    (sibling / "AGENTS.md").write_text("instruction sibling")
    (sibling / "sib_module.py").write_text("# sib module")

    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    # Observe c/module.py: parent-to-child root-first ordering
    records = scoped.observe("repo", "a/b/c/module.py", directory=False)
    assert len(records) == 3

    assert records[0]["repository"] == "repo"
    assert records[0]["path"] == "a/AGENTS.md"
    assert records[0]["text"] == "instruction a"
    assert records[0]["sha256"] == hashlib.sha256(b"instruction a").hexdigest()

    assert records[1]["repository"] == "repo"
    assert records[1]["path"] == "a/b/AGENTS.md"
    assert records[1]["text"] == "instruction b"
    assert records[1]["sha256"] == hashlib.sha256(b"instruction b").hexdigest()

    assert records[2]["repository"] == "repo"
    assert records[2]["path"] == "a/b/c/AGENTS.md"
    assert records[2]["text"] == "instruction c"
    assert records[2]["sha256"] == hashlib.sha256(b"instruction c").hexdigest()

    # Verify root AGENTS.md and sibling instructions were NOT included
    observed_paths = [r["path"] for r in records]
    assert "AGENTS.md" not in observed_paths
    assert "a/sibling/AGENTS.md" not in observed_paths

    # Sibling isolation: observing sibling does not leak b or c
    records_sib = scoped.observe("repo", "a/sibling/sib_module.py", directory=False)
    assert len(records_sib) == 2
    assert [r["path"] for r in records_sib] == ["a/AGENTS.md", "a/sibling/AGENTS.md"]

    # Observe directory explicitly
    records_dir = scoped.observe("repo", "a/b", directory=True)
    assert [r["path"] for r in records_dir] == ["a/AGENTS.md", "a/b/AGENTS.md"]

    # Root directory observation excludes root AGENTS.md
    assert scoped.observe("repo", ".", directory=True) == []

    # Root-level file observation excludes root AGENTS.md
    (repo / "top.py").write_text("# top")
    assert scoped.observe("repo", "top.py", directory=False) == []


def test_missing_files_and_appearance_after_restore(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    a = repo / "a"
    a.mkdir()
    (a / "file.py").write_text("# a file")

    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    # File does not exist: returns absent record
    records = scoped.observe("repo", "a/file.py", directory=False)
    assert len(records) == 1
    assert records[0]["path"] == "a/AGENTS.md"
    assert records[0]["sha256"] is None
    assert records[0]["text"] is None

    # Cache now pins a/AGENTS.md as absent
    pinned = scoped.records()
    assert len(pinned) == 1
    assert pinned[0]["sha256"] is None

    # Now create the absent file on disk
    (a / "AGENTS.md").write_text("newly created")

    # Observe must refuse previously pinned absent file that now exists (no silent refresh)
    with pytest.raises(ValueError, match="instruction_pinned_row_changed"):
        scoped.observe("repo", "a/file.py", directory=False)

    # Restore must also refuse an absent record when file now exists on disk
    scoped2 = ScopedInstructions(tools.roots, tools.path)
    with pytest.raises(ValueError, match="absent_file_now_exists"):
        scoped2.restore([{"repository": "repo", "path": "a/AGENTS.md", "sha256": None, "text": None}])


def test_missing_parent_directories_permissible_for_new_file_edits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    # non_existent_dir and its subdirectories do not exist on disk
    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    records = scoped.observe("repo", "missing_parent/sub/new_file.py", directory=False)
    assert len(records) == 2
    assert records[0]["path"] == "missing_parent/AGENTS.md"
    assert records[0]["sha256"] is None
    assert records[0]["text"] is None
    assert records[1]["path"] == "missing_parent/sub/AGENTS.md"
    assert records[1]["sha256"] is None
    assert records[1]["text"] is None


def test_changed_deleted_and_present_empty_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    # 1. Present empty file
    empty_dir = repo / "empty"
    empty_dir.mkdir()
    (empty_dir / "AGENTS.md").write_bytes(b"")
    (empty_dir / "file.py").write_text("# empty")

    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    recs = scoped.observe("repo", "empty/file.py")
    assert len(recs) == 1
    assert recs[0]["path"] == "empty/AGENTS.md"
    assert recs[0]["text"] == ""
    assert recs[0]["sha256"] == hashlib.sha256(b"").hexdigest()

    # 2. Changed file
    mod_dir = repo / "mod"
    mod_dir.mkdir()
    (mod_dir / "AGENTS.md").write_text("v1")
    (mod_dir / "file.py").write_text("# mod")

    scoped.observe("repo", "mod/file.py")
    (mod_dir / "AGENTS.md").write_text("v2")

    with pytest.raises(ValueError, match="instruction_pinned_row_changed"):
        scoped.observe("repo", "mod/file.py")

    # 3. Deleted file
    del_dir = repo / "del"
    del_dir.mkdir()
    (del_dir / "AGENTS.md").write_text("to delete")
    (del_dir / "file.py").write_text("# del")

    scoped.observe("repo", "del/file.py")
    (del_dir / "AGENTS.md").unlink()

    with pytest.raises(ValueError, match="instruction_pinned_row_changed"):
        scoped.observe("repo", "del/file.py")


def test_invalid_utf8_oversize_and_aggregate_budgets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tools = make_tools(repo)

    # 1. Invalid UTF-8
    bad_dir = repo / "bad_utf8"
    bad_dir.mkdir()
    (bad_dir / "AGENTS.md").write_bytes(b"\xff\xfe\x00\x01")
    (bad_dir / "file.py").write_text("# bad")

    scoped = ScopedInstructions(tools.roots, tools.path)
    with pytest.raises(ValueError, match="instruction_file_not_utf8"):
        scoped.observe("repo", "bad_utf8/file.py")
    assert scoped.records() == []

    # 2. Single file oversize (> 16 KiB)
    big_dir = repo / "big"
    big_dir.mkdir()
    (big_dir / "AGENTS.md").write_bytes(b"a" * (MAX_FILE_BYTES + 1))
    (big_dir / "file.py").write_text("# big")

    with pytest.raises(ValueError, match="instruction_file_too_large"):
        scoped.observe("repo", "big/file.py")
    assert scoped.records() == []

    # File exactly 16 KiB (16384 bytes) is permitted
    (big_dir / "AGENTS.md").write_bytes(b"a" * MAX_FILE_BYTES)
    recs = scoped.observe("repo", "big/file.py")
    assert len(recs) == 1
    assert len(recs[0]["text"].encode("utf-8")) == MAX_FILE_BYTES

    # 3. Aggregate present budget (> 24000 bytes)
    p1 = repo / "p1"
    p1.mkdir()
    p2 = p1 / "p2"
    p2.mkdir()
    (p2 / "file.py").write_text("# p2")
    (p1 / "AGENTS.md").write_bytes(b"x" * 13000)
    (p2 / "AGENTS.md").write_bytes(b"y" * 12000)

    scoped_agg = ScopedInstructions(tools.roots, tools.path)
    # 13000 + 12000 = 25000 > 24000
    with pytest.raises(ValueError, match="instruction_aggregate_budget_exceeded"):
        scoped_agg.observe("repo", "p1/p2/file.py")
    assert scoped_agg.records() == []

    # Leave space for encoded record metadata within the 24000-byte budget
    (p2 / "AGENTS.md").write_bytes(b"y" * 10000)
    recs_agg = scoped_agg.observe("repo", "p1/p2/file.py")
    assert len(recs_agg) == 2


def test_validation_no_partial_cache_mutation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    d1 = repo / "valid_parent"
    d1.mkdir()
    (d1 / "AGENTS.md").write_text("valid parent instructions")

    d2 = d1 / "invalid_child"
    d2.mkdir()
    (d2 / "AGENTS.md").write_bytes(b"\x80\x81")  # invalid UTF-8
    (d2 / "file.py").write_text("# f")

    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    with pytest.raises(ValueError, match="instruction_file_not_utf8"):
        scoped.observe("repo", "valid_parent/invalid_child/file.py")
    # No partial cache mutation: valid_parent must not remain in cache
    assert scoped.records() == []

    # In restore: valid row followed by invalid row
    valid_sha = hashlib.sha256(b"valid parent instructions").hexdigest()
    valid_row = {"repository": "repo", "path": "valid_parent/AGENTS.md", "sha256": valid_sha, "text": "valid parent instructions"}
    invalid_row = {"repository": "repo", "path": "other/AGENTS.md", "sha256": "bad_sha", "text": "foo"}

    with pytest.raises(ValueError):
        scoped.restore([valid_row, invalid_row])
    assert scoped.records() == []


def test_traversal_windows_aliases_and_git_rejection(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    invalid_paths = [
        "../escape.py",
        "a/../escape.py",
        "a\\b.py",
        "C:/file.py",
        "a:stream.py",
        "a/con/file.py",
        "a/aux.txt/file.py",
        "a/nul/file.py",
        "a/com1/file.py",
        "a/prn/file.py",
        "a/foo./file.py",
        "a./file.py",
        "a/foo /file.py",
        "a /file.py",
        "a/GIT~1/file.py",
        "a/PROGRA~1/file.py",
        "a/.git/file.py",
        "a/.GIT/file.py",
        "a/.Git/file.py",
        "a/.git./file.py",
        "a/.git /file.py",
        "/absolute/file.py",
        "   leading_space.py",
    ]

    for p in invalid_paths:
        with pytest.raises(ValueError, match="invalid_repository_path"):
            scoped.observe("repo", p)


def test_symlinks_and_dangling_links(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    # Probe platform symlink support
    test_link = tmp_path / "probe_symlink"
    try:
        os.symlink(repo, test_link)
        test_link.unlink()
    except OSError:
        pytest.skip("Symlinks not permitted or supported on this host")

    real_dir = repo / "real"
    real_dir.mkdir()
    (real_dir / "AGENTS.md").write_text("real agents")
    (real_dir / "file.py").write_text("# real")

    link_dir = repo / "link_dir"
    os.symlink(real_dir, link_dir)

    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    # Observing through a symlink directory must refuse
    with pytest.raises(ValueError, match="repository_link_forbidden"):
        scoped.observe("repo", "link_dir/file.py")

    # Dangling symlink
    dangling = repo / "dangling_link"
    os.symlink(repo / "nonexistent", dangling)

    with pytest.raises(ValueError, match="repository_link_forbidden"):
        scoped.observe("repo", "dangling_link/file.py")


def test_fifo_safety(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo not available on this platform")

    # WSL's shared Windows volume cannot create FIFOs; this tiny fixture follows
    # the native attachment test's POSIX temporary-directory pattern.
    with tempfile.TemporaryDirectory(prefix="tare-instructions-posix-", dir="/tmp") as directory:
        repo = Path(directory)
        fifo_dir = repo / "fifo_dir"
        fifo_dir.mkdir()
        os.mkfifo(fifo_dir / "AGENTS.md")
        (fifo_dir / "file.py").write_text("# fifo")
        tools = make_tools(repo)
        scoped = ScopedInstructions(tools.roots, tools.path)
        with pytest.raises(ValueError, match="regular_file_required"):
            scoped.observe("repo", "fifo_dir/file.py")


def test_no_unrelated_recursive_walk(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    unrelated = repo / "unrelated"
    unrelated.mkdir()
    (unrelated / "deep").mkdir()
    (unrelated / "deep" / "AGENTS.md").write_text("unrelated")

    target = repo / "target"
    target.mkdir()
    (target / "AGENTS.md").write_text("target instructions")
    (target / "file.py").write_text("# target")

    def fail_walk(*args, **kwargs):
        pytest.fail("os.walk must not be called during scoped discovery")

    monkeypatch.setattr(os, "walk", fail_walk)

    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    records = scoped.observe("repo", "target/file.py")
    assert len(records) == 1
    assert records[0]["path"] == "target/AGENTS.md"

    # Unrelated instructions never touched or cached
    all_cached = [r["path"] for r in scoped.records()]
    assert "unrelated/deep/AGENTS.md" not in all_cached


def test_deterministic_roundtrip_restore(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    a = repo / "a"
    a.mkdir()
    (a / "AGENTS.md").write_text("a instructions")

    b = a / "b"
    b.mkdir()
    # b/AGENTS.md absent
    (b / "file.py").write_text("# b")

    tools = make_tools(repo)
    scoped1 = ScopedInstructions(tools.roots, tools.path)

    scoped1.observe("repo", "a/b/file.py")
    rows1 = scoped1.records()
    assert len(rows1) == 2

    # Deterministic roundtrip via JSON serialization
    serialized = json.dumps(rows1, sort_keys=True)
    deserialized = json.loads(serialized)

    scoped2 = ScopedInstructions(tools.roots, tools.path)
    scoped2.restore(deserialized)
    assert scoped2.records() == rows1

    # Restore refuses contradictory snapshots
    contradictory = copy.deepcopy(deserialized)
    contradictory[0]["text"] = "tampered text"
    contradictory[0]["sha256"] = hashlib.sha256(b"tampered text").hexdigest()

    scoped3 = ScopedInstructions(tools.roots, tools.path)
    with pytest.raises(ValueError, match="instruction_source_changed"):
        scoped3.restore(contradictory)


def test_unique_records_and_chain_depth_limits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    # Chain depth > 64
    deep_parts = [f"d{i}" for i in range(MAX_CHAIN_DEPTH + 2)]
    deep_path = "/".join(deep_parts) + "/file.py"
    with pytest.raises(ValueError, match="instruction_chain_depth_exceeded"):
        scoped.observe("repo", deep_path)

    # Unique records > 256
    rows = []
    for i in range(MAX_UNIQUE_RECORDS + 1):
        rows.append({
            "repository": "repo",
            "path": f"dir_{i}/AGENTS.md",
            "sha256": None,
            "text": None,
        })
    with pytest.raises(ValueError, match="instruction_unique_records_limit_exceeded"):
        scoped.restore(rows)


def test_restore_validation_edge_cases(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tools = make_tools(repo)
    scoped = ScopedInstructions(tools.roots, tools.path)

    # Root AGENTS.md in restore is forbidden
    with pytest.raises(ValueError, match="invalid_instruction_path"):
        scoped.restore([{"repository": "repo", "path": "AGENTS.md", "sha256": None, "text": None}])

    # Missing field
    with pytest.raises(ValueError, match="invalid_record_fields"):
        scoped.restore([{"repository": "repo", "path": "a/AGENTS.md"}])

    # Extra field
    with pytest.raises(ValueError, match="invalid_record_fields"):
        scoped.restore([{"repository": "repo", "path": "a/AGENTS.md", "sha256": None, "text": None, "extra": 1}])

    # Sha256 mismatch with text
    with pytest.raises(ValueError, match="sha256_text_mismatch"):
        scoped.restore([{
            "repository": "repo",
            "path": "a/AGENTS.md",
            "sha256": "0" * 64,
            "text": "not matching zeros",
        }])

    # Duplicate contradictory rows in same restore call
    dup1 = {"repository": "repo", "path": "a/AGENTS.md", "sha256": None, "text": None}
    dup2 = {"repository": "repo", "path": "a/AGENTS.md", "sha256": "1" * 64, "text": "foo"}
    with pytest.raises(ValueError, match="sha256_text_mismatch"):
        scoped.restore([dup1, dup2])



def test_encoded_budget_counts_escaping_and_keeps_prior_cache(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    for name in ('a','b'):
        (repo/name).mkdir();(repo/name/'AGENTS.md').write_bytes(b'\n'*10000)
    tools=make_tools(repo);scoped=ScopedInstructions(tools.roots,tools.path)
    scoped.observe('repo','a/file.py');before=scoped.records()
    with pytest.raises(ValueError,match='aggregate_budget'):
        scoped.observe('repo','b/file.py')
    assert scoped.records()==before


def test_opened_file_identity_is_checked_before_read(tmp_path,monkeypatch):
    from compute_plane import native_instructions
    repo=tmp_path/'repo';(repo/'src').mkdir(parents=True)
    source=repo/'src/AGENTS.md';source.write_bytes(b'original')
    replacement=repo/'replacement';replacement.write_bytes(b'replaced')
    real_open=os.open
    def swapped(path,flags,*args,**kwargs):
        if Path(path).name=='AGENTS.md': os.replace(replacement,source)
        return real_open(path,flags,*args,**kwargs)
    tools=make_tools(repo);scoped=ScopedInstructions(tools.roots,tools.path)
    monkeypatch.setattr(native_instructions.os,'open',swapped)
    with pytest.raises(ValueError,match='changed_during_open'):
        scoped.observe('repo','src/file.py')
    assert scoped.records()==[]
