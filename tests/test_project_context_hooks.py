"""Opt-in native adapter integration; fixtures are not host-delivery attestation."""
import json
from pathlib import Path
import subprocess

import pytest

indexer = pytest.importorskip("specgraph.indexer")
from compute_plane.hook_cli import dispatch
from compute_plane.tool_hooks import adapt_codex
from compute_plane import project_context_hooks


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "specgraph.yaml").write_text("schema_paths: []\napi_paths: []\nhook_context:\n  enabled: true\n  role: implementer\n  timeout_ms: 1000\n")
    (root / "src/example.py").write_text("def alpha(): return 1\n")
    monkeypatch.setenv("TARE_SPECGRAPH_STATE_DIR", str(tmp_path / "index-state"))
    monkeypatch.setenv("TARE_HOOK_STATE_DIR", str(tmp_path / "hook-state"))
    monkeypatch.setenv("TARE_HOOK_OBSERVATION_MODE", "fixture")
    indexer.index_repository(root)
    return root


def event(root, name, arguments, phase="PreToolUse", **extra):
    return json.dumps({"hook_event_name": phase, "cwd": str(root), "session_id": "fixture-session",
                       "tool_name": name, "tool_input": arguments, **extra}).encode()


def reports(rendered):
    return json.loads(rendered["hookSpecificOutput"]["additionalContext"])


def test_codex_absolute_patch_root_is_selected_from_declarative_owner(scoped, monkeypatch):
    root = scoped
    def forbidden(*args, **kwargs):
        raise AssertionError("scoped hook launched subprocess or legacy live scan")
    monkeypatch.setattr(subprocess, "run", forbidden)
    patch = f"*** Begin Patch\n*** Update File: {root}/src/example.py\n@@\n-old\n+new\n*** End Patch"
    raw = event(root.parent, "functions.apply_patch", patch)
    output = dispatch("codex", raw)
    report = reports(output)[0]
    assert report["root"] == str(root)
    assert report["files"] == ["src/example.py"]
    assert report["symbols"][0]["id"] == "src/example.py:alpha"
    assert dispatch("codex", raw) == {}  # Repeated mandatory context is silent.


def test_explicit_exec_workdir_and_query_use_snapshot_and_allow_original_tool(scoped):
    root = scoped
    raw = event(root.parent, "functions.exec_command", {"cmd": "rg alpha src", "workdir": str(root)})
    output = dispatch("codex", raw)
    report = reports(output)[0]
    assert report["root"] == str(root)
    assert report["symbols"][0]["id"] == "src/example.py:alpha"
    assert "permissionDecision" not in output["hookSpecificOutput"]
    assert dispatch("codex", event(root, "exec_command", {"cmd": "rg absent src"})) == {}


def test_unavailable_index_has_no_blocking_loop(scoped, monkeypatch):
    monkeypatch.setenv("TARE_SPECGRAPH_STATE_DIR", str(scoped.parent / "absent-state"))
    raw = event(scoped, "Bash", {"command": "rg alpha src"})
    assert dispatch("claude", raw) == {}
    assert dispatch("claude", raw) == {}


def test_post_edit_coalesces_and_next_hint_marks_observed_dirty(scoped, monkeypatch):
    root = scoped
    def forbidden(*args, **kwargs):
        raise AssertionError("post hook attempted live scan")
    monkeypatch.setattr(subprocess, "run", forbidden)
    pre = event(root, "Edit", {"file_path": str(root / "src/example.py")})
    first = reports(dispatch("claude", pre))[0]
    post = event(root, "Edit", {"file_path": str(root / "src/example.py")}, phase="PostToolUse")
    assert dispatch("claude", post) == {}
    assert dispatch("claude", post) == {}
    after = reports(dispatch("claude", pre))[0]
    assert after["digest"] != first["digest"]
    assert after["coverage"]["current_worktree"] == "DIRTY_OBSERVED"
    assert len(after["dirty_paths"]) == 1


def test_failed_native_tool_does_not_mark_dirty(scoped):
    root = scoped
    pre = event(root, "Edit", {"file_path": str(root / "src/example.py")})
    dispatch("claude", pre)
    post = event(root, "Edit", {"file_path": str(root / "src/example.py")},
                 phase="PostToolUse", tool_response={"isError": True, "error": "failed"})
    assert dispatch("claude", post) == {}
    assert dispatch("claude", pre) == {}


def test_multiple_roots_are_kept_separate_in_one_patch(scoped):
    root = scoped
    other = root.parent / "other"
    (other / "src").mkdir(parents=True)
    (other / "specgraph.yaml").write_bytes((root / "specgraph.yaml").read_bytes())
    (other / "src/example.py").write_text("def beta(): return 2\n")
    indexer.index_repository(other)
    patch = (f"*** Begin Patch\n*** Update File: {root}/src/example.py\n@@\n-a\n+b\n"
             f"*** Update File: {other}/src/example.py\n@@\n-a\n+b\n*** End Patch")
    payloads = reports(dispatch("codex", event(root.parent, "apply_patch", {"command": patch})))
    assert {r["root"] for r in payloads} == {str(root), str(other)}
    assert len({r["snapshot"] for r in payloads}) == 2


def test_namespace_normalization_keeps_explicit_workdir(scoped):
    call = adapt_codex(json.loads(event(scoped.parent, "functions.exec_command",
                                       {"cmd": "pwd", "workdir": str(scoped)})))
    assert call.name == "Bash"
    assert call.root == str(scoped)
    assert call.arguments["command"] == "pwd"


def test_post_records_bounded_root_event_without_indexing_or_source_scan(scoped, monkeypatch):
    root = scoped
    recorded = []
    def record(root, paths, **options):
        recorded.append((root, paths, options))
        return True
    def forbidden(*args, **kwargs):
        raise AssertionError("Post callback attempted indexing, process spawn or source scan")
    monkeypatch.setattr(project_context_hooks, "_record_changes", record)
    monkeypatch.setattr(indexer, "index_repository", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    import os
    monkeypatch.setattr(os, "scandir", forbidden)
    post = event(root.parent, "functions.apply_patch", {
        "command": f"*** Begin Patch\n*** Update File: {root}/src/example.py\n*** End Patch"},
        phase="PostToolUse", tool_use_id="call-1")
    assert dispatch("codex", post) == {}
    assert len(recorded) == 1
    selected_root, paths, options = recorded[0]
    assert selected_root == root
    assert paths == [str(root / "src/example.py")]
    assert options["source"] == "codex:PostToolUse"
    assert options["overflow"] is False
    assert options["event_id"].startswith("kernel:")
    assert len(options["event_id"]) <= 256


def test_event_retry_identity_is_namespaced_and_absent_ids_are_not_invented(scoped, monkeypatch):
    identifiers = []
    def record(root, paths, **options):
        identifiers.append(options["event_id"])
        return True
    monkeypatch.setattr(project_context_hooks, "_record_changes", record)
    raw = event(scoped, "Edit", {"file_path": str(scoped / "src/example.py")},
                phase="PostToolUse", tool_use_id="same-host-call")
    dispatch("claude", raw)
    dispatch("claude", raw)
    other = json.loads(raw)
    other["session_id"] = "another-session"
    dispatch("claude", json.dumps(other).encode())
    del other["tool_use_id"]
    dispatch("claude", json.dumps(other).encode())
    assert identifiers[0] == identifiers[1]
    assert identifiers[2] != identifiers[0]
    assert identifiers[3] is None


def test_large_patch_marks_event_overflow_instead_of_claiming_complete_delivery(scoped, monkeypatch):
    recorded = []
    def record(root, paths, **options):
        recorded.append((paths, options))
        return True
    monkeypatch.setattr(project_context_hooks, "_record_changes", record)
    patch = "*** Begin Patch\n" + "\n".join(
        f"*** Update File: {scoped}/src/file_{i}.py" for i in range(65)) + "\n*** End Patch"
    assert dispatch("codex", event(scoped.parent, "apply_patch", {"command": patch}, phase="PostToolUse")) == {}
    assert len(recorded) == 1
    assert len(recorded[0][0]) == 64
    assert recorded[0][1]["overflow"] is True


def test_failed_delivery_is_advisory_deduplicated_and_not_erased_by_new_snapshot(scoped, monkeypatch):
    monkeypatch.setattr(project_context_hooks, "_record_changes", lambda *args, **kwargs: False)
    post = event(scoped, "Edit", {"file_path": str(scoped / "src/example.py")}, phase="PostToolUse")
    assert dispatch("claude", post) == {}
    pre = event(scoped, "Read", {"file_path": str(scoped / "src/example.py")})
    first = reports(dispatch("claude", pre))[0]
    assert first["hook_delivery"].startswith("UNCONFIRMED")
    assert dispatch("claude", pre) == {}
    (scoped / "src/example.py").write_text("def beta(): return 2\n")
    indexer.index_repository(scoped)
    changed = reports(dispatch("claude", pre))[0]
    assert changed["hook_delivery"].startswith("UNCONFIRMED")
    assert changed["snapshot"] != first["snapshot"]


def test_successful_delivery_does_not_add_mutable_dirty_queue_to_session_receipt(scoped, monkeypatch):
    monkeypatch.setattr(project_context_hooks, "_record_changes", lambda *args, **kwargs: True)
    pre = event(scoped, "Read", {"file_path": str(scoped / "src/example.py")})
    dispatch("claude", pre)
    before = project_context_hooks._load(project_context_hooks._receipt(scoped, "fixture-session"))
    post = event(scoped, "Edit", {"file_path": str(scoped / "src/example.py")}, phase="PostToolUse")
    dispatch("claude", post)
    after = project_context_hooks._load(project_context_hooks._receipt(scoped, "fixture-session"))
    assert before == after
    assert not after.get("pending_paths")


def test_unowned_absolute_edit_is_not_recorded_against_unrelated_host_cwd(scoped, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("foreign path was assigned to cwd's journal")
    monkeypatch.setattr(project_context_hooks, "_record_changes", forbidden)
    outside = scoped.parent / "unowned.py"
    outside.write_text("# outside the configured owner\n")
    post = event(scoped, "Edit", {"file_path": str(outside)}, phase="PostToolUse")
    assert dispatch("claude", post) == {}


def test_durable_journal_ack_clears_dirty_even_when_snapshot_is_unchanged(scoped):
    from specgraph.lifecycle import lifecycle_status, reconcile_repository
    from specgraph.storage import RepositoryIndex
    root = scoped
    with RepositoryIndex(root) as index:
        initial = index.metadata()["content_hash"]
    post = event(root.parent, "functions.apply_patch", {
        "command": f"*** Begin Patch\n*** Update File: {root}/src/example.py\n*** End Patch"},
        phase="PostToolUse", tool_use_id="observed-unchanged-edit")
    assert dispatch("codex", post) == {}
    pending = lifecycle_status(root)
    assert pending["event_seq"] == 1
    assert pending["pending_paths"] == ["src/example.py"]
    pre = event(root, "Read", {"file_path": str(root / "src/example.py")})
    dirty = reports(dispatch("codex", pre))[0]
    assert dirty["coverage"]["current_worktree"] == "DIRTY_OBSERVED"
    reconciled = reconcile_repository(root)
    assert reconciled["content_hash"] == initial
    assert reconciled["reconciliation"]["acknowledged"] is True
    settled = lifecycle_status(root)
    assert settled["pending_count"] == 0
    assert settled["reconciled_seq"] == settled["event_seq"]
    clean = reports(dispatch("codex", pre))[0]
    assert clean["coverage"]["current_worktree"] != "DIRTY_OBSERVED"
    assert clean["digest"] != dirty["digest"]
    assert dispatch("codex", pre) == {}


def test_reconciliation_does_not_ack_concurrent_post_event_for_same_path(scoped, monkeypatch):
    from specgraph.lifecycle import lifecycle_status, reconcile_repository
    root = scoped
    source = root / "src/example.py"
    source.write_text("def beta(): return 2\n")
    post = event(root, "Edit", {"file_path": str(source)}, phase="PostToolUse", tool_use_id="before-cutoff")
    dispatch("claude", post)
    real_index = indexer.index_repository
    def concurrent_edit(*args, **kwargs):
        result = real_index(*args, **kwargs)
        source.write_text("def gamma(): return 3\n")
        racing = event(root, "Edit", {"file_path": str(source)}, phase="PostToolUse", tool_use_id="after-cutoff")
        dispatch("claude", racing)
        return result
    monkeypatch.setattr(indexer, "index_repository", concurrent_edit)
    first = reconcile_repository(root)
    assert first["reconciliation"]["events_through"] == 1
    state = lifecycle_status(root)
    assert state["event_seq"] == 2 and state["reconciled_seq"] == 1
    assert state["pending_paths"] == ["src/example.py"]
    pre = event(root, "Read", {"file_path": str(source)})
    assert reports(dispatch("claude", pre))[0]["coverage"]["current_worktree"] == "DIRTY_OBSERVED"
    monkeypatch.setattr(indexer, "index_repository", real_index)
    reconcile_repository(root)
    assert lifecycle_status(root)["pending_count"] == 0


def test_periodic_worker_recovers_edit_with_no_delivered_hook_event(scoped):
    from specgraph.lifecycle import lifecycle_status, watch_repository
    from specgraph.storage import RepositoryIndex
    root = scoped
    # A missing native callback must not imply that the repository is unchanged.
    (root / "src/example.py").write_text("def lost_event_edit(): return 7\n")
    status = lifecycle_status(root)
    assert status["status"] == "NOT_INITIALIZED"
    outcomes = list(watch_repository(root, max_cycles=1, interval_seconds=.1))
    assert outcomes[0]["status"] == "READY"
    assert outcomes[0]["reconciliation"]["reason"] == "periodic_sha256"
    with RepositoryIndex(root) as index:
        assert index.search("lost_event_edit")
        assert not index.search("alpha")
    observed = lifecycle_status(root)
    assert observed["event_seq"] == 0
    assert observed["event_coverage"] == "BEST_EFFORT"
    assert observed["last_observation_completed_at"]


def test_multi_root_delivery_cap_sets_overflow_on_observed_roots(scoped, monkeypatch):
    recorded = []
    def record(root, paths, **options):
        recorded.append((root, paths, options))
        return True
    monkeypatch.setattr(project_context_hooks, "_record_changes", record)
    roots = [scoped]
    for i in range(4):
        other = scoped.parent / f"other-{i}"
        (other / "src").mkdir(parents=True)
        (other / "specgraph.yaml").write_bytes((scoped / "specgraph.yaml").read_bytes())
        roots.append(other)
    patch = "*** Begin Patch\n" + "\n".join(
        f"*** Update File: {root}/src/example.py" for root in roots) + "\n*** End Patch"
    assert dispatch("codex", event(scoped.parent, "apply_patch", {"command": patch}, phase="PostToolUse")) == {}
    assert len(recorded) == 4
    assert all(options["overflow"] for _, _, options in recorded)
    assert {root for root, _, _ in recorded} <= set(roots)
    assert all(paths == [str(root / "src/example.py")] for root, paths, _ in recorded)


def test_legacy_session_pending_migrates_once_and_reconciliation_clears_it(scoped):
    from specgraph.lifecycle import lifecycle_status, reconcile_repository
    receipt = project_context_hooks._receipt(scoped, "fixture-session")
    project_context_hooks._save(receipt, {"pending_paths": [str(scoped / "src/example.py")], "delta_overflow": True})
    pre = event(scoped, "Read", {"file_path": str(scoped / "src/example.py")})
    report = reports(dispatch("claude", pre))[0]
    assert report["coverage"]["current_worktree"] == "DIRTY_OBSERVED"
    state = lifecycle_status(scoped)
    assert state["event_seq"] == 1 and state["overflow"] is True
    assert not project_context_hooks._load(receipt).get("pending_paths")
    assert dispatch("claude", pre) == {}
    assert lifecycle_status(scoped)["event_seq"] == 1
    reconcile_repository(scoped, reuse_snapshot=True)
    report = reports(dispatch("claude", pre))[0]
    assert report["coverage"]["current_worktree"] != "DIRTY_OBSERVED"
    assert lifecycle_status(scoped)["overflow"] is False


def test_context_isolated_by_host_role_subagent_and_resume(scoped):
    arguments = {"file_path": str(scoped / "src/example.py")}
    for host, extra in [("codex", {}), ("claude", {}), ("codex", {"agent_role": "explorer"}),
                        ("codex", {"agent_id": "child-a"}), ("codex", {"source": "resume"})]:
        raw = event(scoped, "Read", arguments, **extra)
        assert reports(dispatch(host, raw))[0]["root"] == str(scoped)
        assert dispatch(host, raw) == {}
    # Returning to the original identity uses its own receipt, not the last child.
    assert dispatch("codex", event(scoped, "Read", arguments)) == {}


def test_scoped_delegation_has_no_synchronous_network_or_process(scoped, monkeypatch):
    import socket
    import urllib.request
    import os
    from compute_plane import tool_hooks
    def forbidden(*args, **kwargs):
        raise AssertionError("scoped handler attempted network/process/source scan")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(tool_hooks, "local_models_up", forbidden)
    monkeypatch.delenv("TARE_LOCAL_MODELS_HEALTH_CACHE", raising=False)
    raw = event(scoped, "functions.spawn_agent", {"message": "Do not interpret this as a role"}, agent_role="planner")
    output = dispatch("codex", raw)
    assert "permissionDecision" not in output.get("hookSpecificOutput", {})
    assert reports(output)[0]["role"] == "planner"
    assert reports(output)[0]["event"] == "delegate"


def test_scoped_health_cache_is_explicit_fresh_and_external_while_legacy_probe_stays(scoped, monkeypatch):
    import time
    from compute_plane import tool_hooks
    cache = scoped.parent / "health.json"
    monkeypatch.setenv("TARE_LOCAL_MODELS_HEALTH_CACHE", str(cache))
    cache.write_text(json.dumps({"schema": "tare.tools/local-models-health/1", "status": "UP", "observed_at": time.time()}))
    raw = event(scoped, "spawn_agent", {"message": "explore"})
    assert dispatch("codex", raw)["hookSpecificOutput"]["permissionDecision"] == "deny"
    cache.write_text(json.dumps({"schema": "tare.tools/local-models-health/1", "status": "UP", "observed_at": time.time() - 60}))
    assert "permissionDecision" not in dispatch("codex", raw).get("hookSpecificOutput", {})
    inside = scoped / "health.json"
    inside.write_bytes(cache.read_bytes())
    monkeypatch.setenv("TARE_LOCAL_MODELS_HEALTH_CACHE", str(inside))
    assert tool_hooks.cached_local_models_up(scoped) is False
    probes = []
    monkeypatch.setattr(tool_hooks, "local_models_up", lambda: probes.append(True) or True)
    unscoped = event(scoped.parent, "spawn_agent", {"message": "explore"})
    assert dispatch("codex", unscoped)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert probes == [True]


def test_over_budget_output_does_not_mark_unreturned_context_as_delivered(scoped, monkeypatch):
    from specgraph import hook_context
    from compute_plane.hook_observations import hook_host_status
    size = [4500]
    previous = []
    def context(*args, **kwargs):
        previous.append(kwargs["previous_digest"])
        return {"status": "HIT", "digest": "same-selection", "snapshot": "abc", "body": "x" * size[0]}
    monkeypatch.setattr(hook_context, "context_hint", context)
    raw = event(scoped, "Read", {"file_path": str(scoped / "src/example.py")})
    assert dispatch("codex", raw) == {}
    status = hook_host_status(scoped, registration_paths={})["hosts"]["codex"]
    assert status["context_output"] == "NOT_OBSERVED"
    size[0] = 10
    assert dispatch("codex", raw)
    assert previous == [None, None]


def test_on_demand_policy_silences_automatic_context_but_keeps_search_and_post(scoped, monkeypatch):
    from specgraph import hook_context
    from specgraph.lifecycle import lifecycle_status
    from compute_plane.hook_observations import hook_host_status
    marker = scoped / "specgraph.yaml"
    marker.write_text(marker.read_text() + "  automatic_roles: []\n")
    pre = event(scoped, "Read", {"file_path": str(scoped / "src/example.py")})
    real_context = hook_context.context_hint
    def forbidden(*args, **kwargs):
        raise AssertionError("automatic context was queried despite on-demand policy")
    monkeypatch.setattr(hook_context, "context_hint", forbidden)
    assert dispatch("codex", pre) == {}
    state = hook_host_status(scoped, registration_paths={})["hosts"]["codex"]
    assert state["observations"][0]["reason"] == "ON_DEMAND_POLICY"
    post = event(scoped, "Edit", {"file_path": str(scoped / "src/example.py")}, phase="PostToolUse")
    assert dispatch("codex", post) == {}
    assert lifecycle_status(scoped)["pending_count"] == 1
    monkeypatch.setattr(hook_context, "context_hint", real_context)
    assert reports(dispatch("codex", event(scoped, "exec_command", {"cmd": "rg alpha src"})))[0]["status"] == "HIT"
    explicit = event(scoped, "Read", {"file_path": str(scoped / "src/example.py")}, symbol_id="src/example.py:alpha")
    assert reports(dispatch("codex", explicit))[0]["status"] == "HIT"
