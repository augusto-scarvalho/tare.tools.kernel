"""Registration, adapter output and host delivery remain distinct facts."""
import json
import sqlite3

from compute_plane import hook_observations as observations


def test_registration_does_not_claim_execution_or_delivery(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("TARE_HOOK_STATE_DIR", str(tmp_path / "state"))
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "functions.apply_patch", "hooks": [
        {"type": "command", "command": "python3 /outside/codex-global.py"}]}]}}))
    status = observations.hook_host_status(root, registration_paths={"codex": config})
    host = status["hosts"]["codex"]
    assert host["registration"]["status"] == "DECLARED_UNVERIFIED"
    assert host["adapter_invocation"] == "NOT_OBSERVED"
    assert host["host_delivery"] == "UNATTESTED"
    assert not (tmp_path / "state").exists()


def test_manual_output_never_becomes_host_delivery_or_stores_opaque_ids(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("TARE_HOOK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TARE_HOOK_OBSERVATION_MODE", "manual")
    assert observations.record_hook_observation(root, host="claude", event="PreToolUse", tool="apply_patch",
        native_tool="functions.apply_patch", session_id="opaque-session-secret", agent_id="opaque-agent-secret",
        role="auditor", outcome="CONTEXT_RETURNED", output_bytes=1024, snapshot="abc", digest="def")
    status = observations.hook_host_status(root, registration_paths={})
    host = status["hosts"]["claude"]
    assert host["adapter_invocation"] == "OBSERVED_LOCALLY"
    assert host["context_output"] == "RETURNED_BY_ADAPTER"
    assert host["host_delivery"] == "UNATTESTED"
    assert host["observation_kinds"] == ["manual"]
    raw = json.dumps(status)
    assert "opaque-session-secret" not in raw and "opaque-agent-secret" not in raw
    assert host["observations"][0]["native_tool"] == "functions.apply_patch"
    monkeypatch.setenv("TARE_HOOK_OBSERVATION_MODE", "HOST_DELIVERED")
    assert observations.record_hook_observation(root, host="claude", event="PreToolUse", tool="Read", outcome="SILENT")
    assert observations.hook_host_status(root, registration_paths={})["hosts"]["claude"]["host_delivery"] == "UNATTESTED"


def test_store_and_doctor_output_are_bounded_and_context_eviction_only_reemits(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("TARE_HOOK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TARE_HOOK_OBSERVATION_MODE", "fixture")
    for i in range(observations.MAX_ROWS + 4):
        assert observations.record_hook_observation(root, host="codex", event="PreToolUse", tool="Read",
            session_id=str(i), outcome="SILENT")
        observations.write_context_receipt(root, str(i), {"digest": str(i)})
    status = observations.hook_host_status(root, registration_paths={})
    assert status["retained_records"] == observations.MAX_ROWS
    assert len(status["hosts"]["codex"]["observations"]) == 8
    assert observations.read_context_receipt(root, "0") == {}
    assert observations.read_context_receipt(root, str(observations.MAX_ROWS + 3))
    with sqlite3.connect(observations._path(root)) as connection:
        assert connection.execute("SELECT count(*) FROM contexts").fetchone()[0] == observations.MAX_ROWS


def test_candidate_state_and_locked_diagnostics_fail_open(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("TARE_HOOK_STATE_DIR", str(root / "state"))
    assert not observations.record_hook_observation(root, host="codex", event="PreToolUse", tool="Read", outcome="SILENT")
    assert observations.hook_host_status(root, registration_paths={})["status"] == "UNAVAILABLE"
    assert not (root / "state").exists()
    monkeypatch.setenv("TARE_HOOK_STATE_DIR", str(tmp_path / "external"))
    assert observations.record_hook_observation(root, host="codex", event="PreToolUse", tool="Read", outcome="SILENT")
    with sqlite3.connect(observations._path(root)) as connection:
        connection.execute("BEGIN EXCLUSIVE")
        assert not observations.record_hook_observation(root, host="codex", event="PreToolUse", tool="Read", outcome="SILENT", timeout_ms=0)
        assert observations.hook_host_status(root, timeout_ms=0, registration_paths={})["status"] == "UNAVAILABLE"


def test_explicit_passive_inspection_ignores_tool_stdout_and_other_roots(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("TARE_HOOK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(observations.Path, "home", lambda: tmp_path)
    transcript = tmp_path / ".codex/sessions/current.jsonl"
    transcript.parent.mkdir(parents=True)
    report = {"schema": "tare.tools/specgraph-hook-context/1", "root": str(root), "snapshot": "snapshot-1",
              "event": "pre", "role": "implementer", "source": "not retained private content"}
    def row(role, report):
        return {"type": "response_item", "timestamp": "2026-09-08T12:00:00Z", "payload": {
            "type": "message", "role": role, "content": [{"text": json.dumps([report])}]}}
    transcript.write_text("\n".join(json.dumps(value) for value in [row("assistant", report),
        row("developer", report | {"root": str(tmp_path / "other")})]) + "\n")
    assert observations.observe_codex_delivery(root, transcript)["status"] == "NOT_OBSERVED"
    with transcript.open("a") as stream:
        stream.write(json.dumps(row("developer", report)) + "\n")
    result = observations.observe_codex_delivery(root, transcript)
    assert result["status"] == "OBSERVED_IN_HOST_TRANSCRIPT"
    assert result["snapshot"] == "snapshot-1"
    assert "not retained private content" not in json.dumps(result)
    status = observations.hook_host_status(root, registration_paths={})
    assert status["hosts"]["codex"]["host_delivery"] == "OBSERVED_IN_HOST_TRANSCRIPT"
    assert status["hosts"]["codex"]["adapter_invocation"] == "NOT_OBSERVED"
    assert status["hosts"]["claude"]["host_delivery"] == "UNATTESTED"


def test_copied_observation_database_is_not_attributed_to_another_root(tmp_path, monkeypatch):
    import shutil
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("TARE_HOOK_STATE_DIR", str(tmp_path / "state"))
    assert observations.record_hook_observation(first, host="codex", event="PreToolUse", tool="Read", outcome="CONTEXT_RETURNED")
    shutil.copyfile(observations._path(first), observations._path(second))
    status = observations.hook_host_status(second, registration_paths={})
    assert status["status"] == "UNAVAILABLE"
    assert status["hosts"]["codex"]["host_delivery"] == "UNATTESTED"
    assert not observations.record_hook_observation(second, host="codex", event="PreToolUse", tool="Read", outcome="SILENT")
