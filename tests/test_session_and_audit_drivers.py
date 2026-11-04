import json
import sys
from uuid import uuid4

import pytest

from compute_plane.audit_driver import ClaudeAuditDriver
from compute_plane.session_hooks import adapt_session, render_session


@pytest.mark.parametrize("vendor", ["claude", "codex", "agy"])
def test_session_adapter_needs_root_and_stop_stays_neutral(vendor):
    payload = {"workspacePaths": ["/repo"]} if vendor == "agy" else {"cwd": "/repo", "hook_event_name": "Stop"}
    event = adapt_session(vendor, payload, event_hint="stop" if vendor == "agy" else None)
    assert render_session(event, "not a stop gate") == {}
    with pytest.raises(ValueError):
        adapt_session(vendor, {})


@pytest.mark.parametrize('vendor', ['claude', 'codex', 'agy'])
@pytest.mark.parametrize('phase', ['start', 'stop', 'end'])
def test_session_codec_phases_resume_and_missing_id(vendor, phase):
    payload = {'workspacePaths': ['/repo']} if vendor == 'agy' else {
        'cwd': '/repo', 'hook_event_name': {'start': 'SessionStart', 'stop': 'Stop', 'end': 'SessionEnd'}[phase]}
    hint = phase if vendor == 'agy' else None
    event = adapt_session(vendor, payload, event_hint=hint)
    assert event.phase == phase and event.root == '/repo' and event.session_id is None and not event.resumed
    payload['source'] = 'resume'
    assert adapt_session(vendor, payload, event_hint=hint).resumed is True
    expected = {} if phase != 'start' else ({'injectSteps': [{'ephemeralMessage': 'ação'}]} if vendor == 'agy'
        else {'hookSpecificOutput': {'hookEventName': 'SessionStart', 'additionalContext': 'ação'}})
    assert render_session(event, 'ação') == expected


@pytest.mark.parametrize('root', [None, '', '  ', 123, []])
@pytest.mark.parametrize('vendor', ['claude', 'codex', 'agy'])
def test_session_codec_rejects_root_after_valid_phase(vendor, root):
    payload = {'workspacePaths': [root]} if vendor == 'agy' else {'cwd': root, 'hook_event_name': 'Stop'}
    with pytest.raises(ValueError, match='root'):
        adapt_session(vendor, payload, event_hint='stop' if vendor == 'agy' else None)


@pytest.mark.parametrize('roots', [None, [], ['/one', '/two'], '/repo'])
def test_session_codec_agy_requires_single_workspace(roots):
    with pytest.raises(ValueError, match='single_workspace'):
        adapt_session('agy', {'workspacePaths': roots}, event_hint='stop')


@pytest.mark.parametrize('vendor,payload,hint', [
    ('claude', {'cwd': '/repo', 'hook_event_name': []}, None),
    ('codex', {'cwd': '/repo', 'hook_event_name': {}}, None),
    ('gemini', {'cwd': '/repo', 'hook_event_name': 'Stop'}, None),
    ('', {'cwd': '/repo', 'hook_event_name': 'Stop'}, None),
    ([], {}, None), ('agy', {'workspacePaths': ['/repo']}, []),
    ('claude', {'cwd': '/repo', 'hook_event_name': 'Stop'}, 'stop'),
])
def test_session_codec_malformed_phase_has_typed_failure(vendor, payload, hint):
    with pytest.raises(ValueError):
        adapt_session(vendor, payload, event_hint=hint)


@pytest.mark.parametrize('session', ['', '   ', '\t\n', 1, [], 'x' * 257])
def test_session_codec_rejects_blank_or_invalid_session(session):
    with pytest.raises(ValueError, match='invalid_session_id'):
        adapt_session('claude', {'cwd': '/repo', 'hook_event_name': 'Stop', 'session_id': session})


@pytest.mark.parametrize('session', ['x' * 256, 'opaque/id', '../opaque', 'ação'])
def test_session_codec_preserves_opaque_session(session):
    assert adapt_session('claude', {'cwd': '/repo', 'hook_event_name': 'Stop', 'session_id': session}).session_id == session


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), 101, True, "1"])
def test_execution_requires_finite_cost(budget):
    driver = ClaudeAuditDriver([sys.executable])
    with pytest.raises(ValueError, match="bounded_cost"):
        driver.build_command(model="claude-test", session=str(uuid4()), resume=False, schema={}, max_usd=budget)


@pytest.mark.parametrize("resume", [False, True])
def test_monetary_cap_is_omitted_unless_selected(resume):
    driver = ClaudeAuditDriver([sys.executable])
    session = str(uuid4())
    kwargs = dict(model="claude-test", session=session, resume=resume, schema={})
    assert "--max-budget-usd" not in driver.build_command(**kwargs)
    unlimited = driver.build_packet_command(**kwargs, effort="xhigh")
    assert "--max-budget-usd" not in unlimited
    assert unlimited[-2:] == ["--resume" if resume else "--session-id", session]
    limited = driver.build_packet_command(**kwargs, effort="xhigh", max_usd=.75)
    assert limited[limited.index("--max-budget-usd") + 1] == "0.75"


def test_run_rejects_command_outside_validated_prefix(tmp_path):
    driver = ClaudeAuditDriver([sys.executable, 'approved_script.py'])
    with pytest.raises(ValueError, match='command_prefix_mismatch'):
        driver.run([sys.executable, '-c', 'print(12345)'], root=tmp_path, prompt=b'', timeout=5)


@pytest.mark.parametrize('flag', ['--tools', '--allowedTools', '--output-format'])
def test_prefix_cannot_inject_provider_flags(flag):
    with pytest.raises(ValueError, match='command_prefix_options'):
        ClaudeAuditDriver([sys.executable, flag, 'Bash'])


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX process-group cancellation')
def test_cancellation_kills_descendant_that_ignores_term(tmp_path):
    import os, signal, subprocess, time
    from pathlib import Path
    pidfile = tmp_path / 'grandchild.pid'
    grandchild = ('import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                 'open(sys.argv[1],"w").write(str(os.getpid())); time.sleep(30)')
    script = ('import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'
              + repr(grandchild) + ',sys.argv[1]]); time.sleep(30)')
    pid = None
    try:
        with pytest.raises(InterruptedError):
            ClaudeAuditDriver([sys.executable]).run([sys.executable, '-c', script, str(pidfile)],
                root=tmp_path, prompt=b'', timeout=10, cancelled=pidfile.exists)
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stat = Path('/proc') / str(pid) / 'stat'
            if not stat.exists() or stat.read_text().split(') ', 1)[1].startswith('Z'):
                break  # dead, possibly waiting for init to reap
            time.sleep(.02)
        else:
            pytest.fail('descendant survived cancellation')
    finally:
        if pid is None and pidfile.exists(): pid = int(pidfile.read_text())
        if pid is not None:
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass


def test_real_leaf_subprocess_receives_stdin_without_shell(tmp_path):
    driver = ClaudeAuditDriver([sys.executable])
    command = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"]
    code, raw = driver.run(command, root=tmp_path, prompt=b'{"fixture":true}', timeout=5)
    assert code == 0 and json.loads(raw) == {"fixture": True}


def test_transport_accepts_complete_context_with_finite_ceiling(tmp_path):
    driver = ClaudeAuditDriver([sys.executable])
    command = [sys.executable, '-c', 'import sys; print(len(sys.stdin.buffer.read()))']
    code, raw = driver.run(command, root=tmp_path, prompt=b'x' * 50000, timeout=5)
    assert code == 0 and int(raw) == 50000
    with pytest.raises(ValueError, match='execution_budget_invalid'):
        driver.run(command, root=tmp_path, prompt=b'x' * 64001, timeout=5)


@pytest.mark.parametrize("script,timeout,error", [
    ("import time; time.sleep(10)", .15, TimeoutError),
    ("import sys; sys.stdout.write('x'*2097153)", 5, ValueError),
    ("import sys; sys.stderr.write('x'*65537)", 5, ValueError),
])
def test_real_process_timeout_and_output_limits(tmp_path, script, timeout, error):
    driver = ClaudeAuditDriver([sys.executable])
    with pytest.raises(error):
        driver.run([sys.executable, "-c", script], root=tmp_path, prompt=b"test", timeout=timeout)
def test_unread_large_stdin_respects_deadline(tmp_path, monkeypatch):
    import sys
    import time
    import pytest
    from compute_plane.audit_driver import ClaudeAuditDriver
    from compute_plane import audit_driver
    driver = ClaudeAuditDriver([sys.executable])
    stopped = []
    terminate = audit_driver.terminate_process_tree
    def record_stop(child):
        stopped.append((time.monotonic(), child))
        terminate(child)
    monkeypatch.setattr(audit_driver, 'terminate_process_tree', record_stop)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        driver.run([sys.executable, "-c", "import time; time.sleep(2)"],
                   root=tmp_path, prompt=b"x" * 24000, timeout=.2)
    # Enforce the deadline before process-tree cleanup. Windows taskkill and
    # handle cleanup have their own finite bound and can take over a second.
    assert len(stopped) == 1 and stopped[0][0] - started < 1.5
    assert stopped[0][1].poll() is not None


def test_stream_fragmentation_duplicate_and_auxiliary_usage():
    from compute_plane.audit_driver import AuditStream
    seen = []
    parser = AuditStream("session", "primary", lambda draft, message: seen.append((draft, message)))
    event = {"type": "assistant", "session_id": "session", "message": {
        "id": "msg", "model": "primary", "content": [{"type": "tool_use", "name": "StructuredOutput",
                                                    "input": {"claim": "ação λ"}}]}}
    raw = (json.dumps(event, ensure_ascii=False) + "\n").encode()
    for byte in raw:
        parser.feed(bytes([byte]))
    parser.feed(raw)
    parser.feed(json.dumps({"type": "result", "session_id": "session", "subtype": "error_max_budget_usd",
                            "modelUsage": {"primary": {}, "auxiliary": {}}}).encode())
    parser.finish()
    assert seen == [({"claim": "ação λ"}, "msg")]
    assert parser.author == "primary" and parser.terminal["subtype"] == "error_max_budget_usd"
    event["message"]["content"][0]["input"] = {"claim": "different"}
    with pytest.raises(ValueError, match="conflicting_review"):
        parser.feed(json.dumps(event).encode() + b"\n")


def test_verbose_cli_heartbeats_do_not_exhaust_small_result_budget(tmp_path):
    from compute_plane.audit_driver import AuditStream
    parser = AuditStream('session', 'primary', lambda *args: None)
    heartbeat = json.dumps({'type': 'system', 'padding': 'x' * 300}) + '\n'
    terminal = json.dumps({'type': 'result', 'session_id': 'session', 'subtype': 'success'}) + '\n'
    script = 'import sys; sys.stdout.write(' + repr(heartbeat) + '*10000+' + repr(terminal) + ')'
    code, raw = ClaudeAuditDriver([sys.executable]).run([sys.executable, '-c', script],
        root=tmp_path, prompt=b'', timeout=10, on_output=parser.feed)
    parser.finish()
    assert code == 0 and len(raw) > 2 * 1024 * 1024
    assert parser.terminal['subtype'] == 'success'
    with pytest.raises(ValueError, match='provider_output_over_budget'):
        while True:
            parser.feed(heartbeat.encode() * 100)


@pytest.mark.parametrize("raw", [b"[]\n", b'{"type":"result","type":"assistant"}\n',
    b'{"type":"assistant","session_id":"other"}\n', b"\xff\n", b'{"broken":',
    b"x" * 131073], ids=["array", "duplicate", "wrong-session", "bad-utf8", "truncated", "over-budget"])
def test_invalid_provider_events_fail_closed(raw):
    from compute_plane.audit_driver import AuditStream
    parser = AuditStream("session", "primary", lambda *args: None)
    with pytest.raises((ValueError, UnicodeError)):
        parser.feed(raw)
        parser.finish()


def test_real_stream_captures_draft_before_timeout_and_cancels(tmp_path):
    from compute_plane.audit_driver import AuditStream
    seen = []
    parser = AuditStream("session", "primary", lambda draft, message: seen.append(draft))
    raw = json.dumps({"type": "assistant", "session_id": "session", "message": {
        "id": "msg", "model": "primary", "content": [{"type": "tool_use", "name": "StructuredOutput",
                                                    "input": {"fixture": True}}]}})
    script = "import time; print(" + repr(raw) + ",flush=True); time.sleep(10)"
    with pytest.raises(TimeoutError):
        ClaudeAuditDriver([sys.executable]).run([sys.executable, "-c", script],
            root=tmp_path, prompt=b"x" * 24000, timeout=.5, on_output=parser.feed)
    assert seen == [{"fixture": True}]
    with pytest.raises(InterruptedError):
        ClaudeAuditDriver([sys.executable]).run([sys.executable, "-c", "import time; time.sleep(10)"],
            root=tmp_path, prompt=b"hi", timeout=5, cancelled=lambda: True)


def test_packet_schema_dialect_translation_preserves_owner_document():
    from compute_plane.audit_driver import packet_wire_schema
    original = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "owner/1",
                "type": "object", "$defs": {"name": {"type": "string"}},
                "properties": {"name": {"$ref": "#/$defs/name"}}}
    wire = packet_wire_schema(original)
    assert "$schema" not in wire and "$defs" not in wire
    assert wire["properties"]["name"]["$ref"] == "#/definitions/name"
    assert original["$defs"]["name"] == wire["definitions"]["name"] and "$schema" in original


def test_startup_stderr_is_preserved_without_provider_events(tmp_path):
    captured = []
    code, raw = ClaudeAuditDriver([sys.executable]).run(
        [sys.executable, "-c", "import sys; sys.stderr.write('invalid schema'); sys.exit(1)"],
        root=tmp_path, prompt=b"", timeout=5, on_stderr=captured.append)
    assert code == 1 and raw == b"" and captured == [b"invalid schema"]
