"""Real child-process tests; no provider, SSH host or user workspace effects."""
import os
import sys
import time
from types import SimpleNamespace

import pytest

from compute_plane.terminal_sessions import TerminalSessions
from compute_plane import terminal_sessions


def start(host, tmp_path, code, *, owner='conversation-a', request_id='run-1', **kwargs):
    code = "import sys; sys.stdout.reconfigure(encoding='utf-8', newline='\\n'); sys.stderr.reconfigure(encoding='utf-8', newline='\\n'); " + code
    return host.start(owner=owner, request_id=request_id,
        argv=[sys.executable, '-I', '-u', '-c', code], cwd=tmp_path,
        env=dict(os.environ), **kwargs)


def until(host, handle, predicate, *, owner='conversation-a'):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = host.read(owner=owner, handle=handle)
        if predicate(result):
            return result
        time.sleep(.01)
    pytest.fail('terminal did not reach expected state: ' + str(result))


def test_live_utf8_input_eof_and_idempotent_effects(tmp_path):
    code = (
        'import sys,time; '
        'sys.stdout.buffer.write(b"ol\\xc3"); sys.stdout.flush(); time.sleep(.03); '
        'sys.stdout.buffer.write(b"\\xa1\\n"); sys.stdout.flush(); '
        'print("diagnostic",file=sys.stderr,flush=True); '
        'print("received:"+sys.stdin.read(),flush=True)')
    with TerminalSessions() as host:
        handle = start(host, tmp_path, code)['handle']
        live = until(host, handle, lambda r: 'olá' in r['output']['stdout']['text'])
        assert live['status'] == 'running'
        assert start(host, tmp_path, code)['handle'] == handle
        with pytest.raises(ValueError, match='request_changed'):
            start(host, tmp_path, 'print("second execution")')
        kwargs = dict(owner='conversation-a', handle=handle, request_id='input-1', text='one line\n')
        assert host.send(**kwargs)['state'] == 'queued'
        host.send(**kwargs)  # lost transport response must not write again
        with pytest.raises(ValueError, match='request_changed'):
            host.send(**{**kwargs, 'text': 'different'})
        host.close_input(owner='conversation-a', handle=handle)
        result = until(host, handle, lambda r: r['cleanup_verified'])
        assert result['status'] == 'exited' and result['exit_code'] == 0
        assert result['output']['stdout']['text'] == 'olá\nreceived:one line\n\n'
        assert result['output']['stderr']['text'] == 'diagnostic\n'
        assert host.send(**kwargs)['state'] == 'written'
        assert start(host, tmp_path, code)['handle'] == handle
        with pytest.raises(ValueError, match='input_closed'):
            host.send(**{**kwargs, 'request_id': 'input-2'})


def test_output_tail_reports_loss_and_supports_independent_readers(tmp_path):
    with TerminalSessions(output_chars=128) as host:
        handle = start(host, tmp_path, 'print("α"*10000,end="")')['handle']
        result = until(host, handle, lambda r: r['cleanup_verified'])
        assert result['output']['stdout'] == dict(text='α'*128, cursor=10000,
                                                 end=10000, truncated=True, more=False)
        page = host.read(owner='conversation-a', handle=handle, limit=32)
        assert len(page['output']['stdout']['text']) == 32
        assert page['output']['stdout']['more']
        assert host.read(owner='conversation-a', handle=handle, limit=32) == page
        tail = host.read(owner='conversation-a', handle=handle,
                         cursors={'stdout': page['output']['stdout']['cursor']})
        assert tail['output']['stdout']['text'] == 'α'*96
        assert not tail['output']['stdout']['truncated']
        with pytest.raises(ValueError, match='cursor_invalid'):
            host.read(owner='conversation-a', handle=handle, cursors={'stdout': 10001})


def test_owner_and_host_lifetime_isolation(tmp_path):
    host = TerminalSessions()
    try:
        handle = start(host, tmp_path, 'import time; print("ready"); time.sleep(60)')['handle']
        until(host, handle, lambda r: 'ready' in r['output']['stdout']['text'])
        for method, extra in ((host.read, {}), (host.stop, {}),
                              (host.send, {'request_id':'x', 'text':'x'}), (host.close_input, {})):
            with pytest.raises(ValueError, match='handle_unavailable'):
                method(owner='conversation-b', handle=handle, **extra)
        with TerminalSessions() as new_host:
            with pytest.raises(ValueError, match='handle_unavailable'):
                new_host.stop(owner='conversation-a', handle=handle)
        # Reconnecting a consumer neither kills nor restarts the still-owned command.
        assert host.read(owner='conversation-a', handle=handle)['status'] == 'running'
    finally:
        host.close()
    result = host.read(owner='conversation-a', handle=handle)
    assert result['status'] == 'stopped' and result['cleanup_verified']
    with pytest.raises(ValueError, match='host_closed'):
        start(host, tmp_path, 'print("never")', request_id='new')


@pytest.mark.parametrize('mode', ['stop', 'timeout', 'leader_exit'])
def test_descendants_cleaned_even_after_leader_exits_zero(tmp_path, mode, monkeypatch):
    elapsed = [0.0]
    if mode == 'timeout':
        # Let the real descendant start before advancing the supervisor deadline.
        # Keep polling and cleanup clocks progressing even on a slow CI host.
        monkeypatch.setattr(terminal_sessions, 'time', SimpleNamespace(
            monotonic=lambda: time.monotonic() + elapsed[0], sleep=time.sleep))
    marker = tmp_path / 'descendant-was-not-cleaned'
    child = ('import time; from pathlib import Path; time.sleep(2); '
             'Path(' + repr(str(marker)) + ').write_text("leak")')
    code = ('import subprocess,sys,time; '
            'subprocess.Popen([sys.executable,"-I","-c",' + repr(child) + ']); '
            'print("spawned",flush=True); ' + ('time.sleep(60)' if mode != 'leader_exit' else ''))
    with TerminalSessions() as host:
        handle = start(host, tmp_path, code, timeout=20)['handle']
        until(host, handle, lambda r: 'spawned' in r['output']['stdout']['text'])
        if mode == 'timeout':
            elapsed[0] += 20
        if mode == 'stop':
            result = host.stop(owner='conversation-a', handle=handle)
        else:
            result = until(host, handle, lambda r: r['cleanup_verified'])
        assert result['cleanup_verified']
        assert result['reason'] == {'stop': 'stopped', 'timeout':'timeout', 'leader_exit':None}[mode]
        if mode == 'leader_exit':
            assert result['exit_code'] == 0
        time.sleep(2.1)
        assert not marker.exists()


def test_unread_input_is_bounded_and_does_not_block_stop(tmp_path):
    # Exceed both Windows and Linux pipe capacity so the writer really blocks.
    size = 2 * 1024 * 1024
    with TerminalSessions(input_bytes=size) as host:
        handle = start(host, tmp_path, 'import time; print("ready"); time.sleep(60)')['handle']
        until(host, handle, lambda r: 'ready' in r['output']['stdout']['text'])
        began = time.monotonic()
        host.send(owner='conversation-a', handle=handle, request_id='full', text='x'*size)
        assert time.monotonic() - began < .5
        with pytest.raises(ValueError, match='queue_full'):
            host.send(owner='conversation-a', handle=handle, request_id='more', text='x'*size)
        result = host.stop(owner='conversation-a', handle=handle)
        assert result['cleanup_verified']
        receipt = host.send(owner='conversation-a', handle=handle, request_id='full', text='x'*size)
        assert receipt['state'] in {'unknown', 'not_written'}


def test_launch_failure_is_retained_and_no_shell_interpolation(tmp_path):
    with TerminalSessions() as host:
        kwargs = dict(owner='a', request_id='missing', argv=[str(tmp_path/'missing.exe')],
                      cwd=tmp_path, env={})
        failed = host.start(**kwargs)
        assert failed['status'] == 'failed' and failed['cleanup_verified']
        assert host.start(**kwargs) == failed
        argv = [sys.executable, '-I', '-c', 'import sys; print(repr(sys.argv[1:]))',
                '', 'space value', '$(touch bad)', '& echo bad']
        handle = host.start(owner='a', request_id='literal', argv=argv, cwd=tmp_path, env=dict(os.environ))['handle']
        result = until(host, handle, lambda r: r['cleanup_verified'], owner='a')
        assert result['output']['stdout']['text'].strip() == repr(argv[4:])
        assert not (tmp_path/'bad').exists()


def test_capacity_preserves_existing_effect_identity(tmp_path):
    with TerminalSessions(max_running=1, max_records=2) as host:
        code = 'import time; time.sleep(60)'
        handle = start(host, tmp_path, code)['handle']
        with pytest.raises(ValueError, match='running_limit'):
            start(host, tmp_path, code, request_id='run-2')
        host.stop(owner='conversation-a', handle=handle)
        second = start(host, tmp_path, 'print("done")', request_id='run-2')['handle']
        until(host, second, lambda r: r['cleanup_verified'])
        with pytest.raises(ValueError, match='record_limit'):
            start(host, tmp_path, code, request_id='run-3')
        assert start(host, tmp_path, code)['handle'] == handle


def test_closing_one_owner_preserves_another_conversation(tmp_path):
    with TerminalSessions() as host:
        first = start(host, tmp_path, 'import time; time.sleep(60)')['handle']
        second = start(host, tmp_path, 'import time; time.sleep(60)', owner='conversation-b')['handle']
        results = host.close_owner('conversation-a')
        assert [r['handle'] for r in results] == [first]
        assert results[0]['cleanup_verified']
        assert host.list(owner='conversation-a') == []
        assert host.list(owner='conversation-a', include_completed=True)[0]['handle'] == first
        assert host.list(owner='conversation-b')[0]['handle'] == second
        assert host.read(owner='conversation-b', handle=second)['status'] == 'running'


def test_thread_start_failure_does_not_leak_pipes_or_child(tmp_path, monkeypatch):
    import threading
    original = threading.Thread.start
    def fail_writer(thread):
        if thread.name == 'tare-terminal-input':
            raise RuntimeError('injected writer failure')
        return original(thread)
    monkeypatch.setattr(threading.Thread, 'start', fail_writer)
    with TerminalSessions() as host:
        result = start(host, tmp_path, 'import time; time.sleep(60)')
        assert result['status'] == 'failed'
        assert result['cleanup_verified']


@pytest.mark.parametrize('timeout', [0, -1, True, float('nan'), float('inf')])
def test_invalid_deadline_never_starts_a_child(tmp_path, timeout):
    with TerminalSessions() as host:
        with pytest.raises(ValueError, match='timeout_invalid'):
            start(host, tmp_path, 'raise AssertionError("must not run")', timeout=timeout)


@pytest.mark.skipif(os.name != 'nt', reason='Windows suspended assignment containment')
def test_failed_windows_job_assignment_reaps_suspended_child(tmp_path, monkeypatch):
    from compute_plane import terminal_sessions
    def fail(*_):
        raise OSError('injected assignment failure')
    monkeypatch.setattr(terminal_sessions.WindowsJob, 'assign_and_resume', fail)
    with TerminalSessions() as host:
        result = start(host, tmp_path, 'from pathlib import Path; Path("leak").write_text("bad")')
        assert result['status'] == 'failed'
        assert result['cleanup_verified'] and not (tmp_path/'leak').exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job termination failure')
def test_failed_job_termination_reports_uncertainty_and_reaps_leader(tmp_path, monkeypatch):
    from compute_plane import terminal_sessions
    children = []
    real_popen = terminal_sessions.subprocess.Popen
    def capture(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    def fail(*_):
        raise OSError('injected Job failure')
    monkeypatch.setattr(terminal_sessions.subprocess, 'Popen', capture)
    monkeypatch.setattr(terminal_sessions.WindowsJob, 'assign_and_resume', fail)
    monkeypatch.setattr(terminal_sessions.WindowsJob, 'terminate', fail)
    host = TerminalSessions()
    result = start(host, tmp_path, 'import time; time.sleep(60)')
    assert result['status'] == 'unknown' and not result['cleanup_verified']
    assert children[0].poll() is not None
    with pytest.raises(RuntimeError, match='cleanup_unknown'):
        start(host, tmp_path, 'print("must not start")', request_id='different')
    with pytest.raises(RuntimeError, match='cleanup_unknown'):
        host.close()
    assert all(s.closed for s in (children[0].stdin, children[0].stdout, children[0].stderr))
