"""Real ConPTY/POSIX PTY behavior, owned processes and no remote service."""
import os
import re
import sys
import time

import pytest

from compute_plane.terminal_sessions import TerminalSessions


def text(result):
    value = result['output']['stdout']['text']
    value = re.sub(r'\x1b\][^\x07]*(?:\x07|\x1b\\)', '', value)
    return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value).replace('\r', '')


def launch(host, tmp_path, code, **kwargs):
    env = dict(os.environ, TERM='xterm-256color')
    return host.start(owner='pty-test', request_id='start',
        argv=[sys.executable, '-I', '-u', '-c', code], cwd=tmp_path, env=env,
        mode='pty', columns=90, rows=30, **kwargs)


def observe(host, handle, predicate):
    until = time.monotonic()+12
    while time.monotonic() < until:
        result = host.read(owner='pty-test', handle=handle)
        if predicate(result):
            return result
        time.sleep(.02)
    pytest.fail('PTY state: '+repr(result))


def test_real_terminal_resize_and_keyboard_input(tmp_path):
    code = (
        'import os,sys; '
        'print("TTY="+str(os.isatty(0))+","+str(os.isatty(1)),flush=True); '
        'print("SIZE="+str(tuple(os.get_terminal_size(1))),flush=True); '
        'value=input(); '
        'print("RECEIVED="+value,flush=True); '
        # ConPTY applies a resize to the child asynchronously: the line can arrive first.
        'import time; deadline=time.monotonic()+5\n'
        'while tuple(os.get_terminal_size(1))==(90,30) and time.monotonic()<deadline: time.sleep(.05)\n'
        'print("RESIZED="+str(tuple(os.get_terminal_size(1))),flush=True)')
    with TerminalSessions() as host:
        result = launch(host, tmp_path, code)
        handle = result['handle']
        ready = observe(host, handle, lambda r: 'SIZE=(90, 30)' in text(r))
        assert ready['mode'] == 'pty' and ready['status'] == 'running'
        assert ready['may_have_executed']
        assert 'TTY=True,True' in text(ready)
        host.resize(owner='pty-test', handle=handle, columns=110, rows=40)
        host.send(owner='pty-test', handle=handle, request_id='line', text='hello-pty\r')
        done = observe(host, handle, lambda r: r['cleanup_verified'])
        assert done['exit_code'] == 0
        assert 'RECEIVED=hello-pty' in text(done)
        assert 'RESIZED=(110, 40)' in text(done)
        assert done['output']['stderr']['text'] == ''
        with pytest.raises(ValueError, match='pty_closed'):
            host.resize(owner='pty-test', handle=handle, columns=80, rows=24)


def test_ctrl_c_reaches_terminal_child(tmp_path):
    code = 'import time\nprint("READY-INT",flush=True)\ntry:\n time.sleep(60)\nexcept KeyboardInterrupt:\n print("INTERRUPTED",flush=True)\n'
    with TerminalSessions() as host:
        handle = launch(host, tmp_path, code)['handle']
        observe(host, handle, lambda r: 'READY-INT' in text(r))
        host.send(owner='pty-test', handle=handle, request_id='interrupt', text='\x03')
        done = observe(host, handle, lambda r: r['cleanup_verified'])
        assert 'INTERRUPTED' in text(done)
        assert done['exit_code'] == 0


@pytest.mark.skipif(os.name != 'nt', reason='Windows inherited Ctrl+C ignore mark')
def test_ctrl_c_reaches_terminal_child_from_host_that_ignores_ctrl_c(tmp_path):
    # The TUI backend starts the agent with CREATE_NEW_PROCESS_GROUP: that host
    # ignores Ctrl+C and Windows passes the mark to every descendant.
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
         '--basetemp', str(tmp_path/'inner'),
         os.path.abspath(__file__)+'::test_ctrl_c_reaches_terminal_child'],
        cwd=root, capture_output=True, text=True, timeout=120,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    assert result.returncode == 0, result.stdout[-4000:]+result.stderr[-2000:]


@pytest.mark.parametrize('mode', ['stop', 'timeout', 'leader_exit'])
def test_pty_descendant_cleanup(tmp_path, mode):
    marker = tmp_path/'pty-leak'
    child = 'import time; from pathlib import Path; time.sleep(2); Path('+repr(str(marker))+').write_text("leak")'
    code = ('import subprocess,sys,time; subprocess.Popen([sys.executable,"-I","-c",'+repr(child)+']); '
            'print("CHILD-STARTED",flush=True); '+('time.sleep(60)' if mode != 'leader_exit' else ''))
    with TerminalSessions() as host:
        handle = launch(host, tmp_path, code, timeout=.6 if mode == 'timeout' else 30)['handle']
        observe(host, handle, lambda r: 'CHILD-STARTED' in text(r))
        if mode == 'stop':
            done = host.stop(owner='pty-test', handle=handle)
        else:
            done = observe(host, handle, lambda r: r['cleanup_verified'])
        assert done['cleanup_verified']
        assert done['reason'] == {'stop':'stopped', 'timeout':'timeout', 'leader_exit':None}[mode]
        time.sleep(2.1)
        assert not marker.exists()


def test_pty_explicit_mode_and_eof_semantics(tmp_path):
    with TerminalSessions() as host:
        handle = launch(host, tmp_path, 'import time; time.sleep(60)')['handle']
        with pytest.raises(ValueError, match='control_key_or_stop'):
            host.close_input(owner='pty-test', handle=handle)
        with pytest.raises(ValueError, match='request_changed'):
            host.start(owner='pty-test', request_id='start', argv=['x'], cwd=tmp_path, env={}, mode='pipes')
        with pytest.raises(ValueError, match='handle_unavailable'):
            host.resize(owner='other-owner', handle=handle, columns=50, rows=20)
        with pytest.raises(ValueError, match='dimensions'):
            host.resize(owner='pty-test', handle=handle, columns=True, rows=30)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX controlling terminal')
def test_posix_foreground_group_and_ctrl_d(tmp_path):
    code = 'import os,sys; print("CONTROL="+str(os.tcgetpgrp(0)==os.getpgrp()),flush=True); print("EOF="+str(sys.stdin.read()==""),flush=True)'
    with TerminalSessions() as host:
        handle = launch(host, tmp_path, code)['handle']
        observe(host, handle, lambda r: 'CONTROL=True' in text(r))
        host.send(owner='pty-test', handle=handle, request_id='eof', text='\x04')
        done = observe(host, handle, lambda r: r['cleanup_verified'])
        assert done['exit_code'] == 0 and 'EOF=True' in text(done)


def test_start_receipt_distinguishes_no_command_from_possible_effects(tmp_path, monkeypatch):
    import threading
    original = threading.Thread.start
    def fail_writer(thread):
        if thread.name == 'tare-terminal-input':
            raise RuntimeError('injected post-spawn failure')
        return original(thread)
    with TerminalSessions() as host:
        missing = host.start(owner='pty-test', request_id='missing',
            argv=[str(tmp_path/'does-not-exist')], cwd=tmp_path, env={}, mode='pipes')
        assert not missing['may_have_executed'] and missing['cleanup_verified']
        monkeypatch.setattr(threading.Thread, 'start', fail_writer)
        started = launch(host, tmp_path, 'import time; time.sleep(60)')
        assert started['may_have_executed'] and started['cleanup_verified']
        assert started['reason'] == 'start_failed'


def test_native_shell_reads_from_real_terminal(tmp_path):
    import shutil
    if os.name == 'nt':
        executable = os.path.join(os.environ['SystemRoot'], 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
        argv = [executable, '-NoLogo', '-NoProfile', '-Command',
                "Write-Output 'SHELL-READY'; $v=Read-Host; Write-Output ('SHELL-INPUT='+$v)"]
    else:
        executable = shutil.which('bash')
        assert executable, 'Bash is required for this supported-host qualification'
        argv = [executable, '--noprofile', '--norc', '-c',
                'printf "SHELL-READY\\n"; read -r value; printf "SHELL-INPUT=%s\\n" "$value"']
    with TerminalSessions() as host:
        handle = host.start(owner='pty-test', request_id='shell', argv=argv, cwd=tmp_path,
            env=dict(os.environ, TERM='xterm-256color'), mode='pty')['handle']
        observe(host, handle, lambda r: 'SHELL-READY' in text(r))
        host.send(owner='pty-test', handle=handle, request_id='line', text='actual-shell\r')
        done = observe(host, handle, lambda r: r['cleanup_verified'])
        assert done['exit_code'] == 0
        assert 'SHELL-INPUT=actual-shell' in text(done)


@pytest.mark.skipif(os.name == 'nt', reason='Linux terminal-session process groups')
def test_other_process_groups_in_terminal_session_are_cleaned(tmp_path):
    marker = tmp_path/'subgroup-leak'
    child = ('import os,signal,time; from pathlib import Path; os.setpgid(0,0); '
             'signal.signal(signal.SIGHUP,signal.SIG_IGN); print("SUBGROUP-READY",flush=True); '
             'time.sleep(2); Path('+repr(str(marker))+').write_text("leak")')
    code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-I","-c",'+repr(child)+']); time.sleep(60)'
    with TerminalSessions() as host:
        handle = launch(host, tmp_path, code)['handle']
        observe(host, handle, lambda r: 'SUBGROUP-READY' in text(r))
        result = host.stop(owner='pty-test', handle=handle)
        assert result['cleanup_verified']
        assert result['containment'] == 'linux-terminal-session'
        time.sleep(2.1)
        assert not marker.exists()


@pytest.mark.skipif(os.name != 'nt', reason='ConPTY failure before suspended child creation')
def test_missing_pty_executable_releases_pseudoconsole(tmp_path):
    with TerminalSessions() as host:
        result = host.start(owner='pty-test', request_id='missing-pty',
            argv=[str(tmp_path/'missing.exe')], cwd=tmp_path, env=dict(os.environ), mode='pty')
        assert result['status'] == 'failed' and result['cleanup_verified']
        assert not result['may_have_executed']


def test_resize_while_output_is_arriving_remains_responsive(tmp_path):
    import threading
    code = ('import time; print("FLOOD-READY",flush=True)\n'
            'for i in range(300):\n print("x"*200,flush=True); time.sleep(.002)\n'
            'input()\n')
    with TerminalSessions() as host:
        handle = launch(host, tmp_path, code)['handle']
        observe(host, handle, lambda r: 'FLOOD-READY' in text(r))
        failures = []
        def resize():
            try:
                for width in range(90, 110):
                    host.resize(owner='pty-test', handle=handle, columns=width, rows=40)
            except Exception as exc:
                failures.append(exc)
        worker = threading.Thread(target=resize, daemon=True)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), 'resize deadlocked with output reader'
        assert not failures
        host.send(owner='pty-test', handle=handle, request_id='finish', text='\r')
        done = observe(host, handle, lambda r: r['cleanup_verified'])
        assert done['exit_code'] == 0
        assert done['columns'] == 109
