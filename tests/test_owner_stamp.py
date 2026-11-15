"""TARE_OWNER stamp, owner_alive, and unsandboxed POSIX children that die with tare (real processes on Linux)."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from compute_plane import managed_process
from compute_plane.managed_process import owner_alive, owner_stamp, run_process

ROOT = Path(__file__).resolve().parents[1]
linux = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='PR_SET_PDEATHSIG and /proc are Linux')


def fake_proc(monkeypatch, table):
    def stat(pid):
        value = table.get(pid, FileNotFoundError())
        if isinstance(value, Exception):
            raise value
        return [value[0]] + ['0'] * 18 + [value[1]]
    monkeypatch.setattr(managed_process, '_proc_stat', stat)


def test_stamp_and_owner_alive_on_fakes(monkeypatch):
    me = os.getpid()
    fake_proc(monkeypatch, {me: ('R', '500'), 41: ('S', '77'), 42: ('Z', '78'), 43: PermissionError()})
    assert owner_stamp() == f'{me}:500'
    assert owner_alive(f'{me}:500') is True
    assert owner_alive('41:77') is True
    assert owner_alive('41:78') is False  # pid reused by a process started at another time
    assert owner_alive('42:78') is False  # zombie
    assert owner_alive('44:1') is False  # gone
    assert owner_alive('43:1') is None  # unreadable
    for garbage in (None, 41, '', '41', '41:', ':77', '0:77', '-41:77', '41:077', '41:7:7', ' 41:77', '٤١:77'):
        assert owner_alive(garbage) is None
    fake_proc(monkeypatch, {})  # no /proc on this platform
    assert owner_stamp() == f'{me}:'
    assert owner_alive('41:77') is None and owner_alive(owner_stamp()) is None


@linux
def test_real_owner_alive():
    start = managed_process._proc_stat(os.getpid())[19]
    assert owner_stamp() == f'{os.getpid()}:{start}'
    assert owner_alive(owner_stamp()) is True
    assert owner_alive(f'{os.getpid()}:{int(start) + 1}') is False
    assert owner_alive('not a stamp') is None


@linux
def test_stamped_environment_is_exact_and_signals_are_default(tmp_path):
    env = {'PATH': os.environ['PATH'], 'LC_CTYPE': 'C'}  # a C locale the shim's interpreter would coerce
    result = run_process(['sh', '-c', 'env; grep SigIgn /proc/self/status'], cwd=tmp_path, dest=tmp_path / 'run',
                         timeout=20, env=env)
    assert result['exit_code'] == 0 and result['cleanup_verified']
    lines = (tmp_path / 'run' / 'stdout.txt').read_text().splitlines()
    ignored = int(lines.pop().split()[1], 16)
    assert not ignored & (1 << signal.SIGPIPE - 1 | 1 << signal.SIGXFSZ - 1)
    assert {'LC_CTYPE=C', 'PATH=' + env['PATH'], 'TARE_OWNER=' + owner_stamp()} <= set(lines)


@linux
def test_a_command_that_cannot_start_still_raises_in_the_parent(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_process([str(tmp_path / 'missing')], cwd=tmp_path, dest=tmp_path / 'run', timeout=20)
    with pytest.raises(PermissionError):
        managed_process.run_pipe_process([str(tmp_path)], cwd=tmp_path, timeout=20)


PARENT = '''
import sys
sys.path.insert(0, sys.argv[1])
from compute_plane.managed_process import run_process
run_process([sys.executable, '-c', sys.argv[3]], cwd=sys.argv[2], dest=sys.argv[2] + '/run', timeout=60)
'''
CHILD = '''
import os, subprocess, sys, time
grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
open('pids.tmp', 'w').write(f'{os.getpid()} {grandchild.pid}')
os.rename('pids.tmp', 'pids')
time.sleep(60)
'''


def running(pid):
    try:
        return managed_process._proc_stat(pid)[0] not in ('Z', 'X')
    except (FileNotFoundError, ProcessLookupError):
        return False


@linux
def test_a_killed_owner_takes_its_child_and_leaves_a_stamped_orphan(tmp_path):
    parent = subprocess.Popen([sys.executable, '-c', PARENT, str(ROOT), str(tmp_path), CHILD])
    grandchild = None
    try:
        until = time.monotonic() + 20
        while not (tmp_path / 'pids').exists():
            assert time.monotonic() < until and parent.poll() is None
            time.sleep(.05)
        child, grandchild = map(int, (tmp_path / 'pids').read_text().split())
        stamp = f'{parent.pid}:{managed_process._proc_stat(parent.pid)[19]}'
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=10)
        until = time.monotonic() + 10
        while running(child) and time.monotonic() < until:
            time.sleep(.05)
        assert not running(child)
        assert running(grandchild)
        environ = Path(f'/proc/{grandchild}/environ').read_bytes().split(b'\0')
        assert ('TARE_OWNER=' + stamp).encode() in environ
        assert owner_alive(stamp) is False
    finally:
        if parent.poll() is None:
            parent.kill()
        if grandchild is not None and running(grandchild):
            os.kill(grandchild, signal.SIGKILL)
