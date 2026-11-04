"""Job accounting must not substitute for process-handle termination confirmation."""
import os
import sys
import time
from types import SimpleNamespace

import pytest

from compute_plane.managed_process import WindowsJob


@pytest.mark.parametrize('state,error', [(0, None), (258, TimeoutError), (0xffffffff, OSError)])
def test_termination_waits_for_owned_member_handles_and_releases_them(state, error):
    calls = []
    job = object.__new__(WindowsJob)
    job.handle = 10
    job._member_handles = lambda: [20]
    job.c = SimpleNamespace(get_last_error=lambda: 6, WinError=lambda code: OSError(code, 'fixture wait failed'))
    job.k = SimpleNamespace(
        TerminateJobObject=lambda *a: calls.append(('terminate', *a)) or True,
        WaitForSingleObject=lambda handle, milliseconds: calls.append(('wait', handle, milliseconds)) or state,
        CloseHandle=lambda handle: calls.append(('close', handle)) or True)
    if error:
        with pytest.raises(error):
            job.terminate()
    else:
        job.terminate()
    assert calls[0] == ('terminate', 10, 1)
    assert calls[1][0:2] == ('wait', 20) and 0 <= calls[1][2] <= 5000
    assert calls[-1] == ('close', 20)


@pytest.mark.skipif(os.name != 'nt', reason='Windows process handle completion')
@pytest.mark.parametrize('streamed', [False, True])
def test_pipe_cleanup_signals_the_same_live_descendant_handle(tmp_path, streamed):
    import ctypes as c
    from ctypes import wintypes as w
    from compute_plane.managed_process import run_pipe_process
    k = c.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes, k.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
    k.WaitForSingleObject.argtypes, k.WaitForSingleObject.restype = [w.HANDLE, w.DWORD], w.DWORD
    k.CloseHandle.argtypes, k.CloseHandle.restype = [w.HANDLE], w.BOOL
    path = tmp_path/'child.pid'
    child = 'import os,time; from pathlib import Path; Path('+repr(str(path))+').write_text(str(os.getpid())); time.sleep(30)'
    parent = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'+repr(child)+']); time.sleep(30)'
    handle = None
    def stop_after_observation():
        nonlocal handle
        if handle is None and path.exists() and (text := path.read_text()):
            handle = k.OpenProcess(0x100000, False, int(text))
            assert handle and k.WaitForSingleObject(handle, 0) == 258
        return handle is not None
    try:
        started = time.monotonic()
        receipt, _, _ = run_pipe_process([sys.executable, '-I', '-c', parent], cwd=tmp_path,
            timeout=2, cancelled=stop_after_observation, on_stdout=(lambda chunk: None) if streamed else None)
        assert handle and receipt['cancelled'] and receipt['cleanup_verified']
        assert k.WaitForSingleObject(handle, 0) == 0
        assert time.monotonic()-started < 3
    finally:
        if handle:
            k.CloseHandle(handle)
