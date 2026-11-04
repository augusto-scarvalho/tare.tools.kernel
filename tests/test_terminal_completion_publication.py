"""Cleanup completion must never be published before the final process status."""
from types import SimpleNamespace

import pytest
from compute_plane.terminal_sessions import _Command


@pytest.mark.parametrize('exit_code',[0,3])
def test_snapshot_during_last_resource_close_does_not_publish_completion(exit_code):
    session=_Command('owner','signature',4096,4096,16,'pipe',80,24)
    session.status='running';session.exit_code=exit_code;session.timeout=30
    observed=[]
    def close_job():
        # This is the deterministic interleaving from the Windows main-CI
        # failure: the process/readers are done, final resource close is not.
        observed.append(session.snapshot())
    session.process=SimpleNamespace(poll=lambda:exit_code,wait=lambda **kw:None,
        returncode=exit_code,stdin=SimpleNamespace(close=lambda:None),
        stdout=SimpleNamespace(close=lambda:None),stderr=SimpleNamespace(close=lambda:None))
    session.job=SimpleNamespace(terminate=lambda:None,active=lambda:False,close=close_job)
    session._cleanup()
    assert len(observed)==1
    assert observed[0]['status']=='running' and not observed[0]['cleanup_verified']
    final=session.snapshot()
    assert final['cleanup_verified'] and final['status']=='exited'
    assert final['exit_code']==exit_code and session.done.is_set()
