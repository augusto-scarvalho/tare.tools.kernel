"""A cleanup deadline bounds waiting, not the final observation of already-finished owners."""
from types import SimpleNamespace
import pytest
from compute_plane import terminal_sessions


@pytest.mark.parametrize('reader_alive,tree_active,verified',[(False,0,True),(True,0,False),(False,1,False)])
def test_cleanup_observes_final_state_after_join_uses_the_wait_budget(monkeypatch,reader_alive,tree_active,verified):
    command=terminal_sessions._Command('owner','signature',1024,1024,8,'pipes',80,24)
    stream=SimpleNamespace(close=lambda:None)
    command.process=SimpleNamespace(pid=1,returncode=0,poll=lambda:0,wait=lambda **kw:0,
        stdin=stream,stdout=stream,stderr=stream)
    observations=[];clock=[0.0]
    command.job=SimpleNamespace(terminate=lambda:None,close=lambda:None,
        active=lambda:observations.append(tree_active) or tree_active)
    command.readers=[SimpleNamespace(ident=1,is_alive=lambda:reader_alive,
        join=lambda **kw:clock.__setitem__(0,clock[0]+kw['timeout']))]
    command.reason='timeout'
    monkeypatch.setattr(terminal_sessions,'time',SimpleNamespace(monotonic=lambda:clock[0],
        sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds)))
    command._cleanup()
    assert observations==[tree_active]
    assert command.cleanup_verified is verified
    assert command.status==('stopped' if verified else 'unknown')
    assert clock[0]==5.0 and command.done.is_set()
