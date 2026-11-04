"""Physical custody/crash recovery without Aider, provider calls or replayed effects."""
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from compute_plane.native_journal import NativeJournal, custody_lock
from tare_tools_agent_runtime import (ConversationLog, ProjectionSpec, BindingRef,
    CancellationToken, ExecutionControl, NativeSession, NativeBudget)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows mandatory byte-range lock')
def test_empty_contended_lock_reports_custody_before_any_write(tmp_path):
    import msvcrt
    path = tmp_path/'writer.lock'
    path.touch()
    # A first writer can hold a region beyond EOF. A contender must attempt
    # locking, not initialize/write that already protected byte beforehand.
    with path.open('r+b', buffering=0) as held:
        msvcrt.locking(held.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            with pytest.raises(ValueError, match='native_custody_has_active_writer'):
                with custody_lock(tmp_path):
                    pytest.fail('An independent handle entered occupied custody')
        finally:
            held.seek(0); msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, 1)
    assert path.read_bytes() == b''
    with custody_lock(tmp_path):
        NativeJournal(tmp_path/'journal.jsonl').append({'kind': 'acquired-after-release'})
    assert len(NativeJournal(tmp_path/'journal.jsonl').snapshot().records) == 1


def test_observation_budget_does_not_relax_tactical_journal(tmp_path):
    from compute_plane.native_journal import NativeObservationJournal
    from tare_tools_agent_runtime import ConversationRejected
    event = {'kind': 'receipt', 'detail': 'x' * 300000}
    with pytest.raises(ConversationRejected):
        NativeJournal(tmp_path/'tactical.jsonl').append(event)
    observations = NativeObservationJournal(tmp_path/'observations.jsonl')
    observations.append(event)
    assert NativeObservationJournal(observations.path).snapshot() == observations.snapshot()
    original = observations.path.read_bytes()
    with pytest.raises(ConversationRejected):
        observations.append({'detail': 'x' * (1024 * 1024)})
    assert observations.path.read_bytes() == original


def test_a_max_size_reply_with_reasoning_fits_and_replays(tmp_path):
    # A 16,384-token reply is ~80 KB of text; the event keeps it twice (reply and provider message).
    text = 'y' * 90000
    event = {'kind': 'model_returned', 'call_id': 's/model/1', 'reply': {
        'text': text, 'metadata_json': json.dumps({'assistant': {'content': text, 'reasoning_content': text[:40000]}})}}
    journal = NativeJournal(tmp_path/'journal.jsonl')
    journal.append(event)
    assert NativeJournal(journal.path).snapshot() == journal.snapshot()


@pytest.mark.parametrize('winerror', [5, 32, 33])
def test_metadata_replace_recovers_from_windows_reader(tmp_path, monkeypatch, winerror):
    from compute_plane import native_journal as journal
    path = tmp_path/'state.json'
    path.write_bytes(b'{"old":true}')
    replace = journal.os.replace
    attempts, pauses = [], []
    def occupied_once(source, target):
        attempts.append(Path(source).read_bytes())
        if len(attempts) == 1:
            assert Path(target).read_bytes() == b'{"old":true}'
            error = PermissionError('reader holds metadata')
            error.winerror = winerror
            raise error
        replace(source, target)
    monkeypatch.setattr(journal.os, 'replace', occupied_once)
    monkeypatch.setattr(journal, 'sleep', pauses.append)
    journal.save_json(path, {'new': True})
    assert attempts == [b'{"new":true}', b'{"new":true}'] and pauses == [.02]
    assert json.loads(path.read_bytes()) == {'new': True}
    assert not path.with_name('state.json.pending').exists()


@pytest.mark.parametrize('winerror,expected_attempts', [(5, 6), (None, 1)])
def test_metadata_replace_failure_is_bounded_and_preserves_old_bytes(tmp_path, monkeypatch, winerror, expected_attempts):
    from compute_plane import native_journal as journal
    path = tmp_path/'state.json'
    path.write_bytes(b'{"old":true}')
    attempts, pauses = [], []
    error = PermissionError('permanent access failure')
    if winerror is not None:
        error.winerror = winerror
    def denied(source, target):
        attempts.append(Path(source).read_bytes())
        raise error
    monkeypatch.setattr(journal.os, 'replace', denied)
    monkeypatch.setattr(journal, 'sleep', pauses.append)
    with pytest.raises(PermissionError) as caught:
        journal.save_json(path, {'new': True})
    assert caught.value is error and len(attempts) == expected_attempts
    assert len(pauses) == expected_attempts - 1 and sum(pauses) <= .62
    assert path.read_bytes() == b'{"old":true}'
    assert path.with_name('state.json.pending').read_bytes() == b'{"new":true}'


def test_projection_survives_default_limit_and_reopen(tmp_path):
    path = tmp_path/'journal.jsonl'
    spec = ProjectionSpec(1, 'native-context-v1', tuple(range(1, 45)))
    with custody_lock(tmp_path):
        journal = NativeJournal(path)
        for i in range(44):
            journal.append({'kind': 'model_returned', 'index': i, 'text': 'x' * 50000})
        projection = journal.project(spec)
        assert len(projection.projected_bytes) > 2 * 1024 * 1024
        identity = projection.sha256
    with custody_lock(tmp_path):
        reopened = NativeJournal(path)
        assert reopened.project(spec).sha256 == identity
        assert len(reopened.snapshot().records) == 44


def test_journal_reopen_preserves_exact_public_projection(tmp_path):
    memory=ConversationLog()
    with custody_lock(tmp_path):
        disk=NativeJournal(tmp_path/'journal.jsonl')
        for event in ({'kind':'one','text':'olá'}, {'kind':'two','value':[1,True,None]}):
            assert disk.append(event)==memory.append(event)
    with custody_lock(tmp_path):
        disk=NativeJournal(tmp_path/'journal.jsonl')
        assert disk.snapshot()==memory.snapshot()
        assert disk.project(ProjectionSpec(1,'test'))==memory.project(ProjectionSpec(1,'test'))


@pytest.mark.parametrize('corruption',['torn','changed','gap','oversized'])
def test_invalid_journal_is_not_repaired_by_discarding_history(tmp_path,corruption):
    path=tmp_path/'journal.jsonl'
    with custody_lock(tmp_path):
        journal=NativeJournal(path);journal.append({'kind':'one'});journal.append({'kind':'two'})
        raw=path.read_bytes()
        if corruption=='torn':raw=raw[:-1]
        elif corruption=='changed':raw=raw.replace(b'one',b'xxx')
        elif corruption=='gap':raw=raw.splitlines(keepends=True)[1]
        else:raw=b'x'*131073
        path.write_bytes(raw)
        with pytest.raises(ValueError):NativeJournal(path)
        assert path.read_bytes()==raw


def test_failure_after_fsync_poison_writer_but_complete_intent_survives(tmp_path,monkeypatch):
    with custody_lock(tmp_path):
        path=tmp_path/'journal.jsonl';journal=NativeJournal(path)
        def fail(_):raise OSError('directory flush failed')
        monkeypatch.setattr('compute_plane.native_journal._sync_directory',fail)
        with pytest.raises(OSError):journal.append({'kind':'model_requested','call_id':'one'})
        assert not journal.snapshot().records
        with pytest.raises(RuntimeError):journal.append({'kind':'another'})
        assert NativeJournal(path).snapshot().records[0].event()['call_id']=='one'


def child(script,root):
    kernel=Path(__file__).resolve().parents[1]
    setup='import sys;sys.path[:0]=sys.argv[1:3];\n'
    return subprocess.run([sys.executable,'-c',setup+script,str(kernel),str(kernel.parent/'runtime/src'),str(root)],
                          capture_output=True,text=True,timeout=20)


def test_physical_writer_exclusion_and_release(tmp_path):
    script='''from compute_plane.native_journal import custody_lock
with custody_lock(sys.argv[3]): print('acquired')
'''
    with custody_lock(tmp_path):
        result=child(script,tmp_path)
        assert result.returncode and 'native_custody_has_active_writer' in result.stderr
    assert child(script,tmp_path).returncode==0


def test_process_exits_after_write_and_resume_never_replays_effect(tmp_path):
    script='''import os,time
from pathlib import Path
from compute_plane.native_journal import custody_lock,NativeJournal
from tare_tools_agent_runtime import (NativeSession,NativeBudget,ModelReply,ToolCall,
 BindingRef,CancellationToken,ExecutionControl)
p=Path(sys.argv[3])
class Model:
 def generate(self,request,control):return ModelReply(tool_calls=(ToolCall('edit-1','edit','{}'),))
class Tools:
 def execute(self,request,control):
  (p/'changed.txt').write_bytes(b'changed once');os._exit(17)
with custody_lock(p):
 NativeSession(session_id='s',instruction='do',capabilities=('edit',),model=Model(),tools=Tools(),
  journal=NativeJournal(p/'journal.jsonl'),budget=NativeBudget(max_model_calls=2)).run(
   ExecutionControl(BindingRef('w','b','o',1,'t'),CancellationToken(),time.monotonic()+10))
'''
    result=child(script,tmp_path);assert result.returncode==17,result.stderr
    class Never:
        def generate(self,*a):pytest.fail('model replayed')
        def execute(self,*a):pytest.fail('effect replayed')
    with custody_lock(tmp_path):
        session=NativeSession(session_id='s',instruction='do',capabilities=('edit',),model=Never(),tools=Never(),
            journal=NativeJournal(tmp_path/'journal.jsonl'),budget=NativeBudget(max_model_calls=2))
        outcome=session.run(ExecutionControl(BindingRef('w','b','o',1,'t'),CancellationToken(),time.monotonic()+10))
    assert outcome.status=='reconciliation_required' and outcome.pending_id=='s/effect/1'
    assert outcome.model_calls==1 and (tmp_path/'changed.txt').read_bytes()==b'changed once'
