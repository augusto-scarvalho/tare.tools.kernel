"""Snapshot-only native lookup and hash-bound expansion with real source bytes."""
from dataclasses import replace
import hashlib
import json
import time

import pytest

from compute_plane.native_capabilities import NativeCapabilities
from compute_plane.native_project_context import NativeProjectContext
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl, EffectRequest, ToolCall


@pytest.fixture
def project(tmp_path):
    from specgraph.indexer import index_repository
    root = tmp_path/'dependency'; (root/'src').mkdir(parents=True)
    (root/'src/lib.py').write_text('def leaf(): return 1\ndef caller(): return leaf()\n')
    state = tmp_path/'index'; index_repository(root, state_dir=state)
    scratch = tmp_path/'scratch'; scratch.mkdir()
    binding = BindingRef('w','b','o',1,'t')
    control = ExecutionControl(binding,CancellationToken(),time.monotonic()+30)
    provider = NativeProjectContext(repositories={'dep':str(root)}, work_id='w',
                                    state_dirs={'dep':state}, timeout_ms=1000)
    tools = NativeCapabilities(session_id='s',binding=binding,repositories={'dep':str(root)},
        writable={},verifiers={},scratch_root=scratch,project_context=provider,specgraph_edit_context=False)
    return root,state,provider,tools,control


def call(tools,control,name,**args):
    request = EffectRequest('s','effect',ToolCall('t',name,json.dumps(args)))
    return tools.execute(request,control)


def test_real_structure_without_specs_then_expand_without_write_authority(project):
    root,_,_,tools,control = project
    reply = call(tools,control,'context',repository='dep',query='caller')
    data = json.loads(reply.output_json)
    assert data['status'] == 'HIT' and reply.progressed
    assert data['coverage']['current_worktree'] == 'NOT_CHECKED' and not data['absence_proven']
    symbol = next(s for s in data['symbols'] if s['id']=='src/lib.py:caller')
    assert symbol['neighbors'] and not symbol['declared_specs']
    snippet = next(s for s in data['snippets'] if s['id']==symbol['id'])
    result = call(tools,control,'read',repository='dep',path='src/lib.py',sha256=snippet['sha256'],
                  start=symbol['range'][0],end=symbol['range'][1])
    assert result.status == 'succeeded' and 'return leaf()' in json.loads(result.output_json)['text']
    assert call(tools,control,'edit',repository='dep',path='src/lib.py',sha256=snippet['sha256'],
                search='return 1',replace='return 2').status == 'refused'
    assert (root/'src/lib.py').read_text().startswith('def leaf(): return 1')
    events = tools.observations()['context']
    assert [e['operation'] for e in events] == ['context','read']
    assert all(e['returned_bytes']>0 and e['duration_ms']>=0 and e['tokens'] is None for e in events)


def test_stale_pinned_expansion_does_not_return_replacement_text(project):
    root,_,_,tools,control = project
    target = root/'src/lib.py'; before = hashlib.sha256(target.read_bytes()).hexdigest()
    target.write_text('def changed(): return "new source"\n')
    result = call(tools,control,'read',repository='dep',path='src/lib.py',sha256=before)
    data = json.loads(result.output_json)
    assert result.status == 'failed' and data['status'] == 'STALE' and 'text' not in data
    assert data['current_sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_missing_snapshot_miss_wrong_root_and_cancel_are_distinct(project,tmp_path):
    root,state,provider,tools,control = project
    miss = call(tools,control,'context',repository='dep',query='neverexistszxq')
    assert json.loads(miss.output_json)['status']=='MISS' and miss.progressed  # a first miss is news
    assert not call(tools,control,'context',repository='dep',query='neverexistszxq').progressed
    assert call(tools,control,'context',repository='unselected',query='caller').status == 'refused'
    empty = tmp_path/'empty';empty.mkdir()
    other = NativeProjectContext(repositories={'dep':str(empty)},work_id='w',state_dirs={'dep':state})
    data = other.lookup('dep','caller',control)
    assert data['status']=='UNAVAILABLE' and data['root']==str(empty) and not data['absence_proven']
    listing = call(tools,control,'list',repository='dep',path='src')
    assert listing.status == 'succeeded' and json.loads(listing.output_json)['entries'] == ['lib.py']
    assert call(tools,control,'list',repository='dep',path='..').status == 'refused'
    stopped = replace(control,deadline_monotonic=time.monotonic()-1)
    assert provider.lookup('dep','caller',stopped)['status']=='CANCELLED'


def test_repeat_has_no_new_progress_and_native_edit_marks_snapshot_dirty(project):
    _,_,_,tools,control = project
    assert call(tools,control,'context',repository='dep',query='caller').progressed
    assert not call(tools,control,'context',repository='dep',query='caller').progressed
    tools._tools.written.add('dep:src/lib.py')
    result = json.loads(call(tools,control,'context',repository='dep',query='caller').output_json)
    assert result['coverage']['current_worktree']=='DIRTY_OBSERVED'


def test_unlimited_session_keeps_bounded_context_lookup(project, monkeypatch):
    _, _, provider, _, control = project
    seen = {}
    def lookup(*args, **kwargs):
        seen.update(kwargs)
        return {'status': 'MISS'}
    monkeypatch.setattr('specgraph.hook_context.context_hint', lookup)
    result = provider.lookup('dep', 'caller', replace(control, deadline_monotonic=float('inf')))
    assert result['status'] == 'MISS'
    assert seen['timeout_ms'] == provider.timeout_ms


def test_context_timeout_and_cancellation_after_query_are_visible(project,monkeypatch):
    _,_,provider,tools,control = project
    monkeypatch.setattr('specgraph.hook_context.context_hint',lambda *a,**kw:{'status':'TIMEOUT'})
    result = call(tools,control,'context',repository='dep',query='caller')
    assert result.status=='failed' and json.loads(result.output_json)['status']=='TIMEOUT'
    def cancel(*a,**kw):
        control.cancellation.cancel();return {'status':'HIT','snippets':['must not deliver after cancellation']}
    monkeypatch.setattr('specgraph.hook_context.context_hint',cancel)
    data = provider.lookup('dep','caller',control)
    assert data['status']=='CANCELLED' and 'snippets' not in data


def test_read_hash_comes_from_delivered_bytes_after_concurrent_change(project,monkeypatch):
    import io
    from pathlib import Path
    root,_,_,tools,control = project
    from compute_plane.native_journal import io_path
    original=Path.open
    reads=[]
    def changing_source(path,*args,**kwargs):
        if path==io_path(root/'src/lib.py') and args==('rb',):
            raw=b'first snapshot\n' if not reads else b'changed source\n'
            reads.append(raw);return io.BytesIO(raw)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',changing_source)
    result=json.loads(call(tools,control,'read',repository='dep',path='src/lib.py').output_json)
    assert reads==[b'first snapshot\n']
    assert result['sha256']==hashlib.sha256(reads[0]).hexdigest()
    assert 'first snapshot' in result['text']
