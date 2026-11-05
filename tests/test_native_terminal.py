import json
import os
from dataclasses import asdict
from types import SimpleNamespace
import sys
import time

import pytest
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl, EffectRequest, ToolCall

from compute_plane.native_capabilities import NativeCapabilities
from compute_plane.native_terminal import NativeTerminal, TERMINAL_SPECS
from compute_plane.terminal_host import TerminalClient, TerminalHost, TerminalTransportError


@pytest.fixture
def setup(tmp_path):
    repo, scratch = tmp_path/'project', tmp_path/'scratch'
    repo.mkdir(); scratch.mkdir()
    (repo/'a.py').write_text('value = 1\n')
    binding = BindingRef('work','binding','owner',1,'token')
    control = ExecutionControl(binding,CancellationToken(),time.monotonic()+30)
    with TerminalHost() as host:
        client = TerminalClient(host.bind(owner='test',repositories={'project':str(repo)},env=dict(os.environ)))
        terminal = NativeTerminal(session_id='s',binding=binding,client=client)
        options = dict(session_id='s',binding=binding,repositories={'project':str(repo)},
            writable={'project':['.']},verifiers={'check':{'repository':'project','argv':[sys.executable,'-c','pass']}},
            scratch_root=scratch,specgraph_edit_context=False,terminal=terminal,
            capabilities=['read','list','edit','verify',*TERMINAL_SPECS])
        yield repo, host, client, NativeCapabilities(**options), control, options


def call(tools, control, name, identity='e', **args):
    return tools.execute(EffectRequest('s',identity,ToolCall(identity,name,json.dumps(args))),control)


def completed(tools, control, handle):
    deadline = time.monotonic()+5
    while time.monotonic()<deadline:
        result = call(tools,control,'terminal_read',handle=handle,wait_ms=200)
        if json.loads(result.output_json).get('cleanup_verified'):
            return result
    pytest.fail('command did not complete')


@pytest.mark.parametrize('ending',['\n','\r\n'])
def test_output_continuation_copies_real_character_cursors_and_stops_at_observed_end(setup, ending):
    _,_,_,tools,control,_=setup
    expected='é🙂0123456789'*1600+ending+'FIM'+ending
    error='erro ç'+ending
    started=call(tools,control,'terminal_start',repository='project',argv=[sys.executable,'-I','-X','utf8','-u','-c',
        'import sys;sys.stdout.reconfigure(newline="");sys.stderr.reconfigure(newline="");sys.stdout.write('+repr(expected)+');sys.stderr.write('+repr(error)+')'])
    handle=json.loads(started.output_json)['handle']
    args=dict(handle=handle,limit=4096,wait_ms=2000,wait_for_exit=True)
    stdout=[];stderr=[];pages=0
    while args is not None:
        result=call(tools,control,'terminal_read',**args)
        assert result.status=='succeeded'
        data=json.loads(result.output_json);pages+=1
        assert pages<=8
        stdout.append(data['output']['stdout']['text']);stderr.append(data['output']['stderr']['text'])
        args=data['next_read']
        if args is not None:
            assert args['stdout_cursor']==data['output']['stdout']['cursor']
            assert args['stderr_cursor']==data['output']['stderr']['cursor']
            assert all(args[s+'_cursor']<=data['output'][s]['end'] for s in ('stdout','stderr'))
        else:
            assert data['status']=='exited' and data['exit_code']==0 and data['cleanup_verified']
            assert all(not p['more'] for p in data['output'].values())
    assert pages>1 and ''.join(stdout)==expected and ''.join(stderr)==error
    invalid=call(tools,control,'terminal_read',handle=handle,stdout_cursor=len(expected)+1,stderr_cursor=len(error))
    assert invalid.status=='refused' and json.loads(invalid.output_json)['reason']=='terminal_cursor_invalid'


def test_changed_continuation_options_do_not_invent_process_progress(setup):
    _,_,_,tools,control,_=setup
    class IdleClient:
        def request(self,operation,**kwargs):
            stream=dict(text='',cursor=0,end=0,more=False,truncated=False)
            return dict(status='running',output={'stdout':stream,'stderr':dict(stream)})
    tools.terminal=NativeTerminal(session_id='s',binding=control.binding,client=IdleClient())
    first=call(tools,control,'terminal_read',handle='idle',limit=4,wait_ms=0)
    second=call(tools,control,'terminal_read',handle='idle',limit=8,wait_ms=0)
    assert first.progressed and not second.progressed
    assert json.loads(first.output_json)['next_read']!=json.loads(second.output_json)['next_read']


def test_native_tools_run_command_and_preserve_output_and_nonzero_exit(setup):
    _,_,_,tools,control,_=setup
    specs=json.loads(tools.tool_specs_json)
    assert specs['terminal_start']['parameters']['properties']['repository']['enum']==['project']
    started=call(tools,control,'terminal_start',repository='project',argv=[sys.executable,'-u','-c','print("actual output"); raise SystemExit(3)'])
    handle=json.loads(started.output_json)['handle']
    result=completed(tools,control,handle)
    assert result.status=='failed'
    assert 'actual output' in json.loads(result.output_json)['output']['stdout']['text']
    listed=call(tools,control,'terminal_list',include_completed=True)
    assert json.loads(listed.output_json)['processes'][0]['handle']==handle


def test_read_wait_is_not_process_timeout_and_never_restarts(setup):
    _,_,_,tools,control,_ = setup
    started = call(tools,control,'terminal_start',argv=[sys.executable,'-u','-c',
        'import time; print("ONCE",flush=True); time.sleep(1.5); print("DONE")'],repository='project',timeout_seconds=5)
    handle = json.loads(started.output_json)['handle']
    early = call(tools,control,'terminal_read',handle=handle,wait_ms=100,wait_for_exit=True)
    value = json.loads(early.output_json)
    assert value['status']=='running' and value['read_wait_expired']
    assert not value['process_timed_out'] and value['timeout_seconds']==5
    assert 'Do not restart' in value['next_action']
    done = call(tools,control,'terminal_read',handle=handle,wait_ms=4000,wait_for_exit=True)
    value = json.loads(done.output_json)
    assert done.status=='succeeded' and value['exit_code']==0 and value['cleanup_verified']
    assert value['output']['stdout']['text'].count('ONCE')==1
    assert 'DONE' in value['output']['stdout']['text']


def test_process_timeout_preserves_partial_output_and_failed_result(setup):
    _,_,_,tools,control,_ = setup
    started = call(tools,control,'terminal_start',argv=[sys.executable,'-u','-c',
        'import time; print("PARTIAL",flush=True); time.sleep(20)'],repository='project',timeout_seconds=1)
    result = call(tools,control,'terminal_read',handle=json.loads(started.output_json)['handle'],
                  wait_for_exit=True,wait_ms=4000)
    value = json.loads(result.output_json)
    assert result.status=='failed' and value['process_timed_out'] and value['cleanup_verified']
    assert value['reason']=='timeout' and 'PARTIAL' in value['output']['stdout']['text']


def test_commands_invalidate_previous_reads_and_verifications(setup):
    repo,_,_,tools,control,options=setup
    read=call(tools,control,'read',identity='read',repository='project',path='a.py')
    verified=tools.execute(EffectRequest('s','verify',ToolCall('v','verify','{"name":"check"}')),control)
    assert verified.status=='succeeded' and tools.observations()['verifications']
    args=dict(repository='project',argv=[sys.executable,'-c','from pathlib import Path; Path("a.py").write_text("value = 2\\n")'])
    started=call(tools,control,'terminal_start',identity='start',**args)
    assert not tools.observations()['reads'] and not tools.observations()['verifications']
    completed(tools,control,json.loads(started.output_json)['handle'])
    assert (repo/'a.py').read_text()=='value = 2\n'
    # Recovery must not resurrect a read from before an authorized command.
    events=[{'kind':'effect_requested','effect_id':'read','tool_call':{'name':'read','arguments_json':json.dumps({'repository':'project','path':'a.py'})}},
            {'kind':'effect_returned','result':asdict(read)},
            {'kind':'effect_requested','effect_id':'start','tool_call':{'name':'terminal_start','arguments_json':json.dumps(args)}},
            {'kind':'effect_returned','result':asdict(started)}]
    records=[SimpleNamespace(sequence=i,event=lambda e=e:e) for i,e in enumerate(events)]
    restored=NativeCapabilities(**options)
    assert restored.restore(SimpleNamespace(snapshot=lambda:SimpleNamespace(records=records)))==[]
    assert not restored.observations()['reads']
    assert call(restored,control,'read',repository='project',path='a.py').status=='succeeded'


def test_terminal_is_explicit_and_for_implementer_only(setup):
    _,_,_,tools,control,options=setup
    ordinary=NativeCapabilities(**{**options,'terminal':None,'capabilities':['read','list']})
    assert 'terminal_start' not in json.loads(ordinary.tool_specs_json)
    assert call(ordinary,control,'terminal_list').status=='refused'
    for role in ('planner','plan-auditor','auditor','explorer'):
        with pytest.raises(ValueError,match='cannot_execute_terminal'):
            NativeCapabilities(**{**options,'role':role,'writable':{},'verifiers':{}})
    terminal_only=NativeCapabilities(**{**options,'capabilities':['terminal_list']})
    assert set(json.loads(terminal_only.tool_specs_json))=={'terminal_list'}
    assert call(terminal_only,control,'read',repository='project',path='a.py').status=='refused'


def test_control_refusal_happens_before_command_effect(setup):
    _,_,client,tools,control,_=setup
    control.cancellation.cancel()
    result=call(tools,control,'terminal_start',repository='project',argv=[sys.executable,'-c','raise AssertionError'])
    assert result.status=='refused' and client.request('list',include_completed=True)==[]


@pytest.mark.parametrize('args', [{'repository':'project','argv':'bad'},
    {'repository':'project','argv':[sys.executable],'owner':'another'},
    {'repository':'project','argv':[sys.executable],'timeout_seconds':True},
    {'repository':'project','argv':[sys.executable],'mode':'unknown'}])
def test_tool_arguments_are_validated_before_dispatch(setup,args):
    _,_,client,tools,control,_=setup
    assert call(tools,control,'terminal_start',**args).status=='refused'
    assert client.request('list',include_completed=True)==[]


@pytest.mark.parametrize('operation',['terminal_start','terminal_resize'])
def test_transport_loss_after_mutation_is_unknown_and_never_retried(setup,operation):
    _,_,client,tools,control,options=setup
    class LostResponse:
        calls=0
        def request(self, operation, **kwargs):
            self.calls+=1
            client.request(operation,**kwargs)
            raise TerminalTransportError('injected lost response')
    lost=LostResponse()
    args = dict(repository='project',argv=[sys.executable,'-c','import time; time.sleep(60)'])
    if operation == 'terminal_resize':
        started = client.request('start',request_id='prepare',mode='pty',**args)
        args = dict(handle=started['handle'],columns=73,rows=21)
    tools.terminal=NativeTerminal(session_id='s',binding=control.binding,client=lost)
    result=call(tools,control,operation,**args)
    assert result.status=='unknown' and lost.calls==1
    assert json.loads(result.output_json)['automatic_retry'] is False
    assert len(client.request('list'))==1
    if operation == 'terminal_resize':
        observed = client.request('read',handle=args['handle'])
        assert observed['columns'] == 73 and observed['rows'] == 21


def test_repeated_poll_is_not_progress_but_bounded_wait_is(setup):
    _,_,_,tools,control,_=setup
    started=call(tools,control,'terminal_start',repository='project',argv=[sys.executable,'-c','import time; time.sleep(60)'])
    handle=json.loads(started.output_json)['handle']
    call(tools,control,'terminal_read',handle=handle,wait_ms=0)
    assert not call(tools,control,'terminal_read',handle=handle,wait_ms=0).progressed
    waited=call(tools,control,'terminal_read',handle=handle,wait_ms=200)
    assert waited.progressed and json.loads(waited.output_json)['waited_ms']>=100
    assert call(tools,control,'terminal_stop',handle=handle).status=='succeeded'


def test_wait_for_exit_waits_past_progress_and_preserves_failure(setup):
    _,_,_,tools,control,_=setup
    started=call(tools,control,'terminal_start',repository='project',argv=[sys.executable,'-u','-c','import time; print("progress",flush=True); time.sleep(.2); raise SystemExit(7)'])
    handle=json.loads(started.output_json)['handle']
    result=call(tools,control,'terminal_read',handle=handle,wait_ms=2000,wait_for_exit=True)
    payload=json.loads(result.output_json)
    assert result.status=='failed' and payload['status']=='exited' and payload['exit_code']==7
    assert 'progress' in payload['output']['stdout']['text']


def test_default_read_returns_on_first_output(setup):
    _,_,_,tools,control,_=setup
    # The child outlives the read by far (terminal_stop ends it): a loaded runner that starts the read late must
    # still find it running.
    started=call(tools,control,'terminal_start',repository='project',argv=[sys.executable,'-u','-c',
        'import time; print("progress",flush=True); time.sleep(60)'])
    handle=json.loads(started.output_json)['handle']
    result=call(tools,control,'terminal_read',handle=handle,wait_ms=2000)
    payload=json.loads(result.output_json)
    assert result.status=='succeeded' and payload['status']=='running'
    assert 'progress' in payload['output']['stdout']['text']
    assert call(tools,control,'terminal_stop',handle=handle).status=='succeeded'


@pytest.mark.parametrize('flag', ['more','truncated','full'])
def test_wait_for_exit_returns_bounded_output_page(setup,flag):
    _,_,_,tools,control,_=setup
    class PageClient:
        calls=0
        def request(self, operation, **kwargs):
            self.calls+=1
            limit=kwargs['limit']
            stream=dict(text='x'*limit if flag=='full' else 'x', cursor=limit, end=limit,
                        more=flag=='more', truncated=flag=='truncated')
            return dict(status='running',output={'stdout':stream,'stderr':dict(stream, text='')})
    client=PageClient()
    tools.terminal=NativeTerminal(session_id='s',binding=control.binding,client=client)
    result=call(tools,control,'terminal_read',handle='page',wait_ms=1000,wait_for_exit=True,limit=4)
    assert result.status=='succeeded' and client.calls==1
    assert json.loads(result.output_json)['output']['stdout']['text'].startswith('x')


def test_wait_for_exit_has_finite_deadline_despite_progress(setup,monkeypatch):
    from compute_plane import native_terminal
    _,_,_,tools,control,_=setup
    clock=SimpleNamespace(now=100.0,sleeps=[])
    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now+=seconds
    # Test the deadline rather than the runner's scheduling latency. Patch only
    # this owner's clock reference, leaving real host/control threads untouched.
    monkeypatch.setattr(native_terminal,'time',SimpleNamespace(monotonic=lambda:clock.now,sleep=sleep))
    class ProgressClient:
        calls=0
        def request(self, operation, **kwargs):
            self.calls+=1
            stream=dict(text='x',cursor=1,end=1,more=False,truncated=False)
            return dict(status='running',output={'stdout':stream,'stderr':dict(stream,text='')})
    client=ProgressClient()
    tools.terminal=NativeTerminal(session_id='s',binding=control.binding,client=client)
    began=clock.now
    result=call(tools,control,'terminal_read',handle='progress',wait_ms=150,wait_for_exit=True)
    elapsed=clock.now-began
    assert result.status=='succeeded' and 2 <= client.calls <= 5
    assert elapsed==pytest.approx(.15)
    assert clock.sleeps and all(0 <= seconds <= .05 for seconds in clock.sleeps)
    assert json.loads(result.output_json)['status']=='running'


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('value,expected', [(True, True), (False, False), ('true', True), ('false', False)])
def test_terminal_boolean_arguments_match_native_values(setup, direct, value, expected):
    _, _, _, tools, control, _ = setup
    dispatched = []
    class Client:
        def request(self, operation, **args):
            dispatched.append((operation, args))
            if operation == 'list':
                assert args['include_completed'] is expected
                return []
            stream = dict(text='progress', cursor=8, end=8, more=False, truncated=False)
            return dict(status='running' if len(dispatched) == 1 else 'exited', exit_code=0,
                        output={'stdout': stream, 'stderr': dict(stream)})
    tools.terminal = NativeTerminal(session_id='s', binding=control.binding, client=Client())
    port = tools.terminal if direct else tools
    result = call(port, control, 'terminal_read', handle='h', wait_for_exit=value, wait_ms=1000)
    assert result.status == 'succeeded'
    assert len(dispatched) == (2 if expected else 1)
    assert json.loads(result.output_json)['status'] == ('exited' if expected else 'running')
    assert call(port, control, 'terminal_list', include_completed=value).status == 'succeeded'


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('name,key', [('terminal_read', 'wait_for_exit'), ('terminal_list', 'include_completed')])
@pytest.mark.parametrize('value', ['TRUE', 'False', ' true', '', 1, None, []])
def test_invalid_terminal_booleans_refused_before_dispatch(setup, direct, name, key, value):
    _, _, _, tools, control, _ = setup
    class Client:
        def request(self, *args, **kwargs):
            pytest.fail('invalid arguments must not dispatch')
    tools.terminal = NativeTerminal(session_id='s', binding=control.binding, client=Client())
    args = {'handle': 'h'} if name == 'terminal_read' else {}
    result = call(tools.terminal if direct else tools, control, name, **args, **{key: value})
    assert result.status == 'refused'
    assert json.loads(result.output_json)['reason'] == 'terminal_tool_argument_type'
