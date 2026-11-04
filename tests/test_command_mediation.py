"""Actual command boundaries and durable claims, with explicitly fake assessment ports."""
import json
from pathlib import Path

import pytest
from compute_plane.action_control import ActionAuthorization, ActionRefused, ActionSnapshot
from compute_plane.mediated_tools import MediatedTools
from compute_plane.native_effect_recovery import EffectReceipts
from compute_plane.terminal_host import TerminalHost, TerminalClient, TerminalRefused
from tare_tools_agent_runtime import EffectRequest, EffectResult, ToolCall


def test_command_denial_prevents_process_start_and_before_effect(tmp_path, monkeypatch):
    effects=[]
    class Deny:
        def execute(self, facts, effect):
            assert facts.argv==('python','fixture.py') and facts.cwd==str(tmp_path)
            assert len(facts.environment_sha256)==len(facts.grant_sha256)==64
            raise ActionRefused('fixture denial')
    monkeypatch.setattr('compute_plane.managed_process.run_process',lambda *a,**kw:pytest.fail('process started'))
    tools=MediatedTools({'repositories':{'r':str(tmp_path)},'role':'implementer','writable':{'r':['.']}},
        scratch_root=tmp_path,before_effect=lambda:effects.append('started'),command_mediation=Deny())
    with pytest.raises(ActionRefused):tools.command(['python','fixture.py'],tmp_path,gate=False)
    assert not effects


def test_named_verifier_can_use_its_own_bound_while_raw_commands_stay_short(tmp_path,monkeypatch):
    timeouts=[]
    def process(argv,*,dest,timeout,**kwargs):
        timeouts.append(timeout)
        (Path(dest)/'stdout.txt').write_text('ok')
        (Path(dest)/'stderr.txt').write_text('')
        return {'exit_code':0,'cleanup_verified':True,'cancelled':False,'timeout':False,'output_limit':False}
    monkeypatch.setattr('compute_plane.managed_process.run_process',process)
    tools=MediatedTools({'repositories':{'r':str(tmp_path)},'role':'implementer','writable':{'r':['.']},
        'operation_timeout':500,'verifiers':{'check':{'repository':'r','argv':['fixture']}}},scratch_root=tmp_path)
    tools.command_seconds=30;tools.verification_seconds=300
    tools.perform('verify',{'name':'check'})
    tools.command(['fixture'],tmp_path,gate=False)
    tools.command(['fixture'],tmp_path,timeout=5,gate=False,verification=True)
    assert timeouts==[300,30,5]


def test_command_executes_frozen_argv_environment_and_timeout(tmp_path, monkeypatch):
    original=['python','fixture.py']; current={'EXACT':'before'}; seen=[]
    monkeypatch.setattr('compute_plane.mediated_tools.isolated_child_environment',lambda:dict(current))
    class Port:
        def execute(self,facts,effect):
            assert json.loads(facts.payload_json)['timeout']==2
            original.append('changed');current['EXACT']='after'
            return effect()
    def process(argv,**kwargs):
        seen.append((argv,kwargs['env'],kwargs['timeout']))
        for name in ('stdout.txt','stderr.txt'):(Path(kwargs['dest'])/name).write_bytes(b'')
        return dict(cleanup_verified=True,cancelled=False,timeout=False,output_limit=False,exit_code=0)
    monkeypatch.setattr('compute_plane.managed_process.run_process',process)
    tools=MediatedTools({'repositories':{'r':str(tmp_path)},'role':'implementer','writable':{'r':['.']}},
                       scratch_root=tmp_path,command_mediation=Port())
    tools.command(original,tmp_path,timeout=2,gate=False)
    assert seen==[(['python','fixture.py'],{'EXACT':'before'},2)]


def test_terminal_start_and_input_are_distinct_mediated_effects_and_stop_remains_available(tmp_path):
    calls=[]; facts=[]
    class Sessions:
        def start(self,**kwargs):calls.append(('start',kwargs));return {'handle':'owned','status':'running'}
        def send(self,**kwargs):calls.append(('send',kwargs));return {'queued':True}
        def stop(self,**kwargs):calls.append(('stop',kwargs));return {'cleanup_verified':True}
        def close(self):pass
    class Port:
        def execute(self,request,effect):
            facts.append(request)
            if request.operation=='terminal_input':raise ActionRefused('input refused separately')
            return effect()
    with TerminalHost(sessions=Sessions()) as host:
        descriptor=host.bind(owner='operator',repositories={'r':str(tmp_path)},env={'X':'y'},command_mediation=Port())
        client=TerminalClient(descriptor)
        assert client.request('start',repository='r',request_id='first',argv=['python','fixture.py'])['handle']=='owned'
        with pytest.raises(TerminalRefused):client.request('send',handle='owned',request_id='input',text='execute something')
        assert client.request('stop',handle='owned')['cleanup_verified']
    assert [call[0] for call in calls]==['start','stop']
    assert [item.operation for item in facts]==['terminal_start','terminal_input']
    assert json.loads(facts[1].payload_json)['text']=='execute something'
    assert json.loads(facts[0].payload_json)['mode']=='pipes'


def test_claim_survives_recorder_restart_and_is_retained_with_the_result(tmp_path):
    request=EffectRequest('s','effect',ToolCall('call','verify','{"name":"check"}'))
    action=ActionSnapshot('w','s','effect','binding',3,'verify','r','/fixture',*(['a'*64]*5),'mediation-only',200.0)
    permit=ActionAuthorization('host','nonce',action.action_sha256,'s','effect','binding',3,190.0)
    recorder=EffectReceipts(tmp_path,'s',{'role':'fixture'})
    assert recorder.claim_action(request,action,permit)
    assert not EffectReceipts(tmp_path,'s',{'role':'fixture'}).claim_action(request,action,permit)
    recorder.save(request,result={'effect_id':'effect','status':'unknown','output_json':'{}','progressed':False})
    rows=list(recorder.root.glob('*.json'))
    assert len(rows)==1 and json.loads(rows[0].read_text())['action_control']['action_sha256']==action.action_sha256
    assert '"nonce"' not in rows[0].read_text()
    assert not recorder.claim_action(request,action,permit)
