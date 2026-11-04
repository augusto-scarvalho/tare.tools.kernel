"""Corrective input preserves the wire prefix and closes only unstarted actions."""
from dataclasses import replace
import json
import pytest
from tare_tools_agent_runtime import CancellationToken, ConversationLog, NativeSession
from compute_plane.native_models import NativeModelAdapter, DELIVERY_TOOL, ModelProfile
from tests.test_native_models import profile, execution, response, Client, SPECS


@pytest.mark.parametrize('protocol',['openai-chat','bedrock-converse'])
def test_cancel_then_message_projects_no_execution_before_user_correction(protocol):
    cancel=CancellationToken()
    client=Client([response(protocol),response(protocol,DELIVERY_TOOL)])
    model=NativeModelAdapter(profile(protocol),client,system='Stable system.',tool_specs_json=SPECS)
    original=model.generate
    def generate(request,control):
        reply=original(request,control)
        if len(client.calls)==1:cancel.cancel()
        return reply
    model.generate=generate
    class Tools:
        def execute(self,*args):pytest.fail('Cancelled action must not execute')
    journal=ConversationLog()
    run=NativeSession(session_id='s',instruction='Original task',capabilities=('read','edit','verify'),model=model,tools=Tools(),journal=journal)
    assert run.run(replace(execution(),cancellation=cancel)).status=='cancelled'
    first=client.calls[0][1]
    run.continue_with(execution(),message_id='correction',text='Use the search tools instead.',interrupts_turn=True)
    assert run.run(execution()).status=='proposed'
    second=client.calls[-1][1]
    assert second['messages'][:len(first['messages'])]==first['messages']
    assert 'CANCELLED_BEFORE_EXECUTION' in json.dumps(second)
    assert second.get('system')==first.get('system')
    assert second.get('tools')==first.get('tools')
    assert not any(r.event()['kind']=='effect_requested' for r in journal.snapshot().records)


@pytest.mark.parametrize('name',['claude-opus-5-5','claude-sonnet-5-5','claude-haiku-5-5','claude-fable-5-1'])
def test_claude_decisions_use_native_tool_calls_before_dispatch(tmp_path,name):
    from compute_plane.native_cli import VendorCliClient
    from tests.test_native_cli import request
    client=VendorCliClient('cli://claude',executable=str(tmp_path/'claude.exe'),version='2.1.293')
    model=NativeModelAdapter(ModelProfile('vendor-cli',name,32768,4096,'fixture',client.endpoint),
        client,system='Review the requested change.',tool_specs_json=SPECS,counter_protocol='utf8-envelope-estimate-v1')
    wire=model.wire(request())
    assert 'response_schema' not in wire and 'reminder' not in wire
    assert {t['name'] for t in wire['native_tools']} >= {'read','edit','verify'}
    assert 'tools' not in json.loads(wire['prompt'])
