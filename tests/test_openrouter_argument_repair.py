"""Serialize malformed-argument repair without changing history/cache identity."""
from dataclasses import replace
import json

import pytest

from compute_plane.native_models import NativeModelAdapter
from tare_tools_agent_runtime import ConversationLog, EffectResult, NativeSession
from tests.test_native_models import Client, SPECS, execution, profile, response


@pytest.mark.parametrize('endpoint,reason,expected',[
    ('https://openrouter.ai/api/v1','capability_arguments_mismatch: missing path',False),
    ('https://openrouter.ai/api/v1','permission denied',None),
    ('http://fixture','capability_arguments_mismatch: missing path',None),
])
def test_only_observed_argument_refusal_enables_openrouter_serial_repair(endpoint,reason,expected):
    raw=response('openai-chat')
    raw['choices'][0]['message']['tool_calls'][0]['function']['arguments']='{}'
    client=Client([raw,response('openai-chat','tare_deliver')])
    client.endpoint=endpoint
    adapter=NativeModelAdapter(replace(profile('openai-chat'),endpoint=endpoint),client,
        system='system',tool_specs_json=SPECS)
    identity=adapter.profile_digest
    class Tools:
        def execute(self,request,_):
            return EffectResult(request.effect_id,'refused',json.dumps({'reason':reason}))
    session=NativeSession(session_id='s',instruction='Inspect app.py',capabilities=('read','edit','verify'),
        model=adapter,tools=Tools(),journal=ConversationLog())
    assert session.run(execution()).status=='proposed'
    calls=[body for path,body in client.calls if path=='/chat/completions']
    assert len(calls)==2
    assert 'parallel_tool_calls' not in calls[0]
    assert calls[1].get('parallel_tool_calls') is expected
    assert calls[0]['messages'][:2]==calls[1]['messages'][:2]
    assert calls[0]['tools']==calls[1]['tools']
    assert adapter.profile_digest==identity
    if endpoint=='https://openrouter.ai/api/v1':
        assert calls[0]['session_id']==calls[1]['session_id']


def test_explicit_parallel_preference_is_not_overridden():
    endpoint='https://openrouter.ai/api/v1'
    raw=response('openai-chat')
    client=Client([raw,response('openai-chat','tare_deliver')]);client.endpoint=endpoint
    selected=replace(profile('openai-chat'),endpoint=endpoint,parameters_json='{"parallel_tool_calls":true}')
    adapter=NativeModelAdapter(selected,client,system='system',tool_specs_json=SPECS)
    class Tools:
        def execute(self,request,_):
            return EffectResult(request.effect_id,'refused',json.dumps({
                'reason':'capability_arguments_mismatch: missing path'}))
    session=NativeSession(session_id='s',instruction='Inspect app.py',capabilities=('read','edit','verify'),
        model=adapter,tools=Tools(),journal=ConversationLog())
    assert session.run(execution()).status=='proposed'
    assert all(body['parallel_tool_calls'] is True for path,body in client.calls if path=='/chat/completions')
