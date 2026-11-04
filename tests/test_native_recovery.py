"""Deterministic reproductions of the observed malformed-history failure."""
from dataclasses import replace
import json

import pytest

from compute_plane.native_models import NativeModelAdapter, DELIVERY_TOOL
from tare_tools_agent_runtime import ConversationLog, EffectResult, NativeBudget, NativeSession, ModelFailure
from tests.test_native_models import Client, SPECS, execution, profile, response, request


@pytest.mark.parametrize('arguments', [
    '{"limit":2200000000000000.0000000000000000E+2147483647000000}',
    '{"path":', '{"path":"a","path":"b"}', '["not-an-object"]',
])
def test_invalid_arguments_are_retained_but_not_replayed_or_executed(arguments):
    bad = response('openai-chat', usage={'prompt_tokens': 20, 'completion_tokens': 7})
    bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = arguments
    good = response('openai-chat', 'edit')
    rows, effects = [], []
    client = Client([bad, good, response('openai-chat', DELIVERY_TOOL)])
    adapter = NativeModelAdapter(profile('openai-chat'), client, system='Stable instructions',
                                 tool_specs_json=SPECS, observe=rows.append)
    class Tools:
        def execute(self, req, control):
            effects.append(req)
            return EffectResult(req.effect_id, 'succeeded', '{}', True)
    journal = ConversationLog()
    outcome = NativeSession(session_id='s', instruction='Fix app.py', capabilities=('read','edit','verify'),
        model=adapter, tools=Tools(), journal=journal, budget=NativeBudget(max_model_retries=0)).run(execution())
    assert outcome.status == 'proposed' and outcome.model_calls == 3
    assert len(effects) == 1 and effects[0].tool_call.name == 'edit'
    second = client.calls[1][1]
    assert second['messages'][:2] == client.calls[0][1]['messages'][:2]
    assert second['tools'] == client.calls[0][1]['tools']
    assert 'No tools from that response were executed' in second['messages'][-2]['content']
    assert second['messages'][-1]['content'].startswith('Runtime budget: 15 model calls remain')
    assert arguments not in json.dumps(second['messages'], ensure_ascii=False)
    saved = next(r.event() for r in journal.snapshot().records if r.event()['kind'] == 'model_returned')
    metadata = json.loads(saved['reply']['metadata_json'])
    assert metadata['assistant'] == bad['choices'][0]['message']
    assert metadata['usage'] == bad['usage'] == rows[0]['usage']
    assert rows[0]['status'] == 'recoverable_tool_arguments'


def test_invalid_member_refuses_entire_batch_before_a_valid_edit():
    bad = response('openai-chat', 'edit')
    calls = bad['choices'][0]['message']['tool_calls']
    calls.append({'id': 'bad', 'type': 'function', 'function': {'name': 'read', 'arguments': '{'}})
    adapter = NativeModelAdapter(profile('openai-chat'), Client([bad]), system='system', tool_specs_json=SPECS)
    reply = adapter.generate(request(), execution())
    assert reply.complete and not reply.tool_calls and not reply.proposes_delivery
    assert json.loads(reply.metadata_json)['recovery']['code'] == 'INVALID_TOOL_ARGUMENTS'



def test_a_call_cut_by_the_output_limit_asks_for_smaller_calls():
    cut = response('openai-chat', 'edit')
    cut['choices'][0]['message']['tool_calls'].append({'id': 'cut', 'type': 'function', 'function': {
        'name': 'malformed_tool_call', 'arguments': '{"intended_function":"edit","error":"The output ended"}'}})
    adapter = NativeModelAdapter(profile('openai-chat'), Client([cut]), system='system', tool_specs_json=SPECS)
    reply = adapter.generate(request(), execution())
    assert reply.complete and not reply.tool_calls and not reply.proposes_delivery
    recovery = json.loads(reply.metadata_json)['recovery']
    assert recovery['code'] == 'TRUNCATED_TOOL_CALL' and 'smaller calls' in recovery['message']

def test_repeated_bad_arguments_obey_existing_stagnation_control():
    bad = response('openai-chat')
    bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{'
    client = Client([bad, bad])
    adapter = NativeModelAdapter(profile('openai-chat'), client, system='system', tool_specs_json=SPECS)
    class NoTools:
        def execute(self, *args):
            pytest.fail('malformed responses cannot execute effects')
    outcome = NativeSession(session_id='s', instruction='task', capabilities=('read','edit','verify'),
        model=adapter, tools=NoTools(), journal=ConversationLog(),
        budget=NativeBudget(max_stagnant_turns=2)).run(execution())
    assert outcome.reason == 'NO_PROGRESS' and outcome.model_calls == 2


def test_http_500_is_not_misclassified_as_a_correctable_model_response():
    client = Client([ModelFailure('HTTP_500')])
    adapter = NativeModelAdapter(profile('openai-chat'), client, system='system', tool_specs_json=SPECS)
    with pytest.raises(ModelFailure, match='HTTP_500') as caught:
        adapter.generate(request(), execution())
    assert not caught.value.retryable and len(client.calls) == 1


def test_summary_cannot_recover_by_requesting_tool_execution():
    bad = response('openai-chat')
    bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{'
    adapter = NativeModelAdapter(profile('openai-chat'), Client([bad]), system='system', tool_specs_json=SPECS)
    with pytest.raises(ModelFailure, match='INVALID_PROVIDER_RESPONSE'):
        adapter.generate(replace(request(), purpose='summary'), execution())


def test_converse_recovery_preserves_usage_and_omits_invalid_tool_history():
    bad = response('bedrock-converse', args=['invalid-object'],
                   usage={'inputTokens':20, 'outputTokens':7})
    client = Client([bad, response('bedrock-converse', DELIVERY_TOOL)])
    rows = []
    adapter = NativeModelAdapter(profile('bedrock-converse'), client, system='system',
                                 tool_specs_json=SPECS, observe=rows.append)
    class NoTools:
        def execute(self, *args):
            pytest.fail('invalid arguments cannot execute effects')
    outcome = NativeSession(session_id='s', instruction='task', capabilities=('read','edit','verify'),
        model=adapter, tools=NoTools(), journal=ConversationLog()).run(execution())
    assert outcome.status == 'proposed' and outcome.model_calls == 2
    assert 'invalid-object' not in json.dumps(client.calls[1][1])
    assert rows[0]['usage'] == {'inputTokens':20, 'outputTokens':7}


def gated_session(monkeypatch, replies, verify_status='succeeded', claims=None, text_final=False):
    """edit/verify/deliver replies through a real session; returns (outcome, recoveries)."""
    import compute_plane.jev_choice_worker as worker
    if claims is None:
        monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    else:
        monkeypatch.setenv('TYPESAFE_API_KEY', 'k')
        monkeypatch.setattr(worker, 'assess', lambda request, **kw: {'choice': 'x', 'probabilities': {
            'success': claims, 'failure': 1 - claims}})
    answers = [response('openai-chat', name) for name in replies]
    if text_final:
        for name, answer in zip(replies, answers):
            if name == DELIVERY_TOOL:
                answer['choices'][0] = {'finish_reason':'stop', 'message':{
                    'role':'assistant','content':'Implemented and verified.'}}
    client = Client(answers)
    adapter = NativeModelAdapter(profile('openai-chat'), client, system='s', tool_specs_json=SPECS,
                                 verified_delivery=True, text_completes_turn=text_final)
    class Tools:
        def execute(self, req, control):
            status = verify_status if req.tool_call.name == 'verify' else 'succeeded'
            return EffectResult(req.effect_id, status, '{}', status == 'succeeded')
    journal = ConversationLog()
    outcome = NativeSession(session_id='s', instruction='Fix app.py', capabilities=('read', 'edit', 'verify'),
        model=adapter, tools=Tools(), journal=journal, budget=NativeBudget(max_model_retries=0)).run(execution())
    recoveries = [json.loads(r.event()['reply']['metadata_json']).get('recovery', {}).get('code')
                  for r in journal.snapshot().records if r.event()['kind'] == 'model_returned']
    return outcome, [code for code in recoveries if code]


def test_a_change_nobody_checked_cannot_be_delivered_until_the_verifier_runs(monkeypatch):
    outcome, recoveries = gated_session(monkeypatch, ['edit', DELIVERY_TOOL, 'verify', DELIVERY_TOOL])
    assert recoveries == ['UNVERIFIED_DELIVERY'] and outcome.status == 'proposed' and outcome.model_calls == 4
    outcome, recoveries = gated_session(monkeypatch, ['read', DELIVERY_TOOL])  # nothing changed: no check needed
    assert recoveries == [] and outcome.status == 'proposed'


def test_plain_text_completion_obeys_the_same_verification_guard(monkeypatch):
    outcome, recoveries = gated_session(monkeypatch, ['edit', DELIVERY_TOOL, 'verify', DELIVERY_TOOL], text_final=True)
    assert recoveries == ['UNVERIFIED_DELIVERY']
    assert outcome.status == 'proposed' and outcome.model_calls == 4


def test_a_failed_check_blocks_only_a_summary_that_claims_success(monkeypatch):
    replies = ['edit', 'verify', DELIVERY_TOOL, DELIVERY_TOOL]
    outcome, recoveries = gated_session(monkeypatch, replies, 'failed', claims=0.97)
    assert recoveries == ['FAILED_CHECK_DELIVERY', 'FAILED_CHECK_DELIVERY'] and outcome.status != 'proposed'
    outcome, recoveries = gated_session(monkeypatch, replies[:3], 'failed', claims=0.2)  # reports the failure
    assert recoveries == [] and outcome.status == 'proposed'
    outcome, recoveries = gated_session(monkeypatch, replies[:3], 'failed')  # no judge: facts only, fail open
    assert recoveries == [] and outcome.status == 'proposed'
