"""Keep uncertain effects fenced, then render only the owner's known resolution."""
from dataclasses import replace
import json

import pytest

from compute_plane.native_effect_recovery import pending
from compute_plane.native_models import DELIVERY_TOOL, NativeModelAdapter
from tare_tools_agent_runtime import ConversationLog, EffectResult, NativeContext, NativeSession
from tests.test_native_models import Client, SPECS, execution, profile, request, response
from tests.test_native_responses import Client as ResponsesClient, adapter as responses_adapter, response as responses_reply


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse', 'openai-codex-responses'])
@pytest.mark.parametrize('resolved_status', ['failed', 'succeeded'])
def test_owner_reconciled_unknown_result_is_rendered_once_without_reexecuting(protocol, resolved_status):
    if protocol == 'openai-codex-responses':
        client = ResponsesClient([responses_reply('verify'), responses_reply(DELIVERY_TOOL)])
        adapter = responses_adapter(client)
    else:
        client = Client([response(protocol, 'verify'), response(protocol, DELIVERY_TOOL)])
        adapter = NativeModelAdapter(profile(protocol), client, system='Stable instructions', tool_specs_json=SPECS)
    class Tools:
        calls = 0
        def execute(self, effect, control):
            self.calls += 1
            return EffectResult(effect.effect_id, 'unknown', '{"reason":"interrupted"}')
    tools, journal = Tools(), ConversationLog()
    session = NativeSession(session_id='s', instruction='Verify and deliver.',
        capabilities=('read', 'edit', 'verify'), model=adapter, tools=tools, journal=journal)
    first = session.run(execution())
    assert first.status == 'reconciliation_required' and len(client.calls) == tools.calls == 1
    req = replace(request(), instruction='Verify and deliver.',
                  context=NativeContext.project(journal))
    # No owner proof: no provider request, even if another user message is offered.
    with pytest.raises(ValueError, match='unmatched tool|pending tool'):
        adapter.wire(req)
    attempted_bypass = ConversationLog()
    for record in journal.snapshot().records:
        attempted_bypass.append(record.event())
    attempted_bypass.append({'kind': 'user_message', 'text': 'Continue anyway',
                             'message_id': 'cannot-bypass', 'after_sequence': 0})
    with pytest.raises(ValueError, match='user input cannot bypass a pending tool'):
        adapter.wire(replace(req, context=NativeContext.project(attempted_bypass)))
    assert len(client.calls) == 1
    effect = pending(journal, 's')
    assert effect is not None
    original = journal.snapshot().records
    result = EffectResult(effect.effect_id, resolved_status,
                          json.dumps({'returncode': 0 if resolved_status == 'succeeded' else None,
                                      'recovery': 'Explicit owner proof'}))
    assert session.reconcile_effect(execution(), effect, result)
    second = session.run(execution())
    assert second.status == 'proposed'
    assert second.model_calls == 2 and len(client.calls) == 2 and tools.calls == 1
    # Unknown remains in custody; model history contains only the known resolution.
    assert [r.sha256 for r in journal.snapshot().records[:len(original)]] == [r.sha256 for r in original]
    wire = json.dumps(client.calls[-1][1])
    assert 'Explicit owner proof' in wire and 'interrupted' not in wire
    assert wire.count('Explicit owner proof') == 1
    assert client.calls[0][1].get('instructions', client.calls[0][1].get('system')) == client.calls[-1][1].get('instructions', client.calls[-1][1].get('system'))
    if protocol == 'openai-codex-responses':
        assert client.calls[0][1]['prompt_cache_key'] == client.calls[-1][1]['prompt_cache_key']
    duplicate = ConversationLog()
    for record in journal.snapshot().records:
        duplicate.append(record.event())
    duplicate.append({'kind': 'effect_returned', 'result': {
        'effect_id': result.effect_id, 'status': result.status,
        'output_json': result.output_json, 'progressed': result.progressed}})
    with pytest.raises(ValueError, match='unmatched tool observation'):
        adapter.wire(replace(req, context=NativeContext.project(duplicate)))
