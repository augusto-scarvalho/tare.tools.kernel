"""Ordinary turns and a follow-up retain every earlier message and tool result."""
import copy
import json

import pytest
from tare_tools_agent_runtime import ConversationLog, EffectResult, NativeSession

from compute_plane.native_models import ModelProfile, NativeModelAdapter, DELIVERY_TOOL
from tests.test_native_models import Client, SPECS, execution, profile, response


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse', 'codex', 'claude', 'agy'])
def test_complete_prefix_survives_tools_delivery_and_user_follow_up(protocol):
    cli = protocol in {'codex', 'claude', 'agy'}
    base = 'openai-chat' if cli else protocol
    client = Client([response(base, name) for name in ('read', 'edit', 'verify', DELIVERY_TOOL, DELIVERY_TOOL)])
    selected = profile(base)
    options = {}
    if cli:
        client.endpoint = 'cli://' + protocol
        client.version = '2.1.293' if protocol == 'claude' else '0.154.0'
        selected = ModelProfile('vendor-cli', selected.model, 32768, 4096, 'fixture', client.endpoint)
        options['counter_protocol'] = 'utf8-envelope-estimate-v1'
    model = NativeModelAdapter(selected, client, system='Stable role instructions', tool_specs_json=SPECS, **options)

    class Tools:
        def execute(self, request, control):
            return EffectResult(request.effect_id, 'succeeded', '{"observed":"original source"}', True)

    journal = ConversationLog()
    session = NativeSession(session_id='stable', instruction='Implement only the selected change.',
        capabilities=('read','edit','verify'), model=model, tools=Tools(), journal=journal)
    assert session.run(execution()).status == 'proposed'
    before = copy.deepcopy(journal.snapshot())
    assert session.continue_with(execution(), message_id='delta', text='Now expand only this criterion.')
    assert session.run(execution()).status == 'proposed'
    assert len(client.calls) == 5
    assert journal.snapshot().records[:len(before.records)] == before.records
    for (_, first), (_, second) in zip(client.calls, client.calls[1:]):
        if cli:
            # Claude declares its catalog natively; the other CLIs quote it beside a response schema.
            assert first['system'] == second['system']
            assert first.get('response_schema') == second.get('response_schema')
            assert first.get('native_tools') == second.get('native_tools')
            one, two = json.loads(first['prompt']), json.loads(second['prompt'])
            assert one.get('action_catalog') == two.get('action_catalog')
            key = 'messages' if 'native_tools' in first else 'conversation_records'
            first_messages, second_messages = one[key], two[key]
        else:
            if protocol == 'bedrock-converse':
                assert first['system'] == second['system']
                assert first['toolConfig'] == second['toolConfig']
            else:
                assert first['tools'] == second['tools']
            first_messages, second_messages = first['messages'], second['messages']
        assert second_messages[:len(first_messages)] == first_messages
