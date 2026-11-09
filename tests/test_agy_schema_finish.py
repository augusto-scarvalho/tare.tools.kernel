"""AGY 1.4.10 can render schema completion as finish without a tool hook."""
import json

import pytest

from compute_plane.native_cli import decode_agy
from tare_tools_agent_runtime import ModelFailure
from tests.test_native_models import execution


@pytest.mark.parametrize('damage', [None, 'task_tool', 'missing_stop', 'extra_inference', 'bad_schema', 'error', 'legacy'])
def test_finish_display_requires_the_complete_typed_reply_and_lifecycle(damage):
    hooks = [{'phase':p,'model':'fixture','invocation':0,
              'termination':'TERMINAL_CUSTOM_HOOK' if p == 'stop' else None}
             for p in ('pre','post','stop')]
    value = {'text': 'Complete', 'actions': []}
    rows = [{'event':'step_update','step_update':{'tool_info':{'name':'finish'}}},
            {'event':'result','result':{'status':'SUCCESS','response':json.dumps(value)}}]
    if damage == 'task_tool':
        rows.insert(0, {'event':'step_update','step_update':{'tool_info':{'name':'read'}}})
    if damage == 'missing_stop':hooks.pop()
    if damage == 'extra_inference':hooks.insert(1,hooks[0])
    if damage == 'bad_schema':rows[-1]['result']['response'] = '{}'
    if damage == 'error':rows[-1]['result']['status'] = 'ERROR'
    if damage is None:
        response = decode_agy(rows,hooks,'fixture',structured=True)
        assert response['choices'][0]['message']['content'] == 'Complete'
    else:
        with pytest.raises((ModelFailure,ValueError)):
            decode_agy(rows,hooks,'fixture',structured=damage != 'legacy')


def test_a_final_action_agy_rejected_is_a_retryable_failure_not_a_blocking_one():
    # AGY 1.5.3 (2026-11-09): the model omitted the required 'text'; AGY marked finish ERROR but still reported
    # SUCCESS with that reply, and the session blocked on CLI_INVALID_RESPONSE.
    hooks = [{'phase': p, 'model': 'fixture', 'invocation': 0,
              'termination': 'TERMINAL_CUSTOM_HOOK' if p == 'stop' else None} for p in ('pre', 'post', 'stop')]
    reply = {'actions': [{'name': 'search', 'arguments_json': '{}'}]}
    rows = [{'event': 'step_update', 'step_update': {'state': 'ACTIVE', 'step_type': 'tool', 'tool_name': 'finish',
                                                     'tool_info': {'name': 'finish', 'parameters': reply}}},
            {'event': 'step_update', 'step_update': {'state': 'ERROR', 'step_type': 'tool', 'tool_name': 'finish',
                                                     'tool_info': {'name': 'finish', 'parameters': reply, 'error': {
                                                         'type': 'TOOL_ERROR',
                                                         'message': "invalid arguments: missing property 'text'"}}}},
            {'event': 'result', 'result': {'status': 'SUCCESS', 'response': json.dumps(reply)}}]
    with pytest.raises(ModelFailure) as failure:
        decode_agy(rows, hooks, 'fixture', structured=True)
    assert failure.value.code == 'CLI_INVALID_RESPONSE' and failure.value.retryable is True
    assert "missing property 'text'" in failure.value.final_action_error


def test_a_final_action_the_cli_rejected_is_repaired_once_naming_the_rejection(tmp_path, monkeypatch):
    # AGY 1.5.3 (2026-11-09): the model omitted 'text', then wrote the key as '"\\"text"'. A plain retry resent the
    # same request; the repair tells the model what was rejected, and counts both attempts' tokens.
    from compute_plane import native_cli
    client = native_cli.VendorCliClient('cli://agy', executable=str(tmp_path/'agy.exe'), version='1.2.7')
    sent = []

    def rejected(usage, attempt):
        failure = ModelFailure('CLI_INVALID_RESPONSE', retryable=True)
        failure.final_action_error = "invalid arguments: missing property 'text'"
        failure.usage = usage
        failure.transport_observation = {'vendor': 'agy', 'attempt': attempt}
        return failure

    answers = [rejected({'input_tokens': 100, 'output_tokens': 10}, 1),
               ({'model': 'm', 'usage': {'input_tokens': 120, 'output_tokens': 12}, 'choices': []}, {'vendor': 'agy'})]

    def exchange(path, body, control):
        sent.append(body)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(client, '_exchange', exchange)
    response, receipt = client.post_observed('', {'model': 'm', 'prompt': 'p'}, execution())
    assert 'reminder' not in sent[0] and "missing property 'text'" in sent[1]['reminder']
    assert response['usage'] == {'input_tokens': 220, 'output_tokens': 22}
    assert receipt['repair_of'] == {'vendor': 'agy', 'attempt': 1} and receipt['inference_invocations_observed'] == 2

    sent.clear()
    answers[:] = [rejected(None, 1), rejected(None, 2)]
    with pytest.raises(ModelFailure) as twice:
        client.post_observed('', {'model': 'm', 'prompt': 'p'}, execution())
    assert len(sent) == 2 and twice.value.transport_observation['repair_of'] == {'vendor': 'agy', 'attempt': 1}

    sent.clear()
    answers[:] = [ModelFailure('CLI_GENERATION_FAILED')]
    with pytest.raises(ModelFailure, match='CLI_GENERATION_FAILED'):
        client.post_observed('', {'model': 'm', 'prompt': 'p'}, execution())
    assert len(sent) == 1  # only a rejection the CLI named is repaired
