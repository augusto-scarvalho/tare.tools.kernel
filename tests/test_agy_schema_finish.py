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


@pytest.mark.parametrize('damage', [None, 'ran', 'hooked', 'other_error', 'plain'])
def test_a_catalog_action_called_as_a_function_is_repaired_only_when_agy_had_no_such_tool(damage):
    # Claude Sonnet 5.5 on AGY 1.5.9 (2026-11-14): verify and search called as functions; AGY answered "unknown
    # tool" for each, fired no tool hook and reported SUCCESS with an empty reply. The session blocked.
    hooks = [{'phase': p, 'model': 'fixture', 'invocation': 0,
              'termination': 'TERMINAL_CUSTOM_HOOK' if p == 'stop' else None} for p in ('pre', 'post', 'stop')]
    def step(index, name, state, error=None):
        info = {'name': name, 'parameters': {}, **({'error': {'type': 'TOOL_ERROR', 'message': error}} if error else {})}
        return {'event': 'step_update', 'step_update': {'state': state, 'step_index': index, 'step_type': 'tool',
                                                        'tool_name': name, 'tool_info': info}}
    unknown = 'unknown tool: "%s" — check spelling'
    rows = [step(2, 'verify', 'ACTIVE'), step(2, 'verify', 'ERROR', unknown % 'verify'),
            step(3, 'search', 'ACTIVE'), step(3, 'search', 'ERROR', unknown % 'search'),
            {'event': 'result', 'result': {'status': 'SUCCESS', 'response': '', 'usage': {'input_tokens': 7}}}]
    if damage == 'ran': rows[3] = step(3, 'search', 'DONE')
    if damage == 'hooked': hooks.insert(1, {'phase': 'tool', 'model': 'fixture', 'tool': 'search', 'decision': 'deny'})
    if damage == 'other_error': rows[3] = step(3, 'search', 'ERROR', 'permission denied')
    with pytest.raises(ModelFailure) as failure:
        decode_agy(rows, hooks, 'fixture', structured=damage != 'plain')
    if damage is None:
        assert failure.value.code == 'CLI_UNEXPECTED_TOOL' and failure.value.retryable is True
        assert 'search, verify' in failure.value.final_action_error and failure.value.usage == {'input_tokens': 7}
    else:
        assert not failure.value.retryable and not hasattr(failure.value, 'final_action_error')


def test_a_final_action_the_cli_rejected_is_repaired_once_naming_the_rejection(tmp_path, monkeypatch):
    # AGY 1.5.3 (2026-11-09): the model omitted 'text', then wrote the key as '"\\"text"'. A plain retry resent the
    # same request; the repair tells the model what was rejected. The response keeps the repair's own usage (the
    # count anchor measures one request); the first attempt's usage stays in the receipt.
    from compute_plane import native_cli
    from compute_plane.native_models import NativeTransportInterrupted
    client = native_cli.VendorCliClient('cli://agy', executable=str(tmp_path/'agy.exe'), version='1.2.7')
    sent = []

    def rejected(usage, attempt):
        failure = ModelFailure('CLI_INVALID_RESPONSE', retryable=True)
        failure.final_action_error = "invalid arguments: missing property 'text'"
        failure.usage = usage
        failure.transport_observation = {'vendor': 'agy', 'attempt': attempt}
        return failure

    answers = []

    def exchange(path, body, control):
        sent.append(body)
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(client, '_exchange', exchange)
    request = {'model': 'm', 'prompt': 'p'}
    answers[:] = [rejected({'input_tokens': 100}, 1),
                  ({'model': 'm', 'usage': {'input_tokens': 120}, 'choices': []}, {'vendor': 'agy', 'attempt': 2})]
    response, receipt = client.post_observed('', request, execution())
    assert 'reminder' not in sent[0] and "missing property 'text'" in sent[1]['reminder']
    assert response['usage'] == {'input_tokens': 120}  # not summed: it anchors the next request's estimate
    assert receipt['repair_of'] == {'vendor': 'agy', 'attempt': 1, 'usage': {'input_tokens': 100}}
    assert receipt['inference_invocations_observed'] == 2

    sent.clear()
    answers[:] = [rejected(None, 1), rejected(None, 2)]
    with pytest.raises(ModelFailure) as twice:
        client.post_observed('', request, execution())
    assert len(sent) == 2 and twice.value.transport_observation['repair_of']['attempt'] == 1

    sent.clear()
    answers[:] = [rejected(None, 1), NativeTransportInterrupted('CLI_INCOMPLETE_EXCHANGE', {'attempt': 2})]
    with pytest.raises(NativeTransportInterrupted) as cut:
        client.post_observed('', request, execution())
    assert cut.value.transport_observation['repair_of']['attempt'] == 1  # still visible when the repair is cut

    sent.clear()
    answers[:] = [rejected(None, 1), ModelFailure('CLI_INPUT_BYTE_LIMIT_EXCEEDED', not_started=True)]
    with pytest.raises(ModelFailure) as full:
        client.post_observed('', request, execution())
    assert full.value.code == 'CLI_INVALID_RESPONSE' and full.value.retryable  # the note did not fit: the rejection
    assert full.value.transport_observation['repair_not_started'] == 'CLI_INPUT_BYTE_LIMIT_EXCEEDED'

    sent.clear()
    answers[:] = [ModelFailure('CLI_GENERATION_FAILED')]
    with pytest.raises(ModelFailure, match='CLI_GENERATION_FAILED'):
        client.post_observed('', request, execution())
    assert len(sent) == 1  # only a rejection the CLI named is repaired

    sent.clear()
    stopping = execution()
    stopping.cancellation.cancel()
    answers[:] = [rejected(None, 1)]
    with pytest.raises(ModelFailure):
        client.post_observed('', request, stopping)
    assert len(sent) == 1  # cancelled: no repair


def test_a_finish_the_model_corrected_in_the_same_run_is_not_a_rejection():
    hooks = [{'phase': p, 'model': 'fixture', 'invocation': 0,
              'termination': 'TERMINAL_CUSTOM_HOOK' if p == 'stop' else None} for p in ('pre', 'post', 'stop')]
    value = {'text': 'Corrected', 'actions': []}
    rows = [{'event': 'step_update', 'step_update': {'state': 'ERROR', 'tool_info': {'name': 'finish', 'error': {
                'type': 'TOOL_ERROR', 'message': "invalid arguments: missing property 'text'"}}}},
            {'event': 'step_update', 'step_update': {'state': 'DONE', 'tool_info': {'name': 'finish'}}},
            {'event': 'result', 'result': {'status': 'SUCCESS', 'response': json.dumps(value)}}]
    assert decode_agy(rows, hooks, 'fixture', structured=True)['choices'][0]['message']['content'] == 'Corrected'
