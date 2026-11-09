"""A CLI release the adapter has not seen answers one probe per model through the same path before the real request."""
import json

import pytest
from tare_tools_agent_runtime import ModelFailure

from compute_plane import native_cli
from compute_plane.native_models import NativeTransportInterrupted
from tests.test_native_models import execution

KEY = '9.9.9 fixture'


def client_with(monkeypatch, tmp_path, probe_outcome, qualification=None, real=None):
    """A client whose exchanges are scripted: the real request first meets an unqualified release 9.9.9."""
    client = native_cli.VendorCliClient('cli://agy', executable=str(tmp_path/'agy.exe'), version='1.2.7',
                                        qualification=qualification)
    sent = []

    def repaired(path, body, control):
        sent.append(body)
        probing = native_cli.PROBE_TEXT in body['prompt']
        if (not probing and KEY not in client._probing and KEY not in client._passed
                and client.qualification.get(KEY) is None):
            raise native_cli._Unqualified(KEY)
        if probing:
            if isinstance(probe_outcome, BaseException):
                raise probe_outcome
            return probe_outcome
        if isinstance(real, BaseException):
            raise real
        return {'model': 'm', 'usage': {'input_tokens': 9}, 'choices': []}, {'vendor': 'agy'}

    monkeypatch.setattr(client, '_repaired', repaired)
    return client, sent


def real_body():
    return native_cli.cli_request({'model': 'fixture', 'messages': [{'role': 'user', 'content': 'x'*5000}],
                                   'tools': [{'function': {'name': 'read', 'parameters': {}}}]}, structured=True)


OK = ({'model': 'm', 'usage': {'input_tokens': 3, 'cache_read_tokens': 0}, 'choices': []},
      {'vendor': 'agy', 'inference_invocations_observed': 2})  # passed after one repair


def test_a_probe_qualifies_a_release_and_model_once_per_machine_through_the_real_path(tmp_path, monkeypatch):
    store = {}
    client, sent = client_with(monkeypatch, tmp_path, OK, store)
    _, receipt = client.post_observed('', real_body(), execution())
    probe = sent[1]
    assert probe['effort'] == sent[0]['effort']  # AGY: a model named for its tier accepts only that effort
    assert 'reminder' not in probe and len(probe['prompt']) < len(sent[0]['prompt'])
    assert json.loads(probe['prompt'])['action_catalog'] == json.loads(sent[0]['prompt'])['action_catalog']
    assert probe['response_schema'] == sent[0]['response_schema']  # same branch and decoder as the real request
    assert receipt['qualification'] == {'version': '9.9.9', 'model': 'fixture', 'status': 'probed',
        'probe_invocations': 2, 'probe_usage': {'input_tokens': 3, 'cache_read_tokens': 0}}
    assert store[KEY]['status'] == 'ok' and store[KEY]['usage_readable'] is True
    again, later = client_with(monkeypatch, tmp_path, OK, store)  # another session, same persisted record
    _, second = again.post_observed('', real_body(), execution())
    assert len(later) == 1 and 'qualification' not in second  # no second probe


def test_other_clis_probe_at_low_effort():
    probe = native_cli.probe_body({'model': 'm', 'effort': 'high', 'prompt': '{"messages": []}', 'system': 's'},
                                  'cli://claude')
    assert probe['effort'] == 'low'


@pytest.mark.parametrize('failure', [ModelFailure('CLI_USAGE_LIMIT'), ModelFailure('CLI_GENERATION_FAILED')])
def test_a_quota_or_provider_failure_of_the_probe_keeps_its_own_code_and_handling(tmp_path, monkeypatch, failure):
    store = {}
    client, sent = client_with(monkeypatch, tmp_path, failure, store)
    with pytest.raises(ModelFailure) as caught:
        client.post_observed('', real_body(), execution())
    # Not renamed "not started": a spent quota still reaches the host's usage-limit handling.
    assert caught.value is failure and not caught.value.not_started
    assert store == {} and len(sent) == 2


def test_an_interrupted_probe_says_the_request_was_not_sent(tmp_path, monkeypatch):
    store = {}
    client, _ = client_with(monkeypatch, tmp_path, NativeTransportInterrupted('CLI_INCOMPLETE_EXCHANGE', {}), store)
    with pytest.raises(ModelFailure) as caught:
        client.post_observed('', real_body(), execution())
    assert caught.value.code == 'CLI_QUALIFICATION_INTERRUPTED'
    assert caught.value.not_started and caught.value.retryable and store == {}


def test_a_shape_failure_is_recorded_with_its_usage_and_stops_before_the_real_request(tmp_path, monkeypatch):
    rejected = ModelFailure('CLI_INVALID_RESPONSE', retryable=True)
    rejected.final_action_error = "invalid arguments: missing property 'text'"
    rejected.usage = {'input_tokens': 7}
    store = {}
    client, sent = client_with(monkeypatch, tmp_path, rejected, store)
    with pytest.raises(ModelFailure) as caught:
        client.post_observed('', real_body(), execution())
    assert caught.value.code == 'CLI_QUALIFICATION_FAILED' and caught.value.not_started and len(sent) == 2
    assert store[KEY]['probe_code'] == 'CLI_INVALID_RESPONSE' and 'missing property' in store[KEY]['detail']
    assert store[KEY]['probe_usage'] == {'input_tokens': 7}  # the probe was paid: its usage is recorded
    assert caught.value.transport_observation['qualification']['version'] == '9.9.9'


def test_the_probe_goes_with_the_real_call_that_paid_for_it_even_when_that_call_fails(tmp_path, monkeypatch):
    client, _ = client_with(monkeypatch, tmp_path, OK, {}, real=ModelFailure('CLI_GENERATION_FAILED'))
    with pytest.raises(ModelFailure) as caught:
        client.post_observed('', real_body(), execution())
    assert caught.value.transport_observation['qualification']['status'] == 'probed'


def test_a_store_that_cannot_be_written_loses_the_verdict_never_the_call(tmp_path, monkeypatch):
    class Unwritable(dict):
        def __setitem__(self, key, value):
            raise PermissionError('locked by another host')
    client, _ = client_with(monkeypatch, tmp_path, OK, Unwritable())
    _, receipt = client.post_observed('', real_body(), execution())
    assert receipt['qualification']['status'] == 'probed'


def test_a_request_without_a_conversation_is_not_probed():
    assert native_cli.probe_body({'model': 'm', 'prompt': '{"other": 1}', 'system': 's'}) is None
