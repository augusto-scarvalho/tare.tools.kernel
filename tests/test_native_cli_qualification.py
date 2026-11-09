"""A CLI version the adapter has not seen answers one probe through the same path before the real request."""
import json

import pytest
from tare_tools_agent_runtime import ModelFailure

from compute_plane import native_cli
from compute_plane.native_models import NativeTransportInterrupted
from tests.test_native_models import execution


def client_with(monkeypatch, tmp_path, probe_outcome, qualification=None):
    """A client whose exchanges are scripted: the real request first meets an unqualified version 9.9.9."""
    client = native_cli.VendorCliClient('cli://agy', executable=str(tmp_path/'agy.exe'), version='1.2.7',
                                        qualification=qualification)
    sent = []

    def repaired(path, body, control):
        sent.append(body)
        probing = body['prompt'].find(native_cli.PROBE_TEXT) >= 0
        if not probing and '9.9.9' not in client._probing and client.qualification.get('9.9.9') is None:
            raise native_cli._Unqualified('9.9.9')
        if probing:
            if isinstance(probe_outcome, BaseException):
                raise probe_outcome
            return probe_outcome
        return {'model': 'm', 'usage': {'input_tokens': 9}, 'choices': []}, {'vendor': 'agy'}

    monkeypatch.setattr(client, '_repaired', repaired)
    return client, sent


def real_body():
    return native_cli.cli_request({'model': 'fixture', 'messages': [{'role': 'user', 'content': 'x'*5000}],
                                   'tools': [{'function': {'name': 'read', 'parameters': {}}}]}, structured=True)


def test_a_probe_qualifies_a_version_once_per_machine_and_uses_the_real_path(tmp_path, monkeypatch):
    ok = ({'model': 'm', 'usage': {'input_tokens': 3, 'cache_read_tokens': 0}, 'choices': []},
          {'vendor': 'agy', 'inference_invocations_observed': 2})  # passed after one repair
    store = {}
    client, sent = client_with(monkeypatch, tmp_path, ok, store)
    _, receipt = client.post_observed('', real_body(), execution())
    probe = sent[1]
    assert probe['effort'] == 'low' and 'reminder' not in probe and len(probe['prompt']) < len(sent[0]['prompt'])
    assert json.loads(probe['prompt'])['action_catalog'] == json.loads(sent[0]['prompt'])['action_catalog']
    assert probe['response_schema'] == sent[0]['response_schema']  # same branch and decoder as the real request
    assert receipt['qualification'] == {'version': '9.9.9', 'status': 'probed', 'probe_invocations': 2,
                                        'probe_usage': {'input_tokens': 3, 'cache_read_tokens': 0}}
    assert store['9.9.9']['status'] == 'ok' and store['9.9.9']['usage_readable'] is True
    again, later = client_with(monkeypatch, tmp_path, ok, store)  # another session, same persisted record
    again.post_observed('', real_body(), execution())
    assert len(later) == 1  # no second probe


@pytest.mark.parametrize('failure', [ModelFailure('CLI_USAGE_LIMIT'), ModelFailure('CLI_GENERATION_FAILED')])
def test_a_quota_or_provider_failure_of_the_probe_is_no_verdict(tmp_path, monkeypatch, failure):
    store = {}
    client, sent = client_with(monkeypatch, tmp_path, failure, store)
    with pytest.raises(ModelFailure) as caught:
        client.post_observed('', real_body(), execution())
    assert caught.value.code == failure.code and caught.value.not_started  # the user's request was not sent
    assert store == {} and len(sent) == 2


def test_an_interrupted_probe_says_the_request_was_not_sent(tmp_path, monkeypatch):
    store = {}
    client, _ = client_with(monkeypatch, tmp_path, NativeTransportInterrupted('CLI_INCOMPLETE_EXCHANGE', {}), store)
    with pytest.raises(ModelFailure) as caught:
        client.post_observed('', real_body(), execution())
    assert caught.value.code == 'CLI_QUALIFICATION_INTERRUPTED'
    assert caught.value.not_started and caught.value.retryable and store == {}


def test_a_shape_failure_is_recorded_and_stops_before_the_real_request(tmp_path, monkeypatch):
    rejected = ModelFailure('CLI_INVALID_RESPONSE', retryable=True)
    rejected.final_action_error = "invalid arguments: missing property 'text'"
    store = {}
    client, sent = client_with(monkeypatch, tmp_path, rejected, store)
    with pytest.raises(ModelFailure) as caught:
        client.post_observed('', real_body(), execution())
    assert caught.value.code == 'CLI_QUALIFICATION_FAILED' and caught.value.not_started and len(sent) == 2
    assert store['9.9.9']['probe_code'] == 'CLI_INVALID_RESPONSE' and "missing property" in store['9.9.9']['detail']
    assert caught.value.transport_observation['qualification']['version'] == '9.9.9'


def test_a_request_without_a_conversation_is_not_probed():
    assert native_cli.probe_body({'model': 'm', 'prompt': '{"other": 1}', 'system': 's'}) is None
