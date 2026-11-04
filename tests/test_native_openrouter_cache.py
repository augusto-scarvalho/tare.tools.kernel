"""OpenRouter affinity survives turns/resume; accounting never invents hits."""
from dataclasses import replace
import json

import pytest

from compute_plane.native_models import ModelProfile, NativeModelAdapter, encode
from compute_plane.native_usage import UsageContract, normalize_usage, reported_cost
from tests.test_native_models import Client, SPECS, profile, request


def adapter(endpoint='https://openrouter.ai/api/v1', parameters=None):
    client = Client([])
    client.endpoint = endpoint
    selected = replace(profile('openai-chat'), endpoint=endpoint,
                       parameters_json=json.dumps(parameters or {}))
    return NativeModelAdapter(selected, client, system='Stable', tool_specs_json=SPECS)


def test_affinity_survives_new_adapter_and_changed_suffix_but_separates_sessions():
    req = request()
    first = adapter().wire(req)
    continued = adapter().wire(replace(req, call_id='s/model/2', suffix='New observation'))
    assert first['session_id'] == continued['session_id']
    assert first['session_id'] != adapter().wire(replace(req, session_id='other'))['session_id']
    assert len(first['session_id']) <= 256
    assert encode(first['tools']) == encode(continued['tools'])
    assert encode(first['messages']) == encode(continued['messages'][:-1])
    assert adapter().wire(replace(req, purpose='summary'))['session_id'] == first['session_id']
    assert adapter(parameters={'session_id': 'explicit'}).wire(req)['session_id'] == 'explicit'


@pytest.mark.parametrize('endpoint', ['http://fixture', 'https://openrouter.ai.example/api/v1'])
def test_affinity_is_not_added_to_other_http_providers(endpoint):
    assert 'session_id' not in adapter(endpoint).wire(request())


def responses_adapter(endpoint='https://openrouter.ai/api/v1', parameters=None):
    client = Client([])
    client.endpoint = endpoint
    selected = ModelProfile('openai-responses', 'fixture', 32768, 4096,
                            'OpenRouter Responses qualification', endpoint,
                            parameters_json=json.dumps(parameters or {}))
    return NativeModelAdapter(selected, client, system='Stable', tool_specs_json=SPECS)


def test_responses_preserves_endpoint_policy_and_conversation_cache_identity():
    policy = {'provider': {'only': ['fixture/zdr'], 'allow_fallbacks': False}, 'transforms': []}
    req = request()
    first = responses_adapter(parameters=policy).wire(req)
    resumed = responses_adapter(parameters=policy).wire(replace(req, call_id='s/model/2', suffix='Changed scope'))
    other = responses_adapter(parameters=policy).wire(replace(req, session_id='other'))
    for field in ('session_id', 'prompt_cache_key', 'instructions', 'tools', 'provider', 'transforms'):
        assert first[field] == resumed[field]
    assert first['session_id'] != other['session_id']
    assert first['prompt_cache_key'] != other['prompt_cache_key']
    assert len(first['prompt_cache_key']) <= 64
    assert first['input'] == resumed['input'][:-1]
    assert first['include'] == ['reasoning.encrypted_content'] and first['store'] is False
    assert first['max_output_tokens'] == 4096
    explicit = responses_adapter(parameters={**policy, 'session_id': 'operator-route',
                                            'prompt_cache_key': 'operator-prefix'}).wire(req)
    assert explicit['session_id'] == 'operator-route' and explicit['prompt_cache_key'] == 'operator-prefix'


@pytest.mark.parametrize('endpoint', ['https://api.openai.com/v1', 'https://openrouter.ai.example/api/v1'])
@pytest.mark.parametrize('field,value', [('provider', {'only': ['fixture']}), ('transforms', []), ('session_id', 'x')])
def test_gateway_responses_parameters_are_not_admitted_elsewhere(endpoint, field, value):
    with pytest.raises(ValueError, match='unsupported OpenAI Responses parameter'):
        responses_adapter(endpoint, {field: value})


CONTRACT = UsageContract('openrouter-inclusive', 'OpenRouter usage accounting; prompt includes cache reads/writes')


@pytest.mark.parametrize('details,uncached,read,write', [
    ({'cached_tokens': 80, 'cache_write_tokens': 10}, 10, 80, 10),
    ({'cached_tokens': 0, 'cache_write_tokens': 100}, 0, 0, 100),
    ({'cached_tokens': 0, 'cache_write_tokens': 0}, 100, 0, 0),
    ({'cached_tokens': 80}, None, 80, None),
    ({'cache_write_tokens': 0}, None, None, 0),
])
def test_cache_categories_remain_disjoint_and_missing_is_unknown(details, uncached, read, write):
    result = normalize_usage({'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                              'prompt_tokens_details': details}, CONTRACT)
    assert not result['issues']
    tokens = result['tokens']
    assert (tokens['uncached_input_tokens'], tokens['cache_read_tokens'], tokens['cache_write_tokens']) == (uncached, read, write)
    assert tokens['total_tokens'] == 120


@pytest.mark.parametrize('details', [
    {'cached_tokens': 80, 'cache_write_tokens': 21},
    {'cached_tokens': True, 'cache_write_tokens': 0},
    {'cached_tokens': 0, 'cache_write_tokens': -1},
])
def test_invalid_cache_receipts_do_not_contribute_exact_totals(details):
    result = normalize_usage({'prompt_tokens': 100, 'completion_tokens': 20,
                              'prompt_tokens_details': details}, CONTRACT)
    assert result['issues'] and all(v is None for v in result['tokens'].values())


def test_responses_gateway_usage_preserves_unknown_writes_and_reported_account_charge():
    contract = UsageContract('openrouter-responses-inclusive', 'OpenRouter Responses usage receipt')
    raw = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
           'input_tokens_details': {'cached_tokens': 80}, 'cost': .004}
    normalized = normalize_usage(raw, contract)
    assert not normalized['issues']
    assert normalized['tokens']['cache_read_tokens'] == 80
    assert normalized['tokens']['cache_write_tokens'] is None
    assert normalized['tokens']['uncached_input_tokens'] is None
    assert reported_cost(raw, contract)['known_amount'] == '0.004'
    assert reported_cost(raw, UsageContract('responses-inclusive', 'Direct OpenAI')) is None
