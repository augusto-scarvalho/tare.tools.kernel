"""Provider timings survive transport/journal without becoming token charges."""
import io
import json

import pytest

from compute_plane.native_models import NativeModelAdapter
from compute_plane.native_stream import chat, converse
from tests.test_native_models import Client, SPECS, execution, profile, request, response, DELIVERY_TOOL
from tests.test_native_streaming import chat_parts, converse_parts, sse, aws
from tests.test_native_usage import run_session, report, chat_usage


@pytest.mark.parametrize('protocol,key,values', [
    ('openai-chat', 'timings', {'cache_n': 6558, 'prompt_n': 25, 'prompt_ms': 193.091,
                              'predicted_n': 85, 'predicted_ms': 999.729,
                              'predicted_per_second': 85.023, 'draft_n_accepted': 61}),
    ('bedrock-converse', 'metrics', {'latencyMs': 125}),
])
def test_provider_timings_survive_reply_and_observation_without_entering_prompt(protocol, key, values):
    rows = []
    result = response(protocol, DELIVERY_TOOL)
    result[key] = values
    adapter = NativeModelAdapter(profile(protocol), Client([result]), system='stable',
                                 tool_specs_json=SPECS, observe=rows.append)
    before = adapter.wire(request())
    reply = adapter.generate(request(), execution())
    expected = {'source': protocol + '.' + key, 'values': values}
    metadata = json.loads(reply.metadata_json)
    assert reply.proposes_delivery
    assert metadata['provider_timing'] == rows[0]['provider_timing'] == expected
    assert key not in metadata['assistant']
    assert adapter.wire(request()) == before
    assert rows[0]['duration_ms'] >= 0  # Transport wall time is a different measurement.


@pytest.mark.parametrize('value', [None, 'unknown', 15, []])
def test_unusable_optional_timing_does_not_break_generation_or_invent_zero(value):
    result = response('openai-chat', DELIVERY_TOOL)
    result['timings'] = value
    adapter, log, rows, outcome = run_session([result])
    assert outcome.status == 'proposed'
    assert report(adapter, log, rows)['calls'][0]['provider_timing'] is None


@pytest.mark.parametrize('lose_observation', [False, True])
def test_usage_keeps_timing_with_saved_reply_fallback_and_no_double_count(lose_observation):
    result = response('openai-chat', DELIVERY_TOOL, usage=chat_usage())
    result['timings'] = {'prompt_n': 20, 'cache_n': 80, 'predicted_n': 20,
                         'prompt_ms': 40, 'predicted_ms': 250}
    adapter, log, rows, _ = run_session([result])
    measured = report(adapter, log, [] if lose_observation else rows)
    assert measured['calls'][0]['provider_timing']['values'] == result['timings']
    assert measured['tokens']['total_tokens']['known_sum'] == 120
    assert measured['tokens']['cache_read_tokens']['known_sum'] == 80
    assert measured['calls'][0]['usage_source'] == ('saved_reply' if lose_observation else 'observation')


def test_stream_keeps_latest_timing_snapshot_instead_of_summing_chunks():
    parts = chat_parts()
    parts.insert(1, sse({'choices': [], 'timings': {'predicted_n': 1}}))
    final = {'cache_n': 80, 'prompt_n': 20, 'prompt_ms': 40, 'predicted_n': 5, 'predicted_ms': 60}
    parts.insert(-1, sse({'choices': [], 'timings': final}))
    emitted = []
    value = chat(io.BytesIO(b''.join(parts)), emitted.append)
    assert value['timings'] == final
    assert value['choices'][0]['message']['content'] == 'olá mundo'
    assert value['usage']['completion_tokens'] == 5
    assert all('timings' not in event for event in emitted)


def test_converse_stream_preserves_provider_latency_with_no_invented_decode_rate():
    parts = converse_parts()
    parts[-1] = aws('metadata', {'usage': {'inputTokens': 20, 'outputTokens': 5},
                                 'metrics': {'latencyMs': 125}})
    result = converse(io.BytesIO(b''.join(parts)), lambda _: None)
    assert result['metrics'] == {'latencyMs': 125}
    assert result['usage']['outputTokens'] == 5
