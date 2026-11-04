"""Offline falsifiers for premature compaction across native provider codecs."""
from dataclasses import asdict, replace
import json

import pytest

from tare_tools_agent_runtime import ConversationLog, ContextPolicy, NativeContext, ProjectionSpec
from compute_plane.native_models import NativeModelAdapter, ModelProfile, encode, sha
from compute_plane.native_counting import input_tokens, added_bytes
from .test_native_models import Client, SPECS, execution, request, response


def conversation(protocol='openai-chat', endpoint='http://fixture', usage=None, *, legacy=False, remaining=None,
                 formatter=False, window=272000, output=32768):
    profile = ModelProfile(protocol, 'fixture-model', window, output, 'fixture', endpoint)
    result = response('bedrock-converse' if protocol == 'bedrock-converse' else 'openai-chat',
                      usage=usage if usage is not None else {'prompt_tokens': 40000})
    if protocol == 'openai-codex-responses':
        result = {'model': 'fixture-model', 'status': 'completed', 'usage': usage, 'output': [
            {'id': 'r1', 'type': 'reasoning', 'encrypted_content': 'opaque' * 1000},
            {'id': 'fc1', 'type': 'function_call', 'call_id': 'provider-id', 'name': 'read',
             'arguments': '{"path":"app.py"}', 'status': 'completed'}]}
    client = Client([result]); client.endpoint = endpoint
    if formatter:
        client.version = '2.1.316'
    client.post_stream_observed = lambda path, body, control, **kw: (client.post(path, body, control), {})
    observations = []
    options = dict(system='Stable', tool_specs_json=SPECS, counter_protocol='utf8-envelope-estimate-v1',
                   streaming=protocol == 'openai-codex-responses')
    adapter = NativeModelAdapter(profile, client, observe=observations.append, **options)
    req = replace(request(), instruction='source ' * 30000, remaining_model_calls=remaining)
    reply = adapter.generate(req, execution())
    if legacy:
        metadata = json.loads(reply.metadata_json); metadata.pop('count_anchor', None)
        reply = replace(reply, metadata_json=encode(metadata))
    log = ConversationLog()
    log.append({'kind': 'model_returned', 'call_id': req.call_id, 'reply': asdict(reply)})
    log.append({'kind': 'effect_returned', 'result': {'effect_id': 's/effect/1',
                'status': 'succeeded', 'output_json': '{"text":"new source"}'}})
    next_request = replace(req, call_id='s/model/2', context=log.project(ProjectionSpec(1, 'fixture')),
                           remaining_model_calls=None if remaining is None else remaining - 1)
    resumed = NativeModelAdapter(profile, client, count_observations=observations if legacy else (), **options)
    return resumed, next_request, log, client, observations


@pytest.mark.parametrize('protocol,endpoint,usage', [
    ('openai-chat', 'http://fixture', {'prompt_tokens': 40000, 'prompt_tokens_details': {'cached_tokens': 39000}}),
    ('openai-codex-responses', 'http://fixture', {'input_tokens': 40000}),
    ('vendor-cli', 'cli://codex', {'input_tokens': 40000, 'cached_input_tokens': 39000}),
    ('vendor-cli', 'cli://claude', {'input_tokens': 1000, 'cache_read_input_tokens': 38000, 'cache_creation_input_tokens': 1000}),
    ('vendor-cli', 'cli://agy', {'input_tokens': 1000, 'cache_read_tokens': 39000}),
    ('bedrock-converse', 'http://fixture', {'inputTokens': 1000, 'cacheReadInputTokens': 38000, 'cacheWriteInputTokens': 1000}),
])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('remaining', [None, 16])
def test_usage_anchor_survives_resume_and_prevents_false_compaction(protocol, endpoint, usage, legacy, remaining):
    adapter, req, log, client, _ = conversation(protocol, endpoint, usage, legacy=legacy, remaining=remaining)
    row = adapter.count(req, execution())
    assert row['status'] == 'estimated' and row['input_tokens'] is None
    assert row['count_basis'] == 'prior_usage_plus_utf8_growth'
    assert row['calibration']['input_tokens'] == 40000
    assert 40000 < row['estimated_input_tokens'] < 60000
    # No measured route or extra model call was used to obtain this estimate.
    assert row['client_invocations'] == 0 and len(client.calls) == 1
    assert row['request_sha256'] == sha(adapter.wire(req))
    policy = ContextPolicy(window=272000, output_reserve=32768, margin=128, allow_estimated_counts=True)
    admitted = NativeContext(policy, counter=adapter.count, summarizer=adapter).prepare(req, log, execution(), 1)
    assert admitted.admitted_request_sha256 == row['request_sha256']
    assert not any(r.event()['kind'].startswith('context_summary') for r in log.snapshot().records)
    # New context still consumes capacity; a previous small count cannot mask it.
    grown = adapter.count(replace(req, suffix='new data ' * 150000), execution())
    assert grown['estimated_input_tokens'] > policy.trigger_limit


def test_budget_notice_stays_in_history_so_the_prior_request_is_an_exact_prefix():
    adapter, req, _, _, _ = conversation(remaining=16)
    first = replace(req, call_id='s/model/1', remaining_model_calls=16,
                    context=ConversationLog().project(ProjectionSpec(1, 'fixture')))
    previous, current = adapter.wire(first)['messages'], adapter.wire(req)['messages']
    assert current[:len(previous)] == previous
    assert 'Runtime budget: 15' in current[-1]['content']


def test_growth_is_charged_at_the_measured_tokens_per_byte():
    adapter, req, _, _, _ = conversation()  # 40000 tokens measured for ~210 KB
    growth = 'print(value)\n' * 10000
    row = adapter.count(replace(req, suffix=growth), execution())
    per_byte = row['calibration']['tokens_per_byte']
    assert 0.1 < per_byte < 1.0
    expected = 40000 + row['calibration']['growth_bytes'] * per_byte + 1024
    assert abs(row['estimated_input_tokens'] - expected) <= 2
    assert row['estimated_input_tokens'] < 40000 + len(growth)  # below the old byte-per-token charge


@pytest.mark.parametrize('change', ['instruction', 'required_context', 'session_id', 'anchor', 'usage', 'summary',
                                   'budget', 'invalid_budget', 'zero_budget', 'missing_budget'])
def test_changed_identity_or_missing_usage_cannot_reuse_anchor(change):
    adapter, req, log, _, _ = conversation(remaining=16 if change == 'missing_budget' else None)
    if change in {'instruction', 'required_context', 'session_id'}:
        req = replace(req, **{change: ('different',) if change == 'required_context' else 'different'})
    else:
        events = req.context.events()
        metadata = json.loads(events[0]['reply']['metadata_json'])
        if change == 'anchor': metadata['count_anchor']['request_sha256'] = '0' * 64
        if change == 'usage': metadata['usage'] = None
        if change == 'budget': metadata['count_anchor']['remaining_model_calls'] = 99
        if change == 'invalid_budget': metadata['count_anchor']['remaining_model_calls'] = True
        if change == 'zero_budget': metadata['count_anchor']['remaining_model_calls'] = 0
        if change == 'missing_budget': metadata['count_anchor'].pop('remaining_model_calls')
        events[0]['reply']['metadata_json'] = encode(metadata)
        changed = ConversationLog()
        if change == 'summary':
            changed.append({'kind': 'context_activated', 'profile_digest': adapter.profile_digest,
                            'source_context_sha256': 'a' * 64, 'summary': 'new compacted history'})
        for event in events: changed.append(event)
        req = replace(req, context=changed.project(ProjectionSpec(1, 'fixture')))
    row = adapter.count(req, execution())
    size = len(encode(adapter.wire(req)).encode())
    if change in {'session_id', 'usage', 'invalid_budget', 'zero_budget'}:  # no measurement to trust
        assert row['count_basis'] == 'serialized_utf8_bytes'
    else:  # the request changed (a compacted history): priced whole at the anchor's measured tokens per byte
        assert row['count_basis'] == 'measured_tokens_per_utf8_byte'
        assert 0 < row['calibration']['tokens_per_byte'] < 1 and row['estimated_input_tokens'] < size + 1024
        if change == 'summary':  # 210 KB of history: about a quarter of the byte-per-token price
            assert row['estimated_input_tokens'] < size / 3


def test_anchor_reconstruction_accepts_native_projection_over_two_mib():
    adapter, req, _, _, _ = conversation()
    events = req.context.events()
    # Reconstruct a prior request with a native-sized context.
    from tare_tools_agent_runtime import ConversationLimits
    from compute_plane.native_journal import NATIVE_JOURNAL_MAX_BYTES
    prefix = ConversationLog(ConversationLimits(max_projection_bytes=NATIVE_JOURNAL_MAX_BYTES))
    for i in range(44):
        prefix.append({'kind': 'host_observation', 'text': 'x' * 50000, 'index': i})
    prior = prefix.project(ProjectionSpec(1, 'count-anchor-v1'))
    assert len(prior.projected_bytes) > 2 * 1024 * 1024
    metadata = json.loads(events[0]['reply']['metadata_json'])
    previous_req = replace(req, call_id=events[0]['call_id'], context=prior)
    metadata['count_anchor']['request_sha256'] = sha(adapter.wire(previous_req))
    events[0]['reply']['metadata_json'] = encode(metadata)
    prefix.append(events[0])
    prefix.append(events[1])
    resumed = replace(req, context=prefix.project(ProjectionSpec(1, 'fixture')))
    assert adapter.count(resumed, execution())['count_basis'] == 'prior_usage_plus_utf8_growth'


@pytest.mark.parametrize('event_size', [70000, 200000])
def test_resumed_anchor_accepts_large_events_from_native_journal(tmp_path, event_size):
    from compute_plane.native_journal import NativeJournal

    adapter, req, _, client, _ = conversation(
        'openai-codex-responses', usage={'input_tokens': 40000}, remaining=16)
    events = req.context.events()
    path = tmp_path / 'journal.jsonl'
    journal = NativeJournal(path)
    journal.append({'kind': 'host_observation', 'text': 'x' * event_size})
    prior = journal.project(ProjectionSpec(1, 'count-anchor-v1'))
    previous = replace(req, call_id=events[0]['call_id'], context=prior,
                       remaining_model_calls=16)
    metadata = json.loads(events[0]['reply']['metadata_json'])
    metadata['count_anchor']['request_sha256'] = sha(adapter.wire(previous))
    events[0]['reply']['metadata_json'] = encode(metadata)
    for event in events:
        journal.append(event)
    before = path.read_bytes()

    # Real custody reload, not an expanded general-purpose in-memory fixture.
    journal = NativeJournal(path)
    resumed = replace(req, context=NativeContext.project(journal))
    wire = adapter.wire(resumed)
    row = adapter.count(resumed, execution())
    assert row['status'] == 'estimated'
    assert row['count_basis'] == 'prior_usage_plus_utf8_growth'
    assert row['calibration']['input_tokens'] == 40000
    assert row['request_sha256'] == sha(wire)
    assert row['client_invocations'] == 0 and len(client.calls) == 1
    policy = ContextPolicy(window=272000, output_reserve=32768, margin=128,
                           allow_estimated_counts=True)
    admitted = NativeContext(policy, counter=adapter.count, summarizer=adapter).prepare(
        resumed, journal, execution(), 1)
    assert admitted.admitted_request_sha256 == row['request_sha256']
    assert adapter.wire(admitted) == wire
    assert path.read_bytes().startswith(before)
    assert not any(r.event()['kind'].startswith('context_summary')
                   for r in journal.snapshot().records)
    assert len(client.calls) == 1


def test_summary_estimates_changes_without_subtracting_old_tokens():
    adapter, req, _, _, _ = conversation()
    row = adapter.count(replace(req, purpose='summary', suffix='Summarize the history.'), execution())
    assert row['count_basis'] == 'prior_usage_plus_utf8_growth'
    assert row['estimated_input_tokens'] >= 40000 + 1024
    # The new summary system prompt is charged, not the whole unchanged history after it.
    assert row['calibration']['growth_bytes'] < 5000


def test_claude_native_summary_reencoding_does_not_double_charge_observed_history():
    adapter, req, _, client, _ = conversation('vendor-cli', 'cli://claude',
        {'input_tokens': 1000, 'cache_read_input_tokens': 108000, 'cache_creation_input_tokens': 1000},
        formatter=True, window=200000, output=16384)
    before = adapter.wire(req)
    assert 'native_tools' in before and set(json.loads(before['prompt'])) == {'messages'}
    summary = replace(req, purpose='summary', suffix='Summarize the supplied history.')
    assert 'messages' in json.loads(adapter.wire(summary)['prompt'])
    calls = len(client.calls)
    counted = adapter.count(summary, execution())
    # The summary contains the same large history already measured at 110k tokens,
    # with a different outer codec; it must fit the original 200k context window.
    assert 110000 <= counted['estimated_input_tokens'] < 200000 - 16384
    assert counted['input_tokens'] is None and counted['status'] == 'estimated'
    assert counted['calibration']['call_id'] == 's/model/1'
    assert len(client.calls) == calls and adapter.wire(req) == before


@pytest.mark.parametrize('bad', [None, {}, {'input_tokens': 1, 'cache_read_input_tokens': None,
                                       'cache_creation_input_tokens': 0}])
def test_claude_summary_codec_change_cannot_calibrate_unknown_usage(bad):
    adapter, req, _, _, _ = conversation('vendor-cli', 'cli://claude', bad or {'unqualified': 1},
                                       formatter=True)
    counted = adapter.count(replace(req, purpose='summary'), execution())
    assert counted['count_basis'] == 'serialized_utf8_bytes' and counted['input_tokens'] is None


def test_duplicate_legacy_observation_is_not_accepted():
    adapter, req, _, _, observations = conversation(legacy=True)
    from compute_plane.native_counting import observation_anchors
    adapter._count_anchors = observation_anchors(observations * 2, adapter.profile.protocol,
                                                 adapter.profile.endpoint, adapter.profile_digest)
    assert adapter.count(req, execution())['count_basis'] == 'serialized_utf8_bytes'


@pytest.mark.parametrize('bad', [None, {}, {'prompt_tokens': True}, {'prompt_tokens': -1}, {'prompt_tokens': 0}])
def test_unknown_counts_are_not_zero_or_measured(bad):
    assert input_tokens('openai-chat', 'http://fixture', bad) is None


@pytest.mark.parametrize('usage,expected', [
    ({'input_tokens':22269,'cache_read_tokens':26446},48715),
    ({'input_tokens':0,'cache_read_tokens':26446},26446),
    ({'input_tokens':22269,'cache_read_tokens':0},22269),
    ({'input_tokens':22269},None), ({'input_tokens':22269,'cache_read_tokens':True},None),
    ({'input_tokens':22269,'cache_read_tokens':-1},None)])
def test_agy_input_includes_cache_and_missing_or_invalid_counts_stay_unknown(usage,expected):
    assert input_tokens('vendor-cli','cli://agy',usage)==expected


def test_unqualified_cli_and_image_counts_remain_explicit_estimates():
    adapter, req, _, _, _ = conversation('vendor-cli', 'cli://agy', {'input_tokens': 40000})
    assert adapter.count(req, execution())['count_basis'] == 'serialized_utf8_bytes'
    from compute_plane.native_counting import estimate
    adapter, req, _, _, _ = conversation()
    assert estimate(adapter, req, adapter.wire(req), has_images=True)['calibration_reason'] == 'image_token_count_not_qualified'
    assert input_tokens('vendor-cli', 'cli://claude', {'input_tokens': 1000}) is None
    assert added_bytes({'x': 'abc', 'removed': 'large'}, {'x': 'abc'}) == 0
    assert added_bytes({'x': 'abc'}, {'x': 'abc中文'}) >= len('中文'.encode())


def test_a_compaction_that_fits_at_the_measured_rate_is_admitted():
    """A compacted request replaced the history its anchor counted. Priced at a byte per token it looked ~3x its
    size, and a summary that fit was thrown away as COMPACTION_TARGET_NOT_REACHED."""
    class Provider:  # charges 0.3 tokens per wire byte
        endpoint = 'http://fixture'

        def post(self, path, body, control):
            usage = {'prompt_tokens': len(encode(body).encode()) * 3 // 10}
            if 'tools' in body:
                return response('openai-chat', usage=usage)
            return {'model': 'fixture-model', 'usage': usage, 'choices': [{'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': 'Objective: fix app.py. Done: read it. Next: edit and verify.'}}]}
    profile = ModelProfile('openai-chat', 'fixture-model', 32768, 6272, 'fixture', 'http://fixture')
    adapter = NativeModelAdapter(profile, Provider(), system='Stable', tool_specs_json=SPECS,
                                 counter_protocol='utf8-envelope-estimate-v1')
    policy = ContextPolicy(window=32768, output_reserve=6272, margin=256, allow_estimated_counts=True,
                           max_compactions=None)
    context, log = NativeContext(policy, counter=adapter.count, summarizer=adapter), ConversationLog()
    for n in range(1, 6):  # each read adds ~10k tokens to a 32k window
        req = replace(request(), call_id=f's/model/{n}', context=NativeContext.project(log))
        reply = adapter.generate(context.prepare(req, log, execution(), None), execution())
        log.append({'kind': 'model_returned', 'call_id': req.call_id, 'reply': asdict(reply)})
        log.append({'kind': 'effect_returned', 'result': {'effect_id': f's/effect/{n}', 'status': 'succeeded',
                    'output_json': json.dumps({'text': 'def handler(x):\n    return x + 1\n' * 1000})}})
    assert sum(r.event()['kind'] == 'context_activated' for r in log.snapshot().records) >= 1
