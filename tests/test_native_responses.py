"""Provider inference only: NativeSession owns tools and continuation."""
import base64
import copy
import io
import json
from dataclasses import replace

import pytest
from tare_tools_agent_runtime import ConversationLog, EffectResult, ModelFailure, NativeSession
from compute_plane.native_models import ModelProfile, NativeModelAdapter
from compute_plane.native_responses_stream import decode
from compute_plane.native_codex_auth import CodexFileAuth
from tests.test_native_models import execution, request, SPECS


def response(name='read', *, partial=False):
    return {'id': 'resp_1', 'model': 'fixture', 'status': 'incomplete' if partial else 'completed',
            'output': [{'id': 'rs_1', 'type': 'reasoning', 'summary': [], 'encrypted_content': 'opaque'},
                       {'id': 'fc_1', 'type': 'function_call', 'status': 'completed', 'call_id': 'call_1',
                        'name': name, 'arguments': json.dumps({'summary': 'verified'} if name == 'tare_deliver'
                                                             else {'path': 'app.py'})}],
            'usage': {'input_tokens': 24, 'output_tokens': 8, 'total_tokens': 32,
                      'input_tokens_details': {'cached_tokens': 12},
                      'output_tokens_details': {'reasoning_tokens': 4}}}


def events(value):
    return [*[{'type': 'response.output_item.done', 'output_index': i, 'item': item}
              for i, item in enumerate(value['output'])],
            {'type': 'response.'+value['status'], 'response': {**value, 'output': []}}]


class Client:
    endpoint = 'http://fixture'
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []
    def post_stream_observed(self, path, body, control, **kwargs):
        self.calls.append((path, copy.deepcopy(body)))
        return decode(iter(events(next(self.replies))), lambda e: None), {}


def adapter(client):
    return NativeModelAdapter(ModelProfile('openai-codex-responses', 'fixture', 32768, 4096,
                                           'fixture', client.endpoint), client,
                              system='Use Tare tools.', tool_specs_json=SPECS,
                              streaming=True, counter_protocol='utf8-envelope-estimate-v1')


def test_session_runs_tools_and_replays_opaque_state_and_results_without_duplicate_on_resume():
    client = Client([response(), response('edit'), response('verify'), response('tare_deliver')])
    model, calls = adapter(client), []
    class Tools:
        def execute(self, req, control):
            calls.append(req.tool_call.name)
            return EffectResult(req.effect_id, 'succeeded', '{"result":"checked"}', True)
    session = NativeSession(session_id='s', instruction='Read edit verify', capabilities=('read','edit','verify'),
                            model=model, tools=Tools(), journal=ConversationLog())
    assert session.run(execution()).status == 'proposed'
    assert calls == ['read', 'edit', 'verify']
    assert all(path == '/responses' for path, body in client.calls)
    wire = client.calls[1][1]
    assert wire['input'][-4:-1] == [*response()['output'],
        {'type': 'function_call_output', 'call_id': 'call_1', 'output': '{"result":"checked"}'}]
    assert wire['input'][-1]['content'].startswith('Runtime budget: 15 model calls remain')
    assert not wire['store'] and wire['include'] == ['reasoning.encrypted_content']
    assert all(tool['strict'] is False for tool in wire['tools'])
    assert 'max_tokens' not in wire and 'messages' not in wire
    assert session.run(execution()).status == 'proposed'
    assert len(client.calls) == 4 and calls == ['read','edit','verify']
    client.replies = iter([response('tare_deliver')])
    assert session.continue_with(execution(), message_id='delta', text='Expand only the changed criterion.')
    assert session.run(execution()).status == 'proposed'
    for (_, previous), (_, current) in zip(client.calls, client.calls[1:]):
        assert current['prompt_cache_key'] == previous['prompt_cache_key']
        assert current['instructions'] == previous['instructions']
        assert current['tools'] == previous['tools']
        assert current['input'][:len(previous['input'])] == previous['input']


def test_cache_accounting_key_is_stable_within_a_session_and_distinct_between_sessions():
    model = adapter(Client([]))
    first = model.wire(request())
    again = model.wire(replace(request(), call_id='different-call'))
    other = model.wire(replace(request(), session_id='other-session'))
    assert first['prompt_cache_key'] == again['prompt_cache_key']
    assert first['prompt_cache_key'] != other['prompt_cache_key']


def test_codex_subscription_cache_identity_keeps_the_supported_request_contract():
    client=Client([])
    def configured(params):
        return NativeModelAdapter(ModelProfile('openai-codex-responses','gpt-6.1-sol',32768,4096,
            'Codex subscription qualification',client.endpoint,parameters_json=json.dumps(params)),client,
            system='Stable',tool_specs_json=SPECS,streaming=True,
            counter_protocol='utf8-envelope-estimate-v1').wire(request())
    wire = configured({})
    assert wire['prompt_cache_key'].startswith('tare-')
    assert 'prompt_cache_options' not in wire
    assert configured({'prompt_cache_key': 'operator-group'})['prompt_cache_key'] == 'operator-group'
    for mode in ('implicit', 'explicit'):
        with pytest.raises(ValueError, match='unsupported Codex Responses parameter'):
            configured({'prompt_cache_options': {'mode': mode}})


def test_count_matches_submitted_wire_and_rejects_mutation_before_dispatch():
    client = Client([response()]); model = adapter(client)
    req = request(); count = model.count(req, execution())
    assert count['status'] == 'estimated' and count['input_tokens'] is None and not client.calls
    with pytest.raises(ModelFailure, match='ADMITTED_REQUEST_CHANGED'):
        model.generate(replace(req, instruction='changed', admitted_request_sha256=count['request_sha256']), execution())
    assert not client.calls
    assert model.generate(replace(req, admitted_request_sha256=count['request_sha256']), execution()).tool_calls
    assert client.calls[0][1] == model.wire(req)
    summary = model.wire(replace(req, purpose='summary'))
    assert not summary.get('tools') or summary['tool_choice'] == 'none'


def test_large_usage_attribution_preserves_receipt_without_overflowing_session(tmp_path):
    from compute_plane.native_journal import NativeJournal, NativeObservationJournal
    value = response('tare_deliver')
    value['usage']['attribution'] = {'items': ['context item ' * 10000]}
    client = Client([value])
    model = adapter(client)
    observations = NativeObservationJournal(tmp_path/'observations.jsonl')
    model._observe = observations.append
    path = tmp_path/'journal.jsonl'
    session = NativeSession(session_id='s', instruction='task', capabilities=('read', 'edit', 'verify'),
                            model=model, tools=None, journal=NativeJournal(path))
    assert session.run(execution()).status == 'proposed'
    reopened = NativeJournal(path)
    replies = [r.event() for r in reopened.snapshot().records if r.event()['kind'] == 'model_returned']
    assert 'context item' not in json.dumps(replies)
    receipts = [r.event() for r in NativeObservationJournal(observations.path).snapshot().records]
    assert receipts[-1]['usage'] == value['usage']
    assert model.observation_failures == 0
    from compute_plane.native_usage import native_usage_report, UsageContract
    contracts = {model.profile_digest: UsageContract(layout='responses-inclusive', source='fixture')}
    usage = native_usage_report(reopened, receipts, contracts=contracts, observation_failures=0)
    assert usage['tokens']['total_tokens'] == {'known_sum': 32, 'unknown_calls': 0}
    assert usage['calls'][0]['raw_usage'] == value['usage']
    changed = copy.deepcopy(receipts)
    changed[-1]['usage']['input_tokens'] += 1
    with pytest.raises(ValueError, match='usage observation conflicts'):
        native_usage_report(reopened, changed, contracts=contracts)
    resumed = NativeSession(session_id='s', instruction='task', capabilities=('read', 'edit', 'verify'),
                            model=model, tools=None, journal=reopened)
    assert resumed.run(execution()).status == 'proposed'
    assert len(client.calls) == 1


def test_incomplete_calls_never_execute_and_model_mismatch_fails():
    assert not adapter(Client([response(partial=True)])).generate(request(), execution()).tool_calls
    value=response(); value['model']='unexpected'
    with pytest.raises(ModelFailure, match='INVALID_PROVIDER_RESPONSE'):
        adapter(Client([value])).generate(request(), execution())


def test_stream_retains_completed_items_when_terminal_output_is_empty_without_done_marker():
    from compute_plane.native_stream import decode as decode_sse
    value=response()
    raw=b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events(value))
    assert decode_sse(io.BytesIO(raw), 'openai-codex-responses', lambda e:None) == value


def test_stream_replays_completed_reasoning_when_terminal_reencrypts_the_same_item():
    value = response()
    stream = events(value)
    terminal = copy.deepcopy(value)
    terminal['output'][0]['encrypted_content'] = 'another-opaque-encryption'
    stream[-1]['response'] = terminal
    original = copy.deepcopy(stream)
    observed = []
    decoded = decode(iter(stream), observed.append)
    # Streaming replay uses output_item.done, as the API specifies. Ciphertexts
    # are opaque; equality is still required for identity, summary and tool data.
    assert decoded == value and decoded['usage'] == terminal['usage']
    assert stream == original
    assert observed == [{'kind': 'usage', 'usage': value['usage']}]


@pytest.mark.parametrize('damage', ['summary', 'identity', 'arguments', 'kind', 'missing_cipher', 'invalid_cipher'])
def test_reencrypted_terminal_still_rejects_changed_content_or_invalid_cipher(damage):
    value = response()
    stream = events(value)
    terminal = copy.deepcopy(value)
    terminal['output'][0]['encrypted_content'] = 'another-opaque-encryption'
    if damage == 'summary': terminal['output'][0]['summary'] = [{'type': 'summary_text', 'text': 'changed'}]
    elif damage == 'identity': terminal['output'][0]['id'] = 'other-reasoning'
    elif damage == 'arguments': terminal['output'][1]['arguments'] = '{"path":"different.py"}'
    elif damage == 'kind': terminal['output'][0]['type'] = 'message'
    elif damage == 'missing_cipher': terminal['output'][0].pop('encrypted_content')
    else: terminal['output'][0]['encrypted_content'] = {'not': 'opaque text'}
    stream[-1]['response'] = terminal
    with pytest.raises(ValueError, match='responses_final_snapshot_changed'):
        decode(iter(stream), lambda e:None)


def test_unknown_progress_events_are_ignored_and_never_replace_the_terminal():
    # A keepalive (or an event a newer API adds) mid-stream ended a 30 s gpt-6-luna call as unsupported on 2026-10-07.
    value, extra = response(), [{'type': 'keepalive'}, {'type': 'response.newer_progress', 'sequence_number': 3}]
    stream = events(value)
    assert decode(iter(stream[:1] + extra + stream[1:]), lambda e: None) == value
    with pytest.raises(ValueError, match='responses_missing_terminal'):
        decode(iter(stream[:-1] + extra), lambda e: None)


@pytest.mark.parametrize('damage', ['truncate','duplicate','changed','open','hosted','wrong_response'])
def test_bad_stream_never_authorizes_an_effect(damage):
    stream=events(response())
    if damage=='truncate': stream.pop()
    if damage=='duplicate': stream.insert(1,copy.deepcopy(stream[0]))
    if damage=='changed': stream[-1]['response']['output']=[{'id':'different'}]
    if damage=='open': stream[0]['type']='response.output_item.added'
    if damage=='hosted': stream[0]['item']['type']='web_search_call'
    if damage=='wrong_response': stream.insert(0,{'type':'response.created','response':{'id':'different'}})
    with pytest.raises(ValueError): decode(iter(stream),lambda e:None)


@pytest.mark.parametrize('damage', [None, 'wrong_id', 'wrong_status', 'missing_id', 'generic_error'])
def test_terminal_failure_closes_call_without_executing_partial_tools(damage):
    from compute_plane.native_models import NativeHttpClient
    from tests.test_native_streaming import serving, sse, real_control
    from tare_tools_agent_runtime import NativeBudget

    failed = {'type': 'response.failed', 'response': {'id': 'resp_1', 'status': 'failed',
              'model': 'fixture', 'usage': {'input_tokens': 42, 'output_tokens': 3},
              'error': {'message': 'private provider detail'}}}
    if damage == 'wrong_id': failed['response']['id'] = 'other'
    if damage == 'wrong_status': failed['response']['status'] = 'in_progress'
    if damage == 'missing_id': failed['response'].pop('id')
    if damage == 'generic_error': failed = {'type': 'error', 'message': 'private provider detail'}
    parts = [sse({'type': 'response.created', 'response': {'id': 'resp_1'}}),
             sse({'type': 'response.output_item.added', 'output_index': 0,
                  'item': {'id': 'fc_1', 'type': 'function_call', 'status': 'in_progress'}}),
             sse({'type': 'response.function_call_arguments.delta', 'delta': '{"path":"partial'}),
             sse(failed)]
    with serving(parts=parts) as (url, seen, *_):
        rows, effects = [], []
        model = adapter(NativeHttpClient(url))
        model._observe = rows.append
        class Tools:
            def execute(self, req, control):
                effects.append(req)
                pytest.fail('Failed inference must never execute partial tools')
        journal = ConversationLog()
        session = NativeSession(session_id='s', instruction='task', capabilities=('read','edit','verify'),
                                model=model, tools=Tools(), journal=journal,
                                budget=NativeBudget(max_model_retries=2))
        outcome = session.run(real_control())
        kinds = [r.event()['kind'] for r in journal.snapshot().records]
        assert len(seen) == 1 and not effects, outcome  # Even with retry budget available.
        assert 'private provider detail' not in json.dumps(rows)
        if damage is None:
            assert outcome.status != 'reconciliation_required'
            assert outcome.pending_id is None
            assert 'model_failed' in kinds and 'invocation_interrupted' not in kinds
            assert rows[0]['error'] == 'PROVIDER_RESPONSE_FAILED'
            assert rows[0]['usage'] == {'input_tokens': 42, 'output_tokens': 3}
        else:
            assert outcome.status == 'reconciliation_required'
            assert 'invocation_interrupted' in kinds and 'model_failed' not in kinds
            assert rows[0]['usage'] is None
            session.run(real_control())
            assert len(seen) == 1


def login(tmp_path, *, exp=2000, mode='chatgpt', account='account'):
    payload=base64.urlsafe_b64encode(json.dumps({'exp':exp}).encode()).decode().rstrip('=')
    path=tmp_path/'auth.json'
    path.write_text(json.dumps({'auth_mode':mode,'tokens':{'access_token':'h.'+payload+'.signature',
                                                         'account_id':account}}))
    return path


def test_auth_read_only_atomic_account_binding_and_no_api_fallback(tmp_path):
    path=login(tmp_path); before=path.read_bytes()
    auth=CodexFileAuth(path,account_id='account',clock=lambda:1000)
    assert auth.headers()['ChatGPT-Account-Id']=='account' and path.read_bytes()==before
    login(tmp_path,account='other')
    with pytest.raises(ModelFailure,match='CODEX_ACCOUNT_CHANGED'): auth.headers()
    login(tmp_path,exp=1000)
    with pytest.raises(ModelFailure,match='CODEX_LOGIN_EXPIRED'): auth.headers()
    login(tmp_path,mode='apikey')
    with pytest.raises(ModelFailure,match='CODEX_CHATGPT_LOGIN_REQUIRED'): auth.headers()
    path.write_text('sensitive-invalid-credential')
    with pytest.raises(ModelFailure,match='CODEX_LOGIN_UNAVAILABLE') as exc: auth.headers()
    assert 'sensitive' not in str(exc.value)


def test_auth_failure_has_proof_of_no_dispatch_and_usage_stays_unknown(tmp_path):
    from compute_plane.native_codex_auth import client
    from compute_plane.native_usage import native_usage_report
    rows=[]; model=adapter(client(tmp_path/'missing.json',account_id='account'))
    model._observe=rows.append
    journal=ConversationLog()
    result=NativeSession(session_id='s',instruction='task',capabilities=('read','edit','verify'),
                         model=model,tools=None,journal=journal).run(execution())
    assert result.status != 'proposed'
    report=native_usage_report(journal,rows,contracts={})
    assert report['coverage']['not_started']==1
    assert report['transport']['http_attempts_upper_bound_for_observed_exchanges']==0
    assert rows[0]['usage'] is None


def test_responses_usage_is_per_call_inclusive_and_missing_is_unknown():
    from compute_plane.native_usage import normalize_usage, UsageContract
    contract=UsageContract('responses-inclusive','fixture')
    value=normalize_usage(response()['usage'],contract)
    assert value['tokens']=={'input_tokens':24,'uncached_input_tokens':12,'cache_read_tokens':12,
                             'cache_write_tokens':0,'output_tokens':8,'reasoning_tokens':4,'total_tokens':32}
    assert normalize_usage(None,contract)['tokens']['input_tokens'] is None


def test_real_worker_responses_cancel_keeps_remote_stop_unknown(monkeypatch):
    from compute_plane.native_models import NativeHttpClient, NativeTransportInterrupted
    from tests import test_native_streaming as fixture
    parts=[fixture.sse({'type':'response.output_text.delta','delta':'provisional'})]
    parts += [fixture.sse(e) for e in events(response())]
    monkeypatch.setattr(fixture,'chat_parts',lambda tool:parts)
    with fixture.serving(gate=True) as (url,seen,release,first):
        ctl=fixture.real_control()
        with pytest.raises(NativeTransportInterrupted) as error:
            NativeHttpClient(url).post_stream_observed('/responses',{},ctl,
                protocol='openai-codex-responses',on_event=lambda e:ctl.cancellation.cancel())
        assert error.value.transport_observation['cleanup_verified']
        assert error.value.transport_observation['remote_stop']=='unknown'
