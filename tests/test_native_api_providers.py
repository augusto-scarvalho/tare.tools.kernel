"""Offline API-provider qualification: fake replies and bounded loopback HTTP only."""
import copy
import io
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from tare_tools_agent_runtime import ConversationLog, EffectResult, ModelFailure, NativeSession
from compute_plane.native_models import ModelProfile, NativeHttpClient, NativeModelAdapter
from compute_plane.native_usage import UsageContract, normalize_usage, price_usage, deployment_contract
from compute_plane.native_counting import input_tokens
from compute_plane.native_stream import decode
from compute_plane.native_anthropic import to_messages
from tests.test_native_models import Client, SPECS, execution, request, response as chat_response
from tests.test_native_responses import response as responses_response, Client as StreamClient


def anthropic(name='read', *, stop='tool_use'):
    return {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'fixture',
            'stop_reason': stop, 'content': [
                {'type': 'thinking', 'thinking': 'private', 'signature': 'opaque',
                 'cache_control': {'type': 'ephemeral'}},
                {'type': 'redacted_thinking', 'data': 'encrypted'},
                {'type': 'tool_use', 'id': 'call_1', 'name': name,
                 'input': {'summary': 'Observed fixtures only.'} if name == 'tare_deliver' else {'path': 'app.py'}}],
            'usage': {'input_tokens': 20, 'output_tokens': 7,
                      'cache_read_input_tokens': 12, 'cache_creation_input_tokens': 4}}


def model(protocol, client, **kwargs):
    profile_args = kwargs.pop('profile_args', {})
    return NativeModelAdapter(ModelProfile(protocol, 'fixture', 32768, 1024, 'fixture',
        client.endpoint, **profile_args), client, system='Stable system', tool_specs_json=SPECS,
        counter_protocol='utf8-envelope-estimate-v1', **kwargs)


@pytest.mark.parametrize('params', [
    {}, {'thinking': {'type': 'adaptive'}},
    {'thinking': {'type': 'enabled', 'budget_tokens': 512}},
    {'thinking': {'type': 'disabled'}}, {'thinking': {}},
    *({'thinking': value} for value in [None, False, True, 'disabled', [], 0]),
])
def test_anthropic_final_delivery_choice_preserves_thinking_and_budget(params):
    params = {**params, 'temperature': 0.7}
    profile_args = {'parameters_json': json.dumps(params)}
    req = replace(request(), remaining_model_calls=1)
    baseline = model('anthropic-messages', Client([]), profile_args=profile_args).wire(req)
    wire = model('anthropic-messages', Client([]), profile_args=profile_args,
                 force_final_delivery=True).wire(req)
    disabled = params.get('thinking') == {'type': 'disabled'}
    expected = {'type': 'tool', 'name': 'tare_deliver'} if disabled else {'type': 'auto'}
    assert wire['tool_choice'] == expected
    assert wire == {**baseline, 'tool_choice': expected}
    assert all(wire[key] == value for key, value in params.items())
    assert ('thinking' in wire) == ('thinking' in params)
    assert wire['messages'][-1]['content'][-1] == {'type': 'text', 'text': (
        'Runtime budget: 1 model calls remain INCLUDING this call. '
        'Summary calls also consume this budget. Batch independent reads or searches when useful; '
        'reuse observed sources. Reserve calls for necessary verification and final delivery. '
        'Do not skip checks or claim completion to fit the budget; report unfinished work honestly.')}


@pytest.mark.parametrize('protocol,expected', [
    ('openai-chat', {'type': 'function', 'function': {'name': 'tare_deliver'}}),
    ('openai-responses', {'type': 'function', 'name': 'tare_deliver'}),
    ('openai-codex-responses', {'type': 'function', 'name': 'tare_deliver'}),
])
def test_openai_final_delivery_keeps_named_tool(protocol, expected):
    req = replace(request(), remaining_model_calls=1)
    streaming = protocol == 'openai-codex-responses'
    client = StreamClient([]) if streaming else Client([])
    baseline = model(protocol, client, streaming=streaming).wire(req)
    wire = model(protocol, client, streaming=streaming, force_final_delivery=True).wire(req)
    assert wire['tool_choice'] == expected
    assert wire == {**baseline, 'tool_choice': expected}


@pytest.mark.parametrize('protocol', [
    'anthropic-messages', 'openai-chat', 'openai-responses', 'openai-codex-responses'])
@pytest.mark.parametrize('force,remaining,purpose', [
    (False, 1, 'model'), (True, 2, 'model'), (True, None, 'model'),
    (True, 1, 'summary'),
])
def test_final_delivery_choice_only_applies_to_last_forced_model_call(protocol, force, remaining, purpose):
    streaming = protocol == 'openai-codex-responses'
    client = StreamClient([]) if streaming else Client([])
    params = {'thinking': {'type': 'disabled'}} if protocol == 'anthropic-messages' else {}
    adapter = model(protocol, client, streaming=streaming, force_final_delivery=force,
        profile_args={'parameters_json': json.dumps(params)})
    wire = adapter.wire(replace(request(), remaining_model_calls=remaining, purpose=purpose))
    if purpose == 'summary':
        assert 'tool_choice' not in wire and 'tools' not in wire
    else:
        assert wire['tool_choice'] == ({'type': 'auto'} if protocol == 'anthropic-messages' else 'auto')


@pytest.mark.parametrize('effort', ['low', 'medium', 'high', 'xhigh', 'max'])
def test_anthropic_effort_reaches_wire_without_changing_cache_history_or_output_limit(effort):
    client = Client([anthropic()])
    base = {'cache_control': {'type': 'ephemeral'}}
    baseline = model('anthropic-messages', client, profile_args={'parameters_json': json.dumps(base)}).wire(request())
    adapter = model('anthropic-messages', client,
                    profile_args={'parameters_json': json.dumps({**base, 'output_config': {'effort': effort}})})
    wire = adapter.wire(request())
    assert wire == {**baseline, 'output_config': {'effort': effort}}
    assert adapter.generate(request(), execution()).tool_calls
    assert client.calls == [('/messages', wire)]


@pytest.mark.parametrize('output', [None, {}, [], 'high', {'effort': 'ultra'}, {'effort': True}, {'effort': []},
    {'effort': 'high', 'format': {}}, {'effort': 'high', 'tool_choice': 'auto'}])
def test_anthropic_bad_output_configuration_is_refused_before_dispatch(output):
    client = Client([])
    with pytest.raises(ValueError, match='output_config.effort'):
        model('anthropic-messages', client, profile_args={'parameters_json': json.dumps({'output_config': output})})
    assert client.calls == []


@pytest.mark.parametrize('streaming', [False, True])
def test_openai_responses_limit_stream_cache_and_existing_decoder(streaming):
    client = StreamClient([responses_response()]) if streaming else Client([responses_response()])
    adapter = model('openai-responses', client, streaming=streaming)
    first = adapter.wire(request())
    assert first['max_output_tokens'] == 1024 and first['stream'] is streaming
    assert 'max_tokens' not in first and not first['store']
    assert 1 <= len(first['prompt_cache_key']) <= 64
    assert first['prompt_cache_key'] == adapter.wire(replace(request(), call_id='s/model/2'))['prompt_cache_key']
    assert first['prompt_cache_key'] != adapter.wire(replace(request(), session_id='other'))['prompt_cache_key']
    resumed = model('openai-responses', client, streaming=streaming)
    assert first['prompt_cache_key'] == resumed.wire(request())['prompt_cache_key']
    configured = model('openai-responses', client, streaming=streaming,
        profile_args={'parameters_json': json.dumps({'prompt_cache_key': 'operator-affinity'})})
    assert configured.wire(request())['prompt_cache_key'] == 'operator-affinity'
    assert adapter.generate(request(), execution()).tool_calls
    assert client.calls[0] == ('/responses', first)


@pytest.mark.parametrize('protocol', ['anthropic-messages', 'openai-responses', 'openai-chat'])
def test_multiturn_replays_original_blocks_signatures_and_results(protocol):
    def raw(name):
        if protocol == 'anthropic-messages':
            return anthropic(name)
        if protocol == 'openai-responses':
            return responses_response(name)
        value = chat_response('openai-chat', name)
        value['model'] = 'fixture'
        value['choices'][0]['message']['extra_content'] = {'google': {'thought_signature': 'message-opaque'}}
        value['choices'][0]['message']['tool_calls'][0]['extra_content'] = {
            'google': {'thought_signature': 'call-opaque', 'unknown': [1, {'x': True}]}}
        return value
    client = Client([raw('read'), raw('edit'), raw('tare_deliver')])
    adapter = model(protocol, client, profile_args=(
        {'cache_system': True, 'cache_support_source': 'Anthropic API fixture evidence'}
        if protocol == 'anthropic-messages' else {}))
    effects = []
    class Tools:
        def execute(self, req, control):
            effects.append(req.tool_call.name)
            return EffectResult(req.effect_id, 'failed' if len(effects) == 2 else 'succeeded',
                                '{"observed":"offline"}', len(effects) != 2)
    session = NativeSession(session_id='s', instruction='Inspect', capabilities=('read', 'edit', 'verify'),
        model=adapter, tools=Tools(), journal=ConversationLog())
    outcome = session.run(execution())
    assert outcome.status == 'proposed', (outcome, session.journal.snapshot())
    assert effects == ['read', 'edit']
    assert session.run(execution()).status == 'proposed' and len(client.calls) == 3
    wire = client.calls[2][1]
    if protocol == 'anthropic-messages':
        assert wire['system'] == [{'type': 'text', 'text': 'Stable system', 'cache_control': {'type': 'ephemeral'}}]
        assert wire['messages'][1]['content'] == raw('read')['content']
        results = [b for m in wire['messages'] for b in m['content'] if b['type'] == 'tool_result']
        assert [b['is_error'] for b in results] == [False, True]
        assert all(b['content'] == '{"observed":"offline"}' and b['tool_use_id'] == 'call_1' for b in results)
    elif protocol == 'openai-responses':
        assert wire['input'][2:4] == raw('read')['output']
        assert wire['input'][4]['type'] == 'function_call_output'
    else:
        assert wire['messages'][3] == raw('read')['choices'][0]['message']
        assert wire['messages'][4]['tool_call_id'] == 'provider-id'


@pytest.mark.parametrize('mutation', ['wrong_model', 'hosted', 'malformed', 'missing_stop', 'duplicate', 'unsigned'])
def test_anthropic_invalid_response_rejects_whole_batch(mutation):
    value = anthropic()
    if mutation == 'wrong_model': value['model'] = 'other'
    if mutation == 'hosted': value['content'].append({'type': 'server_tool_use', 'name': 'web_search'})
    if mutation == 'malformed': value['content'][-1]['input'] = 'not an object'
    if mutation == 'missing_stop': value.pop('stop_reason')
    if mutation == 'duplicate': value['content'].append(copy.deepcopy(value['content'][-1]))
    if mutation == 'unsigned': value['content'][0].pop('signature')
    rows = []
    adapter = model('anthropic-messages', Client([value]), observe=rows.append)
    with pytest.raises(ModelFailure, match='INVALID_PROVIDER_RESPONSE'):
        adapter.generate(request(), execution())
    assert rows[-1]['status'] == 'invalid_response' and rows[-1]['usage'] == value['usage']


def test_anthropic_truncated_batch_has_no_effects_and_streaming_refused():
    client = Client([anthropic(stop='max_tokens')])
    reply = model('anthropic-messages', client).generate(request(), execution())
    assert not reply.tool_calls and not reply.proposes_delivery
    assert json.loads(reply.metadata_json)['recovery']['code'] == 'TRUNCATED_TOOL_CALL'
    with pytest.raises(ValueError):
        model('anthropic-messages', StreamClient([]), streaming=True)
    with pytest.raises(ValueError, match='support evidence'):
        model('anthropic-messages', Client([]), profile_args={'cache_system': True})


def test_anthropic_usage_missing_cache_is_unknown_not_zero_or_free():
    contract = UsageContract('anthropic-exclusive', 'API fixture')
    raw = anthropic()['usage']
    known = normalize_usage(raw, contract)['tokens']
    assert known['input_tokens'] == 36 and known['uncached_input_tokens'] == 20
    assert known['cache_read_tokens'] == 12 and known['cache_write_tokens'] == 4
    assert known['total_tokens'] == 43 and known['reasoning_tokens'] is None
    missing = {k: v for k, v in raw.items() if k != 'cache_read_input_tokens'}
    normalized = normalize_usage(missing, contract)
    assert normalized['tokens']['cache_read_tokens'] is None
    assert normalized['tokens']['input_tokens'] is None and normalized['tokens']['total_tokens'] is None
    assert input_tokens('anthropic-messages', '', missing) is None
    assert price_usage(normalized, None)['known_amount'] is None
    assert deployment_contract({'native_profile': {'protocol': 'anthropic-messages'}})['layout'] == 'anthropic-exclusive'


def test_gemini_stream_retains_opaque_metadata_without_interpretation():
    extension = {'google': {'thought_signature': 'opaque', 'future': [1, {'a': 'b'}]}}
    chunks = [
        {'model': 'fixture', 'choices': [{'index': 0, 'delta': {'role': 'assistant',
            'extra_content': extension, 'tool_calls': [{'index': 0, 'id': 'c1', 'type': 'function',
                'extra_content': extension, 'function': {'name': 'read', 'arguments': '{"path":'}}]}}]},
        {'model': 'fixture', 'choices': [{'index': 0, 'delta': {'tool_calls': [
            {'index': 0, 'function': {'arguments': '"app.py"}'}}]}, 'finish_reason': 'tool_calls'}]}]
    wire = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
    result = decode(io.BytesIO(wire.encode()), 'openai-chat', lambda e: None)
    assistant = result['choices'][0]['message']
    assert assistant['extra_content'] == extension
    assert assistant['tool_calls'][0]['extra_content'] == extension
    assert json.loads(assistant['tool_calls'][0]['function']['arguments']) == {'path': 'app.py'}


@pytest.mark.parametrize('protocol', ['anthropic-messages', 'openai-responses'])
def test_loopback_payload_path_bearer_headers_and_output_limit(protocol):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length']))), dict(self.headers)))
            if protocol == 'openai-responses' and len(seen[-1][1].get('prompt_cache_key', '')) > 64:
                # The real API rejected Tare's 69-character generated key with HTTP 400.
                self.send_error(400)
                return
            data = json.dumps(anthropic('tare_deliver') if protocol == 'anthropic-messages'
                              else responses_response('tare_deliver')).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        headers = {'anthropic-version': '2023-06-01', 'anthropic-workspace-id': 'wrk_fixture'} if protocol == 'anthropic-messages' else {}
        client = NativeHttpClient(f'http://127.0.0.1:{server.server_port}/v1',
                                  credential=lambda: 'fixture-key', request_headers=lambda: headers)
        rows = []
        adapter = model(protocol, client, observe=rows.append)
        assert adapter.generate(request(), execution()).proposes_delivery
        path, body, received = seen[0]
        assert path == '/v1/' + ('messages' if protocol == 'anthropic-messages' else 'responses')
        assert body['max_tokens' if protocol == 'anthropic-messages' else 'max_output_tokens'] == 1024
        received = {k.lower(): v for k, v in received.items()}
        assert received['authorization'] == 'Bearer fixture-key'
        assert all(received[k] == v for k, v in headers.items())
        assert 'chatgpt-account-id' not in received and 'fixture-key' not in json.dumps(rows)
    finally:
        server.shutdown(); server.server_close(); thread.join(2)


@pytest.mark.parametrize('headers', [
    {'anthropic-version': 'bad'}, {'anthropic-workspace-id': 'invalid space'},
    {'anthropic-version': '2023-06-01\r\nx-secret: value'}, {'x-api-key': 'no'}])
def test_invalid_headers_refused_before_http(headers):
    client = NativeHttpClient('http://127.0.0.1:1', request_headers=lambda: headers)
    with pytest.raises(ValueError): client.post('/messages', {}, execution())


def test_cancelled_api_request_does_not_read_credentials():
    control = execution(); control.cancellation.cancel()
    client = NativeHttpClient('http://127.0.0.1:1', credential=lambda: pytest.fail('credential after cancel'))
    with pytest.raises(InterruptedError): client.post('/messages', {}, control)


def test_anthropic_inline_images_and_explicit_cache_preserved():
    wire = {'model': 'fixture', 'max_tokens': 12, 'stream': False, 'cache_control': {'type': 'ephemeral'},
            'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YWJj'}}]}]}
    result = to_messages(wire)
    assert result['messages'][0]['content'][0] == {'type': 'image', 'source': {
        'type': 'base64', 'media_type': 'image/png', 'data': 'YWJj'}}
    assert result['cache_control'] == {'type': 'ephemeral'} and 'cache_control' not in result['system'][0]
