"""CLI inference proposals must never execute through a vendor harness."""
import json
from dataclasses import replace

import pytest
from tare_tools_agent_runtime import ConversationLog, ConversationRejected, EffectResult, ModelFailure, NativeSession
from compute_plane.native_models import ModelProfile, NativeModelAdapter, sha
from compute_plane.native_cli import decode_agy, decode_claude, decode_claude_native, proposal
from tests.test_native_models import execution, request, SPECS


def test_claude_reference_accepts_updates_but_old_executable_cannot_run_opus_55(tmp_path, monkeypatch):
    from compute_plane import native_cli
    executable = str(tmp_path / 'claude.exe')
    for version in ('2.1.278', '2.1.280', '2.1.282'):
        native_cli.VendorCliClient('cli://claude', executable=executable, version=version)
    with pytest.raises(ValueError, match='release CLI version required'):
        native_cli.VendorCliClient('cli://claude', executable=executable, version='unknown')

    def unexpected_dispatch(argv, **kwargs):
        assert argv[-1] == '--version'
        return {'cleanup_verified': True, 'exit_code': 0}, b'2.1.278 (Claude Code)', b''

    monkeypatch.setattr(native_cli, 'run_pipe_process', unexpected_dispatch)
    old = native_cli.VendorCliClient('cli://claude', executable=executable, version='2.1.278')
    with pytest.raises(ModelFailure, match='CLI_INCOMPATIBLE_VERSION'):
        old.post_observed('', {'model': 'claude-opus-5-5'}, execution())


@pytest.mark.parametrize('actual', ['2.1.278', '2.1.280', '2.1.292'])
@pytest.mark.parametrize('structured', [False, True])
def test_haiku_55_rejects_old_cli_before_inference_even_with_structured_output(tmp_path, monkeypatch, actual, structured):
    from compute_plane import native_cli
    calls = []

    def probe_only(argv, **kwargs):
        calls.append(argv)
        assert argv[-1] == '--version'
        return {'cleanup_verified': True, 'exit_code': 0}, (actual+' (Claude Code)').encode(), b''

    monkeypatch.setattr(native_cli, 'run_pipe_process', probe_only)
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.280')
    body = {'model': 'claude-haiku-5-5', **({'response_schema': {'type': 'object'}} if structured else {})}
    with pytest.raises(ModelFailure, match='CLI_INCOMPATIBLE_VERSION') as caught:
        client.post_observed('', body, execution())
    assert caught.value.not_started and len(calls) == 1


@pytest.mark.parametrize('vendor,reference,actual', [
    ('claude', '2.1.280', '2.1.282'), ('claude', '2.1.282', '2.1.282'),
    ('agy', '1.2.7', '1.2.11'), ('agy', '1.2.11', '1.2.11')])
@pytest.mark.parametrize('fault', [None, 'missing_switch', 'timeout', 'bad_version', 'boundary'])
def test_updated_cli_checks_capabilities_and_preserves_boundary(
        tmp_path, monkeypatch, vendor, reference, actual, fault):
    from compute_plane import native_cli
    endpoint = 'cli://' + vendor
    physical = dict(started=True, cleanup_verified=True, exit_code=0,
                    cancelled=False, timeout=False, output_limit=False)
    calls = []
    def process(argv, **options):
        calls.append(argv)
        if '--version' in argv:
            return physical, (actual if fault != 'bad_version' else 'unknown').encode(), b''
        if '--help' in argv:
            switches = native_cli.CLI_SWITCHES[endpoint] - ({'--model'} if fault == 'missing_switch' else set())
            help_text = ' '.join(switches).encode()
            return (physical | {'timeout': fault == 'timeout'},
                    help_text if vendor == 'claude' else b'', help_text if vendor == 'agy' else b'')
        if vendor == 'claude':
            rows = structured_claude_rows()
            if fault == 'boundary': rows[0]['tools'].append('Bash')
        else:
            hooks = [{'phase': phase, 'model': 'fixture'} for phase in ('pre', 'post', 'stop')]
            hooks[-1]['termination'] = 'TERMINAL_CUSTOM_HOOK'
            if fault == 'boundary': hooks.pop(0)
            (options['cwd']/'boundary.jsonl').write_text(
                '\n'.join(json.dumps(h) for h in hooks), encoding='utf-8')
            rows = [{'event': 'result', 'result': {'status': 'SUCCESS',
                'structured_output': {'text': 'ok', 'actions': []}}}]
        return physical, '\n'.join(json.dumps(r) for r in rows).encode(), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient(endpoint, executable=str(tmp_path/(vendor+'.exe')), version=reference)
    body = native_cli.cli_request({'model': 'fixture', 'messages': [],
        'tools': [{'function': {'name': 'read', 'parameters': {}}}]}, structured=True, formatter=vendor == 'claude')
    if fault:
        # An updated CLI that breaks the decoded shape fails its one probe: the real request is never sent.
        code = ('CLI_VERSION_UNAVAILABLE' if fault == 'bad_version' else
                'CLI_QUALIFICATION_FAILED' if fault == 'boundary' else 'CLI_CAPABILITY_UNAVAILABLE')
        with pytest.raises(ModelFailure, match=code) as caught:
            client.post_observed('', body, execution())
        assert caught.value.not_started
        if fault == 'boundary':
            assert client.qualification[actual + ' fixture']['probe_code'] == 'CLI_BOUNDARY_UNCONFIRMED'
            assert caught.value.transport_observation['qualification']['status'] == 'failed'
            dispatched = len(calls)
            with pytest.raises(ModelFailure, match='CLI_QUALIFICATION_FAILED'):
                client.post_observed('', body, execution())  # recorded: no second probe, only --version
            assert [argv[-1] for argv in calls[dispatched:]] == ['--version']
        else:
            assert all(argv[-1] in {'--version', '--help'} for argv in calls)
        return
    receipts = [client.post_observed('', body, execution())[1] for _ in range(2)]
    for receipt in receipts:
        assert receipt['client_version'] == actual
        assert receipt['reference_client_version'] == reference
        assert receipt['task_tools_executed'] == 0
    assert receipts[0]['qualification']['status'] == 'probed' and 'qualification' not in receipts[1]
    assert client.qualification[actual + ' fixture']['status'] == 'ok'
    assert sum('--help' in argv for argv in calls) == 1
    assert sum('--version' in argv for argv in calls) == 4  # first call, its probe, the real request, second call
    assert client.version == reference


def answer(name='read'):
    return json.dumps({'text': '', 'tool_calls': [{'id': 'p1', 'name': name, 'arguments':
        {'summary': 'checked'} if name == 'tare_deliver' else {'path': 'a.py'}}]})


def structured_claude_rows():
    # Qualified 2.1.280 shape: one API message, formatter acknowledgement,
    # then result. num_turns counts the acknowledgement, not two inferences.
    value = {'text': 'Read source', 'actions': [{'name': 'read', 'arguments_json': '{"path":"a.py"}'}]}
    return [
        {'type': 'system', 'subtype': 'init', 'tools': ['StructuredOutput'], 'mcp_servers': []},
        {'type': 'assistant', 'message': {'id': 'msg-1', 'model': 'fixture', 'content': [
            {'type': 'tool_use', 'id': 'format-1', 'name': 'StructuredOutput', 'input': value}]}},
        {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'format-1',
                                                'content': 'Structured output provided successfully'}]}},
        {'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 2,
         'result': json.dumps(value), 'structured_output': value,
         'usage': {'input_tokens': 2, 'iterations': [{'type': 'message'}]}, 'modelUsage': {'fixture': {}}},
    ]


@pytest.mark.parametrize('damage', [None, 'task_tool', 'missing_schema', 'mismatched_data', 'extra_model',
    'extra_inference', 'extra_iteration', 'no_iterations', 'missing_ack', 'bad_ack', 'extra_format',
    'bad_args', 'extra_field', 'mcp', 'enabled_task_tool', 'missing_id'])
def test_claude_schema_accepts_only_one_data_formatter_invocation(damage):
    rows = structured_claude_rows()
    block = rows[1]['message']['content'][0]
    if damage == 'task_tool': block['name'] = 'Bash'
    if damage == 'missing_schema': rows[-1].pop('structured_output')
    if damage == 'mismatched_data': rows[-1]['structured_output'] = {'text': 'different', 'actions': []}
    if damage == 'extra_model': rows[-1]['modelUsage']['other'] = {}
    if damage == 'extra_inference': rows.insert(2, {'type': 'assistant', 'message': {
        'id': 'msg-2', 'model': 'fixture', 'content': [{'type': 'text', 'text': 'extra'}]}})
    if damage == 'extra_iteration': rows[-1]['usage']['iterations'] *= 2
    if damage == 'no_iterations': rows[-1]['usage'].pop('iterations')
    if damage == 'missing_ack': rows.pop(2)
    if damage == 'bad_ack': rows[2]['message']['content'][0]['tool_use_id'] = 'other'
    if damage == 'extra_format': rows[1]['message']['content'].append(dict(block))
    if damage == 'bad_args': block['input']['actions'][0]['arguments_json'] = '[]'
    if damage == 'extra_field': block['input']['unexpected'] = True
    if damage == 'mcp': rows[0]['mcp_servers'] = ['remote']
    if damage == 'enabled_task_tool': rows[0]['tools'].append('Read')
    if damage == 'missing_id': rows[1]['message'].pop('id')
    if damage:
        with pytest.raises((ModelFailure, ValueError, ConversationRejected)):
            decode_claude(rows, 'fixture', structured=True)
    else:
        reply = decode_claude(rows, 'fixture', structured=True)
        assert reply['usage'] == rows[-1]['usage']
        assert json.loads(reply['choices'][0]['message']['tool_calls'][0]['function']['arguments']) == {'path': 'a.py'}


def test_claude_total_estimate_survives_proposal_decode():
    rows = structured_claude_rows()
    rows[-1]['total_cost_usd'] = .034
    reply = decode_claude(rows, 'fixture', structured=True)
    assert reply['usage']['total_cost_usd'] == .034
    assert 'total_cost_usd' not in rows[-1]['usage']


def native_claude_rows(names, calls=(('read', {'path': 'a.py'}),), text=''):
    """Claude Code 2.1.316 shape (2026-11-02 live capture): one API message streamed as one event per block, every
    native call refused by Tare's hook, and the CLI's own denial list."""
    uses = [{'type': 'tool_use', 'id': f'toolu_{i}', 'name': 'mcp__tare__' + n, 'input': a}
            for i, (n, a) in enumerate(calls, 1)]
    rows = [{'type': 'system', 'subtype': 'init', 'tools': ['mcp__tare__' + n for n in names],
             'mcp_servers': [{'name': 'tare', 'status': 'connected', 'source': 'dynamic'}]}]
    blocks = [{'type': 'thinking', 'thinking': 'private'}, *([{'type': 'text', 'text': text}] if text else []), *uses]
    rows += [{'type': 'assistant', 'message': {'id': 'msg-1', 'model': 'fixture', 'content': [b]}} for b in blocks]
    rows += [{'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': u['id'], 'is_error': True,
                                                       'content': 'Tare executes this action.'}]}} for u in uses]
    rows.append({'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 1 + len(uses), 'result': text,
                 'usage': {'input_tokens': 2, 'iterations': [{'type': 'message'}]},
                 'permission_denials': [{'tool_name': u['name'], 'tool_use_id': u['id'], 'tool_input': u['input']}
                                        for u in uses]})
    return rows, [{'phase': 'claude', 'tool': u['name'], 'input': u['input']} for u in uses]


def native_process(rows_hooks, invocations=None, physical=None):
    physical = physical or dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False,
                                output_limit=False)
    def process(argv, **options):
        if '--version' in argv:
            return {**physical, 'exit_code': 0}, b'2.1.280 (Claude Code)', b''
        if invocations is not None:
            invocations.append((argv, options))
        rows, hooks = rows_hooks(json.loads((options['cwd']/'tools.json').read_text(encoding='utf-8')))
        (options['cwd']/'boundary.jsonl').write_text(''.join(json.dumps(h)+'\n' for h in hooks), encoding='utf-8')
        return physical, '\n'.join(json.dumps(r) for r in rows).encode(), b''
    return process


def test_claude_native_transport_declares_only_tare_tools_and_records_refused_calls(tmp_path, monkeypatch):
    from compute_plane import native_cli
    invocations = []
    monkeypatch.setattr(native_cli, 'run_pipe_process', native_process(
        lambda tools: native_claude_rows([t['name'] for t in tools], calls=(('read', {'path': 'a.py'}),
                                                                            ('verify', {'path': 'tests'}))),
        invocations))
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.280')
    model = adapter(client)
    req = request()
    body = model.wire(req)
    assert 'response_schema' not in body and 'tools' not in json.loads(body['prompt'])
    assert {t['name'] for t in body['native_tools']} >= {'read', 'edit', 'verify'}
    assert model.count(req, execution())['request_sha256'] == sha(body)
    response, receipt = client.post_observed('', body, execution())
    (argv, options), = invocations
    assert '--json-schema' not in argv and '--max-turns' not in argv
    assert argv[argv.index('--tools')+1] == '' and '--strict-mcp-config' in argv
    server, = json.loads(argv[argv.index('--mcp-config')+1])['mcpServers'].values()
    assert server['args'][1].endswith('native_cli_mcp.py')
    settings = json.loads(argv[argv.index('--settings')+1])
    command = settings['hooks']['PreToolUse'][0]['hooks'][0]['command']
    assert 'disableAllHooks' not in settings and '\\' not in command and command.endswith('"claude"')
    assert options['env']['ENABLE_TOOL_SEARCH'] == 'false'
    message = response['choices'][0]['message']
    assert [(c['function']['name'], json.loads(c['function']['arguments'])) for c in message['tool_calls']] == [
        ('read', {'path': 'a.py'}), ('verify', {'path': 'tests'})]
    assert receipt['hook_calls'] == 2 and receipt['assistant_tool_names'] == ['read', 'verify']
    assert receipt['task_tools_executed'] == 0 and receipt['inference_invocations_observed'] == 1
    summary = model.wire(replace(req, purpose='summary'))
    assert 'native_tools' not in summary and summary['purpose'] == 'summary'


@pytest.mark.parametrize('damage', [None, 'text_only', 'extra_tool', 'no_server', 'unhooked', 'undenied',
                                    'executed', 'second_inference', 'unknown_tool'])
def test_claude_native_reply_requires_every_call_recorded_and_refused(damage):
    names = ['read', 'edit']
    calls = () if damage == 'text_only' else (('read', {'path': 'a.py'}), ('edit', {'path': 'a.py', 'changes': []}))
    rows, hooks = native_claude_rows(names, calls=calls, text='Reading first.')
    if damage == 'extra_tool': rows[0]['tools'].append('Bash')
    if damage == 'no_server': rows[0]['mcp_servers'] = []
    if damage == 'unhooked': hooks.pop()
    if damage == 'undenied': rows[-1]['permission_denials'].pop()
    if damage == 'executed': rows[-2]['message']['content'][0]['is_error'] = False
    if damage == 'second_inference': rows.insert(-1, {'type': 'assistant', 'message': {
        'id': 'msg-2', 'model': 'fixture', 'content': [{'type': 'text', 'text': 'retrying'}]}})
    if damage == 'unknown_tool':
        rows[3]['message']['content'][0]['name'] = 'mcp__tare__command'
    tools = [{'name': n} for n in names]
    if damage not in (None, 'text_only'):
        with pytest.raises(ModelFailure):
            decode_claude_native(rows, hooks, 'fixture', tools)
        return
    message = decode_claude_native(rows, hooks, 'fixture', tools)['choices'][0]['message']
    assert message['content'] == 'Reading first.'
    assert [c['function']['name'] for c in message['tool_calls']] == [n for n, _ in calls]
    assert [c['id'] for c in message['tool_calls']] == [f'toolu_{i}' for i in range(1, len(calls)+1)]


def test_native_declaration_server_and_hook_never_execute_a_call(tmp_path):
    import subprocess, sys
    from pathlib import Path
    from compute_plane import native_cli
    folder = Path(native_cli.__file__).parent
    (tmp_path/'tools.json').write_text(json.dumps([{'name': 'read', 'description': 'Read.',
        'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}]), encoding='utf-8')
    lines = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}},
             {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
             {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
             {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'read', 'arguments': {}}}]
    served = subprocess.run([sys.executable, '-I', str(folder/'native_cli_mcp.py'), str(tmp_path/'tools.json')],
        input=''.join(json.dumps(l)+'\n' for l in lines).encode(), capture_output=True, timeout=30).stdout
    replies = [json.loads(l) for l in served.splitlines()]
    assert [r['id'] for r in replies] == [1, 2, 3]
    assert replies[1]['result']['tools'][0]['inputSchema']['properties'] == {'path': {'type': 'string'}}
    assert replies[2]['result']['isError'] is True
    event = {'hook_event_name': 'PreToolUse', 'tool_name': 'mcp__tare__read', 'tool_input': {'path': 'é.py'}}
    decided = subprocess.run([sys.executable, '-I', str(folder/'native_cli_hook.py'), str(tmp_path/'b.jsonl'), 'claude'],
        input=json.dumps(event).encode(), capture_output=True, timeout=30).stdout
    decision = json.loads(decided)
    assert decision['continue'] is False and decision['hookSpecificOutput']['permissionDecision'] == 'deny'
    assert json.loads((tmp_path/'b.jsonl').read_text(encoding='utf-8')) == {
        'phase': 'claude', 'tool': 'mcp__tare__read', 'input': {'path': 'é.py'}}


def summary_claude_rows(output):
    return [
        {'type': 'system', 'subtype': 'init', 'tools': [], 'mcp_servers': []},
        {'type': 'assistant', 'message': {'id': 'msg-1', 'model': 'fixture', 'content': [
            {'type': 'thinking', 'thinking': 'private'}, {'type': 'text', 'text': output}]}},
        {'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 1,
         'result': output, 'usage': {'input_tokens': 2}},
    ]


SUMMARY_TEXT = ('Pending: fix {"code": "CLI_TURN_LIMIT_REACHED"} in compute_plane/native_cli.py '
                '[lines 194-259]; checked tests/test_native_cli.py.')


def test_claude_summary_is_plain_text_without_schema_or_turn_limit(tmp_path, monkeypatch):
    # Opus answered a summary in text and --max-turns 1 ended it as error_max_turns.
    from compute_plane import native_cli
    physical = dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False, output_limit=False)
    invocations = []
    def process(argv, **options):
        if '--version' in argv:
            return physical, b'2.1.280 (Claude Code)', b''
        invocations.append(argv)
        return physical, '\n'.join(json.dumps(r) for r in summary_claude_rows(SUMMARY_TEXT)).encode(), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.280')
    model = adapter(client)
    req = request()
    assert 'native_tools' in model.wire(req)  # an ordinary model call uses native tool calls
    body = model.wire(replace(req, purpose='summary'))
    assert 'response_schema' not in body and body['purpose'] == 'summary'
    assert model.count(replace(req, purpose='summary'), execution())['request_sha256'] == sha(body)
    response, receipt = client.post_observed('', body, execution())
    argv, = invocations
    assert '--json-schema' not in argv and '--max-turns' not in argv
    assert argv[argv.index('--tools')+1] == '' and '--strict-mcp-config' in argv
    assert response['choices'][0]['message'] == {'role': 'assistant', 'content': SUMMARY_TEXT, 'tool_calls': []}
    assert receipt['task_tools_executed'] == 0 and receipt['formatting_tools_executed'] == 0


@pytest.mark.parametrize('damage', [None, 'empty', 'blank', 'too_long', 'task_tool', 'structured_tool',
                                    'tool_use', 'turn_limit', 'extra_inference'])
def test_claude_summary_accepts_only_one_toolless_text_reply(damage):
    output = {'empty': '', 'blank': '  \n ', 'too_long': 'x' * 262145}.get(damage, SUMMARY_TEXT)
    rows = summary_claude_rows(output)
    if damage == 'task_tool': rows[0]['tools'] = ['Bash']
    if damage == 'structured_tool': rows[0]['tools'] = ['StructuredOutput']
    if damage == 'tool_use': rows[1]['message']['content'].append({'type': 'tool_use', 'name': 'StructuredOutput'})
    if damage == 'turn_limit': rows[-1].update(subtype='error_max_turns', is_error=True)
    if damage == 'extra_inference': rows.insert(2, {'type': 'assistant', 'message': {
        'id': 'msg-2', 'model': 'fixture', 'content': [{'type': 'text', 'text': 'more'}]}})
    if damage:
        with pytest.raises((ModelFailure, ValueError)):
            decode_claude(rows, 'fixture', summary=True)
        return
    reply = decode_claude(rows, 'fixture', summary=True)
    assert reply['choices'][0]['message']['content'] == SUMMARY_TEXT
    assert reply['choices'][0]['message']['tool_calls'] == [] and reply['usage'] == {'input_tokens': 2}


def test_claude_summary_reaches_the_conversation_as_text_not_a_delivery(tmp_path, monkeypatch):
    # The decoded CLI summary goes through NativeModelAdapter._reply like any chat reply:
    # the whole text is the summary, with no tool calls and no delivery authority.
    from compute_plane import native_cli
    physical = dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False, output_limit=False)
    invocations = []
    def process(argv, **options):
        if '--version' in argv:
            return physical, b'2.1.280 (Claude Code)', b''
        invocations.append(argv)
        return physical, '\n'.join(json.dumps(r) for r in summary_claude_rows(SUMMARY_TEXT)).encode(), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.280')
    model = NativeModelAdapter(ModelProfile('vendor-cli', 'fixture', 32768, 4096, 'fixture', client.endpoint),
        client, system='Perform the task through Tare tools.', tool_specs_json=SPECS,
        counter_protocol='utf8-envelope-estimate-v1', text_completes_turn=True)
    reply = model.generate(replace(request(), call_id='s/summary/1', purpose='summary'), execution())
    argv, = invocations
    assert '--json-schema' not in argv and '--max-turns' not in argv
    assert reply.text == SUMMARY_TEXT and reply.tool_calls == () and reply.complete
    assert not reply.proposes_delivery


def test_a_cli_that_cannot_be_launched_fails_before_dispatch(tmp_path, monkeypatch):
    from compute_plane import native_cli
    def process(argv, **kwargs):
        raise FileNotFoundError(2, 'replaced by an update')  # the executable is momentarily gone
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.280')
    body = adapter(client).wire(request())
    with pytest.raises(ModelFailure) as failure:
        client.post_observed('', body, execution())
    assert failure.value.code == 'CLI_VERSION_UNAVAILABLE' and failure.value.not_started
    assert failure.value.transport_observation['started'] is False


LIMIT_TEXT = "You've hit your limit · resets 5pm (America/Sao_Paulo)"


def claude_limit_rows(event=True):
    rows = [{'type': 'system', 'subtype': 'init', 'tools': ['StructuredOutput'], 'mcp_servers': []},
            {'type': 'assistant', 'message': {'id': 'msg-1', 'model': '<synthetic>',
                                              'content': [{'type': 'text', 'text': LIMIT_TEXT}]}},
            {'type': 'result', 'subtype': 'success', 'is_error': True, 'result': LIMIT_TEXT, 'total_cost_usd': 0}]
    if event:
        rows.insert(1, {'type': 'rate_limit_event', 'rate_limit_info': {
            'status': 'rejected', 'resetsAt': 1790960400, 'rateLimitType': 'five_hour'}})
    return rows


@pytest.mark.parametrize('rows,expected', [
    (claude_limit_rows(), {'resets_at': 1790960400.0, 'kind': 'five_hour'}),
    (claude_limit_rows(event=False), {'resets_at': None, 'kind': None})])
def test_a_spent_claude_subscription_is_a_usage_limit_with_its_reset(rows, expected):
    with pytest.raises(ModelFailure) as failure:
        decode_claude(rows, 'claude-opus-5-5', structured=True)
    assert failure.value.code == 'CLI_USAGE_LIMIT' and failure.value.usage_limit == expected


def test_other_claude_errors_are_not_usage_limits():
    rows = claude_limit_rows()
    rows[1]['rate_limit_info']['status'] = 'allowed'
    rows[-1]['result'] = 'API Error: 500 {"type":"api_error"}'
    with pytest.raises(ModelFailure) as failure:
        decode_claude(rows, 'claude-opus-5-5', structured=True)
    assert failure.value.code == 'CLI_GENERATION_FAILED'


def test_the_transport_receipt_carries_the_usage_limit(tmp_path, monkeypatch):
    from compute_plane import native_cli
    physical = dict(started=True, cleanup_verified=True, exit_code=1, cancelled=False, timeout=False, output_limit=False)
    def process(argv, **options):
        if '--version' in argv:
            return {**physical, 'exit_code': 0}, b'2.1.287 (Claude Code)', b''
        return physical, '\n'.join(json.dumps(r) for r in claude_limit_rows()).encode(), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.287')
    monkeypatch.setattr(client, '_checked_versions', {'2.1.287'})
    with pytest.raises(ModelFailure) as failure:
        client.post_observed('', adapter(client).wire(request()), execution())
    assert failure.value.code == 'CLI_USAGE_LIMIT'
    assert failure.value.transport_observation['usage_limit'] == {'resets_at': 1790960400.0, 'kind': 'five_hour'}


CODEX_LIMIT = 'You’ve hit your usage limit. Try again at 3:45 PM.'


@pytest.mark.parametrize('info,expected', [
    (None, {'resets_at': None, 'kind': None}),
    ({'usage_limit_exceeded': {'limit_id': 'codex', 'resets_at': 1790960400}}, {'resets_at': 1790960400.0, 'kind': 'codex'})])
def test_a_spent_codex_plan_is_a_usage_limit(info, expected):
    from compute_plane.native_cli_codex import decode
    error = {'message': CODEX_LIMIT, **({'codex_error_info': info} if info else {})}
    rows = [{'type': 'thread.started'}, {'type': 'turn.started'}, {'type': 'error', **error},
            {'type': 'turn.failed', 'error': error}]
    with pytest.raises(ModelFailure) as failure:
        decode(rows, 'gpt-6-luna')
    assert failure.value.code == 'CLI_USAGE_LIMIT' and failure.value.usage_limit == expected


def test_a_spent_agy_quota_is_a_usage_limit_before_the_boundary_checks():
    rows = [{'event': 'result', 'result': {'status': 'ERROR', 'error': 'You have exhausted your quota on this model.'}}]
    hooks = [{'phase': 'pre', 'model': 'fixture'}, {'phase': 'stop', 'model': 'fixture', 'termination': 'OTHER'}]
    with pytest.raises(ModelFailure) as failure:
        decode_agy(rows, hooks, 'fixture')
    assert failure.value.code == 'CLI_USAGE_LIMIT'
    # An answer that only talks about quotas is an answer.
    answer = {'event': 'result', 'result': {'status': 'SUCCESS', 'response': json.dumps(
        {'text': 'The server returned quota exceeded (RESOURCE_EXHAUSTED).', 'tool_calls': []})}}
    assert decode_agy([answer], agy_hooks(), 'fixture')['choices'][0]['message']['content'].startswith('The server')


def test_agy_individual_quota_is_a_usage_limit_with_its_relative_reset(monkeypatch):
    # AGY 1.4.12 rows captured 2026-11-02: a planner-sized request was refused (exit 3, hooks pre/stop only),
    # while the same model still answered a tiny request. Before, this surfaced as CLI_BOUNDARY_UNCONFIRMED.
    from compute_plane import native_cli
    monkeypatch.setattr(native_cli.time, 'time', lambda: 1790000000.0)
    rows = [{'event': 'step_update', 'step_update': {'step_index': 1, 'state': 'DONE', 'step_type': 'error_message'}},
            {'event': 'result', 'result': {'status': 'ERROR', 'response': '', 'finished_reason': 'error',
             'error': 'Individual quota reached. Please upgrade your subscription to increase your limits. '
                      'Resets in 53h34m46s.'}}]
    hooks = [{'phase': 'pre', 'model': 'fixture'}, {'phase': 'stop', 'model': 'fixture', 'termination': 'OTHER'}]
    with pytest.raises(ModelFailure) as failure:
        decode_agy(rows, hooks, 'fixture')
    assert failure.value.code == 'CLI_USAGE_LIMIT'
    # Gemini 3.8 Flash answered the same large request meanwhile: the limit belongs to this model only.
    assert failure.value.usage_limit == {'resets_at': 1790000000.0 + 53*3600 + 34*60 + 46, 'kind': 'model'}


def test_remaining_call_budget_is_counted_in_suffix_without_changing_cached_prefix():
    model = adapter(Client())
    req = replace(request(), remaining_model_calls=5)
    first = model.wire(req)
    second = model.wire(replace(req, remaining_model_calls=4))
    before = json.loads(first['prompt'])['conversation_records']
    after = json.loads(second['prompt'])['conversation_records']
    assert before[:-1] == after[:-1]
    assert '5 model calls remain INCLUDING this call' in before[-1]['content']
    assert '4 model calls remain INCLUDING this call' in after[-1]['content']
    count = model.count(req, execution())
    assert count['request_sha256'] == sha(first) != sha(second)
    with pytest.raises(ModelFailure, match='ADMITTED_REQUEST_CHANGED'):
        model.generate(replace(req, remaining_model_calls=4, admitted_request_sha256=count['request_sha256']), execution())


@pytest.mark.parametrize('damage', [None, 'different_id', 'missing_id', 'wrong_model', 'tool'])
def test_claude_thinking_and_text_fragments_require_one_identified_invocation(damage):
    messages = [
        {'id': 'msg-1', 'model': 'fixture', 'content': [{'type': 'thinking', 'thinking': 'private'}]},
        {'id': 'msg-1', 'model': 'fixture', 'content': [{'type': 'text', 'text': answer()}]},
    ]
    if damage == 'different_id': messages[1]['id'] = 'msg-2'
    if damage == 'missing_id': messages[0].pop('id')
    if damage == 'wrong_model': messages[0]['model'] = 'other'
    if damage == 'tool': messages[0]['content'].append({'type': 'tool_use'})
    rows = [{'type': 'system', 'subtype': 'init', 'tools': [], 'mcp_servers': []},
            *({'type': 'assistant', 'message': message} for message in messages),
            {'type': 'result', 'subtype': 'success', 'result': answer(), 'usage': {'input_tokens': 2}}]
    if damage:
        with pytest.raises(ModelFailure, match='CLI_UNEXPECTED_INVOCATION'):
            decode_claude(rows, 'fixture')
    else:
        reply = decode_claude(rows, 'fixture')
        assert reply['usage'] == {'input_tokens': 2}
        assert reply['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'read'


@pytest.mark.parametrize('prefix,suffix,accepted', [
    ('I have read the source. Here is the next proposed operation.\n\n', '', True),
    ('x' * 1025, '', False),
    ('```json\n', '\n```', False),
    ('[', ']', False),
    ('Explanation\n', '\n' + '{"text":"second","tool_calls":[]}', False),
    ('Explanation\n', '\ntrailing prose', False),
])
def test_claude_preamble_preserves_one_strict_data_proposal(prefix, suffix, accepted):
    output = prefix + answer() + suffix
    rows = [{'type': 'system', 'subtype': 'init', 'tools': [], 'mcp_servers': []},
            {'type': 'assistant', 'message': {'model': 'fixture', 'content': [{'type': 'text', 'text': output}]}},
            {'type': 'result', 'subtype': 'success', 'result': output}]
    if accepted:
        reply = decode_claude(rows, 'fixture')['choices'][0]['message']
        assert prefix.strip() in reply['content']
        assert len(reply['tool_calls']) == 1
    else:
        with pytest.raises((ConversationRejected, ValueError)):
            decode_claude(rows, 'fixture')


class Client:
    endpoint = 'cli://agy'
    def __init__(self):
        self.calls = []
        self.names = iter(['read', 'edit', 'verify', 'tare_deliver'])
    def post_observed(self, path, body, control):
        self.calls.append(body)
        return proposal(answer(next(self.names)), 'fixture', {'input_tokens': 10}), {'cleanup_verified': True}


@pytest.mark.parametrize('output,accepted', [
    ('Before editing, I need to confirm the outcome fields.', True),
    ('I ran every test and delivered the change.', True),
    ('', False), ('x' * 16385, False), ('{"text":', False),
    ('```python\nprint(1)\n```', False), ('<tool_call>edit</tool_call>', False),
])
def test_claude_narrative_has_no_executable_or_delivery_authority(output, accepted):
    rows = [{'type': 'system', 'subtype': 'init', 'tools': [], 'mcp_servers': []},
            {'type': 'assistant', 'message': {'id': 'one', 'model': 'fixture',
                'content': [{'type': 'text', 'text': output}]}},
            {'type': 'result', 'subtype': 'success', 'result': output, 'usage': {'input_tokens': 2}}]
    if not accepted:
        with pytest.raises((ConversationRejected, ValueError)):
            decode_claude(rows, 'fixture')
        return
    response = decode_claude(rows, 'fixture')
    assert response['choices'][0]['message']['content'] == output
    assert response['choices'][0]['message']['tool_calls'] == []
    assert response['usage'] == {'input_tokens': 2}


def test_repeated_claude_narrative_stops_at_runtime_stagnation_budget():
    from tare_tools_agent_runtime import NativeBudget
    class NarrativeClient:
        endpoint = 'cli://claude'
        calls = 0
        def post_observed(self, path, body, control):
            self.calls += 1
            rows = [{'type': 'system', 'subtype': 'init', 'tools': [], 'mcp_servers': []},
                    {'type': 'assistant', 'message': {'model': 'fixture', 'content': [{'type': 'text'}]}},
                    {'type': 'result', 'subtype': 'success', 'result': 'I will inspect this next.'}]
            return decode_claude(rows, 'fixture'), {'cleanup_verified': True}
    client = NarrativeClient()
    log = ConversationLog()
    result = NativeSession(session_id='s', instruction='Implement', capabilities=('read', 'edit', 'verify'),
        model=adapter(client), tools=None, journal=log,
        budget=NativeBudget(max_model_calls=4, max_stagnant_turns=2)).run(execution())
    assert result.reason == 'NO_PROGRESS' and client.calls == 2
    assert not any(r.event()['kind'] in {'effect_requested', 'terminal'} for r in log.snapshot().records)


def adapter(client):
    return NativeModelAdapter(ModelProfile('vendor-cli', 'fixture', 32768, 4096, 'fixture', client.endpoint),
        client, system='Perform the task through Tare tools.', tool_specs_json=SPECS,
        counter_protocol='utf8-envelope-estimate-v1')


def test_tare_owns_roundtrip_history_resume_and_count_admission():
    client = Client(); model = adapter(client); effects = []
    class Tools:
        def execute(self, req, control):
            effects.append(req.tool_call.name)
            return EffectResult(req.effect_id, 'succeeded', '{"checked":true}', True)
    session = NativeSession(session_id='s', instruction='Read edit verify', capabilities=('read','edit','verify'),
                            model=model, tools=Tools(), journal=ConversationLog())
    assert session.run(execution()).status == 'proposed'
    assert effects == ['read', 'edit', 'verify'] and len(client.calls) == 4
    messages = json.loads(client.calls[1]['prompt'])['conversation_records']
    assert messages[-2] == {'role': 'tool', 'tool_call_id': 'p1', 'content': '{"checked":true}'}
    assert '15 model calls remain' in messages[-1]['content']
    assert session.run(execution()).status == 'proposed' and len(client.calls) == 4
    req = request(); count = model.count(req, execution())
    assert count['request_sha256'] == sha(model.wire(req)) and count['status'] == 'estimated'
    assert 'hidden vendor context' in count['limitation']
    with pytest.raises(ModelFailure, match='ADMITTED_REQUEST_CHANGED'):
        model.generate(replace(req, instruction='changed', admitted_request_sha256=count['request_sha256']), execution())
    summary = model.wire(replace(req, purpose='summary'))
    assert json.loads(summary['prompt'])['action_catalog'] == []
    assert summary['response_schema']['properties']['actions']['maxItems'] == 0


def agy_hooks():
    return [{'phase': p, 'model': 'fixture', 'invocation': 0,
             'termination': 'TERMINAL_CUSTOM_HOOK' if p == 'stop' else None} for p in ['pre','post','stop']]


@pytest.mark.parametrize('damage', ['missing_hook','extra_inference','tool','truncate','incomplete'])
def test_agy_requires_complete_single_inference_and_no_vendor_tools(damage):
    hooks = agy_hooks(); rows = [{'event': 'result', 'result': {'status': 'SUCCESS', 'response': answer()}}]
    if damage == 'missing_hook': hooks.pop()
    if damage == 'extra_inference': hooks.insert(2, hooks[0])
    if damage == 'tool': rows.insert(0, {'event': 'step_update', 'step_update': {'tool_info': {'name':'run_command'}}})
    if damage == 'truncate': rows[0]['result']['response'] = answer()[:-1]
    if damage == 'incomplete': rows[0]['result']['status'] = 'ERROR'
    with pytest.raises((ModelFailure, ValueError, ConversationRejected)): decode_agy(rows, hooks, 'fixture')


def test_agy_preserves_proposal_and_rejects_duplicate_ids_in_adapter():
    reply = decode_agy([{'event':'result', 'result':{'status':'SUCCESS', 'response':answer()}}], agy_hooks(), 'fixture')
    assert reply['choices'][0]['message']['tool_calls'][0]['id'] == 'p1'
    reply['choices'][0]['message']['tool_calls'] *= 2
    with pytest.raises(ValueError, match='repeated'): adapter(Client())._reply(request(), reply)


@pytest.mark.parametrize('field', ['response', 'structured_output'])
def test_agy_data_actions_use_host_ids_and_preserve_json_arguments(field):
    value = {'text': 'Read source', 'actions': [
        {'name': 'read', 'arguments_json': json.dumps({'path': 'a.py'})},
        {'name': 'read', 'arguments_json': json.dumps({'path': 'b.py'})}]}
    result = {'status': 'SUCCESS', field: json.dumps(value) if field == 'response' else value}
    reply = decode_agy([{'event': 'result', 'result': result}], agy_hooks(), 'fixture', structured=True)
    calls = reply['choices'][0]['message']['tool_calls']
    assert [c['id'] for c in calls] == ['cli-action-1', 'cli-action-2']
    assert json.loads(calls[1]['function']['arguments']) == {'path': 'b.py'}


@pytest.mark.parametrize('damage', ['bad_args', 'extra_field', 'wrong_type', 'extra_inference', 'vendor_tool', 'error'])
def test_agy_structured_output_does_not_relax_boundary_or_schema(damage):
    value = {'text': '', 'actions': [{'name': 'read', 'arguments_json': '{}'}]}
    hooks = agy_hooks()
    if damage == 'bad_args': value['actions'][0]['arguments_json'] = '[]'
    if damage == 'extra_field': value['actions'][0]['command'] = 'execute'
    if damage == 'wrong_type': value['actions'] = {}
    if damage == 'extra_inference': hooks.insert(2, hooks[0])
    if damage == 'vendor_tool': hooks.insert(1, {'phase': 'tool', 'model': 'fixture'})
    rows = [{'event': 'result', 'result': {'status': 'ERROR' if damage == 'error' else 'SUCCESS',
                                         'response': json.dumps(value)}}]
    with pytest.raises((ModelFailure, ValueError, ConversationRejected)):
        decode_agy(rows, hooks, 'fixture', structured=True)


def test_agy_schema_and_action_catalog_are_in_admitted_request():
    model = adapter(Client())
    req = request()
    count = model.count(req, execution())
    wire = model.wire(req)
    assert count['request_sha256'] == sha(wire)
    assert wire['response_schema']['properties']['actions']['items']['properties']['name']['enum']
    assert 'tools' not in json.loads(wire['prompt'])
    assert count['estimated_input_tokens'] >= len(json.dumps(wire['response_schema']).encode())
    # The decision goes through the schema, which AGY may carry through finish; no other function.
    assert 'a finish function, if offered, carries it' in wire['system'] and 'Do not invoke' not in wire['system']


def test_agy_provider_function_error_is_not_a_successful_partial_proposal():
    rows = [{'event': 'result', 'result': {'status': 'ERROR', 'response': answer(),
                                        'error': 'Malformed function call: private vendor diagnostic'}}]
    with pytest.raises(ModelFailure, match='CLI_MALFORMED_FUNCTION_CALL'):
        decode_agy(rows, agy_hooks(), 'fixture')


def test_agy_hook_records_blocked_tool_name_without_arguments(tmp_path, monkeypatch, capsys):
    import io
    from compute_plane import native_cli_hook
    target = tmp_path/'boundary.jsonl'
    monkeypatch.setattr('sys.argv', ['hook', str(target), 'tool'])
    monkeypatch.setattr('sys.stdin', io.TextIOWrapper(io.BytesIO(json.dumps({
        'modelName': 'fixture', 'toolCall': {'name': 'manage_task', 'args': {'Input': 'private content'}}}).encode())))
    native_cli_hook.main()
    result = json.loads(capsys.readouterr().out)
    assert result['decision'] == 'deny'
    raw = target.read_text()
    assert json.loads(raw)['tool'] == 'manage_task' and 'private content' not in raw


def test_agy_finish_hook_captures_only_final_schema_data(tmp_path, monkeypatch, capsys):
    import io
    from compute_plane import native_cli_hook
    target = tmp_path/'boundary.jsonl'
    value = {'text': 'Next action', 'actions': [{'name': 'read', 'arguments_json': '{}'}]}
    monkeypatch.setattr('sys.argv', ['hook', str(target), 'tool'])
    monkeypatch.setattr('sys.stdin', io.TextIOWrapper(io.BytesIO(json.dumps({'modelName': 'fixture', 'toolCall': {
        'name': 'finish', 'args': value | {'toolAction': 'Format response', 'toolSummary': 'Decision'}}}).encode())))
    native_cli_hook.main()
    assert json.loads(capsys.readouterr().out)['decision'] == 'deny'
    row = json.loads(target.read_text())
    assert row['proposal'] == value and row['decision'] == 'deny'
    assert 'toolSummary' not in row['proposal']


def test_agy_finish_hook_keeps_utf8_text_in_an_isolated_interpreter(tmp_path):
    # AGY runs the hook as `python -I`, which ignores PYTHONUTF8; on Windows the locale
    # codec once turned "é" into "Ã©" in every reply delivered through finish.
    import subprocess, sys
    from compute_plane import native_cli_hook
    target = tmp_path/'boundary.jsonl'
    value = {'text': 'Oi! 17 + 25 é 42.', 'actions': []}
    event = json.dumps({'modelName': 'fixture', 'toolCall': {'name': 'finish', 'args': value}}, ensure_ascii=False)
    subprocess.run([sys.executable, '-I', native_cli_hook.__file__, str(target), 'tool'],
                   input=event.encode('utf-8'), capture_output=True, check=True)
    assert json.loads(target.read_text(encoding='utf-8'))['proposal'] == value


@pytest.mark.parametrize('damage', [None, 'task_tool', 'not_denied', 'extra_hook',
    'task_event', 'error', 'invalid_proposal', 'legacy'])
def test_agy_finish_interception_is_one_complete_proposal_not_vendor_execution(damage):
    hooks = agy_hooks()
    hooks.insert(1, {'phase': 'tool', 'model': 'fixture', 'tool': 'finish', 'decision': 'deny',
        'proposal': {'text': 'Read next', 'actions': [{'name': 'read', 'arguments_json': '{"path":"a.py"}'}]}})
    # The hook's typed final arguments are used, not AGY's text rendering of nested JSON.
    rows = [{'event': 'step_update', 'step_update': {'tool_info': {'name': 'finish'}, 'state': 'ERROR'}},
            {'event': 'result', 'result': {'status': 'SUCCESS', 'response': 'rendering is not the proposal'}}]
    if damage == 'task_tool': hooks[1]['tool'] = 'manage_task'
    if damage == 'not_denied': hooks[1]['decision'] = 'allow'
    if damage == 'extra_hook': hooks.insert(2, hooks[1])
    if damage == 'task_event': rows[0]['step_update']['tool_info']['name'] = 'run_command'
    if damage == 'error': rows[-1]['result']['status'] = 'ERROR'
    if damage == 'invalid_proposal': hooks[1]['proposal']['actions'][0]['arguments_json'] = '[]'
    if damage is not None:
        with pytest.raises((ModelFailure, ValueError, ConversationRejected)):
            decode_agy(rows, hooks, 'fixture', structured=damage != 'legacy')
    else:
        reply = decode_agy(rows, hooks, 'fixture', structured=True)
        assert reply['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'read'


@pytest.mark.parametrize('damage', ['tools','mcp','credits','extra_assistant','wrong_model','tool_use'])
def test_claude_rejects_harness_activity_errors_and_model_substitution(damage):
    rows = [{'type':'system','subtype':'init','tools':[],'mcp_servers':[]},
            {'type':'assistant','message':{'model':'fixture','content':[{'type':'text','text':answer()}]}},
            {'type':'result','subtype':'success','is_error':False,'result':answer()}]
    if damage == 'tools': rows[0]['tools'] = ['Bash']
    if damage == 'mcp': rows[0]['mcp_servers'] = ['external']
    if damage == 'credits': rows[-1].update(is_error=True,api_error_code='credits_required')
    if damage == 'extra_assistant': rows.insert(2, rows[1])
    if damage == 'wrong_model': rows[1]['message']['model'] = 'other'
    if damage == 'tool_use': rows[1]['message']['content'] = [{'type':'tool_use'}]
    with pytest.raises(ModelFailure): decode_claude(rows,'fixture')


@pytest.mark.parametrize('damage', ['tool', 'truncated', 'second_turn', 'error'])
def test_codex_rejects_task_tools_and_incomplete_or_repeated_turns(damage):
    from compute_plane.native_cli_codex import decode
    rows = [{'type': 'turn.started'}, {'type':'item.completed','item':{'type':'agent_message','text':answer()}},
            {'type':'turn.completed','usage':{'input_tokens':12}}]
    if damage == 'tool': rows.insert(1, {'type':'item.started','item':{'type':'command_execution'}})
    if damage == 'truncated': rows.pop()
    if damage == 'second_turn': rows.append({'type':'turn.started'})
    if damage == 'error': rows.insert(1, {'type':'error'})
    with pytest.raises(ModelFailure, match='CLI_BOUNDARY_UNCONFIRMED'): decode(rows, 'fixture')


@pytest.mark.parametrize('damage', [None, 'tool', 'second_turn', 'completed'])
def test_codex_provider_refusal_is_a_generation_failure_not_a_boundary_violation(damage):
    # Codex 0.154.0 with gpt-6-luna: the provider answered 400 and the turn failed.
    from compute_plane.native_cli_codex import decode
    rows = [{'type': 'thread.started'}, {'type': 'turn.started'},
            {'type': 'error', 'message': "The 'fixture' model is not supported when using Codex with a ChatGPT account."},
            {'type': 'turn.failed', 'error': {'message': 'unsupported model'}}]
    if damage == 'tool': rows.insert(2, {'type': 'item.started', 'item': {'type': 'command_execution'}})
    if damage == 'second_turn': rows.append({'type': 'turn.started'})
    if damage == 'completed': rows.append({'type': 'turn.completed'})
    code = 'CLI_GENERATION_FAILED' if damage is None else 'CLI_BOUNDARY_UNCONFIRMED'
    with pytest.raises(ModelFailure, match=code): decode(rows, 'fixture')


def test_codex_uses_the_final_message_after_commentary():
    from compute_plane.native_cli_codex import decode
    rows = [{'type': 'turn.started'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Checking the verifier output.'}},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': answer()}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 12}}]
    reply = decode(rows, 'fixture')
    assert reply['choices'][0]['message']['content'] == json.loads(answer())['text']


@pytest.mark.parametrize('text', ['I will now edit app.py.', '```json {"text": "x"} ```', '{"text": "x"}'])
def test_codex_answer_that_is_not_the_json_object_asks_the_model_again(text):
    from compute_plane.native_cli_codex import decode
    from compute_plane.native_models import INVALID_CLI_REPLY, NativeModelAdapter
    from tests.test_native_models import Client, SPECS, execution, profile, request
    rows = [{'type': 'turn.started'}, {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': text}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 12}}]
    reply = decode(rows, 'fixture')
    assert [c['function']['name'] for c in reply['choices'][0]['message']['tool_calls']] == [INVALID_CLI_REPLY]
    reply['model'] = profile('openai-chat').model
    turn = NativeModelAdapter(profile('openai-chat'), Client([reply]), system='s', tool_specs_json=SPECS).generate(
        request(), execution())
    assert not turn.tool_calls and json.loads(turn.metadata_json)['recovery']['code'] == 'INVALID_CLI_REPLY'


def test_codex_catalog_controls_remove_model_selected_tools_and_do_not_mutate_source(tmp_path):
    from compute_plane.native_cli_codex import prepare, decode, REQUIRED_FEATURES
    catalog = {'slug':'fixture','tool_mode':'code_mode_only','experimental_supported_tools':['clock'],
               'apply_patch_tool_type':'freeform','multi_agent_version':'v2'}
    args = prepare(tmp_path, {'model':'fixture','system':'Tare instructions','effort':'high'}, catalog,
                   features=REQUIRED_FEATURES)
    selected = json.loads((tmp_path/'models.json').read_text())['models'][0]
    assert selected['tool_mode'] == 'direct' and selected['experimental_supported_tools'] == []
    assert selected['apply_patch_tool_type'] is None and selected['multi_agent_version'] is None
    assert 'tools.experimental_request_user_input.enabled=false' in args
    assert '--ignore-user-config' in args and '--ephemeral' in args and 'service_tier="default"' in args
    assert catalog['tool_mode'] == 'code_mode_only'
    reply = decode([{'type':'turn.started'}, {'type':'item.completed','item':{'type':'agent_message','text':answer()}},
                    {'type':'turn.completed'}], 'fixture')
    assert reply['_model_identity'] == 'requested_route_only'


@pytest.mark.parametrize('fault', ['cancelled', 'timeout', 'output_limit', 'cleanup'])
def test_cli_never_returns_a_proposal_after_interruption(tmp_path, monkeypatch, fault):
    from compute_plane import native_cli
    from compute_plane.native_models import NativeTransportInterrupted
    physical = dict(started=True,cleanup_verified=True,exit_code=0,cancelled=False,timeout=False,output_limit=False)
    failed = {**physical, ('cleanup_verified' if fault == 'cleanup' else fault): fault != 'cleanup'}
    calls = iter([(physical,b'2.1.278 (Claude Code)',b''),(failed,b'',b'')])
    monkeypatch.setattr(native_cli,'run_pipe_process',lambda *a,**k:next(calls))
    client = native_cli.VendorCliClient('cli://claude',executable=str(tmp_path/'claude.exe'),version='2.1.278')
    with pytest.raises(NativeTransportInterrupted) as err:
        client.post_observed('/inference',{'model':'fixture','effort':'high','system':'test','prompt':'{}'},execution())
    assert err.value.transport_observation['remote_stop'] == 'unknown'
    assert err.value.transport_observation['cleanup_verified'] == (fault != 'cleanup')


@pytest.mark.parametrize('stage', ['version', 'help', 'features'])
def test_preflight_cancellation_is_not_an_unavailable_cli(tmp_path, monkeypatch, stage):
    from compute_plane import native_cli
    from compute_plane.native_cli_codex import discover
    from compute_plane.native_models import NativeTransportInterrupted
    physical = dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False, output_limit=False)
    cancelled = dict(physical, cancelled=True, exit_code=None)
    calls = []
    def process(argv, **options):
        calls.append(argv)
        if stage == 'help' and '--version' in argv:
            return physical, b'2.1.281', b''
        return cancelled, b'', b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    with pytest.raises(NativeTransportInterrupted) as error:
        if stage == 'features':
            discover(str(tmp_path/'codex.exe'), tmp_path, execution(), process,
                dict(started=False, cleanup_verified=True, remote_stop='not_dispatched'))
        else:
            client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.278')
            client.post_observed('/inference', dict(model='fixture', prompt='{}', system='test'), execution())
    receipt = error.value.transport_observation
    assert receipt['cancelled'] and receipt['started'] is False and receipt['cleanup_verified'] is True
    assert receipt['remote_stop'] == 'not_dispatched'
    assert all('--version' in call or '--help' in call or call[1:3] == ['features', 'list'] for call in calls)


@pytest.mark.parametrize('endpoint,effort,accepted', [
    ('cli://claude','xhigh',True), ('cli://claude','max',True),
    ('cli://codex','max',True), ('cli://agy','xhigh',False),
    ('cli://claude','ultracode',False), ('cli://codex','ultra',False)])
def test_reasoning_levels_do_not_enable_vendor_workflows(endpoint, effort, accepted):
    def build():
        return ModelProfile('vendor-cli','fixture',32768,4096,'test',endpoint,
                            parameters_json=json.dumps({'effort':effort}))
    if accepted:
        assert json.loads(build().parameters_json)['effort'] == effort
    else:
        with pytest.raises(ValueError): build()


def test_selected_claude_effort_does_not_inherit_shell_overrides(tmp_path, monkeypatch):
    from compute_plane import native_cli
    physical = dict(started=True,cleanup_verified=True,exit_code=0,cancelled=False,timeout=False,output_limit=False)
    calls = []
    for key in ('CLAUDE_CODE_EFFORT_LEVEL', 'MAX_THINKING_TOKENS', 'CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING'):
        monkeypatch.setenv(key, 'host-override')
    def process(argv, **options):
        calls.append((argv,options))
        if '--version' in argv:
            return physical,b'2.1.278 (Claude Code)',b''
        rows = [{'type':'system','subtype':'init','tools':[],'mcp_servers':[]},
                {'type':'assistant','message':{'model':'fixture','content':[{'type':'text','text':answer('tare_deliver')}]}},
                {'type':'result','subtype':'success','is_error':False,'result':answer('tare_deliver')}]
        return physical,'\n'.join(json.dumps(r) for r in rows).encode(),b''
    monkeypatch.setattr(native_cli,'run_pipe_process',process)
    client = native_cli.VendorCliClient('cli://claude',executable=str(tmp_path/'claude.exe'),version='2.1.278')
    client.post_observed('/inference',{'model':'fixture','effort':'max','system':'test',
        'prompt':json.dumps({'tools':[{'function':{'name':'tare_deliver'}}]})},execution())
    args, options = calls[-1]
    assert args[args.index('--effort')+1] == 'max'
    assert not any(k in options['env'] for k in ('CLAUDE_CODE_EFFORT_LEVEL','MAX_THINKING_TOKENS','CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING'))


@pytest.mark.parametrize('subtype', ['error_max_turns', 'error_max_structured_output_retries',
                                   'error_max_budget_usd', 'PRIVATE-DIAGNOSTIC'])
def test_failed_claude_result_keeps_safe_classification_without_payload(tmp_path, monkeypatch, subtype):
    from compute_plane import native_cli
    physical = dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False, output_limit=False)
    def process(argv, **options):
        if '--version' in argv:
            return physical, b'2.1.280 (Claude Code)', b''
        rows = structured_claude_rows()
        rows[-1].update(subtype=subtype, is_error=True, num_turns=2, errors=['PRIVATE-DIAGNOSTIC'])
        rows[-1].pop('structured_output')
        return physical | {'exit_code': 1}, '\n'.join(json.dumps(r) for r in rows).encode(), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.280')
    code = native_cli.CLAUDE_RESULT_FAILURES.get(subtype, 'CLI_GENERATION_FAILED')
    with pytest.raises(ModelFailure, match=code) as caught:
        client.post_observed('', adapter(client).wire(request()), execution())
    receipt = caught.value.transport_observation
    assert receipt['result_subtype'] == ('unknown' if subtype == 'PRIVATE-DIAGNOSTIC' else subtype)
    assert receipt['result_is_error'] is True and receipt['reported_turns'] == 2
    assert receipt['structured_output_present'] is False
    assert receipt['cleanup_verified'] is True and receipt['task_tools_executed'] is None
    assert 'PRIVATE-DIAGNOSTIC' not in json.dumps(receipt)


@pytest.mark.parametrize('case,observed,code', [
    ('rate_limit', True, 'CLI_GENERATION_FAILED'),
    ('status_mismatch', False, 'CLI_GENERATION_FAILED'),
    ('category_mismatch', False, 'CLI_GENERATION_FAILED'),
    ('conflicting_retry', False, 'CLI_GENERATION_FAILED'),
    ('private_text_only', False, 'CLI_GENERATION_FAILED'),
    ('known_budget', False, 'CLI_USAGE_BUDGET_REACHED'),
    ('credits', False, 'CLI_USAGE_CREDITS_REQUIRED'),
])
def test_claude_rate_limit_observation_does_not_replace_failure(
        tmp_path, monkeypatch, case, observed, code):
    from compute_plane import native_cli

    physical = dict(started=True, cleanup_verified=True, exit_code=1,
                    cancelled=False, timeout=False, output_limit=False)
    rows = structured_claude_rows()
    rows[-1].update(is_error=True, errors=['PRIVATE-DIAGNOSTIC'],
                    result='PRIVATE-DIAGNOSTIC 429')
    rows[-1].pop('structured_output')
    retry = {'type': 'system', 'subtype': 'api_retry', 'error': 'rate_limit',
             'error_status': 429, 'message': 'PRIVATE-DIAGNOSTIC'}
    if case == 'status_mismatch':
        retry['error_status'] = 500
    elif case == 'category_mismatch':
        retry['error'] = 'authentication_failed'
    elif case == 'known_budget':
        rows[-1]['subtype'] = 'error_max_budget_usd'
    elif case == 'credits':
        rows[-1]['api_error_code'] = 'credits_required'
    if case != 'private_text_only':
        rows.insert(1, retry)
    if case == 'conflicting_retry':
        rows.insert(2, {'type': 'system', 'subtype': 'api_retry',
                        'error': 'authentication_failed', 'error_status': 401})

    def process(argv, **options):
        if '--version' in argv:
            return physical | {'exit_code': 0}, b'2.1.280 (Claude Code)', b''
        return physical, '\n'.join(json.dumps(row) for row in rows).encode(), b''

    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://claude',
        executable=str(tmp_path/'claude.exe'), version='2.1.280')
    with pytest.raises(ModelFailure, match=code) as caught:
        client.post_observed('', adapter(client).wire(request()), execution())
    receipt = caught.value.transport_observation
    assert receipt.get('provider_rate_limit_observed') is (True if observed else None)
    assert receipt['cleanup_verified'] is True and receipt['remote_stop'] == 'unknown'
    assert 'PRIVATE-DIAGNOSTIC' not in json.dumps(receipt)


def test_successful_claude_retry_does_not_show_a_failed_rate_limit(tmp_path, monkeypatch):
    from compute_plane import native_cli

    def rows_hooks(tools):
        rows, hooks = native_claude_rows([t['name'] for t in tools])
        rows.insert(1, {'type': 'system', 'subtype': 'api_retry', 'error': 'rate_limit', 'error_status': 429})
        return rows, hooks
    monkeypatch.setattr(native_cli, 'run_pipe_process', native_process(rows_hooks))
    client = native_cli.VendorCliClient('cli://claude',
        executable=str(tmp_path/'claude.exe'), version='2.1.280')
    _, receipt = client.post_observed('', adapter(client).wire(request()), execution())
    assert 'provider_rate_limit_observed' not in receipt


def test_the_formatter_says_the_action_catalog_is_not_callable():
    from compute_plane import native_cli
    tools = [{'type': 'function', 'function': {'name': name, 'description': name, 'parameters': {'type': 'object'}}}
             for name in ('read', 'tare_deliver')]
    body = native_cli.cli_request({'model': 'fixture', 'messages': [], 'tools': tools}, structured=True, formatter=True)
    assert ('The entries of action_catalog (read, tare_deliver) are not functions available to you'
            in body['system'])
    assert body['reminder'].endswith('read, tare_deliver are items of its actions array, not functions to call.')
    assert json.loads(body['prompt'])['action_catalog']  # the records stay plain JSON for counting and checks
    assert 'action_catalog (' not in native_cli.cli_request({'model': 'fixture', 'messages': [], 'tools': []},
                                                            structured=True, formatter=True)['system']


@pytest.mark.parametrize('model,formatter,actions,summary,qualified', [
    ('claude-haiku-5-5', True, True, False, True),
    ('claude-sonnet-5-5', True, True, False, False),
    ('claude-opus-5-5', True, True, False, False),
    ('claude-haiku-5-5', False, True, False, False),
    ('claude-haiku-5-5', True, False, False, False),
    ('claude-haiku-5-5', True, True, True, False),
])
def test_haiku_delivery_guidance_stays_at_its_qualified_transport_boundary(
        model, formatter, actions, summary, qualified):
    from compute_plane import native_cli
    records = [{'role': 'user', 'content': 'Read and report.'}]
    tools = [{'type': 'function', 'function': {'name': 'tare_deliver', 'parameters': {'type': 'object'}}}]
    body = native_cli.cli_request({'model': model, 'messages': records, 'tools': tools if actions else []},
                                 structured=True, formatter=formatter, summary=summary)
    assert (native_cli.HAIKU_FORMAT_GUIDANCE in body['system']) is qualified
    assert (native_cli.HAIKU_FORMAT_GUIDANCE in body.get('reminder', '')) is qualified
    packet = json.loads(body['prompt'])
    assert packet['messages' if summary else 'conversation_records'] == records
    if not summary:
        assert body['response_schema']['properties']['actions']['items']['properties']['arguments_json']['type'] == 'string'


@pytest.mark.parametrize('name,expected', [
    ('mcp__tare__read', 'read'),
    ('read', 'unknown'),
    ('PRIVATE-UNKNOWN-FUNCTION', 'unknown'),
    (None, 'unknown'),
    (['PRIVATE-INVALID-NAME'], 'unknown'),
])
def test_rejected_claude_native_call_reports_cause_without_private_content(tmp_path, monkeypatch, name, expected):
    from compute_plane import native_cli
    def rows_hooks(tools):
        rows, hooks = native_claude_rows([t['name'] for t in tools], calls=(('read', {'path': 'PRIVATE-ARGUMENT'}),))
        next(r for r in rows if r['type'] == 'assistant' and r['message']['content'][0]['type'] == 'tool_use'
             )['message']['content'][0]['name'] = name
        rows[-2]['message']['content'][0].update(content='PRIVATE-ERROR')
        rows[-1].update(subtype='error_max_turns', is_error=True)
        return rows, hooks
    monkeypatch.setattr(native_cli, 'run_pipe_process', native_process(rows_hooks, physical=dict(
        started=True, cleanup_verified=True, exit_code=1, cancelled=False, timeout=False, output_limit=False)))
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.280')
    with pytest.raises(ModelFailure, match='CLI_TURN_LIMIT_REACHED') as caught:
        client.post_observed('', adapter(client).wire(request()), execution())
    receipt = caught.value.transport_observation
    assert receipt['assistant_tool_names'] == [expected] and receipt['tool_result_error'] is True
    assert receipt['task_tools_executed'] is None and receipt['cleanup_verified'] is True
    assert 'PRIVATE-' not in json.dumps(receipt)
