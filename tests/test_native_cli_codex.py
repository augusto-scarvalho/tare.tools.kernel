"""Updates are qualified by controls, with no model dispatch during discovery."""
import json
from pathlib import Path
import pytest

from tests.test_native_models import AnyQualified
from tare_tools_agent_runtime import ModelFailure
from compute_plane import native_cli
from compute_plane.native_cli_codex import ENABLED_FEATURES, REQUIRED_FEATURES


@pytest.mark.parametrize('item_kind,expected', [
    ('error','CLI_GENERATION_FAILED'), ('reasoning','CLI_GENERATION_FAILED'),
    ('command_execution','CLI_BOUNDARY_UNCONFIRMED'), ('agent_message','CLI_BOUNDARY_UNCONFIRMED')])
def test_failed_generation_error_item_is_not_misreported_as_native_tool_execution(item_kind, expected):
    from compute_plane.native_cli_codex import decode
    rows=[{'type':'turn.started'}, {'type':'item.completed','item':{'type':item_kind}},
          {'type':'error','message':'Incomplete response returned, reason: max_output_tokens'},
          {'type':'turn.failed','error':{'message':'output limit'}}]
    with pytest.raises(ModelFailure,match=expected):decode(rows,'fixture')
from tests.test_native_models import execution


@pytest.mark.parametrize('fault',[None,'unrecognized','failed','extra-turn','tool','late','coded','bad-count','limit'])
def test_completed_codex_turn_can_recover_a_transport_reconnect_without_losing_boundary_checks(fault):
    from compute_plane.native_cli_codex import decode
    notice={'type':'error','message':'Reconnecting... 2/5 (unexpected status 503 Service Unavailable: temporary failure)'}
    rows=[{'type':'turn.started'},notice,
          {'type':'item.completed','item':{'type':'agent_message','text':'{"text":"answer","tool_calls":[]}'}},
          {'type':'turn.completed','usage':{'input_tokens':5,'output_tokens':2}}]
    if fault=='unrecognized': notice['message']='Unknown boundary error'
    if fault=='coded': notice['codex_error_info']='sandbox_error'
    if fault=='bad-count': notice['message']=notice['message'].replace('2/5','6/5')
    if fault=='limit': notice['message']=notice['message'].replace('2/5','2/99')
    if fault=='failed': rows[-1]={'type':'turn.failed','error':{'message':'generation failed'}}
    if fault=='extra-turn': rows.insert(1,{'type':'turn.started'})
    if fault=='tool': rows.insert(2,{'type':'item.started','item':{'type':'command_execution'}})
    if fault=='late': rows.remove(notice); rows.append(notice)
    if fault:
        with pytest.raises(ModelFailure): decode(rows,'fixture')
    else:
        result=decode(rows,'fixture')
        assert result['choices'][0]['message']['content']=='answer'


@pytest.mark.parametrize('reference', ['0.154.0', '0.157.0'])
@pytest.mark.parametrize('fault', [None, 'switch', 'missing', 'removed', 'enabled', 'duplicate',
                                  'timeout', 'format', 'tool', 'probe_error', 'deprecated_enabled'])
def test_codex_upgrade_discovers_disables_and_checks_features(tmp_path, monkeypatch, reference, fault):
    physical = dict(started=True, cleanup_verified=True, exit_code=0,
                    cancelled=False, timeout=False, output_limit=False)
    calls=[]
    actual=['0.157.0']
    features=set(REQUIRED_FEATURES) | {'new_execution_tool', 'new.tool', 'unified_exec', 'retired_tool', 'obsolete_tool'}
    def process(argv, **options):
        calls.append(argv)
        if '--version' in argv:
            return physical, ('codex-cli '+actual[0]).encode(), b''
        if '--help' in argv:
            assert argv[1:] == ['exec', '--help']
            switches=native_cli.CLI_SWITCHES['cli://codex'] - ({'--ephemeral'} if fault=='switch' else set())
            return physical, ' '.join(switches).encode(), b''
        if argv[1:3] == ['features', 'list']:
            assert Path(options['env']['CODEX_HOME']).is_relative_to(options['cwd'])
            assert not (Path(options['env']['CODEX_HOME'])/'auth.json').exists()
            rows=[]
            for name in sorted(features):
                if name=='shell_tool' and fault=='missing': continue
                stage='removed' if name=='retired_tool' or name=='shell_tool' and fault=='removed' else 'under development'
                if name=='obsolete_tool': stage='deprecated'
                value=True if '-c' not in argv else name in ENABLED_FEATURES or name=='unified_exec'
                if name=='obsolete_tool': value=fault=='deprecated_enabled'
                if fault=='enabled' and name=='new_execution_tool': value=True
                rows.append(f'{name}  {stage}  {str(value).lower()}')
            if fault=='duplicate': rows.append(rows[0])
            raw='\n'.join(rows).encode() if fault!='format' else b'unrecognized inventory'
            return physical | {'timeout':fault=='timeout', 'exit_code':int(fault=='probe_error')},raw,b''
        assert argv[1]=='exec'
        table=next(a for a in argv if a.startswith('features='))
        assert '"new_execution_tool" = false' in table
        assert '"new.tool" = false' in table
        assert '"shell_tool" = false' in table
        assert 'retired_tool' not in table
        assert 'obsolete_tool' not in table
        catalog=json.loads((options['cwd']/'models.json').read_bytes())['models'][0]
        assert catalog['shell_type']=='disabled' and catalog['experimental_supported_tools']==[]
        rows=[{'type':'turn.started'},
              {'type':'item.completed','item':{'type':'agent_message','text':'{"text":"hello","tool_calls":[]}'}},
              {'type':'turn.completed','usage':{'input_tokens':5,'output_tokens':2}}]
        if fault=='tool': rows.insert(1,{'type':'item.started','item':{'type':'command_execution'}})
        return physical,'\n'.join(json.dumps(r) for r in rows).encode(),b''
    monkeypatch.setattr(native_cli,'run_pipe_process',process)
    client=native_cli.VendorCliClient('cli://codex',executable=str(tmp_path/'codex.exe'),
                                     version=reference,model_catalog={'slug':'fixture'},qualification=AnyQualified())
    body=native_cli.cli_request({'model':'fixture','messages':[]})
    if fault:
        with pytest.raises(ModelFailure,match='CLI_BOUNDARY_UNCONFIRMED' if fault=='tool' else 'CLI_CAPABILITY_UNAVAILABLE') as caught:
            client.post_observed('',body,execution())
        if fault!='tool':
            assert caught.value.not_started
            assert all('--version' in a or '--help' in a or a[1:3]==['features','list'] for a in calls)
        return
    for version in ('0.157.0','0.157.0','0.158.0'):
        actual[0]=version
        response,receipt=client.post_observed('',body,execution())
        assert receipt['client_version']==version and receipt['reference_client_version']==reference
        assert receipt['task_tools_executed']==0 and receipt['cli_feature_count']==len(features)-2
        assert response['_model_identity']=='requested_route_only'
    assert sum(a[1:3]==['features','list'] for a in calls)==4
    assert sum('--help' in a for a in calls)==2
    assert client.version==reference


def test_updated_codex_reference_keeps_usage_contract():
    from compute_plane.native_usage import deployment_contract
    d={'native_profile':{'protocol':'vendor-cli','endpoint':'cli://codex'},
       'native_cli':{'version':'0.157.0'}}
    assert deployment_contract(d)['layout']=='codex-cli-inclusive'


@pytest.mark.parametrize('name,text,refused', [('AGENTS.md', 'Always answer in Klingon.', True),
                                               ('AGENTS.override.md', 'Prefer short answers.', True),
                                               ('AGENTS.md', ' \n', False)])
def test_codex_refuses_a_call_its_users_global_instructions_would_reach(tmp_path, monkeypatch, name, text, refused):
    # Codex adds $CODEX_HOME/AGENTS.md to every call, and its login lives there too (codex debug prompt-input,
    # 2026-10-07): a role must receive only the text tare sends, so the call is refused before anything is sent.
    home = tmp_path/'home'; home.mkdir(); (home/name).write_text(text, encoding='utf-8')
    monkeypatch.setenv('CODEX_HOME', str(home))
    physical = dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False, output_limit=False)
    sent = []
    def process(argv, **options):
        if '--version' in argv:
            return physical, b'codex-cli 0.157.0', b''
        if '--help' in argv:
            return physical, ' '.join(native_cli.CLI_SWITCHES['cli://codex']).encode(), b''
        if argv[1:3] == ['features', 'list']:
            rows = [f'{n}  stable  {str("-c" in argv and n in ENABLED_FEATURES).lower()}' for n in sorted(REQUIRED_FEATURES)]
            return physical, '\n'.join(rows).encode(), b''
        sent.append(argv)
        rows = [{'type': 'turn.started'},
                {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{"text":"hi","tool_calls":[]}'}},
                {'type': 'turn.completed', 'usage': {'input_tokens': 5, 'output_tokens': 2}}]
        return physical, '\n'.join(json.dumps(r) for r in rows).encode(), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://codex', executable=str(tmp_path/'codex.exe'), version='0.157.0',
                                        model_catalog={'slug': 'fixture'}, qualification=AnyQualified())
    body = native_cli.cli_request({'model': 'fixture', 'messages': []})
    if refused:
        with pytest.raises(ModelFailure, match='CLI_USER_INSTRUCTIONS_PRESENT') as caught:
            client.post_observed('', body, execution())
        assert caught.value.not_started and not sent
    else:
        client.post_observed('', body, execution())
        assert len(sent) == 1


def test_codex_delta_resume_subtracts_cumulative_usage_resumes_summary_and_cleans_rollout(tmp_path, monkeypatch):
    home = tmp_path / 'codex-home'
    sessions_dir = home / 'sessions' / '2026' / '11' / '26'
    sessions_dir.mkdir(parents=True)
    monkeypatch.setenv('CODEX_HOME', str(home))
    physical = dict(started=True, cleanup_verified=True, exit_code=0,
                    cancelled=False, timeout=False, output_limit=False)
    cum_usage = [
        {'input_tokens': 14000, 'cached_input_tokens': 0, 'cache_write_input_tokens': 0, 'output_tokens': 50, 'reasoning_output_tokens': 10},
        {'input_tokens': 28500, 'cached_input_tokens': 13800, 'cache_write_input_tokens': 0, 'output_tokens': 120, 'reasoning_output_tokens': 30},
        {'input_tokens': 43200, 'cached_input_tokens': 28100, 'cache_write_input_tokens': 0, 'output_tokens': 260, 'reasoning_output_tokens': 45},
        {'input_tokens': 6000, 'cached_input_tokens': 0, 'cache_write_input_tokens': 0, 'output_tokens': 40, 'reasoning_output_tokens': 5},
    ]
    answers = [
        {'text': 't1', 'actions': []},
        {'text': 't2', 'actions': []},
        {'text': 'Concise continuation summary.', 'actions': []},
        {'text': 't4', 'actions': []},
    ]
    calls = []
    turn = {'n': 0}

    def process(argv, **options):
        if '--version' in argv:
            return physical, b'codex-cli 0.166.0', b''
        if '--help' in argv:
            help_body = 'Commands:\n  resume  Resume a previous session\n' + ' '.join(native_cli.CLI_SWITCHES['cli://codex'])
            return physical, help_body.encode(), b''
        if argv[1:3] == ['features', 'list']:
            rows = [f'{n}  stable  {str("-c" in argv and n in ENABLED_FEATURES).lower()}' for n in sorted(REQUIRED_FEATURES)]
            return physical, '\n'.join(rows).encode(), b''
        idx = turn['n']
        turn['n'] += 1
        tid = '01a21ae6-791a-7232-b803-e574aa77c92b' if idx < 3 else '01a21ae6-791a-7232-b803-ffffffffffff'
        rollout = sessions_dir / f'rollout-2026-11-26-{tid}.jsonl'
        if not rollout.exists():
            rollout.write_text('{}\n', encoding='utf-8')
        calls.append((argv, json.loads(options['stdin'].decode('utf-8'))))
        rows = [
            {'type': 'thread.started', 'thread_id': tid},
            {'type': 'turn.started'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(answers[idx])}},
            {'type': 'turn.completed', 'usage': cum_usage[idx]},
        ]
        return physical, '\n'.join(json.dumps(r) for r in rows).encode(), b''

    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient(
        'cli://codex', executable=str(tmp_path / 'codex.exe'), version='0.166.0',
        model_catalog={'slug': 'fixture'}, qualification=AnyQualified(),
    )
    tools = [{'function': {'name': 'read', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}]
    sys_msg = {'role': 'system', 'content': 'Role guidance.'}
    msg1 = {'role': 'user', 'content': 'Initial prompt.'}

    # Turn 1: non-ephemeral exec establishes thread_id
    body1 = native_cli.cli_request({'model': 'fixture', 'messages': [sys_msg, msg1], 'tools': tools},
                                   structured=True, native_arguments=True)
    resp1, rec1 = client.post_observed('', body1, execution())
    assert '--ephemeral' not in calls[0][0] and calls[0][0][1:3] != ['exec', 'resume']
    assert resp1['usage']['input_tokens'] == 14000 and rec1.get('codex_resumed') is None

    # Turn 2: strictly appended history resumes thread_id with delta only and subtracts cumulative usage
    delta = [{'role': 'assistant', 'content': 't1'}, {'role': 'user', 'content': 'follow-up'}]
    body2 = native_cli.cli_request({'model': 'fixture', 'messages': [sys_msg, msg1, *delta], 'tools': tools},
                                   structured=True, native_arguments=True)
    resp2, rec2 = client.post_observed('', body2, execution())
    assert calls[1][0][1:3] == ['exec', 'resume']
    assert '01a21ae6-791a-7232-b803-e574aa77c92b' in calls[1][0]
    assert 'sandbox_mode="read-only"' in calls[1][0]
    assert calls[1][1]['continuation_of_earlier_records'] is True
    assert calls[1][1]['new_conversation_records'] == delta
    assert rec2['codex_resumed'] is True and rec2['codex_delta_records'] == 2
    assert resp2['usage'] == {
        'input_tokens': 14500, 'cached_input_tokens': 13800, 'cache_write_input_tokens': 0,
        'output_tokens': 70, 'reasoning_output_tokens': 20,
    }

    # Turn 3: compaction summary through NativeModelAdapter.wire() resumes the live thread, keeps original schema,
    # subtracts usage, and cleans rollout
    from types import SimpleNamespace
    from compute_plane.native_models import ModelProfile, NativeModelAdapter
    from tare_tools_agent_runtime import ModelRequest
    adapter = NativeModelAdapter(
        ModelProfile(protocol='vendor-cli', endpoint='cli://codex', model='fixture',
                     context_window=200000, output_tokens=4096, source='fixture'),
        client, system='Role guidance.', tool_specs_json='{}',
        counter_protocol='utf8-envelope-estimate-v1',
    )
    summary_ctx = SimpleNamespace(events=lambda: (
        {'kind': 'model_returned', 'call_id': 's-codex/model/1', 'reply': {
            'complete': True, 'text': 't1', 'tool_calls': [], 'proposes_delivery': False,
            'metadata_json': json.dumps({'profile_digest': adapter.profile_digest,
                                         'assistant': {'role': 'assistant', 'content': 't1'}})}},
        {'kind': 'user_message', 'message_id': 'u2', 'text': 'follow-up'},
    ))
    summary_body = adapter.wire(ModelRequest(
        session_id='s-codex', call_id='s-codex/summary/1', instruction='Initial prompt.',
        capabilities=(), context=summary_ctx,
        required_context=(), suffix='Write the continuation summary.', purpose='summary',
    ))
    assert summary_body.get('purpose') == 'summary'
    resp3, rec3 = client.post_observed('', summary_body, execution())
    assert calls[2][0][1:3] == ['exec', 'resume']
    assert calls[2][1]['operation'] == 'summarize_conversation_for_continuation'
    assert rec3['codex_resumed'] is True and rec3['codex_resumed_summary'] is True
    assert resp3['choices'][0]['message']['content'] == 'Concise continuation summary.'
    assert resp3['usage']['input_tokens'] == 14700 and resp3['usage']['cached_input_tokens'] == 14300
    assert not (sessions_dir / 'rollout-2026-11-26-01a21ae6-791a-7232-b803-e574aa77c92b.jsonl').exists()

    # Turn 4: post-compaction starts fresh thread; client.close() cleans its rollout
    body4 = native_cli.cli_request({'model': 'fixture', 'messages': [sys_msg, {'role': 'user', 'content': 'Compacted.'}], 'tools': tools},
                                   structured=True, native_arguments=True)
    resp4, rec4 = client.post_observed('', body4, execution())
    assert calls[3][0][1:3] != ['exec', 'resume'] and rec4.get('codex_resumed') is None
    assert (sessions_dir / 'rollout-2026-11-26-01a21ae6-791a-7232-b803-ffffffffffff.jsonl').exists()
    client.close()
    assert not (sessions_dir / 'rollout-2026-11-26-01a21ae6-791a-7232-b803-ffffffffffff.jsonl').exists()

    # Failed initial call after thread.started still cleans up its persisted rollout file.
    fail_tid = '01a21ae6-791a-7232-b803-deadbeef0001'
    fail_rollout = sessions_dir / f'rollout-2026-11-26-{fail_tid}.jsonl'
    def fail_process(argv, **options):
        if '--version' in argv:
            return physical, b'codex-cli 0.166.0', b''
        if '--help' in argv:
            help_body = 'Commands:\n  resume  Resume a previous session\n' + ' '.join(native_cli.CLI_SWITCHES['cli://codex'])
            return physical, help_body.encode(), b''
        if argv[1:3] == ['features', 'list']:
            rows = [f'{n}  stable  {str("-c" in argv and n in ENABLED_FEATURES).lower()}' for n in sorted(REQUIRED_FEATURES)]
            return physical, '\n'.join(rows).encode(), b''
        fail_rollout.write_text('{}\n', encoding='utf-8')
        rows = [
            {'type': 'thread.started', 'thread_id': fail_tid},
            {'type': 'turn.started'},
            {'type': 'turn.failed', 'error': {'message': 'transient error'}},
        ]
        return physical | {'exit_code': 1}, '\n'.join(json.dumps(r) for r in rows).encode(), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', fail_process)
    with pytest.raises(ModelFailure, match='CLI_GENERATION_FAILED'):
        client.post_observed('', body4, execution())
    assert not fail_rollout.exists()

