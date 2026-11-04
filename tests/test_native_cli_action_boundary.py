"""AGY action transport admission regressions."""
import json

import pytest
from tare_tools_agent_runtime import ModelFailure
from compute_plane import native_cli
from tests.test_native_models import execution


@pytest.fixture
def transport(tmp_path, monkeypatch):
    observed = []
    reply = {'text': 'Decision', 'actions': []}
    physical = dict(started=True, cleanup_verified=True, exit_code=0,
                    cancelled=False, timeout=False, output_limit=False)

    def process(argv, **options):
        if '--version' in argv:
            return physical, b'1.2.7', b''
        observed.append((argv, options))
        hooks = [{'phase': phase, 'model': 'fixture', 'invocation': 0,
                  'termination': 'TERMINAL_CUSTOM_HOOK' if phase == 'stop' else None}
                 for phase in ('pre', 'post', 'stop')]
        (options['cwd']/'boundary.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in hooks), encoding='utf-8')
        row = {'event': 'result', 'result': {'status': 'SUCCESS', 'response': json.dumps(reply)}}
        return physical, json.dumps(row).encode(), b''

    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://agy', executable=str(tmp_path/'agy.exe'), version='1.2.7')

    def invoke(actions, *, with_tools=True, forced=None):
        reply['actions'] = actions
        chat = {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'Recorded task'}],
                'tools': [{'function': {'name': 'read', 'parameters': {'type': 'object'}}}] if with_tools else []}
        if forced is not None:
            chat['tool_choice'] = {'type':'function','function':{'name':forced}}
        body = native_cli.cli_request(chat, structured=True)
        response, receipt = client.post_observed('/inference', body, execution())
        return response, receipt, body, observed[-1]

    return invoke


def test_read_action_transport(transport):
    action = {'name': 'read', 'arguments_json': '{"path":"a.py"}'}
    response, receipt, body, (argv, options) = transport([action], with_tools=True)

    tool_call = response['choices'][0]['message']['tool_calls'][0]
    assert tool_call['id'] == 'cli-action-1'
    assert json.loads(tool_call['function']['arguments']) == {'path': 'a.py'}
    assert receipt['task_tools_executed'] == 0

    assert '--json-schema' in argv
    schema_idx = argv.index('--json-schema')
    assert json.loads(argv[schema_idx + 1]) == body['response_schema']

    stdin_data = json.loads(options['stdin'])
    content = stdin_data['message']['content']
    parsed_content, boundary = json.JSONDecoder().raw_decode(content)
    assert content[:boundary] == body['prompt']
    assert content[boundary:] == body['reminder']
    assert native_cli.AGY_ACTION_GUIDANCE in body['reminder']
    assert 'conversation_records' in parsed_content
    assert 'action_catalog' in parsed_content


def test_forced_choice_reminder_follows_the_unchanged_agy_input_prefix(transport):
    _,_,body,(_,options)=transport([{'name':'read','arguments_json':'{"path":"a.py"}'}],forced='read')
    content=json.loads(options['stdin'])['message']['content']
    assert content==body['prompt']+body['reminder']
    assert 'not quoted history' in body['reminder'] and '"read"' in body['reminder']


def test_final_choice_preserves_cacheable_system_catalog_and_response_grammar():
    chat={'model':'fixture','messages':[{'role':'system','content':'Stable role'},
        {'role':'user','content':'Observed context'}], 'tools':[{'function':{'name':name,'parameters':{'type':'object'}}}
                                                               for name in ('read','tare_deliver')]}
    normal=native_cli.cli_request(chat,structured=True)
    final=native_cli.cli_request({**chat,'tool_choice':{'type':'function','function':{'name':'tare_deliver'}}},structured=True)
    assert normal['system']==final['system'] and normal['response_schema']==final['response_schema']
    assert json.loads(normal['prompt'])['action_catalog']==json.loads(final['prompt'])['action_catalog']
    assert final['prompt'].startswith(normal['prompt'][:-1])
    assert 'honest outcome and remaining gaps' in final['reminder']


def test_unexpected_tool_proposal(transport):
    action = {'name': 'run_command', 'arguments_json': '{}'}
    with pytest.raises(ModelFailure, match=r'CLI_UNEXPECTED_TOOL_PROPOSAL'):
        transport([action], with_tools=True)


@pytest.mark.parametrize('names', [[], ['other'], ['read','read'], ['read']])
def test_host_enforces_a_forced_choice_before_any_effect_with_a_static_catalog(transport, names):
    actions=[{'name':name,'arguments_json':'{"path":"a.py"}'} for name in names]
    response, receipt, body, _ = transport(actions, forced='read')
    calls=response['choices'][0]['message']['tool_calls']
    assert [c['function']['name'] for c in calls] == (['read'] if names==['read'] else ['invalid_cli_reply'])
    assert receipt['inference_invocations_observed']==1 and receipt['task_tools_executed']==0
    assert json.loads(body['prompt'])['turn_constraints']=={'required_action':'read'}
    if names!=['read']:
        assert receipt['proposal_rejection']=={'stage':'tool_choice'}


def test_with_tools_false(transport):
    response, _, body, _ = transport([], with_tools=False)
    assert response['choices'][0]['message']['tool_calls'] == []

    assert body['response_schema']['properties']['actions']['maxItems'] == 0

    action = {'name': 'read', 'arguments_json': '{"path":"a.py"}'}
    with pytest.raises(ModelFailure, match=r'CLI_UNEXPECTED_TOOL_PROPOSAL'):
        transport([action], with_tools=False)
