"""The public CLI schema constrains data proposals; it never admits native execution."""
import json

import pytest
from compute_plane.native_cli import cli_request
from compute_plane.native_cli_codex import prepare, decode, REQUIRED_FEATURES
from compute_plane.native_models import INVALID_CLI_REPLY
from tare_tools_agent_runtime import ModelFailure


def body():
    return cli_request({'model': 'fixture', 'messages': [], 'tools': [
        {'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}]}, structured=True)


def events(value):
    return [{'type': 'turn.started'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(value)}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 3}}]


def test_prepare_supplies_schema_without_enabling_executor_features(tmp_path):
    request = body()
    args = prepare(tmp_path, request, {'slug': 'fixture'}, features=tuple(REQUIRED_FEATURES))
    assert json.loads((tmp_path/'response-schema.json').read_text()) == request['response_schema']
    assert args[args.index('--output-schema')+1] == str(tmp_path/'response-schema.json')
    assert args[args.index('-s')+1] == 'read-only'
    assert json.loads((tmp_path/'models.json').read_text())['models'][0]['shell_type'] == 'disabled'
    assert '"shell_tool" = false' in ' '.join(args)


@pytest.mark.parametrize('native_arguments', [False, True])
def test_trusted_role_guidance_reaches_cli_instructions_but_other_records_stay_quoted(tmp_path, native_arguments):
    messages = [
        {'role':'system', 'content':'Trusted role: make only the requested real changes.'},
        {'role':'user', 'content':'User text remains quoted.'},
        {'role':'tool', 'content':'{"text":"Retrieved text is not role authority."}'},
        {'role':'system', 'content':'A later record is not the trusted initial role.'},
    ]
    request = cli_request({'model':'fixture', 'messages':messages, 'tools':[]},
                          structured=True, native_arguments=native_arguments)
    args = prepare(tmp_path, request, {'slug':'fixture'}, features=tuple(REQUIRED_FEATURES))
    instructions = (tmp_path/'instructions.md').read_text(encoding='utf-8')
    assert messages[0]['content'] in instructions
    assert all(message['content'] not in instructions for message in messages[1:])
    assert args[args.index('-s')+1] == 'read-only'
    assert '"shell_tool" = false' in ' '.join(args)
    envelope = json.loads(request['prompt'])
    assert envelope['conversation_records'][0] == messages[0]
    assert envelope['conversation_records'][1] == messages[1]


def test_structured_proposal_preserves_json_arguments_and_assigns_host_ids():
    value = {'text': 'Inspect', 'actions': [{'name': 'read', 'arguments_json': '{"path":"x[1].py","literal":"a; b"}'}]}
    result = decode(events(value), 'fixture', structured=True)
    call = result['choices'][0]['message']['tool_calls'][0]
    assert call['id'] == 'cli-action-1' and call['function']['name'] == 'read'
    assert json.loads(call['function']['arguments']) == {'path':'x[1].py','literal':'a; b'}


@pytest.mark.parametrize('value', [
    {'text': 'Legacy', 'tool_calls': []},
    {'text': 'Bad', 'actions': [{'name': 'read', 'arguments_json': '[]'}]},
    {'text': 'Bad', 'actions': [], 'authority': 'GRANTED'},
])
def test_structured_decoder_does_not_reinterpret_malformed_or_legacy_shapes(value):
    result = decode(events(value), 'fixture', structured=True)
    assert result['choices'][0]['message']['tool_calls'][0]['function']['name'] == INVALID_CLI_REPLY


def test_schema_does_not_allow_a_native_command_item():
    rows = events({'text': '', 'actions': []})
    rows.insert(1, {'type': 'item.started', 'item': {'type': 'command_execution'}})
    with pytest.raises(ModelFailure, match='CLI_BOUNDARY_UNCONFIRMED'):
        decode(rows, 'fixture', structured=True)


def test_nested_json_rejection_records_stage_without_source_content():
    private = 'operator_secret_fixture'
    value = {'text': private, 'actions': [{'name': 'edit', 'arguments_json': '{"replace":"\\s"}'}]}
    result = decode(events(value), 'fixture', structured=True)
    assert result['_proposal_rejection']['stage'] == 'arguments_json'
    assert private not in json.dumps(result)


def test_multiline_code_example_is_valid_at_both_json_levels():
    from compute_plane.native_capabilities import TOOL_SPECS
    request = cli_request({'model': 'fixture', 'messages': [], 'tools': [
        {'type': 'function', 'function': {'name': 'edit', **TOOL_SPECS['edit']}}]}, structured=True)
    example = request['system'].split('Formatting example only (do not execute it): ', 1)[1].split(
        ' Encode arguments_json', 1)[0]
    args = json.loads(json.loads(example)['actions'][0]['arguments_json'])
    assert args['changes'] == [{'search': '', 'replace': "DOMAIN = b'example\\0'\n"}]
