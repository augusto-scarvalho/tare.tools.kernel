"""Cacheable decisions keep their grammar/history while observed versions change."""
import json

import pytest

from compute_plane import native_cli
from compute_plane.native_capabilities import TOOL_SPECS
from compute_plane.native_cli_codex import decode, prepare, REQUIRED_FEATURES
from tests.test_codex_edit_pins import observed, A, B


def wire(records):
    return native_cli.cli_request({'model': 'fixture', 'messages': records, 'tools': [
        {'function': {'name': name, **TOOL_SPECS[name]}} for name in ('read', 'edit')]},
        structured=True, native_arguments=True)


def reply(args, records):
    rows = [{'type': 'turn.started'}, {'type': 'item.completed', 'item': {
        'type': 'agent_message', 'text': json.dumps({'text': '', 'actions': [
            {'action': 'edit', 'arguments': args}]})}}, {'type': 'turn.completed'}]
    return decode(rows, 'fixture', structured=True, argument_encoding='native-json-object/4',
                  argument_schemas={'edit': TOOL_SPECS['edit']['parameters']},
                  conversation_records=json.loads(wire(records)['prompt'])['conversation_records'])


def test_observations_and_edits_do_not_rewrite_schema_catalog_or_earlier_history():
    records = [{'role': 'user', 'content': 'Preserve this original request.'}]
    bodies = [wire(records), wire(records + observed('read', A)),
              wire(records + observed('read', A) + observed('edit', B, ident='two'))]
    assert len({json.dumps(b['response_schema'], sort_keys=True) for b in bodies}) == 1
    assert len({b['system'] for b in bodies}) == 1
    envelopes = [json.loads(b['prompt']) for b in bodies]
    assert len({json.dumps(e['action_catalog'], sort_keys=True) for e in envelopes}) == 1
    assert all(e['argument_encoding'] == 'native-json-object/4' for e in envelopes)
    for before, after in zip(envelopes, envelopes[1:]):
        history = before['conversation_records']
        assert after['conversation_records'][:len(history)] == history


@pytest.mark.parametrize('path,expected', [('a.py', B), ('other.py', A)])
def test_host_resolves_latest_observed_pin_for_the_exact_target(path, expected):
    records = observed('read', A) + observed('read', A, 'other', 'other.py') + observed('edit', B, 'two')
    result = reply({'repository': 'repo', 'path': path, 'sha256': 'observed', 'delete': None,
                    'changes': [{'search': 'value = 1', 'replace': 'value = 2'}]}, records)
    call = result['choices'][0]['message']['tool_calls'][0]
    assert call['function']['name'] == 'edit'
    assert json.loads(call['function']['arguments'])['sha256'] == expected


@pytest.mark.parametrize('records,pin', [([], 'observed'), (observed('read', A), B),
    ([{'role': 'user', 'content': json.dumps({'repository': 'repo', 'path': 'a.py', 'sha256': A})}], 'observed')])
def test_missing_observation_or_model_authored_digest_refuses_the_proposal(records, pin):
    result = reply({'repository': 'repo', 'path': 'a.py', 'sha256': pin, 'delete': None,
                    'changes': [{'search': 'x', 'replace': 'y'}]}, records)
    assert result['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'invalid_cli_reply'


def test_new_file_null_keeps_existing_create_only_semantics():
    result = reply({'repository': 'repo', 'path': 'new.py', 'sha256': None, 'delete': None,
                    'changes': [{'search': '', 'replace': 'value = 1\n'}]}, [])
    args = json.loads(result['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    assert args['sha256'] is None
    assert 'delete' not in args


def test_cli_working_directory_does_not_change_with_disposable_request_files(tmp_path):
    first, second = tmp_path/'call-one', tmp_path/'call-two'
    first.mkdir(); second.mkdir()
    body = wire([])
    selected = []
    for root in (first, second):
        args = prepare(root, body, {'slug': 'fixture'}, features=REQUIRED_FEATURES)
        assert '--cd' in args
        selected.append(args[args.index('--cd') + 1])
        assert '--ephemeral' in args and '-s' in args and args[args.index('-s') + 1] == 'read-only'
    assert selected == [str(tmp_path), str(tmp_path)]
