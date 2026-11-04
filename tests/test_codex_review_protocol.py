"""Native approval must match the exact action item, never prose or exit status."""
from copy import deepcopy
import pytest
from compute_plane.action_control import ActionRefused
from compute_plane.codex_review_protocol import NativeReviewBinding


def rows(status='approved'):
    item = {'method': 'item/started', 'params': {'threadId': 'thread', 'turnId': 'turn',
        'item': {'type': 'mcpToolCall', 'id': 'item', 'server': 'tare_action',
                 'tool': 'execute_exact_command', 'arguments': {'argv': ['python', '-V'], 'cwd': '/fixture'}}}}
    params = {'threadId': 'thread', 'turnId': 'turn', 'targetItemId': 'item', 'reviewId': 'review',
              'startedAtMs': 1, 'action': {'type': 'mcpToolCall', 'server': 'tare_action',
                                        'toolName': 'execute_exact_command'}, 'review': {'status': 'inProgress'}}
    started = {'method': 'item/autoApprovalReview/started', 'params': deepcopy(params)}
    params.update(completedAtMs=2, decisionSource='agent', review={'status': status})
    return item, started, {'method': 'item/autoApprovalReview/completed', 'params': params}


def binding():
    return NativeReviewBinding({'argv': ['python', '-V'], 'cwd': '/fixture'}, 'thread', 'turn')


@pytest.mark.parametrize('status', ['approved', 'denied', 'timedOut', 'aborted'])
def test_correlated_native_terminal_is_distinct(status):
    port = binding()
    for row in rows(status): port.observe(row)
    assert port.approved is (status == 'approved')
    assert port.terminal['status'] == status


@pytest.mark.parametrize('key,value', [('targetItemId', 'other'), ('threadId', 'other'),
    ('turnId', 'other'), ('reviewId', 'other'), ('startedAtMs', 0), ('completedAtMs', 0),
    ('decisionSource', 'user'), ('action', {'type': 'command', 'command': 'echo safe'})])
def test_changed_native_completion_never_authorizes(key, value):
    port = binding(); item, start, end = rows()
    port.observe(item); port.observe(start); end['params'][key] = value
    with pytest.raises(ActionRefused): port.observe(end)
    assert not port.approved


def test_no_event_completion_before_start_duplicate_and_other_tool():
    port = binding(); item, start, end = rows()
    port.observe({'method': 'turn/completed', 'params': {}})
    assert not port.approved and port.terminal is None
    with pytest.raises(ActionRefused): port.observe(end)
    port.observe(item); port.observe(start); port.observe(end)
    with pytest.raises(ActionRefused): port.observe(end)
    other = deepcopy(item); other['params']['item']['type'] = 'commandExecution'
    with pytest.raises(ActionRefused): binding().observe(other)


def test_arguments_are_copied_and_changed_command_is_refused():
    args = {'argv': ['python', '-V'], 'cwd': '/fixture'}
    port = NativeReviewBinding(args, 'thread', 'turn'); args['argv'][1] = '-c'
    item, _, _ = rows(); port.observe(item)
    item['params']['item']['arguments']['argv'][1] = '-c'
    with pytest.raises(ActionRefused): binding().observe(item)


def test_malformed_event_keeps_only_structural_failure_metadata():
    port=binding();item,start,end=rows()
    del start['params']['targetItemId']
    port.observe(item)
    with pytest.raises(ActionRefused,match='malformed_native_review'):port.observe(start)
    assert port.failure=={'method':'item/autoApprovalReview/started','error_type':'KeyError',
                          'missing_field':'targetItemId'}
    assert not port.approved


# Offline subprocess coverage: the bridge publishes a request, not an effect.
import json
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

from compute_plane.action_control import ActionAuthorization, ActionSnapshot, ExactActionGuard


def bridge_arguments(root):
    marker = root / 'executed.txt'
    return {'argv': [sys.executable, '-c',
                    f'from pathlib import Path; Path({str(marker)!r}).write_text("executed")'],
            'cwd': str(root)}


def bridge_process(root, arguments, calls, deadline=None, response=None):
    (root / 'packet.json').write_text(json.dumps({
        'arguments': arguments,
        'deadline': time.monotonic() + 10 if deadline is None else deadline,
    }), encoding='utf-8')
    if response is not None:
        (root / 'response.json').write_text(json.dumps(response), encoding='utf-8')
    bridge = Path(__file__).resolve().parents[1] / 'compute_plane' / 'codex_review_bridge.py'
    completed = subprocess.run(
        [sys.executable, '-B', str(bridge), str(root)],
        input=''.join(json.dumps(row) + '\n' for row in calls),
        capture_output=True, text=True, encoding='utf-8', timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ''
    return [json.loads(line) for line in completed.stdout.splitlines()]


def bridge_call(arguments, ident=1, name='execute_exact_command'):
    return {'jsonrpc': '2.0', 'id': ident, 'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments}}


def test_bridge_process_dispatches_exact_arguments_without_executing(tmp_path):
    arguments = bridge_arguments(tmp_path)
    response = {'content': [{'type': 'text', 'text': 'offline host response'}]}
    replies = bridge_process(tmp_path, arguments, [
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
         'params': {'protocolVersion': '2024-11-05'}},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'ping'},
        bridge_call({}, 4), bridge_call({}, 5),
    ], response=response)
    assert [row['id'] for row in replies] == [1, 2, 3, 4, 5]
    assert replies[0]['result']['protocolVersion'] == '2024-11-05'
    tool, = replies[1]['result']['tools']
    assert tool['name'] == 'execute_exact_command'
    assert tool['inputSchema']['properties'] == {}
    assert tool['inputSchema']['required'] == []
    assert json.dumps(arguments,sort_keys=True,allow_nan=False) in tool['description']
    assert tool['inputSchema']['additionalProperties'] is False
    assert replies[2]['result'] == {}
    assert replies[3]['result'] == response
    assert replies[4]['error']['code'] == -32000
    expected = json.dumps(arguments, sort_keys=True, allow_nan=False)
    assert (tmp_path / 'dispatch.json').read_text(encoding='utf-8') == expected
    assert (tmp_path / 'dispatch.tmp').read_text(encoding='utf-8') == expected
    assert not (tmp_path / 'executed.txt').exists()
    # The exact supplied implementation retains its hard-link staging file.
    assert {path.name for path in tmp_path.iterdir()} == {
        'packet.json', 'response.json', 'dispatch.tmp', 'dispatch.json'}


@pytest.mark.parametrize('changed', ['argv', 'cwd', 'action_sha256'])
def test_bridge_process_advertises_typed_schema_but_rejects_changed_arguments(tmp_path, changed):
    arguments = {'action_sha256': 'a' * 64, **bridge_arguments(tmp_path)}
    supplied = deepcopy(arguments)
    if changed == 'argv':
        supplied['argv'].append('--changed')
    elif changed == 'cwd':
        supplied['cwd'] += '-changed'
    else:
        supplied['action_sha256'] = 'b' * 64
    replies = bridge_process(tmp_path, arguments, [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'},
        bridge_call(supplied, 2),
    ])
    assert [row['id'] for row in replies] == [1, 2]
    tool, = replies[0]['result']['tools']
    assert tool['name'] == 'execute_exact_command'
    schema = tool['inputSchema']
    assert schema['type'] == 'object'
    assert schema['properties'] == {}
    assert schema['required'] == []
    assert json.dumps(arguments,sort_keys=True,allow_nan=False) in tool['description']
    assert schema['additionalProperties'] is False
    assert replies[1] == {'jsonrpc': '2.0', 'id': 2,
                          'error': {'code': -32000, 'message': 'Host command refused'}}
    assert not (tmp_path / 'dispatch.json').exists()
    assert not (tmp_path / 'dispatch.tmp').exists()
    assert not (tmp_path / 'executed.txt').exists()
    assert {path.name for path in tmp_path.iterdir()} == {'packet.json'}


@pytest.mark.parametrize('mismatch', ['argv', 'cwd', 'name', 'expired'])
def test_bridge_process_refuses_without_any_dispatch(tmp_path, mismatch):
    arguments = bridge_arguments(tmp_path)
    supplied = {}
    name = 'execute_exact_command'
    deadline = None
    if mismatch == 'argv':
        supplied['argv'] = arguments['argv']+['--changed']
    elif mismatch == 'cwd':
        supplied['cwd'] = arguments['cwd']+'-changed'
    elif mismatch == 'name':
        name = 'other_tool'
    else:
        deadline = time.monotonic() - 1
    replies = bridge_process(tmp_path, arguments,
                             [bridge_call(supplied, name=name)], deadline=deadline)
    assert replies == [{'jsonrpc': '2.0', 'id': 1,
                        'error': {'code': -32000, 'message': 'Host command refused'}}]
    assert {path.name for path in tmp_path.iterdir()} == {'packet.json'}


def revalidation_fixture():
    action = ActionSnapshot(
        work_id='work', session_id='session', effect_id='effect', binding_id='binding',
        fence_seq=1, operation='command', repository='fixture', cwd='/fixture',
        environment_sha256='1' * 64, context_sha256='2' * 64, source_sha256='3' * 64,
        policy_sha256='4' * 64, grant_sha256='5' * 64, execution_profile='offline',
        deadline=20.0, argv=('python', '-V'),
    )
    authorization = ActionAuthorization(
        issuer='host', nonce='nonce', action_sha256=action.action_sha256,
        session_id=action.session_id, effect_id=action.effect_id,
        binding_id=action.binding_id, fence_seq=action.fence_seq, expires_at=15.0,
    )
    state = {'now': 1.0, 'stopped': False}
    control = SimpleNamespace(
        binding=SimpleNamespace(work_id='work', binding_id='binding', fence=1),
        should_stop=lambda: state['stopped'],
    )
    return action, authorization, control, state


def test_revalidation_accepts_unchanged_snapshot_without_claiming():
    action, authorization, control, state = revalidation_fixture()
    verified, refreshed, claims = [], [], []
    guard = ExactActionGuard(
        lambda a, auth: verified.append((a, auth)) or True,
        lambda *args: claims.append(args) or True, clock=lambda: state['now'],
    )
    assert guard.revalidate(action, authorization, control,
                            lambda: refreshed.append(True) or replace(action)) is None
    assert verified == [(action, authorization)]
    assert refreshed == [True]
    assert claims == []


def test_revalidation_rejects_changed_source_without_claiming():
    action, authorization, control, state = revalidation_fixture()
    claims = []
    guard = ExactActionGuard(lambda *args: True,
                             lambda *args: claims.append(args) or True,
                             clock=lambda: state['now'])
    with pytest.raises(ActionRefused, match='snapshot changed while awaiting native review'):
        guard.revalidate(action, authorization, control,
                         lambda: replace(action, source_sha256='6' * 64))
    assert claims == []


@pytest.mark.parametrize('phase', ['verify', 'refresh'])
@pytest.mark.parametrize('change', ['stopped', 'binding', 'fence',
                                    'action_expiry', 'authorization_expiry'])
def test_revalidation_rejects_live_control_and_expiry_changes_without_claiming(phase, change):
    action, authorization, control, state = revalidation_fixture()
    if change == 'action_expiry':
        action = replace(action, deadline=10.0)
        authorization = replace(authorization, action_sha256=action.action_sha256)
    claims, refreshed = [], []

    def mutate():
        if change == 'stopped':
            state['stopped'] = True
        elif change == 'binding':
            control.binding.binding_id = 'changed'
        elif change == 'fence':
            control.binding.fence = 2
        elif change == 'action_expiry':
            state['now'] = action.deadline
        else:
            state['now'] = authorization.expires_at

    def verify(a, auth):
        if phase == 'verify':
            mutate()
        return True

    def refresh():
        refreshed.append(True)
        if phase == 'refresh':
            mutate()
        return action

    guard = ExactActionGuard(verify, lambda *args: claims.append(args) or True,
                             clock=lambda: state['now'])
    with pytest.raises(ActionRefused):
        guard.revalidate(action, authorization, control, refresh)
    assert refreshed == ([] if phase == 'verify' else [True])
    assert claims == []


@pytest.mark.parametrize('as_bytes', [False, True], ids=['str','bytes'])
def test_bridge_decoder_default_stays_64_kib_with_bounded_rpc_opt_in(as_bytes):
    from compute_plane.codex_review_bridge import loads
    document = json.dumps({'padding':'x' * 87000})
    data = document.encode('utf-8') if as_bytes else document
    with pytest.raises(ValueError, match='bounded_json_required'):
        loads(data)
    assert loads(data, max_bytes=131072) == {'padding':'x' * 87000}


@pytest.mark.parametrize('as_bytes', [False, True], ids=['str','bytes'])
def test_bridge_decoder_counts_utf8_bytes(as_bytes):
    from compute_plane.codex_review_bridge import loads
    document = '"' + '\u00e9' * 32767 + '"'
    assert len(document.encode('utf-8')) == 65536
    data = document.encode('utf-8') if as_bytes else document
    assert loads(data) == '\u00e9' * 32767
    document = '"' + '\u00e9' * 32768 + '"'
    data = document.encode('utf-8') if as_bytes else document
    with pytest.raises(ValueError, match='bounded_json_required'):
        loads(data)
    assert loads(data, max_bytes=131072) == '\u00e9' * 32768


@pytest.mark.parametrize('bound', [0, -1, True, False, 65536.0, '131072', None, 131073])
def test_bridge_decoder_rejects_invalid_requested_bound(bound):
    from compute_plane.codex_review_bridge import loads
    with pytest.raises(ValueError, match='bounded_json_required'):
        loads('{}', max_bytes=bound)


@pytest.mark.parametrize('bound', [65536, 131072])
@pytest.mark.parametrize('document', ['{"key":1,"key":2}', 'NaN', 'Infinity', '-Infinity'],
                         ids=['duplicate-key','nan','infinity','negative-infinity'])
def test_bridge_decoder_preserves_strict_json_validation(bound,document):
    from compute_plane.codex_review_bridge import loads
    with pytest.raises(ValueError):
        loads(document, max_bytes=bound)
