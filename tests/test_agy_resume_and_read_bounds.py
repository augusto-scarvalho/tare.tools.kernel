import json
import subprocess
import sys
from pathlib import Path
import pytest

from compute_plane import native_cli, native_cli_hook
from compute_plane.native_capabilities import NativeCapabilities
from compute_plane.native_models import _reads_covered_later, _project_duplicate_read, encode
from tare_tools_agent_runtime import EffectRequest, ModelFailure, ToolCall
from tests.test_native_models import AnyQualified, execution


def test_hook_allows_only_valid_finish_when_opted_in_and_denies_all_other_tools(tmp_path):
    target = tmp_path / 'boundary.jsonl'
    valid_finish = {'modelName': 'fixture', 'invocationNum': 0,
                    'toolCall': {'name': 'finish', 'args': {'text': 'ok', 'actions': []}}}
    # Default 2-arg call keeps deny.
    p_default = subprocess.run(
        [sys.executable, '-I', native_cli_hook.__file__, str(target), 'tool'],
        input=json.dumps(valid_finish).encode(), capture_output=True, check=True)
    assert json.loads(p_default.stdout)['decision'] == 'deny'

    # Opt-in --allow-finish allows ONLY valid finish proposals.
    p_allow = subprocess.run(
        [sys.executable, '-I', native_cli_hook.__file__, str(target), 'tool', '--allow-finish'],
        input=json.dumps(valid_finish).encode(), capture_output=True, check=True)
    assert json.loads(p_allow.stdout)['decision'] == 'allow'

    # Any non-finish tool is STILL denied even with --allow-finish.
    other_tool = {'modelName': 'fixture', 'invocationNum': 0,
                  'toolCall': {'name': 'read', 'args': {'text': 'ok', 'actions': []}}}
    p_other = subprocess.run(
        [sys.executable, '-I', native_cli_hook.__file__, str(target), 'tool', '--allow-finish'],
        input=json.dumps(other_tool).encode(), capture_output=True, check=True)
    assert json.loads(p_other.stdout)['decision'] == 'deny'

    # Malformed finish args are STILL denied even with --allow-finish.
    bad_finish = {'modelName': 'fixture', 'invocationNum': 0,
                  'toolCall': {'name': 'finish', 'args': {'text': 'ok', 'actions': [], 'extra': 1}}}
    p_bad = subprocess.run(
        [sys.executable, '-I', native_cli_hook.__file__, str(target), 'tool', '--allow-finish'],
        input=json.dumps(bad_finish).encode(), capture_output=True, check=True)
    assert json.loads(p_bad.stdout)['decision'] == 'deny'


def test_agy_delta_resume_subtracts_cumulative_usage_bypasses_full_160k_cap_and_falls_back_on_divergence(
        tmp_path, monkeypatch):
    calls = []
    physical = dict(started=True, cleanup_verified=True, exit_code=0,
                    cancelled=False, timeout=False, output_limit=False)
    cum_usage_by_turn = [
        {'input_tokens': 8000, 'cache_read_tokens': 0, 'output_tokens': 60, 'thinking_tokens': 40, 'total_tokens': 8060},
        {'input_tokens': 10500, 'cache_read_tokens': 6000, 'output_tokens': 150, 'thinking_tokens': 90, 'total_tokens': 10650},
        {'input_tokens': 4000, 'cache_read_tokens': 0, 'output_tokens': 50, 'thinking_tokens': 20, 'total_tokens': 4050},
    ]
    turn = {'n': 0}

    def process(argv, **options):
        if argv[-1] == '--version':
            return physical, b'1.5.17', b''
        if argv[-1] == '--help':
            switches = ' '.join(sorted(native_cli.CLI_SWITCHES['cli://agy'] | {'--resume'}))
            return physical, switches.encode(), b''
        idx = turn['n']
        turn['n'] += 1
        calls.append((argv, json.loads(options['stdin'].decode('utf-8'))))
        proposal = {'text': f'reply-{idx}', 'actions': []}
        hooks = [
            {'phase': 'pre', 'model': 'fixture'},
            {'phase': 'tool', 'model': 'fixture', 'tool': 'finish', 'decision': 'allow', 'proposal': proposal},
            {'phase': 'post', 'model': 'fixture'},
            {'phase': 'stop', 'model': 'fixture', 'termination': 'TERMINAL_CUSTOM_HOOK'},
        ]
        (options['cwd'] / 'boundary.jsonl').write_text('\n'.join(json.dumps(h) for h in hooks), encoding='utf-8')
        conv_id = 'conv-1' if idx < 2 else 'conv-2'
        row = {'event': 'result', 'conversation_id': conv_id,
               'result': {'status': 'SUCCESS', 'response': json.dumps(proposal), 'usage': cum_usage_by_turn[idx]}}
        return physical, json.dumps(row).encode(), b''

    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient(
        'cli://agy', executable=str(tmp_path / 'agy.exe'), version='1.5.17',
        qualification=AnyQualified(), request_byte_limit=160 * 1024,
    )
    tools = [{'function': {'name': 'read', 'parameters': {'type': 'object'}}}]
    big_msg = {'role': 'user', 'content': 'A' * (110 * 1024)}

    # Turn 1: stateless (establishes conv-1)
    body1 = native_cli.cli_request({'model': 'fixture', 'messages': [big_msg], 'tools': tools}, structured=True)
    resp1, rec1 = client.post_observed('/inference', body1, execution())
    assert '--resume' not in calls[0][0]
    assert resp1['usage']['input_tokens'] == 8000 and resp1['usage']['cache_read_tokens'] == 0
    assert rec1.get('agy_resumed') is None and rec1['blocked_tool_names'] == []

    # Turn 2: strictly appended history whose FULL body exceeds 160 KiB, while the delta is small.
    # Delta resume succeeds, sends only new records, and reports exact single-turn subtracted usage.
    delta_msgs = [{'role': 'assistant', 'content': 'B' * (70 * 1024)}, {'role': 'user', 'content': 'follow-up'}]
    body2 = native_cli.cli_request(
        {'model': 'fixture', 'messages': [big_msg, *delta_msgs], 'tools': tools}, structured=True)
    assert len(encode(body2).encode('utf-8')) > 160 * 1024
    resp2, rec2 = client.post_observed('/inference', body2, execution())
    assert '--resume' in calls[1][0] and calls[1][0][calls[1][0].index('--resume') + 1] == 'conv-1'
    sent2 = json.loads(calls[1][1]['message']['content'].split('\n\n', 1)[0])
    assert sent2['continuation_of_earlier_records'] is True
    assert sent2['new_conversation_records'] == delta_msgs
    assert rec2['agy_resumed'] is True and rec2['agy_delta_records'] == 2
    assert resp2['usage'] == {
        'input_tokens': 2500, 'cache_read_tokens': 6000, 'output_tokens': 90,
        'thinking_tokens': 50, 'total_tokens': 2590,
    }

    # Turn 3: prefix changed (e.g., after compaction) -> falls back cleanly to stateless without --resume.
    body3 = native_cli.cli_request(
        {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'compacted summary'}], 'tools': tools},
        structured=True)
    resp3, rec3 = client.post_observed('/inference', body3, execution())
    assert '--resume' not in calls[2][0]
    assert rec3.get('agy_resumed') is None
    assert resp3['usage']['input_tokens'] == 4000


def test_mechanical_read_bounds_clamp_pages_and_update_schema(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'sample.py').write_text('\n'.join(f'line_{i} = {"x" * 90}' for i in range(1, 201)) + '\n', encoding='utf-8')
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    ctrl = execution()
    caps = NativeCapabilities(
        session_id='s1', binding=ctrl.binding, repositories={'repo': str(repo)},
        writable={}, verifiers={}, scratch_root=str(scratch), role='planner',
        read_max_lines=40, read_max_bytes=2048,
    )
    specs = json.loads(caps.tool_specs_json)
    assert 'at most 40 lines and 2048 bytes' in specs['read']['description']
    assert 'up to 40 lines from start' in specs['read']['description']

    req = EffectRequest('s1', 's1/effect/1', ToolCall('c1', 'read', json.dumps({
        'repository': 'repo', 'path': 'sample.py', 'start': 1, 'end': 200})))
    res = caps.execute(req, ctrl)
    assert res.status == 'succeeded'
    out = json.loads(res.output_json)
    assert out['start'] == 1 and out['end'] <= 40
    assert len(out['text'].encode('utf-8')) <= 2048
    assert out['truncated'] is True and out['next_start'] == out['end'] + 1


def test_reads_covered_later_elides_only_redundant_unchanged_spans():
    sha1, sha2 = 'a' * 64, 'b' * 64
    events = [
        # Effect 1: read a.py lines 10..30 (sha1) -> later covered by Effect 3 (1..50, sha1) -> ELIDED
        {'kind': 'model_returned', 'call_id': 's/model/1', 'reply': {
            'tool_calls': [{'call_id': 'c1', 'name': 'read', 'arguments_json': json.dumps({'repository': 'r', 'path': 'a.py', 'start': 10, 'end': 30})}]}},
        {'kind': 'effect_returned', 'result': {'effect_id': 's/effect/1', 'status': 'succeeded',
            'output_json': json.dumps({'start': 10, 'end': 30, 'lines': 100, 'sha256': sha1, 'text': '10|hello'})}},
        # Effect 2: read a.py lines 60..80 (sha1) -> disjoint from 1..50 -> KEPT
        {'kind': 'model_returned', 'call_id': 's/model/2', 'reply': {
            'tool_calls': [{'call_id': 'c2', 'name': 'read', 'arguments_json': json.dumps({'repository': 'r', 'path': 'a.py', 'start': 60, 'end': 80})}]}},
        {'kind': 'effect_returned', 'result': {'effect_id': 's/effect/2', 'status': 'succeeded',
            'output_json': json.dumps({'start': 60, 'end': 80, 'lines': 100, 'sha256': sha1, 'text': '60|world'})}},
        # Effect 3: read a.py lines 1..50 (sha1) -> covers Effect 1 -> KEPT
        {'kind': 'model_returned', 'call_id': 's/model/3', 'reply': {
            'tool_calls': [{'call_id': 'c3', 'name': 'read', 'arguments_json': json.dumps({'repository': 'r', 'path': 'a.py', 'start': 1, 'end': 50})}]}},
        {'kind': 'effect_returned', 'result': {'effect_id': 's/effect/3', 'status': 'succeeded',
            'output_json': json.dumps({'start': 1, 'end': 50, 'lines': 100, 'sha256': sha1, 'text': '1|top'})}},
        # Effect 4: edit a.py -> resets coverage so Effect 5 (sha2) does NOT retroactively elide Effect 3 (sha1)
        {'kind': 'model_returned', 'call_id': 's/model/4', 'reply': {
            'tool_calls': [{'call_id': 'c4', 'name': 'edit', 'arguments_json': json.dumps({'repository': 'r', 'path': 'a.py'})}]}},
        {'kind': 'effect_returned', 'result': {'effect_id': 's/effect/4', 'status': 'succeeded',
            'output_json': json.dumps({'sha256': sha2})}},
        {'kind': 'model_returned', 'call_id': 's/model/5', 'reply': {
            'tool_calls': [{'call_id': 'c5', 'name': 'read', 'arguments_json': json.dumps({'repository': 'r', 'path': 'a.py', 'start': 1, 'end': 50})}]}},
        {'kind': 'effect_returned', 'result': {'effect_id': 's/effect/5', 'status': 'succeeded',
            'output_json': json.dumps({'start': 1, 'end': 50, 'lines': 100, 'sha256': sha2, 'text': '1|new'})}},
    ]
    dups = _reads_covered_later(events)
    assert dups == {'s/effect/1'}
    projected = json.loads(_project_duplicate_read(events[1]['result']['output_json']))
    assert 'text' not in projected and projected['sha256'] == sha1 and 'Covered verbatim' in projected['text_omitted']
