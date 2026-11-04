"""Structured reads keep ordinary file scope, source identity and legacy schemas."""
import hashlib
import json
import time
import pytest

from compute_plane.native_capabilities import NativeCapabilities, TOOL_SPECS
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl, EffectRequest, ToolCall


def test_json_numeric_overflow_and_non_ascii_array_indices_are_rejected():
    from compute_plane.native_json_read import project_json
    for raw in (b'1e400', b'{"hidden": 1e400}'):
        with pytest.raises(ValueError, match='finite'):
            project_json(raw, {'json_pointer': ''})
    for args in ({'json_pointer': '/items/\u0661'},
                 {'json_pointer': '', 'json_fields': '/items/\u0661'}):
        with pytest.raises(ValueError, match='array index'):
            project_json(b'{"items":[1,2]}', args)


def test_structured_read_scope_hash_pages_and_legacy_contract(tmp_path):
    root, scratch = tmp_path/'repo', tmp_path/'scratch'
    root.mkdir(); scratch.mkdir()
    raw = json.dumps({'nodes': [{'id': str(i), 'notes': 'x'*6000} for i in range(200)]}).encode()
    (root/'graph.json').write_bytes(raw)
    binding = BindingRef('work', 'binding', 'owner', 1, 'token')
    options = dict(session_id='s', binding=binding, repositories={'repo': str(root)},
                   writable={}, verifiers={}, scratch_root=scratch, role='explorer', capabilities=['read'])
    legacy, tools = NativeCapabilities(**options), NativeCapabilities(**options, structured_reads=True)
    assert 'json_pointer' not in json.loads(legacy.tool_specs_json)['read']['parameters']['properties']
    assert 'json_pointer' not in TOOL_SPECS['read']['parameters']['properties']
    control = ExecutionControl(binding, CancellationToken(), time.monotonic()+30)
    def call(target, **args):
        return target.execute(EffectRequest('s', 's/effect/1', ToolCall('t', 'read', json.dumps(
            {'repository': 'repo', 'path': 'graph.json', **args}))), control)
    assert call(legacy, json_pointer='/nodes').status != 'succeeded'
    result = call(tools, json_pointer='/nodes', json_fields='/id', limit=100)
    assert result.status == 'succeeded'
    data = json.loads(result.output_json)
    assert data['sha256'] == hashlib.sha256(raw).hexdigest()
    assert data['total'] == 200 and data['next_offset'] == 100
    assert len(json.loads(data['text'])) == 100
    directory=call(tools,json_pointer='',json_fields='/nodes')
    assert directory.status=='succeeded'
    projected=json.loads(directory.output_json)
    assert projected['view']=='object directory' and projected['projection_complete'] is False
    assert projected['sha256']==data['sha256'] and len(projected['text'].encode())<=24000
    assert call(tools, path='../escape.json', json_pointer='').status != 'succeeded'
    assert call(tools, json_pointer='/nodes', start=1).status != 'succeeded'
    assert call(tools, json_pointer='/nodes', start_column=1).status != 'succeeded'
    (root/'graph.json').write_text('{"nodes": []}')
    stale = call(tools, json_pointer='/nodes', sha256=data['sha256'])
    assert stale.status == 'failed' and json.loads(stale.output_json)['status'] == 'STALE'
    assert 'repo:graph.json' not in tools.observations()['reads']
    (root/'notes.md').write_text('first\nsecond\nthird\n')
    lines = call(tools, path='notes.md', limit=2)
    assert lines.status == 'succeeded'
    assert json.loads(lines.output_json)['text'] == '1|first\n2|second'
    assert json.loads(lines.output_json)['next_start'] == 3
    assert call(tools, path='notes.md', limit=2, end=2).status != 'succeeded'
