"""Batched oversized reads return bounded pages and still leave a call for delivery.

Deterministic fixture model; no live provider, vendor CLI or network.
"""
import json
import time

from compute_plane.native_capabilities import NativeCapabilities
from tare_tools_agent_runtime import (BindingRef, CancellationToken, ConversationLog, ExecutionControl,
    ModelReply, NativeBudget, NativeSession, ToolCall)


class FixtureModel:
    def __init__(self):
        self.remaining = []

    def generate(self, request, control):
        self.remaining.append(request.remaining_model_calls)
        if len(self.remaining) == 1:
            return ModelReply(text='Reading both sources.', tool_calls=(
                ToolCall('read-a', 'read', json.dumps({'repository': 'repo', 'path': 'a.py', 'start': 1, 'end': 450})),
                ToolCall('read-b', 'read', json.dumps({'repository': 'repo', 'path': 'b.py', 'start': 100, 'end': 500})),
            ))
        return ModelReply(text='Both bounded pages observed; proposing delivery.', proposes_delivery=True)


def test_two_batched_oversized_reads_then_delivery_within_two_calls(tmp_path):
    root, scratch = tmp_path/'repo', tmp_path/'custody'
    root.mkdir(); scratch.mkdir()
    for name in ('a.py', 'b.py'):
        (root/name).write_text(''.join(f'value = {i}\n' for i in range(1, 501)))
    binding = BindingRef('work', 'binding', 'owner', 1, 'token')
    control = ExecutionControl(binding, CancellationToken(), time.monotonic()+20)
    tools = NativeCapabilities(session_id='s', binding=binding, repositories={'repo': str(root)},
        writable={}, verifiers={}, scratch_root=scratch, specgraph_edit_context=False)
    model, log = FixtureModel(), ConversationLog()
    outcome = NativeSession(session_id='s', instruction='Read both sources and deliver', capabilities=('read',),
        model=model, tools=tools, journal=log, budget=NativeBudget(max_model_calls=2)).run(control)

    assert (outcome.status, outcome.reason, outcome.model_calls) == ('proposed', 'DELIVERY_PROPOSED', 2)
    assert model.remaining == [2, 1]
    events = [r.event() for r in log.snapshot().records]
    requested = [e['effect_id'] for e in events if e['kind'] == 'effect_requested']
    results = [e['result'] for e in events if e['kind'] == 'effect_returned']
    assert requested == ['s/effect/1/1', 's/effect/1/2']
    assert [r['status'] for r in results] == ['succeeded', 'succeeded']
    pages = [json.loads(r['output_json']) for r in results]
    for page, (start, end) in zip(pages, [(1, 450), (100, 500)]):
        lines = page['text'].splitlines()
        assert len(lines) == 200 and len(page['text'].encode()) <= 24000
        assert (page['start'], page['end'], page['next_start']) == (start, start+199, start+200)
        assert page['requested_end'] == end and page['truncated'] is True
        assert page['lines'] == 500
        assert lines[0] == f'{start}|value = {start}'
    assert events[-1]['kind'] == 'terminal' and events[-1]['status'] == 'proposed'
