"""Compaction tool policy and repeated scoped guidance on the provider wire."""
from dataclasses import asdict, replace
import json

import pytest

from tare_tools_agent_runtime import ConversationLog, ProjectionSpec
from compute_plane.native_models import NativeModelAdapter, _project_scoped_instructions, sha
from .test_native_models import Client, SPECS, execution, profile, request, response


def guidance(text='scoped-rule-unique-marker', path='nested/AGENTS.md'):
    return {'repository': 'repo', 'path': path, 'text': text, 'sha256': text}


def test_openai_summary_omits_tool_schemas_on_the_counted_and_generated_request():
    raw = {'model': 'fixture-model', 'choices': [{'finish_reason': 'stop',
        'message': {'role': 'assistant', 'content': 'Saved investigation; report remains to be written.'}}]}
    counter = Client([{'prompt': 'summary request'}, {'tokens': [1, 2, 3]}])
    client = Client([raw])
    adapter = NativeModelAdapter(profile('openai-chat'), client, system='Use tools for the task.',
                                 tool_specs_json=SPECS, local_counter=counter)
    req = replace(request(), purpose='summary', suffix='Return a continuation checkpoint only.')
    count = adapter.count(req, execution())
    reply = adapter.generate(replace(req, admitted_request_sha256=count['request_sha256']), execution())
    assert reply.text and not reply.tool_calls and not reply.proposes_delivery
    counted_wire = counter.calls[0][1]
    assert client.calls[0][1] == counted_wire
    assert 'tools' not in counted_wire and 'tool_choice' not in counted_wire
    assert counted_wire['messages'][0]['content'].startswith('Current operation: summarize')
    normal = adapter.wire(request())
    assert normal['tool_choice'] == 'auto'
    assert normal['tools']
    assert count['request_sha256'] == sha(counted_wire)


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse'])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_wire_deduplicates_guidance_only_within_current_projection(protocol, checkpoint):
    adapter = NativeModelAdapter(profile(protocol), Client([response(protocol)] * 2),
                                 system='Stable', tool_specs_json=SPECS)
    log = ConversationLog()
    row = guidance()
    if checkpoint:
        log.append({'kind': 'context_activated', 'profile_digest': adapter.profile_digest,
                    'source_context_sha256': 'a' * 64, 'summary': 'Prior facts',
                    'observed_state': json.dumps({'scoped_project_instructions': [row]})})
    sequences = []
    for index in (1, 2):
        req = replace(request(), call_id=f's/model/{index}')
        reply = adapter.generate(req, execution())
        a = log.append({'kind': 'model_returned', 'call_id': req.call_id, 'reply': asdict(reply)})
        b = log.append({'kind': 'effect_returned', 'result': {'effect_id': f's/effect/{index}',
            'status': 'succeeded', 'output_json': json.dumps({'text': f'file-content-{index}',
            'scoped_project_instructions': [row]}), 'progressed': True}})
        sequences.append((a.sequence, b.sequence))
    before = log.snapshot()
    req = replace(request(), context=log.project(ProjectionSpec(1, 'fixture')))
    wire = adapter.wire(req)
    # Count text itself, not its matching fixture hash.
    raw = json.dumps(wire)
    assert raw.count('scoped-rule-unique-marker') == 2
    assert 'file-content-1' in raw and 'file-content-2' in raw
    assert log.snapshot() == before
    assert adapter.wire(req) == wire
    # No adapter-global "already sent" cache: the rule returns when the
    # earlier observation/checkpoint is absent from a later projection.
    later = replace(req, context=log.project(ProjectionSpec(1, 'tail-only', sequences[-1])))
    assert json.dumps(adapter.wire(later)).count('scoped-rule-unique-marker') == 2


def test_projection_preserves_guidance_changes_deletions_and_scopes():
    seen = {}
    original = guidance()
    changed = guidance('changed-rule')
    missing = guidance(None)
    sibling = guidance(path='sibling/AGENTS.md')
    for row, expected in [(original, [original]), (original, []), (changed, [changed]),
                          (original, [original]), (missing, [missing]),
                          (original, [original]), (sibling, [sibling])]:
        raw = json.dumps({'scoped_project_instructions': [row], 'text': 'unchanged tool content'})
        projected = json.loads(_project_scoped_instructions(raw, seen))
        assert projected['scoped_project_instructions'] == expected
        assert projected['text'] == 'unchanged tool content'
    for raw in ['plain checkpoint text', '[]', '{"observations": []}']:
        assert _project_scoped_instructions(raw, seen) == raw


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse'])
@pytest.mark.parametrize('enabled', [False, True])
def test_reads_of_files_edited_later_are_sent_as_markers_only_when_enabled(protocol, enabled):
    steps = [('read', {'repository': 'repo', 'path': 'app.py'}, {'text': '1: old app body', 'start': 1, 'end': 1,
                                                                  'lines': 1, 'sha256': 'old'}),
             ('edit', {'repository': 'repo', 'path': 'app.py'}, {'sha256': 'new'}),
             ('read', {'repository': 'repo', 'path': 'other.py'}, {'text': '1: other body', 'start': 1, 'end': 1,
                                                                    'lines': 1, 'sha256': 'o'})]
    adapter = NativeModelAdapter(profile(protocol), Client([response(protocol, n, a) for n, a, _ in steps]),
                                 system='Stable', tool_specs_json=SPECS, elide_edited_reads=enabled)
    log = ConversationLog()
    for index, (name, args, output) in enumerate(steps, 1):
        req = replace(request(), call_id=f's/model/{index}')
        reply = adapter.generate(req, execution())
        log.append({'kind': 'model_returned', 'call_id': req.call_id, 'reply': asdict(reply)})
        log.append({'kind': 'effect_returned', 'result': {'effect_id': f's/effect/{index}', 'status': 'succeeded',
                    'output_json': json.dumps(output), 'progressed': True}})
    before = log.snapshot()
    raw = json.dumps(adapter.wire(replace(request(), context=log.project(ProjectionSpec(1, 'fixture')))))
    assert ('old app body' in raw) is not enabled
    assert 'other body' in raw  # a file that was not edited keeps its text
    assert ('text_omitted' in raw) is enabled
    assert log.snapshot() == before


def test_failed_or_unrelated_edits_do_not_hide_reads():
    from compute_plane.native_models import _reads_edited_later
    def reply(i, name, path):
        return {'kind': 'model_returned', 'call_id': f's/model/{i}', 'reply': {'tool_calls': [
            {'call_id': 'c', 'name': name, 'arguments_json': json.dumps({'repository': 'r', 'path': path})}]}}
    def result(i, status='succeeded'):
        return {'kind': 'effect_returned', 'result': {'effect_id': f's/effect/{i}', 'status': status}}
    events = [reply(1, 'read', 'a.py'), result(1), reply(2, 'edit', 'a.py'), result(2, 'failed'),
              reply(3, 'edit', 'b.py'), result(3), reply(4, 'read', 'a.py'), result(4),
              reply(5, 'rename', 'c.py'), result(5)]
    assert _reads_edited_later(events) == set()
    events += [reply(6, 'edit', 'a.py'), result(6)]
    assert _reads_edited_later(events) == {'s/effect/1', 's/effect/4'}


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse'])
@pytest.mark.parametrize('keep', [None, 2])
def test_reads_and_searches_older_than_the_last_replies_are_sent_as_ranges_and_locations(protocol, keep):
    steps = [('read', {'repository': 'repo', 'path': 'a.py'}, {'text': '1: first body', 'start': 1, 'end': 1,
                                                              'lines': 1, 'sha256': 'a'}),
             ('search', {'repository': 'repo', 'query': 'value'}, {'query': 'value', 'matches': [
                 {'path': 'b.py', 'line': 7, 'text': 'value = second match text'}]}),
             ('read', {'repository': 'repo', 'path': 'c.py'}, {'text': '1: third body', 'start': 1, 'end': 1,
                                                              'lines': 1, 'sha256': 'c'}),
             ('read', {'repository': 'repo', 'path': 'd.py'}, {'text': '1: fourth body', 'start': 1, 'end': 1,
                                                              'lines': 1, 'sha256': 'd'})]
    adapter = NativeModelAdapter(profile(protocol), Client([response(protocol, n, a) for n, a, _ in steps]),
                                 system='Stable', tool_specs_json=SPECS, elide_outputs_after=keep)
    log = ConversationLog()
    for index, (name, args, output) in enumerate(steps, 1):
        req = replace(request(), call_id=f's/model/{index}')
        reply = adapter.generate(req, execution())
        log.append({'kind': 'model_returned', 'call_id': req.call_id, 'reply': asdict(reply)})
        log.append({'kind': 'effect_returned', 'result': {'effect_id': f's/effect/{index}', 'status': 'succeeded',
                    'output_json': json.dumps(output), 'progressed': True}})
    before = log.snapshot()
    raw = json.dumps(adapter.wire(replace(request(), context=log.project(ProjectionSpec(1, 'fixture')))))
    elided = keep is not None
    assert ('first body' in raw) is not elided and ('second match text' in raw) is not elided
    assert 'third body' in raw and 'fourth body' in raw  # the last two replies keep their text
    assert ('b.py:7' in raw) is elided  # an older search keeps where it matched
    assert log.snapshot() == before
    with pytest.raises(ValueError):
        NativeModelAdapter(profile(protocol), Client([]), system='Stable', tool_specs_json=SPECS, elide_outputs_after=0)


@pytest.mark.parametrize('force', [False, True])
def test_the_last_budgeted_call_must_deliver_only_when_enabled(force):
    adapter = NativeModelAdapter(profile('openai-chat'), Client([]), system='Stable', tool_specs_json=SPECS,
                                 force_final_delivery=force)
    named = {'type': 'function', 'function': {'name': 'tare_deliver'}}
    assert adapter.wire(replace(request(), remaining_model_calls=2))['tool_choice'] == 'auto'
    assert adapter.wire(replace(request(), remaining_model_calls=1))['tool_choice'] == (named if force else 'auto')
    from compute_plane.native_responses import to_responses
    wire = adapter.wire(replace(request(), remaining_model_calls=1))
    wire['messages'] = wire['messages'][:2]
    assert to_responses(wire)['tool_choice'] == ({'type': 'function', 'name': 'tare_deliver'} if force else 'auto')
