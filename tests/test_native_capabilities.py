"""Real mediated effects and containment, without a model service or Aider imports."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from compute_plane.native_capabilities import NativeCapabilities
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl, EffectRequest, ToolCall


@pytest.fixture
def setup(tmp_path):
    root, scratch = tmp_path/'repo', tmp_path/'custody'
    root.mkdir(); scratch.mkdir()
    (root/'a.py').write_text('value = 1\n')
    binding = BindingRef('work', 'binding', 'owner', 1, 'token')
    control = ExecutionControl(binding, CancellationToken(), time.monotonic()+20)
    tools = NativeCapabilities(session_id='s', binding=binding, repositories={'repo': str(root)},
        writable={'repo': ['a.py']}, verifiers={'unit': {'repository': 'repo', 'argv': [sys.executable, '-c',
            "from pathlib import Path; assert Path('a.py').read_text() == 'value = 2\\n'"]}},
        scratch_root=scratch, specgraph_edit_context=False)
    return root, scratch, tools, control


def call(tools, control, capability, **args):
    return tools.execute(EffectRequest('s', 's/effect/1', ToolCall('t', capability, json.dumps(args))), control)


@pytest.mark.parametrize('operation', [{}, {'search': 'value = 1'}, {'replace': 'value = 2'}])
def test_model_edit_schema_requires_an_operation_without_weakening_runtime(setup, operation):
    from jsonschema import Draft202012Validator
    root, _, tools, control = setup
    schema = json.loads(tools.tool_specs_json)['edit']['parameters']
    args = dict(repository='repo', path='a.py', sha256=hashlib.sha256((root/'a.py').read_bytes()).hexdigest())
    validator = Draft202012Validator(schema)
    assert not validator.is_valid(args | operation)
    result = call(tools, control, 'edit', **(args | operation))
    assert result.status == 'refused'
    assert (root/'a.py').read_text() == 'value = 1\n'
    payload = {'changes': [{'search': 'value = 1', 'replace': 'value = 2'}]}
    assert validator.is_valid(args | payload)
    # Historical direct calls keep their executor contract and all fences.
    assert call(tools, control, 'edit', **(args | {'search': 'value = 1', 'replace': 'value = 2'})).status == 'succeeded'


@pytest.mark.parametrize('width', [10, 400])
def test_oversized_read_returns_bounded_contiguous_pages_without_retry(setup, width):
    root, _, tools, control = setup
    (root/'a.py').write_text(('x' * width + '\n') * 430)
    args = dict(repository='repo', path='a.py', start=197, end=430)
    observed = []
    while True:
        result = call(tools, control, 'read', **args)
        assert result.status == 'succeeded'
        page = json.loads(result.output_json)
        lines = page['text'].splitlines()
        assert len(lines) <= 200 and len(page['text'].encode()) <= 24000
        assert page['requested_end'] == 430
        assert page['truncated'] == (page['end'] < 430)
        observed.extend(int(line.split('|',1)[0]) for line in lines)
        if not page['truncated']:
            break
        args.update(start=page['next_start'], sha256=page['sha256'])
    assert observed == list(range(197, 431))
    (root/'a.py').write_text('changed\n')
    stale = call(tools, control, 'read', repository='repo', path='a.py', start=1, end=430,
                 sha256=page['sha256'])
    assert stale.status == 'failed' and json.loads(stale.output_json)['status'] == 'STALE'


def test_long_repository_paths_keep_scope_hashes_and_file_operations(tmp_path):
    from compute_plane.mediated_tools import MediatedTools
    from compute_plane.native_journal import io_path
    root=tmp_path/'project';root.mkdir()
    folder='nested/'+'x'*90+'/'+'y'*90
    name=folder+'/source.txt';target=folder+'/renamed.txt'
    source=io_path(root/name);source.parent.mkdir(parents=True)
    source.write_bytes(b'value = 1\r\n')
    assert len(str(root/name))>260
    tools=MediatedTools({'role':'implementer','repositories':{'repo':str(root)},
                         'writable':{'repo':['nested']},'specgraph_edit_context':False})
    assert tools.path('repo',name)==root/name
    assert 'source.txt' in tools.perform('list',{'repository':'repo','path':folder})['entries']
    read=tools.perform('read',{'repository':'repo','path':name})
    edit=tools.perform('edit',{'repository':'repo','path':name,'sha256':read['sha256'],
                             'search':'value = 1\n','replace':'value = 2\n'})
    assert source.read_bytes()==b'value = 2\r\n'
    with pytest.raises(ValueError,match='source_changed_read_again'):
        tools.perform('edit',{'repository':'repo','path':name,'sha256':read['sha256'],
                              'changes':[], 'delete':True})
    with pytest.raises(ValueError,match='outside_writable_scope'):
        tools.perform('rename',{'repository':'repo','path':name,'sha256':edit['sha256'],'target':'outside.txt'})
    tools.perform('rename',{'repository':'repo','path':name,'sha256':edit['sha256'],'target':target})
    assert not source.exists() and io_path(root/target).read_bytes()==b'value = 2\r\n'
    tools.perform('edit',{'repository':'repo','path':target,'sha256':edit['sha256'],
                          'changes':[], 'delete':True})
    assert not io_path(root/target).exists()


def test_real_read_exact_edit_verify_and_scoped_scratch(setup):
    root, scratch, tools, control = setup
    assert call(tools, control, 'verify', name='unit').status == 'failed'
    read = call(tools, control, 'read', repository='repo', path='a.py')
    sha = json.loads(read.output_json)['sha256']
    assert call(tools, control, 'edit', repository='repo', path='a.py', sha256=sha,
                search='value = 1', replace='value = 2').status == 'succeeded'
    verified = call(tools, control, 'verify', name='unit')
    data = json.loads(verified.output_json)
    assert verified.status == 'succeeded' and data['process']['cleanup_verified']
    assert (root/'a.py').read_text() == 'value = 2\n'
    assert [p.name for p in scratch.iterdir()] == ['effect-receipts']


def test_verifier_command_scratch_crossing_windows_path_limit_stays_in_custody(tmp_path):
    from compute_plane.mediated_tools import MediatedTools
    from compute_plane.native_journal import io_path
    root=tmp_path/'repo';root.mkdir()
    scratch=tmp_path/('s'*max(1,244-len(str(tmp_path))-1))
    io_path(scratch).mkdir()
    tools=MediatedTools({'role':'implementer','repositories':{'repo':str(root)},'writable':{},
        'verifiers':{'unit':{'repository':'repo','argv':[sys.executable,'-c','print("verified")']}},
        'specgraph_edit_context':False},scratch_root=scratch)
    result=tools.perform('verify',{'name':'unit'})
    assert result['returncode']==0 and 'verified' in result['output']
    assert result['process']['cleanup_verified'] and list(io_path(scratch).iterdir())==[]


def test_repeated_read_is_not_new_progress(setup):
    _, _, tools, control = setup
    assert call(tools,control,'read',repository='repo',path='a.py').progressed
    result = call(tools,control,'read',repository='repo',path='a.py')
    assert result.status == 'succeeded' and not result.progressed


def test_read_after_own_edit_and_verification_preserves_receipt_but_external_change_clears_it(setup):
    root, _, tools, control = setup
    read = call(tools, control, 'read', repository='repo', path='a.py')
    before = json.loads(read.output_json)['sha256']
    assert call(tools, control, 'edit', repository='repo', path='a.py', sha256=before,
                search='value = 1', replace='value = 2').status == 'succeeded'
    # Editing is not a new read attestation for the modified source.
    assert tools.observations()['reads']['repo:a.py'] == before
    assert call(tools, control, 'verify', name='unit').status == 'succeeded'
    assert call(tools, control, 'read', repository='repo', path='a.py').status == 'succeeded'
    assert tools.observations()['verifications']['unit']['returncode'] == 0
    (root/'a.py').write_text('value = 3\n')
    call(tools, control, 'read', repository='repo', path='a.py')
    assert tools.observations()['verifications'] == {}


def test_next_edit_invalidates_verified_state_immediately(setup):
    _, _, tools, control = setup
    sha = json.loads(call(tools, control, 'read', repository='repo', path='a.py').output_json)['sha256']
    edited = call(tools, control, 'edit', repository='repo', path='a.py', sha256=sha,
                  search='value = 1', replace='value = 2')
    assert call(tools, control, 'verify', name='unit').status == 'succeeded'
    call(tools, control, 'edit', repository='repo', path='a.py',
         sha256=json.loads(edited.output_json)['sha256'], search='value = 2', replace='value = 3')
    assert tools.observations()['verifications'] == {}


@pytest.mark.parametrize('role', ['planner', 'plan-auditor', 'auditor', 'explorer'])
def test_role_is_enforced_at_the_capability_boundary(setup, role):
    root, scratch, _, control = setup
    before = (root/'a.py').read_bytes()
    tools = NativeCapabilities(session_id='s', binding=control.binding, repositories={'repo': str(root)},
        writable={}, verifiers={}, scratch_root=scratch, role=role, specgraph_edit_context=False)
    assert call(tools, control, 'read', repository='repo', path='a.py').status == 'succeeded'
    assert call(tools, control, 'edit', repository='repo', path='a.py',
        sha256=hashlib.sha256((root/'a.py').read_bytes()).hexdigest(), search='1', replace='2').status == 'refused'
    assert (root/'a.py').read_bytes() == before


def test_read_start_defaults_to_a_bounded_page_at_that_line(setup):
    root,_,tools,control=setup
    (root/'a.py').write_text(''.join(f'value = {i}\n' for i in range(1,501)))
    result=call(tools,control,'read',repository='repo',path='a.py',start=250)
    assert result.status=='succeeded',result
    lines=json.loads(result.output_json)['text'].splitlines()
    assert len(lines)==200 and lines[0]=='250|value = 250' and lines[-1]=='449|value = 449'
    result=call(tools,control,'read',repository='repo',path='a.py',start=450)
    assert result.status=='succeeded' and len(json.loads(result.output_json)['text'].splitlines())==51
    page = call(tools,control,'read',repository='repo',path='a.py',start=250,end=450)
    assert page.status == 'succeeded' and json.loads(page.output_json)['truncated']


@pytest.mark.parametrize('replacement', ['new rules', 'old rules'])
def test_instruction_edit_reports_boundary_without_replacing_pins(setup, replacement):
    root, scratch, _, control = setup
    (root/'nested').mkdir()
    (root/'nested/AGENTS.md').write_text('old rules', encoding='utf-8')
    (root/'nested/a.py').write_text('value = 1\n', encoding='utf-8')
    tools = NativeCapabilities(session_id='s', binding=control.binding, repositories={'repo':str(root)},
        writable={'repo':['nested']}, verifiers={}, scratch_root=scratch,
        specgraph_edit_context=False, scoped_instructions=True)
    page = json.loads(call(tools, control, 'read', repository='repo', path='nested/AGENTS.md').output_json)
    changed = call(tools, control, 'edit', repository='repo', path='nested/AGENTS.md', sha256=page['sha256'],
                   changes=[{'search':'old rules', 'replace':replacement}])
    assert changed.status == 'succeeded', changed.output_json
    assert (root/'nested/AGENTS.md').read_text(encoding='utf-8') == replacement
    assert tools.instructions.records()[0]['sha256'] == page['sha256']
    following = call(tools, control, 'read', repository='repo', path='nested/a.py')
    if replacement == 'new rules':
        boundary = json.loads(changed.output_json)['instruction_update']
        assert boundary['applied'] and boundary['requires_new_conversation'] and not boundary['guidance_adopted']
        assert following.status == 'refused'
        refusal = json.loads(following.output_json)
        assert refusal['reason'].startswith('instruction_pinned_row_changed:')
        assert 'start a new conversation' in refusal['next_action']
    else:
        assert 'instruction_update' not in json.loads(changed.output_json)
        assert following.status == 'succeeded'
    fresh = NativeCapabilities(session_id='fresh', binding=control.binding, repositories={'repo':str(root)},
        writable={'repo':['nested']}, verifiers={}, scratch_root=scratch,
        specgraph_edit_context=False, scoped_instructions=True)
    reopened = fresh.execute(EffectRequest('fresh', 'fresh/effect/1', ToolCall('fresh-read', 'read',
        json.dumps({'repository':'repo','path':'nested/a.py'}))), control)
    assert reopened.status == 'succeeded'
    assert fresh.instructions.records()[0]['text'] == replacement


@pytest.mark.parametrize('indent', ['  ', '\t', ' \t'])
@pytest.mark.parametrize('newline', ['\n', '\r\n'], ids=['lf', 'crlf'])
def test_read_source_copies_into_exact_edit_without_separator_indentation(setup, indent, newline):
    root, _, tools, control = setup
    lines = ['heading', indent+'Recognize pre-existing caf\u00e9 | source',
             indent+'whitespace policy.', 'tail']
    (root/'a.py').write_bytes((newline.join(lines)+newline).encode())
    page = json.loads(call(tools, control, 'read', repository='repo', path='a.py', start=2, end=3).output_json)
    copied = '\n'.join(line.split('|', 1)[1] for line in page['text'].splitlines())
    assert copied == '\n'.join(lines[1:3])
    before = (root/'a.py').read_bytes()
    refused = call(tools, control, 'edit', repository='repo', path='a.py', sha256=page['sha256'],
                   changes=[{'search':' '+copied, 'replace':'wrong'}])
    assert refused.status == 'refused' and (root/'a.py').read_bytes() == before
    changed = call(tools, control, 'edit', repository='repo', path='a.py', sha256=page['sha256'],
                   changes=[{'search':copied, 'replace':copied.replace('pre-existing', 'portable')}])
    assert changed.status == 'succeeded', changed.output_json
    assert (root/'a.py').read_bytes() == before.replace(b'pre-existing', b'portable')


@pytest.mark.parametrize('start,end', [(1,240), (760,960)])
def test_oversized_dogfood_reads_recover_with_bounded_pages(setup, start, end):
    root, _, tools, control = setup
    (root/'a.py').write_text(''.join(f'value = {i}\n' for i in range(1,1001)))
    page = call(tools, control, 'read', repository='repo', path='a.py', start=start, end=end)
    assert page.status == 'succeeded'
    data = json.loads(page.output_json)
    assert (data['start'], data['end'], data['next_start']) == (start, start+199, start+200)
    assert len(data['text'].splitlines()) == 200
    assert data['lines'] == 1000
    assert data['requested_end'] == end and data['truncated']
    following = call(tools, control, 'read', repository='repo', path='a.py',
                     start=data['next_start'], end=end, sha256=data['sha256'])
    assert following.status == 'succeeded'
    assert json.loads(following.output_json)['text'].splitlines()[-1] == f'{end}|value = {end}'


@pytest.mark.parametrize('root_length', [245, 254])
def test_atomic_edit_temporary_name_may_exceed_windows_max_path(tmp_path, root_length):
    root=tmp_path/('d'*(root_length-len(str(tmp_path))-1))
    # Materialize like Git core.longpaths, retaining the ordinary path at ingress.
    directory = Path('\\\\?\\'+str(root)) if os.name == 'nt' else root
    directory.mkdir()
    scratch=tmp_path/'scratch';scratch.mkdir()
    source=root/'a.py';source.write_bytes(b'old\n')
    assert len(str(source))<260 and len(str(root/'.tare-edit-12345678'))>=260
    binding=BindingRef('work','binding','owner',1,'token')
    control=ExecutionControl(binding,CancellationToken(),time.monotonic()+10)
    tools=NativeCapabilities(session_id='s',binding=binding,repositories={'repo':str(root)},
        writable={'repo':['a.py']},verifiers={},scratch_root=scratch,specgraph_edit_context=False)
    result=call(tools,control,'edit',repository='repo',path='a.py',
        sha256=hashlib.sha256(b'old\n').hexdigest(),search='old',replace='new')
    assert result.status=='succeeded',result
    assert source.read_bytes()==b'new\n' and list(root.iterdir())==[source]


@pytest.mark.parametrize('changes', [
    {'sha256':'stale'}, {'path':'../escape'}, {'path':'other.py','sha256':None},
    {'search':'value=1'}, {'search':1}, {'argv':['rm']},
])
def test_invalid_or_unfenced_edit_has_no_effect(setup, changes):
    root, _, tools, control = setup
    args = dict(repository='repo', path='a.py', sha256=hashlib.sha256((root/'a.py').read_bytes()).hexdigest(),
                search='value = 1', replace='value = 2')
    args.update(changes)
    assert call(tools, control, 'edit', **args).status == 'refused'
    assert (root/'a.py').read_text() == 'value = 1\n'


@pytest.mark.parametrize('sha', ['', 'stale', 'a'*60, 'a'*63, 'a'*65, 'g'*64, 'A'*64, 'a'*64+'\n'])
def test_invalid_hash_diagnostic_does_not_read_source_or_start_effect(setup, monkeypatch, sha):
    from jsonschema import Draft202012Validator
    root, _, tools, control = setup
    specs = json.loads(tools.tool_specs_json)
    assert not Draft202012Validator(specs['read']['parameters']).is_valid(
        dict(repository='repo', path='a.py', sha256=sha))
    assert not Draft202012Validator(specs['edit']['parameters']).is_valid(
        dict(repository='repo', path='a.py', sha256=sha, changes=[dict(search='1', replace='2')]))
    original = (root/'a.py').read_bytes()
    before = tools.observations()
    def unexpected(*args):
        pytest.fail('invalid hash must be rejected before observing bytes')
    monkeypatch.setattr(tools._tools, 'observed', unexpected)
    result = call(tools, control, 'edit', repository='repo', path='a.py',
                  sha256=sha, search='1', replace='2')
    assert result.status == 'refused' and not result.progressed
    reason = json.loads(result.output_json)['reason']
    assert reason.startswith('invalid_sha256:') and 'JSON null' in reason
    assert 'source_changed' not in reason
    assert tools.observations() == before and not tools._started
    assert (root/'a.py').read_bytes() == original


def test_string_null_creates_like_json_null_and_never_overwrites(setup):
    # Qwen's XML tool calls carry every argument as text, so NInfer delivers "null"; it is never a hash.
    root, _, tools, control = setup
    (root/'a.py').unlink()
    args = dict(repository='repo', path='a.py', search='', replace='value = 2\n')
    made = call(tools, control, 'edit', sha256='null', **args)
    assert made.status == 'succeeded' and (root/'a.py').read_bytes() == b'value = 2\n'
    assert tools.observations()['effects'] == 1
    # Neither null must ever authorize overwriting an existing file.
    for sha in (None, 'null'):
        stale = call(tools, control, 'edit', sha256=sha, **args)
        assert stale.status == 'refused'
        assert json.loads(stale.output_json)['reason'] == 'source_changed_read_again'
    assert tools.observations()['effects'] == 1


def test_string_null_create_keeps_its_review_snapshot(setup, tmp_path):
    from compute_plane.native_review import receipt_path
    root, _, tools, control = setup
    tools.review_root = tmp_path/'review'
    (root/'a.py').unlink()
    made = call(tools, control, 'edit', repository='repo', path='a.py', sha256='null', search='', replace='value = 2\n')
    assert made.status == 'succeeded' and tools.review_errors == []
    receipt = json.loads(receipt_path(tools.review_root, made.effect_id).read_text())
    assert receipt['before'] is None and receipt['after'] == json.loads(made.output_json)['sha256']


def test_valid_stale_hash_still_reports_changed_source(setup):
    root, _, tools, control = setup
    sha = json.loads(call(tools, control, 'read', repository='repo', path='a.py').output_json)['sha256']
    (root/'a.py').write_bytes(b'value = 3\n')
    result = call(tools, control, 'edit', repository='repo', path='a.py',
                  sha256=sha, search='', replace='value = 2\n')
    assert result.status == 'refused'
    assert json.loads(result.output_json)['reason'] == 'source_changed_read_again'
    assert (root/'a.py').read_bytes() == b'value = 3\n' and tools.observations()['effects'] == 0


@pytest.mark.parametrize('before,search,replacement,after', [
    (b'# note\r\nvalue = 1\r\n', '1', '2', b'# note\r\nvalue = 2\r\n'),
    (b'# note\nvalue = 1\n', '1', '2', b'# note\nvalue = 2\n'),
    (b'# note\rvalue = 1\r', '1', '2', b'# note\rvalue = 2\r'),
    (b'keep\r\nold\r\nline\r\nend', 'old\nline\n', 'new\nlines\n', b'keep\r\nnew\r\nlines\r\nend'),
    (b'keep\nold\r\nline\r\nend\n', 'old\nline', 'new\nlines', b'keep\nnew\r\nlines\r\nend\n'),
    (b'keep\r\nold\nend\r', 'old', 'new\nline', b'keep\r\nnew\nline\nend\r'),
    (b'keep\r\nold', 'old', 'new\nline', b'keep\r\nnew\r\nline'),
    (b'keep\r\n', '', 'new\n', b'keep\r\nnew\r\n'),
    (b'old', 'old', 'new', b'new'),
    (b'\xef\xbb\xbf# caf\xc3\xa9\r\nold\r\n', 'old', 'new', b'\xef\xbb\xbf# caf\xc3\xa9\r\nnew\r\n'),
    (b'', '', 'new\r\nline\n', b'new\r\nline\n'),
])
def test_edit_preserves_original_bytes_outside_match_and_local_newlines(setup,before,search,replacement,after):
    root, _, tools, control = setup
    source = root/'a.py'
    source.write_bytes(before)
    read = call(tools,control,'read',repository='repo',path='a.py')
    sha = json.loads(read.output_json)['sha256']
    result = call(tools,control,'edit',repository='repo',path='a.py',sha256=sha,
                  search=search,replace=replacement)
    assert result.status == 'succeeded', result
    assert source.read_bytes() == after
    assert json.loads(result.output_json)['sha256'] == hashlib.sha256(after).hexdigest()


def test_newline_normalization_does_not_make_ambiguous_search_unique(setup):
    root, _, tools, control = setup
    before = b'old\r\nline\r\nold\nline\n'
    (root/'a.py').write_bytes(before)
    result = call(tools,control,'edit',repository='repo',path='a.py',
                  sha256=hashlib.sha256(before).hexdigest(),search='old\nline',replace='new')
    assert result.status == 'refused'
    assert json.loads(result.output_json)['reason'] == 'exact_unique_search_required'
    assert (root/'a.py').read_bytes() == before and tools.observations()['effects'] == 0


def test_recheck_control_after_pre_edit_hook_and_unknown_after_write(setup, monkeypatch):
    from compute_plane.tool_hooks import ToolGateResult, ToolDecision
    root, _, tools, control = setup
    tools._tools.config['specgraph_edit_context'] = True
    def gate(_, event):
        control.cancellation.cancel()
        return ToolGateResult(ToolDecision.ALLOW)
    monkeypatch.setattr('compute_plane.mediated_tools.SpecGraphEditPolicy.evaluate', gate)
    args = dict(repository='repo', path='a.py', sha256=hashlib.sha256((root/'a.py').read_bytes()).hexdigest(),
                search='value = 1', replace='value = 2')
    assert call(tools, control, 'edit', **args).status == 'refused'
    assert (root/'a.py').read_text() == 'value = 1\n'
    def post_failure(_, event):
        if event.event == 'PostToolUse': raise ValueError('post-write failure')
        return ToolGateResult(ToolDecision.ALLOW)
    monkeypatch.setattr('compute_plane.mediated_tools.SpecGraphEditPolicy.evaluate', post_failure)
    control = replace(control, cancellation=CancellationToken())
    assert call(tools, control, 'edit', **args).status == 'unknown'
    assert (root/'a.py').read_text() == 'value = 2\n'


def test_wrong_binding_or_session_and_model_selected_command_refused(setup):
    _, _, tools, control = setup
    assert call(tools, replace(control,binding=replace(control.binding,fence=2)), 'read', repository='repo',path='a.py').status == 'refused'
    assert call(tools, control, 'verify', name='unit', argv=['anything']).status == 'refused'
    assert call(tools, control, 'run', repository='repo', argv=['anything']).status == 'refused'
    req = EffectRequest('other', 'e', ToolCall('t','read','{"repository":"repo","path":"a.py"}'))
    assert tools.execute(req,control).status == 'refused'


def test_argument_refusals_name_the_fields_to_fix(setup, tmp_path):
    _, _, tools, control = setup
    wrong = call(tools, control, 'read', path='a.py', offset=120, limit=90)
    assert wrong.status == 'refused'
    assert json.loads(wrong.output_json)['reason'] == ('capability_arguments_mismatch: missing repository; '
        'unexpected limit, offset; accepted end, path, repository, sha256, start, start_column')
    from compute_plane.mediated_tools import MediatedTools
    (tmp_path/'p').mkdir(); (tmp_path/'p'/'a.py').write_text('x\n')
    mediated = MediatedTools({'role': 'explorer', 'repositories': {'repo': str(tmp_path/'p')},
                              'writable': {}, 'specgraph_edit_context': False})
    with pytest.raises(ValueError, match=r'^read_choose_source_lines_or_json_pointer: .*json_pointer'):
        mediated.perform('read', {'repository': 'repo', 'path': 'a.py', 'offset': 1})


def test_verifier_deadline_keeps_cleanup_receipt_and_unknown_effect(setup):
    _, scratch, tools, control = setup
    tools._tools.config['verifiers']['unit']['argv'] = [sys.executable,'-c','import time; time.sleep(10)']
    result = call(tools, replace(control,deadline_monotonic=time.monotonic()+.3), 'verify', name='unit')
    assert result.status == 'unknown'
    receipt = json.loads(result.output_json)['process']
    assert receipt['cleanup_verified'] and (receipt['cancelled'] or receipt['timeout'])
    assert not list(scratch.iterdir())


def test_native_import_and_effects_do_not_load_aider_or_legacy_protocol(tmp_path):
    kernel = Path(__file__).resolve().parents[1]
    runtime = kernel.parent/'runtime/src'
    script = '''import sys,importlib.abc,pathlib,json,hashlib,time
sys.path[:0]=sys.argv[1:3]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='aider' or fullname.startswith('aider.') or fullname in {'compute_plane.agent_tools','compute_plane.agent_protocol','compute_plane.aider_driver','compute_plane.agent_session'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from compute_plane.native_capabilities import NativeCapabilities
from tare_tools_agent_runtime import BindingRef,CancellationToken,ExecutionControl,EffectRequest,ToolCall
p=pathlib.Path(sys.argv[3]); (p/'a').write_text('one'); (p/'scratch').mkdir()
b=BindingRef('w','b','o',1,'t'); c=ExecutionControl(b,CancellationToken(),time.monotonic()+5)
t=NativeCapabilities(session_id='s',binding=b,repositories={'r':str(p)},writable={'r':['a']},verifiers={},scratch_root=p/'scratch',specgraph_edit_context=False)
args={'repository':'r','path':'a','sha256':hashlib.sha256(b'one').hexdigest(),'search':'one','replace':'two'}
assert t.execute(EffectRequest('s','e',ToolCall('t','edit',json.dumps(args))),c).status=='succeeded'
assert (p/'a').read_text()=='two'
'''
    p = subprocess.run([sys.executable,'-c',script,str(kernel),str(runtime),str(tmp_path)],capture_output=True,text=True,timeout=15)
    assert p.returncode == 0,p.stderr


def test_a_refused_search_says_whether_it_is_missing_or_repeated(setup):
    root, _, tools, control = setup
    before = (b'import re\n'
              b'CREATE_TABLE_REGEX = re.compile(r"CREATE\s+TABLE\s+([A-Za-z0-9_\\"`]+)\s*\((.*?)\);")\n'
              b'x = 1\nx = 1\n')
    (root/'a.py').write_bytes(before)
    sha = hashlib.sha256(before).hexdigest()
    mistyped = 'CREATE_TABLE_REGEX = re.compile(r"CREATE\s+TABLE\s+([A-Za-z0-9_"`]+)\s*\((.*?)\);")'
    missing = json.loads(call(tools, control, 'edit', repository='repo', path='a.py', sha256=sha,
                              search=mistyped, replace='y').output_json)
    assert missing['reason'] == 'exact_unique_search_required'  # the code stays exact
    assert missing['detail'].startswith('The search text is not in the file') and 'line 2|CREATE_TABLE_REGEX' in missing['detail']
    repeated = json.loads(call(tools, control, 'edit', repository='repo', path='a.py', sha256=sha,
                               search='x = 1', replace='y').output_json)
    assert repeated['detail'].startswith('The search text occurs 2 times (lines 3, 4)')
    assert (root/'a.py').read_bytes() == before


def test_an_unknown_verifier_is_refused_by_name(setup):
    # A session asking for a verifier the host did not configure got a bare KeyError back (OS walkthrough, 2026-10-07).
    _, _, tools, control = setup
    result = call(tools, control, 'verify', name='tests')
    assert result.status == 'refused' and not result.progressed
    assert 'unknown_verifier:tests' in result.output_json and 'unit' in result.output_json


def test_protected_command_requires_explicit_capability_and_host_bindings(setup):
    root, scratch, ordinary, control = setup
    assert 'command' not in ordinary.names
    tools = NativeCapabilities(session_id='s', binding=control.binding, repositories={'repo':str(root)},
        writable={'repo':['.']}, verifiers={}, scratch_root=scratch, capabilities=['command'],
        specgraph_edit_context=False)
    result=call(tools,control,'command',repository='repo',argv=['python','check.py'])
    assert result.status=='refused'
    assert json.loads(result.output_json)['reason']=='protected_command_host_binding_required'
    for role in ('explorer','planner','auditor'):
        with pytest.raises(ValueError,match='native_capabilities_invalid'):
            NativeCapabilities(session_id='s',binding=control.binding,repositories={'repo':str(root)},
                writable={},verifiers={},scratch_root=scratch,capabilities=['command'],role=role,
                specgraph_edit_context=False)


def test_protected_command_uses_existing_mediated_process_and_real_result(setup,monkeypatch):
    root,scratch,_,control=setup
    seen=[]
    class Mediation:
        def execute(self,facts,effect):
            seen.append(('review',facts.argv));return effect()
    tools=NativeCapabilities(session_id='s',binding=control.binding,repositories={'repo':str(root)},
        writable={'repo':['.']},verifiers={},scratch_root=scratch,capabilities=['command'],
        specgraph_edit_context=False)
    tools.bind_command_mediation(Mediation())
    tools.bind_command_confinement(lambda *a,**kw: None,{},10)
    def process(argv,**options):
        seen.append(('effect',tuple(argv)))
        assert options['process_factory'] is tools._tools.process_factory
        Path(options['dest'],'stdout.txt').write_bytes(b'HOST OUTPUT')
        Path(options['dest'],'stderr.txt').write_bytes(b'')
        return {'exit_code':7,'cleanup_verified':True,'cancelled':False,'timeout':False,'output_limit':False}
    monkeypatch.setattr('compute_plane.managed_process.run_process',process)
    result=call(tools,control,'command',repository='repo',argv=['python','check.py'])
    assert result.status=='failed'
    assert seen==[('review',('python','check.py')),('effect',('python','check.py'))]
    output=json.loads(result.output_json)
    assert output['returncode']==7 and output['output']=='HOST OUTPUT'
    assert tools.observations()['reads']=={} and tools.observations()['verifications']=={}


@pytest.mark.parametrize('staged,fault,expected',[
    (True,{'output_limit':True},'failed'),(True,{'timeout':True},'failed'),
    (False,{'output_limit':True},'unknown'),
    (True,{'output_limit':True,'cleanup_verified':False},'unknown'),
    (True,{'output_limit':True,'cancelled':True},'unknown'),
    (True,{'output_limit':True,'isolation_violation':True},'unknown')])
def test_only_proved_staged_bounds_allow_inspection_without_replaying_or_claiming_success(
        setup,monkeypatch,staged,fault,expected):
    from dataclasses import asdict
    from compute_plane.native_journal import NativeJournal
    root,scratch,_,control=setup
    class Mediation:
        def execute(self,facts,effect):return effect()
    tools=NativeCapabilities(session_id='s',binding=control.binding,repositories={'repo':str(root)},
        writable={'repo':['a.py']},verifiers={},scratch_root=scratch,capabilities=['command','read'],
        specgraph_edit_context=False)
    tools.bind_command_mediation(Mediation())
    tools.bind_command_confinement(lambda *a,**kw:None,{},10,staged_effects=staged)
    tools._tools.reads['repo:a.py']=hashlib.sha256((root/'a.py').read_bytes()).hexdigest()
    calls=[]
    def process(argv,**options):
        calls.append(argv);(root/'a.py').write_text('partial change\n')
        return {'exit_code':None,'cleanup_verified':True,'cancelled':False,'timeout':False,
                'output_limit':False,**fault}
    monkeypatch.setattr('compute_plane.managed_process.run_process',process)
    request=EffectRequest('s','s/effect/1',ToolCall('t','command',json.dumps({'repository':'repo','argv':['fixture']})))
    journal=NativeJournal(scratch/'journal.jsonl')
    journal.append({'kind':'session_started','session_id':'s'})
    journal.append({'kind':'effect_requested','effect_id':request.effect_id,'tool_call':asdict(request.tool_call)})
    result=tools.execute(request,control);value=json.loads(result.output_json)
    assert result.status==expected and len(calls)==1 and (root/'a.py').read_text()=='partial change\n'
    assert tools.observations()['reads']=={} and tools.observations()['verifications']=={}
    if expected=='failed':
        assert value['effects_may_be_partial'] and 'Do not replay' in value['next_action']
        recovered=tools.recover_effect(journal)
        assert recovered==(request,result) and len(calls)==1
        read=tools.execute(EffectRequest('s','s/effect/2',ToolCall('r','read',
            '{"repository":"repo","path":"a.py"}')),control)
        assert read.status=='succeeded' and 'partial change' in json.loads(read.output_json)['text']


@pytest.mark.parametrize('argv',[[], 'python', ['python',None], ['python','x'*8193]],
                         ids=['empty','text','invalid-item','oversized-item'])
def test_protected_command_invalid_argv_never_reaches_mediation(setup,argv):
    root,scratch,_,control=setup
    class Never:
        def execute(self,*args):raise AssertionError('invalid argv reached review')
    tools=NativeCapabilities(session_id='s',binding=control.binding,repositories={'repo':str(root)},
        writable={'repo':['.']},verifiers={},scratch_root=scratch,capabilities=['command'],
        specgraph_edit_context=False)
    tools.bind_command_mediation(Never());tools.bind_command_confinement(lambda *a,**kw:None,{},10)
    result=call(tools,control,'command',repository='repo',argv=argv)
    assert result.status=='refused' and json.loads(result.output_json)['reason']=='protected_command_argv_invalid'


def test_delete_is_not_a_standalone_capability(setup):
    from compute_plane.mediated_tools import MediatedTools, SCHEMAS
    from compute_plane.native_capabilities import TOOL_SPECS
    root, _, tools, control = setup
    assert 'delete' not in SCHEMAS
    assert 'delete' not in tools._tools.schemas()
    assert 'delete' not in TOOL_SPECS
    assert 'delete' not in tools.names
    specs = json.loads(tools.tool_specs_json)
    assert 'delete' not in specs
    edit = specs['edit']
    assert edit['parameters']['properties']['delete']['type'] == 'boolean'
    assert 'delete=true, changes=[]' in edit['description']
    assert 'current non-null SHA-256' in edit['description']
    config = dict(role='implementer', repositories={'repo': str(root)},
                  writable={'repo': ['a.py']}, specgraph_edit_context=False)
    with pytest.raises(ValueError, match='invalid_admitted_capabilities'):
        MediatedTools(config | {'admitted_capabilities': ['edit', 'delete']})


def test_standalone_delete_is_refused_without_effects(setup):
    root, _, tools, control = setup
    before = (root/'a.py').read_bytes()
    args = dict(repository='repo', path='a.py', sha256=hashlib.sha256(before).hexdigest())
    hooks = []
    tools._tools.before_effect = lambda: hooks.append('before_effect')
    tools._tools.before_edit = lambda *args: hooks.append('before_edit')
    tools._tools.callbacks['delete'] = lambda **args: hooks.append('callback')
    with pytest.raises(ValueError, match='capability_or_arguments_not_admitted:delete'):
        tools._tools.perform('delete', args)
    assert call(tools, control, 'delete', **args).status == 'refused'
    assert not hooks
    assert tools._tools.effects == 0
    assert not tools._tools.written
    assert (root/'a.py').read_bytes() == before


def test_edit_delete_lifecycle(setup):
    from jsonschema import Draft202012Validator
    root, _, tools, control = setup
    args = dict(repository='repo', path='a.py')
    sha = json.loads(call(tools, control, 'read', **args).output_json)['sha256']
    tools._tools.verifications['old'] = {}
    deletion = dict(args, sha256=sha, changes=[], delete=True)
    assert Draft202012Validator(json.loads(tools.tool_specs_json)['edit']['parameters']).is_valid(deletion)
    result = call(tools, control, 'edit', **deletion)
    assert result.status == 'succeeded'
    assert json.loads(result.output_json)['sha256'] is None
    assert not (root/'a.py').exists()
    assert 'repo:a.py' not in tools._tools.reads
    assert not tools._tools.verifications
    assert 'repo:a.py' in tools._tools.written
    assert tools._tools.effects == 1
    missing = call(tools, control, 'read', **args)
    assert missing.status == 'succeeded'
    assert json.loads(missing.output_json) == {'exists': False, 'sha256': None}
    created = call(tools, control, 'edit', **args, sha256=None,
                   changes=[{'search': '', 'replace': 'value = 2\n'}], delete=False)
    assert created.status == 'succeeded'
    assert (root/'a.py').read_bytes() == b'value = 2\n'
    assert tools._tools.effects == 2


@pytest.mark.parametrize('override', [
    {'sha256': None}, {'sha256': 'null'}, {'sha256': 'bad'}, {'sha256': '0'*64},
    {'delete': 1}, {'delete': 'TRUE'}, {'delete': ' false'}, {'delete': ''}, {'delete': None},
    {'changes': [{'search': '', 'replace': ''}]}, {'changes': None},
    {'search': '', 'replace': ''}, {'delete': False}, {'path': 'outside.py'},
])
def test_delete_refuses_invalid_requests_without_effect(setup, override):
    root, _, tools, control = setup
    before = (root/'a.py').read_bytes()
    (root/'outside.py').write_bytes(before)
    args = dict(repository='repo', path='a.py', sha256=hashlib.sha256(before).hexdigest(),
                changes=[], delete=True)
    result = call(tools, control, 'edit', **(args | override))
    assert result.status == 'refused'
    assert (root/'a.py').read_bytes() == before
    assert (root/'outside.py').read_bytes() == before
    assert tools._tools.effects == 0


def test_delete_missing_source_and_missing_changes(setup):
    root, _, tools, control = setup
    sha = hashlib.sha256((root/'a.py').read_bytes()).hexdigest()
    args = dict(repository='repo', path='a.py', sha256=sha, delete=True)
    assert call(tools, control, 'edit', **args).status == 'refused'
    (root/'a.py').unlink()
    assert call(tools, control, 'edit', **args, changes=[]).status == 'refused'
    assert not (root/'a.py').exists()
    assert tools._tools.effects == 0


@pytest.mark.parametrize('guard', ['pre_effect', 'pre_write_sha'])
def test_delete_pre_effect_guards_preserve_source(setup, monkeypatch, guard):
    root, _, tools, control = setup
    before = (root/'a.py').read_bytes()
    args = dict(repository='repo', path='a.py', sha256=hashlib.sha256(before).hexdigest(),
                changes=[], delete=True)
    if guard == 'pre_effect':
        def refused():
            raise ValueError('pre_effect_refused')
        monkeypatch.setattr(tools._tools, 'before_effect', refused)
    else:
        original = tools._tools.observed
        calls = 0
        def changed(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                (root/'a.py').write_bytes(b'newer source\n')
            return original(*args)
        monkeypatch.setattr(tools._tools, 'observed', changed)
    result = call(tools, control, 'edit', **args)
    assert result.status == 'refused'
    expected = before if guard == 'pre_effect' else b'newer source\n'
    assert (root/'a.py').read_bytes() == expected
    assert tools._tools.effects == 0


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('delete', [True, 'true'])
def test_qwen_text_delete_reaches_executor(setup, direct, delete):
    root, _, tools, control = setup
    args = dict(repository='repo', path='a.py',
                sha256=hashlib.sha256((root/'a.py').read_bytes()).hexdigest(),
                changes=[], delete=delete)
    original = dict(args)
    if direct:
        from compute_plane.mediated_tools import MediatedTools
        mediated = MediatedTools({'role': 'implementer', 'repositories': {'repo': str(root)},
                                  'writable': {'repo': ['a.py']}, 'specgraph_edit_context': False})
        output = mediated.perform('edit', args)
    else:
        result = call(tools, control, 'edit', **args)
        assert result.status == 'succeeded'
        output = json.loads(result.output_json)
    assert output['sha256'] is None
    assert not (root/'a.py').exists()
    assert args == original
    assert (mediated if direct else tools._tools).effects == 1


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('delete', [False, 'false'])
def test_text_false_edits_content_without_converting_text(setup, direct, delete):
    root, _, tools, control = setup
    args = dict(repository='repo', path='a.py',
                sha256=hashlib.sha256((root/'a.py').read_bytes()).hexdigest(), delete=delete,
                changes=[dict(search='value = 1', replace='true false')])
    if direct:
        from compute_plane.mediated_tools import MediatedTools
        mediated = MediatedTools({'role': 'implementer', 'repositories': {'repo': str(root)},
                                  'writable': {'repo': ['a.py']}, 'specgraph_edit_context': False})
        mediated.perform('edit', args)
    else:
        assert call(tools, control, 'edit', **args).status == 'succeeded'
    assert (root/'a.py').read_text() == 'true false\n'
    assert args['delete'] == delete


def test_boolean_normalization_is_schema_scoped_and_nonmutating():
    from compute_plane.native_arguments import normalize_booleans
    args = dict(flag='true', text='false', unknown='true', changes=[dict(replace='true')])
    schema = {'properties': {'flag': {'type': 'boolean'}, 'text': {'type': 'string'}}}
    assert normalize_booleans(args, schema) == dict(args, flag=True)
    assert args['flag'] == 'true'
    for value in (True, False, None, 1, [], 'TRUE', 'False', ' true', ''):
        assert normalize_booleans({'flag': value}, schema)['flag'] is value
