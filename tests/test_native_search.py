"""Real ripgrep through the scoped native port, without model inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from compute_plane.native_capabilities import NativeCapabilities
from compute_plane.mediated_tools import CommandInterrupted
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl, EffectRequest, ToolCall


@pytest.fixture
def setup(tmp_path):
    # This suite qualifies a real host dependency; absence must not pass silently.
    assert shutil.which('rg'), 'Install ripgrep on the test host PATH'
    root, scratch = tmp_path/'repo', tmp_path/'scratch'
    root.mkdir(); scratch.mkdir()
    (root/'src').mkdir()
    (root/'src/a.py').write_text('value = 1\nfind.me = 2\nFIND_ME = 3\n', encoding='utf-8')
    (root/'notes.txt').write_text('find.me\n', encoding='utf-8')
    (root/'.gitignore').write_text('ignored.txt\n', encoding='utf-8')
    (root/'ignored.txt').write_text('private-marker\n', encoding='utf-8')
    binding = BindingRef('work', 'binding', 'owner', 1, 'token')
    control = ExecutionControl(binding, CancellationToken(), time.monotonic()+20)
    tools = NativeCapabilities(session_id='s', binding=binding, repositories={'repo': str(root)},
        writable={}, verifiers={}, scratch_root=scratch, role='explorer',
        capabilities=['search', 'read', 'list'], specgraph_edit_context=False)
    return root, scratch, tools, control


def call(setup, query='find.me', **args):
    _, _, tools, control = setup
    result = tools.execute(EffectRequest('s', 's/effect/1', ToolCall('t', 'search',
        json.dumps({'repository': 'repo', 'query': query, **args}))), control)
    return result, json.loads(result.output_json)


def test_current_scoped_literal_regex_case_and_glob(setup):
    root, scratch, tools, _ = setup
    result, data = call(setup)
    assert result.status == 'succeeded' and result.progressed
    assert [(m['path'], m['line']) for m in data['matches']] == [('notes.txt', 1), ('src/a.py', 2)]
    assert data['process']['cleanup_verified']
    assert [p.name for p in scratch.iterdir()] == ['effect-receipts']
    receipt = json.loads(next((scratch/'effect-receipts').glob('*.json')).read_bytes())
    assert receipt['result']['status'] == 'succeeded'
    assert receipt['result']['effect_id'] == 's/effect/1'
    assert not call(setup)[0].progressed
    _, data = call(setup, query='find.me', mode='regex', case='insensitive', path='src', glob='*.py')
    assert [m['line'] for m in data['matches']] == [2, 3]
    (root/'src/a.py').write_text('find.me changed\n', encoding='utf-8')
    assert call(setup)[0].progressed
    assert tools.observations()['effects'] == 0 and tools.observations()['reads'] == {}
    observed = tools.observations()['context'][-1]
    assert observed['operation'] == 'search' and observed['tokens'] is None
    assert observed['returned_bytes'] > 0 and observed['duration_ms'] >= 0


def test_one_mediated_search_claims_inventory_and_match_separately_without_replay(setup):
    from types import SimpleNamespace
    from compute_plane.action_control import ActionSnapshot,ActionAuthorization,ActionRefused
    _,scratch,tools,control=setup
    claims=[]
    def execute(facts,effect):
        action=ActionSnapshot('work','s',facts.effect_id,'binding',1,facts.operation,
            facts.repository,facts.cwd,facts.environment_sha256,*(['a'*64]*3),facts.grant_sha256,
            'fixture',control.deadline_monotonic,argv=facts.argv,payload_json=facts.payload_json)
        permit=ActionAuthorization('fixture','nonce',action.action_sha256,'s',facts.effect_id,'binding',1,
                                   control.deadline_monotonic)
        if not tools.claim_action(action,permit):raise ActionRefused('claim was not granted')
        claims.append(facts.effect_id)
        return effect()
    tools.bind_command_mediation(SimpleNamespace(execute=execute))
    result,data=call(setup,path='src')
    assert result.status=='succeeded' and data['status']=='HIT'
    assert claims==['s/effect/1/command/1','s/effect/1/command/2']
    assert len(list((scratch/'effect-receipts').glob('*.json')))==3  # two intents and one parent result
    again,_=call(setup,path='src')
    assert again.status=='refused' and len(claims)==2


def test_published_search_schema_matches_observed_defaults_and_pattern_choice(setup):
    specs = json.loads(setup[2].tool_specs_json)['search']['parameters']['properties']
    assert specs['mode']['enum'] == ['literal', 'regex']
    assert specs['case']['enum'] == ['sensitive', 'insensitive']
    result, omitted = call(setup, query='find.me|FIND_ME')
    assert omitted['matches'] == []  # Alternation must never be inferred silently.
    assert omitted['mode'] == specs['mode']['default'] == 'literal'
    assert omitted['case'] == specs['case']['default'] == 'sensitive'
    assert [m['line'] for m in call(setup, query='find.me|FIND_ME', mode='regex', path='src')[1]['matches']] == [2, 3]
    (setup[0]/'many.txt').write_text('needle\n'*60, encoding='utf-8')
    assert len(call(setup, query='needle')[1]['matches']) == specs['limit']['default'] == 50


def test_miss_ignored_files_and_flag_shaped_query(setup):
    result, data = call(setup, query='private-marker')
    # A first miss is news: a feature task starts on red tests and looks for what does not exist yet, and two
    # misses after a red verify blocked such a session with NO_PROGRESS at its third call. Its repeat is not news.
    assert result.status == 'succeeded' and result.progressed and data['status'] == 'MISS'
    assert not call(setup, query='private-marker')[0].progressed
    (setup[0]/'flags.txt').write_text('--pre=evil\n', encoding='utf-8')
    assert call(setup, query='--pre=evil')[1]['matches'][0]['path'] == 'flags.txt'


def test_explicit_large_file_can_be_searched_and_read_without_broadening_directory_scan(setup):
    import hashlib
    root, _, tools, control = setup
    raw = b'padding\n' * 300000 + b'i2i01 evidence\n'
    assert len(raw) > 2*1024*1024
    (root/'work-graph.json').write_bytes(raw)
    broad = call(setup, query='i2i01')[1]
    assert not broad['matches'] and broad['coverage']['files_skipped'] == 1
    found = call(setup, query='i2i01', path='work-graph.json')[1]
    assert found['status'] == 'HIT' and found['coverage']['files_searched'] == 1
    line = found['matches'][0]['line']
    result = tools.execute(EffectRequest('s', 'read-large', ToolCall('r', 'read', json.dumps({
        'repository': 'repo', 'path': 'work-graph.json', 'start': line, 'end': line}))), control)
    page = json.loads(result.output_json)
    assert result.status == 'succeeded' and page['text'] == f'{line}|i2i01 evidence'
    assert page['sha256'] == hashlib.sha256(raw).hexdigest() and len(page['text'].encode()) <= 24000
    assert tools.observations()['effects'] == 0


def test_excluded_explicit_source_never_reports_a_miss(setup):
    from compute_plane.mediated_tools import MAX_READ_BYTES
    (setup[0]/'too-large.json').write_bytes(b'x'*(MAX_READ_BYTES+1))
    data = call(setup, query='anything', path='too-large.json')[1]
    assert data['status'] == 'PARTIAL' and data['truncated']
    assert data['coverage']['files_searched'] == 0 and 'not_inspected' in data['reason']


def test_limit_long_lines_and_empty_repository(setup):
    root = setup[0]
    (root/'many.txt').write_text(('find.me\n')*220, encoding='utf-8')
    result, data = call(setup, limit=2)
    assert result.progressed and data['status'] == 'PARTIAL' and len(data['matches']) == 2
    (root/'long.txt').write_text('x'*4000+'needle\n', encoding='utf-8')
    data = call(setup, query='needle', path='long.txt')[1]
    assert data['matches'][0]['text_truncated'] and len(data['matches'][0]['text']) == 2048
    (root/'empty').mkdir()
    assert call(setup, path='empty')[1]['status'] == 'MISS'


@pytest.mark.parametrize('filename',['compact.json','ação report.json'])
def test_large_compact_line_returns_locations_without_raising_output_budget(setup,filename):
    root,_,tools,_=setup
    (root/filename).write_text(json.dumps({'padding':'x'*1100000,'target':'needle'}),encoding='utf-8')
    result,data=call(setup,query='needle',path=filename)
    assert result.status=='succeeded' and data['status']=='PARTIAL'
    assert data['matches']==[{'path':filename,'line':1,'text':'','text_truncated':True}]
    assert data['includes_location_only'] and data['process']['cleanup_verified']
    assert 'interruption' not in data['process']
    recovered=data['process']['recovered_output_limits']
    assert recovered[0]['stage']=='match' and recovered[0]['process']['output_limit']
    assert len(json.dumps(data).encode())<24000
    assert tools.observations()['effects']==0 and tools.observations()['reads']=={}


def test_long_line_fallback_preserves_matches_from_earlier_batches(setup):
    root=setup[0]
    (root/'000-known.txt').write_text('needle is here\n',encoding='utf-8')
    for index in range(40):
        (root/f'filler-{index:02}.txt').write_text('not a match\n',encoding='utf-8')
    (root/'zz-compact.json').write_text(json.dumps({'padding':'x'*1100000,'target':'needle'}),encoding='utf-8')
    result,data=call(setup,query='needle',limit=100)
    assert result.status=='succeeded' and data['status']=='PARTIAL'
    found={row['path']:row for row in data['matches']}
    assert found['000-known.txt']['text']=='needle is here'
    assert found['zz-compact.json']=={'path':'zz-compact.json','line':1,'text':'','text_truncated':True}
    assert data['includes_location_only'] and data['process']['cleanup_verified']


@pytest.mark.parametrize('query,found',[('find.me',True),('absent',False)])
def test_clean_partial_inventory_searches_only_complete_captured_names_and_never_claims_absence(setup,monkeypatch,query,found):
    tools=setup[2]._tools
    original=tools.command
    def partial(argv,*args,**kwargs):
        if '--files' in argv:
            return {'output_limit':True,'returncode':None,'output_truncated':True,
                'output':'notes.txt\0src/a.py\0untrusted-incomplete-tail',
                'process':{'output_limit':True,'cleanup_verified':True,'cancelled':False,'timeout':False,'wall_ms':1}}
        return original(argv,*args,**kwargs)
    monkeypatch.setattr(tools,'command',partial)
    result,data=call(setup,query=query)
    assert result.status=='succeeded' and data['status']=='PARTIAL'
    assert data['inventory_complete'] is False and data['truncated']
    assert data['coverage']['files_selected']==data['coverage']['files_searched']==2
    assert bool(data['matches']) is found and tools.reads=={} and tools.effects==0
    assert 'interruption' not in data['process']
    recovered=data['process']['recovered_output_limits']
    assert recovered[0]['stage']=='inventory' and recovered[0]['process']['cleanup_verified']


@pytest.mark.parametrize('binary',[b'needle before\n\0needle after\n',b'PK\x03\x04\0header\nneedle in archive\n'])
@pytest.mark.parametrize('limit',[1,50])
def test_binary_ripgrep_matches_are_filtered_using_the_end_record_before_returning_a_page(setup,binary,limit):
    root=setup[0]
    (root/'000-binary.zip').write_bytes(binary)
    (root/'text.txt').write_text('needle text\n',encoding='utf-8')
    result,data=call(setup,query='needle',limit=limit)
    assert result.status=='succeeded'
    assert [row['path'] for row in data['matches']]==['text.txt']
    assert data['coverage']['binary_files_skipped']==1


@pytest.mark.parametrize('fault',[None,'cleanup_verified','cancelled','timeout','isolation_violation'])
def test_partial_command_stdout_requires_read_only_and_proved_cleanup(setup,monkeypatch,fault):
    tools=setup[2]._tools
    receipt=dict(cleanup_verified=True,cancelled=False,timeout=False,output_limit=True,exit_code=None)
    if fault: receipt[fault]=False if fault=='cleanup_verified' else True
    def process(argv,**options):
        (Path(options['dest'])/'stdout.txt').write_bytes(b'notes.txt\0src/a.py\0unfinished')
        (Path(options['dest'])/'stderr.txt').write_bytes(b'not a filename')
        return receipt
    monkeypatch.setattr('compute_plane.managed_process.run_process',process)
    args=dict(gate=False,read_only=True,output_bytes=64,partial_read=True)
    if fault:
        with pytest.raises(CommandInterrupted): tools.command(['fixture'],setup[0],**args)
    else:
        result=tools.command(['fixture'],setup[0],**args)
        assert result['output_limit'] and result['output_truncated']
        assert result['output']=='notes.txt\0src/a.py\0unfinished'
        with pytest.raises(ValueError,match='partial_output_requires_read_only'):
            tools.command(['fixture'],setup[0],**dict(args,read_only=False))


def test_more_than_one_batch_and_non_ascii_paths(setup):
    root = setup[0]
    for i in range(40):
        (root/f'file {i:02}.py').write_text('batchneedle\n', encoding='utf-8')
    (root/'ação.py').write_text('batchneedle\n', encoding='utf-8')
    data = call(setup, query='batchneedle', limit=100)[1]
    assert data['status'] == 'HIT' and len(data['matches']) == 41
    assert data['coverage']['files_searched'] == data['coverage']['files_selected']
    assert data['process']['count'] >= 3


@pytest.mark.parametrize('args', [{'path': '../outside'}, {'path': '.git'}, {'path': '/tmp'},
    {'path': 'C:/secret'}, {'path': 'missing'}, {'mode': 'shell'}, {'case': 'auto'},
    {'limit': True}, {'limit': 0}, {'limit': 201}, {'query': ''}, {'query': 'a\nb'}])
def test_bad_arguments_refused_without_effect(setup, args):
    result, _ = call(setup, **args)
    assert result.status == 'refused' and setup[2].observations()['effects'] == 0


def test_regex_error_is_not_miss_and_missing_executable_is_actionable(setup, monkeypatch):
    result, data = call(setup, query='[', mode='regex')
    assert result.status == 'failed' and data['status'] == 'ERROR'
    monkeypatch.setattr('compute_plane.native_search.host_executable', lambda name: None)
    result, data = call(setup)
    assert result.status == 'failed' and data['reason'] == 'ripgrep_not_available_on_host_PATH'


def test_interrupted_search_is_read_only_and_preserves_cleanup_receipt(setup, monkeypatch):
    def interrupted(*a, **kw):
        raise CommandInterrupted({'cancelled': True, 'cleanup_verified': True})
    monkeypatch.setattr(setup[2]._tools, 'command', interrupted)
    result, data = call(setup)
    assert result.status == 'refused' and data['process']['cleanup_verified']


@pytest.mark.parametrize('stage', ['inventory', 'match'])
def test_clean_output_limit_gives_scoped_recovery_without_claiming_miss(setup, monkeypatch, stage):
    tools = setup[2]._tools
    original = tools.command
    receipt = {'output_limit': True, 'cleanup_verified': True,
               'cancelled': False, 'timeout': False, 'wall_ms': 1}
    def overflow(argv, *args, **kwargs):
        if ('--files' in argv) == (stage == 'inventory'):
            raise CommandInterrupted(receipt)
        return original(argv, *args, **kwargs)
    monkeypatch.setattr(tools, 'command', overflow)
    result, data = call(setup)
    assert result.status == 'failed' and not result.progressed
    assert data['status'] == 'ERROR' and data['truncated'] and data['matches'] == []
    assert data['reason'].startswith(f'search_{stage}_output_limit: narrow path')
    assert data['process']['interruption'] == receipt
    assert data['coverage']['files_searched'] == 0
    assert tools.effects == 0 and tools.reads == {}
    monkeypatch.setattr(tools, 'command', original)
    assert call(setup, path='src')[1]['status'] == 'HIT'


@pytest.mark.parametrize('fault', [{'cleanup_verified': False}, {'cancelled': True}, {'timeout': True},
                                 {'isolation_violation':True}])
def test_output_limit_with_uncertain_cleanup_or_stop_preserves_interruption(setup, monkeypatch, fault):
    receipt = dict(output_limit=True, cleanup_verified=True, cancelled=False, timeout=False) | fault
    def interrupted(*args, **kwargs):
        raise CommandInterrupted(receipt)
    monkeypatch.setattr(setup[2]._tools, 'command', interrupted)
    result, data = call(setup)
    assert result.status == 'refused' and data['process'] == receipt
    assert 'search_inventory_output_limit' not in data.get('reason', '')


def test_config_file_cannot_inject_preprocessor(setup, monkeypatch):
    config = setup[0].parent/'rg-config'
    config.write_text('--pre\nnonexistent-dangerous-preprocessor\n', encoding='utf-8')
    monkeypatch.setenv('RIPGREP_CONFIG_PATH', str(config))
    assert call(setup)[1]['status'] == 'HIT'


def test_outside_evaluator_is_not_read_through_link_or_junction(setup):
    root = setup[0]
    outside = root.parent/'evaluator'; outside.mkdir()
    (outside/'oracle.py').write_text('outside-evaluator-marker\n', encoding='utf-8')
    link = root/'linked'
    if os.name == 'nt':
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside)], check=True, capture_output=True)
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        data = call(setup, query='outside-evaluator-marker')[1]
        assert data['matches'] == []
        assert call(setup, path='linked')[0].status == 'refused'
        assert call(setup, path='linked/oracle.py')[0].status == 'refused'
    finally:
        if os.name == 'nt':
            link.rmdir()
        else:
            link.unlink()


def test_explicit_selection_never_grants_edit_or_shell_to_explorer(setup):
    root, scratch, _, control = setup
    for names in (['edit'], ['run'], ['context'], [], ['search', 'search']):
        with pytest.raises(ValueError, match='native_capabilities_invalid'):
            NativeCapabilities(session_id='s', binding=control.binding, repositories={'repo': str(root)},
                writable={}, verifiers={}, scratch_root=scratch, role='explorer', capabilities=names)


def test_repository_choices_are_session_scoped_without_mutating_shared_schema(setup):
    from compute_plane.native_capabilities import TOOL_SPECS
    root, scratch, tools, control = setup
    original = json.dumps(TOOL_SPECS, sort_keys=True)
    other = NativeCapabilities(session_id='other', binding=tools.binding,
        repositories={'owner': str(root)}, writable={}, verifiers={}, scratch_root=scratch,
        role='explorer', capabilities=['search', 'read', 'list'], specgraph_edit_context=False)
    for port, name in ((tools, 'repo'), (other, 'owner'), (tools, 'repo')):
        for spec in json.loads(port.tool_specs_json).values():
            assert spec['parameters']['properties']['repository']['enum'] == [name]
    assert json.dumps(TOOL_SPECS, sort_keys=True) == original


def test_a_case_given_as_mode_and_a_missing_folder_are_corrected_not_refused(setup):
    result, data = call(setup, query='FIND_ME', mode='insensitive')
    assert result.status == 'succeeded' and data['case'] == 'insensitive' and len(data['matches']) == 1
    result, data = call(setup, query='value', path='src/missing/deeper')
    assert result.status == 'succeeded' and data['path'] == 'src' and data['matches']
    assert data['scope_widened'] == 'src/missing/deeper does not exist; searched src instead'
    result, _ = call(setup, query='value', path='../outside')
    assert result.status == 'refused'  # widening never leaves the repository


def test_a_link_inside_a_folder_already_proven_is_still_refused(setup):
    root = setup[0]
    outside = root.parent/'evaluator-nested'; outside.mkdir()
    (outside/'oracle.py').write_text('nested-outside-marker\n', encoding='utf-8')
    link = root/'src'/'linked'
    if os.name == 'nt':
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside)], check=True, capture_output=True)
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        # src/a.py proves src first; files under src/linked must still be checked whole and refused.
        data = call(setup, query='nested-outside-marker')[1]
        assert data['matches'] == [] and data['coverage']['files_selected'] >= 2
        _, data = call(setup, query='value', path='src')
        assert [m['path'] for m in data['matches']] == ['src/a.py']
    finally:
        if os.name == 'nt':
            link.rmdir()
        else:
            link.unlink()
