"""Interrupted AGY exchanges retain diagnostic structure without leaking proposals."""
import json
from pathlib import Path
import pytest
from compute_plane import native_cli
from compute_plane.native_models import NativeTransportInterrupted
from tests.test_native_models import execution


def test_agy_plugin_reference_resolves_from_the_agent_directory(tmp_path):
    native_cli.prepare_agy(tmp_path,'Fixture system.')
    agent=tmp_path/'.agents/agents/tare-inference/agent.md'
    relative=agent.read_text(encoding='utf-8').split('plugins:\n  - ',1)[1].splitlines()[0]
    assert not Path(relative).is_absolute()
    assert (agent.parent/relative).resolve()==(tmp_path/'.agents/plugins/tare-inference').resolve()


def test_incomplete_agy_keeps_safe_progress_without_accepting_a_captured_proposal(tmp_path,monkeypatch):
    physical=dict(started=True,cleanup_verified=True,exit_code=0,cancelled=False,timeout=False,output_limit=False)
    def process(argv,**options):
        if '--version' in argv:return physical,b'1.4.12',b''
        if '--help' in argv:return physical,b'',(' '.join(native_cli.CLI_SWITCHES['cli://agy'])).encode()
        hooks=[{'phase':'pre','model':'fixture'}, {'phase':'tool','tool':'finish',
            'proposal':{'text':'PRIVATE-PROPOSAL','actions':[]}}]
        (options['cwd']/'boundary.jsonl').write_text('\n'.join(json.dumps(h) for h in hooks))
        raw=json.dumps({'event':'step_update','step_update':{'tool_info':{'name':'finish','args':'PRIVATE-ARGS'}}}).encode()
        return {**physical,'timeout':True,'exit_code':None},raw+b'\n{"event":',b'PRIVATE-STDERR'
    monkeypatch.setattr(native_cli,'run_pipe_process',process)
    client=native_cli.VendorCliClient('cli://agy',executable=str(tmp_path/'agy.exe'),version='1.2.7')
    body=native_cli.cli_request({'model':'fixture','messages':[],
        'tools':[{'function':{'name':'read','parameters':{}}}]},structured=True)
    with pytest.raises(NativeTransportInterrupted) as caught:client.post_observed('',body,execution())
    row=caught.value.transport_observation
    assert row['hook_phases']==['pre','tool'] and row['hook_proposal_observed']
    assert row['step_tool_names']==['finish'] and row['stream_event_types']==['step_update']
    assert row['incomplete_stream_lines']==1 and row['working_directory_present']
    assert row['request_bytes']<=row['request_byte_limit'] and row['stdin_bytes']>0
    assert row['task_tools_executed'] is None and row['remote_stop']=='unknown'
    assert 'PRIVATE-' not in json.dumps(row)
