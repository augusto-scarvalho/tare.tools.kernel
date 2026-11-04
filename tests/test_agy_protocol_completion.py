"""A terminal, validated exchange can end a CLI whose background threads linger."""
import json
from pathlib import Path
import sys
import pytest
from compute_plane import managed_process, native_cli
from tests.test_native_models import execution


def boundary(model='fixture'):
    return [{'phase':'pre','model':model},{'phase':'post','model':model},
            {'phase':'stop','model':model,'termination':'TERMINAL_CUSTOM_HOOK'}]


def result():
    return {'event':'result','result':{'status':'SUCCESS',
        'structured_output':{'text':'Observed result.','actions':[]}}}


@pytest.mark.parametrize('fault',[None,'missing_stop','wrong_model','duplicate','truncated','native_tool','error'])
def test_only_full_valid_terminal_protocol_can_request_cleanup(tmp_path,fault):
    hooks=boundary('wrong' if fault=='wrong_model' else 'fixture')
    if fault=='missing_stop':hooks.pop()
    (tmp_path/'boundary.jsonl').write_text('\n'.join(json.dumps(h) for h in hooks))
    rows=[result()]
    if fault=='duplicate':rows.append(result())
    if fault=='error':rows[0]['result']['status']='FAILED'
    if fault=='native_tool':rows.insert(0,{'event':'step_update','step_update':{'tool_info':{'name':'read'}}})
    raw='\n'.join(json.dumps(r) for r in rows).encode()+ (b'\n{"event":' if fault=='truncated' else b'\n')
    observer=native_cli.AgyCompletion(tmp_path,'fixture',True)
    observer.feed(raw[:17]);assert not observer.complete()
    observer.feed(raw[17:])
    assert observer.complete() is (fault is None)


def test_validated_result_cleans_lingering_process_without_waiting_for_timeout(tmp_path,monkeypatch):
    executable=tmp_path/'worker.py'
    executable.write_text('import json,time\nfrom pathlib import Path\n'
        f'Path("boundary.jsonl").write_text({json.dumps(chr(10).join(json.dumps(h) for h in boundary()))})\n'
        f'print({json.dumps(json.dumps(result()))},flush=True)\ntime.sleep(30)\n')
    physical=dict(started=True,cleanup_verified=True,exit_code=0,cancelled=False,timeout=False,output_limit=False)
    def process(argv,**options):
        if '--version' in argv:return physical,b'1.4.12',b''
        if '--help' in argv:return physical,b'',(' '.join(native_cli.CLI_SWITCHES['cli://agy'])).encode()
        return managed_process.run_pipe_process([sys.executable,'-I',str(executable)],**options)
    monkeypatch.setattr(native_cli,'run_pipe_process',process)
    client=native_cli.VendorCliClient('cli://agy',executable=str(tmp_path/'agy.exe'),version='1.2.7',timeout_seconds=10)
    body=native_cli.cli_request({'model':'fixture','messages':[],
        'tools':[{'function':{'name':'read','parameters':{}}}]},structured=True)
    response,receipt=client.post_observed('',body,execution())
    assert response['choices'][0]['message']['content']=='Observed result.'
    assert receipt['protocol_complete'] and receipt['cleanup_verified'] and receipt['exit_code'] is None
    assert not receipt['timeout'] and receipt['wall_ms']<8000
    assert receipt['task_tools_executed']==0 and receipt['remote_stop']=='unknown'
