import json
import pytest
from compute_plane import native_cli
from tare_tools_agent_runtime import ModelFailure


@pytest.mark.parametrize('formatter,native_arguments,summary,tools,expected',[
    (False,False,False,True,True),(True,False,False,True,False),
    (False,True,False,True,False),(False,False,True,True,False),(False,False,False,False,False)])
def test_agy_external_action_reminder_keeps_the_existing_prefix_and_other_protocols(
        formatter,native_arguments,summary,tools,expected):
    records=[{'role':'user','content':'Leia o arquivo e audite a alteração.'}]
    chat={'model':'fixture','messages':records,'tools':[{'function':{'name':'read',
          'parameters':{'type':'object'}}}] if tools else []}
    body=native_cli.cli_request(chat,structured=True,formatter=formatter,
        native_arguments=native_arguments,summary=summary)
    assert (native_cli.AGY_ACTION_GUIDANCE in body.get('reminder','')) is expected
    assert native_cli.AGY_ACTION_GUIDANCE not in body['system']
    assert native_cli.AGY_ACTION_GUIDANCE not in body['prompt']
    assert json.loads(body['prompt'])['messages' if summary else 'conversation_records']==records


def test_agy_native_read_then_structured_result_is_still_rejected():
    hooks=[{'phase':phase,'model':'fixture'} for phase in ('pre','post','stop')]
    hooks[-1]['termination']='TERMINAL_CUSTOM_HOOK'
    rows=[{'event':'step_update','step_update':{'tool_info':{'name':'read'}}},
          {'event':'result','result':{'status':'SUCCESS','structured_output':{'text':'ok','actions':[]}}}]
    with pytest.raises(ModelFailure,match='CLI_UNEXPECTED_TOOL'):
        native_cli.decode_agy(rows,hooks,'fixture',structured=True)


def test_agy_read_example_is_valid_action_data():
    from jsonschema import validate
    body=native_cli.cli_request({'model':'fixture','messages':[],'tools':[
        {'function':{'name':'read','parameters':{'type':'object'}}}]},structured=True)
    sample=native_cli.AGY_ACTION_GUIDANCE.split('Formatting example only: ',1)[1]
    value,_=json.JSONDecoder().raw_decode(sample)
    validate(value,body['response_schema'])
    assert json.loads(value['actions'][0]['arguments_json'])['path']=='example.py'


@pytest.mark.parametrize('name,expected',[('read','read'),('PRIVATE-TOOL','unknown'),(['PRIVATE'],'unknown')])
def test_updated_agy_rejection_reports_only_allowlisted_step_names(tmp_path,monkeypatch,name,expected):
    from tests.test_native_models import execution
    physical={'started':True,'cleanup_verified':True,'exit_code':0,'cancelled':False,'timeout':False,'output_limit':False}
    def process(argv,**options):
        if argv[-1]=='--version':return physical,b'1.4.10',b''
        if argv[-1]=='--help':return physical,' '.join(native_cli.CLI_SWITCHES['cli://agy']).encode(),b''
        hooks=[{'phase':phase,'model':'fixture'} for phase in ('pre','post','stop')]
        hooks[-1]['termination']='TERMINAL_CUSTOM_HOOK'
        (options['cwd']/'boundary.jsonl').write_text('\n'.join(json.dumps(h) for h in hooks))
        rows=[{'event':'step_update','step_update':{'tool_info':{'name':name}}},
              {'event':'result','result':{'status':'SUCCESS','structured_output':{'text':'ok','actions':[]}}}]
        return physical,'\n'.join(json.dumps(r) for r in rows).encode(),b''
    monkeypatch.setattr(native_cli,'run_pipe_process',process)
    client=native_cli.VendorCliClient('cli://agy',executable=str(tmp_path/'agy.exe'),version='1.4.7')
    body=native_cli.cli_request({'model':'fixture','messages':[],'tools':[
        {'function':{'name':'read','parameters':{'type':'object'}}}]},structured=True)
    with pytest.raises(ModelFailure,match='CLI_UNEXPECTED_TOOL') as caught:
        client.post_observed('',body,execution())
    receipt=caught.value.transport_observation
    assert receipt['step_tool_names']==[expected] and receipt['client_version']=='1.4.10'
    assert receipt['task_tools_executed'] is None and 'PRIVATE' not in json.dumps(receipt)
