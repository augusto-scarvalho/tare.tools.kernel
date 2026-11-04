"""Reject malformed data unchanged while reporting only finite syntax categories."""
import json
import pytest
from compute_plane.native_cli_claude import rejection_reason,json_rejection_details


@pytest.mark.parametrize('raw,category',[
    ('{"secret path":"first\nsecond"}','CONTROL_CHARACTER'),
    ('{"a":1 "b":2}','MISSING_COMMA'),
    ('{"a":"unterminated}','UNTERMINATED_STRING'),
    ('{"a":1} {"b":2}','EXTRA_DATA'),
    ('{"a":"bad\\q"}','INVALID_ESCAPE'),
    ('{"a":1,}','OBJECT_KEY'),
])
def test_finite_syntax_diagnostics_do_not_keep_private_content(raw,category):
    assert rejection_reason(raw)=='INVALID_JSON'
    details=json_rejection_details(raw)
    assert details['category']==category
    assert set(details)=={'category','position','line','column'}
    assert all(type(details[k]) is int for k in ('position','line','column'))
    assert 'secret' not in json.dumps(details) and raw not in json.dumps(details)


def test_valid_json_has_no_syntax_diagnostic_and_duplicate_fields_remain_rejected():
    assert json_rejection_details('{"text":"","tool_calls":[]}')=={}
    raw='{"secret":1,"secret":2}'
    assert rejection_reason(raw)=='INVALID_JSON'
    assert json_rejection_details(raw)=={'category':'EVENT_REJECTED'}


def test_transport_keeps_rejected_usage_and_private_free_location(tmp_path,monkeypatch):
    from compute_plane import native_cli
    from tests.test_native_models import execution
    model='claude-sonnet-5-5';text='\n<PRIVATE broken proposal>'
    rows=[{'type':'system','subtype':'init','tools':[],'mcp_servers':[]},
        {'type':'assistant','message':{'id':'one','model':model,'content':[{'type':'text','text':text}]}},
        {'type':'result','subtype':'success','is_error':False,'result':text,
         'usage':{'input_tokens':30,'output_tokens':5},'total_cost_usd':.001}]
    physical=dict(started=True,cleanup_verified=True,exit_code=0,
                  cancelled=False,timeout=False,output_limit=False)
    def process(argv,**options):
        if '--version' in argv:return physical,b'2.1.280',b''
        return physical,'\n'.join(json.dumps(r) for r in rows).encode(),b''
    monkeypatch.setattr(native_cli,'run_pipe_process',process)
    client=native_cli.VendorCliClient('cli://claude',executable=str(tmp_path/'claude.exe'),version='2.1.280')
    body=native_cli.cli_request({'model':model,'messages':[],
        'tools':[{'function':{'name':'read','parameters':{}}}]})
    response,receipt=client.post_observed('',body,execution())
    assert response['_proposal_rejection']==receipt['proposal_rejection']
    assert receipt['proposal_rejection']['json_error']=={
        'category':'EXPECTED_VALUE','position':1,'line':2,'column':1}
    assert response['choices'][0]['message']['tool_calls'][0]['function']['name']=='invalid_cli_reply'
    assert response['usage']['input_tokens']==30 and response['usage']['total_cost_usd']==.001
    assert receipt['task_tools_executed']==0 and receipt['inference_invocations_observed']==1
    assert 'PRIVATE' not in json.dumps(receipt)
