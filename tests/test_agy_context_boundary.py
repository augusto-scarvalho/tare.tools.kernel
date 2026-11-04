import json
import pytest
from compute_plane import native_cli
from tare_tools_agent_runtime import ModelFailure
from tests.test_native_models import execution


@pytest.mark.parametrize('text', ['x' * (160 * 1024), '\u00e7' * (90 * 1024)], ids=['ascii','utf8'])
def test_oversized_agy_request_is_refused_before_process_or_inference(tmp_path, monkeypatch, text):
    def forbidden(*args, **kwargs):pytest.fail('must refuse before CLI dispatch')
    monkeypatch.setattr(native_cli, 'run_pipe_process', forbidden)
    client=native_cli.VendorCliClient('cli://agy',executable=str(tmp_path/'agy.exe'),version='1.2.7')
    body={'prompt':text};before=json.dumps(body)
    with pytest.raises(ModelFailure,match='CLI_INPUT_BYTE_LIMIT_EXCEEDED') as caught:
        client.post_observed('',body,execution())
    proof=caught.value.transport_observation
    assert caught.value.not_started and proof['started'] is False and proof['cleanup_verified'] is True
    assert proof['request_bytes']>proof['request_byte_limit']==160*1024
    assert json.dumps(body)==before and text not in json.dumps(proof)


def test_agy_bound_cannot_be_raised_implicitly_and_other_cli_is_unchanged(tmp_path):
    kwargs={'executable':str(tmp_path/'cli.exe'),'version':'2.1.280','request_byte_limit':1024*1024}
    assert native_cli.VendorCliClient('cli://agy',**kwargs).request_byte_limit==160*1024
    assert native_cli.VendorCliClient('cli://claude',**kwargs).request_byte_limit==1024*1024
    kwargs['request_byte_limit']=4096
    assert native_cli.VendorCliClient('cli://agy',**kwargs).request_byte_limit==4096


def test_agy_declares_small_batches_without_rewriting_quoted_history():
    messages=[{'role':'user','content':'PRIVATE EVIDENCE'}]
    chat={'model':'fixture','messages':messages,'tools':[{'function':{'name':'read','parameters':{'type':'object'}}}]}
    agy=native_cli.cli_request(chat,structured=True)
    claude=native_cli.cli_request(chat,structured=True,formatter=True)
    assert agy['response_schema']['properties']['actions']['maxItems']==4
    assert claude['response_schema']['properties']['actions']['maxItems']==16
    assert json.loads(agy['prompt'])['conversation_records']==messages
    assert '60 lines' in agy['reminder']


def test_agy_refuses_oversized_returned_batch():
    hooks=[{'phase':phase,'model':'fixture'} for phase in ('pre','post','stop')]
    hooks[-1]['termination']='TERMINAL_CUSTOM_HOOK'
    value={'text':'','actions':[{'name':'read','arguments_json':'{}'} for _ in range(5)]}
    rows=[{'event':'result','result':{'status':'SUCCESS','structured_output':value}}]
    with pytest.raises(ModelFailure,match='CLI_ACTION_LIMIT_EXCEEDED') as caught:
        native_cli.decode_agy(rows,hooks,'fixture',structured=True)
    assert caught.value.proposed_action_count==5
