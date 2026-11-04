"""Malformed proposal data may receive feedback; failed custody never does."""
import json
from dataclasses import replace

import pytest
from tare_tools_agent_runtime import ModelFailure
from compute_plane import native_cli
from compute_plane.native_models import INVALID_CLI_REPLY, ModelProfile, NativeModelAdapter
from tests.test_native_models import Client, SPECS, execution, profile, request


@pytest.fixture
def transport(tmp_path, monkeypatch):
    physical = dict(started=True, cleanup_verified=True, exit_code=0,
                    cancelled=False, timeout=False, output_limit=False)
    rows = [
        {'type':'system', 'subtype':'init', 'tools':[], 'mcp_servers':[]},
        {'type':'assistant', 'message':{'id':'one', 'model':'fixture',
            'content':[{'type':'thinking', 'thinking':'private'}]}},
        {'type':'assistant', 'message':{'id':'one', 'model':'fixture',
            'content':[{'type':'text', 'text':'invalid proposal'}]}},
        {'type':'result', 'subtype':'success', 'result':'{"verdict":"ACCEPT"}',
            'usage':{'input_tokens':12, 'output_tokens':4}, 'total_cost_usd':0.706},
    ]
    dispatched = []
    def process(argv, **options):
        if '--version' in argv:
            return physical, b'2.1.280', b''
        dispatched.append((argv,options))
        return physical, b'\n'.join(json.dumps(r).encode() for r in rows), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    client = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version='2.1.280')
    def invoke(*, forced=None, summary=False, structured=False):
        chat = {'model':'fixture', 'messages':[{'role':'user','content':'Audit the candidate'}],
            'tools':[{'function':{'name':name, 'parameters':{'type':'object'}}} for name in ('read','tare_deliver')]}
        if forced:
            chat['tool_choice'] = {'type':'function','function':{'name':forced}}
        return client.post_observed('', native_cli.cli_request(chat, summary=summary, structured=structured), execution())
    return invoke, rows, physical, dispatched


@pytest.mark.parametrize('answer', [
    '{"verdict":"ACCEPT"}', '{"text":', '```json\n{}\n```',
    '{"text":"one","text":"two","tool_calls":[]}',
    'The finding concerns [path:12].',
])
def test_clean_malformed_claude_proposal_uses_existing_bounded_feedback(transport, answer):
    invoke, rows, _, dispatched = transport
    rows[-1]['result'] = answer
    response, receipt = invoke()
    assert len(dispatched) == 1  # The adapter never retries inference itself.
    calls = response['choices'][0]['message']['tool_calls']
    assert [c['function']['name'] for c in calls] == [INVALID_CLI_REPLY]
    assert response['usage'] == {'input_tokens':12, 'output_tokens':4, 'total_cost_usd':0.706}
    assert receipt['task_tools_executed'] == 0
    assert receipt['proposal_rejection']['stage']=='response_schema'
    assert receipt['proposal_rejection']['reason'] in {'INVALID_JSON','ENVELOPE_KEYS'}
    response['model'] = profile('openai-chat').model
    turn = NativeModelAdapter(profile('openai-chat'), Client([response]), system='s',
        tool_specs_json=SPECS).generate(request(), execution())
    assert not turn.tool_calls and not turn.proposes_delivery
    assert json.loads(turn.metadata_json)['recovery']['code'] == 'INVALID_CLI_REPLY'
    feedback=json.loads(turn.metadata_json)['recovery']['message']
    assert '"tool_calls"' in feedback and 'arguments is a JSON object' in feedback
    assert 'verdict' not in feedback and 'path:12' not in feedback


@pytest.mark.parametrize('damage', ['model','native_tool','extra_result','exit','cancel','cleanup'])
def test_failed_custody_or_native_tools_never_become_repair_feedback(transport, damage):
    invoke, rows, physical, _ = transport
    if damage == 'model': rows[1]['message']['model'] = 'other'
    if damage == 'native_tool': rows[0]['tools'] = ['read']
    if damage == 'extra_result': rows.append(dict(rows[-1]))
    if damage == 'exit': physical['exit_code'] = 1
    if damage == 'cancel': physical['cancelled'] = True
    if damage == 'cleanup': physical['cleanup_verified'] = False
    with pytest.raises((ModelFailure, native_cli.NativeTransportInterrupted)):
        invoke()


def test_enforced_final_choice_still_stops_once(transport):
    invoke, rows, _, dispatched = transport
    invalid = {'verdict':'ACCEPT'}
    rows[0]['tools'] = ['StructuredOutput']
    rows[1]['message']['content'] = [{'type':'tool_use', 'id':'format-one',
        'name':'StructuredOutput', 'input':invalid}]
    rows.pop(2)
    rows[-1].update(structured_output=invalid, modelUsage={'fixture':{}},
        usage={'input_tokens':12, 'iterations':[{'type':'message'}]})
    rows.insert(-1, {'type':'user','message':{'content':[{'type':'tool_result',
        'tool_use_id':'format-one'}]}})
    response, receipt = invoke(forced='tare_deliver', structured=True)
    assert receipt['proposal_rejection'] == {'stage':'tool_choice'}
    assert response['_proposal_rejection'] == {'stage':'tool_choice'}
    client = Client([response]); client.endpoint = 'cli://claude'
    cli = ModelProfile('vendor-cli','fixture',8192,1024,'offline fixture','cli://claude',
        parameters_json='{"effort":"high"}')
    with pytest.raises(ModelFailure, match='CLI_REQUIRED_ACTION'):
        NativeModelAdapter(cli, client, system='s', tool_specs_json=SPECS,
            force_final_delivery=True, counter_protocol='utf8-envelope-estimate-v1').generate(
                replace(request(), remaining_model_calls=1), execution())
    assert len(dispatched) == 1


def test_invalid_summary_still_fails_without_tool_repair(transport):
    invoke, rows, _, _ = transport
    rows[-1]['result'] = ''
    with pytest.raises(ModelFailure, match='CLI_INVALID_RESPONSE'):
        invoke(summary=True)


def test_custody_failure_is_not_repaired_even_if_it_inherits_value_error(transport, monkeypatch):
    invoke, _, _, _ = transport
    class FutureModelFailure(ValueError):
        pass
    def refuse(*args, **kwargs):
        raise FutureModelFailure('CLI_BOUNDARY_UNCONFIRMED')
    monkeypatch.setattr(native_cli, 'ModelFailure', FutureModelFailure)
    monkeypatch.setattr(native_cli, 'decode_claude', refuse)
    with pytest.raises(FutureModelFailure, match='CLI_BOUNDARY_UNCONFIRMED'):
        invoke()


@pytest.mark.parametrize('names',[[],['read'],['tare_deliver','read'],['tare_deliver']])
def test_legacy_opus_required_delivery_is_transmitted_and_enforced(transport,names):
    invoke,rows,_,dispatched=transport
    rows[-1]['result']=json.dumps({'text':'Actual findings and gaps','tool_calls':[
        {'id':str(i),'name':name,'arguments':{'summary':'Actual result'}} for i,name in enumerate(names)]})
    response,receipt=invoke(forced='tare_deliver')
    input_row=json.loads(dispatched[-1][1]['stdin'])
    assert dispatched[-1][0][dispatched[-1][0].index('--input-format')+1]=='stream-json'
    wire=''.join(b['text'] for b in input_row['message']['content'])
    record,reminder=wire.split('\n\nReturn exactly one',1)
    assert json.loads(record)['turn_constraints']=={'required_action':'tare_deliver'}
    assert 'tool_calls item named tare_deliver' in reminder
    calls=response['choices'][0]['message']['tool_calls']
    if names==['tare_deliver']:
        assert calls[0]['function']['name']=='tare_deliver'
        assert 'proposal_rejection' not in receipt
    else:
        assert [c['function']['name'] for c in calls]==[INVALID_CLI_REPLY]
        assert receipt['proposal_rejection']=={'stage':'tool_choice'}
    assert len(dispatched)==1 and receipt['task_tools_executed']==0


def test_legacy_opus_final_choice_keeps_the_cacheable_prefix():
    chat={'model':'claude-opus-5-5','messages':[{'role':'user','content':'Stable history'}],
        'tools':[{'function':{'name':'tare_deliver','parameters':{'type':'object'}}}]}
    normal=native_cli.cli_request(chat)
    final=native_cli.cli_request({**chat,'tool_choice':{'type':'function','function':{'name':'tare_deliver'}}})
    assert normal['system']==final['system']
    assert final['prompt'].startswith(normal['prompt'][:-1])
    assert json.loads(normal['prompt'])['tools']==json.loads(final['prompt'])['tools']
    assert 'response_schema' not in final  # Retain Opus's qualified toolless codec.
    with pytest.raises(ModelFailure,match='CLI_TOOL_CHOICE_UNSUPPORTED') as error:
        native_cli.cli_request({**chat,'tool_choice':{'type':'function','function':{'name':'missing'}}})
    assert error.value.not_started
