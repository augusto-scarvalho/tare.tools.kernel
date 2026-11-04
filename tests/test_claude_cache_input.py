"""Cache boundaries preserve quoted data, including a changing runtime allowance."""
import json
import pytest
from compute_plane.native_cli import cli_request
from compute_plane.native_cli_claude import input_record, rejection_reason, provider_failure


def body(records, **extra):
    return cli_request({'model':'claude-sonnet-5-5', 'messages':records,
        'tools':[{'function':{'name':'read','parameters':{'type':'object'}}}], **extra})


@pytest.mark.parametrize('structured',[False,True])
def test_one_input_preserves_the_full_record_and_the_final_reminder(structured):
    b=cli_request({'model':'fixture','messages':[{'role':'system','content':'trusted'},
        {'role':'user','content':'quoted \\ file\nsource'}], 'tools':[]},structured=structured)
    b['reminder']='\n\nCURRENT REQUIREMENT'
    record=input_record(b);blocks=record['message']['content']
    assert record['type']=='user' and record['message']['role']=='user'
    assert blocks[-1]['text']==b['reminder']
    assert json.loads(''.join(x['text'] for x in blocks[:-1]))==json.loads(b['prompt'])
    assert sum('cache_control' in x for x in blocks)==1
    assert next(x['cache_control'] for x in blocks if 'cache_control' in x)=={'type':'ephemeral','ttl':'1h'}
    assert '\n' not in json.dumps(record)  # Exactly one wire input, no replayed turns.


def test_appending_results_and_changing_budget_preserves_the_previous_cache_boundary():
    history=[{'role':'system','content':'Stable policy'}, {'role':'user','content':'Stable task'}]
    old=input_record(body([*history,{'role':'user','content':'Runtime budget: 3 remain'}]))['message']['content']
    new=input_record(body([*history,{'role':'assistant','content':'Observed decision'},
        {'role':'tool','content':'Observed result'}, {'role':'user','content':'Runtime budget: 2 remain'}]))['message']['content']
    boundary=next(i for i,b in enumerate(old) if 'cache_control' in b)
    assert [x['text'] for x in old[:boundary+1]]==[x['text'] for x in new[:boundary+1]]
    assert 'Runtime budget:' not in ''.join(x['text'] for x in old[:boundary+1])
    assert 'Runtime budget: 3' in ''.join(x['text'] for x in old)
    assert 'Runtime budget: 2' in ''.join(x['text'] for x in new)


def test_required_delivery_changes_only_the_uncached_tail():
    normal=body([{'role':'user','content':'Stable history'}])
    forced=body([{'role':'user','content':'Stable history'}],
                tool_choice={'type':'function','function':{'name':'read'}})
    first=input_record(normal)['message']['content'];last=input_record(forced)['message']['content']
    boundary=next(i for i,b in enumerate(first) if 'cache_control' in b)
    assert first[:boundary+1]==last[:boundary+1]
    assert json.loads(''.join(x['text'] for x in last[:-1]))['turn_constraints']=={'required_action':'read'}


@pytest.mark.parametrize('reply,reason',[
    ('{"PRIVATE-KEY":"PRIVATE-VALUE"}','ENVELOPE_KEYS'),
    ('{"text":','INVALID_JSON'),
    ('{"text":"one","text":"two","tool_calls":[]}','INVALID_JSON'),
    ('{"text":"","tool_calls":[{"name":"PRIVATE-NAME","arguments":{}}]}','CALL_KEYS'),
    ('{"text":"","tool_calls":[{"id":"one","name":"read","arguments":"PRIVATE"}]}','ARGUMENTS_TYPE'),
])
def test_rejection_diagnostics_expose_only_a_finite_schema_reason(reply,reason):
    assert rejection_reason(reply)==reason
    assert 'PRIVATE' not in rejection_reason(reply)


def test_provider_diagnostics_never_retain_account_or_request_text():
    report=provider_failure('API Error: 400 invalid_request_error: maximum cache_control content; PRIVATE-KEY sk-secret@example.com')
    assert report=={'status':400,'markers':['cache_control','maximum','content','invalid_request_error']}
    assert 'PRIVATE' not in json.dumps(report) and 'secret' not in json.dumps(report)


