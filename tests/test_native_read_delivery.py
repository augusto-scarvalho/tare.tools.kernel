import hashlib
import json
from dataclasses import replace
import pytest
from tests.test_native_capabilities import setup, call
from tests.test_native_models import Client, profile, response, request, execution, SPECS
from compute_plane.native_models import NativeModelAdapter
from tare_tools_agent_runtime import ModelFailure


def test_default_read_returns_useful_pages_without_false_full_coverage(setup):
    root,_,tools,control=setup
    content=''.join(f'{i} '+('é'*150)+'\n' for i in range(250))
    (root/'a.py').write_text(content,encoding='utf-8')
    result=call(tools,control,'read',repository='repo',path='a.py')
    page=json.loads(result.output_json)
    assert result.status=='succeeded' and page['truncated']
    assert 0<len(page['text'].encode())<=24000 and page['start']==1
    assert page['next_start']==page['end']+1 and page['end']<200
    second=json.loads(call(tools,control,'read',repository='repo',path='a.py',start=page['next_start']).output_json)
    assert second['text'].startswith(str(page['end']+1)+'|')
    assert tools.observations()['reads']['repo:a.py']==hashlib.sha256((root/'a.py').read_bytes()).hexdigest()


@pytest.mark.parametrize('args', [{'start':501}])
def test_refused_range_does_not_attest_content(setup,args):
    root,_,tools,control=setup
    # An invalid source location must not attest delivered content.
    (root/'a.py').write_text(('x'*500+'\n')*250)
    result=call(tools,control,'read',repository='repo',path='a.py',**args)
    assert result.status=='refused' and not result.progressed
    assert 'repo:a.py' not in tools.observations()['reads']
    reason=json.loads(result.output_json)['reason']
    assert 'start=1' in reason


def test_long_single_line_read_replaces_old_hash_only_after_content_is_delivered(setup):
    root,_,tools,control=setup
    call(tools,control,'read',repository='repo',path='a.py')
    (root/'a.py').write_text('x'*25000)
    result=call(tools,control,'read',repository='repo',path='a.py')
    assert result.status=='succeeded'
    page=json.loads(result.output_json)
    assert page['truncated'] and page['partial_line'] and page['next_start']==1
    assert tools.observations()['reads']['repo:a.py']==hashlib.sha256((root/'a.py').read_bytes()).hexdigest()


@pytest.mark.parametrize('content',['<testsuite>'+('<testcase name="ok"/>'*2400)+'</testsuite>',
                                   'ação 🦜 '*7000],ids=['junit','unicode'])
def test_long_line_pages_reconstruct_exact_unicode_without_omission_or_more_than_24k(setup,content):
    root,_,tools,control=setup
    (root/'a.py').write_text(content+'\nlast\n',encoding='utf-8')
    args=dict(repository='repo',path='a.py',start=1,end=1)
    fragments=[]
    for _ in range(20):
        result=call(tools,control,'read',**args)
        assert result.status=='succeeded',result.output_json
        page=json.loads(result.output_json)
        assert len(page['text'].encode('utf-8'))<=24000 and page['partial_line']
        fragments.append(page['text'].split('|',1)[1])
        assert page['start_column']==1+sum(map(len,fragments[:-1]))
        if page['next_start']!=1: break
        assert page['truncated'] and page['next_start_column']>page['start_column']
        args['start_column']=page['next_start_column']
    else: pytest.fail('Long-line cursor did not finish')
    assert ''.join(fragments)==content and page['next_start']==2
    tail=json.loads(call(tools,control,'read',repository='repo',path='a.py',start=2).output_json)
    assert tail['text']=='2|last'


@pytest.mark.parametrize('column',[0,-1,True,1.5,9999])
def test_invalid_column_never_attests_unread_content(setup,column):
    _,_,tools,control=setup
    result=call(tools,control,'read',repository='repo',path='a.py',start_column=column)
    assert result.status=='refused' and 'repo:a.py' not in tools.observations()['reads']


def test_column_continuation_still_rejects_a_changed_pinned_source(setup):
    root,_,tools,control=setup
    (root/'a.py').write_text('x'*40000)
    page=json.loads(call(tools,control,'read',repository='repo',path='a.py').output_json)
    (root/'a.py').write_text('y'*40000)
    result=call(tools,control,'read',repository='repo',path='a.py',start=1,
                start_column=page['next_start_column'],sha256=page['sha256'])
    assert result.status=='failed' and json.loads(result.output_json)['status']=='STALE'
    assert 'repo:a.py' not in tools.observations()['reads']


def test_pinned_hash_mismatch_does_not_attest_substituted_content(setup):
    _,_,tools,control=setup
    result=call(tools,control,'read',repository='repo',path='a.py',sha256='a'*64)
    assert result.status=='failed' and 'repo:a.py' not in tools.observations()['reads']


@pytest.mark.parametrize('protocol',['openai-chat','bedrock-converse'])
def test_missing_delivery_field_has_non_provider_diagnostic_and_usage(protocol):
    from pydantic import BaseModel
    class Delivery(BaseModel):
        summary: str
        remaining: list[str]
    value=response(protocol,'tare_deliver',usage={'inputTokens':10,'outputTokens':2} if protocol=='bedrock-converse'
                   else {'prompt_tokens':10,'completion_tokens':2})
    rows=[];client=Client([value])
    adapter=NativeModelAdapter(profile(protocol),client,system='test',tool_specs_json=SPECS,
        validate_delivery=lambda data:Delivery.model_validate(data).model_dump(),
        delivery_schema_json=json.dumps(Delivery.model_json_schema()),observe=rows.append)
    # This diagnostic contract exercises the last call, when correction cannot
    # be offered. An unbounded request now correctly offers one repair.
    with pytest.raises(ModelFailure) as caught:
        adapter.generate(replace(request(), remaining_model_calls=1), execution())
    assert caught.value.code=='INVALID_DELIVERY' and not caught.value.retryable
    assert rows[0]['status']=='invalid_delivery' and rows[0]['usage']
    assert rows[0]['delivery_validation']['issues']==[{'field':'remaining','code':'missing'}]
    assert len(client.calls)==1


@pytest.mark.parametrize('protocol',['openai-chat','bedrock-converse'])
def test_invalid_delivery_never_logs_rejected_input_values(protocol):
    from pydantic import BaseModel
    class Delivery(BaseModel):
        summary: int
    rows=[]
    value=response(protocol,'tare_deliver')
    if protocol=='openai-chat':value['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps({'summary':'PRIVATE-TEXT'})
    else:value['output']['message']['content'][-1]['toolUse']['input']={'summary':'PRIVATE-TEXT'}
    adapter=NativeModelAdapter(profile(protocol),Client([value]),system='test',tool_specs_json=SPECS,
        validate_delivery=lambda data:Delivery.model_validate(data).model_dump(),
        delivery_schema_json=json.dumps(Delivery.model_json_schema()),observe=rows.append)
    with pytest.raises(ModelFailure):adapter.generate(request(),execution())
    assert 'PRIVATE-TEXT' not in json.dumps(rows)
    assert rows[0]['delivery_validation']['issues'][0]['field']=='summary'


@pytest.mark.parametrize('lines,text,accepted', [(1,'',False),(0,'',True),(1,'1: value = 1',True)])
def test_restore_does_not_turn_legacy_empty_eof_into_read_evidence(setup,lines,text,accepted):
    from types import SimpleNamespace
    root,_,tools,_=setup
    if lines==0:(root/'a.py').write_bytes(b'')
    sha=hashlib.sha256((root/'a.py').read_bytes()).hexdigest()
    events=[{'kind':'effect_requested','effect_id':'old/read','tool_call':{
        'name':'read','arguments_json':json.dumps({'repository':'repo','path':'a.py','start':500 if not text else 1})}},
        {'kind':'effect_returned','result':{'effect_id':'old/read','status':'succeeded',
         'output_json':json.dumps({'sha256':sha,'lines':lines,'text':text})}}]
    records=[SimpleNamespace(sequence=i,event=lambda event=event:event) for i,event in enumerate(events)]
    tools.restore(SimpleNamespace(snapshot=lambda:SimpleNamespace(records=records)))
    assert ('repo:a.py' in tools.observations()['reads'])==accepted
