"""Stream fragmentation must not amplify a small valid response past journal ingress."""
from dataclasses import asdict,replace
import io,json

import pytest

from compute_plane.native_models import NativeHttpClient,NativeModelAdapter
from compute_plane.native_stream import chat
from tare_tools_agent_runtime import ConversationLog,ConversationRejected,ModelFailure,ModelReply,ProjectionSpec
from tests.test_native_models import Client,execution,profile,request,SPECS
from tests.test_native_streaming import sse,chat_parts


def parts(details):
    return [sse({'model':'fixture','choices':[{'index':0,'delta':{
        'reasoning_details':[detail]},'finish_reason':None}]}) for detail in details]+chat_parts()


@pytest.mark.parametrize('kind,field',[('reasoning.text','text'),('reasoning.summary','summary')])
def test_many_small_fragments_fit_existing_ingress_and_round_trip_every_character(kind,field):
    fragment={field:'olá ', 'type':kind,'format':'unknown','index':0}
    response=chat(io.BytesIO(b''.join(parts([fragment]*1000))),lambda _:None)
    adapter=NativeModelAdapter(replace(profile('openai-chat'),model='fixture'),NativeHttpClient('http://fixture'),
        system='system',tool_specs_json=SPECS)
    reply=adapter._reply(request(),response)
    expected=[{**fragment,field:'olá '*1000}]
    assert json.loads(reply.metadata_json)['assistant']['reasoning_details']==expected
    log=ConversationLog();log.append({'kind':'model_returned','call_id':request().call_id,'reply':asdict(reply)})
    followup=replace(request(),call_id='s/model/2',context=log.project(ProjectionSpec(1,'native-full-v1')))
    wire=adapter.wire(followup)
    assert next(m for m in wire['messages'] if m['role']=='assistant')['reasoning_details']==expected
    assert len(reply.metadata_json.encode())<12000


@pytest.mark.parametrize('difference',[
    {'id':'different'}, {'format':'different'}, {'index':1}, {'signature':'signed'},
    {'extension':{'opaque':True}}, {'type':'reasoning.encrypted','data':'opaque'},
])
def test_different_block_or_opaque_metadata_is_preserved_without_rewriting(difference):
    first={'type':'reasoning.text','text':'first','index':0,'format':'unknown'}
    second={**first,**difference,'text':'second'}
    expected=[first,second]
    response=chat(io.BytesIO(b''.join(parts(expected))),lambda _:None)
    assert response['choices'][0]['message']['reasoning_details']==expected


def test_encrypted_and_unindexed_blocks_remain_discrete_even_with_equal_identity():
    details=[{'type':'reasoning.encrypted','data':'opaque','index':0}]*2+[
        {'type':'reasoning.text','text':'unindexed'}]*2
    response=chat(io.BytesIO(b''.join(parts(details))),lambda _:None)
    assert response['choices'][0]['message']['reasoning_details']==details


def metadata_rejects(size):
    try: ModelReply(metadata_json=json.dumps({'data':'x'*size}))
    except ConversationRejected: return True
    return False


def test_actual_oversized_opaque_content_still_fails_once_with_fixed_reason_and_usage():
    # The Agent Runtime bounds reply metadata (64 KiB in 0.14.4, 128 KiB since 0.14.6) without
    # exporting the number, so size the payload from the installed ModelReply itself.
    oversized=next(2**k for k in range(10,24) if metadata_rejects(2**k))
    response=chat(io.BytesIO(b''.join(parts([{'type':'reasoning.encrypted',
        'data':'private-provider-text'*(oversized//21+1),'index':0}]))),lambda _:None)
    response['model']='fixture-model';response['usage']={'prompt_tokens':20,'completion_tokens':5}
    client=Client([response]);observed=[]
    adapter=NativeModelAdapter(profile('openai-chat'),client,system='system',
        tool_specs_json=SPECS,observe=observed.append)
    with pytest.raises(ModelFailure,match='INVALID_PROVIDER_RESPONSE'):
        adapter.generate(request(),execution())
    assert len(client.calls)==1
    assert observed[-1]['validation_rejection']=='EVENT_TOO_LARGE'
    assert observed[-1]['usage']==response['usage']
    assert 'private-provider-text' not in json.dumps(observed)
