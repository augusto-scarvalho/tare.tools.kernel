"""Fleet counter identity admission, no fleet/SDK dependency in the Kernel."""
from dataclasses import asdict, replace
import json

import pytest

from compute_plane.native_models import ModelProfile, NativeModelAdapter, sha
from tare_tools_agent_runtime import ModelFailure
from tests.test_native_models import execution, request, response


class Fleet:
    endpoint='http://fixture'
    def __init__(self):self.calls=[];self.change=None;self.binding=None
    def post(self,path,body,control):
        self.calls.append((path,body))
        if path=='/fleet/count':
            self.binding={'schema':'tare.tools/fleet-count/1','requested_model':'fixture-model',
                'model':'fixture-model','request_sha256':sha(body['request']),'fingerprint':'a'*64}
            live={'model':'fixture-model','context_window':8192,'fingerprint':'a'*64,'template_sha256':'b'*64}
            if self.change=='window':live['context_window']=4096
            if self.change=='model':live['model']='wrong'
            if self.change=='template':live['template_sha256']='c'*64
            if self.change=='request':self.binding['request_sha256']='c'*64
            return {'schema':'tare.tools/fleet-count/1','profile':live,'binding':dict(self.binding),'input_tokens':20}
        assert path=='/chat/completions'
        reply=response('openai-chat', 'tare_deliver')
        if self.change!='unconfirmed':reply['tare_fleet_observation']={'binding':dict(self.binding)}
        return reply


def adapter(client,**kwargs):
    from tests.test_native_models import SPECS
    profile=ModelProfile('openai-chat','fixture-model',8192,1024,'fixture','http://fixture',template_sha256='b'*64)
    return NativeModelAdapter(profile,client,system='stable',tool_specs_json=SPECS,**kwargs)


def test_qualified_count_is_used_by_generation_and_saved_reply():
    client=Fleet();model=adapter(client,counter_protocol='slop-fleet-v1')
    req=request();count=model.count(req,execution())
    assert count['status']=='observed' and count['input_tokens']==20 and count['client_invocations']==1
    reply=model.generate(replace(req,admitted_request_sha256=count['request_sha256']),execution())
    assert reply.proposes_delivery
    assert client.calls[-1][1]['tare_fleet_binding']==count['fleet_count']['binding']
    assert json.loads(reply.metadata_json)['fleet_generation']['binding']==count['fleet_count']['binding']


@pytest.mark.parametrize('fault',['window','model','template','request'])
def test_mismatched_count_does_not_admit_generation(fault):
    client=Fleet();client.change=fault;model=adapter(client,counter_protocol='slop-fleet-v1')
    count=model.count(request(),execution())
    assert count['status']=='unavailable' and count['reason']=='FLEET_COUNT_IDENTITY_MISMATCH'
    with pytest.raises(ModelFailure,match='FLEET_COUNT_REQUIRED'):
        model.generate(request(),execution())
    assert len(client.calls)==1


def test_generation_requires_count_and_confirmation():
    client=Fleet();model=adapter(client,counter_protocol='slop-fleet-v1')
    with pytest.raises(ModelFailure,match='FLEET_COUNT_REQUIRED'):model.generate(request(),execution())
    assert not client.calls
    count=model.count(request(),execution());client.change='unconfirmed'
    with pytest.raises(ModelFailure,match='FLEET_GENERATION_BINDING_UNCONFIRMED'):
        model.generate(replace(request(),admitted_request_sha256=count['request_sha256']),execution())


def test_legacy_digest_and_request_are_byte_compatible():
    client=Fleet();model=adapter(client)
    from tests.test_native_models import SPECS
    legacy={k:v for k,v in asdict(model.profile).items() if k not in ('session_header','billing_sources')}  # earlier fields
    assert model.profile_digest==sha({'profile':legacy,'system':'stable','tools':json.loads(SPECS)})
    assert 'tare_fleet_binding' not in model.wire(request())
    with pytest.raises(ValueError,match='generation gateway'):
        adapter(client,counter_protocol='slop-fleet-v1',local_counter=type('Elsewhere',(),{'endpoint':'http://other'})())
