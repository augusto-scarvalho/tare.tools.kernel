"""Real streaming HTTP, wire reconstruction and cancellation without inference."""
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io,json,struct,time,zlib
from threading import Event,Thread

import pytest
from compute_plane.native_stream import chat,converse
from compute_plane.native_models import NativeHttpClient,NativeModelAdapter,NativeTransportInterrupted
from tare_tools_agent_runtime import ConversationLog,NativeSession
from tests.test_native_models import profile,request,execution,SPECS


def aws(kind,value):
    headers=b''
    for key,text in [(':message-type','event'),(':event-type',kind),(':content-type','application/json')]:
        k,v=key.encode(),text.encode();headers+=bytes([len(k)])+k+b'\x07'+struct.pack('!H',len(v))+v
    payload=json.dumps(value).encode();pre=struct.pack('!II',16+len(headers)+len(payload),len(headers))
    frame=pre+struct.pack('!I',zlib.crc32(pre))+headers+payload
    return frame+struct.pack('!I',zlib.crc32(frame))


def sse(value):return b'data: '+json.dumps(value,ensure_ascii=False).encode()+b'\n\n'


def chat_parts(tool=False):
    if tool:
        deltas=[{'role':'assistant','tool_calls':[{'index':0,'id':'t1','type':'function','function':{'name':'tare_deliver','arguments':'{"summary":'}}]},
                {'tool_calls':[{'index':0,'function':{'arguments':'"olá"}'}}]}]
    else:deltas=[{'role':'assistant','content':'olá '},{'content':'mundo'}]
    return [*[sse({'model':'fixture','choices':[{'index':0,'delta':d,'finish_reason':None}]}) for d in deltas],
        sse({'model':'fixture','choices':[{'index':0,'delta':{},'finish_reason':'tool_calls' if tool else 'stop'}]}),
        sse({'model':'fixture','choices':[],'usage':{'prompt_tokens':20,'completion_tokens':5}}),b'data: [DONE]\n\n']


def converse_parts(tool=False):
    parts=[aws('messageStart',{'role':'assistant'}),
        aws('contentBlockDelta',{'contentBlockIndex':0,'delta':{'reasoningContent':{'text':'retained'}}}),
        aws('contentBlockDelta',{'contentBlockIndex':0,'delta':{'reasoningContent':{'signature':'signature'}}}),
        aws('contentBlockStop',{'contentBlockIndex':0})]
    if tool:
        parts += [aws('contentBlockStart',{'contentBlockIndex':1,'start':{'toolUse':{'toolUseId':'t1','name':'tare_deliver'}}}),
            aws('contentBlockDelta',{'contentBlockIndex':1,'delta':{'toolUse':{'input':'{"summary":'}}}),
            aws('contentBlockDelta',{'contentBlockIndex':1,'delta':{'toolUse':{'input':'"olá"}'}}})]
    else:
        parts += [aws('contentBlockDelta',{'contentBlockIndex':1,'delta':{'text':t}}) for t in ('olá ','mundo')]
    return parts+[aws('contentBlockStop',{'contentBlockIndex':1}),
        aws('messageStop',{'stopReason':'tool_use' if tool else 'end_turn'}),
        aws('metadata',{'usage':{'inputTokens':20,'outputTokens':5,'cacheReadInputTokens':0,'cacheWriteInputTokens':0}})]


@contextmanager
def serving(protocol='openai-chat',tool=False,gate=False,cut=False,parts=None):
    seen,release,first= [],Event(),Event()
    if parts is None:
        parts=chat_parts(tool) if protocol=='openai-chat' else converse_parts(tool)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));seen.append((self.path,body))
            if self.path.endswith('/count-tokens'):
                raw=json.dumps({'inputTokens':20}).encode();self.send_response(200);self.end_headers();self.wfile.write(raw);return
            self.send_response(200)
            self.send_header('Content-Type','text/event-stream' if protocol=='openai-chat' else 'application/vnd.amazon.eventstream')
            self.end_headers()
            try:
                for i,part in enumerate(parts):
                    # Network write boundaries deliberately split UTF-8 and AWS frames.
                    for start in range(0,len(part),7):self.wfile.write(part[start:start+7]);self.wfile.flush()
                    if i==0:
                        first.set()
                        # A buffering client cannot obtain a completed response
                        # by waiting out the gate instead of observing the delta.
                        if gate and not release.wait(5):return
                    if cut and i==1:return
            except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port),seen,release,first
    finally:release.set();server.shutdown();server.server_close();thread.join(2)


def real_control():
    return replace(execution(),deadline_monotonic=time.monotonic()+10,monotonic=time.monotonic)


@pytest.mark.parametrize('tool',[False,True])
def test_openrouter_final_usage_frame_repeats_finish_without_replaying_delivery(tool):
    parts=chat_parts(tool)
    reason='tool_calls' if tool else 'stop'
    parts[-2]=sse({'model':'fixture','choices':[{'index':0,
        'delta':{'role':'assistant','content':''},'finish_reason':reason,'native_finish_reason':reason}],
        'usage':{'prompt_tokens':20,'completion_tokens':5}})
    with serving(parts=parts) as (url,seen,*_):
        observations=[]
        adapter=NativeModelAdapter(replace(profile('openai-chat'),endpoint=url,model='fixture'),NativeHttpClient(url),
            system='system',tool_specs_json=SPECS,streaming=True,observe=observations.append)
        reply=adapter.generate(request(),real_control())
        assert reply.complete and reply.proposes_delivery==tool and len(seen)==1
        assert reply.text==('olá' if tool else 'olá mundo')
        assert observations[0]['usage']=={'prompt_tokens':20,'completion_tokens':5}


@pytest.mark.parametrize('damage',['content','tool','opaque','finish','no_usage','duplicate','index'])
def test_post_finish_choice_only_allows_one_content_free_matching_usage_frame(damage):
    frame={'model':'fixture','choices':[{'index':0,'delta':{'role':'assistant','content':''},
        'finish_reason':'stop'}],'usage':{'prompt_tokens':20,'completion_tokens':5}}
    choice=frame['choices'][0]
    if damage=='content':choice['delta']['content']='later content'
    if damage=='tool':choice['delta']['tool_calls']=[]
    if damage=='opaque':choice['delta']['extension']={'private':'value'}
    if damage=='finish':choice['finish_reason']='tool_calls'
    if damage=='no_usage':frame.pop('usage')
    if damage=='index':choice['index']=1
    parts=chat_parts()[:-2]+[sse(frame)]*(2 if damage=='duplicate' else 1)+[b'data: [DONE]\n\n']
    with pytest.raises(ValueError,match='stream_choice_order'):
        chat(io.BytesIO(b''.join(parts)),lambda event:None)


def test_openrouter_reasoning_fragments_survive_stream_and_next_request_exactly():
    from dataclasses import asdict
    from tare_tools_agent_runtime import ProjectionSpec
    details=[{'type':'reasoning.text','text':'first fragment','signature':None,'index':0},
             {'type':'reasoning.text','text':'second fragment','signature':'opaque-signature','index':0},
             {'type':'reasoning.encrypted','data':'opaque-bytes','format':'unknown','index':1}]
    parts=[sse({'model':'fixture','choices':[{'index':0,'delta':{'reasoning_details':[detail]},'finish_reason':None}]})
           for detail in details]+chat_parts()
    with serving(parts=parts) as (url,*_):
        adapter=NativeModelAdapter(replace(profile('openai-chat'),endpoint=url,model='fixture'),NativeHttpClient(url),
            system='system',tool_specs_json=SPECS,streaming=True)
        reply=adapter.generate(request(),real_control())
        assert json.loads(reply.metadata_json)['assistant']['reasoning_details']==details
        log=ConversationLog();log.append({'kind':'model_returned','call_id':request().call_id,'reply':asdict(reply)})
        wire=adapter.wire(replace(request(),call_id='s/model/2',context=log.project(ProjectionSpec(1,'native-full-v1'))))
        assert next(m for m in wire['messages'] if m['role']=='assistant')['reasoning_details']==details


@pytest.mark.parametrize('details',[{},['wrong'],[None]])
def test_malformed_reasoning_detail_sequence_is_not_accepted(details):
    parts=[sse({'model':'fixture','choices':[{'index':0,'delta':{'reasoning_details':details}}]})]+chat_parts()
    with pytest.raises(ValueError,match='stream_reasoning_details_shape'):
        chat(io.BytesIO(b''.join(parts)),lambda event:None)


@pytest.mark.parametrize('reason',['stream_role','stream_opaque_metadata_changed'])
def test_chat_shape_failure_keeps_a_fixed_diagnostic_without_provider_data(reason):
    deltas=([{'role':'user','content':'private-provider-value'}] if reason=='stream_role' else
        [{'role':'assistant','extension':{'token':'private-provider-value'}},
         {'extension':{'token':'another-private-value'}}])
    parts=[sse({'model':'fixture','choices':[{'index':0,'delta':delta,'finish_reason':None}]}) for delta in deltas]
    with serving(parts=parts) as (url,seen,*_):
        with pytest.raises(NativeTransportInterrupted) as caught:
            NativeHttpClient(url).post_stream_observed('/chat/completions',{},real_control(),protocol='openai-chat')
        receipt=caught.value.transport_observation
        assert receipt['stream_failure_reason']==reason and receipt['cleanup_verified']
        assert receipt['remote_stop']=='unknown' and len(seen)==1
        assert 'private-provider-value' not in json.dumps(receipt) and 'another-private-value' not in json.dumps(receipt)


@pytest.mark.parametrize('protocol',['openai-chat','bedrock-converse'])
@pytest.mark.parametrize('tool',[False,True])
def test_stream_reassembles_text_tools_usage_and_signed_reasoning(protocol,tool):
    with serving(protocol,tool) as (url,seen,release,first):
        rows,events=[],[];p=replace(profile(protocol),endpoint=url,model='fixture')
        adapter=NativeModelAdapter(p,NativeHttpClient(url),system='system',tool_specs_json=SPECS,
                                  streaming=True,on_progress=events.append,observe=rows.append)
        reply=adapter.generate(request(),real_control())
        assert reply.complete and reply.proposes_delivery==tool and reply.text==('olá' if tool else 'olá mundo')
        assert events and all(e['provisional'] for e in events)
        assert len(seen)==1 and rows[0]['usage'] is not None
        assert rows[0]['transport_calls'][0]['stream_status']=='COMPLETE'
        metadata=json.loads(reply.metadata_json)
        if protocol=='bedrock-converse':
            assert metadata['assistant']['content'][0]['reasoningContent']['reasoningText']=={'text':'retained','signature':'signature'}
        else:assert seen[0][1]['stream'] and seen[0][1]['stream_options']=={'include_usage':True}


def test_first_delta_is_delivered_before_server_releases_the_rest():
    with serving(gate=True) as (url,seen,release,first):
        events=[]
        def progress(event):
            if not events:
                # The event can reach the observer before the server thread sets
                # `first` after flush. The release barrier is the actual contract.
                assert not release.is_set()
            events.append(event)
            release.set()
        response,receipt=NativeHttpClient(url).post_stream_observed('/chat/completions',{},real_control(),protocol='openai-chat',on_event=progress)
        assert [event['text'] for event in events] == ['olá ', 'mundo']
        assert receipt['stream_observer_failures'] == 0 and receipt['first_content_ms'] is not None
        # The observer releases the remaining bytes; rounded clocks may tie.
        assert response['usage']['completion_tokens'] == 5
        assert 0 <= receipt['first_content_ms'] <= receipt['wall_ms'] + 0.001


def test_observer_exception_does_not_discard_completed_response():
    with serving() as (url,*_):
        def fail(event):raise OSError('display unavailable')
        response,receipt=NativeHttpClient(url).post_stream_observed('/chat/completions',{},real_control(),protocol='openai-chat',on_event=fail)
        assert response['usage']['completion_tokens']==5 and receipt['stream_observer_failures']==2


@pytest.mark.parametrize('protocol',['openai-chat','bedrock-converse'])
def test_cut_stream_never_becomes_delivery_and_resume_does_not_retry(protocol):
    with serving(protocol,tool=True,cut=True) as (url,seen,*_):
        log=ConversationLog();rows=[]
        adapter=NativeModelAdapter(replace(profile(protocol),model='fixture',endpoint=url),NativeHttpClient(url),
                                  system='system',tool_specs_json=SPECS,streaming=True,observe=rows.append)
        session=NativeSession(session_id='s',instruction='task',capabilities=('read','edit','verify'),model=adapter,tools=None,journal=log)
        outcome=session.run(real_control())
        assert outcome.status=='reconciliation_required' and outcome.pending_id=='s/model/1'
        assert session.run(real_control()).status=='reconciliation_required' and len(seen)==1
        assert rows[0]['usage'] is None and rows[0]['transport_calls'][0]['stream_status']=='INCOMPLETE'
        if protocol == 'openai-chat':
            assert rows[0]['transport_calls'][0]['stream_failure_reason'] == 'stream_missing_done'
            assert rows[0]['transport_calls'][0]['worker_error_type'] == 'ValueError'


def test_cancel_after_provisional_delta_closes_worker_and_retains_unknown_effect():
    with serving(gate=True) as (url,seen,release,first):
        ctl=real_control()
        with pytest.raises(NativeTransportInterrupted) as error:
            NativeHttpClient(url).post_stream_observed('/chat/completions',{},ctl,protocol='openai-chat',
                on_event=lambda e:ctl.cancellation.cancel())
        physical=error.value.transport_observation
        assert physical['cleanup_verified'] and physical['cancelled'] and physical['remote_stop']=='unknown'


@pytest.mark.parametrize('damage',['crc','truncate','stop_missing'])
def test_bad_aws_frame_or_unfinished_message_is_rejected(damage):
    data=b''.join(converse_parts())
    if damage=='crc':data=data[:15]+bytes([data[15]^1])+data[16:]
    if damage=='truncate':data=data[:-2]
    if damage=='stop_missing':data=b''.join(converse_parts()[:-2])
    with pytest.raises(ValueError):converse(io.BytesIO(data),lambda e:None)


def test_missing_usage_is_unknown_and_terminal_usage_is_not_summed():
    parts=chat_parts();raw=b''.join(parts[:-2]+parts[-1:])
    assert chat(io.BytesIO(raw),lambda e:None)['usage'] is None
    assert chat(io.BytesIO(b''.join(parts)),lambda e:None)['usage']=={'prompt_tokens':20,'completion_tokens':5}


def test_streaming_changes_counted_wire_and_profile_identity_only_when_selected():
    p=profile('openai-chat');client=NativeHttpClient(p.endpoint)
    normal=NativeModelAdapter(p,client,system='system',tool_specs_json=SPECS)
    stream=NativeModelAdapter(p,client,system='system',tool_specs_json=SPECS,streaming=True)
    assert stream.profile_digest!=normal.profile_digest and stream.wire(request())['stream'] is True
    assert normal.wire(request())['stream'] is False
