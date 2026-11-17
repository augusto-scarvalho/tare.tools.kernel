"""Real streaming HTTP, wire reconstruction and cancellation without inference."""
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io,json,struct,time,zlib
from threading import Event,Thread

import pytest
from compute_plane.native_stream import chat,converse,messages
from compute_plane.native_models import ModelProfile,NativeHttpClient,NativeModelAdapter,NativeTransportInterrupted
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


SIGNATURE='Eq/+=ólá-signature=='


def event(value):return b'event: '+value['type'].encode()+b'\ndata: '+json.dumps(value,ensure_ascii=False).encode()+b'\n\n'


def anthropic_events(tool=False):
    delta=lambda i,d:{'type':'content_block_delta','index':i,'delta':d}
    stop=lambda i:{'type':'content_block_stop','index':i}
    start=lambda i,b:{'type':'content_block_start','index':i,'content_block':b}
    events=[{'type':'message_start','message':{'id':'msg_1','type':'message','role':'assistant','model':'fixture',
            'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':20,'output_tokens':1}}},
        start(0,{'type':'thinking','thinking':'','signature':''}),{'type':'ping'},
        delta(0,{'type':'thinking_delta','thinking':'retained '}),delta(0,{'type':'thinking_delta','thinking':'thought'}),
        delta(0,{'type':'signature_delta','signature':SIGNATURE}),stop(0),
        start(1,{'type':'redacted_thinking','data':'opaque-bytes'}),stop(1)]
    if tool:
        events+=[start(2,{'type':'tool_use','id':'t1','name':'tare_deliver','input':{}}),
            delta(2,{'type':'input_json_delta','partial_json':''}),
            delta(2,{'type':'input_json_delta','partial_json':'{"summary":'}),
            delta(2,{'type':'input_json_delta','partial_json':'"olá"}'}),stop(2)]
    else:
        events+=[start(2,{'type':'text','text':''}),delta(2,{'type':'text_delta','text':'olá '}),
            {'type':'future_event','x':1},delta(2,{'type':'text_delta','text':'mundo'}),stop(2)]
    return events+[{'type':'message_delta','delta':{'stop_reason':'tool_use' if tool else 'end_turn','stop_sequence':None},
            'usage':{'output_tokens':5}},{'type':'message_stop'}]


def anthropic_parts(tool=False):return [event(e) for e in anthropic_events(tool)]


def anthropic_adapter(url,**kwargs):
    return NativeModelAdapter(ModelProfile('anthropic-messages','fixture',32768,1024,'fixture',url),NativeHttpClient(url),
        system='system',tool_specs_json=SPECS,**kwargs)


def decode_events(events):return messages(io.BytesIO(b''.join(map(event,events))),lambda e:None)


@contextmanager
def serving(protocol='openai-chat',tool=False,gate=False,cut=False,parts=None):
    seen,release,first= [],Event(),Event()
    if parts is None:
        parts={'openai-chat':chat_parts,'anthropic-messages':anthropic_parts}.get(protocol,converse_parts)(tool)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));seen.append((self.path,body))
            if self.path.endswith('/count-tokens'):
                raw=json.dumps({'inputTokens':20}).encode();self.send_response(200);self.end_headers();self.wfile.write(raw);return
            self.send_response(200)
            self.send_header('Content-Type','application/vnd.amazon.eventstream' if protocol=='bedrock-converse' else 'text/event-stream')
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


def test_opencode_usage_frame_is_kept_whole():
    usage = {'prompt_tokens': 20, 'completion_tokens': 5, 'cost': 0,
             'opencode': {'billing_source': 'go', 'allowance': 0.0001, 'tokens': {'input': 20, 'output': 5}}}
    parts = chat_parts()
    parts[-2] = sse({'model': 'fixture', 'choices': [], 'usage': usage})
    assert chat(io.BytesIO(b''.join(parts)), lambda e: None)['usage'] == usage


@pytest.mark.parametrize('tool',[False,True])
def test_anthropic_stream_reassembles_text_tools_usage_and_signed_thinking(tool):
    with serving('anthropic-messages',tool) as (url,seen,*_):
        rows,events=[],[]
        adapter=anthropic_adapter(url,streaming=True,on_progress=events.append,observe=rows.append)
        reply=adapter.generate(request(),real_control())
        assert reply.complete and reply.proposes_delivery==tool and reply.text==('olá' if tool else 'olá mundo')
        assert len(seen)==1 and seen[0][0]=='/messages' and seen[0][1]['stream'] is True and 'stream_options' not in seen[0][1]
        assert rows[0]['transport_calls'][0]['stream_status']=='COMPLETE'
        # Cumulative counts: the later output_tokens replaces the first, input_tokens is not summed.
        assert rows[0]['usage']=={'input_tokens':20,'output_tokens':5}
        assert [e['channel'] for e in events if e['channel']!='text']==['reasoning_content']*2+(['tool_arguments']*2 if tool else [])
        assert all(e['provisional'] for e in events) and (tool or [e['text'] for e in events if e['channel']=='text']==['olá ','mundo'])
        content=json.loads(reply.metadata_json)['assistant']['_anthropic_content']
        assert content[:2]==[{'type':'thinking','thinking':'retained thought','signature':SIGNATURE},
                             {'type':'redacted_thinking','data':'opaque-bytes'}]
        assert content[2]==({'type':'tool_use','id':'t1','name':'tare_deliver','input':{'summary':'olá'}} if tool
                            else {'type':'text','text':'olá mundo'})


def test_anthropic_stream_next_request_replays_the_signature_exactly():
    from dataclasses import asdict
    from tare_tools_agent_runtime import ProjectionSpec
    with serving('anthropic-messages',True) as (url,*_):
        adapter=anthropic_adapter(url,streaming=True)
        reply=adapter.generate(request(),real_control())
        log=ConversationLog();log.append({'kind':'model_returned','call_id':request().call_id,'reply':asdict(reply)})
        wire=adapter.wire(replace(request(),call_id='s/model/2',context=log.project(ProjectionSpec(1,'native-full-v1'))))
        assistant=next(m for m in wire['messages'] if m['role']=='assistant')['content']
        assert assistant[0]['signature']==SIGNATURE and assistant[1]['data']=='opaque-bytes'


def test_anthropic_nonstreaming_request_is_unchanged():
    plain=anthropic_adapter('http://fixture').wire(request())
    assert plain['stream'] is False and 'stream_options' not in plain
    assert anthropic_adapter('http://fixture',streaming=True).wire(request())=={**plain,'stream':True}


def test_anthropic_first_delta_is_delivered_before_server_releases_the_rest():
    parts=anthropic_parts();parts=[b''.join(parts[:4]),*parts[4:]]  # the gate holds after a first delta
    with serving('anthropic-messages',parts=parts,gate=True) as (url,seen,release,first):
        events=[]
        def progress(event):
            if not events: assert not release.is_set()
            events.append(event);release.set()
        response,receipt=NativeHttpClient(url).post_stream_observed('/messages',{},real_control(),
            protocol='anthropic-messages',on_event=progress)
        assert events[0]['channel']=='reasoning_content' and receipt['stream_observer_failures']==0
        assert response['stop_reason']=='end_turn' and response['usage']['output_tokens']==5


def test_anthropic_cut_stream_never_becomes_delivery_and_resume_does_not_retry():
    with serving('anthropic-messages',tool=True,cut=True) as (url,seen,*_):
        log=ConversationLog();rows=[]
        session=NativeSession(session_id='s',instruction='task',capabilities=('read','edit','verify'),
            model=anthropic_adapter(url,streaming=True,observe=rows.append),tools=None,journal=log)
        assert session.run(real_control()).status=='reconciliation_required'
        assert session.run(real_control()).status=='reconciliation_required' and len(seen)==1
        call=rows[0]['transport_calls'][0]
        assert call['stream_status']=='INCOMPLETE' and call['stream_failure_reason']=='stream_missing_completion'


def test_anthropic_error_event_keeps_a_fixed_diagnostic_without_provider_text():
    parts=anthropic_parts()[:3]+[event({'type':'error','error':{'type':'overloaded_error','message':'private-provider-value'}})]
    with serving('anthropic-messages',parts=parts) as (url,seen,*_):
        with pytest.raises(NativeTransportInterrupted) as caught:
            NativeHttpClient(url).post_stream_observed('/messages',{},real_control(),protocol='anthropic-messages')
        receipt=caught.value.transport_observation
        assert receipt['stream_failure_reason']=='stream_provider_error' and 'private-provider-value' not in json.dumps(receipt)


def test_anthropic_decoder_keeps_the_nonstreaming_shape_and_ignores_unknown_events():
    got=decode_events(anthropic_events())
    assert list(got)==['id','type','role','model','content','stop_reason','stop_sequence','usage']
    assert got['stop_reason']=='end_turn' and got['stop_sequence'] is None and got['usage']=={'input_tokens':20,'output_tokens':5}


def test_anthropic_tool_input_is_parsed_once_and_empty_input_is_an_object():
    events=anthropic_events(True)
    assert decode_events(events)['content'][2]['input']=={'summary':'olá'}
    bare=[e for e in events if e.get('delta',{}).get('type')!='input_json_delta' or e['delta']['partial_json']=='']
    assert decode_events(bare)['content'][2]['input']=={}


def drop(events,at):return events[:at]+events[at+1:]


def twice(events,at):return events[:at]+[events[at]]+events[at:]


SIGNATURE_AT,REDACTED,TEXT_START,TEXT_DELTA,TEXT_STOP,MESSAGE_DELTA,MESSAGE_STOP=5,7,9,10,13,14,15  # positions in anthropic_events()


def test_anthropic_signature_in_parts_and_a_start_without_its_key_are_rebuilt_whole():
    """A signature may come in more than one delta and a thinking block may start without the key: both are what
    the API sends, not damage, and the signature is replayed to the API, so it is joined byte for byte."""
    import io
    from compute_plane import native_stream
    events=anthropic_events()
    whole=events[SIGNATURE_AT]['delta']['signature']
    first=dict(events[SIGNATURE_AT],delta={'type':'signature_delta','signature':whole[:3]})
    second=dict(events[SIGNATURE_AT],delta={'type':'signature_delta','signature':whole[3:]})
    events=events[:SIGNATURE_AT]+[first,second]+events[SIGNATURE_AT+1:]
    for i,event in enumerate(events):
        if event.get('type')=='content_block_start' and event['content_block'].get('type')=='thinking':
            events[i]=dict(event,content_block={'type':'thinking','thinking':''})
    raw=''.join('event: %s\ndata: %s\n\n'%(e['type'],json.dumps(e)) for e in events).encode()
    answer=native_stream.messages(io.BytesIO(raw),lambda event:None)
    thinking=next(b for b in answer['content'] if b['type']=='thinking')
    assert thinking['signature']==whole


@pytest.mark.parametrize('damage,reason',[
    ('second_start','stream_message_start'),('no_start','stream_missing_start'),
    ('duplicate_index','stream_duplicate_block'),('gap_index','stream_block_index'),
    ('delta_unknown_block','stream_unknown_block'),('delta_closed_block','stream_unknown_block'),
    ('stop_unknown_block','stream_unknown_block'),('after_stop','stream_after_message_stop'),
    ('unclosed','stream_unclosed_blocks'),('cut_before_stop','stream_missing_completion'),
    ('no_stop_reason','stream_missing_completion'),('hosted_block','stream_unsupported_block'),
    ('unknown_delta','stream_unsupported_delta'),('wrong_delta_for_block','stream_block_type_changed'),
    ('text_not_empty','stream_block_start'),
    ('redacted_without_data','stream_block_start'),('delta_changes_content','stream_message_delta'),
    ('delta_not_text','stream_text_required'),('block_after_message_delta','stream_block_after_message_delta'),
    ('message_delta_before_block_closed','stream_unclosed_blocks')])
def test_anthropic_decoder_refuses_what_it_cannot_rebuild(damage,reason):
    events=anthropic_events()
    if damage=='second_start':events=twice(events,0)
    if damage=='no_start':events=events[1:]
    if damage=='duplicate_index':events=twice(events,TEXT_START)
    if damage=='gap_index':events[TEXT_START]=dict(events[TEXT_START],index=5)
    if damage=='delta_unknown_block':events[TEXT_DELTA]=dict(events[TEXT_DELTA],index=9)
    if damage=='delta_closed_block':events[TEXT_DELTA]=dict(events[TEXT_DELTA],index=0)
    if damage=='stop_unknown_block':events[TEXT_STOP]=dict(events[TEXT_STOP],index=9)
    if damage=='after_stop':events+=[{'type':'ping'}]
    if damage=='unclosed':events=drop(events,TEXT_STOP)
    if damage=='cut_before_stop':events=events[:MESSAGE_STOP]
    if damage=='no_stop_reason':events[MESSAGE_DELTA]=dict(events[MESSAGE_DELTA],delta={'stop_reason':None})
    if damage=='hosted_block':
        events[TEXT_START]=dict(events[TEXT_START],content_block={'type':'server_tool_use','id':'s','name':'web_search','input':{}})
    if damage=='unknown_delta':events[TEXT_DELTA]=dict(events[TEXT_DELTA],delta={'type':'citations_delta','citation':{}})
    if damage=='wrong_delta_for_block':
        events[TEXT_DELTA]=dict(events[TEXT_DELTA],delta={'type':'thinking_delta','thinking':'x'})
    if damage=='text_not_empty':events[TEXT_START]=dict(events[TEXT_START],content_block={'type':'text','text':'x'})
    if damage=='block_after_message_delta':  # the terminal delta moved before the last block: a block after the end
        events=events[:TEXT_START]+[events[MESSAGE_DELTA]]+events[TEXT_START:MESSAGE_DELTA]+events[MESSAGE_DELTA+1:]
    if damage=='message_delta_before_block_closed':
        events=events[:TEXT_STOP]+[events[MESSAGE_DELTA]]+events[TEXT_STOP:MESSAGE_DELTA]+events[MESSAGE_DELTA+1:]
    if damage=='redacted_without_data':events[REDACTED]=dict(events[REDACTED],content_block={'type':'redacted_thinking'})
    if damage=='delta_changes_content':events[MESSAGE_DELTA]=dict(events[MESSAGE_DELTA],delta={'stop_reason':'end_turn','content':[]})
    if damage=='delta_not_text':events[TEXT_DELTA]=dict(events[TEXT_DELTA],delta={'type':'text_delta','text':7})
    with pytest.raises(ValueError,match=reason):decode_events(events)


def test_anthropic_event_stream_without_the_terminal_blank_line_or_cut_mid_event_is_refused():
    raw=b''.join(anthropic_parts())
    # the last event has no blank line after it (its data is pending), or is cut before its data line (nothing pending)
    with pytest.raises(ValueError,match='stream_truncated_event'):messages(io.BytesIO(raw[:-2]),lambda e:None)
    with pytest.raises(ValueError,match='stream_truncated_event|stream_missing_completion'):
        messages(io.BytesIO(raw[:-30]),lambda e:None)
    whole_events=b''.join(anthropic_parts()[:-1])  # ends cleanly, one event short of message_stop
    with pytest.raises(ValueError,match='stream_missing_completion'):messages(io.BytesIO(whole_events),lambda e:None)
    # a complete answer followed by an event cut short (it may be the provider's error) is never a delivery
    with pytest.raises(ValueError,match='stream_truncated_event'):
        messages(io.BytesIO(raw+b'event: error\ndata: {"type":"error","error":'),lambda e:None)
    with pytest.raises(ValueError,match='stream_missing_done'):chat(io.BytesIO(b''.join(chat_parts()[:-1])),lambda e:None)
