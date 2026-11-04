import base64
from dataclasses import replace
import io
import json
import os
import random
import tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from PIL import Image
import pytest

from tare_tools_agent_runtime import ConversationLog, ModelFailure, NativeSession
from compute_plane.native_attachments import AttachmentError, NativeAttachmentStore, inspect_content
from compute_plane.native_models import NativeHttpClient, NativeModelAdapter, DELIVERY_TOOL
from tests.test_native_models import Client, SPECS, execution, profile, request, response


def picture(format='PNG', color='red', size=(20,10)):
    output = io.BytesIO()
    Image.new('RGB', size, color).save(output, format=format)
    return output.getvalue()


def adapter(store, protocol='openai-chat', client=None, **options):
    return NativeModelAdapter(profile(protocol), client or Client([]), system='Stable',
        tool_specs_json=SPECS, attachment_loader=store.load,
        image_media_types=('image/png','image/jpeg','image/gif','image/webp'), **options)


def test_file_snapshot_survives_original_edit_deletion_and_store_reopening(tmp_path):
    source = tmp_path/'operator notes.md'
    source.write_bytes('First version\r\nAção 中文\n'.encode())
    store = NativeAttachmentStore(tmp_path/'assets')
    ref = store.import_file(source)
    original = store.load(ref)
    source.write_bytes(b'new version'); source.unlink()
    assert NativeAttachmentStore(store.root).load(ref) == original
    assert inspect_content(ref, original).text == original.decode()
    assert store.put_bytes(ref.name, original) == ref
    assert not list(store.root.glob('.attachment-*'))
    (store.root/ref.sha256).write_bytes(b'corrupted')
    with pytest.raises(AttachmentError, match='SNAPSHOT_CHANGED'):
        store.put_bytes(ref.name, original)
    assert (store.root/ref.sha256).read_bytes() == b'corrupted'


def test_snapshot_custody_and_import_support_deep_paths(tmp_path):
    store = NativeAttachmentStore(tmp_path/('nested-'*20)/('history-'*15))
    ref = store.put_bytes('notes.txt', b'deeply nested content')
    assert len(str(store.root/ref.sha256)) > 260
    assert NativeAttachmentStore(store.root).load(ref) == b'deeply nested content'
    copied = NativeAttachmentStore(tmp_path/'copy').import_file(store.root/ref.sha256, media_type='text/plain')
    assert copied.sha256 == ref.sha256 and copied.size_bytes == ref.size_bytes


@pytest.mark.parametrize('format,mime', [('PNG','image/png'),('JPEG','image/jpeg'),('GIF','image/gif'),('WEBP','image/webp')])
def test_supported_image_bytes_and_dimensions_are_verified(tmp_path, format, mime):
    store = NativeAttachmentStore(tmp_path/'assets')
    raw = picture(format)
    ref = store.put_bytes('screenshot', raw)
    assert ref.media_type == mime
    info = inspect_content(ref, store.load(ref))
    assert (info.width, info.height, info.text) == (20,10,None)
    with pytest.raises(AttachmentError):
        store.put_bytes('corrupt', raw[:len(raw)//2], mime)


@pytest.mark.parametrize('name,data', [('bad.txt',b'\xff'),('binary',b'a\x00b'),('report.pdf',b'%PDF-1.7'),('report.docx',b'PK\x03\x04')])
def test_unsupported_documents_are_not_silently_sent_as_plain_text(tmp_path, name, data):
    store = NativeAttachmentStore(tmp_path/'assets')
    with pytest.raises(AttachmentError) as failure:
        store.put_bytes(name, data)
    assert failure.value.not_started
    assert not store.root.exists()


def test_image_mismatch_animation_and_dimensions_refuse_before_storage(tmp_path):
    store = NativeAttachmentStore(tmp_path/'assets')
    with pytest.raises(AttachmentError, match='TYPE_MISMATCH'):
        store.put_bytes('wrong.jpg', picture(), 'image/jpeg')
    with pytest.raises(AttachmentError, match='DIMENSIONS_EXCEEDED'):
        store.put_bytes('wide.png', picture(size=(4001,4000)))
    output=io.BytesIO()
    Image.new('RGB',(2,2),'red').save(output,format='GIF',save_all=True,
        append_images=[Image.new('RGB',(2,2),'blue')],duration=100,loop=0)
    with pytest.raises(AttachmentError, match='ANIMATED'):
        store.put_bytes('animated.gif',output.getvalue())


@pytest.mark.skipif(os.name == 'nt', reason='POSIX FIFO and symlink behavior')
def test_special_and_linked_files_refuse_without_waiting_for_a_writer():
    # WSL's shared Windows volume cannot create FIFOs. Keep this tiny POSIX
    # fixture on the native filesystem and remove every special file on exit.
    with tempfile.TemporaryDirectory(prefix='tare-attachment-posix-', dir='/tmp') as directory:
        root = Path(directory)
        store = NativeAttachmentStore(root/'assets')
        source = root/'pipe'; os.mkfifo(source)
        link = root/'link'
        try:
            with pytest.raises(AttachmentError, match='REGULAR_FILE_REQUIRED'):
                store.import_file(source)
            ref = store.put_bytes('notes.txt', b'original')
            blob = store.root/ref.sha256; blob.unlink(); os.mkfifo(blob)
            try:
                with pytest.raises(AttachmentError, match='REGULAR_FILE_REQUIRED'):
                    store.load(ref)
            finally:
                blob.unlink()
            link.symlink_to(source)
            with pytest.raises(ValueError, match='link_forbidden'):
                store.import_file(link)
        finally:
            link.unlink(missing_ok=True); source.unlink()


@pytest.mark.parametrize('protocol', ['openai-chat','bedrock-converse'])
def test_initial_and_followup_content_reaches_wire_and_reopen_does_not_repeat(tmp_path, protocol):
    store = NativeAttachmentStore(tmp_path/'assets')
    first = store.put_bytes('first.png', picture())
    second = store.put_bytes('second.png', picture(color='blue'))
    doc = store.put_bytes('error.log', b'Error E312: unmatched separator')
    client = Client([response(protocol, DELIVERY_TOOL),response(protocol, DELIVERY_TOOL)])
    model = adapter(store, protocol, client)
    class Tools:
        def execute(self, *args):
            pytest.fail('Attachment input alone must not execute a tool')
    log = ConversationLog()
    options = dict(session_id='s', instruction='Compare the screenshots', capabilities=('read','edit','verify'),
                   model=model, tools=Tools(), journal=log, attachments=(first,doc))
    session = NativeSession(**options)
    assert session.run(execution()).status == 'proposed'
    session.continue_with(execution(), message_id='next', text='', attachments=(second,))
    completed = session.run(execution())
    assert completed.status == 'proposed' and len(client.calls) == 2
    encoded = json.dumps(client.calls[-1][1])
    assert base64.b64encode(store.load(first)).decode() in encoded
    assert base64.b64encode(store.load(second)).decode() in encoded
    assert 'Error E312: unmatched separator' in encoded
    assert base64.b64encode(store.load(first)).decode() not in json.dumps([r.event() for r in log.snapshot().records])
    assert NativeSession(**options).run(execution()) == completed
    assert len(client.calls) == 2


def test_image_input_requires_configuration_and_text_only_profile_stays_compatible(tmp_path):
    store = NativeAttachmentStore(tmp_path/'assets')
    ref = store.put_bytes('screen.png',picture())
    client = Client([])
    original = NativeModelAdapter(profile('openai-chat'),client,system='Stable',tool_specs_json=SPECS)
    enabled = NativeModelAdapter(profile('openai-chat'),client,system='Stable',tool_specs_json=SPECS,attachment_loader=store.load)
    assert original.profile_digest == enabled.profile_digest
    assert original.wire(request()) == enabled.wire(request())
    for model, code in ((original,'LOADER_REQUIRED'),(enabled,'IMAGE_INPUT_NOT_CONFIGURED')):
        with pytest.raises(AttachmentError, match=code):
            model.generate(replace(request(),attachments=(ref,)),execution())
    assert not client.calls


def test_count_uses_exact_image_bytes_and_changed_snapshot_never_dispatches(tmp_path):
    store = NativeAttachmentStore(tmp_path/'assets')
    ref = store.put_bytes('screen.png',picture())
    client = Client([{'inputTokens':350},response('bedrock-converse',DELIVERY_TOOL)])
    model = adapter(store,'bedrock-converse',client)
    req = replace(request(),attachments=(ref,))
    count = model.count(req,execution())
    assert count['input_tokens'] == 350 and count['status'] == 'observed'
    model.generate(replace(req,admitted_request_sha256=count['request_sha256']),execution())
    assert client.calls[0][1]['input']['converse']['messages'] == client.calls[1][1]['messages']
    image = next(part['image'] for part in client.calls[1][1]['messages'][0]['content'] if 'image' in part)
    assert image == {'format':'png','source':{'bytes':base64.b64encode(store.load(ref)).decode()}}
    (store.root/ref.sha256).write_bytes(b'changed')
    with pytest.raises(AttachmentError, match='SNAPSHOT_CHANGED') as failure:
        model.generate(replace(req,admitted_request_sha256=count['request_sha256']),execution())
    assert failure.value.not_started and len(client.calls)==2


def test_text_tokenizer_is_not_claimed_to_measure_vision_tokens(tmp_path):
    store = NativeAttachmentStore(tmp_path/'assets')
    ref = store.put_bytes('screen.png',picture())
    client = Client([])
    req = replace(request(),attachments=(ref,))
    for protocol in ('server-template-tokenize','slop-fleet-v1'):
        row = adapter(store,client=client,counter_protocol=protocol).count(req,execution())
        assert row['status']=='unavailable' and row['input_tokens'] is None
        assert row['reason']=='image_token_count_not_qualified'
    row = adapter(store,client=client,counter_protocol='utf8-envelope-estimate-v1').count(req,execution())
    assert row['status']=='estimated' and row['input_tokens'] is None
    assert 'image token usage is also unmeasured' in row['limitation']
    assert not client.calls


def test_replacing_an_admitted_attachment_refuses_before_provider_dispatch(tmp_path):
    store = NativeAttachmentStore(tmp_path/'assets')
    original = store.put_bytes('screen.png', picture())
    replacement = store.put_bytes('screen.png', picture(color='blue'))
    client = Client([{'inputTokens':350}])
    model = adapter(store, 'bedrock-converse', client)
    req = replace(request(), attachments=(original,))
    count = model.count(req, execution())
    with pytest.raises(ModelFailure, match='ADMITTED_REQUEST_CHANGED') as failure:
        model.generate(replace(req, attachments=(replacement,),
            admitted_request_sha256=count['request_sha256']), execution())
    assert failure.value.not_started and len(client.calls) == 1
    (store.root/original.sha256).unlink()
    with pytest.raises(AttachmentError, match='SNAPSHOT_UNAVAILABLE') as missing:
        model.generate(req, execution())
    assert missing.value.not_started and len(client.calls) == 1


def test_request_budget_rejects_before_loading_excess_content(tmp_path):
    from tare_tools_agent_runtime import AttachmentRef
    from tare_tools_agent_runtime.native_content import MAX_ATTACHMENT_BYTES
    store = NativeAttachmentStore(tmp_path/'assets')
    first = store.put_bytes('notes.txt', b'hello')
    excess = AttachmentRef('large.txt', 'text/plain', 'a'*64, MAX_ATTACHMENT_BYTES)
    loaded = []
    def load(ref):
        loaded.append(ref)
        return store.load(ref)
    client = Client([])
    model = NativeModelAdapter(profile('openai-chat'), client, system='Stable',
        tool_specs_json=SPECS, attachment_loader=load)
    with pytest.raises(AttachmentError, match='REQUEST_CONTENT_LIMIT_EXCEEDED'):
        model.generate(replace(request(), attachments=(first,excess)), execution())
    assert loaded == [first] and not client.calls


def test_oversized_source_is_refused_before_creating_snapshot(tmp_path):
    from tare_tools_agent_runtime.native_content import MAX_ATTACHMENT_BYTES
    source = tmp_path/'large.txt'
    with source.open('wb') as stream:
        stream.truncate(MAX_ATTACHMENT_BYTES+1)
    store = NativeAttachmentStore(tmp_path/'assets')
    with pytest.raises(AttachmentError, match='FILE_SIZE_EXCEEDED'):
        store.import_file(source)
    assert not store.root.exists()


@pytest.mark.parametrize('oversize', ['width', 'bytes'])
def test_bedrock_image_limits_refuse_before_http(tmp_path, oversize):
    if oversize == 'width':
        raw = picture(size=(8001,1))
    else:
        output = io.BytesIO()
        Image.frombytes('RGB',(1300,1000),random.Random(8).randbytes(3_900_000)).save(output,format='PNG')
        raw = output.getvalue()
        assert len(raw) > 3_750_000
    store = NativeAttachmentStore(tmp_path/'assets')
    ref = store.put_bytes('screen.png', raw)
    client = Client([])
    with pytest.raises(AttachmentError, match='BEDROCK_IMAGE_LIMIT_EXCEEDED') as failure:
        adapter(store, 'bedrock-converse', client).generate(
            replace(request(), attachments=(ref,)), execution())
    assert failure.value.not_started and not client.calls


def test_large_image_crosses_real_http_worker_only_with_explicit_request_limit(tmp_path):
    output=io.BytesIO()
    Image.frombytes('RGB',(1200,400),random.Random(7).randbytes(1200*400*3)).save(output,format='PNG')
    raw=output.getvalue(); assert len(raw)>1024*1024
    store=NativeAttachmentStore(tmp_path/'assets'); ref=store.put_bytes('large.png',raw)
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body=json.dumps(response('openai-chat',DELIVERY_TOOL)).encode()
            self.send_response(200); self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        endpoint='http://127.0.0.1:'+str(server.server_port)
        p=replace(profile('openai-chat'),endpoint=endpoint)
        options=dict(system='Stable',tool_specs_json=SPECS,attachment_loader=store.load,image_media_types=('image/png',))
        req=replace(request(),attachments=(ref,))
        small=NativeModelAdapter(p,NativeHttpClient(endpoint),**options)
        with pytest.raises(Exception) as failure:
            small.generate(req,execution())
        assert failure.value.not_started and not seen
        rows=[]
        large=NativeModelAdapter(p,NativeHttpClient(endpoint,request_byte_limit=4*1024*1024),observe=rows.append,**options)
        assert large.generate(req,execution()).proposes_delivery
        image=next(part['image_url']['url'] for part in seen[0]['messages'][1]['content'] if part.get('type')=='image_url')
        assert base64.b64decode(image.partition(',')[2]) == raw
        assert len(seen)==1 and rows[0]['transport_calls'][0]['cleanup_verified']
        assert base64.b64encode(raw).decode() not in json.dumps(rows)
    finally:
        server.shutdown();server.server_close();thread.join(2)
