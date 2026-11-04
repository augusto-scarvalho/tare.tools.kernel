"""EX-01 offline public-component evaluation. No production dependency or model calls."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import urllib.request

REVISION = '66101c7d5f810dc744b5be18b933af3441e00210'
EXPECTED_BEDROCK = 'cba9dee70757238558b836d8b0ab2baee990aa3f4f74bee0e30fb0ac13c2918f'


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')


def prepare(root):
    url = f'https://codeload.github.com/strands-agents/harness-sdk/tar.gz/{REVISION}'
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(32 * 1024 * 1024 + 1)
    assert len(raw) <= 32 * 1024 * 1024
    inventory = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts[1:]
            if not member.isfile() or not parts or '..' in parts:
                continue
            rel = '/'.join(parts)
            if not (rel.startswith('strands-py/src/strands/') and rel.endswith('.py') or rel in
                    {'strands-py/pyproject.toml', 'strands-py/LICENSE', 'strands-py/NOTICE', 'LICENSE', 'NOTICE'}):
                continue
            assert member.size < 1024 * 1024
            body = archive.extractfile(member).read()
            dest = root / 'upstream' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            inventory[rel] = {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    assert inventory['strands-py/src/strands/models/bedrock.py']['sha256'] == EXPECTED_BEDROCK
    save(root / 'upstream.json', {'revision': REVISION, 'url': url,
        'download_sha256': hashlib.sha256(raw).hexdigest(), 'files': inventory,
        'scope': 'selected Python source and licenses; full repository archive not retained'})
    print(json.dumps({'phase': 'prepared', 'files': len(inventory), 'source_bytes': sum(x['bytes'] for x in inventory.values())}))


def evaluate(root):
    import asyncio
    from copy import deepcopy
    from dataclasses import asdict
    import importlib.metadata as metadata
    import threading
    import time
    from types import SimpleNamespace

    kernel = Path(__file__).resolve().parents[1]
    source = root / 'upstream/strands-py/src'
    assert hashlib.sha256((source / 'strands/models/bedrock.py').read_bytes()).hexdigest() == EXPECTED_BEDROCK
    save(root / 'protocol.json', {'case': 'EX-01', 'revision': REVISION,
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'network': 'no model calls; injected boto_session fake, no credentials',
        'acceptance': 'same native read/edit/verify/proposal; public APIs only; explicit count provenance and profile; Runtime owns retry; bounded local cancellation',
        'scope': 'characterization with synthetic AWS-format replies; not provider, performance or full SDK qualification',
        'stop_rule': 'concrete incompatible contract closes component adoption evaluation; no private overrides'})
    sys.path[:0] = [str(source), str(kernel), str(kernel.parent / 'runtime/src')]
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    import strands.models.bedrock as upstream
    from strands.models import BedrockModel
    from compute_plane.native_models import ModelProfile, NativeModelAdapter
    from tare_tools_agent_runtime import (ConversationLog, NativeSession, EffectResult, ExecutionControl,
                                         BindingRef, CancellationToken)
    assert Path(upstream.__file__).resolve().is_relative_to(source)
    versions = {d.metadata['Name']: d.version for d in metadata.distributions()}
    save(root / 'environment.json', {'python': sys.version, 'imported_bedrock': upstream.__file__,
         'installed_distribution_versions': versions, 'strands_source_overrides_installed_distribution': True})
    fields = {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'}}
    usage = {'inputTokens': 10, 'outputTokens': 5, 'cacheReadInputTokens': 8, 'cacheWriteInputTokens': 0, 'totalTokens': 23}

    def response(name):
        args = {'summary': 'fixture verified'} if name == 'tare_deliver' else {'path': 'fixture.py'}
        return {'stopReason': 'tool_use', 'output': {'message': {'role': 'assistant', 'content': [
            {'reasoningContent': {'reasoningText': {'text': 'fixture reasoning', 'signature': 'opaque-signature'}}},
            {'toolUse': {'toolUseId': 'upstream-id', 'name': name, 'input': args}}]}},
            'usage': deepcopy(usage), 'metrics': {'latencyMs': 1}}

    class FakeClient:
        def __init__(self, replies=(), count=123):
            self.replies, self.count = iter(replies), count
            self.calls = []
            self.meta = SimpleNamespace(region_name='us-east-1')
        def converse(self, **request):
            self.calls.append(('converse', deepcopy(request)))
            reply = next(self.replies)
            if isinstance(reply, Exception):
                raise reply
            return deepcopy(reply)
        def count_tokens(self, **request):
            self.calls.append(('count', deepcopy(request)))
            if isinstance(self.count, Exception):
                raise self.count
            return {'inputTokens': self.count}

    class FakeSession(boto3.Session):
        def __init__(self, client):
            super().__init__(region_name='us-east-1', aws_access_key_id='fixture', aws_secret_access_key='fixture')
            self.injected = client
        def client(self, **kwargs):
            self.client_args = kwargs
            return self.injected

    def component(client, model='fixture-model', **extra):
        return BedrockModel(boto_session=FakeSession(client), model_id=model, max_tokens=4096,
            streaming=False, include_tool_result_status=True, additional_request_fields=deepcopy(fields),
            boto_client_config=Config(retries={'total_max_attempts': 1}), **extra)

    async def collect(model, wire, **kwargs):
        return [e async for e in model.stream(wire['messages'],
            tool_specs=[t['toolSpec'] for t in wire['toolConfig']['tools']],
            system_prompt_content=wire['system'], tool_choice=wire['toolConfig']['toolChoice'], **kwargs)]

    def assemble(events):
        # Only the text/reasoning/tool fixtures below. This is not a production stream codec.
        blocks, stop, counters, cursor = {}, None, None, 0
        for event in events:
            if 'contentBlockStart' in event:
                e = event['contentBlockStart']; blocks[e.get('contentBlockIndex', cursor)] = deepcopy(e['start'])
            if 'contentBlockDelta' in event:
                e = event['contentBlockDelta']; block = blocks.setdefault(e.get('contentBlockIndex', cursor), {})
                for kind, delta in e['delta'].items():
                    if kind == 'toolUse':
                        b = block.setdefault(kind, {}); b['input'] = b.get('input', '') + delta['input']
                    elif kind == 'reasoningContent':
                        b = block.setdefault(kind, {}).setdefault('reasoningText', {})
                        for key, value in delta.items(): b[key] = b.get(key, '') + value
                    elif kind == 'text': block[kind] = block.get(kind, '') + delta
                    else: raise AssertionError('fixture codec does not support ' + kind)
            if 'contentBlockStop' in event: cursor = event['contentBlockStop'].get('contentBlockIndex', cursor) + 1
            if 'messageStop' in event: stop = event['messageStop']['stopReason']
            if 'metadata' in event: counters = event['metadata']['usage']
        for block in blocks.values():
            if 'toolUse' in block: block['toolUse']['input'] = json.loads(block['toolUse']['input'])
        assert stop is not None
        return {'stopReason': stop, 'output': {'message': {'role': 'assistant', 'content': [blocks[i] for i in sorted(blocks)]}}, 'usage': counters}

    specs = {n: {'description': n, 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}
             for n in ('read', 'edit', 'verify')}
    profile = ModelProfile('bedrock-converse', 'fixture-model', 32768, 4096, 'EX-01 fixture', 'http://fixture',
        parameters_json=json.dumps({'additionalModelRequestFields': fields}), cache_system=True, cache_support_source='EX-01 fixture')
    outcomes, wires, stream_events = {}, {}, []
    for branch in ('direct', 'strands'):
        client = FakeClient([response(n) for n in ('read', 'edit', 'verify', 'tare_deliver')])
        model = component(client)
        class Bridge:
            endpoint = 'http://fixture'
            def post(self, path, body, control):
                assert path.endswith('/converse')
                if branch == 'direct': return client.converse(modelId=profile.model, **body)
                events = asyncio.run(collect(model, body)); stream_events.append(events)
                return assemble(events)
        class Capabilities:
            def __init__(self): self.calls = []
            def execute(self, request, control):
                self.calls.append(request.tool_call.name)
                return EffectResult(request.effect_id, 'succeeded', '{"fixture":true}', True)
        adapter = NativeModelAdapter(profile, Bridge(), system='fixed context', tool_specs_json=json.dumps(specs))
        tools, log = Capabilities(), ConversationLog()
        session = NativeSession(session_id='ex01', instruction='fixture task', capabilities=tuple(specs),
            model=adapter, tools=tools, journal=log)
        live = ExecutionControl(BindingRef('fixture','binding','test-owner',1,'fixture'), CancellationToken(), time.monotonic()+15)
        outcome = session.run(live)
        save(root / (branch + '-diagnostic.json'), {'outcome': asdict(outcome), 'effects': tools.calls,
             'client_calls': client.calls, 'events': [r.event() for r in log.snapshot().records], 'stream_events': stream_events})
        assert outcome.status == 'proposed' and tools.calls == ['read','edit','verify']
        assert len(client.calls) == 4
        outcomes[branch] = {'outcome': asdict(outcome), 'effects': tools.calls}
        wires[branch] = [wire for _, wire in client.calls]
        save(root / (branch + '-journal.json'), [r.event() for r in log.snapshot().records])
    assert wires['direct'] == wires['strands'], 'public component changed representative wire'
    save(root / 'functional.json', {'outcomes': outcomes, 'wires': wires, 'stream_events': stream_events})
    wire = wires['direct'][0]
    model = component(FakeClient())
    kwargs = {'messages': wire['messages'], 'tool_specs': [t['toolSpec'] for t in wire['toolConfig']['tools']], 'system_prompt_content': wire['system']}
    automatic = model.format_request(**kwargs, tool_choice={'auto': {}})
    forced = model.format_request(**kwargs, tool_choice={'any': {}})
    assert automatic['additionalModelRequestFields']['thinking'] == fields['thinking']
    assert 'thinking' not in forced.get('additionalModelRequestFields', {})
    assert model.get_config()['additional_request_fields']['thinking'] == fields['thinking']
    count_results = {}
    for name, value in [('native', 123), ('denied', ClientError({'Error': {'Code': 'AccessDeniedException', 'Message': 'fixture denied'}}, 'CountTokens'))]:
        client = FakeClient(count=value); model = component(client, model='fixture-'+name, use_native_token_count=True)
        results = [asyncio.run(model.count_tokens(**kwargs)) for _ in range(2)]
        assert all(type(v) is int for v in results)
        count_results[name] = {'returns': results, 'client_calls': client.calls}
    assert len(count_results['native']['client_calls']) == 2
    assert len(count_results['denied']['client_calls']) == 1
    throttled = FakeClient([ClientError({'Error': {'Code':'ThrottlingException','Message':'fixture throttled'}}, 'Converse')])
    try: asyncio.run(collect(component(throttled), wire))
    except Exception as error: throttle_type = type(error).__name__
    else: raise AssertionError('throttle swallowed')
    assert throttle_type == 'ModelThrottledException' and len(throttled.calls) == 1
    # This public provider contains an additional retry independent of SDK retry config.
    mixed = deepcopy(wire)
    mixed['messages'] = [{'role':'assistant','content':[{'toolUse':{'toolUseId':'old','name':'read','input':{'path':'fixture.py'}}}]},
        {'role':'user','content':[{'toolResult':{'toolUseId':'old','content':[{'text':'fixture result'}],'status':'success'}}]},
        {'role':'user','content':[{'text':'next instruction'}]}]
    validation = ClientError({'Error': {'Code':'ValidationException','Message':'Conversation blocks and tool result blocks cannot be provided in the same turn.'}}, 'Converse')
    client = FakeClient([validation, response('tare_deliver')])
    asyncio.run(collect(component(client), mixed))
    assert len(client.calls) == 2
    save(root / 'characterization.json', {'thinking': {'automatic': automatic, 'forced': forced},
        'counts': count_results, 'throttle': {'type': throttle_type, 'calls': len(throttled.calls)},
        'validation_repair_calls': client.calls})
    save(root / 'result.json', {'functional_equivalence': True, 'public_api_only': True,
        'decision': 'RETAIN_DIRECT_ADAPTER', 'component_adoption': 'REJECTED_FOR_THIS_CONTRACT',
        'reasons': ['count returns indistinguishable int for native and fallback',
                    'forced tool choice removes configured thinking from wire',
                    'validation repair performs second physical client call despite SDK total_max_attempts=1'],
        'not_tested': ['real AWS transport, SigV4 and billing', 'physical cancellation', 'upstream full suite', 'performance'],
        'scope': 'EX-01 functional substitution demonstrated; current complete provider rejected, not Strands as a product'})
    print('EX-01: functional equivalence PASS; complete provider adoption rejected for explicit recorded reasons')


if __name__ == '__main__':
    root = Path(os.environ['TARE_EXPERIMENT_RUN_DIR']).resolve()
    assert (root / '.experiment.json').is_file()
    assert sys.argv[1] in {'prepare', 'evaluate'}
    (prepare if sys.argv[1] == 'prepare' else evaluate)(root)
