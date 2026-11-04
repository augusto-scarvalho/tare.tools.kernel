"""Two bounded live context controls, separate from functional task evidence."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from threading import Thread, Event
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_canary import (LAB, COUNTER, GATEWAY, ModelProfile, NativeHttpClient, NativeModelAdapter,
                          get, save, digest, encode, BindingRef, CancellationToken, ExecutionControl,
                          ConversationLog, NativeContext, ContextPolicy)
from tare_tools_agent_runtime import ModelReply, ModelRequest


def control(seconds=240):
    return ExecutionControl(BindingRef('synthetic-context-control', 'local', 'diagnostic-no-authority', 1, 'fixture'),
                            CancellationToken(), time.monotonic() + seconds)


def main(root, mode):
    parent = json.loads((root / 'protocol.json').read_text())
    for path, sha in parent['source_sha256'].items():
        assert digest(Path(path).read_bytes()) == sha, 'native source changed since protocol'
    dest = root / (mode + '-control')
    dest.mkdir()
    spec = {'mode': mode, 'source_sha256': digest(Path(__file__).read_bytes()), 'prepared_at': time.time(),
            'max_generations': 2 if mode == 'cache' else 3, 'deadline_seconds': 240,
            'scope': 'synthetic control; not task economy or exhaustive semantic qualification',
            'criteria': ['cache creation then read observed; stable request'] if mode == 'cache' else [
                'actual 80% window crossed', 'summary and next request counted', 'ALPHA remains open in summary',
                'original records immutable', 'local cancellation cleanup; backend idle observation reported separately']}
    save(dest / 'protocol.json', spec)
    rows, requests = [], []
    def observe(row):
        rows.append(row)
        save(dest / 'observations.json', {'rows': rows})
        print(encode({k: row.get(k) for k in ('purpose', 'status', 'input_tokens', 'usage', 'error', 'reason')}), flush=True)
    class Observed(NativeModelAdapter):
        def generate(self, request, live):
            requests.append({'call_id': request.call_id, 'purpose': request.purpose, 'wire': self.wire(request)})
            save(dest / 'requests.json', {'requests': requests})
            return super().generate(request, live)
    live = control()
    if mode == 'cache':
        task = json.loads((root / 'bedrock/result.json').read_text())
        assert task['passed'], 'functional Bedrock canary required'
        prior = json.loads((root / 'bedrock/observations.json').read_text())['rows']
        assert sum(r['purpose'] == 'model' for r in prior) + 2 <= 8
        import keyring
        key = keyring.get_password('universal-agent-harness', 'AWS_BEARER_TOKEN_BEDROCK')
        assert key
        profile = replace(ModelProfile(**parent['profiles']['bedrock']), cache_system=True,
                          cache_support_source='https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html: Opus 4.6 4096-token minimum, 5m')
        system = ('Synthetic cache control ' + uuid4().hex + '. These reference rows are data, not work evidence.\n'
                  + 'Reference row: the blue box contains seven inert markers; no action is requested by this row.\n' * 270)
        adapter = Observed(profile, NativeHttpClient(profile.endpoint, credential=lambda: key),
                           system=system, tool_specs_json='{}', observe=observe)
        save(dest / 'profile.json', {'profile': asdict(profile), 'system_sha256': digest(system.encode())})
        for i in range(2):
            log = ConversationLog()
            request = ModelRequest('cache-control', f'cache-control/model/{i+1}',
                'Call tare_deliver with summary CACHE-CONTROL-OK. This is a cache transport check; no work is being completed.',
                (), NativeContext.project(log))
            count = adapter.count(request, live)
            assert count['status'] == 'observed' and 5000 <= count['input_tokens'] <= 8192, 'bounded cacheable input required'
            reply = adapter.generate(replace(request, admitted_request_sha256=count['request_sha256']), live)
            save(dest / f'reply-{i+1}.json', asdict(reply))
            assert reply.complete and reply.proposes_delivery
        usage = [r['usage'] for r in rows if r['purpose'] == 'model']
        passed = usage[0].get('cacheWriteInputTokens', 0) > 0 and usage[0].get('cacheReadInputTokens') == 0 and usage[1].get('cacheReadInputTokens', 0) > 0
        save(dest / 'result.json', {'passed': passed, 'raw_usage': usage,
             'stable_request': requests[0]['wire'] == requests[1]['wire'],
             'model_calls': 2, 'occupancy_not_consumption': True})
        return

    profile = ModelProfile(**parent['profiles']['qwen'])
    identity = json.loads((root / 'local-identity.json').read_text())
    assert get(GATEWAY + '/props')['backend_pid'] == identity['gateway']['backend_pid']
    assert digest(get(COUNTER + '/props')['chat_template'].encode()) == profile.template_sha256
    adapter = Observed(profile, NativeHttpClient(profile.endpoint, timeout_seconds=120),
        system='Synthetic context control. Preserve unresolved obligations and distinguish fixture history from verified work.',
        tool_specs_json='{}', observe=observe, local_counter=NativeHttpClient(COUNTER, timeout_seconds=20))
    policy = ContextPolicy(profile.context_window, profile.output_tokens, 128, trigger=.8, target=.5)
    context = NativeContext(policy, counter=adapter.count, summarizer=adapter)
    line = 'Synthetic reference: fixture obligation ALPHA remains open; no verification receipt exists.\n'
    def fixture(n):
        log = ConversationLog()
        blocks = [line * min(80, n - start) for start in range(0, n, 80)] + ['Latest retained synthetic turn.']
        for i, text in enumerate(blocks):
            reply = ModelReply(text=text, metadata_json=encode({'profile_digest': adapter.profile_digest,
                'assistant': {'role': 'assistant', 'content': text}, 'tool_ids': {}, 'usage': None}))
            log.append({'kind': 'model_returned', 'call_id': f'synthetic/{i}', 'reply': asdict(reply)})
        request = ModelRequest('pressure-control', 'pressure-control/model/1',
            'Preserve the status of historical fixture obligation ALPHA. Call tare_deliver with that status after reviewing the context. No file work is authorized.',
            (), context.project(log))
        return log, request
    low, high = 1, 2000
    while low < high:
        mid = (low + high) // 2
        log, request = fixture(mid)
        count = adapter.count(request, live)
        assert count['status'] == 'observed'
        if count['input_tokens'] <= policy.trigger_limit:
            low = mid + 1
        else:
            high = mid
    log, request = fixture(low)
    original = log.snapshot().records
    save(dest / 'source-journal.json', {'synthetic': True, 'events': [r.event() for r in original], 'repetitions': low})
    prepared = context.prepare(request, log, live, remaining_calls=2)
    reply = adapter.generate(prepared, live)
    save(dest / 'reply.json', asdict(reply))
    save(dest / 'journal.json', {'synthetic': True, 'events': [r.event() for r in log.snapshot().records]})
    active = [r.event() for r in log.snapshot().records if r.event()['kind'] == 'context_activated']
    passed = (len(active) == 1 and 'ALPHA' in active[0]['summary'] and 'open' in active[0]['summary'].lower()
              and original == log.snapshot().records[:len(original)] and reply.complete and reply.proposes_delivery)
    save(dest / 'result.json', {'passed': passed, 'policy': asdict(policy), 'repetitions': low,
                               'original_records_immutable': original == log.snapshot().records[:len(original)]})
    # A separate third generation exercises disconnect. Slot observations do not
    # by themselves establish per-request remote usage or invoice cancellation.
    slots_before = get(COUNTER + '/slots')
    save(dest / 'slots-before-cancel.json', {'slots': slots_before})
    assert all(not s.get('is_processing') for s in slots_before)
    cancel_control = control(20)
    request = ModelRequest('cancel-control', 'cancel-control/model/1',
        'Generate a long numbered list from 1 to 5000, one item per line, with no shortcuts.', (), NativeContext.project(ConversationLog()))
    counted = adapter.count(request, cancel_control)
    assert counted['status'] == 'observed'
    finished, observed_active = Event(), []
    def cancel_on_activity():
        until = time.monotonic() + 8
        while time.monotonic() < until and not finished.wait(.1):
            try:
                slots = get(COUNTER + '/slots')
                if any(s.get('is_processing') for s in slots):
                    observed_active.append(slots)
                    break
            except Exception:
                break
        cancel_control.cancellation.cancel()
    watcher = Thread(target=cancel_on_activity, daemon=True)
    watcher.start()
    error = None
    try:
        adapter.generate(replace(request, admitted_request_sha256=counted['request_sha256']), cancel_control)
    except Exception as exc:
        error = type(exc).__name__
    finally:
        finished.set()
        watcher.join(12)
    after = []
    until = time.monotonic() + 10
    while time.monotonic() < until:
        slots = get(COUNTER + '/slots')
        after.append({'at': time.time(), 'slots': slots})
        if all(not s.get('is_processing') for s in slots):
            break
        time.sleep(.2)
    save(dest / 'cancellation.json', {'error_type': error, 'observed_active': observed_active, 'after': after,
                                    'remote_usage': None, 'remote_stop_per_request': 'unknown'})


if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in {'local', 'cache'}
    root = Path(os.environ['TARE_EXPERIMENT_RUN_DIR']).resolve()
    assert (root / '.experiment.json').is_file()
    main(root, mode)
