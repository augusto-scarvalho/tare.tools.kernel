"""Bounded M2 diagnostic, explicitly invoked inside an experiment-storage lot.

No production Work authority. Synthetic tools and criteria qualify transport,
not task utility. Credentials are read only in the Windows child that uses them.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

KERNEL = Path(__file__).resolve().parents[1]
LAB = KERNEL.parent
sys.path.insert(0, str(KERNEL))
import tare_tools_agent_runtime
from compute_plane.native_models import ModelProfile, NativeHttpClient, NativeModelAdapter, encode
from compute_plane.native_usage import UsageContract, native_usage_report
from compute_plane.managed_process import run_pipe_process
from tare_tools_agent_runtime import (BindingRef, CancellationToken, ConversationLog, ExecutionControl,
    EffectResult, NativeSession, NativeBudget, NativeContext, ContextPolicy)

GATEWAY = (os.environ.get('TARE_LOCAL_MODELS') or 'http://127.0.0.1:8080').rstrip('/')
COUNTER = (os.environ.get('TARE_LOCAL_TOKEN_COUNTER') or 'http://127.0.0.1:18082').rstrip('/')
BEDROCK = 'https://bedrock-runtime.us-east-1.amazonaws.com'
INITIAL = 'def clamp(value, lower, upper):\n    return value\n'
TASK = ('Implement clamp in fixture.py: return lower below the lower bound, upper above the upper bound, '
        'otherwise value unchanged. Equal bounds are valid. Raise ValueError if lower > upper. '
        'Use read, then edit with the returned sha256, then verify, then tare_deliver. '
        'Only fixture.py is editable. Verification must pass on the latest bytes. '
        'Keep the source to one function with no imports or external effects. This is a synthetic transport canary.')
SYSTEM = 'You implement the exact bounded task using the supplied native tools. One tool request per response. Preserve constraints.'
SPECS = encode({
    'read': {'description': 'Read fixture.py and its current SHA-256.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
    'edit': {'description': 'Replace only fixture.py using its current SHA-256.', 'parameters': {'type': 'object', 'properties': {
        'sha256': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['sha256', 'content'], 'additionalProperties': False}},
    'verify': {'description': 'Run independent behavioral checks on current fixture.py.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
})


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    raw = encode(value).encode()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('canary evidence bound exceeded')
    path.write_bytes(raw)


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read(1024 * 1024))


def prepare(root):
    if (root / 'protocol.json').exists():
        raise ValueError('protocol already frozen')
    sg = Path(os.environ['TARE_SPECGRAPH_ROOT']).expanduser().resolve()
    if not (sg / 'tools/pilot_profiles.py').is_file():
        raise ValueError('TARE_SPECGRAPH_ROOT must contain tools/pilot_profiles.py')
    gateway, props = get(GATEWAY + '/props'), get(COUNTER + '/props')
    assert gateway['current_model'] == 'qwen38' and gateway['backend_healthy']
    assert props['default_generation_settings']['n_ctx'] == 32768
    save(root / 'local-identity.json', {'gateway': gateway, 'backend': props})
    import runpy
    recipes = runpy.run_path(str(sg / 'tools/pilot_profiles.py'))
    params = recipes['request_parameters'](recipes['PROFILES']['qwen-low-2048'])
    params.update(params.pop('extra_body'))
    output = params.pop('max_tokens')
    params.pop('stream')
    qwen = ModelProfile('openai-chat', 'qwen38', 32768, output, 'local-identity.json: live props plus pilot_profiles.py qwen-low-2048',
                        GATEWAY + '/v1', parameters_json=encode(params),
                        template_sha256=digest(props['chat_template'].encode()))
    opus = ModelProfile('bedrock-converse', 'us.anthropic.claude-opus-4-6-v1', 1000000, 4096,
        'https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-6.html; configured, not stress tested',
        BEDROCK, parameters_json=encode({'additionalModelRequestFields': {
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'}}}),
        count_model='anthropic.claude-opus-4-6-v1', count_model_source='docs/HTTP_QUALIFICATION_20260909.md; explicit base model counter')
    sources = [Path(__file__).resolve(), *(KERNEL / 'compute_plane' / f for f in (
        'native_models.py', 'native_usage.py', 'native_http_worker.py', 'managed_process.py')),
        *(Path(tare_tools_agent_runtime.__file__).resolve().parent / f
          for f in ('native.py', 'native_context.py', 'conversation.py')),
        sg / 'tools/pilot_profiles.py']
    save(root / 'protocol.json', {
        'schema': 'native-m2-canary/1', 'prepared_at': time.time(), 'task': TASK,
        'scope': 'synthetic functional qualification, not paired utility or Work settlement',
        'criteria': ['read/edit/verify observed', 'six independent behavioral cases pass', 'latest source hash matches verification',
                     'delivery proposed', 'full request count available before dispatch', 'thinking parameters preserved'],
        'profiles': {'qwen': asdict(qwen), 'bedrock': asdict(opus)},
        'limits': {'calls_per_task': 6, 'wall_seconds_per_task': 240, 'retries': 0, 'max_compactions': 0,
                   'bedrock_max_input_per_call': 8192, 'bedrock_total_generation_calls': 8,
                   'bedrock_budget_usd': '3.00', 'budget_envelope_usd': '2.29376',
                   'budget_basis': '8 * (8192 * 10 + 4096 * 50) / 1M; conservative USD rates, not observed tariff or invoice',
                   'cache_control_calls_reserved': 2, 'disk_reservation_bytes': 16777216},
        'source_sha256': {str(p): digest(p.read_bytes()) for p in sources},
        'no_hidden_glm_recovery': True, 'no_reuse_as_blind_task': True})
    print(encode({'phase': 'prepared', 'backend_pid': gateway['backend_pid'], 'thinking': ['qwen low 2048', 'opus adaptive high']}), flush=True)


VERIFY = '''import sys,ast,json
text=open(sys.argv[1],encoding='utf-8').read()
tree=ast.parse(text)
assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name=='clamp'
assert not tree.body[0].decorator_list
assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Attribute,ast.Global,ast.Nonlocal)) for n in ast.walk(tree))
ns={'__builtins__':{'ValueError':ValueError}}
exec(compile(tree,sys.argv[1],'exec'),ns)
f=ns['clamp']
for args,expected in [((-3,0,5),0),((9,0,5),5),((2,0,5),2),((2,2,2),2),((-.5,-1.,0.),-.5)]:
 assert f(*args)==expected,(args,expected)
try: f(2,5,0)
except ValueError: pass
else: raise AssertionError('reversed bounds')
print(json.dumps({'passed':6}))
'''


class Tools:
    def __init__(self, dest):
        self.dest, self.path, self.verified = dest, dest / 'fixture.py', None
        self.path.write_text(INITIAL)
        self.actions = []

    def verify(self, control):
        receipt, out, err = run_pipe_process([sys.executable, '-I', '-c', VERIFY, str(self.path)],
            cwd=self.dest, timeout=min(5, control.remaining_seconds()), cancelled=control.should_stop, max_output=8192)
        passed = receipt['exit_code'] == 0 and receipt['cleanup_verified'] and not receipt['timeout']
        sha = digest(self.path.read_bytes())
        if passed:
            self.verified = sha
        return {'passed': passed, 'sha256': sha, 'process': receipt,
                'stdout': out.decode(errors='replace'), 'stderr': err.decode(errors='replace')}

    def execute(self, request, control):
        name, args = request.tool_call.name, json.loads(request.tool_call.arguments_json)
        self.actions.append(name)
        try:
            if control.should_stop():
                raise InterruptedError('control stopped')
            if name == 'read' and not args:
                value = {'content': self.path.read_text(), 'sha256': digest(self.path.read_bytes())}
            elif name == 'edit' and set(args) == {'sha256', 'content'}:
                assert args['sha256'] == digest(self.path.read_bytes()), 'source changed'
                assert isinstance(args['content'], str) and len(args['content'].encode()) <= 8192
                self.path.write_bytes(args['content'].encode())
                self.verified = None
                value = {'sha256': digest(self.path.read_bytes())}
            elif name == 'verify' and not args:
                value = self.verify(control)
            else:
                raise ValueError('invalid capability arguments')
            return EffectResult(request.effect_id, 'succeeded', encode(value), True)
        except Exception as exc:
            return EffectResult(request.effect_id, 'refused', encode({'error': type(exc).__name__}), False)


def run(root, provider):
    started = time.monotonic()
    protocol = json.loads((root / 'protocol.json').read_text())
    for path, sha in protocol['source_sha256'].items():
        assert digest(Path(path).read_bytes()) == sha, 'source changed since protocol'
    dest = root / provider
    dest.mkdir()  # Never overwrite or silently rerun a previous attempt.
    p = ModelProfile(**protocol['profiles'][provider])
    if provider == 'qwen':
        identity = json.loads((root / 'local-identity.json').read_text())
        gateway, props = get(GATEWAY + '/props'), get(COUNTER + '/props')
        assert gateway['backend_pid'] == identity['gateway']['backend_pid']
        assert digest(props['chat_template'].encode()) == p.template_sha256
        assert props['default_generation_settings']['n_ctx'] == p.context_window
        credential = lambda: None
    else:
        import keyring
        key = keyring.get_password('universal-agent-harness', 'AWS_BEARER_TOKEN_BEDROCK')
        assert key, 'credential unavailable'
        credential = lambda: key
    rows, wires, log = [], [], ConversationLog()
    def observe(row):
        rows.append(row)
        save(dest / 'observations.json', {'rows': rows})
        print(encode({'provider': provider, 'purpose': row['purpose'], 'status': row['status'],
                      'error': row.get('error') or row.get('reason'), 'usage': row.get('usage')}), flush=True)
    class Observed(NativeModelAdapter):
        def wire(self, request):
            value = super().wire(request)
            wires.append({'call_id': request.call_id, 'purpose': request.purpose, 'wire': value})
            save(dest / 'requests.json', {'requests': wires})
            return value
    adapter = Observed(p, NativeHttpClient(p.endpoint, credential=credential, timeout_seconds=90),
        system=SYSTEM, tool_specs_json=SPECS, observe=observe,
        local_counter=NativeHttpClient(COUNTER, timeout_seconds=15) if provider == 'qwen' else None)
    control = ExecutionControl(BindingRef('synthetic-canary', provider, 'diagnostic-no-authority', 1, 'fixture'),
                               CancellationToken(), started + 240)
    tools = Tools(dest)
    baseline = tools.verify(control)
    save(dest / 'baseline.json', baseline)
    assert not baseline['passed'], 'oracle must reject original defect'
    def count(request, live):
        row = adapter.count(request, live)
        if provider == 'bedrock' and row.get('input_tokens') is not None and row['input_tokens'] > 8192:
            raise ValueError('paid canary input cap exceeded')
        return row
    session = NativeSession(session_id='native-canary-' + provider, instruction=TASK,
        capabilities=('read', 'edit', 'verify'), model=adapter, tools=tools, journal=log,
        budget=NativeBudget(max_model_calls=6, max_model_retries=0),
        context=NativeContext(ContextPolicy(p.context_window, p.output_tokens, 128,
            trigger=.8 if provider == 'qwen' else .5, target=.5 if provider == 'qwen' else .25, max_compactions=0),
            counter=count, summarizer=adapter))
    outcome = session.run(control)
    verified_by_model = tools.verified
    # Independent final verifier has its own bounded local deadline, even on host failure.
    independent = tools.verify(ExecutionControl(control.binding, CancellationToken(), time.monotonic() + 10))
    passed = (outcome.status == 'proposed' and independent['passed'] and verified_by_model == independent['sha256']
              and all(n in tools.actions for n in ('read', 'edit', 'verify')))
    save(dest / 'journal.json', {'events': [r.event() for r in log.snapshot().records]})
    save(dest / 'usage-report.json', native_usage_report(log, rows, contracts={}, observation_failures=adapter.observation_failures))
    result = {'provider': provider, 'outcome': asdict(outcome), 'independent': independent, 'actions': tools.actions,
              'passed': passed, 'elapsed_from_run_start_ms': (time.monotonic() - started) * 1000,
              'preparation_time_excluded': True, 'usage_contract_pending_live_review': True}
    save(dest / 'result.json', result)
    print(encode({'provider': provider, 'passed': passed, 'outcome': asdict(outcome)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'qwen', 'bedrock'])
    args = parser.parse_args()
    root = Path(os.environ['TARE_EXPERIMENT_RUN_DIR']).resolve()
    assert (root / '.experiment.json').is_file(), 'registered experiment lot required'
    prepare(root) if args.mode == 'prepare' else run(root, args.mode)
