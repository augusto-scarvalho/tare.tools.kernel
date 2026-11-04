"""Physical loopback/process checks. No provider inference or credentials needed."""
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Event, Thread
import time

import pytest

from tare_tools_agent_runtime import ConversationLog, ExecutionControl, NativeSession, NativeBudget, ModelFailure
from compute_plane.managed_process import run_pipe_process
from compute_plane.native_models import NativeHttpClient, NativeModelAdapter, NativeTransportInterrupted, MAX_WIRE_BYTES, DELIVERY_TOOL
from compute_plane.native_usage import native_usage_report
from tests.test_native_models import execution, profile, request, response, SPECS


@contextmanager
def endpoint(mode='stall'):
    received, entered, release = [], Event(), Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers['Content-Length']))))
            entered.set()
            try:
                if mode == 'stall':
                    release.wait(5)
                    return
                if mode == 'redirect':
                    self.send_response(307)
                    self.send_header('Location', '/redirect-target')
                    self.end_headers()
                    return
                if mode in ('retry', 'unavailable') and len(received) == 1:
                    self.send_response(429 if mode == 'retry' else 503)
                    self.end_headers()
                    return
                if mode in ('gpu_busy', 'other_503') and len(received) == 1:
                    # A gateway's text may echo anything: only Retry-After and a known busy type may leave.
                    kind = 'gpu_busy' if mode == 'gpu_busy' else 'qualified_model_gateway_error'
                    body = json.dumps({'error': {'type': kind, 'message': 'SECRET-sk-echoed', 'model': 'm'}}).encode()
                    self.send_response(503)
                    self.send_header('Retry-After', '42')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                if mode == 'drip':
                    self.send_header('Content-Length', '10000')
                    self.end_headers()
                    while not release.wait(.01):
                        self.wfile.write(b' ')
                        self.wfile.flush()
                    return
                body = (b'not-json' if mode == 'invalid-json' else b' ' * (MAX_WIRE_BYTES + 1) if mode == 'oversize'
                        else json.dumps(response('openai-chat', DELIVERY_TOOL)).encode())
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', received, entered, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


def real_control(seconds=10):
    old = execution()
    return ExecutionControl(old.binding, old.cancellation, time.monotonic() + seconds)


def is_running(pid):
    if Path('/proc').is_dir():
        try:
            return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] not in ('Z', 'X')
        except FileNotFoundError:
            return False
    return False  # Platform cleanup receipt is checked independently.


@pytest.mark.parametrize('mode', ['stall', 'drip'])
def test_cancel_closes_local_client_during_headers_or_body_without_claiming_remote_stop(mode):
    with endpoint(mode) as (url, seen, entered, release):
        control = real_control()
        def cancel():
            if entered.wait(3):
                control.cancellation.cancel()
        trigger = Thread(target=cancel)
        trigger.start()
        client, rows, log = NativeHttpClient(url), [], ConversationLog()
        adapter = NativeModelAdapter(replace(profile('openai-chat'), endpoint=url), client,
                                     system='system', tool_specs_json=SPECS, observe=rows.append)
        session = NativeSession(session_id='s', instruction='task', capabilities=('read', 'edit', 'verify'),
                                model=adapter, tools=None, journal=log)
        started = time.monotonic()
        outcome = session.run(control)
        trigger.join(3)
        assert time.monotonic() - started < 2
        assert outcome.status == 'cancelled' and outcome.pending_id == 's/model/1'
        physical = rows[0]['transport_calls'][0]
        assert physical['cleanup_verified'] and physical['cancelled'] and physical['started']
        assert physical['remote_stop'] == 'unknown' and not is_running(physical['pid'])
        assert rows[0]['usage'] is None and len(seen) == 1
        assert session.run(real_control()).status == 'reconciliation_required'
        assert len(seen) == 1  # Cancellation never blindly retries remote work.
        report = native_usage_report(log, rows, contracts={}, observation_failures=0)
        assert report['tokens']['total_tokens']['unknown_calls'] == 1
        assert report['transport']['local_cleanup_confirmed'] == 1
        assert report['transport']['remote_stop_unknown'] == 1


def test_deadline_stops_a_trickling_response_that_does_not_hit_socket_inactivity_timeout():
    with endpoint('drip') as (url, seen, entered, release):
        client = NativeHttpClient(url, timeout_seconds=.35)
        started = time.monotonic()
        with pytest.raises(NativeTransportInterrupted) as caught:
            client.post('/chat/completions', {}, real_control())
        obs = caught.value.transport_observation
        assert obs['timeout'] and obs['cleanup_verified']
        assert time.monotonic() - started < 2 and len(seen) == 1


def test_count_uses_same_physical_cancellation_and_preserves_unknown_observation():
    with endpoint() as (url, seen, entered, release):
        client, rows = NativeHttpClient(url, timeout_seconds=5), []
        control = real_control()
        def cancel():
            if entered.wait(5):
                control.cancellation.cancel()
        trigger = Thread(target=cancel)
        trigger.start()
        adapter = NativeModelAdapter(replace(profile('bedrock-converse'), endpoint=url), client,
                                     system='system', tool_specs_json=SPECS, observe=rows.append)
        try:
            result = adapter.count(request(), control)
        finally:
            release.set()
            trigger.join(5)
        assert result['status'] == 'unavailable' and result['input_tokens'] is None
        assert result['transport_calls'][0]['cleanup_verified']
        assert result['transport_calls'][0]['cancelled']
        assert result['transport_calls'][0]['remote_stop'] == 'unknown'
        assert len(seen) == 1 and seen[0][0].endswith('/count-tokens')


@pytest.mark.parametrize('mode,code', [('retry', 'HTTP_429'), ('unavailable', 'HTTP_503')])
def test_runtime_alone_retries_429_with_one_http_attempt_per_intent(mode, code, monkeypatch):
    # 503: a provider briefly unavailable ended a session in the OS phase 3 comparison; one retry, after a wait.
    from compute_plane import native_models
    monkeypatch.setattr(native_models, 'UNAVAILABLE_WAIT', 0)
    with endpoint(mode) as (url, seen, entered, release):
        rows, log = [], ConversationLog()
        adapter = NativeModelAdapter(replace(profile('openai-chat'), endpoint=url), NativeHttpClient(url),
                                     system='system', tool_specs_json=SPECS, observe=rows.append)
        session = NativeSession(session_id='s', instruction='task', capabilities=('read', 'edit', 'verify'),
                                model=adapter, tools=None, journal=log,
                                budget=NativeBudget(max_model_retries=1))
        assert session.run(real_control()).status == 'proposed'
        assert len(seen) == 2 and seen[0][0] == seen[1][0]
        first, retry = (json.loads(body) for _, body in seen)
        first_budget, retry_budget = first['messages'].pop(), retry['messages'].pop()
        assert first_budget['content'].startswith('Runtime budget: 16 model calls remain')
        assert retry_budget['content'].startswith('Runtime budget: 15 model calls remain')
        assert first == retry  # Only the cumulative budget changes between intents.
        assert rows[0]['error'] == code
        assert all(r['transport_calls'][0]['http_attempts_upper_bound'] == 1 for r in rows)
        assert all(r['transport_calls'][0]['cleanup_verified'] for r in rows)


@pytest.mark.parametrize('mode', ['redirect', 'oversize'])
def test_redirect_and_oversized_reply_are_contained_without_followup_requests(mode):
    with endpoint(mode) as (url, seen, entered, release):
        with pytest.raises(NativeTransportInterrupted) as caught:
            NativeHttpClient(url).post('/chat/completions', {}, real_control())
        assert caught.value.transport_observation['cleanup_verified']
        assert len(seen) == 1


def test_finite_pipe_input_cannot_block_deadline_when_worker_never_reads(tmp_path):
    started = time.monotonic()
    receipt, out, err = run_pipe_process([sys.executable, '-I', '-c', 'import time; time.sleep(60)'],
        cwd=tmp_path, timeout=.15, stdin=b'x' * (512 * 1024))
    assert receipt['timeout'] and receipt['cleanup_verified']
    assert time.monotonic() - started < 2 and not list(tmp_path.iterdir())
    assert not is_running(receipt['pid'])


def test_complete_invalid_json_is_failed_without_automatic_retry():
    with endpoint('invalid-json') as (url, seen, entered, release):
        with pytest.raises(ModelFailure, match='INVALID_HTTP_RESPONSE') as caught:
            NativeHttpClient(url).post('/chat/completions', {}, real_control())
        assert not caught.value.retryable
        assert caught.value.transport_observation['cleanup_verified']
        assert len(seen) == 1


def test_cancellation_during_credential_resolution_never_dispatches_or_invents_charge():
    control, log, rows = real_control(), ConversationLog(), []
    def credential():
        control.cancellation.cancel()
        return 'fixture-secret'
    adapter = NativeModelAdapter(profile('openai-chat'), NativeHttpClient('http://fixture', credential=credential),
                                 system='system', tool_specs_json=SPECS, observe=rows.append)
    session = NativeSession(session_id='s', instruction='task', capabilities=('read', 'edit', 'verify'),
                            model=adapter, tools=None, journal=log)
    outcome = session.run(control)
    assert outcome.status == 'cancelled' and outcome.pending_id is None
    report = native_usage_report(log, rows, contracts={}, observation_failures=0)
    assert report['coverage']['not_started'] == 1
    assert report['tokens']['total_tokens']['unknown_calls'] == 0
    assert report['transport']['http_attempts_upper_bound_for_observed_exchanges'] == 0
    assert 'fixture-secret' not in json.dumps(report)


def test_worker_is_selected_by_absolute_source_path_despite_foreign_cwd_and_pythonpath(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path / 'unrelated-install'))
    with endpoint('success') as (url, seen, entered, release):
        value, receipt = NativeHttpClient(url).post_observed('/chat/completions', {}, real_control())
        assert value['model'] == 'fixture-model' and receipt['cleanup_verified']
        assert not list(tmp_path.iterdir())


def test_output_limit_stops_a_worker_contract_violation(tmp_path):
    receipt, out, err = run_pipe_process([sys.executable, '-I', '-c', 'print("x" * 100000)'],
                                       cwd=tmp_path, timeout=2, max_output=1024)
    assert receipt['output_limit'] and receipt['cleanup_verified']
    assert len(out) <= 1025


def test_child_holding_output_pipe_is_cleaned_even_after_parent_exits_zero(tmp_path):
    code = ('import subprocess,sys; p=subprocess.Popen([sys.executable,"-I","-c",'
            '"import time; time.sleep(60)"]); print(p.pid,flush=True)')
    receipt, out, err = run_pipe_process([sys.executable, '-I', '-c', code], cwd=tmp_path, timeout=.3)
    assert receipt['timeout'] and receipt['cleanup_verified']
    assert not is_running(int(out.strip()))
    assert not list(tmp_path.iterdir())


def test_worker_dns_wait_is_inside_same_cancellable_containment(tmp_path):
    worker = Path(__file__).resolve().parents[1] / 'compute_plane/native_http_worker.py'
    code = ('import socket,time,runpy; socket.getaddrinfo=lambda *a,**k: time.sleep(60); '
            f'runpy.run_path({str(worker)!r},run_name="__main__")')
    source = json.dumps({'url': 'http://127.0.0.1:9', 'body': '{}', 'headers': {}, 'timeout': 10}).encode()
    receipt, out, err = run_pipe_process([sys.executable, '-I', '-c', code],
        cwd=tmp_path, timeout=.3, stdin=source)
    assert receipt['timeout'] and receipt['cleanup_verified'] and not out


def test_unknown_cleanup_refuses_even_an_apparently_complete_response(monkeypatch):
    from compute_plane import managed_process
    def uncertain(*args, **kwargs):
        return dict(started=True, cleanup_verified=False), b'{"kind":"response"}\n{}', b''
    monkeypatch.setattr(managed_process, 'run_pipe_process', uncertain)
    with pytest.raises(NativeTransportInterrupted, match='CLEANUP_UNCONFIRMED') as caught:
        NativeHttpClient('http://fixture').post('/chat/completions', {}, real_control())
    assert not caught.value.transport_observation['cleanup_verified']


def test_completed_response_racing_cancel_is_saved_but_never_executes_tool(monkeypatch):
    from compute_plane import managed_process
    control, rows, log = real_control(), [], ConversationLog()
    def completed(*args, **kwargs):
        control.cancellation.cancel()
        return dict(started=True, cleanup_verified=True, output_limit=False, cancelled=True, timeout=False), (
            b'{"kind":"response"}\n' + json.dumps(response('openai-chat', 'edit')).encode()), b''
    monkeypatch.setattr(managed_process, 'run_pipe_process', completed)
    adapter = NativeModelAdapter(profile('openai-chat'), NativeHttpClient('http://fixture'),
                                 system='system', tool_specs_json=SPECS, observe=rows.append)
    session = NativeSession(session_id='s', instruction='task', capabilities=('read', 'edit', 'verify'),
                            model=adapter, tools=None, journal=log)
    outcome = session.run(control)
    assert outcome.status == 'cancelled'
    assert any(r.event()['kind'] == 'model_returned' for r in log.snapshot().records)
    assert not any(r.event()['kind'] == 'effect_requested' for r in log.snapshot().records)
    assert rows[0]['transport_calls'][0]['response_received']


@pytest.mark.parametrize('stream', [False, True])
def test_unlimited_http_is_still_cancellable_and_has_no_socket_default(stream):
    with endpoint('stall') as (url, seen, entered, release):
        control = real_control(float('inf'))
        def cancel():
            assert entered.wait(3)
            control.cancellation.cancel()
        trigger = Thread(target=cancel); trigger.start()
        client = NativeHttpClient(url, timeout_seconds=None)
        with pytest.raises(NativeTransportInterrupted) as caught:
            client.post_observed('/chat/completions', {}, control,
                                 **({'stream_protocol': 'openai-chat'} if stream else {}))
        trigger.join(3)
        obs = caught.value.transport_observation
        assert obs['cancelled'] and obs['cleanup_verified'] and not obs['timeout']
        assert obs['timeouts'] == {'request_seconds': None, 'connect_seconds': None, 'read_seconds': None}
        assert len(seen) == 1 and obs['remote_stop'] == 'unknown'


def test_finite_enclosing_deadline_constrains_unlimited_request():
    with endpoint('drip') as (url, seen, entered, release):
        with pytest.raises(NativeTransportInterrupted) as caught:
            NativeHttpClient(url, timeout_seconds=None).post('/chat/completions', {}, real_control(.5))
        assert caught.value.transport_observation['timeout']
        assert caught.value.transport_observation['cleanup_verified']


@pytest.mark.parametrize('read_timeout', [.1, None])
def test_read_wait_independent_of_connection_and_total_timeout(read_timeout):
    # Headers deliberately arrive after the configured connection timeout.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            time.sleep(.35)
            try:
                self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers()
                self.wfile.write(b'{}')
            except (BrokenPipeError, ConnectionResetError): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    worker = Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        client = NativeHttpClient(f'http://127.0.0.1:{server.server_port}', timeout_seconds=None,
                                  connect_timeout_seconds=.1, read_timeout_seconds=read_timeout)
        if read_timeout is None:
            body, receipt = client.post_observed('/test', {}, real_control(float('inf')))
            assert body == {} and receipt['cleanup_verified']
            assert receipt['timeouts']['request_seconds'] is None
        else:
            with pytest.raises(NativeTransportInterrupted) as caught:
                client.post('/test', {}, real_control(float('inf')))
            assert caught.value.transport_observation['cleanup_verified']
    finally:
        server.shutdown(); server.server_close(); worker.join(2)


@pytest.mark.parametrize('value', [0, -1, True, float('nan'), float('inf')])
def test_optional_http_timeouts_reject_invalid_values(value):
    for field in ('timeout_seconds', 'connect_timeout_seconds', 'read_timeout_seconds'):
        with pytest.raises(ValueError): NativeHttpClient('http://localhost', **{field: value})


@pytest.mark.parametrize('mode', ['gpu_busy', 'other_503'])
def test_a_busy_local_gpu_costs_no_blind_wait_and_says_how_long(mode, monkeypatch):
    # Contract gpu-lease/1: the caller decides to wait or go elsewhere; 5 s asleep on every busy reply bought nothing.
    from compute_plane import native_models
    if mode == 'other_503':
        monkeypatch.setattr(native_models, 'UNAVAILABLE_WAIT', 0)
    with endpoint(mode) as (url, seen, entered, release):
        rows, log = [], ConversationLog()
        adapter = NativeModelAdapter(replace(profile('openai-chat'), endpoint=url), NativeHttpClient(url),
                                     system='system', tool_specs_json=SPECS, observe=rows.append)
        session = NativeSession(session_id='s', instruction='task', capabilities=('read', 'edit', 'verify'),
                                model=adapter, tools=None, journal=log, budget=NativeBudget(max_model_retries=1))
        started = time.monotonic()
        assert session.run(real_control()).status == 'proposed'
        receipt = rows[0]['transport_calls'][0]
        assert rows[0]['error'] == 'HTTP_503' and receipt['retry_after'] == 42
        assert receipt.get('busy_reason') == ('gpu_busy' if mode == 'gpu_busy' else None)
        assert 'SECRET' not in json.dumps(rows)
        if mode == 'gpu_busy':
            assert time.monotonic() - started < native_models.UNAVAILABLE_WAIT


def test_busy_hints_keep_a_long_wait_up_to_a_day():
    # A 12-hour training run on the node: the wait reads as 12 h, not an hour; anything longer stops at a day.
    from compute_plane.native_http_worker import busy_hints
    class Reply:
        code = 503
        def __init__(self, after):
            self.headers = {'Retry-After': str(after)}
        def read(self, size):
            return b'{"error": {"type": "gpu_busy"}}'
    assert busy_hints(Reply(43200)) == {'retry_after': 43200, 'busy_reason': 'gpu_busy'}
    assert busy_hints(Reply(10**7))['retry_after'] == 24 * 3600


def test_a_reserved_gpu_is_a_busy_reason_with_its_time_left():
    from compute_plane.native_http_worker import busy_hints
    class Reply:
        code, headers = 503, {'Retry-After': '5400'}
        def read(self, size):
            return b'{"error": {"type": "gpu_reserved", "message": "reserved for bench"}}'
    assert busy_hints(Reply()) == {'retry_after': 5400, 'busy_reason': 'gpu_reserved'}
