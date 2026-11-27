"""Connection failures must be proven undispatched, never inferred from silence."""
from dataclasses import replace
import json
import socket
import sys

import pytest

from compute_plane import managed_process
from compute_plane.native_models import NativeHttpClient, NativeModelAdapter, NativeTransportInterrupted
from compute_plane.native_usage import native_usage_report
from tare_tools_agent_runtime import ConversationLog, ModelFailure, NativeBudget, NativeSession
from tests.test_native_http_cancellation import endpoint, real_control
from tests.test_native_models import profile, SPECS


def inject_worker(monkeypatch, code, *, once=False):
    original = managed_process.run_pipe_process
    calls = []
    def run(argv, **kwargs):
        calls.append(1)
        if not once or len(calls) == 1:
            argv = [sys.executable, '-I', '-c', code + '\nimport runpy\n'
                    + f'runpy.run_path({argv[-1]!r}, run_name="__main__")']
        return original(argv, **kwargs)
    monkeypatch.setattr(managed_process, 'run_pipe_process', run)
    return calls


DNS_FAILURE = """import socket
def fail(*a, **kw): raise socket.gaierror(-2, 'private-host-secret')
socket.getaddrinfo = fail
"""


@pytest.mark.parametrize('stream', [False, True])
def test_real_refused_connection_has_no_unresolved_remote_call(stream):
    # Reserve a port without accepting connections; no race with another listener.
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0))
        client = NativeHttpClient(f'http://127.0.0.1:{reserved.getsockname()[1]}', connect_timeout_seconds=5)
        with pytest.raises(ModelFailure, match='HTTP_CONNECTION_REFUSED') as caught:
            client.post_observed('/test', {}, real_control(),
                                 **({'stream_protocol': 'openai-chat'} if stream else {}))
    assert caught.value.not_started and caught.value.retryable
    obs = caught.value.transport_observation
    assert obs['started'] and obs['cleanup_verified']  # worker started, request did not
    assert obs['request_transmission'] == 'not_started'
    assert obs['remote_stop'] == 'not_dispatched' and obs['http_attempts_upper_bound'] == 0


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('retry', [False, True])
def test_dns_failure_can_resume_or_use_explicit_retry_budget(monkeypatch, stream, retry):
    from tests.test_native_streaming import serving
    context = serving(tool=True) if stream else endpoint('success')
    with context as (url, seen, *_):
        calls = inject_worker(monkeypatch, DNS_FAILURE, once=True)
        rows, log = [], ConversationLog()
        selected = replace(profile('openai-chat'), endpoint=url, model='fixture' if stream else 'fixture-model')
        adapter = NativeModelAdapter(selected, NativeHttpClient(url), system='system',
                                     tool_specs_json=SPECS, observe=rows.append, streaming=stream)
        session = NativeSession(session_id='s', instruction='task', capabilities=('read', 'edit', 'verify'),
                                model=adapter, tools=None, journal=log,
                                budget=NativeBudget(max_model_retries=int(retry)))
        outcome = session.run(real_control())
        if not retry:
            assert outcome.pending_id is None and outcome.reason == 'HTTP_DNS_FAILED'
            assert not seen and len(calls) == 1
            report = native_usage_report(log, rows, contracts={}, observation_failures=0)
            assert report['coverage']['not_started'] == 1
            assert report['tokens']['total_tokens']['unknown_calls'] == 0
            outcome = session.run(real_control())
        assert outcome.status == 'proposed' and outcome.pending_id is None
        assert len(calls) == 2 and len(seen) == 1
        assert rows[0]['status'] == 'not_started'
        assert rows[0]['transport_calls'][0]['connection_failure_reason'] == 'DNS_FAILED'
        assert 'private-host-secret' not in json.dumps(rows)


@pytest.mark.parametrize('kind,expected,retryable', [
    ('TimeoutError', 'CONNECT_TIMEOUT', True),
    ('ssl.SSLCertVerificationError', 'TLS_CERTIFICATE_FAILED', False),
    ('ssl.SSLError', 'TLS_FAILED', False),
])
def test_pre_send_diagnostics_are_fixed_codes(monkeypatch, kind, expected, retryable):
    inject_worker(monkeypatch, f'''import socket, ssl
def fail(*a, **kw): raise {kind}('secret endpoint and credential')
socket.create_connection = fail
''')
    with pytest.raises(ModelFailure) as caught:
        NativeHttpClient('https://fixture').post('/test', {}, real_control())
    assert caught.value.code == 'HTTP_' + expected
    assert caught.value.not_started and caught.value.retryable is retryable
    assert caught.value.transport_observation['transport_failure_phase'] == 'connect'
    assert caught.value.transport_observation['worker_error_cause'] == (
        'TIMEOUT' if expected == 'CONNECT_TIMEOUT' else expected)
    assert 'secret' not in json.dumps(caught.value.transport_observation)


@pytest.mark.parametrize('stream', [False, True])
def test_urlerror_after_request_send_is_still_uncertain(monkeypatch, stream):
    inject_worker(monkeypatch, '''import http.client, urllib.error
def fail(*a, **kw): raise urllib.error.URLError('private detail')
http.client.HTTPConnection.getresponse = fail
''')
    with endpoint('success') as (url, seen, entered, _):
        with pytest.raises(NativeTransportInterrupted) as caught:
            NativeHttpClient(url).post_observed('/test', {}, real_control(),
                **({'stream_protocol': 'openai-chat'} if stream else {}))
        assert entered.wait(2) and len(seen) == 1
        obs = caught.value.transport_observation
        assert obs['remote_stop'] == 'unknown' and obs['http_attempts_upper_bound'] == 1
        assert 'connection_failure_reason' not in obs
        assert obs['transport_failure_phase'] == 'response_headers'
        assert 'worker_error_cause' not in obs  # A text reason is not a typed cause.
        assert 'private detail' not in json.dumps(obs)


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('kind,expected', [
    ('ConnectionResetError', 'CONNECTION_RESET'), ('TimeoutError', 'TIMEOUT'),
    ('ssl.SSLError', 'TLS_FAILED'),
    ('http.client.RemoteDisconnected', 'REMOTE_DISCONNECTED'),
])
def test_post_send_typed_urlerror_records_cause_without_authorizing_retry(monkeypatch, stream, kind, expected):
    inject_worker(monkeypatch, f'''import http.client, urllib.error, ssl
def fail(*a, **kw): raise urllib.error.URLError({kind}('private endpoint credential'))
http.client.HTTPConnection.getresponse = fail
''')
    with endpoint('success') as (url, seen, entered, _):
        with pytest.raises(NativeTransportInterrupted) as caught:
            NativeHttpClient(url).post_observed('/test', {}, real_control(),
                **({'stream_protocol': 'openai-chat'} if stream else {}))
        assert entered.wait(2) and len(seen) == 1
    obs = caught.value.transport_observation
    assert obs['transport_failure_phase'] == 'response_headers'
    assert obs['worker_error_cause'] == expected
    assert obs['remote_stop'] == 'unknown' and obs['http_attempts_upper_bound'] == 1
    assert 'request_transmission' not in obs and 'private' not in json.dumps(obs)


@pytest.mark.parametrize('stream', [False, True])
def test_body_timeout_is_distinct_from_waiting_for_response_headers(monkeypatch, stream):
    inject_worker(monkeypatch, '''import http.client
def fail(*a, **kw): raise TimeoutError('private body detail')
http.client.HTTPResponse.read = fail
http.client.HTTPResponse.readline = fail
''')
    with endpoint('success') as (url, seen, entered, _):
        with pytest.raises(NativeTransportInterrupted) as caught:
            NativeHttpClient(url).post_observed('/test', {}, real_control(),
                **({'stream_protocol': 'openai-chat'} if stream else {}))
        assert entered.wait(2) and len(seen) == 1
    obs = caught.value.transport_observation
    assert obs['transport_failure_phase'] == 'response_body'
    assert obs['worker_error_cause'] == 'TIMEOUT'
    assert obs['remote_stop'] == 'unknown' and 'private' not in json.dumps(obs)


def test_send_failure_is_uncertain_even_without_received_output(monkeypatch):
    inject_worker(monkeypatch, '''import http.client
def fail(*a, **kw): raise BrokenPipeError('private send detail')
http.client.HTTPConnection.send = fail
''')
    with endpoint('success') as (url, *_):
        with pytest.raises(NativeTransportInterrupted) as caught:
            NativeHttpClient(url).post_stream_observed('/test', {}, real_control(), protocol='openai-chat')
    obs = caught.value.transport_observation
    assert obs['transport_failure_phase'] == 'request_send'
    assert obs['worker_error_cause'] == 'BROKEN_PIPE'
    assert obs['stream_events'] == 0 and obs['remote_stop'] == 'unknown'
    assert obs['http_attempts_upper_bound'] == 1 and 'private' not in json.dumps(obs)


@pytest.mark.parametrize('value', ['private endpoint credential', {'private': 'detail'}])
def test_parent_ignores_unrecognized_diagnostic_values(monkeypatch, value):
    def worker(*a, **kw):
        physical = dict(started=True, cleanup_verified=True, output_limit=False,
                        cancelled=False, timeout=False, exit_code=0)
        return physical, json.dumps({'kind': 'interrupted', 'phase': value,
                                     'cause': value}).encode() + b'\n', b''
    monkeypatch.setattr(managed_process, 'run_pipe_process', worker)
    with pytest.raises(NativeTransportInterrupted) as caught:
        NativeHttpClient('http://fixture').post_stream_observed('/test', {}, real_control(), protocol='openai-chat')
    obs = caught.value.transport_observation
    assert 'transport_failure_phase' not in obs and 'worker_error_cause' not in obs
    assert 'private' not in json.dumps(obs)


@pytest.mark.parametrize('fault', ['timeout', 'cancelled', 'exit_code', 'extra_frame', 'stderr'])
def test_incomplete_worker_attestation_cannot_authorize_retry(monkeypatch, fault):
    def worker(*a, **kw):
        physical = dict(started=True, cleanup_verified=True, output_limit=False,
                        cancelled=False, timeout=False, exit_code=0)
        raw = b'{"kind":"not_dispatched","reason":"DNS_FAILED"}\n'
        if fault in {'timeout', 'cancelled'}: physical[fault] = True
        if fault == 'exit_code': physical[fault] = 1
        if fault == 'extra_frame': raw = b'{"kind":"delta"}\n' + raw
        return physical, raw, b'error' if fault == 'stderr' else b''
    monkeypatch.setattr(managed_process, 'run_pipe_process', worker)
    with pytest.raises(NativeTransportInterrupted):
        NativeHttpClient('http://fixture').post_stream_observed('/test', {}, real_control(), protocol='openai-chat')


def test_happy_eyeballs_interleaves_address_families():
    from compute_plane.native_http_worker import _interleave_addrinfo
    infos = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2606:4700::1', 443, 0, 0)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2606:4700::2', 443, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('104.18.10.66', 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('104.18.11.66', 443)),
    ]
    ordered = _interleave_addrinfo(infos)
    assert [i[0] for i in ordered] == [socket.AF_INET6, socket.AF_INET, socket.AF_INET6, socket.AF_INET]
    assert [i[4][0] for i in ordered] == ['2606:4700::1', '104.18.10.66', '2606:4700::2', '104.18.11.66']


def test_happy_eyeballs_falls_back_fast_when_first_family_blackholes(monkeypatch):
    import time
    with endpoint('success') as (url, seen, *_):
        port = int(url.rsplit(':', 1)[1])
        # Simulate dual-stack DNS returning two blackholed IPv6 addresses before the live IPv4 address.
        inject_worker(monkeypatch, f'''import socket, time
_real_socket = socket.socket
class _BlackholeIPv6Socket(_real_socket):
    def connect(self, address):
        if self.family == socket.AF_INET6:
            time.sleep(30)
            raise TimeoutError('blackholed ipv6')
        return super().connect(address)
socket.socket = _BlackholeIPv6Socket
def _dual_stack_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2606:4700::1', {port}, 0, 0)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2606:4700::2', {port}, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', {port})),
    ]
socket.getaddrinfo = _dual_stack_getaddrinfo
''')
        t0 = time.monotonic()
        resp, receipt = NativeHttpClient(f'http://dual-stack.example:{port}', connect_timeout_seconds=10).post_observed(
            '/test', {}, real_control())
        elapsed = time.monotonic() - t0
        assert 'choices' in resp and receipt['response_received'] is True and len(seen) == 1
        assert elapsed < 3.0, f'Expected fast IPv4 fallback, took {elapsed:.2f}s'

