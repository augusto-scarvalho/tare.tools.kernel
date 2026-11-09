"""One cancellable HTTP exchange. Launched by absolute path with isolated Python.

    stdin: bounded JSON containing URL, serialized body, headers and timeout.
    stdout: one JSON status line then bounded raw response bytes.

No model/session logic, imports from installed packages, retry or durable files.
Parent containment owns cancellation, including waits inside DNS/TLS libraries.
"""
import http.client
import json
import re
import socket
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

MAX_BODY = 1024 * 1024
MAX_REQUEST = 32 * MAX_BODY
MAX_INPUT = 6 * MAX_REQUEST + 65536  # Escaped request plus bounded headers.

# Fixed diagnostics only: never echo exception text, response data or credentials.
STREAM_REASONS = frozenset({
    'stream_missing_done', 'stream_missing_completion', 'stream_byte_limit',
    'stream_event_limit', 'stream_provider_exception', 'stream_truncated_frame',
    'stream_object_required', 'stream_text_required', 'stream_opaque_metadata_changed',
    'stream_model_changed', 'stream_duplicate_fleet_receipt', 'stream_usage_changed',
    'stream_one_choice_required', 'stream_choice_order', 'stream_role',
    'stream_legacy_function_unsupported', 'stream_tool_index', 'stream_tool_type',
    'stream_reasoning_details_shape',
    'responses_item_shape', 'responses_item_identity', 'responses_duplicate_item_id',
    'responses_duplicate_item_index', 'responses_item_changed', 'responses_unfinished_item',
    'responses_unfinished_function', 'responses_missing_terminal', 'responses_response_identity',
    'responses_delta_text', 'responses_terminal_status', 'responses_unclosed_items',
    'responses_missing_output_index', 'responses_output_shape', 'responses_final_snapshot_changed',
    'responses_output_kind', 'responses_provider_error',
})
# A local GPU's gateway refusing text because the GPU is taken (contract gpu-lease/1, docs/GPU_LEASE_CONTRACT.md).
GPU_BUSY_REASONS = frozenset({'gpu_busy', 'gpu_lease_timeout', 'gpu_lease_cancelled', 'gpu_reserved'})
MAX_RETRY_AFTER = 24 * 3600  # a 12-hour training run on the node says 12 h
MAX_PROVIDER_MESSAGE = 300
# The only request headers whose values a provider's message may keep; account, session and any future header go.
PLAIN_HEADERS = frozenset({'content-type', 'accept', 'content-length', 'user-agent'})
# A key, a bearer token or a masked key a provider echoes (e.g. "Incorrect API key provided: sk-ab***yz").
SECRET_SHAPES = re.compile(r'(?i)bearer\s+\S+|\bsk-[\w.*-]+|[A-Za-z0-9+/_=-]{32,}')


def busy_hints(exc, body=None):
    """What an error reply says about waiting, without its text: Retry-After as whole seconds and, for a 503, the
    gateway's error type when it is one of GPU_BUSY_REASONS. Nothing else of a 429 or 5xx reply leaves this process."""
    hints = {}
    after = (exc.headers.get('Retry-After') or '').strip() if exc.headers else ''
    if after.isdigit():
        hints['retry_after'] = min(int(after), MAX_RETRY_AFTER)
    if exc.code == 503:
        try:
            error = json.loads(exc.read(4096) if body is None else body).get('error')
        except (ValueError, OSError, AttributeError):
            error = None
        if isinstance(error, dict) and error.get('type') in GPU_BUSY_REASONS:
            hints['busy_reason'] = error['type']
            seconds = error.get('retry_after_seconds')
            if 'retry_after' not in hints and type(seconds) is int and seconds >= 0:
                hints['retry_after'] = min(seconds, MAX_RETRY_AFTER)
    return hints

def provider_error(status, body, headers):
    """A refused request's own reason: the provider's error type and message, so an HTTP 400 can be diagnosed. Every
    request header value (but the plain content ones) and anything shaped like a key is removed from the message
    first. A 429 keeps only its error type, a single word, which tells a spent plan (OpenCode's GoUsageLimitError)
    from a passing rate limit; a gateway's 429 or 5xx text may echo anything and stays out."""
    if not 400 <= status < 500:
        return {}
    try:
        error = json.loads(body).get('error')
    except (ValueError, AttributeError):
        return {}
    if not isinstance(error, dict):
        return {}
    found = {}
    kind = error.get('type') or error.get('code')
    if isinstance(kind, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', kind) and not SECRET_SHAPES.search(kind):
        found['type'] = kind
    if status == 429:
        return {'provider_error': found} if found else {}
    message = error.get('message')
    if isinstance(message, str) and message.strip():
        for name, value in headers.items():
            if isinstance(value, str) and name.lower() not in PLAIN_HEADERS:
                for secret in (value, value.split()[-1] if value.split() else value):
                    if len(secret) >= 8:
                        message = message.replace(secret, '[redacted]')
        found['message'] = SECRET_SHAPES.sub('[redacted]', message)[:MAX_PROVIDER_MESSAGE]
    return {'provider_error': found} if found else {}


CONNECT_REASONS = frozenset({'DNS_FAILED', 'CONNECTION_REFUSED', 'CONNECT_TIMEOUT',
                             'TLS_CERTIFICATE_FAILED', 'TLS_FAILED', 'CONNECTION_FAILED'})

TRANSPORT_PHASES = frozenset({'prepare', 'connect', 'request_send',
                              'response_headers', 'response_body'})
ERROR_CAUSES = (
    (socket.gaierror, 'DNS_FAILED'),
    (ssl.SSLCertVerificationError, 'TLS_CERTIFICATE_FAILED'),
    (ssl.SSLError, 'TLS_FAILED'),
    (http.client.RemoteDisconnected, 'REMOTE_DISCONNECTED'),
    (http.client.IncompleteRead, 'INCOMPLETE_READ'),
    (ConnectionRefusedError, 'CONNECTION_REFUSED'),
    (ConnectionResetError, 'CONNECTION_RESET'),
    (ConnectionAbortedError, 'CONNECTION_ABORTED'),
    (BrokenPipeError, 'BROKEN_PIPE'),
    (TimeoutError, 'TIMEOUT'),
    (OSError, 'OS_ERROR'),
)
ERROR_CAUSE_CODES = frozenset(code for _, code in ERROR_CAUSES)


def error_cause(exc):
    """Only typed causes: URLError.reason may instead contain private text."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return next((code for kind, code in ERROR_CAUSES if isinstance(reason, kind)), None)


def connection_failure(exc):
    """Classify by exception type, never by text containing URLs or credentials."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, socket.gaierror):
        return 'DNS_FAILED'
    if isinstance(reason, ssl.SSLCertVerificationError):
        return 'TLS_CERTIFICATE_FAILED'
    if isinstance(reason, ssl.SSLError):
        return 'TLS_FAILED'
    if isinstance(reason, ConnectionRefusedError):
        return 'CONNECTION_REFUSED'
    if isinstance(reason, TimeoutError):
        return 'CONNECT_TIMEOUT'
    if isinstance(reason, OSError):
        return 'CONNECTION_FAILED'
    return None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('redirect_not_admitted')


def main():
    connect_attempted = False
    transmission_started = False
    phase = 'prepare'
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT + 1)
        if len(raw) > MAX_INPUT:
            raise ValueError('input_limit')
        request = json.loads(raw)
        request_limit = request.get('request_byte_limit', MAX_BODY)
        if type(request_limit) is not int or not MAX_BODY <= request_limit <= MAX_REQUEST:
            raise ValueError('invalid_request_limit')
        payload = request['body'].encode('utf-8')
        if len(payload) > request_limit:
            raise ValueError('body_limit')
        method = request.get('method', 'POST')
        if method not in {'POST', 'GET'} or method == 'GET' and (payload or request.get('stream_protocol')):
            raise ValueError('invalid_method')
        # Public stdlib connection/handler extension points. Connection/TLS and
        # socket read waits have independent policy; the parent owns total time,
        # cancellation and cleanup (including a blocked resolver).
        read_timeout = request.get('read_timeout', request['timeout'])
        class TransmissionBoundary:
            def send(self, data):
                nonlocal transmission_started, phase
                # Connect/TLS must finish before any application bytes can leave.
                # Mark before sendall: partial writes and later errors are uncertain.
                if self.sock is None:
                    self.connect()
                transmission_started = True
                phase = 'request_send'
                return super().send(data)
            def getresponse(self):
                nonlocal phase
                phase = 'response_headers'
                return super().getresponse()
        class HTTPConnection(TransmissionBoundary, http.client.HTTPConnection):
            def connect(self):
                nonlocal connect_attempted, phase
                connect_attempted = True
                phase = 'connect'
                try:
                    super().connect()
                except Exception:
                    phase = 'connect'  # Includes proxy setup, which may itself send.
                    raise
                self.sock.settimeout(read_timeout)
        class HTTPSConnection(TransmissionBoundary, http.client.HTTPSConnection):
            def connect(self):
                nonlocal connect_attempted, phase
                connect_attempted = True
                phase = 'connect'
                try:
                    super().connect()
                except Exception:
                    phase = 'connect'
                    raise
                self.sock.settimeout(read_timeout)
        class HTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, req):
                return self.do_open(HTTPConnection, req)
        class HTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, req):
                return self.do_open(HTTPSConnection, req)
        opener = urllib.request.build_opener(NoRedirect(), HTTPHandler(), HTTPSHandler())
        http_request = urllib.request.Request(request['url'], data=None if method == 'GET' else payload,
                                              headers=request['headers'], method=method)
        try:
            with opener.open(http_request, timeout=request.get('connect_timeout', request['timeout'])) as response:
                phase = 'response_body'
                if request.get('stream_protocol'):
                    sys.path.insert(0,str(Path(__file__).resolve().parent))
                    from native_stream import decode
                    def emit(value):
                        sys.stdout.buffer.write(json.dumps(value,ensure_ascii=False).encode()+b'\n')
                        sys.stdout.buffer.flush()
                    body = json.dumps(decode(response,request['stream_protocol'],emit),ensure_ascii=False).encode()
                else:
                    body = response.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                raise ValueError('response_limit')
        except urllib.error.HTTPError as exc:
            # Status, busy_hints and a refused request's redacted reason only: errors may echo credentials.
            try:
                body = exc.read(8192)
            except (OSError, AttributeError, ValueError):
                body = b''
            frame = {'kind': 'http_error', 'status': exc.code, **busy_hints(exc, body),
                     **provider_error(exc.code, body, request['headers'])}
            exc.close()
            sys.stdout.buffer.write(json.dumps(frame).encode() + b'\n')
            return
        if request.get('stream_protocol'):
            emit({'kind':'stream_response','body':json.loads(body)})
        else:
            sys.stdout.buffer.write(b'{"kind":"response"}\n' + body)
    except Exception as exc:
        # No exception text, response headers, URL or secret values in diagnostics.
        diagnostic = {'kind': 'interrupted', 'error_type': type(exc).__name__, 'phase': phase}
        cause = error_cause(exc)
        if cause is not None:
            diagnostic['cause'] = cause
        reason = connection_failure(exc)
        if connect_attempted and not transmission_started and reason in CONNECT_REASONS:
            diagnostic.update(kind='not_dispatched', reason=reason)
        if type(exc) is ValueError and str(exc) in STREAM_REASONS:
            diagnostic['reason'] = str(exc)
        sys.stdout.buffer.write(json.dumps(diagnostic).encode() + b'\n')


if __name__ == '__main__':
    main()
