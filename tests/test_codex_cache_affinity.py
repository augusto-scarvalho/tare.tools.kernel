"""Subscription affinity reaches the actual HTTP worker; no provider or login."""
import copy
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from compute_plane.native_codex_auth import CodexHttpClient
from tests.test_native_models import execution


@contextmanager
def receiver():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path, self.headers.get('session-id'), body))
            self.send_response(200)
            streaming = body.get('stream') is True
            self.send_header('Content-Type', 'text/event-stream' if streaming else 'application/json')
            self.end_headers()
            self.wfile.write(b'data: {"type":"response.completed","response":'
                b'{"id":"fixture","status":"completed","output":[]}}\n\n'
                if streaming else b'{"ok":true}')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def control():
    return replace(execution(), deadline_monotonic=time.monotonic() + 10,
                   monotonic=time.monotonic)


def test_same_key_reaches_worker_on_every_exchange_without_mutating_shared_headers_or_body():
    auth_headers = {'Accept': 'application/json'}
    with receiver() as (url, seen):
        client = CodexHttpClient(url, request_headers=lambda: auth_headers,
                                timeout_seconds=5, request_byte_limit=2 * 1024 * 1024)
        for key in ('session-one', 'session-one', 'explicit-shared-prefix'):
            body = {'prompt_cache_key': key, 'input': [{'role': 'user', 'content': 'task'}]}
            before = copy.deepcopy(body)
            response, receipt = client.post_observed('/responses', body, control())
            assert response == {'ok': True} and receipt['cleanup_verified']
            assert body == before and auth_headers == {'Accept': 'application/json'}
        client.post_observed('/other', {'prompt_cache_key': 'unrelated'}, control())
        client.post_observed('/responses', {'input': []}, control())
    assert [row[1] for row in seen] == [
        'session-one', 'session-one', 'explicit-shared-prefix', None, None]


def test_streaming_uses_the_same_affinity_and_cancellation_still_prevents_dispatch():
    with receiver() as (url, seen):
        client = CodexHttpClient(url)
        body = {'prompt_cache_key': 'stable-prefix', 'stream': True}
        result, receipt = client.post_stream_observed('/responses', body, control(),
                                                     protocol='openai-codex-responses')
        assert result['status'] == 'completed' and receipt['cleanup_verified']
        cancelled = control()
        cancelled.cancellation.cancel()
        from compute_plane.native_models import NativeTransportInterrupted
        with pytest.raises(NativeTransportInterrupted):
            client.post_stream_observed('/responses', body, cancelled,
                                        protocol='openai-codex-responses')
    assert len(seen) == 1 and seen[0][1] == 'stable-prefix'


@pytest.mark.parametrize('key', ['', None, {'bad': 'value'}, 'x\r\nInjected: value', 'x' * 8193])
def test_invalid_affinity_key_is_rejected_before_network_dispatch(key):
    with receiver() as (url, seen):
        client = CodexHttpClient(url)
        with pytest.raises(ValueError, match='invalid request header value'):
            client.post_observed('/responses', {'prompt_cache_key': key}, control())
        assert seen == []
