"""GET observations use the same real isolated transport as model POSTs."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread
import time

import pytest
from compute_plane.native_models import NativeHttpClient, NativeTransportInterrupted
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl, ModelFailure


@pytest.mark.parametrize('response', ['ok', 'refused', 'redirect', 'oversize', 'cancel'])
def test_get_is_bounded_authenticated_and_never_retried_or_redirected(response):
    entered, release = Event(), Event()
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            seen.append((self.path, self.headers.get('Authorization'), self.headers.get('Content-Length')))
            entered.set()
            if response == 'cancel': release.wait(10)
            raw = json.dumps({'data': [{'id': 'fixture'}]}).encode()
            if response == 'refused':
                self.send_response(403); raw = b'private-credential-must-not-escape'
            elif response == 'redirect':
                self.send_response(302); self.send_header('Location', '/must-not-follow')
            else:
                self.send_response(200)
                if response == 'oversize': raw = b'x'*(1024*1024+1)
            self.send_header('Content-Length', str(len(raw)))
            try:
                self.end_headers(); self.wfile.write(raw)
            except OSError: pass
        def do_POST(self):
            seen.append(('POST',)); self.send_error(500)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    token = CancellationToken()
    control = ExecutionControl(BindingRef('check', 'binding', 'owner', 1, 'digest'), token, time.monotonic()+10)
    client = NativeHttpClient(f'http://127.0.0.1:{server.server_port}/v1', credential=lambda:'private-credential', timeout_seconds=5)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(client.get_observed, '/models', control)
            assert entered.wait(5)
            if response == 'cancel': token.cancel()
            if response == 'ok':
                value, receipt = future.result(8)
                assert value == {'data': [{'id': 'fixture'}]}
            else:
                with pytest.raises((ModelFailure, NativeTransportInterrupted)) as caught:
                    future.result(8)
                receipt = caught.value.transport_observation
                assert 'private-credential' not in str(caught.value)
            assert receipt['cleanup_verified']
            assert receipt['method'] == 'GET' and receipt['operation_path'] == '/models'
            assert receipt['http_attempts_upper_bound'] == 1
            assert 'private-credential' not in json.dumps(receipt)
            assert seen == [('/v1/models', 'Bearer private-credential', None)]
    finally:
        release.set(); server.shutdown(); server.server_close(); thread.join(3)
