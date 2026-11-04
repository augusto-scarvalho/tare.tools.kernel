"""The gpu-lease/2 reservation client against a loopback fake node: no real node, no network beyond 127.0.0.1."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
import time

import pytest

from compute_plane import gpu_lease_client as client


@contextmanager
def node(refuse=None, lose_after=None):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, headers=()):
            raw = json.dumps(body).encode()
            self.send_response(status)
            for name, value in headers:
                self.send_header(name, value)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def handle_any(self, method):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'] or 0)) or b'{}')
            seen.append((method, self.path, self.headers['Authorization'], body))
            if refuse:
                return self.reply(refuse[0], {'error': {'type': refuse[1]}}, [('Retry-After', '120')])
            if self.path == '/v1/gpu/reservations':
                return self.reply(201, {'id': 'r1', 'secret': 's3cret', 'fence': 7, 'expires_at': 1, 'hard_until': 2})
            renewals = sum(1 for m, p, *_ in seen if p.endswith('/renew'))
            if self.path.endswith('/renew') and lose_after is not None and renewals > lose_after:
                return self.reply(404, {'error': {'type': 'reservation_not_found'}})
            return self.reply(200, {'expires_at': 3})

        def do_POST(self):
            self.handle_any('POST')

        def do_DELETE(self):
            self.handle_any('DELETE')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_hold_reserves_renews_and_releases_with_the_node_token():
    with node() as (origin, seen):
        with client.hold(origin, 'tok', 'bench', 7200, holder='acer', ttl=0.3, wait=2) as reservation:
            assert reservation['secret'] == 's3cret' and reservation['fence'] == 7
            time.sleep(0.45)
        methods = [(m, p) for m, p, *_ in seen]
    assert methods[0] == ('POST', '/v1/gpu/reservations') and methods[-1] == ('DELETE', '/v1/gpu/reservations/r1')
    assert ('POST', '/v1/gpu/reservations/r1/renew') in methods
    assert all(auth == 'Bearer tok' for _, _, auth, _ in seen)
    assert seen[0][3] == {'name': 'bench', 'holder': 'acer', 'max_seconds': 7200, 'ttl_seconds': 0, 'wait_seconds': 2}
    assert all(body == {'secret': 's3cret', 'fence': 7} for _, path, _, body in seen[1:])
    assert client.RESERVATION_HEADER == 'X-Tare-GPU-Reservation'


@pytest.mark.parametrize('status,kind', [(409, 'gpu_reserved'), (503, 'gpu_busy'), (401, 'unauthorized')])
def test_a_refused_reservation_says_why_and_when_to_retry(status, kind):
    with node(refuse=(status, kind)) as (origin, _):
        with pytest.raises(client.ReservationRefused) as refused:
            client.acquire(origin, 'tok', 'bench', 60)
    assert refused.value.status == status and refused.value.kind == kind and refused.value.retry_after == 120


def test_a_lost_reservation_is_reported_once_and_not_released():
    lost = []
    with node(lose_after=1) as (origin, seen):
        with client.hold(origin, 'tok', 'bench', 60, ttl=0.15, wait=1, on_lost=lost.append):
            time.sleep(0.4)
        methods = [m for m, *_ in seen]
    assert len(lost) == 1 and isinstance(lost[0], client.LeaseLost) and 'DELETE' not in methods
