"""Client of a GPU node's named reservations (contract gpu-lease/2, docs/GPU_LEASE_CONTRACT.md).

`hold()` reserves the node's GPU for text by name, renews it every third of its ttl in the background and releases
it on exit. Requests that should use the reserved GPU carry the reservation's secret in RESERVATION_HEADER; the
node refuses everyone else with 503 gpu_reserved. The kernel never decides to reserve: its caller does.
"""
import contextlib
import json
import threading
import urllib.error
import urllib.request

RESERVATION_HEADER = 'X-Tare-GPU-Reservation'


class ReservationRefused(RuntimeError):
    """The node did not reserve: `kind` gpu_reserved (another holder), gpu_busy (a job kept the GPU), unauthorized,
    reservations_disabled...; `retry_after` in seconds when the node said."""

    def __init__(self, status, kind, retry_after=None):
        super().__init__(f'{status} {kind}' + (f', retry after {retry_after} s' if retry_after else ''))
        self.status, self.kind, self.retry_after = status, kind, retry_after


class LeaseLost(RuntimeError):
    """The reservation ended without its holder: its ttl ran out, its maximum time, or a newer holder."""


def _call(origin, path, token, body, *, method='POST', timeout=30):
    request = urllib.request.Request(origin.rstrip('/') + path, data=json.dumps(body).encode(), method=method,
                                     headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), response.headers
    except urllib.error.HTTPError as exc:
        with contextlib.suppress(ValueError):
            return exc.code, json.load(exc), exc.headers
        return exc.code, {}, exc.headers


def acquire(origin, token, name, max_seconds, *, holder='', ttl=60, wait=30):
    """{'origin', 'id', 'secret', 'fence', 'hard_until', 'ttl'}, or ReservationRefused."""
    status, body, headers = _call(origin, '/v1/gpu/reservations', token,
                                  {'name': name, 'holder': holder, 'max_seconds': int(max_seconds),
                                   'ttl_seconds': int(ttl), 'wait_seconds': wait}, timeout=wait + 15)
    if status == 201:
        return {'origin': origin, 'id': body['id'], 'secret': body['secret'], 'fence': body['fence'],
                'hard_until': body.get('hard_until'), 'ttl': ttl}
    after = str((headers or {}).get('Retry-After') or '').strip()
    raise ReservationRefused(status, (body.get('error') or {}).get('type') or 'refused',
                             int(after) if after.isdigit() else None)


def renew(reservation, token):
    status, body, _ = _call(reservation['origin'], f"/v1/gpu/reservations/{reservation['id']}/renew", token,
                            {'secret': reservation['secret'], 'fence': reservation['fence']})
    if status != 200:
        raise LeaseLost(f"{status} {(body.get('error') or {}).get('type') or 'refused'}")
    return body.get('expires_at')


def release(reservation, token):
    """True when the node released it; False when it was already over."""
    status, _, _ = _call(reservation['origin'], f"/v1/gpu/reservations/{reservation['id']}", token,
                         {'secret': reservation['secret'], 'fence': reservation['fence']}, method='DELETE')
    return status == 200


@contextlib.contextmanager
def hold(origin, token, name, max_seconds, *, holder='', ttl=60, wait=30, on_lost=None):
    """Reserve, renew every ttl/3 until the block ends, release. A lost reservation calls `on_lost(error)` once and
    stops renewing; the block goes on (its requests then get 503 gpu_reserved or a free GPU, like anyone's)."""
    reservation = acquire(origin, token, name, max_seconds, holder=holder, ttl=ttl, wait=wait)
    stop, lost = threading.Event(), []

    def renewer():
        while not stop.wait(ttl / 3):
            try:
                renew(reservation, token)
            except LeaseLost as exc:
                lost.append(exc)
                if on_lost is not None:
                    on_lost(exc)
                return
            except OSError:
                continue  # a network blip: the next renewal, or the ttl, decides

    thread = threading.Thread(target=renewer, name='gpu-reservation-renew', daemon=True)
    thread.start()
    try:
        yield reservation
    finally:
        stop.set()
        thread.join(5)
        if not lost:
            with contextlib.suppress(OSError):
                release(reservation, token)
