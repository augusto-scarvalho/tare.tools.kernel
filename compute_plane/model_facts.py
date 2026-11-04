"""Read-only bounded observations from model catalog/status endpoints."""
import hashlib
import json
import math
import time
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def read_json_observation(url, *, timeout=2, ttl=5, max_bytes=65536):
    """GET only; no credentials, model preload, tokenizer or generation request."""
    started = time.time()
    result = {'source': url, 'observed_at': started, 'expires_at': started + ttl,
              'status': 'UNKNOWN'}
    try:
        request = urllib.request.Request(url, headers={'Accept': 'application/json'})
        deadline = time.monotonic() + timeout
        with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
            chunks = []; size = 0
            while size <= max_bytes:
                if time.monotonic() > deadline:
                    raise ValueError('MODEL_FACTS_TIMEOUT')
                block = response.read1(min(65536, max_bytes + 1 - size))
                if not block:
                    break
                chunks.append(block); size += len(block)
            raw = b''.join(chunks)
        if len(raw) > max_bytes:
            raise ValueError('MODEL_FACTS_OVERFLOW')
        def invalid_constant(value):
            raise ValueError('MODEL_FACTS_NONFINITE_NUMBER')
        data = json.loads(raw, parse_constant=invalid_constant)
        if not isinstance(data, dict) or time.time() > result['expires_at']:
            raise ValueError('MODEL_FACTS_INVALID_OR_EXPIRED')
        result.update(status='OBSERVED', value=data, response_sha256=hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError):
        result['reason'] = 'MODEL_FACTS_UNAVAILABLE'
    return result


def gpu_availability(observation, model, *, now=None):
    """A free lease is an admission observation, never a reservation or VRAM proof."""
    now = time.time() if now is None else now
    result = {k: observation.get(k) for k in ('source', 'observed_at', 'expires_at', 'response_sha256')}
    result.update(status='UNKNOWN', reason='GPU_STATUS_UNKNOWN', immediate=False)
    try:
        if (observation['status'] != 'OBSERVED' or not observation['observed_at'] <= now <= observation['expires_at']):
            return result
        value = observation['value']
        lease = value['gpu_coordination']
        if (value.get('role') != 'qualified-model-gateway' or value.get('status') != 'ok'
                or lease.get('enabled') is not True or lease.get('error')
                or type(lease.get('held')) is not bool):
            return result
        if model not in value.get('available_models', []):
            result.update(status='UNAVAILABLE', reason='GPU_MODEL_NOT_ADVERTISED')
        elif isinstance(lease.get('reservation'), dict):
            # Contract gpu-lease/2: a named reservation; only its holder's requests (with its secret) get the GPU.
            result.update(status='BUSY', reason='GPU_RESERVED', owner_kind='reservation',
                          reservation=str(lease['reservation'].get('name') or ''))
        elif any(t.get('kind') == 'image' for t in lease.get('queue') or [] if isinstance(t, dict)):
            # The gateway answers text 503 gpu_busy while an image job waits for the GPU.
            result.update(status='BUSY', reason='GPU_IMAGE_WAITING', owner_kind='image')
        elif lease['held'] and (lease.get('owner') or {}).get('kind') == 'text':
            # Another text request: the gateway serves waiting text in order, so this admits (after it).
            result.update(status='QUEUED', reason='GPU_TEXT_QUEUE', immediate=True, queued=True, owner_kind='text',
                          resident=value.get('current_model') == model,
                          loading_required=value.get('current_model') != model)
        elif lease['held']:
            kind = (lease.get('owner') or {}).get('kind')
            result.update(status='BUSY', reason='GPU_LEASE_HELD', owner_kind='image' if kind == 'image' else 'unknown')
        else:
            result.update(status='AVAILABLE', reason='GPU_LEASE_FREE', immediate=True,
                          resident=value.get('current_model') == model,
                          loading_required=value.get('current_model') != model)
        eta = lease.get('eta_seconds')  # contract gpu-lease/1: until text gets the GPU, from the node's estimates
        if result['status'] in ('BUSY', 'QUEUED') and type(eta) in (int, float) and math.isfinite(eta) and eta >= 0:
            result['eta_seconds'] = eta
    except (KeyError, TypeError, AttributeError):
        pass
    return result
