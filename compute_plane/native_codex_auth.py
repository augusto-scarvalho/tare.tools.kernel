"""Read one explicitly selected Codex ChatGPT login; never run a CLI agent.

Credentials remain in their existing store. This initial reader does not renew
or rewrite shared login state; an expired login is an explicit authentication
failure, never a fallback to a separately billed API key.
"""
import base64
import math
import time
from pathlib import Path

from tare_tools_agent_runtime import ModelFailure
from .native_models import NativeHttpClient

CODEX_ENDPOINT = 'https://chatgpt.com/backend-api/codex'


class CodexFileAuth:
    def __init__(self, path, *, account_id=None, clock=time.time):
        self.path = Path(path)
        self.account_id = account_id
        self.clock = clock

    def headers(self):
        try:
            with self.path.open('rb') as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                raise ValueError('oversized login')
            from .native_models import object_json
            auth = object_json(raw)
            if auth.get('auth_mode') != 'chatgpt':
                raise ModelFailure('CODEX_CHATGPT_LOGIN_REQUIRED', not_started=True)
            tokens = auth['tokens']
            access, account = tokens['access_token'], tokens['account_id']
            if not isinstance(access, str) or not isinstance(account, str) or not access or not account:
                raise ValueError('missing login fields')
            if self.account_id is not None and account != self.account_id:
                raise ModelFailure('CODEX_ACCOUNT_CHANGED', not_started=True)
            payload = access.split('.')[1]
            claims = object_json(base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4)))
            # This local expiry check is not signature verification. The provider
            # authenticates the token; no token claim grants local authority.
            expiry = claims.get('exp')
            if type(expiry) not in (int, float) or not math.isfinite(expiry) or not expiry > self.clock() + 30:
                raise ModelFailure('CODEX_LOGIN_EXPIRED', not_started=True)
            return {'Authorization': 'Bearer ' + access, 'ChatGPT-Account-Id': account,
                    'originator': 'tare.tools', 'User-Agent': 'tare.tools/native',
                    'Accept': 'text/event-stream'}
        except ModelFailure:
            raise
        except Exception:
            raise ModelFailure('CODEX_LOGIN_UNAVAILABLE', not_started=True) from None


class CodexHttpClient(NativeHttpClient):
    """Keep the subscription's HTTP cache affinity aligned with the wire key."""

    def post_observed(self, path, body, control, *, stream_protocol=None, on_event=None, headers=None):
        exchange_headers = headers

        def headers():
            values = self.request_headers() if self.request_headers is not None else {}
            if path == '/responses' and isinstance(body, dict) and 'prompt_cache_key' in body:
                values = {**values, 'session-id': body['prompt_cache_key']}
            return values

        # Headers belong to this exchange, even if a caller shares the client.
        exchange = NativeHttpClient(self.endpoint, credential=self.credential,
            request_headers=headers, timeout_seconds=self.timeout_seconds,
            connect_timeout_seconds=self.connect_timeout_seconds,
            read_timeout_seconds=self.read_timeout_seconds,
            request_byte_limit=self.request_byte_limit)
        return exchange.post_observed(path, body, control,
            stream_protocol=stream_protocol, on_event=on_event, headers=exchange_headers)


def client(path, *, account_id=None, **transport):
    """Credentials can only be sent to the fixed Codex subscription endpoint."""
    auth = CodexFileAuth(path, account_id=account_id)
    return CodexHttpClient(CODEX_ENDPOINT, request_headers=auth.headers, **transport)
