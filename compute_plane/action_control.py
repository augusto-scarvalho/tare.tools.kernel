# ActionSnapshot: host-prepared, immutable description of one action.
# It carries no authority and no sandbox; action_sha256 covers every field.
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass

_HEX64 = re.compile(r'[0-9a-f]{64}')
_DOMAIN = b'tare.action_snapshot/v1;'
_IDENTITY = ('work_id', 'session_id', 'effect_id', 'binding_id', 'operation', 'repository', 'cwd', 'execution_profile')
_DIGESTS = ('environment_sha256', 'context_sha256', 'source_sha256', 'policy_sha256', 'grant_sha256')
_NUL = chr(0)

def _no_constant(token: str) -> None:
    raise ValueError(f'payload_json contains {token}')

def _is_finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False

def _check_payload(text: object) -> None:
    if type(text) is not str:
        raise ValueError('payload_json must be a str')
    try:
        json.loads(text, parse_constant=_no_constant, parse_float=lambda token: float(token) if math.isfinite(float(token)) else _no_constant(token))
    except ValueError as exc:
        raise ValueError(f'payload_json must be valid JSON without NaN or Infinity: {exc}') from None

def _framed(tag: str, data: bytes) -> bytes:
    return tag.encode() + len(data).to_bytes(8, 'big') + data

def _text_bytes(value: str) -> bytes:
    return value.encode('utf-8', 'surrogatepass')

def _optional(value: object, encode) -> bytes:
    if value is None:
        return bytes(1)
    return bytes([1]) + encode(value)

def _argv_bytes(argv: tuple) -> bytes:
    items = b''.join(_framed('a', _text_bytes(item)) for item in argv)
    return len(argv).to_bytes(8, 'big') + items

@dataclass(frozen=True)
class ActionSnapshot:
    work_id: str
    session_id: str
    effect_id: str
    binding_id: str
    fence_seq: int
    operation: str
    repository: str
    cwd: str
    environment_sha256: str
    context_sha256: str
    source_sha256: str
    policy_sha256: str
    grant_sha256: str
    execution_profile: str
    deadline: float
    argv: tuple[str, ...] | None = None
    script: bytes | None = None
    stdin: bytes | None = None
    payload_json: str | None = None

    def __post_init__(self) -> None:
        for name in _IDENTITY:
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f'{name} must be a non-empty string')
        for name in _DIGESTS:
            value = getattr(self, name)
            if type(value) is not str or _HEX64.fullmatch(value) is None:
                raise ValueError(f'{name} must be 64 lowercase hex characters')
        if type(self.fence_seq) is not int or self.fence_seq < 0:
            raise ValueError('fence_seq must be an int >= 0')
        if not _is_finite_number(self.deadline):
            raise ValueError('deadline must be a finite number')
        if self.argv is not None:
            if type(self.argv) is not tuple or not self.argv:
                raise ValueError('argv must be a non-empty tuple of strings')
            if any(type(item) is not str or _NUL in item for item in self.argv):
                raise ValueError('argv items must be strings without NUL')
            if self.script is not None:
                raise ValueError('argv and script cannot coexist')
        if self.script is not None and type(self.script) is not bytes:
            raise ValueError('script must be bytes')
        if self.stdin is not None and type(self.stdin) is not bytes:
            raise ValueError('stdin must be bytes')
        if self.payload_json is not None:
            _check_payload(self.payload_json)

    @property
    def action_sha256(self) -> str:
        parts = [_DOMAIN]
        parts += [_framed(name, _text_bytes(getattr(self, name))) for name in _IDENTITY + _DIGESTS]
        parts.append(_framed('fence_seq', str(self.fence_seq).encode()))
        parts.append(_framed('deadline', repr(self.deadline).encode()))
        parts.append(_framed('argv', _optional(self.argv, _argv_bytes)))
        parts.append(_framed('script', _optional(self.script, bytes)))
        parts.append(_framed('stdin', _optional(self.stdin, bytes)))
        parts.append(_framed('payload_json', _optional(self.payload_json, _text_bytes)))
        return hashlib.sha256(b''.join(parts)).hexdigest()


class ActionRefused(ValueError):
    # Raised before an effect runs when any guard check fails.
    pass


@dataclass(frozen=True)
class ActionAuthorization:
    # Envelope naming one snapshot. It proves nothing without the host verifier.
    issuer: str
    nonce: str
    action_sha256: str
    session_id: str
    effect_id: str
    binding_id: str
    fence_seq: int
    expires_at: float

    def __post_init__(self) -> None:
        for name in ('issuer', 'nonce', 'session_id', 'effect_id', 'binding_id'):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f'{name} must be a non-empty string')
        if type(self.action_sha256) is not str or _HEX64.fullmatch(self.action_sha256) is None:
            raise ValueError('action_sha256 must be 64 lowercase hex characters')
        if type(self.fence_seq) is not int or self.fence_seq < 0:
            raise ValueError('fence_seq must be an int >= 0')
        if not _is_finite_number(self.expires_at):
            raise ValueError('expires_at must be a finite number')


class ExactActionGuard:
    # Runs one effect only when every check passes at call time.
    # Limits: this guard signs nothing and proves no authority; the host verify
    # callback must establish authenticity. It does not sandbox the effect.
    # Exactly-once behaviour rests on the host's atomic, durable claim port.
    # Only the snapshot and refresh() are checked; other external state is not.

    def __init__(self, verify_authorization, claim, clock=time.monotonic) -> None:
        if not (callable(verify_authorization) and callable(claim) and callable(clock)):
            raise TypeError('verify_authorization, claim and clock must be callable')
        self._verify = verify_authorization
        self._claim = claim
        self._clock = clock

    def revalidate(self, action, authorization, control, refresh):
        if not isinstance(action, ActionSnapshot) or not isinstance(authorization, ActionAuthorization):
            raise TypeError('action must be ActionSnapshot and authorization ActionAuthorization')
        if not callable(refresh):
            raise TypeError('refresh must be callable')
        self._check(action, authorization, control)
        if self._verify(action, authorization) is not True:
            raise ActionRefused('authorization verifier did not accept the envelope')
        self._check(action, authorization, control)
        fresh = refresh()
        if not isinstance(fresh, ActionSnapshot) or fresh.action_sha256 != action.action_sha256:
            raise ActionRefused('snapshot changed while awaiting native review')
        self._check(action, authorization, control)

    def run(self, action, authorization, control, refresh, effect):
        if not isinstance(action, ActionSnapshot) or not isinstance(authorization, ActionAuthorization):
            raise TypeError('action must be ActionSnapshot and authorization ActionAuthorization')
        if not callable(getattr(control, 'should_stop', None)):
            raise TypeError('control must expose should_stop()')
        if not (callable(refresh) and callable(effect)):
            raise TypeError('refresh and effect must be callable')
        self._check(action, authorization, control)
        if self._verify(action, authorization) is not True:
            raise ActionRefused('authorization verifier did not accept the envelope')
        self._check(action, authorization, control)
        refreshed = refresh()
        if not isinstance(refreshed, ActionSnapshot) or refreshed.action_sha256 != action.action_sha256:
            raise ActionRefused('refreshed snapshot does not match the original action')
        if (refreshed.work_id, refreshed.binding_id, refreshed.fence_seq) != (action.work_id, action.binding_id, action.fence_seq):
            raise ActionRefused('work, binding or fence changed')
        self._check(action, authorization, control)
        if self._claim(action, authorization) is not True:
            raise ActionRefused('claim was not granted')
        self._check(action, authorization, control)
        refreshed = refresh()
        if not isinstance(refreshed, ActionSnapshot) or refreshed.action_sha256 != action.action_sha256:
            raise ActionRefused('snapshot changed while consuming the authorization')
        self._check(action, authorization, control)
        return effect()

    def _check(self, action, authorization, control) -> None:
        if control.should_stop():
            raise ActionRefused('execution control is stopped')
        binding = getattr(control, 'binding', None)
        if (binding is None or type(getattr(binding, 'fence', None)) is not int
                or (getattr(binding, 'work_id', None), getattr(binding, 'binding_id', None), binding.fence)
                != (action.work_id, action.binding_id, action.fence_seq)):
            raise ActionRefused('live control does not match the action binding')
        now = self._clock()
        if not _is_finite_number(now):
            raise ActionRefused('clock did not return a finite monotonic number')
        if now >= action.deadline or now >= authorization.expires_at:
            raise ActionRefused('action or authorization deadline has passed')
        mine = (authorization.session_id, authorization.effect_id, authorization.binding_id, authorization.fence_seq, authorization.action_sha256)
        if mine != (action.session_id, action.effect_id, action.binding_id, action.fence_seq, action.action_sha256):
            raise ActionRefused('authorization identity does not match the action')
