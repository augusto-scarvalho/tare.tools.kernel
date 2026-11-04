"""Host-prepared command bytes at existing execution boundaries; no policy or sandbox."""
from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Protocol, Any


def environment_digest(environment):
    return hashlib.sha256(json.dumps(sorted(environment.items()), ensure_ascii=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class CommandFacts:
    operation: str
    owner: str
    repository: str
    cwd: str
    environment_sha256: str
    grant_sha256: str
    argv: tuple[str, ...] | None
    payload_json: str
    session_id: str = ''
    effect_id: str = ''
    work_id: str = ''
    binding_id: str = ''
    fence_seq: int = 0


class CommandMediationPort(Protocol):
    def execute(self, facts: CommandFacts, effect: Callable[[], Any]) -> Any: ...


class CallbackCommandMediation:
    def __init__(self, callback):
        self._callback = callback

    def execute(self, facts, effect):
        return self._callback(facts, effect)


def invoke(port, facts, effect):
    """No fallback on denial, absence of verdict or an adapter exception."""
    if port is None:
        return effect()
    return port.execute(facts, effect)
