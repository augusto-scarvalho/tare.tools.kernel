"""Consultative, identity-bound action assessment contracts."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

from tare_tools_agent_runtime import ExecutionControl
from compute_plane.action_control import ActionAuthorization, ActionSnapshot, _check_payload, _is_finite_number

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_STAGES = {"classifier", "reviewer"}
_STATUSES = {"OBSERVED", "ABSTAINED", "UNAVAILABLE", "TIMEOUT", "MALFORMED", "CANCELLED"}
_VERDICTS = {
    "classifier": {"ROUTINE_ELIGIBLE", "REVIEW_REQUIRED", "DENY_RECOMMENDED", "INSUFFICIENT_EVIDENCE"},
    "reviewer": {"APPROVE_RECOMMENDED", "DENY_RECOMMENDED", "CLARIFICATION_REQUIRED", "NO_VERDICT"},
}

def _identity(value: object, name: str) -> None:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")

def _digest(value: object, name: str) -> None:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hex characters")

def _frame(tag: str, value: bytes) -> bytes:
    return tag.encode() + len(value).to_bytes(8, "big") + value

@dataclass(frozen=True)
class AssessmentRequest:
    action: ActionSnapshot
    stage: str
    backend: str
    public_context: str
    max_output_tokens: int
    deadline: float

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionSnapshot):
            raise ValueError("action must be an ActionSnapshot")
        if self.stage not in _STAGES:
            raise ValueError("invalid assessment stage")
        _identity(self.backend, "backend")
        if type(self.public_context) is not str:
            raise ValueError("public_context must be a string")
        if hashlib.sha256(self.public_context.encode("utf-8")).hexdigest() != self.action.context_sha256:
            raise ValueError("public_context does not match action context digest")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if not _is_finite_number(self.deadline) or self.deadline > self.action.deadline:
            raise ValueError("deadline must be finite and no later than the action deadline")

    @property
    def request_sha256(self) -> str:
        parts = [b"tare.assessment_request/v1;"]
        for tag, value in (("action", self.action.action_sha256), ("stage", self.stage),
                           ("backend", self.backend), ("context", self.public_context),
                           ("budget", str(self.max_output_tokens)), ("deadline", repr(self.deadline))):
            parts.append(_frame(tag, value.encode("utf-8")))
        return hashlib.sha256(b"".join(parts)).hexdigest()

@dataclass(frozen=True)
class AssessmentObservation:
    stage: str
    backend: str
    backend_version: str
    request_sha256: str
    action_sha256: str
    context_sha256: str
    policy_sha256: str
    status: str
    verdict: str | None
    reason_codes: tuple[str, ...]
    elapsed_ms: float
    result_origin: str
    usage_json: str | None = None
    cost_json: str | None = None
    authority: str = "NONE"

    def __post_init__(self) -> None:
        if self.authority != "NONE":
            raise ValueError("assessment observations have no authority")
        if self.stage not in _STAGES or self.status not in _STATUSES:
            raise ValueError("invalid stage or status")
        for name in ("backend", "backend_version", "result_origin"):
            _identity(getattr(self, name), name)
        for name in ("request_sha256", "action_sha256", "context_sha256", "policy_sha256"):
            _digest(getattr(self, name), name)
        if self.status == "OBSERVED":
            if type(self.verdict) is not str or self.verdict not in _VERDICTS[self.stage]:
                raise ValueError("OBSERVED requires a valid stage verdict")
        elif self.verdict is not None:
            raise ValueError("only OBSERVED may have a verdict")
        if type(self.reason_codes) is not tuple or any(type(code) is not str or not code for code in self.reason_codes):
            raise ValueError("reason_codes must be an immutable tuple of non-empty strings")
        if not _is_finite_number(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        for name in ("usage_json", "cost_json"):
            value = getattr(self, name)
            if value is not None and type(value) is not str:
                raise ValueError(f"{name} must be a string or None")
            if value is not None:
                _check_payload(value)

    def matches(self, request: AssessmentRequest) -> bool:
        return (isinstance(request, AssessmentRequest) and self.stage == request.stage and
                self.backend == request.backend and self.request_sha256 == request.request_sha256 and
                self.action_sha256 == request.action.action_sha256 and
                self.context_sha256 == request.action.context_sha256 and
                self.policy_sha256 == request.action.policy_sha256)

class ActionClassifierPort(Protocol):
    def assess(self, request: AssessmentRequest, control: ExecutionControl) -> AssessmentObservation: ...

class ActionReviewerPort(Protocol):
    def assess(self, request: AssessmentRequest, control: ExecutionControl) -> AssessmentObservation: ...

class ReviewedExecutionPort(Protocol):
    def execute(self, action: ActionSnapshot, authorization: ActionAuthorization,
                control: ExecutionControl) -> tuple[AssessmentObservation, str]: ...
