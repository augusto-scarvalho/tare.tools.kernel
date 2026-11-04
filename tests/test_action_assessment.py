import hashlib
from dataclasses import FrozenInstanceError

import pytest

from compute_plane.action_assessment import AssessmentObservation, AssessmentRequest
from compute_plane.action_control import ActionSnapshot


def fixtures():
    context = "public fixture context"
    action = ActionSnapshot("w", "s", "e", "b", 1, "verify", "repo", "/fixture",
        *(["a" * 64] * 5), "mediation-only", 100.0, payload_json="{}")
    return action, context


def request(**changes):
    action, context = fixtures()
    action = __import__("dataclasses").replace(action, context_sha256=hashlib.sha256(context.encode()).hexdigest())
    values = dict(action=action, stage="classifier", backend="local-fixture",
                  public_context=context, max_output_tokens=128, deadline=90.0)
    values.update(changes)
    return AssessmentRequest(**values)


def observation(req, **changes):
    values = dict(stage=req.stage, backend=req.backend, backend_version="v1",
        request_sha256=req.request_sha256, action_sha256=req.action.action_sha256,
        context_sha256=req.action.context_sha256, policy_sha256=req.action.policy_sha256,
        status="OBSERVED", verdict="ROUTINE_ELIGIBLE", reason_codes=(),
        elapsed_ms=1.0, result_origin="fixture")
    values.update(changes)
    return AssessmentObservation(**values)


def test_request_is_frozen_and_digest_binds_parameters():
    req = request()
    with pytest.raises(FrozenInstanceError):
        req.backend = "changed"
    assert len({req.request_sha256, request(backend="other").request_sha256,
                request(stage="reviewer").request_sha256,
                request(max_output_tokens=129).request_sha256,
                request(public_context="other", action=__import__("dataclasses").replace(
                    req.action, context_sha256=hashlib.sha256(b"other").hexdigest())).request_sha256}) == 5


def test_context_must_match_and_token_count_is_strict():
    action, _ = fixtures()
    with pytest.raises(ValueError): request(public_context="wrong")
    with pytest.raises(ValueError): request(max_output_tokens=True)
    with pytest.raises(ValueError): request(deadline=float("inf"))


def test_observation_matches_all_subject_digests_and_backend():
    req = request()
    obs = observation(req)
    assert obs.matches(req)
    assert not obs.matches(request(backend="other"))
    assert not obs.matches(request(stage="reviewer"))
    assert not obs.matches(request(action=__import__("dataclasses").replace(req.action, policy_sha256="b" * 64)))


@pytest.mark.parametrize("change", [
    {"authority": "APPROVE"}, {"status": "ABSTAINED"},
    {"status": "OBSERVED", "verdict": None},
    {"status": "OBSERVED", "verdict": "APPROVE_RECOMMENDED"},
    {"elapsed_ms": float("nan")}, {"elapsed_ms": float("inf")},
    {"reason_codes": ["x"]}, {"reason_codes": {"x": 1}},
])
def test_invalid_observations_rejected(change):
    req = request()
    with pytest.raises(ValueError): observation(req, **change)


def test_unknown_accounting_remains_none_and_observation_is_frozen():
    obs = observation(request(), status="UNAVAILABLE", verdict=None)
    assert obs.usage_json is None and obs.cost_json is None
    with pytest.raises(FrozenInstanceError):
        obs.backend = "changed"
