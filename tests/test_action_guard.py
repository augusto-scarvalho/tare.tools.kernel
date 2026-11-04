"""Independent pre-effect falsifiers; fake owner/claim ports, no model or operator data."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Lock

import pytest
from compute_plane.action_control import ActionAuthorization, ActionSnapshot, ExactActionGuard
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl


def fixture():
    action = ActionSnapshot('w', 's', 'e', 'b', 3, 'verify', 'r', '/fixture',
        *(['a' * 64] * 5), 'mediation-only', 200.0, argv=('python', '-m', 'pytest'))
    permit = ActionAuthorization('owner', 'nonce', action.action_sha256, 's', 'e', 'b', 3, 150.0)
    clock = [100.0]
    control = ExecutionControl(BindingRef('w', 'b', 'operator', 3, 'opaque'), CancellationToken(),
        199.0, monotonic=lambda: clock[0])
    consumed, effects, lock = set(), [], Lock()
    def claim(subject, authorization):
        with lock:
            key = (subject.session_id, subject.effect_id)
            if key in consumed: return False
            consumed.add(key)
            return True
    verifier = lambda subject, authorization: authorization.issuer == 'owner' and authorization.nonce == 'nonce'
    def guard(**overrides):
        return ExactActionGuard(**{'verify_authorization': verifier, 'claim': claim,
                                  'clock': lambda: clock[0], **overrides})
    def effect():
        effects.append('once')
        return {'status': 'succeeded'}
    return action, permit, control, clock, consumed, effects, guard, effect


def test_exact_effect_runs_once_across_guard_instances_with_the_same_claim_port():
    a, p, c, _, claims, effects, guard, effect = fixture()
    assert guard().run(a, p, c, lambda: a, effect) == {'status': 'succeeded'}
    with pytest.raises(ValueError): guard().run(a, p, c, lambda: a, effect)
    assert effects == ['once'] and claims == {('s', 'e')}


@pytest.mark.parametrize('change', [{'work_id': 'other'}, {'binding_id': 'other'}, {'fence': 4}])
def test_live_control_must_match_the_exact_action_binding(change):
    a, p, c, _, claims, effects, guard, effect = fixture()
    wrong = replace(c, binding=replace(c.binding, **change))
    with pytest.raises(ValueError): guard().run(a, p, wrong, lambda: a, effect)
    assert not effects and not claims


@pytest.mark.parametrize('change', [
    {'source_sha256': 'b'*64}, {'policy_sha256': 'b'*64}, {'grant_sha256': 'b'*64},
    {'environment_sha256': 'b'*64}, {'context_sha256': 'b'*64}, {'execution_profile': 'other'},
    {'argv': ('python', 'different')}, {'argv': None, 'script': b'echo x'}, {'stdin': b'input'},
    {'payload_json': '{"changed":true}'}, {'cwd': '/other'}, {'fence_seq': 4},
    {'session_id': 'other'}, {'effect_id': 'other'}, {'binding_id': 'other'}, {'work_id': 'other'},
    # Dataclass equality considers these equal, but the exact representation changed.
    {'deadline': 200},
])
def test_changed_subject_is_refused_before_claim(change):
    a, p, c, _, claims, effects, guard, effect = fixture()
    with pytest.raises(ValueError): guard().run(a, p, c, lambda: replace(a, **change), effect)
    assert not effects and not claims


@pytest.mark.parametrize('change', [
    {'issuer': 'intruder'}, {'nonce': 'invented'}, {'action_sha256': 'b'*64},
    {'session_id': 'other'}, {'effect_id': 'other'}, {'binding_id': 'other'}, {'fence_seq': 4},
    {'expires_at': 100.0},
])
def test_wrong_or_expired_permit_never_claims(change):
    a, p, c, _, claims, effects, guard, effect = fixture()
    with pytest.raises(ValueError): guard().run(a, replace(p, **change), c, lambda: a, effect)
    assert not effects and not claims


@pytest.mark.parametrize('where', ['verifier', 'refresh', 'claim'])
def test_cancellation_inside_a_port_prevents_the_effect(where):
    a, p, c, _, claims, effects, guard, effect = fixture()
    def cancel_and_return(value):
        c.cancellation.cancel()
        return value
    overrides = {'verify_authorization': lambda *args: cancel_and_return(True)} if where == 'verifier' else {}
    if where == 'claim': overrides['claim'] = lambda *args: cancel_and_return(True)
    refresh = (lambda: cancel_and_return(a)) if where == 'refresh' else lambda: a
    with pytest.raises(ValueError): guard(**overrides).run(a, p, c, refresh, effect)
    assert not effects
    if where != 'claim': assert not claims


@pytest.mark.parametrize('unknown', [False, True])
def test_crash_or_unknown_result_never_reexecutes(unknown):
    a, p, c, _, claims, effects, guard, _ = fixture()
    def effect():
        effects.append('once')
        if unknown: return {'status': 'unknown'}
        raise RuntimeError('crash after effect intent')
    if unknown: assert guard().run(a, p, c, lambda: a, effect)['status'] == 'unknown'
    else:
        with pytest.raises(RuntimeError): guard().run(a, p, c, lambda: a, effect)
    with pytest.raises(ValueError): guard().run(a, p, c, lambda: a, effect)
    assert effects == ['once'] and claims == {('s', 'e')}


@pytest.mark.parametrize('port', ['verify_authorization', 'claim'])
def test_truthy_nonboolean_port_results_cannot_allow_an_effect(port):
    a, p, c, _, _, effects, guard, effect = fixture()
    with pytest.raises(ValueError): guard(**{port: lambda *args: 1}).run(a, p, c, lambda: a, effect)
    assert not effects


def test_shared_atomic_claim_prevents_simultaneous_duplicate_effects():
    a, p, c, _, _, effects, guard, effect = fixture()
    def run():
        try: return guard().run(a, p, c, lambda: a, effect)
        except ValueError: return 'refused'
    with ThreadPoolExecutor(max_workers=2) as executor: results = list(executor.map(lambda _: run(), range(2)))
    assert results.count('refused') == 1 and effects == ['once']


def test_source_changed_while_claiming_is_refused_after_claim_without_an_effect():
    a, p, c, _, _, effects, guard, effect = fixture()
    current = [a]
    def claim(*args):
        current[0] = replace(a, source_sha256='b'*64)
        return True
    with pytest.raises(ValueError): guard(claim=claim).run(a, p, c, lambda: current[0], effect)
    assert not effects


@pytest.mark.parametrize('now', [float('nan'), float('inf'), True])
def test_invalid_clock_cannot_bypass_expiration(now):
    a, p, c, _, _, effects, guard, effect = fixture()
    with pytest.raises(ValueError): guard(clock=lambda: now).run(a, p, c, lambda: a, effect)
    assert not effects
