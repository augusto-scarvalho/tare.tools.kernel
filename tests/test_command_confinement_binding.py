"""The same mediated runner consumes a host-owned, immutable confinement binding."""
import json
from pathlib import Path
import pytest
from tests.test_native_capabilities import setup, call
from compute_plane.mediated_tools import MediatedTools, validate_command_confinement


def test_factory_environment_and_budget_match_command_facts(tmp_path, monkeypatch):
    seen = {}
    factory = lambda *args, **kwargs: None
    environment = {'SystemRoot': 'fixture', 'TMP': 'scratch'}
    class Port:
        def execute(self, facts, effect):
            seen['facts'] = facts
            return effect()
    tools = MediatedTools({'role': 'implementer', 'repositories': {'repo': str(tmp_path)},
                           'writable': {}, 'specgraph_edit_context': False, 'operation_timeout': 20},
                          process_factory=factory, command_environment=environment,
                          command_seconds=3, command_mediation=Port())
    environment['SECRET'] = 'must never reach child'
    def runner(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        Path(kwargs['dest'], 'stdout.txt').write_bytes(b'ok')
        Path(kwargs['dest'], 'stderr.txt').write_bytes(b'')
        return dict(exit_code=0, cleanup_verified=True, cancelled=False, timeout=False, output_limit=False)
    monkeypatch.setattr('compute_plane.managed_process.run_process', runner)
    assert tools.command(['fixture.exe', 'arg'], tmp_path, gate=False)['output'] == 'ok'
    assert seen['process_factory'] is factory and seen['timeout'] == 3
    assert seen['env'] == {'SystemRoot': 'fixture', 'TMP': 'scratch'}
    from compute_plane.command_mediation import environment_digest
    assert seen['facts'].environment_sha256 == environment_digest(seen['env'])
    assert json.loads(seen['facts'].payload_json)['timeout'] == 3


def test_binding_copies_environment_and_rejects_rebind_and_late_install(setup):
    _, _, tools, control = setup
    env = {'TMP': 'owned'}; factory = lambda *args, **kwargs: None
    tools.bind_command_confinement(factory, env, 5)
    env['SECRET'] = 'operator value'
    assert tools._tools.command_environment == {'TMP': 'owned'}
    assert 'bind_command_confinement' not in tools.names
    with pytest.raises(ValueError, match='binding_invalid'):
        tools.bind_command_confinement(factory, env, 5)
    call(tools, control, 'read', repository='repo', path='a.py')
    tools._tools.process_factory = None
    with pytest.raises(ValueError, match='binding_invalid'):
        tools.bind_command_confinement(factory, env, 5)


@pytest.mark.parametrize('environment,seconds', [
    ({'a': 'b', 'A': 'c'}, 5), ({'': 'b'}, 5), ({'a=b': 'c'}, 5),
    ({'a': '\0'}, 5), ({'a': 'b'}, True), ({'a': 'b'}, float('inf')), ({'a': 'b'}, 0),
])
def test_invalid_binding_does_not_partially_install(setup, environment, seconds):
    _, _, tools, _ = setup
    with pytest.raises(ValueError):
        tools.bind_command_confinement(lambda *args, **kwargs: None, environment, seconds)
    assert tools._tools.process_factory is None and tools._tools.command_environment is None


def test_a_configured_factory_failure_never_uses_popen(tmp_path, monkeypatch):
    def denied(*args, **kwargs): raise OSError('native sandbox unavailable')
    def legacy(*args, **kwargs): pytest.fail('must not fall back')
    monkeypatch.setattr('compute_plane.managed_process.subprocess.Popen', legacy)
    tools = MediatedTools({'role':'implementer','repositories':{'repo':str(tmp_path)},'writable':{}},
                          process_factory=denied, command_environment={'TMP': 'owned'}, command_seconds=2)
    with pytest.raises(OSError, match='unavailable'):
        tools.command(['fixture.exe'], tmp_path, gate=False)


@pytest.mark.parametrize('unbounded,parent_seconds', [(False, 1000), (True, 1000), (True, float('inf'))])
def test_host_can_remove_only_verification_cap_while_parent_and_command_limits_remain(setup, monkeypatch,
                                                                                   unbounded, parent_seconds):
    import math
    import time
    root, _, tools, control = setup
    control = type(control)(control.binding, control.cancellation, time.monotonic()+parent_seconds)
    seen = []
    def runner(argv, **kwargs):
        seen.append(kwargs)
        Path(kwargs['dest'], 'stdout.txt').write_bytes(b'verified')
        Path(kwargs['dest'], 'stderr.txt').write_bytes(b'')
        return dict(exit_code=0, cleanup_verified=True, cancelled=False, timeout=False, output_limit=False)
    monkeypatch.setattr('compute_plane.managed_process.run_process', runner)
    tools.bind_command_confinement(lambda *a, **kw: None, {'TMP': 'owned'}, 3,
        verification_seconds=300, unbounded_verification=unbounded)
    assert call(tools, control, 'verify', name='unit').status == 'succeeded'
    if not unbounded:
        assert seen[0]['timeout'] == 300
    elif math.isfinite(parent_seconds):
        assert 990 < seen[0]['timeout'] <= 1000  # enclosing deadline still wins
    else:
        assert math.isinf(seen[0]['timeout'])
    tools._tools.command(['ordinary-command'], root, gate=False, read_only=True)
    assert seen[-1]['timeout'] == 3  # no change to ordinary command confinement
    assert all(v['process_factory'] is tools._tools.process_factory for v in seen)


def test_unbounded_verification_still_receives_operator_cancellation(setup, monkeypatch):
    import math
    _, _, tools, control = setup
    control = type(control)(control.binding, control.cancellation, math.inf)
    def runner(argv, **kwargs):
        control.cancellation.cancel()
        assert kwargs['cancelled']()
        Path(kwargs['dest'], 'stdout.txt').write_bytes(b'')
        Path(kwargs['dest'], 'stderr.txt').write_bytes(b'')
        return dict(exit_code=None, cleanup_verified=True, cancelled=True, timeout=False, output_limit=False)
    monkeypatch.setattr('compute_plane.managed_process.run_process', runner)
    tools.bind_command_confinement(lambda *a, **kw: None, {'TMP': 'owned'}, 3,
                                  verification_seconds=300, unbounded_verification=True)
    result = call(tools, control, 'verify', name='unit')
    assert result.status != 'succeeded'
    assert json.loads(result.output_json)['process']['cancelled']


@pytest.mark.parametrize('value', [None, 1, 'yes'])
def test_unbounded_verification_requires_an_explicit_host_boolean(setup, value):
    _, _, tools, _ = setup
    with pytest.raises(ValueError, match='unbounded_verification_must_be_bool'):
        tools.bind_command_confinement(lambda *a, **kw: None, {'TMP': 'owned'}, 3,
                                      unbounded_verification=value)
    assert tools._tools.process_factory is None
