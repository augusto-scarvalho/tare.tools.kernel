import dataclasses
import math

import pytest

from compute_plane.action_control import ActionAuthorization, ActionRefused, ActionSnapshot, ExactActionGuard

H = 'a' * 64
H2 = 'b' * 64


def make(**overrides):
    fields = dict(
        work_id='w1', session_id='s1', effect_id='e1', binding_id='b1', fence_seq=3,
        operation='run', repository='repo', cwd='C:/work', environment_sha256=H,
        context_sha256=H, source_sha256=H, policy_sha256=H, grant_sha256=H,
        execution_profile='default', deadline=12.5, argv=('python', '-m', 'pytest'),
        payload_json='[1]',
    )
    fields.update(overrides)
    return ActionSnapshot(**fields)


def test_digest_is_deterministic_and_snapshot_is_immutable():
    snap = make()
    assert snap.action_sha256 == make().action_sha256
    assert len(snap.action_sha256) == 64
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.work_id = 'other'


@pytest.mark.parametrize('change', [
    {'work_id': 'w2'}, {'cwd': 'C:/other'}, {'fence_seq': 4}, {'deadline': 13.0},
    {'environment_sha256': H2}, {'context_sha256': H2}, {'source_sha256': H2},
    {'policy_sha256': H2}, {'grant_sha256': H2}, {'execution_profile': 'strict'},
    {'argv': ('python', '-m', 'pytest ')}, {'argv': None, 'script': b'echo 1'},
    {'argv': None, 'stdin': b'x'}, {'payload_json': '[1.0]'}, {'payload_json': ' [1]'},
])
def test_changed_field_changes_digest(change):
    assert make(**change).action_sha256 != make().action_sha256


@pytest.mark.parametrize('change', [
    {'fence_seq': True}, {'fence_seq': -1}, {'deadline': True}, {'deadline': math.nan},
    {'deadline': math.inf}, {'source_sha256': 'A' * 64}, {'policy_sha256': 'b' * 63},
    {'work_id': ''}, {'argv': ['python', '-m']}, {'argv': ('python', 'x' + chr(0))},
    {'argv': ('python',), 'script': b'echo'}, {'argv': None, 'script': 'echo'},
    {'argv': None, 'stdin': 'x'}, {'payload_json': '['}, {'payload_json': '[NaN]'},
    {'payload_json': '[Infinity]'},
])
def test_invalid_snapshot_is_refused(change):
    with pytest.raises(ValueError):
        make(**change)


@pytest.mark.parametrize('payload', ['[1e309]', '[-1e309]', '{"a": [1e309]}'])
def test_overflowing_float_payload_is_refused(payload):
    with pytest.raises(ValueError):
        make(payload_json=payload)


@pytest.mark.parametrize('payload', ['[1e308]', '[-1e-308]', '{"a": [0.5, 2E3]}', '[1.5]'])
def test_finite_float_payload_is_accepted_verbatim(payload):
    snap = make(payload_json=payload)
    assert snap.payload_json == payload
    assert len(snap.action_sha256) == 64

