import copy

import pytest

from compute_plane.model_facts import gpu_availability


def observed(held=False, kind=None):
    return {'status': 'OBSERVED', 'observed_at': 100, 'expires_at': 105, 'source': 'http://fixture/v1/fleet/status',
            'value': {'status': 'ok', 'role': 'qualified-model-gateway', 'current_model': None,
                      'available_models': ['qwen38'], 'gpu_coordination': {
                          'enabled': True, 'held': held, 'owner': {'kind': kind, 'nonce': 'never-export-this'}}}}


@pytest.mark.parametrize('kind', ['image', None, 'other'])
def test_a_lease_held_by_an_image_or_unknown_owner_is_not_immediate_capacity(kind):
    result = gpu_availability(observed(True, kind), 'qwen38', now=102)
    assert result['status'] == 'BUSY' and result['immediate'] is False
    assert 'never-export' not in str(result)


def test_text_behind_text_is_admitted_as_queued():
    row = observed(True, 'text'); row['value']['current_model'] = 'qwen38'
    result = gpu_availability(row, 'qwen38', now=102)
    assert result['status'] == 'QUEUED' and result['immediate'] is True and result['queued'] is True
    assert result['resident'] is True and 'never-export' not in str(result)


@pytest.mark.parametrize('held', [False, True])
def test_an_image_job_waiting_makes_text_busy_even_before_it_holds(held):
    row = observed(held, 'text' if held else None)
    row['value']['gpu_coordination']['queue'] = [{'kind': 'text'}, {'kind': 'image', 'reason': 'cohort-26'}]
    result = gpu_availability(row, 'qwen38', now=102)
    assert result['status'] == 'BUSY' and result['reason'] == 'GPU_IMAGE_WAITING' and result['immediate'] is False


def test_free_gpu_is_eligible_but_does_not_assert_loaded_model():
    result = gpu_availability(observed(), 'qwen38', now=102)
    assert result['immediate'] is True and result['loading_required'] is True
    assert result['resident'] is False


@pytest.mark.parametrize('fault', ['expired', 'future', 'disabled', 'error', 'unknown', 'wrong-gateway', 'malformed'])
def test_stale_or_unverifiable_gpu_never_means_free(fault):
    row = observed(); now = 102
    if fault == 'expired': now = 106
    elif fault == 'future': now = 99
    elif fault == 'disabled': row['value']['gpu_coordination']['enabled'] = False
    elif fault == 'error': row['value']['gpu_coordination']['error'] = 'fcntl_unavailable'
    elif fault == 'unknown': row['status'] = 'UNKNOWN'
    elif fault == 'wrong-gateway': row['value']['role'] = 'ordinary-llama-server'
    else: row['value']['gpu_coordination'] = None
    assert gpu_availability(row, 'qwen38', now=now)['immediate'] is False


def test_unadvertised_model_is_unavailable():
    assert gpu_availability(observed(), 'other', now=102)['reason'] == 'GPU_MODEL_NOT_ADVERTISED'


@pytest.mark.parametrize('eta,shown', [(240, 240), (-1, None), ('soon', None), (float('nan'), None)])
def test_a_busy_gpu_carries_the_nodes_estimate_of_the_wait(eta, shown):
    # Contract gpu-lease/1: the caller decides to wait or go elsewhere from how long the GPU will be away.
    row = observed(True, 'image'); row['value']['gpu_coordination']['eta_seconds'] = eta
    assert gpu_availability(row, 'qwen38', now=102).get('eta_seconds') == shown
    free = observed(); free['value']['gpu_coordination']['eta_seconds'] = 240
    assert 'eta_seconds' not in gpu_availability(free, 'qwen38', now=102)


def test_a_named_reservation_is_busy_for_everyone_but_its_holder_with_its_time_left():
    # Contract gpu-lease/2: the gateway holds the GPU for one holder; a facts reader is not that holder.
    row = observed(True, 'text'); coord = row['value']['gpu_coordination']
    coord['reservation'] = {'id': 'r1', 'name': 'bench', 'holder': 'acer', 'fence': 3, 'seconds_left': 5400}
    coord['eta_seconds'] = 5400
    result = gpu_availability(row, 'qwen38', now=102)
    assert result['status'] == 'BUSY' and result['reason'] == 'GPU_RESERVED' and result['immediate'] is False
    assert result['reservation'] == 'bench' and result['eta_seconds'] == 5400
