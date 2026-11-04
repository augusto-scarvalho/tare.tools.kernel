import copy
import json
from types import SimpleNamespace
import sys

import pytest

from compute_plane.choice_assessment import request_digest, run_assessment, validate_answer
from compute_plane.laya_choice_worker import check_envelope, LayaChoiceModel


REQUEST = {'state': 'Fix one spelling error.', 'question': 'Which task?',
           'options': {'bounded': 'Mechanical change.', 'reasoning': 'Complex reasoning.'}}
ANSWER = {'choice': 'bounded', 'probabilities': {'bounded': 0.8, 'reasoning': 0.2},
          'usage': {'input_tokens': 20, 'output_tokens': 0}, 'model_identity': {'test': True}}


def command(answer):
    return [sys.executable, '-c', 'print(' + repr(json.dumps(answer)) + ')']


def test_bounded_worker_receipt():
    result = run_assessment(command(ANSWER), REQUEST)
    assert result['status'] == 'OBSERVED'
    assert result['answer'] == ANSWER
    assert result['inference_attempts'] == 1 and result['elapsed_ms'] > 0
    assert result['authority'] == 'NONE'


def test_option_order_is_part_of_input_identity():
    reordered = {**REQUEST, 'options': dict(reversed(list(REQUEST['options'].items())))}
    assert request_digest(REQUEST) != request_digest(reordered)


@pytest.mark.parametrize('field,value', [('choice', []), ('choice', 'other'),
    ('probabilities', {'bounded': 0.8}), ('probabilities', {'bounded': float('nan'), 'reasoning': 0.2}),
    ('probabilities', {'bounded': 0.3, 'reasoning': 0.7}),
    ('usage', {'input_tokens': True, 'output_tokens': 0}), ('model_identity', {})])
def test_reject_invalid_answer(field, value):
    with pytest.raises(ValueError):
        validate_answer({**ANSWER, field: value}, REQUEST)


def test_timeout_is_finite_and_not_retried():
    result = run_assessment([sys.executable, '-c', 'import time; time.sleep(2)'], REQUEST, timeout_seconds=0.05)
    assert result['reason'] == 'ASSESSMENT_TIMEOUT' and result['inference_attempts'] == 1


def test_input_overflow_never_launches_worker():
    result = run_assessment(command(ANSWER), {**REQUEST, 'state': 'x' * 8193})
    assert result['reason'] == 'ASSESSMENT_INPUT_OVERFLOW' and result['inference_attempts'] == 0


@pytest.mark.parametrize('value', [0, 61, float('inf'), True])
def test_invalid_budget(value):
    result = run_assessment(command(ANSWER), REQUEST, timeout_seconds=value)
    assert result['inference_attempts'] == 0


def test_output_overflow_and_failure():
    result = run_assessment([sys.executable, '-c', "print('x'*65537)"], REQUEST)
    assert result['reason'] == 'ASSESSMENT_OUTPUT_OVERFLOW'
    assert run_assessment(command({'failure': 'ASSESSMENT_INPUT_OVERFLOW'}), REQUEST)['reason'] == 'ASSESSMENT_INPUT_OVERFLOW'
    assert run_assessment(command({'failure': 'ASSESSMENT_INSUFFICIENT_RAM'}), REQUEST)['reason'] == 'ASSESSMENT_INSUFFICIENT_RAM'
    assert run_assessment(command({'failure': 'ASSESSMENT_CPU_NOT_ADMITTED'}), REQUEST)['reason'] == 'ASSESSMENT_CPU_NOT_ADMITTED'
    assert run_assessment(command('bad'), REQUEST)['reason'] == 'ASSESSMENT_ANSWER_INVALID'


def test_nonzero_worker_preserves_controlled_failure_and_transport_evidence():
    source = ('import sys; print(\'{"failure":"ASSESSMENT_TIMEOUT","stage":"CPU_SCORING"}\'); '
              'sys.stderr.write("private task text"); sys.exit(124)')
    result = run_assessment([sys.executable, '-c', source], REQUEST)
    assert result['status'] == 'FAILED' and result['reason'] == 'ASSESSMENT_TIMEOUT'
    assert result['worker_exit_code'] == 124 and result['worker_stage'] == 'CPU_SCORING'
    assert result['worker_stderr_bytes'] == len('private task text')
    assert 'private task text' not in json.dumps(result)
    assert result['inference_attempts'] == 1 and 'answer' not in result


@pytest.mark.parametrize('value', [ANSWER, {'failure': 'private task text'}, {'failure': []}])
def test_nonzero_exit_never_promotes_stdout_to_success(value):
    argv = command(value)
    argv[-1] += '; raise SystemExit(9)'
    result = run_assessment(argv, REQUEST)
    assert result['reason'] == 'ASSESSMENT_WORKER_FAILED'
    assert result['worker_exit_code'] == 9 and 'answer' not in result
    assert 'private task text' not in json.dumps(result)


def test_worker_reported_failure_is_not_misclassified_as_invalid_answer():
    result = run_assessment(command({'failure': 'ASSESSMENT_WORKER_FAILED'}), REQUEST)
    assert result['reason'] == 'ASSESSMENT_WORKER_FAILED'
    assert result['worker_exit_code'] == 0


def test_crashed_worker_without_json_retains_exit_code():
    result = run_assessment([sys.executable, '-c', 'raise SystemExit(124)'], REQUEST)
    assert result['reason'] == 'ASSESSMENT_WORKER_FAILED'
    assert result['worker_exit_code'] == 124 and result['worker_stdout_bytes'] == 0


class Tokenizer:
    mask_token = '[MASK]'
    def __call__(self, text, **kwargs):
        return {'input_ids': text.split()}


@pytest.mark.parametrize('fault', ['state', 'option', 'head', 'mask'])
def test_sdk_silent_truncation_is_refused(fault):
    request = copy.deepcopy(REQUEST)
    agent = SimpleNamespace(tok=Tokenizer(), cfg={'max_len': 100, 'head_max_len': 60})
    check_envelope(agent, request)
    if fault == 'state': request['state'] = 'word ' * 101
    elif fault == 'option': request['options']['bounded'] = 'word ' * 50
    elif fault == 'head': request['question'] = 'word ' * 61
    else: request['state'] += ' [MASK]'
    with pytest.raises(ValueError, match='INPUT_OVERFLOW'):
        check_envelope(agent, request)


def test_checkpoint_refuses_before_optional_import(tmp_path):
    manifest = tmp_path / 'checkpoint.json'
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='CHECKPOINT_CHANGED'):
        LayaChoiceModel(manifest)
