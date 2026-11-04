"""Bounded choice observations. Callers retain all routing authority."""
import hashlib
import json
import math
import subprocess
import tempfile
import time


WORKER_FAILURES = frozenset({
    'ASSESSMENT_INPUT_OVERFLOW', 'ASSESSMENT_CHECKPOINT_CHANGED',
    'ASSESSMENT_INSUFFICIENT_RAM', 'ASSESSMENT_RESOURCE_UNAVAILABLE',
    'ASSESSMENT_EVIDENCE_INSUFFICIENT', 'ASSESSMENT_WORKER_FAILED',
    'ASSESSMENT_TIMEOUT', 'ASSESSMENT_WORKER_UNAVAILABLE',
    'ASSESSMENT_ENVIRONMENT_CHANGED', 'ASSESSMENT_BACKEND_UNAVAILABLE',
    'ASSESSMENT_CANDIDATE_TOKEN_INVALID', 'ASSESSMENT_INCOMPLETE_READOUT',
    'ASSESSMENT_MODIFIED_READOUT', 'ASSESSMENT_MISSING_CANDIDATE',
    'ASSESSMENT_CANCELLED', 'ASSESSMENT_RESOURCE_BUSY',
    'ASSESSMENT_CAPACITY_EXCEEDED',
    'ASSESSMENT_CPU_NOT_ADMITTED',
    'ASSESSMENT_EXECUTION_CONFIG_INVALID',
})


# Methods that read one generated token's alternatives; they honestly report one output token.
FIRST_TOKEN_METHODS = frozenset({'litjev-native-first-token-v1', 'litjev-ninfer-chat-first-token-v1'})


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def request_digest(request):
    # Option order affects this model's input and must survive canonical hashing.
    return digest({**request, 'options': list(request['options'].items())})


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {'state', 'question', 'options'}:
        raise ValueError('ASSESSMENT_REQUEST_INVALID')
    if any(not isinstance(request[k], str) or not request[k].strip() for k in ('state', 'question')):
        raise ValueError('ASSESSMENT_REQUEST_INVALID')
    options = request['options']
    if (not isinstance(options, dict) or not 2 <= len(options) <= 6
            or any(not isinstance(k, str) or not k or len(k) > 32
                   or not isinstance(v, str) or not v for k, v in options.items())):
        raise ValueError('ASSESSMENT_OPTIONS_INVALID')
    if len(json.dumps(request, ensure_ascii=False).encode()) > 8192:
        raise ValueError('ASSESSMENT_INPUT_OVERFLOW')


def validate_answer(answer, request):
    if (not isinstance(answer, dict) or not isinstance(answer.get('choice'), str)
            or answer['choice'] not in request['options']):
        raise ValueError('ASSESSMENT_ANSWER_INVALID')
    probs = answer.get('probabilities')
    if (not isinstance(probs, dict) or set(probs) != set(request['options'])
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in probs.values()) or abs(sum(probs.values()) - 1) > 0.001
            or probs[answer['choice']] != max(probs.values())):
        raise ValueError('ASSESSMENT_DISTRIBUTION_INVALID')
    usage = answer.get('usage')
    if (not isinstance(usage, dict) or type(usage.get('input_tokens')) is not int
            or usage['input_tokens'] < 1 or type(usage.get('output_tokens')) is not int
            or usage['output_tokens'] not in (0, 1)):
        raise ValueError('ASSESSMENT_USAGE_INVALID')
    if usage['output_tokens'] == 1 and (not isinstance(answer.get('model_identity'), dict)
            or answer['model_identity'].get('method') not in FIRST_TOKEN_METHODS):
        raise ValueError('ASSESSMENT_USAGE_INVALID')
    if not isinstance(answer.get('model_identity'), dict) or not answer['model_identity']:
        raise ValueError('ASSESSMENT_IDENTITY_MISSING')
    # Raw probabilities and entropy confidence are not calibrated success rates.
    return {k: answer[k] for k in ('choice', 'probabilities', 'usage', 'model_identity')}


def run_assessment(command, request, *, timeout_seconds=30):
    """Run an operator-configured worker once, with no shell or automatic retries.

    Worker stdout is a single bounded JSON answer. The command comes from trusted
    configuration, never from task text or the model. Temporary output avoids
    buffering an unbounded worker response in the parent process.
    """
    started = time.monotonic()
    receipt = {'schema': 'tare.kernel/choice-assessment/1', 'status': 'FAILED',
               'authority': 'NONE', 'inference_attempts': 0}
    try:
        validate_request(request)
        receipt['request_sha256'] = request_digest(request)
        if (not isinstance(command, list) or not command
                or any(not isinstance(a, str) or not a or '\x00' in a for a in command)
                or type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
            raise ValueError('ASSESSMENT_EXECUTION_CONFIG_INVALID')
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            receipt['inference_attempts'] = 1
            try:
                result = subprocess.run(command, input=json.dumps(request, ensure_ascii=False).encode(), stdout=output,
                                        stderr=errors, timeout=timeout_seconds, check=False)
                receipt['worker_exit_code'] = result.returncode
            finally:
                # Keep transport evidence without copying arbitrary stderr (which
                # can contain task text or credentials) into user-facing receipts.
                receipt['worker_stderr_bytes'] = errors.seek(0, 2)
                receipt['worker_stdout_bytes'] = output.seek(0, 2)
            output.seek(0)
            raw = output.read(65537)
        if len(raw) > 65536:
            raise ValueError('ASSESSMENT_OUTPUT_OVERFLOW')
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ValueError('ASSESSMENT_WORKER_FAILED' if result.returncode
                             else 'ASSESSMENT_OUTPUT_INVALID') from None
        if (isinstance(value, dict) and isinstance(value.get('failure'), str)
                and value['failure'] in WORKER_FAILURES):
            if value.get('stage') in ('CPU_INPUT_PREFLIGHT', 'CPU_MODEL_LOAD', 'CPU_SCORING', 'RESIDENT_SCORING'):
                receipt['worker_stage'] = value['stage']
            raise ValueError(value['failure'])
        if result.returncode or isinstance(value, dict) and 'failure' in value:
            raise ValueError('ASSESSMENT_WORKER_FAILED')
        receipt.update(status='OBSERVED', answer=validate_answer(value, request))
    except subprocess.TimeoutExpired:
        receipt['reason'] = 'ASSESSMENT_TIMEOUT'
        receipt['timeout_seconds'] = timeout_seconds
    except OSError:
        receipt['reason'] = 'ASSESSMENT_WORKER_UNAVAILABLE'
    except ValueError as exc:
        receipt['reason'] = str(exc)
    receipt['elapsed_ms'] = round((time.monotonic() - started) * 1000, 3)
    return receipt
