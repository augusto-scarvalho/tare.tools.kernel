"""Optional hosted Jev (TypeSafe System One) choice adapter. Sends the request off-host.

Key from TYPESAFE_API_KEY; model pinned so tuned thresholds survive alias moves
(override with TYPESAFE_DEFAULT_MODEL). One HTTP call, no retries: callers own fallback.
"""
import json
import math
import os
import sys
import urllib.error
import urllib.request

from .choice_assessment import validate_answer, validate_request

MODEL = 'jev-1.13.0'
BASE_URL = 'https://api.typesafe.ai'
BUSY = {429, 529}


def body(request, model):
    return {'model': model, 'state': request['state'],
            'questions': {'q': {'type': 'choice', 'instructions': request['question'],
                                'criteria': request['options']}}}


def answer(request, response):
    """Map Jev's reply onto the kernel answer; confidence is left to callers' own calibration."""
    reply = response['answers']['q']
    if reply.get('type') != 'choice':
        raise ValueError('ASSESSMENT_WORKER_FAILED')
    probabilities = reply['probabilities']
    # Jev rounds; the top option must stay the reported choice after renormalising.
    total = sum(probabilities.values()) if set(probabilities) == set(request['options']) else 0
    if not total:
        raise ValueError('ASSESSMENT_WORKER_FAILED')
    probabilities = {k: probabilities[k] / total for k in request['options']}
    choice = max(probabilities, key=probabilities.get)
    # Output tokens are not generated text and are not billed; input tokens are the cost.
    return validate_answer({'choice': choice, 'probabilities': probabilities,
                            'usage': {'input_tokens': response['usage']['input_tokens'], 'output_tokens': 0},
                            'model_identity': {'model': response['model'], 'method': 'jev-systemone-choice-v1',
                                               'hosted': 'api.typesafe.ai'}}, request)


def post(payload, *, key, base_url, timeout):
    call = urllib.request.Request(base_url.rstrip('/') + '/v1/systemone', method='POST',
                                  data=json.dumps(payload).encode(),
                                  headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(call, timeout=timeout) as reply:
            return json.loads(reply.read(1 << 20))
    except urllib.error.HTTPError as exc:
        if exc.code == 400 and b'max_tokens_exceeded' in exc.read(4096):
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW') from None  # over the model's window (about 32 k tokens)
        raise ValueError('ASSESSMENT_RESOURCE_BUSY' if exc.code in BUSY else
                         'ASSESSMENT_BACKEND_UNAVAILABLE' if exc.code in (401, 403) else
                         'ASSESSMENT_WORKER_FAILED') from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError('ASSESSMENT_BACKEND_UNAVAILABLE') from None


def assess(request, *, key, model=MODEL, base_url=BASE_URL, timeout=25):
    validate_request(request)
    return answer(request, post(body(request, model), key=key, base_url=base_url, timeout=timeout))


# Measured 2026-10-04: a request costs about 300 input tokens before any content, a Noul question about 17 more,
# and 160 questions over one state answered in 0.66 s; the state is paid once per request.
MAX_QUESTIONS = 160


def assess_many(state, questions, *, key, model=MODEL, base_url=BASE_URL, timeout=25):
    """Questions over one shared state, in one request: ({name: answer}, input tokens). A question is either yes/no
    instructions (a Noul; its answer is the probability of yes) or {'question': instructions, 'options': {key:
    description}} (a Choice of 2 to 255 options; its answer is {'choice', 'probabilities'}, renormalised as assess
    does). ValueError with an ASSESSMENT_* reason on failure; no retries."""
    def wire(value):
        if isinstance(value, str) and value.strip():
            return {'type': 'noul', 'instructions': value}
        if (isinstance(value, dict) and set(value) == {'question', 'options'} and isinstance(value['question'], str)
                and value['question'].strip() and isinstance(value['options'], dict)
                and 2 <= len(value['options']) <= 255
                and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in value['options'].items())):
            return {'type': 'choice', 'instructions': value['question'], 'criteria': value['options']}
        raise ValueError('ASSESSMENT_REQUEST_INVALID')
    if (not isinstance(state, str) or not state.strip() or not isinstance(questions, dict)
            or not 1 <= len(questions) <= MAX_QUESTIONS
            or any(not isinstance(k, str) or not k or len(k) > 32 for k in questions)):
        raise ValueError('ASSESSMENT_REQUEST_INVALID')
    asked = {k: wire(v) for k, v in questions.items()}
    response = post({'model': model, 'state': state, 'questions': asked}, key=key, base_url=base_url, timeout=timeout)
    if response.get('model') != model:
        raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
    answers = {}
    try:
        for k, value in asked.items():
            reply = response['answers'][k]
            if value['type'] == 'noul':
                answers[k] = float(reply['noul'])
                if not math.isfinite(answers[k]) or not 0 <= answers[k] <= 1:
                    raise ValueError('ASSESSMENT_MODIFIED_READOUT')
                continue
            probabilities = reply['probabilities']
            total = sum(probabilities.values()) if set(probabilities) == set(value['criteria']) else 0
            if not total or any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
                raise ValueError('ASSESSMENT_MODIFIED_READOUT')
            probabilities = {o: probabilities[o]/total for o in value['criteria']}
            answers[k] = {'choice': max(probabilities, key=probabilities.get), 'probabilities': probabilities}
        tokens = int(response['usage']['input_tokens'])
    except (KeyError, TypeError, AttributeError):
        raise ValueError('ASSESSMENT_INCOMPLETE_READOUT') from None
    return answers, tokens


def main():
    from .choice_assessment import WORKER_FAILURES
    try:
        raw = sys.stdin.buffer.read(8193)
        if len(raw) > 8192:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        key = os.environ.get('TYPESAFE_API_KEY', '').strip()  # pasted keys often carry a newline
        if not key:
            raise ValueError('ASSESSMENT_BACKEND_UNAVAILABLE')
        result = assess(json.loads(raw), key=key, model=os.environ.get('TYPESAFE_DEFAULT_MODEL') or MODEL,
                        base_url=os.environ.get('TYPESAFE_BASE_URL') or BASE_URL)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        result = {'failure': str(exc) if str(exc) in WORKER_FAILURES else 'ASSESSMENT_WORKER_FAILED'}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
