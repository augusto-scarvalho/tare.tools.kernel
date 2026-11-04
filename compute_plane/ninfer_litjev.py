"""LitJEV's first-token method through a NInfer Chat Completions endpoint.

Current NInfer uses the OpenAI `logprobs=true` contract. The older 0.11 server,
screened on 2026-09-29 with --first-token-logprobs, rejects that field before
inference and accepts only `top_logprobs`. Only its explicit validation error
permits repeating the request without `logprobs`, within the original timeout.
Thinking is off and one token is generated; the answer is the restricted softmax
over the option codes, not the sampled token.
The caller decides the model is resident and idle; this module only asks.
"""
import json
import math
import time
import urllib.error
import urllib.request

from .choice_assessment import validate_answer, validate_request

METHOD = 'litjev-ninfer-chat-first-token-v1'
SYSTEM = ('Evaluate the state using the question and labeled options that follow. '
          'Return only the option code. Do not explain or reason aloud.')


def body(request, model):
    keys = list(request['options'])
    question = 'Question: ' + json.dumps({'type': 'choice', 'instructions': request['question'], 'options': [
        {'code': code, 'option': key, 'description': request['options'][key]}
        for code, key in zip('ABCDEF', keys)]}, ensure_ascii=False, allow_nan=False) + '\nAnswer:'
    return {'model': model, 'max_tokens': 1, 'temperature': 1.0, 'top_p': 1.0, 'logprobs': True,
            'top_logprobs': 20, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': request['state'] + '\n\n' + question}]}


def answer(request, response):
    keys = list(request['options'])
    codes = dict(zip('ABCDEF', keys))
    top = response['choices'][0]['logprobs']['content'][0]['top_logprobs']
    scores = {}
    for entry in top:  # 'A' and ' A' are the same answer: sum them
        code = entry['token'].strip()
        if code in codes and isinstance(entry.get('logprob'), (int, float)) and math.isfinite(entry['logprob']):
            scores[code] = math.log(math.exp(scores[code]) + math.exp(entry['logprob'])) if code in scores \
                else entry['logprob']
    if not scores:
        raise ValueError('ASSESSMENT_MISSING_CANDIDATE')
    peak = max(scores.values())
    weights = {codes[c]: math.exp(scores[c] - peak) for c in scores}
    total = sum(weights.values())
    probabilities = {key: weights.get(key, 0.0) / total for key in keys}  # outside the top 20: below every listed one
    return validate_answer({'choice': max(probabilities, key=probabilities.get), 'probabilities': probabilities,
                            'usage': {'input_tokens': max(1, (response.get('usage') or {}).get('prompt_tokens') or 1),
                                      'output_tokens': 1},
                            'model_identity': {'model': response.get('model'), 'method': METHOD,
                                               'score_semantics': 'restricted-next-token-not-calibrated-success'}},
                           request)


def _legacy_refusal(error):
    """Recognize only the old server's bounded, pre-inference validation error."""
    if error.code != 400:
        return False
    try:
        raw = error.read(4097)
        if len(raw) > 4096:
            return False
        value = json.loads(raw)
        detail = value.get('error') if isinstance(value, dict) else None
        return (isinstance(detail, dict) and detail.get('type') == 'invalid_request_error'
                and detail.get('code') == 'logprobs_not_supported' and detail.get('param') == 'logprobs')
    except (OSError, ValueError):
        return False


def assess(request, *, url, model, timeout=10):
    """One question to `url` (…/v1/chat/completions); ValueError with an ASSESSMENT_* reason on failure."""
    validate_request(request)
    payload, deadline = body(request, model), time.monotonic() + timeout
    for modern in (True, False):
        if not modern:
            payload.pop('logprobs')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError('ASSESSMENT_BACKEND_UNAVAILABLE')
        call = urllib.request.Request(url, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(call, timeout=remaining) as reply:
                response = json.loads(reply.read(1 << 20))
            break
        except urllib.error.HTTPError as exc:
            try:
                legacy = modern and _legacy_refusal(exc)
            finally:
                exc.close()
            if legacy:
                continue
            raise ValueError('ASSESSMENT_BACKEND_UNAVAILABLE' if exc.code in (400, 404) else 'ASSESSMENT_WORKER_FAILED') from None
        except (urllib.error.URLError, TimeoutError):
            raise ValueError('ASSESSMENT_BACKEND_UNAVAILABLE') from None
    try:
        return answer(request, response)
    except (KeyError, IndexError, TypeError):
        raise ValueError('ASSESSMENT_INCOMPLETE_READOUT') from None
