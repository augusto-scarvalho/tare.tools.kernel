"""Single-question LitJEV method on a caller-admitted resident llama backend.

The serving owner must hold its GPU lease for this entire function. This module
does not load/switch models or acquire execution authority. One transport token
is counted honestly; its sampled text is never interpreted as the decision.
"""
import json
import math

from .choice_assessment import validate_answer, validate_request

METHOD = 'litjev-native-first-token-v1'
UPSTREAM = 'https://github.com/zhengxuyu/litjev/tree/e7fb109a7466da9709028eb9c4e9f16eaeb4e2a3'


def assess(request, post, identity):
    validate_request(request)
    keys = list(request['options'])
    codes = list('ABCDEF')[:len(keys)]
    messages = [{'role': 'system', 'content':
        'Evaluate the state using the question and labeled options that follow. '
        'Return only the option code. Do not explain or reason aloud.'},
        {'role': 'user', 'content': request['state']}]
    prefix = post('/apply-template', {'messages': messages, 'add_generation_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    suffix = 'Question: ' + json.dumps({'type': 'choice', 'instructions': request['question'],
        'options': [{'code': code, 'option': key, 'description': request['options'][key]}
                    for code, key in zip(codes, keys)]}, ensure_ascii=False, allow_nan=False) + '\nAnswer:'

    def tokenize(text):
        return post('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']

    base = tokenize(suffix)
    candidates = []
    for code in codes:
        appended = tokenize(suffix + ' ' + code)
        if len(appended) != len(base)+1 or appended[:-1] != base:
            raise ValueError('ASSESSMENT_CANDIDATE_TOKEN_INVALID')
        candidates.append(appended[-1])
    if len(set(candidates)) != len(candidates):
        raise ValueError('ASSESSMENT_CANDIDATE_TOKEN_INVALID')
    tokens = tokenize(prefix) + base
    if not 1 <= len(tokens) <= 4096:
        raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
    raw = post('/completion', {'prompt': tokens, 'n_predict': 1, 'n_probs': 128,
        'post_sampling_probs': False, 'temperature': 1.0, 'top_k': 0, 'top_p': 1.0,
        'min_p': 0.0, 'repeat_penalty': 1.0, 'presence_penalty': 0.0,
        'frequency_penalty': 0.0, 'cache_prompt': False, 'return_tokens': True,
        'stream': False, 'seed': 219, 'speculative.n_max': 0})
    rows = raw.get('completion_probabilities', [])
    if (raw.get('truncated') is not False or raw.get('tokens_evaluated') != len(tokens)
            or raw.get('tokens_predicted') != 1 or len(rows) != 1):
        raise ValueError('ASSESSMENT_INCOMPLETE_READOUT')
    settings = raw.get('generation_settings', {})
    if settings.get('post_sampling_probs') is not False or settings.get('logit_bias') or settings.get('grammar'):
        raise ValueError('ASSESSMENT_MODIFIED_READOUT')
    entries = rows[0].get('top_logprobs', [])
    scores = {entry['id']: entry['logprob'] for entry in entries}
    if len(scores) != len(entries) or any(t not in scores or type(scores[t]) not in (int, float)
            or not math.isfinite(scores[t]) for t in candidates):
        raise ValueError('ASSESSMENT_MISSING_CANDIDATE')
    peak = max(scores[t] for t in candidates)
    weights = [math.exp(scores[t]-peak) for t in candidates]
    probabilities = dict(zip(keys, (w/sum(weights) for w in weights)))
    return validate_answer({'choice': max(probabilities, key=probabilities.get),
        'probabilities': probabilities, 'usage': {'input_tokens': len(tokens), 'output_tokens': 1},
        'model_identity': {**identity, 'method': METHOD, 'upstream': UPSTREAM,
            'score_semantics': 'restricted-next-token-not-calibrated-success'}}, request)
