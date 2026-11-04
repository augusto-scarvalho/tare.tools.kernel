import copy
import pytest
from compute_plane.choice_assessment import validate_answer
from compute_plane.litjev_choice import assess

REQUEST = {'state': 'fixture', 'question': 'choose', 'options': {'a': 'first', 'b': 'second'}}


class Backend:
    def __init__(self, fault=None):
        self.fault = fault
        self.calls = []

    def __call__(self, path, data):
        self.calls.append(path)
        if path == '/apply-template':
            return {'prompt': 'prefix'}
        if path == '/tokenize':
            text = data['content']
            if text == 'prefix':
                return {'tokens': [1]*4097 if self.fault == 'overflow' else [1]}
            if text.endswith(' A'):
                return {'tokens': [2, 3, 10]}
            if text.endswith(' B'):
                return {'tokens': [2, 3, 11, 12] if self.fault == 'split' else [2, 3, 11]}
            return {'tokens': [2, 3]}
        result = {'truncated': False, 'tokens_evaluated': 3, 'tokens_predicted': 1,
            'content': 'deliberately not the winning code',
            'generation_settings': {'post_sampling_probs': False, 'logit_bias': [], 'grammar': ''},
            'completion_probabilities': [{'top_logprobs': [
                {'id': 10, 'logprob': -1002.0}, {'id': 11, 'logprob': -1000.0}]}]}
        if self.fault == 'missing': result['completion_probabilities'][0]['top_logprobs'].pop()
        if self.fault == 'nan': result['completion_probabilities'][0]['top_logprobs'][0]['logprob'] = float('nan')
        if self.fault == 'truncate': result['truncated'] = True
        if self.fault == 'sampled': result['generation_settings']['post_sampling_probs'] = True
        if self.fault == 'usage': result['tokens_evaluated'] = 2
        return result


def test_full_distribution_not_sampled_text_and_honest_token_usage():
    backend = Backend()
    answer = assess(REQUEST, backend, {'model': 'fixture'})
    assert answer['choice'] == 'b'
    assert answer['probabilities']['b'] == pytest.approx(.88079707798)
    assert answer['usage'] == {'input_tokens': 3, 'output_tokens': 1}
    assert backend.calls.count('/completion') == 1
    invalid = copy.deepcopy(answer); invalid['model_identity'] = {'model': 'other'}
    with pytest.raises(ValueError, match='USAGE'): validate_answer(invalid, REQUEST)
    invalid['model_identity'] = []; invalid['usage']['output_tokens'] = 1
    with pytest.raises(ValueError, match='USAGE'): validate_answer(invalid, REQUEST)


@pytest.mark.parametrize('fault', ['missing', 'nan', 'truncate', 'sampled', 'usage', 'split', 'overflow'])
def test_refuse_incomplete_or_modified_readout(fault):
    backend = Backend(fault)
    with pytest.raises(ValueError): assess(REQUEST, backend, {'model': 'fixture'})
    if fault in {'split', 'overflow'}: assert '/completion' not in backend.calls
