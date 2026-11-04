import io
from contextlib import nullcontext
import json
from types import SimpleNamespace
import sys

import pytest

from compute_plane import openjev_choice_worker as worker

REQUEST = {'state': 'Both pairs meet the quality requirement.', 'question': 'Which pair costs less?',
           'options': {'cheap': 'Pair A costs USD 1.', 'dear': 'Pair B costs USD 3.'}}


def test_entailment_normalization_is_stable_and_not_raw_entailment_logit_ranking():
    # Second hypothesis has a higher raw entailment logit but overwhelming contradiction.
    rows = [{'logits': [0., 10., 0.], 'input_tokens': 10},
            {'logits': [100., 11., 0.], 'input_tokens': 12}]
    result = worker.choice_answer(REQUEST, rows, {'test': True})
    assert result['choice'] == 'cheap'
    assert sum(result['probabilities'].values()) == pytest.approx(1)
    assert result['usage'] == {'input_tokens': 22, 'output_tokens': 0}
    reversed_request = {**REQUEST, 'options': dict(reversed(list(REQUEST['options'].items())))}
    assert worker.choice_answer(reversed_request, list(reversed(rows)), {'test': True})['choice'] == 'cheap'
    shifted = [{**r, 'logits': [v+10000 for v in r['logits']]} for r in rows]
    assert worker.choice_answer(REQUEST, shifted, {'test': True})['probabilities'] == pytest.approx(result['probabilities'])


@pytest.mark.parametrize('logits', [[0, 1], [0, float('nan'), 0], [0, float('inf'), 0], [True, 1, 2]])
def test_invalid_nli_scores_are_not_decisions(logits):
    with pytest.raises(ValueError, match='DISTRIBUTION_INVALID'):
        worker.distribution(logits)


def test_preflight_all_pairs_before_any_inference():
    model = object.__new__(worker.OpenJevChoiceModel)
    calls = []
    def encode(premise, hypothesis):
        calls.append(hypothesis)
        if 'Pair B' in hypothesis:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        return {}
    model.encode = encode
    model.predict_encoded = lambda _: pytest.fail('partial inference')
    with pytest.raises(ValueError, match='INPUT_OVERFLOW'): model.assess(REQUEST)
    assert len(calls) == 2


def test_tokenizer_cannot_silently_truncate():
    model = object.__new__(worker.OpenJevChoiceModel)
    model.max_tokens = 3
    model.model = SimpleNamespace(config=SimpleNamespace(nli_template='Premise: {premise}\nHypothesis: {hypothesis}'))
    def tokenizer(text, **kwargs):
        assert kwargs['truncation'] is False
        assert text == 'Premise: p\nHypothesis: h'
        return {'input_ids': SimpleNamespace(shape=(1, 4))}
    model.tokenizer = tokenizer
    with pytest.raises(ValueError, match='INPUT_OVERFLOW'): model.encode('p', 'h')


def test_artifact_drift_and_label_permutation_rejected_before_import(tmp_path):
    root = tmp_path/'weights'; root.mkdir()
    config = {'architectures': ['Qwen3_5ForSequenceClassification'], 'id2label': worker.LABELS,
              'nli_template': 'Premise: {premise}\nHypothesis: {hypothesis}'}
    for name in worker.FILES:
        (root/name).write_text(json.dumps(config) if name == 'config.json' else 'fixture')
    manifest = {'schema': 'tare.local-labs/openjev-checkpoint/1', 'directory': 'weights',
                'files': {name: worker.sha256_file(root/name) for name in worker.FILES}}
    path = tmp_path/'checkpoint.json'; path.write_text(json.dumps(manifest))
    assert worker.checkpoint(path)[0] == root
    (root/'model.safetensors').write_text('changed')
    with pytest.raises(ValueError, match='CHECKPOINT_CHANGED'): worker.checkpoint(path)
    manifest['files']['model.safetensors'] = worker.sha256_file(root/'model.safetensors')
    config['id2label'] = {'0': 'entailment', '1': 'contradiction', '2': 'neutral'}
    (root/'config.json').write_text(json.dumps(config))
    manifest['files']['config.json'] = worker.sha256_file(root/'config.json')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='CHECKPOINT_CHANGED'): worker.checkpoint(path)


def test_cpu_cli_refuses_low_memory_before_loading_model(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=16*1024**3)))
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(json.dumps(REQUEST).encode())))
    monkeypatch.setattr(worker, 'OpenJevChoiceModel', lambda *a, **kw: pytest.fail('memory gate bypassed'))
    worker.main()
    assert json.loads(capsys.readouterr().out) == {'failure': 'ASSESSMENT_INSUFFICIENT_RAM'}


def test_offloaded_meta_weights_do_not_select_the_input_device():
    model = object.__new__(worker.OpenJevChoiceModel)
    model.input_device = 'cpu'
    model.torch = SimpleNamespace(inference_mode=nullcontext)
    class Logits:
        def float(self): return self
        def cpu(self): return self
        def tolist(self): return [0., 2., 0.]
    class OffloadedModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device='meta'))
        def __call__(self, **kwargs):
            assert kwargs['input_ids'] == 'real-cpu-input'
            return SimpleNamespace(logits=[Logits()])
    model.model = OffloadedModel()
    def move(device):
        assert device == 'cpu', 'must never copy input to the meta placeholder'
        return 'real-cpu-input'
    result = model.predict_encoded({'input_ids': SimpleNamespace(shape=(1, 3), to=move)})
    assert result['nli_probabilities']['entailment'] > .7


def test_cpu_preflight_counts_every_exact_pair_before_weight_loading(tmp_path, monkeypatch):
    for key in ('USE_TF', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.setenv(key, 'fixture-before-preflight')
    root = tmp_path / 'tokenizer'; root.mkdir()
    (root / 'config.json').write_text(json.dumps({'nli_template': 'Premise: {premise}\nHypothesis: {hypothesis}'}))
    calls = []
    def checkpoint(path, **kwargs):
        assert kwargs == {'include_weights': False}
        return root, {'transformers_version': 'fixture'}, 'hash'
    monkeypatch.setattr(worker, 'checkpoint', checkpoint)
    def tokenize(text, **kwargs):
        assert kwargs == {'truncation': False}
        calls.append(text)
        return {'input_ids': list(range(11))}
    def load(path, **kwargs):
        assert path == root
        assert kwargs == {'local_files_only': True, 'trust_remote_code': False}
        return tokenize
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        __version__='fixture', AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    assert worker.preflight_cpu_tokens('manifest', REQUEST, 22) == 22
    expected = ['Premise: '+p.strip()+'\nHypothesis: '+h.strip() for p, h in worker.assessment_pairs(REQUEST)]
    assert calls == expected
    with pytest.raises(ValueError, match='CAPACITY_EXCEEDED'):
        worker.preflight_cpu_tokens('manifest', REQUEST, 21)
    assert calls == expected * 2


@pytest.mark.parametrize('limit', [None, True, 0, 24577])
def test_cpu_preflight_requires_explicit_bounded_capacity(limit):
    with pytest.raises(ValueError, match='EXECUTION_CONFIG_INVALID'):
        worker.preflight_cpu_tokens('nonexistent-checkpoint', REQUEST, limit)


def test_token_only_checkpoint_still_rejects_tokenizer_drift(tmp_path):
    root = tmp_path / 'weights'; root.mkdir()
    config = {'architectures': ['Qwen3_5ForSequenceClassification'], 'id2label': worker.LABELS,
              'nli_template': 'Premise: {premise}\nHypothesis: {hypothesis}'}
    for name in worker.FILES:
        (root / name).write_text(json.dumps(config) if name == 'config.json' else 'fixture')
    manifest = {'schema': 'tare.local-labs/openjev-checkpoint/1', 'directory': 'weights',
                'files': {name: worker.sha256_file(root / name) for name in worker.FILES}}
    path = tmp_path / 'checkpoint.json'; path.write_text(json.dumps(manifest))
    (root / 'model.safetensors').unlink()
    assert worker.checkpoint(path, include_weights=False)[0] == root
    (root / 'tokenizer.json').write_text('changed')
    with pytest.raises(ValueError, match='CHECKPOINT_CHANGED'):
        worker.checkpoint(path, include_weights=False)
