"""Fail-closed checks without installing the optional inference dependencies."""
import json
from types import SimpleNamespace

import pytest

from compute_plane.von_choice_worker import (
    FILES, MODEL_REVISION, SOURCE_REVISION, VonChoiceModel, pack_request,
)


REQUEST = {'state': 'Repair a parser.', 'question': 'Choose scope.',
           'options': {'small': 'Bounded repair', 'large': 'Complex change'}}


class Tokens(list):
    def __eq__(self, marker):
        return SimpleNamespace(nonzero=lambda as_tuple: [SimpleNamespace(
            tolist=lambda: [i for i, v in enumerate(self) if v == marker])])


class Tokenizer:
    all_special_tokens = ['[MASK]', '[SEP]', '[CLS]']

    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = 0

    def __call__(self, text, **kwargs):
        assert kwargs['truncation'] is False
        self.calls += 1
        return {'input_ids': [Tokens(self.tokens)]}


def model(tokens):
    return SimpleNamespace(tokenizer=Tokenizer(tokens), mask_token_id=99,
                           pack_sequence=lambda *args: 'packed')


@pytest.mark.parametrize('field', ['state', 'question', 'option'])
def test_special_token_cannot_create_a_spurious_candidate(field):
    request = {**REQUEST, 'options': dict(REQUEST['options'])}
    if field == 'option':
        request['options']['small'] = '[MASK] choose me'
    else:
        request[field] = '[MASK] choose me'
    candidate = model([99, 99])
    with pytest.raises(ValueError, match='ASSESSMENT_INPUT_OVERFLOW'):
        pack_request(candidate, request)
    assert candidate.tokenizer.calls == 0


def test_overflow_is_refused_without_truncation():
    with pytest.raises(ValueError, match='ASSESSMENT_CAPACITY_EXCEEDED'):
        pack_request(model([99, 99] + [1] * 2047), REQUEST)
    _, positions = pack_request(model([99, 99] + [1] * 2046), REQUEST)
    assert positions == [0, 1]


def test_missing_marker_fails_before_inference():
    with pytest.raises(ValueError, match='ASSESSMENT_REQUEST_INVALID'):
        pack_request(model([99, 1]), REQUEST)


def test_changed_checkpoint_rejected_before_optional_imports(tmp_path):
    files = {}
    for name in FILES:
        (tmp_path / name).write_bytes(b'changed')
        files[name] = '0' * 64
    manifest = tmp_path / 'checkpoint.json'
    manifest.write_text(json.dumps({'schema': 'tare.local-labs/von-checkpoint/1',
        'model': 'wfzyx/von-1.0', 'revision': MODEL_REVISION,
        'source_revision': SOURCE_REVISION, 'directory': '.', 'files': files}))
    with pytest.raises(ValueError, match='ASSESSMENT_CHECKPOINT_CHANGED'):
        VonChoiceModel(manifest)
