"""One guarded commit for several replacements, including an interrupted result."""
import json
import pytest
from tests.test_native_capabilities import setup, call
from tests.test_native_effect_recovery import request, journal_for


def batch(root):
    import hashlib
    return dict(repository='repo', path='a.py',
                sha256=hashlib.sha256((root/'a.py').read_bytes()).hexdigest(),
                changes=[{'search': 'value = 1', 'replace': 'value = 2'},
                         {'search': 'value = 2', 'replace': 'value = 3'}])


def test_batch_commits_once_and_retains_new_sha_and_line_endings(setup):
    root, _, tools, control = setup
    (root/'a.py').write_bytes(b'value = 1\r\n')
    args = batch(root)
    result = call(tools, control, 'edit', **args)
    assert result.status == 'succeeded'
    assert (root/'a.py').read_bytes() == b'value = 3\r\n'
    assert json.loads(result.output_json)['effects'] == 1
    stale = call(tools, control, 'edit', **args)
    assert stale.status == 'refused'
    assert tools._tools.effects == 1


@pytest.mark.parametrize('change', [
    {'search': 'absent', 'replace': 'x'}, {'search': 'value = 2', 'replace': None},
    {'search': 'value = 2', 'replace': 'x', 'extra': True},
])
def test_invalid_later_replacement_does_not_write_or_start_an_effect(setup, change):
    root, scratch, tools, control = setup
    args = batch(root); args['changes'][1] = change
    before = (root/'a.py').read_bytes()
    result = call(tools, control, 'edit', **args)
    assert result.status == 'refused'
    assert (root/'a.py').read_bytes() == before
    assert tools._tools.effects == 0 and not tools._started
    receipts = [json.loads(path.read_bytes()) for path in scratch.rglob('*.json')]
    assert len(receipts) == 1 and receipts[0]['result']['status'] == 'refused'
    assert 'edit' not in receipts[0]


@pytest.mark.parametrize('cut', ['intent', 'replace', 'result'])
def test_batch_recovery_proves_bytes_without_reexecution(setup, monkeypatch, cut):
    root, _, tools, control = setup
    req = request('edit', **batch(root))
    original = tools._tools.before_edit
    def before(*args):
        original(*args)
        if cut == 'intent': raise SystemExit()
    monkeypatch.setattr(tools._tools, 'before_edit', before)
    save = tools._save_result
    def saved(*args):
        if cut == 'result': save(*args)
        raise SystemExit()
    monkeypatch.setattr(tools, '_save_result', saved)
    with pytest.raises(SystemExit): tools.execute(req, control)
    captured = (root/'a.py').read_bytes()
    monkeypatch.setattr(tools._tools, 'perform', lambda *args: pytest.fail('must not replay'))
    _, result = tools.recover_effect(journal_for(req))
    assert result.status == ('refused' if cut == 'intent' else 'succeeded')
    assert (root/'a.py').read_bytes() == captured
