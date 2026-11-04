"""Crash windows around an atomic edit and a completed verifier."""
from dataclasses import asdict
import json
import pytest
from tare_tools_agent_runtime import ConversationLog, EffectRequest, ToolCall
from compute_plane.native_capabilities import NativeCapabilities
from tests.test_native_capabilities import setup


def request(capability, **args):
    return EffectRequest('s', 's/effect/1', ToolCall('call', capability, json.dumps(args)))


def journal_for(req):
    log = ConversationLog()
    log.append({'kind': 'effect_requested', 'effect_id': req.effect_id, 'tool_call': asdict(req.tool_call)})
    return log


def edit(root):
    import hashlib
    return request('edit', repository='repo', path='a.py',
        sha256=hashlib.sha256((root/'a.py').read_bytes()).hexdigest(), search='value = 1', replace='value = 2')


@pytest.mark.parametrize('cut', ['intent', 'replace', 'result'])
def test_edit_recovers_without_reexecution(setup, monkeypatch, cut):
    root, scratch, tools, control = setup
    req = edit(root)
    original = tools._tools.before_edit
    def before(*args):
        original(*args)
        if cut == 'intent': raise SystemExit()
    monkeypatch.setattr(tools._tools, 'before_edit', before)
    original_save = tools._save_result
    def saved(*args):
        if cut == 'result': original_save(*args)
        raise SystemExit()
    monkeypatch.setattr(tools, '_save_result', saved)
    with pytest.raises(SystemExit): tools.execute(req, control)
    before_bytes = (root/'a.py').read_bytes()
    pending, result = tools.recover_effect(journal_for(req))
    assert pending == req
    assert result.status == ('refused' if cut == 'intent' else 'succeeded')
    assert (root/'a.py').read_bytes() == before_bytes
    if cut != 'intent':
        (root/'a.py').write_text('user changes\n')
        with pytest.raises(ValueError, match='newer changes were preserved'):
            tools.recover_effect(journal_for(req))
        assert (root/'a.py').read_text() == 'user changes\n'


def test_completed_verifier_uses_saved_receipt_and_unknown_one_is_not_retried(setup, monkeypatch):
    root, scratch, tools, control = setup
    req = request('verify', name='unit')
    result = tools.execute(req, control)
    def forbidden(*args): pytest.fail('must not execute again')
    monkeypatch.setattr(tools._tools, 'perform', forbidden)
    assert tools.recover_effect(journal_for(req)) == (req, result)
    missing = EffectRequest('s', 's/effect/2', req.tool_call)
    assert tools.recover_effect(journal_for(missing)) is None


def test_foreign_receipt_and_changed_request_are_rejected(setup):
    root, scratch, tools, control = setup
    req = edit(root)
    tools.execute(req, control)
    other = EffectRequest('s', req.effect_id, ToolCall('call', 'edit', req.tool_call.arguments_json.replace('value = 2', 'value = 3')))
    with pytest.raises(ValueError, match='does not match'):
        tools.recover_effect(journal_for(other))


def test_edit_does_not_overwrite_changes_while_intent_is_saved(setup, monkeypatch):
    root, scratch, tools, control = setup
    req = edit(root)
    original = tools._tools.before_edit
    def changed(*args):
        original(*args)
        (root/'a.py').write_text('newer user change\n')
    monkeypatch.setattr(tools._tools, 'before_edit', changed)
    assert tools.execute(req, control).status == 'unknown'
    with pytest.raises(ValueError, match='newer changes were preserved'):
        tools.recover_effect(journal_for(req))
    assert (root/'a.py').read_text() == 'newer user change\n'
