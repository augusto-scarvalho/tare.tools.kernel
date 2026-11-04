"""Prepare a transport protocol offline using explicit and installed sources."""
import hashlib
import json
from pathlib import Path

import pytest
import tare_tools_agent_runtime

from tools import native_canary


def test_missing_specgraph_configuration_is_rejected_before_probing(tmp_path, monkeypatch):
    monkeypatch.delenv('TARE_SPECGRAPH_ROOT', raising=False)
    monkeypatch.setattr(native_canary, 'get', lambda url: pytest.fail('network probe before configuration'))
    with pytest.raises(KeyError, match='TARE_SPECGRAPH_ROOT'):
        native_canary.prepare(tmp_path)
    assert not list(tmp_path.iterdir())


def test_prepare_hashes_imported_runtime_and_explicit_profile_source(tmp_path, monkeypatch):
    sg = tmp_path / 'arbitrary checkout'
    profiles = sg / 'tools/pilot_profiles.py'
    profiles.parent.mkdir(parents=True)
    profiles.write_text(
        "PROFILES = {'qwen-low-2048': {}}\n"
        "def request_parameters(profile):\n"
        "    return {'extra_body': {}, 'max_tokens': 4096, 'stream': False}\n",
        encoding='utf-8',
    )
    monkeypatch.setenv('TARE_SPECGRAPH_ROOT', str(sg))
    def observation(url):
        if url == native_canary.GATEWAY + '/props':
            return {'current_model': 'qwen38', 'backend_healthy': True, 'backend_pid': 123}
        assert url == native_canary.COUNTER + '/props'
        return {'default_generation_settings': {'n_ctx': 32768}, 'chat_template': 'test template'}
    monkeypatch.setattr(native_canary, 'get', observation)
    lot = tmp_path / 'lot'
    lot.mkdir()
    native_canary.prepare(lot)
    protocol = json.loads((lot / 'protocol.json').read_text())
    sources = protocol['source_sha256']
    runtime = Path(tare_tools_agent_runtime.__file__).resolve().parent
    for path in [profiles, *(runtime / name for name in ['native.py', 'native_context.py', 'conversation.py'])]:
        assert sources[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert protocol['limits']['calls_per_task'] == 6
    assert protocol['profiles']['qwen']['endpoint'] == native_canary.GATEWAY + '/v1'
