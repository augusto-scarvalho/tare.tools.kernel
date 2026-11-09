import importlib.util
import io
from pathlib import Path
import urllib.error

import pytest

spec = importlib.util.spec_from_file_location('install_test_ripgrep',
                                              Path(__file__).parents[1]/'tools'/'install_test_ripgrep.py')
installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)


def test_a_transient_download_failure_is_retried_and_a_missing_asset_is_not(monkeypatch):
    # 2026-11-09: one 504 from GitHub's release download failed every CI shard.
    answers, slept = [504, 504, b'archive'], []
    def urlopen(url, timeout):
        answer = answers.pop(0)
        if isinstance(answer, int):
            raise urllib.error.HTTPError(url, answer, 'Gateway Time-out', {}, None)
        return io.BytesIO(answer)
    monkeypatch.setattr(installer.urllib.request, 'urlopen', urlopen)
    assert installer.fetch('https://example.test/a', sleep=slept.append) == b'archive' and slept == [2, 4]
    answers[:] = [504]*8
    with pytest.raises(urllib.error.HTTPError):
        installer.fetch('https://example.test/a', sleep=slept.append)
    assert slept[2:] == [2, 4, 8, 16, 32, 60, 60]  # about 3 minutes, then it gives up
    del slept[2:]
    answers[:] = [404]
    with pytest.raises(urllib.error.HTTPError):
        installer.fetch('https://example.test/a', sleep=slept.append)
    assert slept == [2, 4] and answers == []  # one call, no wait
