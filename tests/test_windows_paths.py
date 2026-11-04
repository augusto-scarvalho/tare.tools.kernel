r"""Windows' realpath may keep the \\?\ prefix; tare strips it only where the plain path names the same file."""
import ntpath
import os
from pathlib import Path

import pytest

from compute_plane import windows_paths
from compute_plane.windows_paths import PREFIX, plain


@pytest.mark.parametrize('result,requested,expected', [
    (PREFIX + 'C:\\work\\prepared\\session', 'C:/work/prepared/session', 'C:\\work\\prepared\\session'),
    (PREFIX + 'C:\\work\\x', PREFIX + 'C:\\work\\x', PREFIX + 'C:\\work\\x'),  # the caller asked for it
    (PREFIX + 'UNC\\server\\share\\x', 'x', PREFIX + 'UNC\\server\\share\\x'),
    (PREFIX + 'C:\\work\\CON', 'x', PREFIX + 'C:\\work\\CON'),  # plain would name the console device
    (PREFIX + 'C:\\work\\nul.txt\\a', 'x', PREFIX + 'C:\\work\\nul.txt\\a'),
    (PREFIX + 'C:\\work\\trailing.', 'x', PREFIX + 'C:\\work\\trailing.'),  # plain would drop the dot
    (PREFIX + 'C:\\' + 'a' * 300, 'x', PREFIX + 'C:\\' + 'a' * 300),  # only the prefix reaches past MAX_PATH
    ('C:\\work\\x', 'x', 'C:\\work\\x'),
    (b'\\\\?\\C:\\x', b'x', b'\\\\?\\C:\\x'),
])
def test_only_an_equivalent_plain_path_loses_the_prefix(result, requested, expected):
    assert plain(result, requested) == expected


@pytest.mark.skipif(os.name != 'nt', reason='ntpath is os.path only on Windows')
def test_importing_compute_plane_makes_resolve_plain(monkeypatch, tmp_path):
    import compute_plane  # noqa: F401 - installs the wrapper
    assert getattr(ntpath.realpath, 'tare_plain', False)
    keeps_prefix = lambda path, *a, **k: PREFIX + str(tmp_path)  # CPython's race, made deterministic
    monkeypatch.setattr(ntpath, 'realpath', keeps_prefix)
    windows_paths.install()
    assert Path(tmp_path).resolve() == tmp_path and not str(Path(tmp_path).resolve()).startswith(PREFIX)
