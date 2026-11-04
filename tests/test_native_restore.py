import os
import stat

import pytest

from compute_plane.native_review import restore_bytes


def test_preserves_binary_bytes_newlines_mode_and_removes_temporary(tmp_path):
    path = tmp_path/'file'
    path.write_bytes(b'after\r\n')
    if os.name != 'nt': path.chmod(0o751)
    original_mode = stat.S_IMODE(path.stat().st_mode)
    seen = []
    restore_bytes(path, b'\xef\xbb\xbfbefore\r\n\x00', b'after\r\n', before_write=lambda: seen.append(True))
    assert path.read_bytes() == b'\xef\xbb\xbfbefore\r\n\x00'
    assert seen == [True]
    assert stat.S_IMODE(path.stat().st_mode) == original_mode
    assert list(tmp_path.iterdir()) == [path]


def test_rechecks_after_durable_intent_and_preserves_concurrent_change(tmp_path):
    path = tmp_path/'file'
    path.write_bytes(b'after')
    with pytest.raises(ValueError, match='changed before restoration'):
        restore_bytes(path, b'before', b'after', before_write=lambda: path.write_bytes(b'user change'))
    assert path.read_bytes() == b'user change'
    assert list(tmp_path.iterdir()) == [path]


def test_intent_failure_never_mutates_and_cleans_up(tmp_path):
    path = tmp_path/'file'
    path.write_bytes(b'after')
    def failure(): raise OSError('full disk')
    with pytest.raises(OSError, match='full disk'):
        restore_bytes(path, b'before', b'after', before_write=failure)
    assert path.read_bytes() == b'after'
    assert list(tmp_path.iterdir()) == [path]


def test_symlink_target_refused(tmp_path):
    source, link = tmp_path/'source', tmp_path/'link'
    source.write_bytes(b'after')
    try: link.symlink_to(source)
    except OSError: pytest.skip('Host does not permit symlinks')
    with pytest.raises(ValueError, match='link_forbidden'):
        restore_bytes(link, b'before', b'after', before_write=lambda: None)
    assert source.read_bytes() == b'after'
