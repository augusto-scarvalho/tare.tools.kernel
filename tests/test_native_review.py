"""Review remains available when external history exceeds Windows MAX_PATH."""
import json

import pytest

from compute_plane.native_journal import io_path
from compute_plane.native_review import capture, digest, receipt_path, save_edit


def test_deep_review_history_preserves_snapshots_receipts_and_corruption(tmp_path):
    root = tmp_path/('history-'+'x'*70)/('session-'+'y'*70)/'review'
    io_path(root.parent).mkdir(parents=True)
    before, after = b'value = 1\r\n', b'value = 2\r\n'
    args = {'repository':'project', 'path':'a.py', 'sha256':digest(before)}
    output = {'sha256':digest(after)}
    assert len(str(root/'blobs'/digest(before))) > 260
    save_edit(root, 'effect-1', args, before, after, output)
    receipt = json.loads(receipt_path(root,'effect-1').read_text())
    assert (receipt['before'],receipt['after']) == (digest(before),digest(after))
    assert capture(root/'blobs'/digest(before)) == before
    assert capture(root/'blobs'/digest(after)) == after
    damaged = io_path(root/'blobs'/digest(after))
    damaged.write_bytes(b'changed externally')
    with pytest.raises(ValueError,match='review_blob_corrupt'):
        save_edit(root, 'effect-2', args, before, after, output)
    assert damaged.read_bytes() == b'changed externally'
    assert not receipt_path(root,'effect-2').exists()
