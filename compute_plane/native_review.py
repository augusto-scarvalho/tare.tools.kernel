"""Optional local edit snapshots. Never included in model messages."""
import hashlib
import os
import tempfile
from pathlib import Path

from .native_journal import io_path, regular, save_json, _sync_directory
from .native_arguments import normalize_booleans


def edit_args(args):
    """Interpret text-only edit arguments consistently for execution and recovery."""
    args = normalize_booleans(args, {'properties': {'delete': {'type': 'boolean'}}})
    return {**args, 'sha256': None} if args.get('sha256') == 'null' else args


def digest(raw):
    return None if raw is None else hashlib.sha256(raw).hexdigest()


def receipt_path(root, effect_id):
    return regular(io_path(root / (hashlib.sha256(effect_id.encode()).hexdigest() + '.json')))


def capture(path):
    path = regular(io_path(path))
    if not path.exists():
        return None
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('review_file_too_large')
    return raw


def save_edit(root, effect_id, args, before, after, output):
    """Accept only bytes matching both the effect precondition and its result."""
    if digest(before) != args['sha256'] or digest(after) != output['sha256']:
        raise ValueError('review_source_changed')
    root = regular(io_path(root))
    root.mkdir(exist_ok=True)
    blobs = regular(root / 'blobs')
    blobs.mkdir(exist_ok=True)
    for raw in (before, after):
        if raw is None:
            continue
        path = regular(blobs / digest(raw))
        if path.exists():
            if capture(path) != raw:
                raise ValueError('review_blob_corrupt')
            continue
        temporary = regular(path.with_suffix('.pending'))
        with temporary.open('wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    save_json(receipt_path(root, effect_id), {
        'schema': 'tare.tools/edit-review/1', 'effect_id': effect_id,
        'repository': args['repository'], 'path': args['path'],
        'before': digest(before), 'after': digest(after)})


def restore_bytes(path, before, after, *, before_write):
    """Restore verified bytes under the host's session/workspace custody.

    The host validates authority and snapshot provenance. Recheck the current
    bytes after preparing the replacement and persisting intent. This shares
    the file tool's cooperative-lock contract; unrelated editors do not honor
    it, so this is not an OS-wide compare-and-swap transaction. An `after` of
    None undoes a deletion: the file must still be absent, and `before` returns.
    """
    if (after is not None and not isinstance(after, bytes) or before is not None and not isinstance(before, bytes)
            or before is None and after is None
            or after is not None and len(after) > 2 * 1024 * 1024 or before is not None and len(before) > 2 * 1024 * 1024):
        raise ValueError('invalid_restore_snapshot')
    path = regular(io_path(path))
    if capture(path) != after:
        raise ValueError('File changed since this edit; newer changes were preserved.')
    temporary = None
    try:
        if before is not None:
            descriptor, temporary = tempfile.mkstemp(prefix='.tare-undo-', dir=str(path.parent))
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(before)
                stream.flush()
                os.fsync(stream.fileno())
            if after is not None:  # a deleted file has no mode to keep; the new one gets the default
                os.chmod(temporary, path.stat().st_mode)
        before_write()
        if capture(path) != after:
            raise ValueError('File changed before restoration; newer changes were preserved.')
        if before is None:
            path.unlink()
        else:
            os.replace(temporary, path)
        _sync_directory(path.parent)
        if capture(path) != before:
            raise ValueError('Restoration could not be confirmed; inspect the current file.')
        return digest(before)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
