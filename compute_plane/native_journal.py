"""Bounded fsynced native journal under exclusive local process custody."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from time import sleep

from tare_tools_agent_runtime import ConversationLimits, ConversationLog, ConversationRejected


def io_path(path):
    """Use extended Windows paths only where callers perform filesystem I/O."""
    path = Path(path).absolute()
    name = str(path)
    if os.name != 'nt' or name.startswith(('\\\\?\\', '\\\\.\\')):
        return path
    return Path('\\\\?\\UNC\\'+name[2:] if name.startswith('\\\\') else '\\\\?\\'+name)


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in (path, *path.parents)):
        raise ValueError('native_custody_link_forbidden')
    return path


@contextmanager
def custody_lock(root):
    root = regular(root)
    if not root.is_dir():
        raise ValueError('native_custody_directory_required')
    path = regular(root/'writer.lock')
    with path.open('a+b') as lock:
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            # Windows permits locking beyond EOF. Initializing the byte before
            # acquisition races with a first writer already holding that region.
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError:
            raise ValueError('native_custody_has_active_writer') from None
        try:
            yield root
        finally:
            lock.seek(0); release()


def read_json(path, *, max_bytes=4*1024*1024):
    """Custody-file budget, independent of the provider wire-envelope budget."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('native_metadata_budget_invalid')
    with regular(path).open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError('native_metadata_over_budget')
    try:
        return ConversationLog(ConversationLimits(max_event_bytes=max_bytes)).append_json(raw).event()
    except ConversationRejected as exc:
        raise ValueError('invalid_native_metadata_json') from exc


def save_json(path, value, *, max_bytes=4*1024*1024):
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('native_metadata_budget_invalid')
    path = regular(path)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > max_bytes:
        raise ValueError('native_metadata_over_budget')
    temporary = regular(path.with_name(path.name+'.pending'))
    with temporary.open('wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    # Windows readers/scanners can briefly deny replacing an otherwise valid
    # metadata file. Retry only the atomic rename, never the recorded effect.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except OSError as error:
            if getattr(error, 'winerror', None) not in {5, 32, 33} or attempt == 5:
                raise
            sleep(.02 * 2**attempt)
    _sync_directory(path.parent)


def _sync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


NATIVE_JOURNAL_MAX_BYTES = 16 * 1024 * 1024
NATIVE_JOURNAL_MAX_EVENT_BYTES = 256 * 1024


class NativeJournal:
    """Owner holds custody_lock for this object's entire lifetime.

    Torn/corrupt data is refused, never truncated to make a session resumable.
    Durability covers process crashes with successful fsync; it does not attest
    administrator tampering, cross-host filesystem locks or power-loss hardware.
    """
    def __init__(self, path, *, max_bytes=NATIVE_JOURNAL_MAX_BYTES,
                 max_event_bytes=NATIVE_JOURNAL_MAX_EVENT_BYTES):
        self.path = regular(path)
        self.max_bytes = max_bytes
        # The durable journal already caps the selected history. Its projection
        # must not inherit ConversationLog's smaller, unrelated 2 MiB default:
        # context counting and compaction happen only after projection.
        # One model reply is one event: a 16K-token answer with its reasoning, stored as reply
        # text and provider message, reaches ~180 KB (64 KiB crashed xhigh sessions). The
        # tactical journal stays well below the 1 MiB observation journal; readers share it.
        self._limits = ConversationLimits(max_projection_bytes=max_bytes,
                                          max_event_bytes=max_event_bytes)
        self._log = ConversationLog(self._limits)
        frame_limit = max(131072, max_event_bytes + 1024)
        self._chain = '0'*64
        self._poisoned = False
        self._size = 0
        if self.path.exists():
            with self.path.open('rb') as stream:
                for raw in iter(lambda: stream.readline(frame_limit + 1), b''):
                    self._size += len(raw)
                    if self._size > max_bytes or len(raw) > frame_limit or not raw.endswith(b'\n'):
                        raise ValueError('native_journal_incomplete_or_over_budget')
                    row = json.loads(raw)
                    event = ConversationLog(self._limits).append(row['event'])
                    expected = self._frame(event.event())
                    if raw != expected:
                        raise ValueError('native_journal_identity_or_chain_invalid')
                    self._log.append(event.event())
                    self._chain = hashlib.sha256(raw).hexdigest()

    def _frame(self, event):
        return (json.dumps({'sequence': self._log.snapshot().next_sequence, 'previous': self._chain,
                            'event': event}, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False)+'\n').encode()

    def append(self, event):
        if self._poisoned:
            raise RuntimeError('native_journal_requires_reopen_after_write_failure')
        checked = ConversationLog(self._limits).append(event)
        if len(self._log.snapshot().records) >= self._log.limits.max_events:
            raise ValueError('native_journal_full')
        raw = self._frame(checked.event())
        if self._size + len(raw) > self.max_bytes:
            raise ValueError('native_journal_full')
        try:
            with self.path.open('ab') as stream:
                if stream.write(raw) != len(raw):
                    raise OSError('short_native_journal_write')
                stream.flush(); os.fsync(stream.fileno())
            _sync_directory(self.path.parent)
            record = self._log.append(checked.event())
        except BaseException:
            self._poisoned = True
            raise
        self._size += len(raw)
        self._chain = hashlib.sha256(raw).hexdigest()
        return record

    def snapshot(self):
        return self._log.snapshot()

    def project(self, spec):
        return self._log.project(spec)


class NativeObservationJournal(NativeJournal):
    """Provider receipts retain wire detail outside the tactical conversation."""

    def __init__(self, path, *, max_bytes=NATIVE_JOURNAL_MAX_BYTES):
        super().__init__(path, max_bytes=max_bytes, max_event_bytes=1024 * 1024)
