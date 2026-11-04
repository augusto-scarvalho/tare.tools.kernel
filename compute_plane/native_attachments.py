"""Explicit content snapshots for native input; no project or storage discovery."""
from dataclasses import dataclass, field
import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile

from tare_tools_agent_runtime import AttachmentRef, ModelFailure
from tare_tools_agent_runtime.native_content import MAX_ATTACHMENT_BYTES

from .native_journal import custody_lock, regular, _sync_directory, io_path as _io_path


IMAGE_TYPES = {'image/png': 'PNG', 'image/jpeg': 'JPEG',
               'image/webp': 'WEBP', 'image/gif': 'GIF'}
TEXT_TYPES = {'application/json', 'application/xml', 'application/yaml', 'image/svg+xml'}
MAX_IMAGE_PIXELS = 16_000_000


class AttachmentError(ModelFailure):
    """Local validation failed before any provider invocation."""
    def __init__(self, code):
        super().__init__('ATTACHMENT_'+code, not_started=True)


def verify_bytes(ref, data):
    if not isinstance(ref, AttachmentRef) or not isinstance(data, bytes):
        raise AttachmentError('INVALID_CONTENT')
    if len(data) != ref.size_bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
        raise AttachmentError('SNAPSHOT_CHANGED')


@dataclass(frozen=True)
class ContentInfo:
    text: str | None = field(default=None, repr=False)
    width: int | None = None
    height: int | None = None


def inspect_content(ref, data):
    verify_bytes(ref, data)
    if ref.media_type in IMAGE_TYPES:
        try:
            from PIL import Image
        except ImportError:
            raise AttachmentError('IMAGE_DECODER_UNAVAILABLE') from None
        try:
            with Image.open(io.BytesIO(data), formats=list(IMAGE_TYPES.values())) as image:
                if image.format != IMAGE_TYPES[ref.media_type]:
                    raise AttachmentError('IMAGE_TYPE_MISMATCH')
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                    raise AttachmentError('IMAGE_DIMENSIONS_EXCEEDED')
                if getattr(image, 'is_animated', False):
                    raise AttachmentError('ANIMATED_IMAGE_UNSUPPORTED')
                image.verify()
            with Image.open(io.BytesIO(data), formats=list(IMAGE_TYPES.values())) as image:
                image.load()
        except AttachmentError:
            raise
        except Exception:
            raise AttachmentError('INVALID_IMAGE') from None
        return ContentInfo(width=width, height=height)
    if ref.media_type.startswith('text/') or ref.media_type in TEXT_TYPES:
        try:
            text = data.decode('utf-8-sig')
        except UnicodeError:
            raise AttachmentError('DOCUMENT_REQUIRES_UTF8') from None
        if '\x00' in text:
            raise AttachmentError('BINARY_DOCUMENT_UNSUPPORTED')
        return ContentInfo(text=text)
    raise AttachmentError('UNSUPPORTED_MEDIA_TYPE')


def detect_media_type(name, data):
    for prefix, mime in ((b'\x89PNG\r\n\x1a\n', 'image/png'), (b'\xff\xd8', 'image/jpeg'),
                         (b'GIF87a', 'image/gif'), (b'GIF89a', 'image/gif')):
        if data.startswith(prefix):
            return mime
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    # Known binary formats must not accidentally be sent as decoded text.
    if data.startswith((b'%PDF-', b'PK\x03\x04')):
        raise AttachmentError('DOCUMENT_FORMAT_UNSUPPORTED')
    return {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.webp': 'image/webp', '.gif': 'image/gif', '.json': 'application/json',
            '.svg': 'image/svg+xml', '.xml': 'application/xml', '.md': 'text/markdown',
            '.csv': 'text/csv'}.get(Path(name).suffix.lower(), 'text/plain')


def _read_regular(path):
    # NONBLOCK permits a prompt fstat refusal for FIFOs on POSIX.
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(regular(_io_path(path)), flags)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise AttachmentError('REGULAR_FILE_REQUIRED')
        if before.st_size > MAX_ATTACHMENT_BYTES:
            raise AttachmentError('FILE_SIZE_EXCEEDED')
        data = stream.read(MAX_ATTACHMENT_BYTES+1)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise AttachmentError('SOURCE_CHANGED')
    return data


class NativeAttachmentStore:
    """The host supplies an external directory and owns retention of its blobs."""
    def __init__(self, root):
        self.root = Path(root).absolute()
        regular(_io_path(self.root))

    def load(self, ref):
        if not isinstance(ref, AttachmentRef):
            raise AttachmentError('INVALID_REFERENCE')
        try:
            data = _read_regular(self.root/ref.sha256)
        except (OSError, ValueError):
            raise AttachmentError('SNAPSHOT_UNAVAILABLE') from None
        verify_bytes(ref, data)
        return data

    def put_bytes(self, name, data, media_type=None):
        if not isinstance(data, bytes) or len(data) > MAX_ATTACHMENT_BYTES:
            raise AttachmentError('FILE_SIZE_EXCEEDED')
        ref = AttachmentRef(name, media_type or detect_media_type(name, data),
                            hashlib.sha256(data).hexdigest(), len(data))
        inspect_content(ref, data)
        root = regular(_io_path(self.root))
        root.mkdir(parents=True, exist_ok=True)
        with custody_lock(root):
            destination = regular(root/ref.sha256)
            if destination.exists():
                self.load(ref)  # Preserve corrupt snapshots for explicit reconciliation.
                return ref
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, prefix='.attachment-', delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, destination)
                _sync_directory(root)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return ref

    def import_file(self, path, *, media_type=None):
        source = regular(_io_path(path))
        try:
            data = _read_regular(source)
        except OSError:
            raise AttachmentError('SOURCE_UNAVAILABLE') from None
        return self.put_bytes(source.name, data, media_type)
