# Native input attachments

Paired with Runtime's `AttachmentRef` contract. The OS
supplies storage, selected files and model capabilities; Kernel validates content
and builds provider requests. This foundation does not yet expose a TUI picker.

## Composition

Create `NativeAttachmentStore(external_directory)` and call `import_file(path)`
or `put_bytes(name, data)`. Pass the returned immutable references to Runtime's
initial or follow-up message. Inject `store.load` as `attachment_loader` into
`NativeModelAdapter`. Explicitly configure `image_media_types` for a deployment
that supports those inputs. A model name does not establish vision support.

The optional `attachments` package extra installs Pillow for image validation.
Text attachments do not require Pillow. Legacy text-only adapter imports and
requests remain usable with older Runtime releases; attachment use requires a
Runtime that carries `AttachmentRef`, which the declared dependency range does.

## Content and custody

- Snapshots retain original bytes independently of subsequent source edits or
  deletion. Names, MIME types, byte lengths and SHA-256 hashes travel in Runtime
  messages; file paths and base64 image bodies do not enter the journal.
- Every load checks size and hash. Missing or corrupt snapshots refuse the
  request before dispatch. Existing corrupt blobs are preserved rather than
  silently repaired. The host owns retention and orphan cleanup.
- Reads require regular, unlinked files, are bounded to 16 MiB and check file
  metadata around the read. Writes use a custody lock, a flushed temporary file
  and atomic replacement. This follows existing filesystem custody semantics;
  it is not protection against a privileged process replacing parent paths.
- PNG, JPEG, static GIF and WebP are decoded and verified. Images are limited
  to 16 million pixels. Animated images refuse. UTF-8 text, Markdown, CSV, JSON,
  XML and YAML are transmitted as text; SVG is XML text, not a rendered image.
  PDF and Office extraction remain unimplemented and known binary formats refuse.
- Each request has a shared 16 MiB raw attachment budget across initial and
  subsequent messages. References remain subject to Runtime's 16-per-message
  limit. Bedrock additionally checks 20 images per provider message, 8,000 pixels
  per dimension and a conservative 3,750,000-byte ceiling per image.

## Provider requests and counting

OpenAI Chat requests use text parts and `image_url` data URLs containing the
actual snapshot. Bedrock Converse requests use text and image blocks with base64
`source.bytes` in the raw HTTP JSON. CountTokens receives those same content
blocks. These shapes follow the primary documentation for
[OpenAI image inputs](https://developers.openai.com/api/docs/guides/images-vision),
[Bedrock image sources](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html)
and [Bedrock message limits](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Message.html).

The serialized request hash binds admission to hydrated content. A changed
reference after counting refuses before generation. Local text tokenizers and
the current fleet counter are not qualified to count image inputs: they report
unavailable without issuing a count request. Explicit UTF-8 envelope estimation
remains an estimate, with image token usage marked unmeasured. The Runtime
context policy decides whether estimates may admit a request. A loading error
during context preparation blocks it as `CONTEXT_COUNT_ERROR`; no generation
call is started.

`NativeHttpClient` keeps its default 1 MiB request bound. The host may explicitly
set `request_byte_limit` up to 32 MiB for attachment requests. The isolated worker
enforces that limit for regular and streaming calls. Response bounds, redirects,
cancellation and process cleanup retain their existing behavior. Attachment
validation and request-size failures are classified as not started. Telemetry
contains request hashes, not attachment bodies.

## Qualification

`tests/test_native_attachments.py` checks byte custody, original-file deletion,
corruption, image decoding/limits, exact provider content, first/follow-up turns,
completed-session reopening, admission changes, and a real HTTP worker carrying
an image larger than the default request limit. Existing native model, streaming,
cancellation and fleet-counter tests cover the surrounding transport behavior.
Deep external history paths are covered on Windows using extended paths only at
the I/O boundary; saved path spelling, link checks and content verification remain
unchanged. POSIX tests still exercise special-file and symlink refusal.
These fixtures do not qualify vision on a real model. OS owns composer and CLI
qualification in its `docs/guides/NATIVE_ATTACHMENTS.md` guide.
