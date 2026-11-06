# Changelog

Notable changes to `tare.tools.kernel` are recorded here, newest first.

## Unreleased

- `restore_bytes` undoes a deletion: with `after=None` it recreates the file's earlier bytes while the file is still absent, and preserves a file created in the meantime. Undoing a turn that deleted a file failed with `invalid_restore_snapshot`.
- Boolean tool arguments accept the text `"true"`/`"false"` that text-only call formats (Qwen's XML form) send, as `"null"` already was for a missing file's hash, so a local model can delete a file; Codex's strict replies carry `delete: null` for an ordinary edit.
- The native `edit` tool deletes a file: `delete=true`, `changes=[]` and the file's current SHA-256, within the writable scope and review, rechecking the hash before the unlink and recording the intent so an interrupted deletion is recovered without replay. A native implementer could create and change files but never remove one, so a file it created by mistake stayed in its candidate. It is the only deletion: the standalone `delete` operation of `mediated_tools`, which no host offered and which lacked those guards, is removed.
- CI runs on GitHub-hosted Linux and Windows runners (`Tests (linux)`, `Tests (windows)`) with the README's own install commands: the runtime and SpecGraph come from their public repositories at the pinned commits, without deploy keys or a hand-made environment. The repository variables `CI_RUNNER_LINUX` and `CI_RUNNER_WINDOWS` move it to your own runners, and pull requests from forks then skip it. The README installs the runtime and SpecGraph at the commits CI pins; the ones it named are not published.
- `test_default_read_returns_on_first_output` keeps its child running for 60 s instead of 1 s: on a loaded runner the read could start after the child had exited.
- CI pins the agent runtime and SpecGraph at the first commits of their public repositories; the commits it pinned before are not published.
- First public release. The repository starts from a single commit; earlier history is not published. It contains the
  kernel described in the README (provider adapters, mediated tools, Git workspaces, terminal sessions, hooks and
  append-only session journals), its specifications and contracts, among them the GPU lease contract, and the test
  suite.
