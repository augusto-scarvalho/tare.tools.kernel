# Session and audit adapters

Local additive surface, independent of tool policy and provider rollout.

- `compute_plane.session_hooks`: codecs for start/stop/end observations.
  Claude/Codex use lifecycle event names; AGY requires an explicit phase hint.
  The caller must verify real host capabilities; this module does not prove
  native SessionEnd availability. Stop/end always render neutral output.
- `compute_plane.audit_driver.ClaudeAuditDriver`: explicit executable, model,
  UUID, resume flag, verdict schema, optional cost ceiling and deadline. Read-only built-in
  tools, safe mode, no fallback, no latest-session selection, no shell wrapper.
- `inspect_audit_candidate`: read-only Git/byte checks before and after a run.
  Never creates commits, cleans candidates or changes worktrees.
- Packet profile: explicit effort, no navigation tools, stream-json/verbose,
  incremental `AuditStream` parsing of the launched channel, bounded structured
  drafts and separate author/auxiliary usage. No transcript directory lookup.
  StructuredOutput and exact producer identity passed a bounded real pilot with
  Claude Code 2.1.263 / claude-fable-5-1, requested effort xhigh. The adapter
  translates the supported schema subset to the CLI dialect, supplies a small
  explicit system prompt, and preserves bounded startup stderr. This does not
  qualify other providers, versions, or real multi-round session resumption.
- `audit_changes`/`audit_source`: explicit base and WORKTREE/INDEX/commit reads;
  no semantic dependency engine. OS owns source selection and packet sealing.
- Input is a finite temporary file, avoiding Windows stdin pipe stalls. Deadline
  and cooperative cancellation use the existing child-tree termination path.
  Omitted cost ceiling sends no monetary limit to the provider; a supplied
  ceiling is soft, not a subscription-quota guarantee. Input supports up to
  64000 bytes; OS selects the per-packet limit (24000 by default).

The command passed to `run` must start with the exact validated executable and
positional prefix. Flag-like prefix arguments are rejected so they cannot
override the provider options. POSIX cleanup terminates the dedicated process
group even if its leader exits before a descendant. Windows cleanup uses
`taskkill /T /F` and remains best-effort for already orphaned descendants.

OS owns Work selection, roster/lineage checks, finite case policy, one-shot
dispatch reservation, receipt verification and result custody. Agent Runtime
continues owning its managed tactical loop; these adapters do not replace it.

The CLI's tool restriction is not a filesystem sandbox or model attestation.
Only the Claude execution transport is implemented; do not infer equivalent
Codex/AGY audit executors from their session codecs. Host deployment remains
separate; the OS owns the scoped real-pilot receipt and its limitations.

Run the focused tests with:

```text
python -B -m pytest -q tests/test_session_and_audit_drivers.py tests/test_tool_hooks.py
```

The tests make no model calls. No new runtime dependencies.
