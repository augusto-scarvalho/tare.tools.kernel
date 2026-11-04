# Scoped project context hooks

An enabled repository consumes the published SpecGraph snapshot as bounded advice.
SpecGraph owns extraction, indexing and reconciliation; Kernel selects the declared
root and normalizes native events. OS composes explicit Work/lifecycle context.
No hook infers Work from prompt, branch or directory.

Install a matching SpecGraph build providing `specgraph.hook_context` and
`specgraph.lifecycle` in the same environment as `tare-hook`. Add this to the
repository's existing `specgraph.yaml` and explicitly publish its snapshot:

```yaml
hook_context:
  enabled: true
  role: implementer
  automatic_roles: []
  max_bytes: 4096
  timeout_ms: 50
```

`automatic_roles: []` keeps context on demand: literal searches and explicit payload
`spec_id`/`symbol_id` request it; ordinary reads/edits stay quiet. Omission retains
automatic planner/implementer/auditor/explorer context. Explicit `agent_role` can
change the role. Work selection alone does not request a packet.

Use the registrations in [Canonical tool hooks](CANONICAL_TOOL_HOOKS.md). Modern
Codex `functions.exec_command`/`tools.exec_command`, object/string `apply_patch` and
namespaced subagent calls are normalized at ingress; host matchers must admit the
names that host emits. Installation does not change global hook configuration.
An absolute file selects its nearest declarative root even from a parent cwd.

## Context and lifecycle boundaries

- Root discovery checks known config filenames along a bounded ancestor chain;
  it does not scan source trees or query Library synchronously.
- Root, snapshot, selection and coverage remain visible. A published snapshot
  does not prove instantaneous worktree freshness.
- Output is at most 4096 UTF-8 bytes. Required units stay whole; oversized output
  is neither silently truncated nor marked as delivered.
- Selection is bounded to 64 edit paths and four roots, with overflow recorded;
  context selects at most 16 files per root.
- Successful Post callbacks append/coalesce observations through the SpecGraph SDK.
  They do not index, hash sources, start a worker or launch another process.
- The separate worker or explicit reconciliation acknowledges changes and preserves
  concurrent edits. Periodic SHA reconciliation recovers missed callbacks. The
  adapter does not start a daemon.
- Root/host/session/explicit-agent/role/resume identities deduplicate context.
  Retention is bounded to 128 observation scopes and 128 context identities per
  root; eviction may re-emit advice. Counters saturate rather than growing history.

State defaults to `~/.local/state/tare-hooks`, configurable with
`TARE_HOOK_STATE_DIR`. Candidate-contained state and foreign-root databases are
rejected. Failed event delivery stays conservatively `UNCONFIRMED`; a different
snapshot alone does not prove a raced event was reconciled. Diagnostics fail open.

## Scoped local-model policy

Scoped delegation uses only the external receipt explicitly selected through
`TARE_LOCAL_MODELS_HEALTH_CACHE`, bounded to 4096 bytes:

```json
{"schema":"tare.tools/local-models-health/1","status":"UP","observed_at":1788876000}
```

An independent observer must publish a current Unix timestamp. Only `UP` aged
0–30 seconds can block a frontier spawn. Missing, stale, malformed or future-dated
receipts are unknown and fail open. The synchronous scoped handler performs no
network, LLM or source-scan work; it does not fetch/refresh health. The unscoped
LocalModelsFirstGate, including its bounded probe, remains separate.

## Host diagnostics and qualified scope

`compute_plane.hook_observations.hook_host_status(root)` separates registration,
local adapter invocation, returned context and observed host delivery. It reads
known registration files and at most eight observation rows per host. Callbacks
cannot promote themselves to host-delivery evidence. Session/agent IDs are hashed;
diagnostics do not store prompts or tool arguments.

`observe_codex_delivery(root, transcript)` is explicit and offline. It reads at most
1 MiB from a selected local transcript and retains metadata plus a matching
developer-payload hash. Hooks never call it. `OBSERVED_IN_HOST_TRANSCRIPT` is a local
observation, not an attestation of all sessions, installations or releases.

September 8 qualification observed Codex delivery in the active SpecGraph repository
and real Claude `PreToolUse:Read` in a disposable Windows fixture. Antigravity/Aider
callbacks were manually exercised; native delivery remains unqualified. Upstream
forwarding may normalize names; Kernel does not invent a lost original tool name.

Paired microtasks did not demonstrate the 65% savings aspiration, and wall time
increased. Keep context on demand until end-to-end tasks demonstrate total benefit,
including all agent tokens, cache, retries and time. Historical receipts are owned
by SpecGraph under `docs/evidence/2026-09-08-c7-c8/`; adapter tests do not replace
real-host or effectiveness measurements. No frozen harness integration is involved.

## Managed Work receipt transport

`ObservedWorkProgress.evidence_bytes` carries the exact bounded JSON bytes whose
SHA-256 is `evidence_sha256`. OS/SpecGraph can inspect owner-observed command receipts
without reconstructing them from counts or model prose. Legacy instances default
to `b""`; absence stays explicit. Neither bytes nor hash grant execution authority
or Tier 1 attestation.
