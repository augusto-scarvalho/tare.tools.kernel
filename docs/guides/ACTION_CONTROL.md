# Exact command mediation

`compute_plane.action_control` supplies immutable action/authorization envelopes
and `ExactActionGuard`. `action_assessment` supplies advisory classifier/reviewer
ports. Assessment observations have `authority: NONE`; a favorable observation
does not authorize an effect.

The host supplies the authorization verifier, atomic durable claim port, live
Runtime control and fresh snapshot callback. The guard checks exact digests,
session/effect/binding/fence, deadlines and cancellation before consuming the
claim, and revalidates after the claim. A crash or unknown result never retries
the effect. This is mediation, not a filesystem/network sandbox or a signer.

`CommandMediationPort.execute(facts, effect)` is an optional, explicitly injected
host seam. `CommandFacts` contains the executed argv, payload, cwd and environment
identity without raw environment values. The existing execution paths consume it:

- `MediatedTools.command(..., command_mediation=...)` through its constructor;
- `NativeCapabilities(..., command_mediation=...)`, or its host-only
  `bind_command_mediation(port)` before the first effect;
- `TerminalHost.bind(..., command_mediation=...)`, separately for start and input.

Native capabilities add their Runtime session/effect/binding identity to facts.
Terminal facts are selected by the existing owner grant. Terminal input carries
the launch root, not proof of a shell's current directory. An argv-only restricted
profile must refuse interactive input/PTY rather than treating cwd as confinement.
Read/stop/cleanup controls retain their existing owner checks.

Absent an injected port, existing local execution is preserved. A configured
port's denial or exception never falls back to unmediated execution. Ports are
trusted host code and must call the provided effect only through the exact guard.

`NativeCapabilities.claim_action(action, authorization)` delegates to the existing
effect recorder with the active request. `EffectReceipts.claim_action` creates an
exclusive intent file and flushes it before effects. Reopening the recorder cannot
reclaim an existing effect. Partial/crashed records fail closed. Recording a result
retains the intent and cannot erase an already proved result on a later replay.

The OS owns policy, reviewer combinations, public context, source snapshots and
permit issuance. Physical confinement, signing/key custody, producer authenticity,
model calibration and native Codex review integration require separate evidence.

An optional [confined execution profile](CONFINED_EXECUTION.md) now supplies
Windows LPAC execution and host-signed output evidence. It uses the same command
seam and supervisor. Other platforms, terminal input, production key custody,
classifier calibration and native review retain their separate qualification scope.
