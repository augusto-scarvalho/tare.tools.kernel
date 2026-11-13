# SPEC-KERNEL-001: Kernel planes and bounded execution

- **Status:** `CANONICAL_SSOT`
- **Canonical repository:** `tare.tools.kernel`
- **Governing decision:** none. ADR-001, the original ecosystem plan, was removed in
  pull request #103 because it described a state store and landing notary that this
  repository never contained. This specification now governs the Kernel ontology
  ([`ontology/domain_ontology.yaml`](../ontology/domain_ontology.yaml)).
- **Version:** 1.1.0
- **Relocated from:** `tare.tools.library@d5473e69:specs/SPEC-KERNEL-001.md`

## Purpose

State the criteria the Kernel was planned around (decoupled control, compute, data,
assurance and experience planes, atomic state persistence, honest isolation and
attested output) and record which of them the code meets. The Kernel ships the
compute and assurance planes: bounded execution through provider adapters,
processes and terminals, optional Windows confinement and producer receipts. It keeps
no state store; state persistence lives in tare.tools.os.

## Verifiable acceptance criteria

- **AC-01 — Five-plane boundary enforcement:** compute adapters do not obtain
  network credentials or direct data-plane access without control-plane policy.
- **AC-02 — Single-writer atomic CAS:** state mutations use transactions and
  expected-version checks so concurrent writers cannot silently overwrite one
  another.
- **AC-03 — Honest isolation boundary:** execution is constrained by the
  platform mechanisms actually enabled; unsupported hard sandboxing is never
  claimed.
- **AC-04 — Cryptographic envelope attestation:** execution output carries a
  SHA-256 digest and an identity-bound receipt.

## Implementation status

The criteria above are the original target. This table says what the code in this
repository does today.

| Criterion | What exists | Status |
|---|---|---|
| AC-01 | `control_plane/` and `data_plane/` are empty packages, so there is no control-plane policy to enforce. HTTP model adapters take their endpoint and credentials from the caller (`NativeHttpClient` in `compute_plane/native_models.py`). | Not implemented. |
| AC-02 | The Kernel keeps no state store; the session journal (`compute_plane/native_journal.py`) is append-only. Its only compare-and-swap advances one Git ref (`land_candidate_ref` in `compute_plane/candidate_git.py`, via `git update-ref <ref> <new> <old>`). State persistence lives in tare.tools.os. | Not implemented. |
| AC-03 | Project roots are not a command sandbox. The optional confinement profiles (on Windows `compute_plane/windows_mxc.py`, `codex_sandbox.py`, `confined_workspace.py`; on Linux `linux_bwrap.py`) apply only when a host injects them, and the documentation claims no hard sandbox. | Implemented as a stated boundary; confinement is optional, on Windows and on Linux with bubblewrap. |
| AC-04 | `assurance_plane/producer_receipts.py` signs and verifies Ed25519 receipts over the SHA-256 of supplied output bytes. Nothing here attaches a receipt to every execution output; a host signs what it chooses. | Partial: signer and verifier only. |

The ontology concepts `CASPersistence` and `MonotonicLeaseControl` are shared
vocabulary; their presence does not mean the Kernel implements them.

