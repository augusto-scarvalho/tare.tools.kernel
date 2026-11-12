# Documentation index — tare.tools.kernel

The Kernel executes model calls and their effects for tare.tools.os. Start with the
[README](../README.md); this page lists the maintained documents.

## Planes

* **Compute plane (`compute_plane/`):** model adapters, mediated tools, linked-worktree
  qualification, a credential-minimal child environment and bounded checks. A hard
  filesystem or network sandbox is not assumed.
* **Assurance plane (`assurance_plane/`):** Ed25519 producer receipts that bind a
  producer to output bytes; they do not certify correctness.
* **Control, data and experience planes:** empty packages kept from the original
  five-plane plan. State, landing, leases and the operator interfaces live in tare.tools.os.

## Specification and contracts

* **[SPEC-KERNEL-001.md](SPEC-KERNEL-001.md):** five-plane boundaries, CAS concurrency,
  honest isolation and execution-bound receipts, with what is implemented today (the
  plane boundary and CAS persistence are not). It governs the Kernel ontology.
* **[GPU_LEASE_CONTRACT.md](GPU_LEASE_CONTRACT.md):** contract `gpu-lease/1` and `/2` for a
  shared local GPU: what the node reports (busy, expected wait, reservations) and what the
  Kernel does with it.

## Hooks

* **[CANONICAL_TOOL_HOOKS.md](CANONICAL_TOOL_HOOKS.md):** configuration and qualification of
  the Claude Code, Codex and Antigravity adapters for the SpecGraph-first policy.
* **[SCOPED_PROJECT_CONTEXT_HOOKS.md](SCOPED_PROJECT_CONTEXT_HOOKS.md):** on-demand SpecGraph
  context, durable events, bounded host diagnostics and observed verification bytes.
* **[SESSION_AND_AUDIT_ADAPTERS.md](SESSION_AND_AUDIT_ADAPTERS.md):** session lifecycle codecs
  and the Claude audit driver.

## Native execution

* [Native model adapters](guides/NATIVE_MODEL_ADAPTERS.md)
* [Vendor guidance: what a vendor CLI adds on its own](VENDOR_GUIDANCE.md)
* [Gemini compatibility usage](guides/GEMINI_COMPAT_USAGE.md)
* [Native input attachments](NATIVE_ATTACHMENTS.md)
* [Invalid edit hashes and nullable arguments](NATIVE_HASH_DIAGNOSTIC.md)
* [Host-owned terminal sessions](TERMINAL_SESSIONS.md)

## Action control and confinement

* [Exact command mediation](guides/ACTION_CONTROL.md)
* [Confined execution and producer receipts](guides/CONFINED_EXECUTION.md)

## Optional choice workers

* [Hosted Jev](JEV_CHOICE.md), [resident LitJEV](LITJEV_CHOICE.md),
  [OpenJEV](OPENJEV_CHOICE_WORKER.md) and [Von](VON_CHOICE.md)

## Ontology

* **[`../ontology/domain_ontology.yaml`](../ontology/domain_ontology.yaml):** normative concepts
  and invariants for federated discovery; external catalogs keep pinned pointers. A concept's
  presence does not prove that a host implements, runs or delivers it.

## Repository maintenance

* [Changelog policy](CHANGELOG_POLICY.md)
* [Distributed content and private records](guides/REPOSITORY_CONTENT.md)
