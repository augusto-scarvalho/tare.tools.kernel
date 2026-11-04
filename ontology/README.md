# Kernel ontology

This directory is the canonical, repository-owned vocabulary for the concepts
of `tare.tools.kernel`, governed by
[SPEC-KERNEL-001](../docs/SPEC-KERNEL-001.md). A concept here is vocabulary, not
proof of implementation: the specification's implementation status says which
criteria the code meets.

- `domain_ontology.yaml` is authoritative for Kernel concepts.
- Other repositories may keep a pinned pointer to this file, but must not copy
  its payload.
- Performance thresholds and host-specific capability claims belong in specs,
  tests and execution receipts rather than in concept definitions.

The vocabulary is intentionally small. A concept is added only when it is
stable, useful for deterministic joins and governed by a Kernel decision or
specification.
