# Kernel contributor instructions

Kernel owns its five microkernel planes, provider adapters, process containment
and bounded execution ports. Read `docs/SPEC-KERNEL-001.md` and the relevant
owner contract before changing behavior. `tare.tools.agent-runtime` owns the
tactical loop; OS owns settlement; other satellites own their documents.
Change each owner in its own repository. The retired `tare.tools.harness` is
frozen and must not be imported, copied or resumed.

Keep ordinary tests offline and independent of operator storage or historical
records. See `docs/guides/REPOSITORY_CONTENT.md` for private record custody and
`docs/guides/NATIVE_MODEL_ADAPTERS.md` for maintained native-adapter behavior.
Preserve source hashes and original qualification outcomes. Live provider calls,
deployment changes and model downloads need separate operator authorization.

Write code, comments, prompts and documentation in English; agents still reply in
the user's language. The Portuguese version for Brazilian readers lives only in
`README.pt-BR.md`: update it with its English source.

Add an Unreleased changelog entry for material changes without rewriting older
entries. Run relevant bounded tests locally; complete CI runs the core and
source-integration suites on Windows and Linux with pinned dependencies.
Integrate main through a reviewed PR after the complete CI checks pass.
