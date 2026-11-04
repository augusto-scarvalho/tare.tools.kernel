# Distributable sources and private records

Kernel distributes its five planes, provider/process ports, contracts, maintained
guides and offline tests. The current tactical loop belongs to
`tare.tools.agent-runtime`; OS owns settlement. The Aider and aggregate HTTP-session drivers are removed. Their old guides
remain private historical records, not installation instructions.

Dated qualification reports and raw execution receipts have private custody.
[RECORDS_ARCHIVE.json](../RECORDS_ARCHIVE.json) records their original owner
commit, paths, hashes and sizes without copying their content. Historical links
refer to that immutable commit and may require authorized access. Their outcomes
are unchanged; they do not attest the current source or installation.

Before integrating a deletion into an existing checkout, back up its local
records and verify their hashes. Restore them to the original paths after the
update; `.gitignore` excludes those paths from new changes. Do not overwrite a
locally modified record. A clean checkout does not need these files to run tests.

CI validates the current code on Windows and Linux with explicitly pinned owner
dependencies. Opt-in live probes require separately supplied operator
configuration and authorization; normal validation does not contact providers,
download models or read operator storage. Current endpoint setup is documented
by the native-adapter guide and its owning tools.
