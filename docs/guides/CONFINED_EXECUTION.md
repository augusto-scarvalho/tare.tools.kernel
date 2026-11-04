# Confined execution and producer receipts

Windows hosts can also inject `WindowsMxcProfile` or `CodexSandboxProfile` with
an immutable `MxcPolicy(read_only, read_write, denied, registry_read=False)`.
The native Tare factory encodes Microsoft's PSEC 1.0/1.1 schema using FlatBuffers,
loads only System32 `processmodel.dll`, retains its security environment until
process cleanup, and launches suspended with an explicit three-handle list.
It changes no project ACLs. The existing WindowsJob assigns and resumes the child
and owns process/memory budgets, cancellation, timeout and descendant cleanup.
Both Windows factories cap the complete descendant tree at 16 processes,
256 MiB per process and 512 MiB total. The process count includes Python test
runners, Git launchers, native shells and transport helpers.
Only explicit local directories, exact regular-file scopes and an admitted
absolute workload executable are accepted; no launch error retries outside the
sandbox. File scopes never grant their parents. Missing file paths remain exact
scopes; native command creation can still be denied without a writable parent.
The model's scoped edit tool can create a declared file before verification.
Scope types, reparse aliases, hardlinks and cwd are revalidated before launch.

The host may bind `verification_seconds` separately from `seconds` for exact
registered verifiers. Free commands retain the shorter bound, and both remain
limited by execution control. Native search keeps one parent result while its
inventory and match subprocesses have separate once-only command claims; an
interruption never turns a partial search into a proved absence or permits
replaying a raw command.

`staged_effects=True` is a host-only recovery contract: the caller must confine
every write to its disposable candidate/scratch and deny network. A command
stopped by an output/time bound may then return failed after proved cleanup,
with possible partial effects and invalidated read/check observations. The
agent can inspect that staging area; no command is automatically replayed and
application still belongs to the host's verification/review gate. This is off
by default. Cancelled/stopped control, isolation violations and uncertain
cleanup remain unknown; it makes no claim about external command effects.

An optional `enumerate_only` tuple grants directory query/list access without file
content reads, including volume-root metadata needed by path safety checks. It
uses PSEC 1.1's `fs_enumerate` slot and requires both native support flag `0x4`
and contract version 1.1. These paths are part of the finished policy digest and
are revalidated before launch. Unsupported hosts fail without broader read
grants or ACL changes. Enumeration can expose directory entry names; it never
admits an executable or file contents by itself. Read/write/deny scopes still
refuse volume roots. See Microsoft's [PSEC schema](https://github.com/microsoft/mxc/blob/ad54c69452147bd4ca15d027388ad6074903dbff/external/windows-sdk/ProcessSecurityEnvironment.fbs).

The optional Codex factory consumes the public `codex sandbox` typed permission
state with MXC forced, restricted filesystem entries and denied network. The host
supplies an absolute CLI path and its expected SHA-256. Managed installation
junctions are resolved once; later retargeting or changed executable bytes refuse
dispatch. It records the launcher PID and requested-policy digest, without
claiming these prove the workload's native identity. Codex 0.162.1 was qualified
with owned Windows fixtures. Its MXC backend always admits `registryRead`, so the
adapter requires that explicit policy capability and refuses a narrower request.
It calls no model and depends on no Codex provider session or API key.

Native Codex command review uses a separate ephemeral MCP bridge. Its sole tool
advertises the exact host-bound command/cwd for review and accepts only `{}`;
the forwarding model cannot replace command text, paths or hashes. Native
approval must still correlate to that one tool item, and the host compares the
dispatch against its original packet and revalidates the action before effect.
Denied or malformed review never authorizes execution. Failure receipts retain
only structural diagnostics; no model reasoning is recorded there.

Owned controls passed for both launchers: permitted input reads and scratch
writes; denial of outside reads/writes, read-only writes and protected metadata
writes; denial against a reachable host loopback listener; cancellation and timeout
with descendant cleanup. These results cover the exercised Windows build/profile,
not arbitrary hostile code, hard disk quotas or other platforms. Microsoft's
[MXC SDK](https://github.com/microsoft/mxc) is an early preview; the fixed ABI
matches revision `ad54c69452147bd4ca15d027388ad6074903dbff`. The native factory
requires the PSEC exports, and explicit deny rules require native deny support.
No installation enables these factories by default. Classification, review,
producer signing and settlement retain their existing separate owners.

`ConfinedWorkspace` is an explicitly injected Windows LPAC process factory for
the existing `run_process` supervisor. It provisions a fresh host-owned directory,
copies only supplied input/runtime bytes, grants its fresh package SID read/execute
on those copies and modify access to scratch, and never edits installed runtimes,
operator homes or existing project ACLs. Its constructor refuses existing roots,
unsafe names, links, hardlinks and unavailable platforms. No failure enables an
ordinary Popen fallback. Terminal/PTY use is outside this first profile.

The host binds it before any request using
`NativeCapabilities.bind_command_confinement(profile, profile.environment, profile.seconds)`.
The named verifier must use an absolute admitted executable in the copied runtime
and the input root as its launch directory. Existing local execution remains the
default when no profile is injected. This profile is not enabled by installation.

Before resume, Kernel verifies the native AppContainer SID and exact capability
SID set. Creation always requests LPAC opt-out of ALL_APPLICATION_PACKAGES;
AccessCheck confirms denial against that generic grant and admission against its
own SID. Class 46 does not provide a working Win32 query on qualified builds, so
this combination is behavioral evidence, not a purpose-built IsLPAC query.
The mechanisms follow [Microsoft's LPAC launch contract](https://learn.microsoft.com/en-us/windows/win32/secauthz/implementing-an-appcontainer)
and [token information definitions](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ne-winnt-token_information_class).

Default profiles request no capabilities. Python compatibility separately requires
`system_registry_read=True`: the host derives only `registryRead`, records its SID
in the profile identity, and verifies it in the child token. This expands common
registry read access, must be part of the threat scope, and grants no network
capability. Local controls include a protected fixture registry key and socket
connection denial. The [capability derivation API](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-derivecapabilitysidsfromname)
owns neither filesystem grants nor assessment authority.

The native Job bounds active processes to four, per-process memory to 256 MiB and
aggregate memory to 512 MiB. The existing supervisor handles cancellation,
timeout, output bounds and descendant cleanup. Scratch and fresh AppContainer
storage are monitored for 512 entries, 64 files and 8 MiB; collection separately
bounds all exported bytes and each stdout/stderr file. These disk/output checks
can overshoot between polls and are not hard filesystem quotas.

Collection requires the matching process ID, native identity, unchanged copied
manifests and proved cleanup, then snapshots complete exported files without
following reparse points or accepting hardlinks. The host closes the fresh
AppContainer and revokes only its own grants; failed cleanup remains explicit
and retryable. Workspace data stays with its custody owner.

`assurance_plane.producer_receipts` signs a canonical manifest with an injected
Ed25519 private key. The executor receives neither that key nor an authority to
choose the admitted public-key registry. Consumers pass their own expected
action/session/effect/binding/fence/run/source/environment/runtime/profile subject,
their own admitted key registry and every exported byte. Missing, extra, changed
or self-attested output is rejected. Stdout, stderr and process status are artifacts.
A signature proves producer identity and bytes, not correctness or permission to
publish. Execution replay prevention stays with existing durable EffectReceipts;
settlement and consumption remain owner responsibilities.

This is a controlled Windows profile, not certification for arbitrary third-party
commands. Windows retains ambient resources granted to restricted application
packages, common registry reads broaden the compatibility profile, and this
implementation adds no hard disk quota, VM isolation or Linux/macOS profile.
Those boundaries must be qualified before a broader deployment. Provider-model
classification, review calibration and native Codex review are separate ports.
