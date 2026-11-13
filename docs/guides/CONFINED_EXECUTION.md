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

These are controlled Windows and Linux profiles, not certification for arbitrary
third-party commands. Windows retains ambient resources granted to restricted
application packages, common registry reads broaden the compatibility profile, and
this implementation adds no hard disk quota, VM isolation or macOS profile.
Those boundaries must be qualified before a broader deployment. Provider-model
classification, review calibration and native Codex review are separate ports.

## Linux: bubblewrap

Linux hosts can inject `LinuxBwrapProfile` with an immutable
`BwrapPolicy(read_only, read_write, denied)` of absolute POSIX paths. It is a
`process_factory` for the same `run_process` supervisor, which keeps its POSIX
process group, timeouts, output bound and cleanup; receipts carry backend
`linux-bwrap`. `require_support()` launches the full profile once and otherwise
raises `OSError` whose message starts with a code: `bwrap_linux_only`,
`bwrap_architecture_unsupported` (x86_64 and aarch64 only), `bwrap_missing`,
`bwrap_prlimit_missing`, `bwrap_too_old` (bubblewrap before 0.8),
`bwrap_userns_refused` or `bwrap_launch_failed`. `bwrap_userns_refused` means the
kernel or a security module refuses unprivileged user namespaces, as stock Ubuntu
24.04 does with `kernel.apparmor_restrict_unprivileged_userns=1`; the message
lists the user-namespace sysctls the host has. The Kernel only reports the cause
and changes no system setting. No failure falls back to a weaker sandbox.

Inside, only the policy's paths exist, each at its own path. The host lists the
system roots (`SYSTEM_ROOTS`: `/usr`, `/bin`, `/sbin`, `/lib*`, `/etc`) in the
policy itself, so the digest names everything bound. bubblewrap adds a fresh
`/proc`, a minimal `/dev` and an empty `/tmp`; the root is read-only. Binds go
shallowest first, so a read-only `.git` inside a writable tree stays read-only;
absent paths are skipped. Every denied path that exists at launch is masked after
the binds, wherever its real file shows through a bound path: mode 000 and
read-only (a tmpfs for a folder, an empty file otherwise), so a read fails with
EACCES rather than returning an empty file. The sandbox has its own user, PID,
IPC and network namespaces, no capabilities, no nested user namespaces
(`--disable-userns`), and dies with its parent. `prlimit` inside sets 1024 tasks
and no core dumps. A seccomp filter assembled in Python and passed by descriptor
refuses (EPERM) sockets other than AF_UNIX, AF_INET, AF_INET6 and AF_NETLINK, the
`io_uring` calls, and syscalls of another architecture: an AF_VSOCK socket reaches
the host through any network namespace, and on WSL2 it can start Windows programs.

Before each launch the profile compares the real path of every scope and the
identity (real path, device, inode) of every denied path with what it saw when
created, and refuses a change (`bwrap_scope_changed`, `bwrap_denied_path_changed`).
Otherwise a check could replace a missing write scope with a link to the home
folder, or rename a folder above a masked secret, and the next launch would bind
or expose it. A terminal on stdio, another cwd or environment, and a program
outside the scopes are refused. bubblewrap itself runs outside the sandbox, so it
starts with an empty environment and hands the check exactly the bound one
(`--clearenv`, `--setenv`): no loader variable can make it load a check's library.

What Linux does not match: there is no memory cap (none is possible without
privilege or a delegated cgroup); the task cap is 1024 rather than 16 processes,
because Linux counts threads; loopback is private rather than denied, so a check's
own servers work while host listeners stay unreachable; `/`, the names of the
folders above each granted path, the whole of `/etc` and host kernel facts in
`/proc` are visible; a host process that replaces a masked secret by rename (an
editor's atomic save) unmasks it for the check running at that moment, and the
next launch is refused. A host without bubblewrap 0.8 or unprivileged user
namespaces (stock Ubuntu 24.04, many containers, WSL1) has no sandbox. Measured on
WSL2 (Ubuntu 24.04, kernel 6.6, bubblewrap 0.9.0); CI's `Linux sandbox
(bubblewrap)` job runs the same tests on a GitHub-hosted Ubuntu 24.04 runner after
lowering the AppArmor restriction there. Other kernels, arm64 and older bubblewrap
are not measured.
