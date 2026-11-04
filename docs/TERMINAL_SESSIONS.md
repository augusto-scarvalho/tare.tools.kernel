# Host-owned terminal sessions

`compute_plane.terminal_sessions.TerminalSessions` owns command processes for a
long-lived caller. Its lifetime can span several Runtime turns. The composing
OS host must keep the instance alive, authorize commands and provide the working
directory and environment explicitly. The development OS TUI now composes the
authenticated host and native tools described below; a visual terminal renderer
remains pending.

## Public operations

| Operation | Behavior |
|---|---|
| `start(owner, request_id, argv, cwd, env, timeout=300, mode='pipes', columns=80, rows=24)` | Start an argv command without implicit shell parsing and return a handle immediately. Keywords are required. |
| `read(owner, handle, cursors=None, limit=16384)` | Return status and an independent page from each output stream. Does not consume another reader's output. |
| `send(owner, handle, request_id, text)` | Queue bounded UTF-8 input without blocking on an unread pipe. |
| `close_input(owner, handle)` | Close pipe stdin after previously queued input. Repeated closure is harmless. PTY input requires application-specific control keys or stop. |
| `resize(owner, handle, columns, rows)` | Resize a running PTY; dimensions must be integers from 1 to 1000. |
| `stop(owner, handle)` | Stop the owned process tree and return observed cleanup status. May wait for cleanup. |
| `list(owner, include_completed=False)` | Return owner-specific process summaries without output. |
| `close_owner(owner)` | Stop the owner's currently registered commands, preserving other owners. |
| `close()` / context manager exit | Stop all commands and refuse success if cleanup remains unknown. |

The API accepts an argv array. PowerShell, Bash, Git, test runners, build tools
and SSH are normal executables selected by the caller. Shell syntax requires an
explicit shell executable and its command argument. The caller must decide
which environment variables and credentials those commands may inherit.

`owner` is a composition-selected session identity, not an authentication token.
Never accept an arbitrary model-supplied owner as authorization. Wrong-owner and
unknown handles both return `terminal_handle_unavailable`. Handles are random,
in-memory identities; no operation adopts a PID from disk or another host.

## Output and input semantics

Each output stream retains a bounded Unicode tail (262144 characters by default).
Snapshots and listings include `command`, a 240-character display label using
the executable basename. It is not a quoted executable command or a complete
argument record; it does not contain the child environment.
UTF-8 is decoded incrementally; invalid bytes use replacement characters. Cursors
count decoded characters separately for stdout and stderr, rather than bytes or
terminal cells. Each page has `text`, `cursor`, `end`, `truncated` and `more`.
A slow reader receives an explicit loss indication if its cursor precedes the
retained tail. Output is drained continuously even with no attached reader.
Carriage returns and escape sequences are preserved; display layers must render
them safely. Relative ordering between stdout and stderr is not guaranteed.

Input is bounded to 65536 pending bytes and 32 queued writes by default. A
successful `send` first reports `queued`; replaying that same request reports
its current state. `written` means bytes were flushed to the pipe, **not** that
the child consumed or acted on them. A partial/failed write reports `unknown`;
queued input discarded during shutdown reports `not_written`.

The start request ID and input request IDs prevent repeated effects within the
same host lifetime. Reusing an ID with different arguments is refused. Completed
and failed start receipts are retained; a failed executable launch is not silently
retried under its old ID. History has explicit bounds (1024 command records,
4096 input receipts per command); exhausted storage refuses new effects instead
of discarding their identities. Identical requests from different owners are
independent. There is no cross-host or crash-recovery exactly-once guarantee.

`may_have_executed` distinguishes failure before a process was obtained from
failure after a child might have started producing effects. A successful cleanup
does not prove an interrupted command left files or external systems unchanged.
The native capability composition must retain uncertainty for post-spawn startup
failures and never infer a safe retry merely from `reason='start_failed'`.

## Lifetime and containment

The default command deadline is 300 seconds; an explicitly supplied `None`
allows a persistent command until completion, stop or host shutdown. The host
can close or switch a UI panel without closing its command owner. Ending one
agent turn need not close the process supervisor.

Windows starts a suspended child and assigns it to the existing kill-on-close
Job Object before resuming it. Pipe mode on POSIX creates a dedicated process
group; Linux PTY mode creates a controlling-terminal session and also cleans
the additional foreground/background groups a shell creates inside it. Normal
leader exit, deadline and explicit stop all clean up descendants and drain pipes.
A leader that exits zero while leaving a child is not sufficient cleanup evidence.
Unknown cleanup blocks new starts in that host. Cleanup
observes the final process tree and reader state after bounded joins, including
when the join consumed the wait budget. This final observation adds no waiting;
remaining processes or readers still leave cleanup unknown. Linux terminal-session cleanup
and POSIX process groups do not contain deliberate daemonization into another
session; abrupt POSIX host death has no parent-death
guarantee. Working directory and process containment do not impose filesystem
or network access restrictions.

## PTY behavior and remaining composition

Select `mode='pty'` for a real terminal on Windows or Linux. Other PTY platforms
are explicitly unsupported; pipe mode keeps its previous POSIX support. PTY
stdout and stderr are merged into the `stdout` terminal stream. Terminal escape
sequences, echo and carriage returns are preserved for a terminal renderer.
The `rows` and `columns` receipt fields report accepted terminal geometry.

`send` forwards UTF-8 text, including control characters such as Ctrl+C. Their
meaning belongs to the child/terminal mode. For example, Ctrl+D produces EOF at
the start of an empty line in a normal Linux cooked terminal; it is not a
portable replacement for closing a pipe. Closing the input pipe is therefore
refused for PTYs. Use an application's exit command or the explicit stop action.

ConPTY uses invalid startup stdio handles to prevent redirected host output
from bypassing the terminal. Its reader continues draining during terminal
shutdown; native resizing does not hold the output lock. Linux acquires its
controlling tty in a fresh Python exec trampoline before replacing that process
with the requested executable, without `preexec_fn` in a threaded parent.

Background commands survive consumer turns only while their owning host instance
remains alive. The OS composes command permissions and TUI actions. The Runtime
continues to own the agent loop.

Platform references: [Microsoft's ConPTY lifecycle](https://learn.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session)
and [Python's process creation/thread restrictions](https://docs.python.org/3/library/subprocess.html).

## Authenticated host and native agent tools

`compute_plane.terminal_host.TerminalHost` owns one supervisor and a bounded
loopback listener. Trusted composition calls `bind(owner, repositories, env)`
and passes the returned descriptor directly to `TerminalClient`. The descriptor
contains an ephemeral port and bearer credential; keep it outside model context,
journals and saved session configuration. Requests cannot choose an owner,
environment or arbitrary initial cwd. A grant cannot be retargeted, even after
revocation. Selected roots determine working directories; these commands retain
host filesystem and network access and do not constitute a repository sandbox.

The transport uses length-delimited JSON with request/response bounds, message
deadlines and a bounded connection count. It never retries an inconclusive
exchange. `describe` lets composition verify the conversation and project grant
before constructing its native capability port. Shutdown closes connections and
commands; reopening a host does not adopt old PIDs, credentials or handles.

`compute_plane.native_terminal.NativeTerminal` implements the Runtime effect port
for `terminal_start/read/input/stop/list/resize/eof`. Admit those exact names in
`NativeCapabilities` and provide the port explicitly. Other roles and existing
default capability sets do not gain terminal access. Start/input identities come
from Runtime effect IDs, not model-supplied arguments. Native read returns bounded
pages and can wait for output while honoring cancellation. A running process is
not a completed test, and an accepted input write is not proof of consumption.

Native command mutations invalidate earlier read and verifier observations,
including during journal restoration. Transport loss after a possible mutation
returns `unknown`, allowing Runtime reconciliation rather than an automatic retry.
Tools do not claim to undo command effects or preserve commands after host death.

## Local qualification

`tests/test_terminal_sessions.py` exercises real child processes, UTF-8 split
across writes, pre-completion output, stdin/EOF, duplicate requests, bounded
output, owner separation, blocked input, deadlines and descendant cleanup after
both cancellation and exit zero. Fault injection covers failed thread startup
and Windows Job assignment. No model, remote SSH host or publication is used.

Local results on 2026-09-15: 18 tests passed on Windows/Python 3.12; 16 passed
on Linux/WSL/Python 3.12, with the two Windows-specific fault cases skipped.
The external `tui-experience-20260915` experiment lot contains the process
receipts and logs (executions 19 and 20); both supervisors verified cleanup.
The WSL runner uses in-memory pytest capture and omits optional pytest `current`
symlinks, which Windows storage accounting cannot stat on the shared drive.
These are focused local checks; the coordinated release CI remains pending.

PTY qualification in the same lot: execution 24 passed 27 Windows tests with
two Linux-specific skips. After the resize/output-lock adjustment, execution 25
passed all 10 Windows PTY cases with those same two skips, including concurrent
output/resizing and a real PowerShell `Read-Host` exchange. Execution 26 passed
27 Linux tests with three Windows-specific skips, including real Bash input,
Ctrl+D, controlling-terminal ownership and cleanup of an extra process group
that ignores SIGHUP. All final experiment supervisors verified cleanup. These
tests exercise the Kernel API; they do not attest a TUI terminal renderer,
authenticated transport, remote SSH lifecycle or native model tool integration.

Authenticated transport/native tool qualification: execution 31 passed 51 Windows
tests across host, native terminal and existing native file capabilities;
execution 32 passed 11 Linux host tests, including two separate consumer processes
and a discarded response with the original start ID replayed without duplication.
Execution 36 rechecked the native terminal cases after classifying a lost resize
response as an uncertain mutation, using a real PTY resize. No implicit retry.
The paired OS test crossed TUI, HTTP inference fixture, CLI and this host across
two user messages, then verified cleanup on screen closure (executions 30/34/35).
All experiment wrappers verified cleanup. Visual terminal controls and remote SSH
remain unqualified, and no release CI was run.
