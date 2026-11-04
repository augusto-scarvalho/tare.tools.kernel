"""Host-owned command sessions with bounded, incremental pipe or PTY I/O.

The composing host owns this object's lifetime and supplies authorization, cwd
and environment. This is process containment, not a filesystem/network sandbox.
No daemon, durable PID adoption, shell parsing or automatic command replay.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from uuid import uuid4

from .managed_process import WindowsJob, _group_live


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + '_must_be_positive_integer')
    return value


def _key(value):
    if not isinstance(value, str) or not value or len(value) > 256 or '\0' in value:
        raise ValueError('bounded_nonempty_identity_required')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


class _Output:
    """Unicode character cursors; each stream has its own bounded tail."""

    def __init__(self, limit):
        self.limit, self.text, self.end = limit, '', 0
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    def append(self, chunk, *, final=False):
        text = self.decoder.decode(chunk, final=final)
        self.end += len(text)
        self.text = (self.text + text)[-self.limit:]

    def read(self, cursor, limit):
        if type(cursor) is not int or not 0 <= cursor <= self.end:
            raise ValueError('terminal_cursor_invalid')
        start = self.end - len(self.text)
        offset = max(cursor, start)
        text = self.text[offset-start:offset-start+limit]
        return dict(text=text, cursor=offset+len(text), end=self.end,
                    truncated=cursor < start, more=offset+len(text) < self.end)


class _Command:
    def __init__(self, owner, signature, output_chars, input_bytes, max_requests, mode, columns, rows):
        self.id, self.owner, self.signature = uuid4().hex, owner, signature
        self.lock = threading.RLock()
        self.resize_lock = threading.Lock()
        self.done, self.stop_requested = threading.Event(), threading.Event()
        self.input = queue.Queue(maxsize=32)
        self.input_bytes, self.queued_bytes, self.max_requests = input_bytes, 0, max_requests
        self.input_closed, self.input_error = False, None
        self.input_receipts = {}
        self.output = {s: _Output(output_chars) for s in ('stdout', 'stderr')}
        self.process = self.job = self.supervisor = None
        self.readers, self.writer = [], None
        self.started = time.monotonic()
        self.status, self.exit_code = 'starting', None
        self.cleanup_verified, self.error = False, None
        self.reason = None
        self.mode, self.columns, self.rows = mode, columns, rows
        self.may_have_executed = False
        self._terminal_close_attempted = False

    def launch(self, argv, cwd, env, timeout):
        self.timeout = timeout
        self.label = ' '.join([Path(argv[0]).name,*argv[1:]])[:240]
        try:
            if os.name == 'nt':
                self.job = WindowsJob()
            if self.mode == 'pty':
                from .terminal_pty import open_pty
                self.process = open_pty(argv, cwd=cwd, env=env, columns=self.columns, rows=self.rows)
            else:
                self.process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                    **({'creationflags': 0x4} if self.job else {'start_new_session': True}))
            # Conservative across resume/assignment failure: cleanup of a child
            # never proves it produced no effects before startup later failed.
            self.may_have_executed = True
            if self.job:
                self.job.assign_and_resume(self.process)
            for name in self.output:
                reader = threading.Thread(target=self._read, args=(name,), daemon=True,
                                          name='tare-terminal-' + name)
                self.readers.append(reader)
                reader.start()
            self.writer = threading.Thread(target=self._write, daemon=True, name='tare-terminal-input')
            self.writer.start()
            self.status = 'running'
            self.supervisor = threading.Thread(target=self._supervise, args=(timeout,), daemon=True,
                                               name='tare-terminal-supervisor')
            self.supervisor.start()
        except Exception as exc:
            self.error = type(exc).__name__
            self.reason = 'start_failed'
            self._cleanup()

    def _read(self, name):
        stream = getattr(self.process, name)
        try:
            while chunk := stream.read1(4096):
                with self.lock:
                    self.output[name].append(chunk)
        except OSError as exc:
            with self.lock:
                self.error = type(exc).__name__
                self.reason = self.reason or 'output_failed'
            self.stop_requested.set()
        finally:
            with self.lock:
                self.output[name].append(b'', final=True)
            stream.close()

    def _write(self):
        try:
            while not self.stop_requested.is_set():
                try:
                    item = self.input.get(timeout=.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                request_id, data = item
                try:
                    self.process.stdin.write(data)
                    self.process.stdin.flush()
                    state = 'written'
                except (BrokenPipeError, OSError):
                    state = 'unknown'
                    raise
                finally:
                    with self.lock:
                        self.queued_bytes -= len(data)
                        self.input_receipts[request_id]['state'] = state
        except (BrokenPipeError, OSError) as exc:
            with self.lock:
                self.input_error = type(exc).__name__
        finally:
            # Terminating the owned process releases a writer blocked on a full pipe.
            try:
                self.process.stdin.close()
            except OSError:
                pass
            with self.lock:
                self.input_closed = True
                for receipt in self.input_receipts.values():
                    if receipt['state'] == 'queued':
                        receipt['state'] = 'not_written'

    def _supervise(self, timeout):
        try:
            while self.process.poll() is None:
                if self.stop_requested.wait(.02):
                    self.reason = self.reason or 'stopped'
                    break
                if timeout is not None and time.monotonic() - self.started >= timeout:
                    self.reason = 'timeout'
                    break
            self.exit_code = self.process.poll()
        except Exception as exc:
            self.error, self.reason = type(exc).__name__, 'supervision_failed'
        finally:
            self._cleanup()

    def _cleanup(self):
        cleanup_verified = False
        self.stop_requested.set()
        try:
            if self.process is not None:
                if self.job:
                    self.job.terminate()
                    # Assignment may have failed; also reap the suspended leader.
                    if self.process.poll() is None:
                        self.process.kill()
                elif self.mode == 'pty':
                    self.process.terminate_tree()
                else:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self.process.wait(timeout=5)
                if self.exit_code is None:
                    self.exit_code = self.process.returncode
                if self.mode == 'pty' and not self.job:
                    # A child can fork while the first session scan is being
                    # signalled. Finish those groups before joining tty readers.
                    deadline = time.monotonic() + 5
                    while self.process.tree_active() and time.monotonic() < deadline:
                        self.process.terminate_tree()
                        time.sleep(.02)
                terminal_closed = self._close_terminal()
                threads = [*self.readers, *([self.writer] if self.writer else [])]
                deadline = time.monotonic() + 5
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(timeout=max(0, deadline-time.monotonic()))
                while True:
                    active = (self.job.active() if self.job else self.process.tree_active()
                              if self.mode == 'pty' else _group_live(self.process.pid))
                    if not active:
                        cleanup_verified = terminal_closed and all(not t.is_alive() for t in threads)
                        break
                    if time.monotonic() >= deadline:
                        break
                    if self.mode == 'pty' and not self.job:
                        self.process.terminate_tree()
                    time.sleep(.02)
            else:
                cleanup_verified = True
        except Exception as exc:
            self.error = type(exc).__name__
        finally:
            # Also reap a suspended leader if Job termination/assignment failed.
            # This fallback does not attest cleanup of the entire process tree.
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except Exception as exc:
                    self.error = type(exc).__name__
                    cleanup_verified = False
            if self.job:
                self.job.close()
            if self.process is not None and not self._terminal_close_attempted:
                try:
                    self._close_terminal()
                except Exception as exc:
                    self.error = type(exc).__name__
                    cleanup_verified = False
            if self.process is not None:
                # Partial startup can leave streams without reader/writer
                # ownership even when process-tree cleanup remains unknown.
                owners = [self.writer, *self.readers]
                owners += [None] * (3-len(owners))
                for stream, thread in zip((self.process.stdin, self.process.stdout, self.process.stderr), owners):
                    if thread is None or not thread.is_alive():
                        try:
                            stream.close()
                        except OSError:
                            pass
                if self.mode == 'pty':
                    self.process.close_process()
            with self.lock:
                self.input_closed = True
                # Publish completion and outcome together: observers must never
                # treat a still-running snapshot as verified final cleanup.
                self.cleanup_verified = cleanup_verified
                self.status = ('unknown' if not self.cleanup_verified else
                               'failed' if self.reason == 'start_failed' else
                               'stopped' if self.reason else 'exited')
                self.finished = time.monotonic()
            self.done.set()

    def _close_terminal(self):
        if self.mode != 'pty':
            return True
        self._terminal_close_attempted = True
        # A final ConPTY frame needs a reader even after partial startup.
        if not self.process.stdout.closed and (not self.readers or not self.readers[0].is_alive()):
            reader = threading.Thread(target=self._read, args=('stdout',), daemon=True)
            if self.readers:
                self.readers[0] = reader
            else:
                self.readers.append(reader)
            reader.start()
        return self.process.close_terminal()

    def snapshot(self, cursors=None, limit=16384):
        with self.lock:
            cursors = {} if cursors is None else cursors
            if not isinstance(cursors, dict) or set(cursors) - self.output.keys():
                raise ValueError('terminal_cursors_invalid')
            return dict(handle=self.id, status=self.status, exit_code=self.exit_code,
                timeout_seconds=self.timeout, process_timed_out=self.reason == 'timeout',
                command=getattr(self,'label',''),
                reason=self.reason, error=self.error, cleanup_verified=self.cleanup_verified,
                input_closed=self.input_closed, input_error=self.input_error,
                may_have_executed=self.may_have_executed,
                elapsed_ms=round((getattr(self, 'finished', time.monotonic())-self.started)*1000),
                containment='windows-job' if os.name == 'nt' else
                            'linux-terminal-session' if self.mode == 'pty' else 'posix-process-group',
                mode=self.mode, columns=self.columns if self.mode == 'pty' else None,
                rows=self.rows if self.mode == 'pty' else None,
                output={s: out.read(cursors.get(s, 0), limit) for s, out in self.output.items()})

    def send(self, request_id, data):
        raw = data.encode('utf-8')
        if not raw or len(raw) > self.input_bytes:
            raise ValueError('terminal_input_size_invalid')
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            if previous := self.input_receipts.get(request_id):
                if previous['sha256'] != digest:
                    raise ValueError('terminal_input_request_changed')
                return {k: v for k, v in previous.items() if k != 'sha256'}
            if self.input_closed or self.status != 'running' or self.stop_requested.is_set():
                raise ValueError('terminal_input_closed')
            if len(self.input_receipts) >= self.max_requests:
                raise ValueError('terminal_input_request_limit')
            if self.queued_bytes + len(raw) > self.input_bytes or self.input.full():
                raise ValueError('terminal_input_queue_full')
            receipt = dict(request_id=request_id, sha256=digest, state='queued', bytes=len(raw))
            self.input_receipts[request_id] = receipt
            self.queued_bytes += len(raw)
            self.input.put_nowait((request_id, raw))
            return {k: v for k, v in receipt.items() if k != 'sha256'}


class TerminalSessions:
    """Explicit owner-scoped process handles, valid only in this host lifetime.

    Keep one instance in the long-lived composition host to span agent turns.
    Start/input request IDs prevent replay within that lifetime. Their bounded
    receipts are never evicted to make room for another potentially repeated
    effect. An exhausted host must be closed before replacing it.
    """

    def __init__(self, *, max_running=16, max_records=1024, output_chars=262144,
                 input_bytes=65536, max_input_requests=4096):
        for name, value in locals().copy().items():
            if name != 'self':
                _positive(value, name)
        self.max_running, self.max_records = max_running, max_records
        self.output_chars, self.input_bytes = output_chars, input_bytes
        self.max_input_requests = max_input_requests
        self._lock, self._closed = threading.RLock(), False
        self._commands, self._requests = {}, {}

    def start(self, *, owner, request_id, argv, cwd, env, timeout=300,
              mode='pipes', columns=80, rows=24):
        _key(owner); _key(request_id)
        from .terminal_pty import dimensions
        if mode not in ('pipes', 'pty'):
            raise ValueError('terminal_mode_invalid')
        dimensions(columns, rows)
        if (not isinstance(argv, list) or not argv or len(argv) > 256
                or any(not isinstance(v, str) or '\0' in v for v in argv) or not argv[0]
                or sum(len(v) for v in argv) > 65536):
            raise ValueError('terminal_argv_invalid')
        cwd = Path(cwd)
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError('terminal_absolute_cwd_required')
        if (not isinstance(env, dict) or any(not isinstance(k, str) or not k or '=' in k or '\0' in k
                or not isinstance(v, str) or '\0' in v for k, v in env.items())):
            raise ValueError('terminal_explicit_environment_required')
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('terminal_timeout_invalid')
        signature = _digest([argv, str(cwd.resolve()), env, timeout, mode, columns, rows])
        with self._lock:
            if self._closed:
                raise ValueError('terminal_host_closed')
            if handle := self._requests.get((owner, request_id)):
                command = self._commands[handle]
                if command.signature != signature:
                    raise ValueError('terminal_start_request_changed')
                return command.snapshot()
            if any(c.status == 'unknown' for c in self._commands.values()):
                raise RuntimeError('terminal_cleanup_unknown')
            if len(self._commands) >= self.max_records:
                raise ValueError('terminal_record_limit')
            if sum(not c.done.is_set() for c in self._commands.values()) >= self.max_running:
                raise ValueError('terminal_running_limit')
            command = _Command(owner, signature, self.output_chars, self.input_bytes,
                               self.max_input_requests, mode, columns, rows)
            self._commands[command.id] = command
            self._requests[owner, request_id] = command.id
            command.launch(list(argv), str(cwd), dict(env), timeout)
            return command.snapshot()

    def _get(self, owner, handle):
        _key(owner); _key(handle)
        with self._lock:
            command = self._commands.get(handle)
            if command is None or command.owner != owner:
                raise ValueError('terminal_handle_unavailable')
            return command

    def read(self, *, owner, handle, cursors=None, limit=16384):
        _positive(limit, 'terminal_read_limit')
        if limit > 65536:
            raise ValueError('terminal_read_limit_too_large')
        return self._get(owner, handle).snapshot(cursors, limit)

    def send(self, *, owner, handle, request_id, text):
        _key(request_id)
        if not isinstance(text, str):
            raise ValueError('terminal_input_text_required')
        return self._get(owner, handle).send(request_id, text)

    def close_input(self, *, owner, handle):
        command = self._get(owner, handle)
        if command.mode == 'pty':
            raise ValueError('terminal_pty_use_control_key_or_stop')
        with command.lock:
            if not command.input_closed:
                if command.input.full():
                    raise ValueError('terminal_input_queue_full')
                command.input.put_nowait(None)
                command.input_closed = True

    def resize(self, *, owner, handle, columns, rows):
        from .terminal_pty import dimensions
        dimensions(columns, rows)
        command = self._get(owner, handle)
        with command.resize_lock:
            with command.lock:
                if command.mode != 'pty':
                    raise ValueError('terminal_resize_requires_pty')
                if command.status != 'running' or command.stop_requested.is_set():
                    raise ValueError('terminal_pty_closed')
            # Native resize may emit output. Keep the output lock available to
            # its reader while ConPTY processes the geometry change.
            command.process.resize(columns, rows)
            with command.lock:
                command.columns, command.rows = columns, rows
        return command.snapshot()

    def stop(self, *, owner, handle):
        command = self._get(owner, handle)
        command.stop_requested.set()
        command.done.wait(12)
        return command.snapshot()

    def list(self, *, owner, include_completed=False):
        _key(owner)
        if type(include_completed) is not bool:
            raise ValueError('terminal_include_completed_boolean_required')
        with self._lock:
            selected = [c for c in self._commands.values() if c.owner == owner
                        and (include_completed or not c.done.is_set())]
        return [{k: v for k, v in c.snapshot(limit=1).items() if k != 'output'} for c in selected]

    def close_owner(self, owner):
        _key(owner)
        with self._lock:
            selected = [c for c in self._commands.values() if c.owner == owner]
            for command in selected:
                command.stop_requested.set()
        return [self.stop(owner=owner, handle=c.id) for c in selected]

    def close(self):
        with self._lock:
            self._closed = True
            selected = list(self._commands.values())
            for command in selected:
                command.stop_requested.set()
        results = [self.stop(owner=c.owner, handle=c.id) for c in selected]
        if any(not r['cleanup_verified'] for r in results):
            raise RuntimeError('terminal_cleanup_unknown')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
