"""Bounded host execution; descendants are contained even after an exit-zero timeout.

Windows children inherit a kill-on-close Job Object. The initial process is
assigned while suspended, before it can create an uncontained child. POSIX uses
a new process group (deliberate daemonisation is outside this containment).
Output goes to files so a descendant holding a pipe cannot stall communicate().
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
from threading import Thread, Event
import time


def host_executable(name, *, required=False):
    """Resolve a native host tool from absolute PATH entries, never implicit cwd.

    Absolute PATH entries remain operator host configuration. Keep launcher links
    intact so vendor updates can replace their target without pinning an old release.
    Windows passive tools use .exe, never implicit batch shims.
    """
    if Path(name).name != name:
        raise ValueError('host_executable_requires_basename')
    if os.name == 'nt' and not Path(name).suffix:
        name += '.exe'
    current = Path.cwd().resolve()
    for entry in os.get_exec_path():
        folder = Path(entry)
        try:
            if not folder.is_absolute() or folder.resolve() == current:
                continue
            candidate = folder/name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.absolute())
        except (OSError, RuntimeError):
            continue  # An inaccessible or looping PATH entry cannot hide a later host tool.
    if required:
        raise FileNotFoundError('Native host executable not available on absolute PATH: '+name)
    return None


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        if process.poll() is not None:
            return
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if process.poll() is None:
            process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        # The leader can exit on TERM while a descendant ignores it.
        # Always finish cleaning the dedicated group, even after leader exit.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class WindowsJob:
    def __init__(self, *, resource_limits=None):
        import ctypes as c
        from ctypes import wintypes as w
        self.c = c
        self.k = c.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'TerminateJobObject': ([w.HANDLE, w.UINT], w.BOOL),
            'QueryInformationJobObject': ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
            'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'IsProcessInJob': ([w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)], w.BOOL),
            'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
            'CloseHandle': ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.k, name); fn.argtypes = args; fn.restype = result

        class Basic(c.Structure):
            _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64), ('flags', w.DWORD),
                        ('min_ws', c.c_size_t), ('max_ws', c.c_size_t), ('active_limit', w.DWORD),
                        ('affinity', c.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]

        class IO(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in ('read_ops', 'write_ops', 'other_ops', 'read', 'write', 'other')]

        class Extended(c.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', c.c_size_t),
                        ('job_memory', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]

        class Accounting(c.Structure):
            _fields_ = [(name, c.c_int64) for name in ('user', 'kernel', 'period_user', 'period_kernel')]
            _fields_ += [(name, w.DWORD) for name in ('faults', 'total', 'active', 'terminated')]

        self.Accounting = Accounting
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = Extended(); limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if resource_limits is not None:
            if (not isinstance(resource_limits, tuple) or len(resource_limits) != 3
                    or any(type(v) is not int or v <= 0 for v in resource_limits)
                    or resource_limits[0] > 32 or resource_limits[1] > 2 * 1024**3
                    or not resource_limits[1] <= resource_limits[2] <= 4 * 1024**3):
                self.close(); raise ValueError('invalid_windows_job_resource_limits')
            limits.basic.flags |= 0x8 | 0x100 | 0x200
            limits.basic.active_limit, limits.process_memory, limits.job_memory = resource_limits
        if not self.k.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            self.close(); raise c.WinError(c.get_last_error())

    def assign_and_resume(self, proc):
        if not self.k.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise self.c.WinError(self.c.get_last_error())
        resume = self.c.WinDLL('ntdll').NtResumeProcess
        resume.argtypes = [self.c.c_void_p]; resume.restype = self.c.c_long
        status = resume(int(proc._handle))
        if status != 0:
            raise OSError(f'NtResumeProcess failed: {status}')

    def active(self):
        info = self.Accounting()
        if not self.k.QueryInformationJobObject(self.handle, 1, self.c.byref(info), self.c.sizeof(info), None):
            raise self.c.WinError(self.c.get_last_error())
        return info.active

    def _member_handles(self):
        """Pin current job members before termination, never reopen a historical PID."""
        from ctypes import wintypes as w
        c, handles = self.c, []
        class Members(c.Structure):
            _fields_ = [('assigned', w.DWORD), ('listed', w.DWORD), ('pids', c.c_size_t * 4096)]
        members = Members()
        if not self.k.QueryInformationJobObject(self.handle, 3, c.byref(members), c.sizeof(members), None):
            raise c.WinError(c.get_last_error())
        if members.listed != members.assigned or members.listed > 4096:
            raise OSError('windows_job_member_inventory_incomplete')
        try:
            for pid in members.pids[:members.listed]:
                handle = self.k.OpenProcess(0x101000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
                if not handle:
                    error = c.get_last_error()
                    if error == 87:  # The observed process already exited and its PID no longer exists.
                        continue
                    raise c.WinError(error)
                member = w.BOOL()
                if not self.k.IsProcessInJob(handle, self.handle, c.byref(member)):
                    error = c.get_last_error()
                    self.k.CloseHandle(handle)
                    raise c.WinError(error)
                if member.value:
                    handles.append(handle)
                else:
                    self.k.CloseHandle(handle)  # PID reuse cannot turn a foreign process into our member.
            return handles
        except BaseException:
            for handle in handles:
                self.k.CloseHandle(handle)
            raise

    def terminate(self):
        handles = self._member_handles()
        try:
            if not self.k.TerminateJobObject(self.handle, 1):
                raise self.c.WinError(self.c.get_last_error())
            # ActiveProcesses can reach zero before an exiting member's handle is signaled.
            # Share one finite wait budget across the pinned members; callers still check accounting.
            until = time.monotonic() + 5
            for handle in handles:
                state = self.k.WaitForSingleObject(handle, max(0, int((until-time.monotonic())*1000)))
                if state == 258:
                    raise TimeoutError('windows_job_member_termination_unconfirmed')
                if state != 0:
                    raise self.c.WinError(self.c.get_last_error())
        finally:
            for handle in handles:
                self.k.CloseHandle(handle)

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle); self.handle = None


def _group_live(pgid):
    # Linux /proc distinguishes harmless unreaped zombies from running children.
    if Path('/proc').is_dir():
        for folder in Path('/proc').iterdir():
            if folder.name.isdecimal():
                try:
                    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
                    if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                        return True
                except (FileNotFoundError, ProcessLookupError):
                    pass
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def run_process(argv, *, cwd, dest, timeout, env=None, stdin=b'', cancelled=lambda: False, max_output=2097152, process_factory=None):
    """Return only after cleanup; unknown cleanup is a hard stop for the runner."""
    if timeout <= 0:
        raise ValueError('positive wall budget required')
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    (dest / 'stdin.txt').write_bytes(stdin)
    started = time.perf_counter(); job = None; proc = None
    result = {'exit_code': None, 'timeout': False, 'cancelled': False, 'output_limit': False, 'cleanup_verified': False,
              'containment': 'windows-job' if os.name == 'nt' else 'posix-process-group'}
    if cancelled():
        result.update(cancelled=True, cleanup_verified=True, wall_ms=0)
        return result
    try:
        if os.name == 'nt':
            resource_limits = getattr(process_factory, 'job_limits', None)
            job = WindowsJob(resource_limits=resource_limits) if resource_limits is not None else WindowsJob()
            if resource_limits is not None:
                result['resource_limits'] = list(resource_limits)
        with (dest / 'stdin.txt').open('rb') as inp, (dest / 'stdout.txt').open('wb') as out, (dest / 'stderr.txt').open('wb') as err:
            factory = process_factory if process_factory is not None else subprocess.Popen
            proc = factory(argv, cwd=cwd, env=env, stdin=inp, stdout=out, stderr=err,
                **({'creationflags': 0x4} if job else {'start_new_session': True}))
            result['pid'] = proc.pid
            confinement = getattr(proc, 'confinement_receipt', None)
            if confinement is not None:
                result['confinement'] = dict(confinement)
            if job:
                job.assign_and_resume(proc)
            try:
                until = time.monotonic() + timeout
                while proc.poll() is None:
                    resource_check = getattr(process_factory, 'resource_check', None)
                    if resource_check is not None:
                        try:
                            valid = resource_check() is True
                        except Exception:
                            valid = False
                        if not valid:
                            result['isolation_violation'] = True
                            break
                    if cancelled():
                        result['cancelled'] = True
                        break
                    if time.monotonic() >= until:
                        result['timeout'] = True
                        break
                    if out.tell() + err.tell() > max_output:
                        result['output_limit'] = True
                        break
                    time.sleep(.02)
                result['output_limit'] = out.tell() + err.tell() > max_output
                result['exit_code'] = proc.poll()
            except subprocess.TimeoutExpired:
                result['timeout'] = True
            finally:
                # Always clean up, including a host's partial result with exit 0.
                result['descendants_at_exit'] = job.active() if job else int(_group_live(proc.pid))
                if job:
                    job.terminate()
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                proc.wait(timeout=5)
                until = time.monotonic() + 5
                while time.monotonic() < until:
                    if (job.active() == 0) if job else not _group_live(proc.pid):
                        result['cleanup_verified'] = True
                        break
                    time.sleep(.02)
    finally:
        try:
            # Also handles assignment/resume failures before proc.wait().
            if proc and proc.poll() is None:
                proc.kill(); proc.wait(timeout=5)
        finally:
            try:
                if job:
                    job.close()
            finally:
                close_process = getattr(proc, 'close_process', None) if proc else None
                if callable(close_process):
                    close_process()
    result['wall_ms'] = (time.perf_counter() - started) * 1000
    return result


def run_pipe_process(argv, *, cwd, timeout, stdin=b'', cancelled=lambda: False,
                     max_output=2097152, env=None, on_stdout=None, completed=None):
    """Finite trusted worker exchange, without persisting input or credentials.

    An owned finite writer keeps unread stdin off the supervising thread, including
    Windows where communicate(input, timeout) writes synchronously. The worker
    must bound its own output; the observed ceiling additionally stops violations.
    Reuses host containment, never signals a PID recovered from a prior session.
    With on_stdout, a trusted completed callback may prove a terminal protocol
    before process exit. Its output is still drained and cleanup verified.
    """
    if timeout <= 0 or type(max_output) is not int or max_output < 1:
        raise ValueError('positive execution bounds required')
    started = time.monotonic()
    result = dict(exit_code=None, timeout=False, cancelled=False, output_limit=False,
                  cleanup_verified=False, containment='windows-job' if os.name == 'nt' else 'posix-process-group',
                  started=False)
    if cancelled():
        result.update(cancelled=True, cleanup_verified=True, wall_ms=0)
        return result, b'', b''
    job = proc = writer = None
    readers, buffers, overflow = [], [bytearray(), bytearray()], Event()
    out = err = b''
    try:
        if os.name == 'nt':
            job = WindowsJob()
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            **({'creationflags': 0x4} if job else {'start_new_session': True}))
        result.update(started=True, pid=proc.pid)
        if job:
            job.assign_and_resume(proc)
        source = proc.stdin
        proc.stdin = None  # communicate owns output; the finite writer owns input.
        def write_source():
            try:
                source.write(stdin)
                source.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    source.close()
                except OSError:
                    pass
        writer = Thread(target=write_source, daemon=True, name='tare-http-input')
        writer.start()
        if on_stdout is not None:
            result['observer_failures'] = 0
            def read_output(stream, index):
                observer = on_stdout if index == 0 else None
                try:
                    while chunk := stream.read1(4096):
                        room = max(0, max_output + 1 - len(buffers[index]))
                        buffers[index].extend(chunk[:room])
                        if len(buffers[0]) + len(buffers[1]) > max_output:
                            overflow.set()
                        if observer is not None and not overflow.is_set():
                            try:
                                observer(chunk)  # Trusted bounded observer; never owns effects.
                            except Exception:
                                result['observer_failures'] += 1
                                observer = None
                finally:
                    stream.close()
            for index, stream in enumerate((proc.stdout, proc.stderr)):
                reader = Thread(target=read_output, args=(stream,index), daemon=True, name='tare-http-output')
                readers.append(reader); reader.start()
        while True:
            if cancelled():
                result['cancelled'] = True
                break
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                result['timeout'] = True
                break
            if on_stdout is not None:
                if overflow.is_set():
                    result['output_limit'] = True
                    break
                if proc.poll() is not None:
                    result['exit_code'] = proc.returncode
                    break
                if completed is not None and completed() is True:
                    # The trusted adapter proved its terminal protocol boundary.
                    # Cleanup still drains the pipes and verifies the owned tree.
                    result['protocol_complete'] = True
                    break
                time.sleep(min(.02, remaining))
                continue
            try:
                out, err = proc.communicate(timeout=min(.02, remaining))
                result['exit_code'] = proc.returncode
                break
            except subprocess.TimeoutExpired as exc:
                out, err = exc.output or b'', exc.stderr or b''
                if len(out) + len(err) > max_output:
                    result['output_limit'] = True
                    break
    finally:
        try:
            if proc:
                if job:
                    job.terminate()
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                # Termination releases unread stdin and descendant-held pipes.
                # A failure here propagates: cleanup is never declared by timeout.
                if on_stdout is None:
                    out, err = proc.communicate(timeout=5)
                else:
                    for reader in readers:
                        reader.join(timeout=5)
                    out, err = (bytes(b) for b in buffers)
                proc.wait(timeout=5)
                if writer:
                    writer.join(timeout=5)
                until = time.monotonic() + 5
                while time.monotonic() < until:
                    if (job.active() == 0) if job else not _group_live(proc.pid):
                        result['cleanup_verified'] = (writer is None or not writer.is_alive()) and all(not r.is_alive() for r in readers)
                        break
                    time.sleep(.02)
        finally:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            if job:
                job.close()
    result['output_limit'] |= len(out) + len(err) > max_output
    result['wall_ms'] = (time.monotonic() - started) * 1000
    return result, out[:max_output + 1], err[:max_output + 1]
