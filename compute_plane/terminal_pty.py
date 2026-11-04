"""PTY transport adapters for the Kernel's owned terminal sessions.

Windows uses native ConPTY and a suspended child; the caller must assign its
Windows Job before resuming. POSIX uses a fresh exec trampoline for controlling
tty acquisition, avoiding Python preexec_fn in the threaded parent.
"""
from __future__ import annotations

import errno
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
import signal


def dimensions(columns, rows):
    if any(type(v) is not int or not 1 <= v <= 1000 for v in (columns, rows)):
        raise ValueError('terminal_dimensions_must_be_1_to_1000')


class _PosixReader:
    def __init__(self, stream):
        self.stream = stream

    @property
    def closed(self):
        return self.stream.closed

    def close(self):
        self.stream.close()

    def read1(self, size):
        try:
            return self.stream.read1(size)
        except OSError as exc:
            if exc.errno == errno.EIO:
                return b''  # Linux PTYs signal the final slave close as EIO.
            raise


class PosixPtyProcess:
    def __init__(self, argv, *, cwd, env, columns, rows):
        import pty
        self._lock = threading.Lock()
        self.master, slave = pty.openpty()
        self.process = None
        self.stdin = self.stdout = None
        self.stderr = io.BytesIO()
        try:
            self.resize(columns, rows)
            self.stdin = os.fdopen(os.dup(self.master), 'wb')
            self.stdout = _PosixReader(os.fdopen(os.dup(self.master), 'rb'))
            self.process = subprocess.Popen(
                [sys.executable, '-I', str(Path(__file__).with_name('terminal_pty_exec.py')), *argv],
                cwd=cwd, env=env, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
            self.pid = self.process.pid
        except BaseException:
            for stream in (self.stdin, self.stdout, self.stderr):
                if stream is not None:
                    stream.close()
            self.close_terminal()
            raise
        finally:
            os.close(slave)

    @property
    def returncode(self):
        return self.process.returncode

    def poll(self):
        return self.process.poll()

    def wait(self, timeout=None):
        return self.process.wait(timeout=timeout)

    def kill(self):
        self.process.kill()

    def _members(self):
        members = []
        for folder in Path('/proc').iterdir():
            if not folder.name.isdecimal():
                continue
            try:
                fields = (folder/'stat').read_text().rsplit(')', 1)[1].split()
                # A tty shell creates foreground/background process groups in
                # the same session. Those groups remain ours after shell exit.
                if int(fields[3]) == self.pid and fields[0] not in ('Z', 'X'):
                    members.append(int(folder.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        return members

    def terminate_tree(self):
        for pid in self._members():
            try:
                # Recheck membership immediately before signalling a live PID.
                if os.getsid(pid) == self.pid:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def tree_active(self):
        return bool(self._members())

    def resize(self, columns, rows):
        import fcntl
        import struct
        import termios
        dimensions(columns, rows)
        with self._lock:
            if self.master is None:
                raise ValueError('terminal_pty_closed')
            fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack('HHHH', rows, columns, 0, 0))

    def close_terminal(self):
        with self._lock:
            if self.master is not None:
                os.close(self.master)
                self.master = None
        return True

    def close_process(self):
        pass  # Popen owns its process bookkeeping; wait has already reaped it.


class WindowsPtyProcess:
    def __init__(self, argv, *, cwd, env, columns, rows):
        import ctypes as c
        from ctypes import wintypes as w
        import msvcrt
        import shutil

        dimensions(columns, rows)
        self.c, self.w = c, w
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self._lock = threading.Lock()
        self._handle = self.hpc = None
        self.returncode, self.pid = None, None
        self.stdin = self.stdout = None
        self.stderr = io.BytesIO()

        class Coord(c.Structure):
            _fields_ = [('X', c.c_short), ('Y', c.c_short)]

        class Startup(c.Structure):
            _fields_ = [('cb', w.DWORD), ('reserved', w.LPWSTR), ('desktop', w.LPWSTR),
                ('title', w.LPWSTR), ('x', w.DWORD), ('y', w.DWORD), ('xsize', w.DWORD),
                ('ysize', w.DWORD), ('xchars', w.DWORD), ('ychars', w.DWORD),
                ('fill', w.DWORD), ('flags', w.DWORD), ('show', w.WORD), ('reserved2size', w.WORD),
                ('reserved2', c.c_void_p), ('stdin', w.HANDLE), ('stdout', w.HANDLE), ('stderr', w.HANDLE)]

        class StartupEx(c.Structure):
            _fields_ = [('startup', Startup), ('attributes', c.c_void_p)]

        class ProcessInfo(c.Structure):
            _fields_ = [('process', w.HANDLE), ('thread', w.HANDLE), ('pid', w.DWORD), ('tid', w.DWORD)]

        self.Coord = Coord
        signatures = {
            'CreatePipe': ([c.POINTER(w.HANDLE), c.POINTER(w.HANDLE), c.c_void_p, w.DWORD], w.BOOL),
            'CloseHandle': ([w.HANDLE], w.BOOL),
            'CreatePseudoConsole': ([Coord, w.HANDLE, w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)], c.c_long),
            'ResizePseudoConsole': ([w.HANDLE, Coord], c.c_long),
            'ClosePseudoConsole': ([w.HANDLE], None),
            'InitializeProcThreadAttributeList': ([c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t)], w.BOOL),
            'UpdateProcThreadAttribute': ([c.c_void_p, w.DWORD, c.c_size_t, c.c_void_p,
                                          c.c_size_t, c.c_void_p, c.c_void_p], w.BOOL),
            'DeleteProcThreadAttributeList': ([c.c_void_p], None),
            'CreateProcessW': ([w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p, w.BOOL,
                                w.DWORD, c.c_void_p, w.LPCWSTR, c.c_void_p, c.POINTER(ProcessInfo)], w.BOOL),
            'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
            'GetExitCodeProcess': ([w.HANDLE, c.POINTER(w.DWORD)], w.BOOL),
            'TerminateProcess': ([w.HANDLE, w.UINT], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, result

        handles, attributes, initialized = [], None, False
        info = ProcessInfo()
        try:
            for _ in range(2):
                read, write = w.HANDLE(), w.HANDLE()
                self._check(self.k.CreatePipe(c.byref(read), c.byref(write), None, 0))
                handles.extend([read.value, write.value])
            input_read, input_write, output_read, output_write = handles
            hpc = w.HANDLE()
            self._hr(self.k.CreatePseudoConsole(Coord(columns, rows), input_read, output_write, 0, c.byref(hpc)))
            self.hpc = hpc.value
            # Convert only our ends to Python-owned binary streams.
            fd = msvcrt.open_osfhandle(input_write, os.O_WRONLY | os.O_BINARY)
            handles.remove(input_write)
            self.stdin = os.fdopen(fd, 'wb')
            fd = msvcrt.open_osfhandle(output_read, os.O_RDONLY | os.O_BINARY)
            handles.remove(output_read)
            self.stdout = os.fdopen(fd, 'rb')
            size = c.c_size_t()
            self.k.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
            attributes = c.create_string_buffer(size.value)
            self._check(self.k.InitializeProcThreadAttributeList(attributes, 1, 0, c.byref(size)))
            initialized = True
            self._check(self.k.UpdateProcThreadAttribute(attributes, 0, 0x20016, self.hpc,
                                                        c.sizeof(w.HANDLE), None, None))
            startup = StartupEx()
            startup.startup.cb = c.sizeof(startup)
            # Prevent CRT startup from copying the redirected parent handles;
            # ConPTY supplies this child's console handles instead.
            startup.startup.flags = 0x100  # STARTF_USESTDHANDLES
            startup.startup.stdin = startup.startup.stdout = startup.startup.stderr = w.HANDLE(-1).value
            startup.attributes = c.cast(attributes, c.c_void_p)
            executable = argv[0]
            if not os.path.isabs(executable):
                executable = shutil.which(executable, path=env.get('PATH', os.defpath))
            if not executable or not os.path.isfile(executable):
                raise FileNotFoundError(argv[0])
            # A host started with CREATE_NEW_PROCESS_GROUP ignores Ctrl+C and
            # Windows passes that mark to every child. The trampoline clears it
            # in its own fresh process only, so ^C written to ConPTY reaches the
            # command while this host keeps its console-control state.
            argv = [sys.executable, '-I', str(Path(__file__).with_name('terminal_pty_win_exec.py')),
                    executable, *argv]
            executable = sys.executable
            command = c.create_unicode_buffer(subprocess.list2cmdline(argv))
            environment = c.create_unicode_buffer('\0'.join(k+'='+v for k, v in sorted(env.items(), key=lambda x: x[0].upper()))+'\0\0')
            # CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT
            self._check(self.k.CreateProcessW(executable, command, None, None, False,
                        0x4 | 0x400 | 0x80000, environment, cwd, c.byref(startup), c.byref(info)))
            self._handle, self.pid = info.process, info.pid
            self.k.CloseHandle(info.thread)
        except BaseException:
            # No user code has resumed yet. Drain output while closing ConPTY;
            # even startup failure can emit a final terminal frame.
            if self.stdout is not None:
                def discard():
                    try:
                        while self.stdout.read1(4096):
                            pass
                    except OSError:
                        pass
                    finally:
                        self.stdout.close()
                reader = threading.Thread(target=discard, daemon=True)
                reader.start()
            else:
                reader = None
            if self._handle is not None:
                self.kill()
                self.wait(timeout=5)
                self.close_process()
            # Release local pseudoconsole pipe references before waiting for EOF.
            for handle in handles:
                self.k.CloseHandle(handle)
            handles.clear()
            self.close_terminal()
            if self.stdin is not None:
                self.stdin.close()
            if reader:
                reader.join(timeout=5)
            raise
        finally:
            if initialized:
                self.k.DeleteProcThreadAttributeList(attributes)
            for handle in handles:
                self.k.CloseHandle(handle)

    def _check(self, result):
        if not result:
            raise self.c.WinError(self.c.get_last_error())

    @staticmethod
    def _hr(result):
        if result < 0:
            raise OSError('ConPTY HRESULT: '+hex(result & 0xffffffff))

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.k.WaitForSingleObject(self._handle, 0) == 0:
            value = self.w.DWORD()
            self._check(self.k.GetExitCodeProcess(self._handle, self.c.byref(value)))
            self.returncode = value.value
        return self.returncode

    def wait(self, timeout=None):
        if self.poll() is None:
            result = self.k.WaitForSingleObject(self._handle, 0xffffffff if timeout is None else max(0, int(timeout*1000)))
            if result == 258:
                raise subprocess.TimeoutExpired('ConPTY child', timeout)
            if result != 0:
                raise self.c.WinError(self.c.get_last_error())
        return self.poll()

    def kill(self):
        if self.poll() is None:
            if not self.k.TerminateProcess(self._handle, 1):
                error = self.c.get_last_error()
                # TerminateJobObject may already be finishing this same child.
                if error == 5 and self.k.WaitForSingleObject(self._handle, 1000) == 0:
                    self.poll()
                    return
                raise self.c.WinError(error)

    def resize(self, columns, rows):
        dimensions(columns, rows)
        with self._lock:
            if self.hpc is None:
                raise ValueError('terminal_pty_closed')
            self._hr(self.k.ResizePseudoConsole(self.hpc, self.Coord(columns, rows)))

    def close_terminal(self):
        # Never let an older Windows ConPTY close block the composition thread.
        def close():
            with self._lock:
                if self.hpc is not None:
                    handle, self.hpc = self.hpc, None
                    self.k.ClosePseudoConsole(handle)
        thread = threading.Thread(target=close, daemon=True, name='tare-conpty-close')
        thread.start()
        thread.join(timeout=5)
        return not thread.is_alive()

    def close_process(self):
        if self._handle is not None and self.poll() is not None:
            self.k.CloseHandle(self._handle)
            self._handle = None


def open_pty(argv, *, cwd, env, columns, rows):
    dimensions(columns, rows)
    if os.name != 'nt' and not sys.platform.startswith('linux'):
        raise ValueError('terminal_pty_platform_not_supported')
    adapter = WindowsPtyProcess if os.name == 'nt' else PosixPtyProcess
    return adapter(argv, cwd=cwd, env=env, columns=columns, rows=rows)
