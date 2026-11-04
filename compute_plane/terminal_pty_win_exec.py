"""Windows ConPTY trampoline: the command receives Ctrl+C from its terminal.

CREATE_NEW_PROCESS_GROUP makes a process ignore Ctrl+C, and every descendant
created later inherits that mark. A host launched that way would pass it on to
terminal commands, which then ignore the ^C written to the pseudoconsole. The
host keeps its own console-control state; only this fresh process clears the
mark before creating the command, then waits and forwards its exit code.
"""
import ctypes
from ctypes import wintypes
import subprocess
import sys


def main():
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    kernel32.ExitProcess.argtypes = [wintypes.UINT]
    # A null handler with FALSE clears the inherited ignore mark; the command
    # created below inherits the cleared state.
    if not kernel32.SetConsoleCtrlHandler(None, False):
        raise ctypes.WinError(ctypes.get_last_error())
    # The trampoline must outlive the command to report its result. Handlers
    # are not inherited, so consuming Ctrl+C/Ctrl+Break here leaves the
    # command's own reaction untouched.
    consume = handler_type(lambda event: event in (0, 1))
    if not kernel32.SetConsoleCtrlHandler(consume, True):
        raise ctypes.WinError(ctypes.get_last_error())
    code = subprocess.call(sys.argv[2:], executable=sys.argv[1])
    kernel32.ExitProcess(code & 0xffffffff)


if __name__ == '__main__':
    main()
