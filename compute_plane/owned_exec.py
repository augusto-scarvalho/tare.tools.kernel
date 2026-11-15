"""Linux exec shim: the command dies with the tare thread that started it (PR_SET_PDEATHSIG).

A fresh interpreter sets it, so the threaded host needs no preexec_fn (as terminal_pty_exec.py).
"""
import ctypes
import os
from pathlib import Path
import signal
import sys


if __name__ == '__main__':
    report, parent, argv = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != parent:
        os._exit(1)  # The parent died before prctl, which then never fires.
    # Undo what this interpreter's start did to what the command inherits: SIGPIPE and SIGXFSZ
    # ignored and, in a C locale, LC_CTYPE coerced. /proc keeps the environment the parent passed.
    for name in ('SIGPIPE', 'SIGXFSZ'):
        signal.signal(getattr(signal, name), signal.SIG_DFL)
    env = dict(item.split(b'=', 1) for item in Path('/proc/self/environ').read_bytes().split(b'\0') if b'=' in item)
    # The group tare kills when the command ends. A sweep after tare died ends only stamped processes still in
    # it: a daemon the command detached on purpose (a build daemon other programs share) left it and is spared.
    env[b'TARE_GROUP'] = str(os.getpgrp()).encode()
    os.set_inheritable(report, False)  # A successful exec closes it: the parent reads end of file.
    try:
        os.execvpe(argv[0], argv, env)
    except OSError as error:
        os.write(report, str(error.errno).encode())
        os._exit(127)
