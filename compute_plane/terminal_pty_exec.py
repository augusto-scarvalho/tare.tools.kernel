"""POSIX exec trampoline: acquire the controlling tty outside the threaded host."""
import os
import sys


if __name__ == '__main__':
    import fcntl
    import termios
    # Popen already made this process a session leader. No preexec_fn runs in
    # the multithreaded parent; only this fresh interpreter touches tty state.
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
