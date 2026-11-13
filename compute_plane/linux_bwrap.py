"""Immutable path policy and a bubblewrap process factory for Linux, the counterpart of windows_mxc.

Inside, only the policy's paths exist, each at its own path, read-only or writable, plus a fresh /proc, a minimal
/dev and an empty /tmp; every denied path that exists at launch is masked (mode 000, read-only). The sandbox has its
own user, PID, IPC and network namespaces (a private loopback, so host listeners are unreachable), no capabilities,
no nested user namespaces, at most 1024 tasks (threads included) and no core dumps. A seccomp filter refuses
sockets other than AF_UNIX, AF_INET, AF_INET6 and AF_NETLINK (an AF_VSOCK socket reaches the host, and in WSL2
starts Windows programs, network namespace or not), io_uring and syscalls of another architecture.

Requires bubblewrap 0.8 or newer and unprivileged user namespaces. Anything less is no sandbox, never a weaker one.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import posixpath
import re
import signal
import struct
import subprocess
import sys

# The host puts these in the policy itself, so the digest names everything bound. Absent ones are skipped.
SYSTEM_ROOTS = ('/usr', '/bin', '/sbin', '/lib', '/lib32', '/lib64', '/libx32', '/etc')

# bwrap's words when the kernel or a security module (AppArmor on stock Ubuntu 24.04) refuses its namespaces.
_USERNS_REFUSED = ('setting up uid map', 'No permissions to create', 'Creating new namespace failed',
                   'loopback: Failed RTM_NEW')
_USERNS_SETTINGS = ('kernel/apparmor_restrict_unprivileged_userns', 'kernel/unprivileged_userns_clone',
                    'user/max_user_namespaces')

# Classic BPF over struct seccomp_data {int nr; u32 arch; u64 ip; u64 args[6]}: nr at 0, arch at 4, the low word
# of args[0] at 16 (both machines are little-endian).
_LD, _JEQ, _JGE, _RET = 0x20, 0x15, 0x35, 0x06  # BPF_LD|W|ABS, BPF_JMP|JEQ|K, BPF_JMP|JGE|K, BPF_RET|K
_ALLOW, _EPERM = 0x7FFF0000, 0x00050001         # SECCOMP_RET_ALLOW, SECCOMP_RET_ERRNO | EPERM
# uname machine: AUDIT_ARCH, socket, socketpair, and whether x32 syscalls (nr | 0x40000000) share that arch.
_MACHINES = {'x86_64': (0xC000003E, 41, 53, True), 'aarch64': (0xC00000B7, 198, 199, False)}
_IO_URING = (425, 426, 427)  # io_uring_setup, io_uring_enter, io_uring_register, on both machines
_FAMILIES = (1, 2, 10, 16)   # AF_UNIX, AF_INET, AF_INET6, AF_NETLINK


def _paths(values, group):
    if isinstance(values, (str, bytes)):
        raise TypeError(f'{group} must be a sequence of paths')
    result = []
    for path in values:
        if not isinstance(path, str):
            raise TypeError(f'{group} paths must be strings')
        # Validate the original spelling, before normpath can erase traversal.
        if not path.startswith('/') or '\0' in path or '..' in path.split('/'):
            raise ValueError(f'{group} requires absolute POSIX paths without NUL or traversal')
        canonical = '/' + posixpath.normpath(path).lstrip('/')
        if canonical == '/':
            raise ValueError(f'{group} cannot include the root')
        if canonical in result:
            raise ValueError(f'{group} contains a duplicate path')
        result.append(canonical)
    return tuple(result)


@dataclass(frozen=True)
class BwrapPolicy:
    read_only: tuple[str, ...]
    read_write: tuple[str, ...]
    denied: tuple[str, ...] = ()

    def __post_init__(self):
        for group in ('read_only', 'read_write', 'denied'):
            object.__setattr__(self, group, _paths(getattr(self, group), group))
        if not (self.read_only or self.read_write):
            raise ValueError('read_only or read_write scope must be nonempty')
        if set(self.read_only) & set(self.read_write):
            raise ValueError('a path cannot be both read_only and read_write')


def policy_sha256(policy):
    if not isinstance(policy, BwrapPolicy):
        raise TypeError('policy must be a BwrapPolicy')
    canonical = json.dumps(['linux-bwrap/1', policy.read_only, policy.read_write, policy.denied],
                           separators=(',', ':'))
    return hashlib.sha256(canonical.encode('ascii')).hexdigest()


def seccomp_program(machine):
    """The raw sock_filter array bwrap loads with `--seccomp FD`. EPERM for another architecture (an i386
    socketcall cannot be filtered by family) and x32, for io_uring (it opens sockets without socket()), and for
    socket or socketpair outside _FAMILIES. Everything else is allowed."""
    arch, socket, socketpair, x32 = _MACHINES[machine]
    code = [(_LD, 4, None, None), (_JEQ, arch, None, 'deny'), (_LD, 0, None, None)]
    if x32:
        code.append((_JGE, 0x40000000, 'deny', None))
    code += [(_JEQ, socket, 'family', None), (_JEQ, socketpair, 'family', None)]
    code += [(_JEQ, number, 'deny', None) for number in _IO_URING]
    code += [(_RET, _ALLOW, None, None), 'family', (_LD, 16, None, None)]
    code += [(_JEQ, family, 'allow', None) for family in _FAMILIES]
    code += ['deny', (_RET, _EPERM, None, None), 'allow', (_RET, _ALLOW, None, None)]
    labels, program = {}, []
    for item in code:
        if isinstance(item, str):
            labels[item] = len(program)
        else:
            program.append(item)
    jump = lambda at, label: 0 if label is None else labels[label] - at - 1
    return b''.join(struct.pack('<HBBI', op, jump(at, true), jump(at, false), k)
                    for at, (op, k, true, false) in enumerate(program))


def _masks(secrets, scopes):
    """Where each existing secret shows inside: at every bound scope whose real path holds it, below that scope's
    own path. `secrets` maps real paths to whether they are folders, `scopes` bound paths to their real paths.
    Shallowest first; one inside a masked folder is dropped (hidden already, and bwrap could not mount there)."""
    found = sorted({(path + real[len(root):], folder) for real, folder in secrets.items()
                    for path, root in scopes.items() if real == root or real.startswith(root + '/')},
                   key=lambda mask: (mask[0].count('/'), mask[0]))
    masks = []
    for path, folder in found:
        if not any(hidden and path.startswith(above + '/') for above, hidden in masks):
            masks.append((path, folder))
    return masks


def bwrap_arguments(binds, masks, *, cwd, env, seccomp_fd, empty_fd, prlimit, argv, task_limit):
    """bwrap's arguments. `binds` are (path, source, writable), `masks` (path, folder) from _masks, and
    `empty_fd()` returns a new descriptor that reads as empty (bwrap closes each one it reads)."""
    args = ['--unshare-user', '--disable-userns', '--unshare-pid', '--unshare-net', '--unshare-ipc',
            '--die-with-parent', '--cap-drop', 'ALL', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    # bwrap itself starts with no environment, so no loader variable can make it run a check's library.
    # ponytail: these values show in bwrap's world-readable cmdline; pass them with --args FD if one is ever secret.
    args.append('--clearenv')
    for name, value in sorted(env.items()):
        args += ['--setenv', name, value]
    # Shallowest first, so a nested grant (a read-only .git inside a writable tree) lands on top.
    for path, source, writable in sorted(binds, key=lambda bind: (bind[0].count('/'), bind[0])):
        args += ['--bind' if writable else '--ro-bind', source, path]
    # Masks last, so they cover every bind. Mode 000 reads as EACCES, never as an empty file.
    for path, folder in masks:
        if folder:
            args += ['--perms', '000', '--tmpfs', path, '--remount-ro', path]
        else:
            args += ['--perms', '000', '--ro-bind-data', str(empty_fd()), path]
    return [*args, '--remount-ro', '/', '--chdir', cwd, '--seccomp', str(seccomp_fd), '--',
            prlimit, f'--nproc={task_limit}', '--core=0', '--', *argv]


def _host_tools():
    from .managed_process import host_executable
    if sys.platform != 'linux':
        raise OSError('bwrap_linux_only')
    if os.uname().machine not in _MACHINES:
        raise OSError('bwrap_architecture_unsupported')
    bwrap, prlimit = host_executable('bwrap'), host_executable('prlimit')
    if not bwrap:
        raise OSError('bwrap_missing')
    if not prlimit:
        raise OSError('bwrap_prlimit_missing')
    return bwrap, prlimit


def require_support():
    """Return only if one real launch of the full profile (system roots, a masked folder and a masked file, the
    filter, the task limit) runs `true` to exit 0 within 5 s. Otherwise OSError, its message starting with a
    code: bwrap_linux_only, bwrap_architecture_unsupported, bwrap_missing, bwrap_prlimit_missing, bwrap_too_old
    (older than 0.8), bwrap_userns_refused (the kernel or a security module such as AppArmor refuses unprivileged
    user namespaces; the message gives the userns sysctls this host has) or bwrap_launch_failed."""
    import tempfile
    from .managed_process import host_executable
    bwrap, _ = _host_tools()
    version = subprocess.run([bwrap, '--version'], capture_output=True, text=True, timeout=5).stdout.strip()
    found = re.search(r'(\d+)\.(\d+)', version)
    if not found or tuple(map(int, found.groups())) < (0, 8):
        raise OSError(f'bwrap_too_old: {version or "no version"}; 0.8 or newer is required')
    with tempfile.TemporaryDirectory(prefix='tare-bwrap-') as folder, \
            open(os.devnull, 'rb') as stdin, open(os.devnull, 'wb') as stdout, tempfile.TemporaryFile() as stderr:
        os.mkdir(os.path.join(folder, 'secret'))
        open(os.path.join(folder, 'secret.txt'), 'wb').close()
        policy = BwrapPolicy((*SYSTEM_ROOTS, folder), (),
                             (os.path.join(folder, 'secret'), os.path.join(folder, 'secret.txt')))
        try:
            profile = LinuxBwrapProfile(policy, cwd=folder, environment={})
            process = profile([host_executable('true', required=True)], cwd=folder, env={},
                              stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=True)
        except ValueError as error:  # e.g. prlimit or true outside the system roots
            raise OSError(f'bwrap_launch_failed: {error}') from error
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 'timeout'
        if code != 0:
            stderr.seek(0)
            detail = stderr.read(2000).decode('utf-8', 'replace').strip()
            if any(words in detail for words in _USERNS_REFUSED):
                settings = []
                for name in _USERNS_SETTINGS:
                    try:
                        with open('/proc/sys/' + name) as setting:
                            settings.append(name.replace('/', '.') + '=' + setting.read().strip())
                    except OSError:
                        pass
                raise OSError(f'bwrap_userns_refused: {"; ".join(settings) or "no userns sysctl"}; {detail}')
            raise OSError(f'bwrap_launch_failed: exit {code}: {detail}')


class LinuxBwrapProfile:
    """Explicit bubblewrap process factory for managed_process.run_process (its POSIX process-group supervisor)."""
    # Linux counts threads: Node and cargo need hundreds. Set inside the sandbox, where it counts only its tasks.
    task_limit = 1024

    def __init__(self, policy, *, cwd, environment):
        from .mediated_tools import validate_command_confinement
        self._bwrap, self._prlimit = _host_tools()
        if not isinstance(policy, BwrapPolicy):
            raise TypeError('bwrap_policy_required')
        self._policy = policy
        self._policy_sha256 = policy_sha256(policy)
        self._environment, _ = validate_command_confinement(self, environment, 1)
        self._program = seccomp_program(os.uname().machine)
        self._scopes = self._resolve()
        self._secrets = self._identify()
        if not self._admitted(self._prlimit):
            raise ValueError('bwrap_prlimit_outside_policy')
        if (not isinstance(cwd, str) or not cwd.startswith('/') or cwd != posixpath.normpath(cwd)
                or not os.path.isdir(cwd) or not self._admitted(cwd)):
            raise ValueError('bwrap_cwd_outside_policy')
        self._cwd = cwd

    def _admitted(self, path):
        return any(path == scope or path.startswith(scope + '/') for scope in self._scopes)

    def _resolve(self):
        # A bind follows links in its source. A scope whose real path moves (a check that replaced a missing
        # write scope by a link to the home folder) would bind elsewhere next launch, so launch compares these.
        scopes = {path: os.path.realpath(path) for path in (*self._policy.read_only, *self._policy.read_write)}
        if '/' in scopes.values():
            raise ValueError('bwrap_scope_is_root')
        return scopes

    def _identify(self):
        # A check can carry a secret out of its mask by renaming a folder above it; the next launch would then
        # leave it readable at its new path. Launch compares these identities and refuses any change.
        secrets = {}
        for path in self._policy.denied:
            real = os.path.realpath(path)
            try:
                info = os.stat(real)
            except OSError:  # absent, or not reachable by this user, and so not by the sandbox either
                secrets[path] = None
            else:
                secrets[path] = (real, info.st_dev, info.st_ino)
        return secrets

    def __call__(self, argv, *, cwd, env, stdin, stdout, stderr, start_new_session):
        if start_new_session is not True or cwd != self._cwd or env != self._environment:
            raise ValueError('bwrap_launch_binding_changed')
        if policy_sha256(self._policy) != self._policy_sha256:
            raise ValueError('bwrap_policy_changed')
        if (not isinstance(argv, (list, tuple)) or not argv or
                any(not isinstance(a, str) or '\0' in a for a in argv)):
            raise ValueError('bwrap_argv_invalid')
        if (not argv[0].startswith('/') or argv[0] != posixpath.normpath(argv[0])
                or not self._admitted(argv[0])):
            raise ValueError('bwrap_executable_outside_policy')
        # No --new-session: with no terminal there is none to protect, and the sandbox stays in bwrap's process
        # group, which the supervisor kills and checks.
        if any(not hasattr(s, 'fileno') or os.isatty(s.fileno()) for s in (stdin, stdout, stderr)):
            raise ValueError('bwrap_terminal_stdio_refused')
        if self._resolve() != self._scopes:
            raise ValueError('bwrap_scope_changed')
        if self._identify() != self._secrets:
            raise ValueError('bwrap_denied_path_changed')
        binds = [(path, real, path in self._policy.read_write)
                 for path, real in self._scopes.items() if os.path.exists(real)]
        masks = _masks({s[0]: os.path.isdir(s[0]) for s in self._secrets.values() if s},
                       {path: real for path, real, _ in binds})
        fds = []
        try:
            fds.append(os.memfd_create('tare-seccomp'))
            os.write(fds[0], self._program)
            os.lseek(fds[0], 0, os.SEEK_SET)

            def empty_fd():
                fds.append(os.open(os.devnull, os.O_RDONLY))
                return fds[-1]

            args = bwrap_arguments(binds, masks, cwd=self._cwd, env=env, seccomp_fd=fds[0], empty_fd=empty_fd,
                                   prlimit=self._prlimit, argv=argv, task_limit=self.task_limit)
            process = subprocess.Popen([self._bwrap, *args], cwd=self._cwd, env={}, stdin=stdin,
                                       stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=fds)
        finally:
            for fd in fds:
                os.close(fd)
        # What this launch enforced, and what it did not: no memory cap exists without privilege.
        process.confinement_receipt = {
            'backend': 'linux-bwrap', 'policy_sha256': self._policy_sha256, 'pid': process.pid,
            'bwrap': self._bwrap, 'task_limit': self.task_limit, 'memory_limit': None,
            'network': 'private-loopback', 'seccomp_sha256': hashlib.sha256(self._program).hexdigest(),
            'masked': [path for path, _ in masks]}
        return process
