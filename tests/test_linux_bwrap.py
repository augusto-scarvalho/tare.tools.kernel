"""Linux bubblewrap sandbox: pure checks run anywhere; real-sandbox checks need Linux with bubblewrap.

Real-sandbox checks skip where require_support() fails, unless TARE_REQUIRE_SANDBOX=1 (the CI sandbox job), which
turns that skip into a failure.
"""

from dataclasses import FrozenInstanceError
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

from compute_plane import linux_bwrap
from compute_plane.linux_bwrap import (SYSTEM_ROOTS, BwrapPolicy, LinuxBwrapProfile, _masks, bwrap_arguments,
                                       policy_sha256, require_support, seccomp_program)

X86_64, AARCH64, I386, ARM = 0xC000003E, 0xC00000B7, 0x40000003, 0x40000028
ALLOW, EPERM = 0x7FFF0000, 0x00050001


# Policy

def test_paths_are_normalized_copied_and_frozen():
    read_only, read_write, denied = ['/usr/./lib/', '/etc'], ['/home/u//p'], ['/home/u/p/.env']
    policy = BwrapPolicy(read_only, read_write, denied)
    read_only.append('/opt')
    denied.clear()
    assert policy == BwrapPolicy(('/usr/lib', '/etc'), ('/home/u/p',), ('/home/u/p/.env',))
    with pytest.raises(FrozenInstanceError):
        policy.denied = ()


@pytest.mark.parametrize('path', ['', 'relative', './x', '/', '//', '/.', '/a/../b', '/a/..', '/a\x00b', 'C:/x'])
def test_invalid_paths_rejected_in_every_group(path):
    for group in ('read_only', 'read_write', 'denied'):
        groups = {'read_only': ['/usr'], 'read_write': []}
        groups[group] = [path]
        with pytest.raises(ValueError):
            BwrapPolicy(**groups)


def test_duplicates_overlap_types_and_nonempty_scope():
    with pytest.raises(ValueError, match='duplicate'):
        BwrapPolicy(['/usr', '/usr/'], [])
    with pytest.raises(ValueError, match='both'):
        BwrapPolicy(['/p'], ['/p/.'])
    with pytest.raises(ValueError, match='nonempty'):
        BwrapPolicy([], [], ['/p/.env'])
    with pytest.raises(TypeError):
        BwrapPolicy('/usr', [])
    with pytest.raises(TypeError, match='strings'):
        BwrapPolicy([b'/usr'], [])


def test_digest_is_the_documented_canonical_json():
    policy = BwrapPolicy(['/usr', '/p'], ['/p/out'], ['/p/.env'])
    text = '["linux-bwrap/1",["/usr","/p"],["/p/out"],["/p/.env"]]'
    assert policy_sha256(policy) == hashlib.sha256(text.encode()).hexdigest()
    assert policy_sha256(policy) != policy_sha256(BwrapPolicy(['/usr', '/p', '/p/out'], [], ['/p/.env']))
    for value in (None, {}, 'policy'):
        with pytest.raises(TypeError, match='BwrapPolicy'):
            policy_sha256(value)


# Seccomp: an independent classic-BPF interpreter runs the exact bytes.

def _verdict(program, arch, nr, family=0):
    data = struct.pack('<IIQ6Q', nr & 0xFFFFFFFF, arch, 0, family, 0, 0, 0, 0, 0)
    code = [struct.unpack_from('<HBBI', program, at) for at in range(0, len(program), 8)]
    at = accumulator = 0
    while True:
        op, true, false, k = code[at]
        if op == 0x20:
            accumulator, at = struct.unpack_from('<I', data, k)[0], at + 1
        elif op == 0x15:
            at += 1 + (true if accumulator == k else false)
        elif op == 0x35:
            at += 1 + (true if accumulator >= k else false)
        elif op == 0x06:
            return k
        else:
            raise AssertionError(f'unexpected opcode {op:#x}')


@pytest.mark.parametrize('machine,arch,socket_nr,pair_nr,foreign', [
    ('x86_64', X86_64, 41, 53, I386), ('aarch64', AARCH64, 198, 199, ARM)])
def test_seccomp_program_refuses_exactly_what_it_should(machine, arch, socket_nr, pair_nr, foreign):
    program = seccomp_program(machine)
    assert len(program) % 8 == 0 and len(program) // 8 <= 20
    for family in (1, 2, 10, 16, 0x1_0000_0001):  # AF_UNIX, AF_INET, AF_INET6, AF_NETLINK; the kernel reads an int
        assert _verdict(program, arch, socket_nr, family) == ALLOW
        assert _verdict(program, arch, pair_nr, family) == ALLOW
    for family in (0, 3, 9, 17, 31, 40, 44, 45):  # AF_PACKET 17, AF_BLUETOOTH 31, AF_VSOCK 40, AF_XDP 44, ...
        assert _verdict(program, arch, socket_nr, family) == EPERM
        assert _verdict(program, arch, pair_nr, family) == EPERM
    for nr in (425, 426, 427):
        assert _verdict(program, arch, nr) == EPERM
    for nr in (0, 1, 57, 59, 63, 220, 221, 424, 428, 435):
        assert _verdict(program, arch, nr, family=40) == ALLOW
    assert _verdict(program, foreign, 0) == EPERM
    assert _verdict(program, foreign, socket_nr, 1) == EPERM
    if machine == 'x86_64':
        assert _verdict(program, arch, 0x40000000 | socket_nr, 1) == EPERM  # x32
        assert _verdict(program, arch, 0x40000000) == EPERM


def test_seccomp_program_refuses_an_unknown_machine():
    with pytest.raises(KeyError):
        seccomp_program('riscv64')


# Arguments and masks

def test_arguments_for_one_policy():
    binds = [('/p/.git', '/p/.git', False), ('/usr', '/usr', False), ('/p', '/p', True), ('/lib', '/usr/lib', False)]
    masks = [('/p/.ssh', True), ('/p/.env', False)]
    descriptors = iter([7])
    assert bwrap_arguments(binds, masks, cwd='/p', env={'PATH': '/usr/bin', 'HOME': '/p/out'}, seccomp_fd=3,
                           empty_fd=lambda: next(descriptors), prlimit='/usr/bin/prlimit',
                           argv=['/usr/bin/python3', '-c', 'pass'], task_limit=1024) == [
        '--unshare-user', '--disable-userns', '--unshare-pid', '--unshare-net', '--unshare-ipc',
        '--die-with-parent', '--cap-drop', 'ALL', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
        '--clearenv', '--setenv', 'HOME', '/p/out', '--setenv', 'PATH', '/usr/bin',
        '--ro-bind', '/usr/lib', '/lib', '--bind', '/p', '/p', '--ro-bind', '/usr', '/usr',
        '--ro-bind', '/p/.git', '/p/.git',
        '--perms', '000', '--tmpfs', '/p/.ssh', '--remount-ro', '/p/.ssh',
        '--perms', '000', '--ro-bind-data', '7', '/p/.env',
        '--remount-ro', '/', '--chdir', '/p', '--seccomp', '3', '--',
        '/usr/bin/prlimit', '--nproc=1024', '--core=0', '--', '/usr/bin/python3', '-c', 'pass']


def test_masks_follow_every_alias_and_skip_what_is_hidden_or_unbound():
    scopes = {'/p': '/p', '/lib': '/usr/lib', '/alias': '/p/sub', '/pp': '/pp'}
    secrets = {'/p/.env': False, '/p/sub/.ssh': True, '/p/sub/.ssh/id': False, '/elsewhere/.env': False,
               '/usr/lib/x/.env': False, '/pq/.env': False}
    assert _masks(secrets, scopes) == [('/alias/.ssh', True), ('/p/.env', False), ('/lib/x/.env', False),
                                       ('/p/sub/.ssh', True)]
    assert _masks({'/p': True}, {'/p': '/p', '/p/in': '/p/in'}) == [('/p', True)]


def test_a_non_linux_host_has_no_sandbox(monkeypatch):
    monkeypatch.setattr(linux_bwrap, 'sys', SimpleNamespace(platform='darwin'))
    with pytest.raises(OSError, match='^bwrap_linux_only$'):
        require_support()
    with pytest.raises(OSError, match='^bwrap_linux_only$'):
        LinuxBwrapProfile(BwrapPolicy(['/usr'], []), cwd='/usr', environment={})


# Real sandbox

@cache
def _support_error():
    try:
        require_support()
    except OSError as error:
        return str(error)
    return None


def _runtime_roots():  # as spelled, so sys.executable is inside them
    return [p for p in dict.fromkeys(os.path.normpath(p) for p in (sys.base_prefix, sys.prefix))
            if not any(p == root or p.startswith(root + '/') for root in SYSTEM_ROOTS)]


@pytest.fixture
def box(tmp_path):
    """A project with a read-only and a writable folder, fake secrets in both, and a folder outside the grants."""
    error = _support_error()
    if error:
        if os.environ.get('TARE_REQUIRE_SANDBOX') == '1':
            pytest.fail(f'this run requires the sandbox: {error}')
        pytest.skip(error)
    ro, rw, outside = tmp_path / 'ro', tmp_path / 'rw', tmp_path / 'outside'
    for folder in (ro / '.ssh', rw / 'nested', outside):
        folder.mkdir(parents=True)
    (ro / 'readme.txt').write_text('granted')
    (ro / '.env').write_text('fake-secret-file')
    (ro / '.ssh' / 'id_test').write_text('fake-secret-key')
    (ro / 'npmrc-target').write_text('fake-secret-behind-link')
    (ro / '.npmrc').symlink_to('npmrc-target')
    (rw / 'nested' / '.env').write_text('fake-secret-in-writable')
    (outside / 'other.txt').write_text('outside')
    policy = BwrapPolicy((*SYSTEM_ROOTS, *_runtime_roots(), str(ro)), (str(rw), str(rw / 'later')),
                         (str(ro / '.env'), str(ro / '.ssh'), str(ro / '.npmrc'), str(rw / 'nested' / '.env')))
    profile = LinuxBwrapProfile(policy, cwd=str(rw), environment={})
    return profile, policy, tmp_path


def _run(profile, tmp_path, code, *args):
    from compute_plane.managed_process import run_process
    dest = tmp_path / 'runs' / uuid.uuid4().hex
    result = run_process([sys.executable, '-I', '-c', code, *map(str, args)], cwd=profile._cwd, dest=dest,
                         timeout=60, env={}, process_factory=profile)
    return result, (dest / 'stdout.txt').read_text(), (dest / 'stderr.txt').read_text()


def _json(run):
    result, out, err = run
    assert result['exit_code'] == 0, err
    return json.loads(out)


ATTEMPT = '''
import json, os, sys
def attempt(action):
    try:
        return action()
    except OSError as error:
        return error.errno
'''


def test_filesystem_grants_absences_and_masks(box):
    profile, policy, tmp_path = box
    found = _json(_run(profile, tmp_path, ATTEMPT + '''
ro, rw, outside = sys.argv[1:4]
print(json.dumps({
    'read': attempt(lambda: open(ro + '/readme.txt').read()),
    'write_read_only': attempt(lambda: open(ro + '/new.txt', 'w')),
    'write_writable': attempt(lambda: open(rw + '/new.txt', 'w').write('x')),
    'outside': os.path.exists(outside), 'listed_parent': sorted(os.listdir(os.path.dirname(ro))),
    'write_root': attempt(lambda: open('/new.txt', 'w')),
    'secret_file': attempt(lambda: open(ro + '/.env').read()),
    'secret_folder': attempt(lambda: os.listdir(ro + '/.ssh')),
    'secret_in_folder': attempt(lambda: open(ro + '/.ssh/id_test').read()),
    'secret_behind_link': attempt(lambda: open(ro + '/.npmrc').read()),
    'secret_link_target': attempt(lambda: open(ro + '/npmrc-target').read()),
    'secret_in_writable': attempt(lambda: open(rw + '/nested/.env').read()),
    'secret_overwrite': attempt(lambda: open(rw + '/nested/.env', 'w')),
}))''', tmp_path / 'ro', tmp_path / 'rw', tmp_path / 'outside'))
    assert found == {'read': 'granted', 'write_read_only': 30, 'write_writable': 1, 'outside': False,
                     'listed_parent': ['ro', 'rw'], 'write_root': 30, 'secret_file': 13, 'secret_folder': 13,
                     'secret_in_folder': 13, 'secret_behind_link': 13, 'secret_link_target': 13,
                     'secret_in_writable': 13, 'secret_overwrite': 30}  # O_TRUNC meets the read-only mask first
    assert (tmp_path / 'rw' / 'new.txt').read_text() == 'x'
    assert not (tmp_path / 'ro' / 'new.txt').exists()
    assert (tmp_path / 'rw' / 'nested' / '.env').read_text() == 'fake-secret-in-writable'


def test_a_secret_moved_by_a_check_refuses_the_next_launch(box):
    profile, policy, tmp_path = box
    found = _json(_run(profile, tmp_path, ATTEMPT + '''
rw = sys.argv[1]
moved = attempt(lambda: os.rename(rw + '/nested', rw + '/moved'))
print(json.dumps({'moved': moved, 'read': attempt(lambda: open(rw + '/moved/.env').read()),
                  'unlink': attempt(lambda: os.unlink(rw + '/moved/.env'))}))''', tmp_path / 'rw'))
    # The mask travels with its folder, and the masked entry itself cannot be removed.
    assert found == {'moved': None, 'read': 13, 'unlink': 16}
    with pytest.raises(ValueError, match='bwrap_denied_path_changed'):
        _run(profile, tmp_path, 'pass')


def test_a_scope_replaced_by_a_link_refuses_the_next_launch(box):
    profile, policy, tmp_path = box
    _json(_run(profile, tmp_path, 'import os, sys; os.symlink(sys.argv[1], sys.argv[2]); print(1)',
               tmp_path / 'outside', tmp_path / 'rw' / 'later'))
    with pytest.raises(ValueError, match='bwrap_scope_changed'):
        _run(profile, tmp_path, 'pass')


def test_network_is_a_private_loopback_and_vm_sockets_are_refused(box):
    profile, policy, tmp_path = box
    with socket.create_server(('127.0.0.1', 0)) as listener:
        found = _json(_run(profile, tmp_path, ATTEMPT + '''
import ctypes, socket
def own_loopback():
    with socket.create_server(('127.0.0.1', 0)) as server, socket.create_connection(server.getsockname(), 5) as client:
        connection, _ = server.accept()
        client.sendall(b'ping')
        return connection.recv(4).decode()
libc = ctypes.CDLL(None, use_errno=True)
def call(function, *args):
    return 0 if function(*args) != -1 else ctypes.get_errno()
print(json.dumps({
    'host_listener': attempt(lambda: socket.create_connection(('127.0.0.1', int(sys.argv[1])), 5) and 0),
    'own_loopback': attempt(own_loopback),
    'off_machine': attempt(lambda: socket.create_connection(('192.0.2.1', 80), 5) and 0),
    'unix_pair': attempt(lambda: len(socket.socketpair())),
    'vsock': attempt(lambda: socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM).close()),
    'io_uring_setup': call(libc.syscall, 425, 1, ctypes.create_string_buffer(120)),
    'nested_user_namespace': call(libc.unshare, 0x10000000),  # CLONE_NEWUSER
}))''', listener.getsockname()[1]))
        listener.setblocking(False)
        with pytest.raises(BlockingIOError):
            listener.accept()  # nothing from the sandbox reached the host's listener
    assert found['host_listener'] == 111  # ECONNREFUSED: the sandbox's own loopback has no such listener
    assert found['own_loopback'] == 'ping'
    assert found['off_machine'] == 101  # ENETUNREACH
    assert found['unix_pair'] == 2
    assert found['vsock'] == 1  # EPERM from the filter, not EAFNOSUPPORT (97) from a kernel without vsock
    assert found['io_uring_setup'] == 1
    assert found['nested_user_namespace'] == 28  # ENOSPC: --disable-userns (bwrap without it allows one)


def test_task_limit_holds_and_core_dumps_are_off(box):
    profile, policy, tmp_path = box
    found = _json(_run(profile, tmp_path, '''
import json, resource, threading
limits = [resource.getrlimit(resource.RLIMIT_NPROC), resource.getrlimit(resource.RLIMIT_CORE)]
threading.stack_size(65536)
stop, started = threading.Event(), 0
try:
    for _ in range(1100):
        threading.Thread(target=stop.wait, daemon=True).start()
        started += 1
except RuntimeError:
    pass
stop.set()
print(json.dumps({'limits': limits, 'started': started}))'''))
    assert found['limits'] == [[1024, 1024], [0, 0]]
    assert 900 < found['started'] < 1024  # stopped by the limit, which counts threads, not by anything else


def test_killing_the_parent_leaves_nothing_behind(box, tmp_path):
    profile, policy, _ = box
    marker = 'tare-bwrap-' + uuid.uuid4().hex
    helper = subprocess.Popen([sys.executable, '-c', '''
import json, os, sys, time
from compute_plane.linux_bwrap import BwrapPolicy, LinuxBwrapProfile
policy, cwd, argv = json.loads(sys.argv[1])
profile = LinuxBwrapProfile(BwrapPolicy(*policy), cwd=cwd, environment={})
with open(os.devnull, 'rb') as stdin, open(os.devnull, 'wb') as stdout:
    profile(argv, cwd=cwd, env={}, stdin=stdin, stdout=stdout, stderr=stdout, start_new_session=True)
    time.sleep(60)''', json.dumps([[policy.read_only, policy.read_write, policy.denied], profile._cwd, [
        sys.executable, '-I', '-c', 'import subprocess, sys, time; subprocess.Popen([sys.executable, "-I", "-c", '
        '"import time; time.sleep(60)", sys.argv[1] + "-child"]); time.sleep(60)', marker]])],
        cwd=Path(linux_bwrap.__file__).parents[1])

    def running(suffix=b''):  # zombies have an empty cmdline
        found = []
        for folder in Path('/proc').iterdir():
            try:
                if folder.name.isdecimal() and marker.encode() + suffix in (folder / 'cmdline').read_bytes():
                    found.append(folder.name)
            except OSError:
                pass
        return found

    try:
        until = time.monotonic() + 20
        while not running(b'-child'):
            assert time.monotonic() < until and helper.poll() is None, 'the sandboxed grandchild never started'
            time.sleep(.05)
    finally:
        helper.kill()  # SIGKILL: bwrap's parent dies without any cleanup of its own
        helper.wait()
    until = time.monotonic() + 10
    while running():
        assert time.monotonic() < until, f'left behind: {running()}'
        time.sleep(.05)


def test_receipt_states_what_was_enforced(box):
    profile, policy, tmp_path = box
    result, out, err = _run(profile, tmp_path, 'pass')
    assert result['exit_code'] == 0, err
    assert result['containment'] == 'posix-process-group'
    assert result['cleanup_verified'] is True and result['descendants_at_exit'] == 0
    root = str(tmp_path)
    assert result['confinement'] == {
        'backend': 'linux-bwrap', 'policy_sha256': policy_sha256(policy), 'pid': result['pid'],
        'bwrap': shutil.which('bwrap'), 'task_limit': 1024, 'memory_limit': None, 'network': 'private-loopback',
        'seccomp_sha256': hashlib.sha256(seccomp_program(os.uname().machine)).hexdigest(),
        'masked': [root + '/ro/.env', root + '/ro/.ssh', root + '/ro/npmrc-target', root + '/rw/nested/.env']}


def test_the_check_gets_exactly_its_environment(box):
    profile, policy, tmp_path = box
    environment = {'PATH': '/usr/bin:/bin', 'TARE_PROBE': 'value with spaces'}
    profile = LinuxBwrapProfile(policy, cwd=profile._cwd, environment=environment)
    from compute_plane.managed_process import run_process
    dest = tmp_path / 'runs' / 'environment'
    result = run_process([sys.executable, '-I', '-c', 'import json, os; print(json.dumps(dict(os.environ)))'],
                         cwd=profile._cwd, dest=dest, timeout=60, env=environment, process_factory=profile)
    assert result['exit_code'] == 0, (dest / 'stderr.txt').read_text()
    found = json.loads((dest / 'stdout.txt').read_text())
    assert found.pop('PWD') == profile._cwd  # bwrap's, after --chdir
    found.pop('LC_CTYPE', None)  # Python's own C-locale coercion (PEP 538)
    assert found == environment


def test_launch_refuses_a_changed_binding_an_outside_program_and_a_terminal(box):
    profile, policy, tmp_path = box
    with open(os.devnull, 'rb') as stdin, open(os.devnull, 'wb') as stdout:
        launch = dict(cwd=profile._cwd, env={}, stdin=stdin, stdout=stdout, stderr=stdout, start_new_session=True)
        for change, code in (({'start_new_session': False}, 'binding_changed'), ({'env': {'A': 'B'}}, 'binding_changed'),
                             ({'cwd': str(tmp_path)}, 'binding_changed')):
            with pytest.raises(ValueError, match=code):
                profile([sys.executable], **{**launch, **change})
        for argv in ([str(tmp_path / 'outside' / 'tool')], ['python3'], [str(tmp_path / 'ro' / '..' / 'outside')], []):
            with pytest.raises(ValueError, match='argv_invalid|executable_outside_policy'):
                profile(argv, **launch)
        primary, secondary = os.openpty()
        try:
            with open(secondary, 'wb') as terminal, pytest.raises(ValueError, match='terminal_stdio_refused'):
                profile([sys.executable], **{**launch, 'stdout': terminal})
        finally:
            os.close(primary)


def test_stock_host_reports_refused_user_namespaces():
    """Stock Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor. Where that restriction is on and
    bwrap is refused, the support check says so. The CI sandbox job runs this before lowering the restriction."""
    try:
        restricted = Path('/proc/sys/kernel/apparmor_restrict_unprivileged_userns').read_text().strip() == '1'
    except OSError:
        restricted = False
    bwrap = shutil.which('bwrap')
    if not restricted or not bwrap or subprocess.run(
            [bwrap, '--unshare-user', '--unshare-net', '--ro-bind', '/', '/', 'true'],
            capture_output=True).returncode == 0:
        pytest.skip('this host does not refuse bubblewrap its user namespaces')
    with pytest.raises(OSError, match='^bwrap_userns_refused: .*apparmor_restrict_unprivileged_userns=1'):
        require_support()
