"""Opt-in, disposable Windows LPAC workspace; monitored limits are not hard quotas."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from uuid import uuid4

from .mediated_tools import validate_command_confinement
from .windows_appcontainer import AppContainerProfile


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')

def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()

def _safe(path, directory=False):
    info = path.lstat()
    if (info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)):
        raise ValueError('confined_unsafe_path')
    return info

def _snapshot(path, limit):
    before = _safe(path)
    if before.st_size > limit:
        raise ValueError('confined_file_limit')
    with path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_ino != before.st_ino:
            raise ValueError('confined_file_changed')
        data = stream.read(limit + 1)
        after_fd = os.fstat(stream.fileno())
    after = _safe(path)
    if (len(data) > limit or len(data) != before.st_size or
            any((s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev, s.st_nlink) !=
                (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev, before.st_nlink)
                for s in (after_fd, after))):
        raise ValueError('confined_file_changed')
    return data

def _names(files, count, size):
    if type(files) is not dict or len(files) > count:
        raise ValueError('confined_file_count')
    seen, directories, manifest, total = set(), set(), {}, 0
    reserved = {'con', 'prn', 'aux', 'nul', 'conin$', 'conout$'}
    reserved.update(f'{prefix}{n}' for prefix in ('com', 'lpt') for n in range(1, 10))
    for name, data in files.items():
        if not isinstance(name, str) or not name or len(name) > 240 or not name.isascii() or type(data) is not bytes:
            raise ValueError('confined_invalid_file')
        parts = name.split('/')
        if any(not part or part in ('.', '..') or part[-1] in '. ' or
               not re.fullmatch(r'[A-Za-z0-9_.$ -]+', part) or
               part.split('.')[0].casefold() in reserved for part in parts):
            raise ValueError('confined_invalid_name')
        key = name.casefold()
        prefixes = {'/'.join(parts[:i]).casefold() for i in range(1, len(parts))}
        if key in seen or key in directories or prefixes & seen:
            raise ValueError('confined_name_collision')
        seen.add(key)
        directories.update(prefixes)
        total += len(data)
        if total > size:
            raise ValueError('confined_file_bytes')
        manifest[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return manifest

def _walk(root, *, read_bytes=True):
    _safe(root, True)
    files, entries, size = {}, 0, 0
    def visit(folder, prefix=''):
        nonlocal entries, size
        for entry in os.scandir(folder):
            entries += 1
            if entries > 512:
                raise ValueError('confined_entry_limit')
            path = Path(entry.path)
            relative = prefix + entry.name
            info = _safe(path, entry.is_dir(follow_symlinks=False))
            if stat.S_ISDIR(info.st_mode):
                visit(path, relative + '/')
            else:
                if len(files) >= 64 or size + info.st_size > 8 * 1024 * 1024:
                    raise ValueError('confined_output_limit')
                data = _snapshot(path, 8 * 1024 * 1024 - size) if read_bytes else info.st_size
                size += len(data) if read_bytes else data
                files[relative.replace('\\', '/')] = data
    visit(root)
    return files, entries


class ConfinedWorkspace:
    """A single pending execution; no automatic retry on an unknown result."""
    def __init__(self, root, runtime_files: dict[str, bytes], input_files: dict[str, bytes], seconds=10,
                 *, system_registry_read=False):
        if os.name != 'nt':
            raise OSError('confinement_platform_unavailable')
        if type(system_registry_read) is not bool:
            raise ValueError('registry_read_switch_must_be_boolean')
        _, seconds = validate_command_confinement(lambda: None, {}, seconds)
        runtime = _names(runtime_files, 4096, 64 * 1024 * 1024)
        source = _names(input_files, 64, 8 * 1024 * 1024)
        if not isinstance(root, (str, os.PathLike)):
            raise ValueError('confined_root_invalid')
        root = Path(root)
        if (not root.is_absolute() or not re.fullmatch(r'tare-confined\.[0-9a-f]{32}', root.name)
                or os.path.lexists(root)):
            raise ValueError('confined_root_invalid')
        for ancestor in (root.parent, *root.parent.parents):
            _safe(ancestor, True)
        self.root = root
        self.runtime_root, self.input_root, self.scratch_root = (root / part for part in ('runtime', 'input', 'scratch'))
        self.seconds = seconds
        self.runtime_sha256, self.source_sha256 = _digest(runtime), _digest(source)
        self._manifests = (runtime, source)
        self._profile = None
        self._granted = []
        self._closed = False
        self._pending = None
        root.mkdir()
        try:
            for folder, files in ((self.runtime_root, runtime_files), (self.input_root, input_files)):
                folder.mkdir()
                for name, data in files.items():
                    target = folder.joinpath(*name.split('/'))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream:
                        stream.write(data)
            self.scratch_root.mkdir()
            self._profile = AppContainerProfile('tare.tcp04.' + uuid4().hex,
                                                system_registry_read=system_registry_read)
            (self.scratch_root / 'Packages' / self._profile.name / 'AC' / 'Temp').mkdir(parents=True)
            system = os.environ['SystemRoot']
            system = Path(system)
            if not system.is_absolute():
                raise ValueError('confined_systemroot_invalid')
            self._environment = {'SystemRoot': str(system), 'WINDIR': str(system),
                'PATH': str(self.runtime_root) + ';' + str(system / 'System32'),
                'USERPROFILE': str(self.scratch_root), 'APPDATA': str(self.scratch_root),
                'LOCALAPPDATA': str(self.scratch_root), 'TEMP': str(self.scratch_root),
                'TMP': str(self.scratch_root)}
            icacls = str(system / 'System32' / 'icacls.exe')
            for folder, access in ((self.runtime_root, 'RX'), (self.input_root, 'RX'),
                                   (self.scratch_root, 'M')):
                self._granted.append((folder, icacls))  # Revoke even a partially applied grant.
                self._acl(icacls, folder, '/grant', '*' + self._profile.sid_text + ':(OI)(CI)(' + access + ')')
            self._acl(icacls, self.scratch_root, '/setintegritylevel', '(OI)(CI)L')
            self.profile_sha256 = _digest({'profile': self._profile.name, 'sid': self._profile.sid_text,
                'root': str(root), 'runtime_root': str(self.runtime_root), 'input_root': str(self.input_root),
                'scratch_root': str(self.scratch_root), 'storage_root': str(self._profile.storage_root),
                'runtime_sha256': self.runtime_sha256, 'source_sha256': self.source_sha256,
                'environment': self._environment, 'job_limits': self.job_limits, 'seconds': seconds,
                'capability_names': self._profile.capability_names, 'capability_sids': self._profile.capability_sids,
                'environment_semantics': 'supplied-block+windows-appcontainer-redirects'})
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _acl(exe, folder, command, value):
        subprocess.run([exe, str(folder), command, value], timeout=10, capture_output=True, check=True)

    @property
    def environment(self):
        return dict(self._environment)

    @property
    def job_limits(self):
        return self._profile.job_limits

    @property
    def run_id(self):
        if self._pending is None:
            raise ValueError('confined_run_unavailable')
        return self._pending['run_id']

    def validate(self):
        """Host preflight before assessments and claims, with no effects or retries."""
        if self._closed or self._pending is not None:
            raise ValueError('confined_unavailable')
        self._verify_manifests()
        return True

    def _verify_manifests(self):
        for folder, expected in zip((self.runtime_root, self.input_root), self._manifests):
            observed = {}
            def visit(directory, prefix=''):
                _safe(directory, True)
                for entry in os.scandir(directory):
                    path = Path(entry.path)
                    key = prefix + entry.name
                    info = _safe(path, entry.is_dir(follow_symlinks=False))
                    if stat.S_ISDIR(info.st_mode):
                        if not any(name.startswith(key + '/') for name in expected):
                            raise ValueError('confined_manifest_extra_directory')
                        visit(path, key + '/')
                    else:
                        if key not in expected:
                            raise ValueError('confined_manifest_extra')
                        data = _snapshot(path, expected[key]['bytes'])
                        observed[key] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            visit(folder)
            if observed != expected:
                raise ValueError('confined_manifest_changed')

    def __call__(self, argv, *, cwd, env, stdin, stdout, stderr, creationflags):
        if self._closed or self._pending is not None:
            raise ValueError('confined_unavailable')
        if not isinstance(cwd, str) or os.path.normcase(os.path.abspath(cwd)) != os.path.normcase(str(self.input_root)):
            raise ValueError('confined_cwd_mismatch')
        if type(env) is not dict or env != self._environment:
            raise ValueError('confined_environment_mismatch')
        self._verify_manifests()
        if not isinstance(argv, (tuple, list)) or not argv or not isinstance(argv[0], str):
            raise ValueError('confined_executable_invalid')
        executable = Path(argv[0])
        if not executable.is_absolute() or executable.suffix.casefold() != '.exe':
            raise ValueError('confined_executable_invalid')
        try:
            relative = executable.relative_to(self.runtime_root).as_posix()
        except ValueError:
            raise ValueError('confined_executable_invalid') from None
        if relative not in self._manifests[0]:
            raise ValueError('confined_executable_invalid')
        _safe(executable)
        paths = []
        for stream in (stdin, stdout, stderr):
            if not callable(getattr(stream, 'fileno', None)) or not isinstance(getattr(stream, 'name', None), str):
                raise ValueError('confined_stdio_invalid')
            path = Path(stream.name)
            if not path.is_absolute() or not path.is_file():
                raise ValueError('confined_stdio_invalid')
            _safe(path)
            for ancestor in (path.parent, *path.parent.parents):
                _safe(ancestor, True)
            resolved = path.resolve()
            if resolved.is_relative_to(self.root) or resolved.is_relative_to(self._profile.storage_root):
                raise ValueError('confined_stdio_inside_workspace')
            paths.append(path)
        if len({os.path.normcase(str(path)) for path in paths}) != 3:
            raise ValueError('confined_stdio_alias')
        run_id = uuid4().hex
        self._pending = {'run_id': run_id, 'paths': tuple(paths), 'proc': None}
        # A failed/unknown spawn remains pending; only custody can resolve it.
        proc = self._profile.spawn(argv, cwd=cwd, env=env, stdin=stdin, stdout=stdout,
                                   stderr=stderr, creationflags=creationflags)
        self._pending['proc'] = proc
        self._pending['pid'] = proc.pid
        return proc

    def resource_check(self) -> bool:
        try:
            files_a, entries_a = _walk(self.scratch_root, read_bytes=False)
            files_b, entries_b = _walk(self._profile.storage_root, read_bytes=False)
            return (entries_a + entries_b <= 512 and len(files_a) + len(files_b) <= 64
                    and sum((*files_a.values(), *files_b.values())) <= 8 * 1024 * 1024)
        except (OSError, ValueError):
            return False

    def collect(self, process_receipt) -> dict[str, bytes]:
        pending = self._pending
        if self._closed or pending is None or pending['proc'] is None or type(process_receipt) is not dict:
            raise ValueError('confined_collection_unavailable')
        proc = pending['proc']
        if (process_receipt.get('cleanup_verified') is not True or process_receipt.get('pid') != proc.pid
                or any(process_receipt.get(flag) for flag in ('cancelled', 'timeout', 'output_limit', 'isolation_violation'))
                or proc.returncode is None):
            raise ValueError('confined_receipt_invalid')
        identity = process_receipt.get('confinement')
        if (not isinstance(identity, dict) or identity.get('appcontainer_sid') != self._profile.sid_text
                or identity.get('capability_count') != len(self._profile.capability_sids)
                or identity.get('capability_sids') != list(self._profile.capability_sids)
                or identity.get('capability_names') != list(self._profile.capability_names)):
            raise ValueError('confined_identity_invalid')
        self._verify_manifests()
        if not self.resource_check():
            raise ValueError('confined_resource_violation')
        _, out, err = pending['paths']
        result = {'stdout.txt': _snapshot(out, 1024 * 1024), 'stderr.txt': _snapshot(err, 1024 * 1024),
                  'process.json': _canonical(process_receipt)}
        for folder, prefix in ((self.scratch_root, 'scratch/'), (self._profile.storage_root, 'appdata/')):
            files, _ = _walk(folder)
            result.update((prefix + name, data) for name, data in files.items())
        if len(result) > 64 or sum(map(len, result.values())) > 8 * 1024 * 1024 or not self.resource_check():
            raise ValueError('confined_collection_limit')
        self._pending = None
        return result

    def close(self):
        if self._closed:
            return
        if self._pending is not None and self._pending['proc'] is not None:
            proc = self._pending['proc']
            if proc.returncode is None and proc.poll() is None:
                raise ValueError('confined_process_alive')
        failures = []
        if self._profile is not None:
            for item in list(reversed(self._granted)):
                folder, exe = item
                try:
                    self._acl(exe, folder, '/remove:g', '*' + self._profile.sid_text)
                    self._granted.remove(item)
                except Exception as exc:
                    failures.append(exc)
            try:
                self._profile.close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise OSError('confined_cleanup_unconfirmed') from failures[0]
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
