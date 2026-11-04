"""Immutable PSEC policy and explicit native Windows MXC process factory.

ABI: microsoft/mxc commit ad54c69452147bd4ca15d027388ad6074903dbff (PSEC 1.0/1.1).
WindowsMxcProfile validates live paths and uses the existing process supervisor.
"""

from dataclasses import dataclass
import hashlib
import ntpath

import flatbuffers


# Bounded native job UI restrictions: clipboard read/write, system parameters,
# display settings, global atoms, desktop access, and exit-Windows.
# GUI HANDLES (bit 0x1) is excluded from 0xFE.
# This is not a claim of execution isolation; Win32k calls remain permitted.
_UI_RESTRICTIONS = 0xFE


def _paths(values, group: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f'{group} must be a sequence of paths')
    copied = tuple(values)
    if len(copied) > 32:
        raise ValueError(f'{group} allows at most 32 paths')
    result = []
    seen = set()
    for path in copied:
        if not isinstance(path, str):
            raise TypeError(f'{group} paths must be strings')
        # Validate the original spelling, before normpath can erase traversal.
        if not path or '\x00' in path or any(c in path for c in '*?'):
            raise ValueError(f'{group} contains an empty, NUL, or wildcard path')
        spelling = path.replace('/', '\\')
        if '..' in spelling.split('\\'):
            raise ValueError(f'{group} contains a traversal segment')
        drive, tail = ntpath.splitdrive(spelling)
        drive_path = (len(drive) == 2 and drive[0] in
                      'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                      and drive[1] == ':' and tail.startswith('\\'))
        unc_parts = drive[2:].split('\\') if drive.startswith('\\\\') else []
        unc_path = (len(unc_parts) == 2 and
                    all(part and part not in ('.', '..') and ':' not in part
                        for part in unc_parts) and
                    (not tail or tail.startswith('\\')))
        if not (drive_path or unc_path):
            raise ValueError(f'{group} requires absolute Windows drive or UNC paths')
        canonical = ntpath.normpath(path)
        _, canonical_tail = ntpath.splitdrive(canonical)
        if canonical_tail in ('', '\\') and group != 'enumerate_only':
            raise ValueError(f'{group} cannot include a volume root')
        if canonical in seen:
            raise ValueError(f'{group} contains a duplicate path')
        seen.add(canonical)
        result.append(canonical)
    return tuple(result)


@dataclass(frozen=True)
class MxcPolicy:
    read_only: tuple[str, ...]
    read_write: tuple[str, ...]
    denied: tuple[str, ...] = ()
    registry_read: bool = False
    enumerate_only: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.registry_read) is not bool:
            raise TypeError('registry_read must be a bool')
        for group in ('read_only', 'read_write', 'denied', 'enumerate_only'):
            object.__setattr__(self, group, _paths(getattr(self, group), group))
        if not (self.read_only or self.read_write):
            raise ValueError('read_only or read_write scope must be nonempty')


def _string_vector(builder, paths):
    strings = [builder.CreateString(path) for path in paths]
    builder.StartVector(4, len(strings), 4)
    for offset in reversed(strings):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def build_psec_spec(policy: MxcPolicy) -> bytes:
    """Encode the fixed deny-network profile using the upstream FlatBuffers builder."""
    if not isinstance(policy, MxcPolicy):
        raise TypeError('policy must be an MxcPolicy')
    builder = flatbuffers.Builder(256)
    read_write = _string_vector(builder, policy.read_write)
    read_only = _string_vector(builder, policy.read_only)
    denied = _string_vector(builder, policy.denied)
    enumerate_only = _string_vector(builder, policy.enumerate_only) if policy.enumerate_only else None
    capability = builder.CreateString('registryRead') if policy.registry_read else None

    # EndpointPolicy: default_action=deny (0); allow and deny arrays absent.
    builder.StartObject(3)
    builder.PrependInt8Slot(0, 0, 0)
    egress = builder.EndObject()
    # NetworkPolicy: ONLY egress, with no proxy or AppContainer peer.
    builder.StartObject(3)
    builder.PrependUOffsetTRelativeSlot(1, egress, 0)
    network = builder.EndObject()

    builder.StartObject(9 if enumerate_only is not None else 8)
    if enumerate_only is not None:
        builder.PrependUOffsetTRelativeSlot(8, enumerate_only, 0)
    builder.PrependUOffsetTRelativeSlot(7, network, 0)
    builder.PrependUOffsetTRelativeSlot(6, denied, 0)
    builder.PrependUOffsetTRelativeSlot(5, read_only, 0)
    builder.PrependUOffsetTRelativeSlot(4, read_write, 0)
    builder.PrependUint64Slot(3, _UI_RESTRICTIONS, 0)
    builder.PrependBoolSlot(2, False, False)
    if capability is not None:
        builder.PrependUOffsetTRelativeSlot(1, capability, 0)
    # Required SchemaVersion is inline: construct at the current root offset,
    # not as a child table or before starting the root object.
    builder.Prep(2, 4)
    builder.PrependUint16(1 if enumerate_only is not None else 0)  # minor
    builder.PrependUint16(1)  # major
    version = builder.Offset()
    builder.PrependStructSlot(0, version, 0)
    root = builder.EndObject()
    builder.Finish(root, file_identifier=b'PSEC')
    return bytes(builder.Output())


def policy_sha256(policy: MxcPolicy) -> str:
    """Hash precisely the finished PSEC bytes, not an alternate policy encoding."""
    return hashlib.sha256(build_psec_spec(policy)).hexdigest()


def _load_mxc_library():
    import os
    if os.name != 'nt':
        raise OSError('MxcEnvironment requires Windows')
    import ctypes
    return ctypes.WinDLL('processmodel.dll', winmode=0x800)


def _path_alias(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)


class MxcEnvironment:
    """Own one native PSEC context without launching a process."""

    def __init__(self, policy: MxcPolicy):
        if not isinstance(policy, MxcPolicy):
            raise TypeError('policy must be an MxcPolicy')
        import ctypes

        self._dll = _load_mxc_library()
        create = self._dll.CreateProcessSecurityEnvironment
        query = self._dll.QueryProcessSecurityEnvironmentSupport
        self._close = self._dll.CloseProcessSecurityEnvironment
        create.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int32
        query.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int32
        self._close.argtypes = [ctypes.c_void_p]
        self._close.restype = None

        support = ctypes.c_uint64()
        hr = query(ctypes.byref(support))
        if hr < 0:
            raise OSError('QueryProcessSecurityEnvironmentSupport failed: '
                          f'HRESULT 0x{hr & 0xFFFFFFFF:08X}')
        self.support_flags = support.value
        if policy.denied and not (self.support_flags & 1):
            raise OSError('Process security environment deny support unavailable')
        if policy.enumerate_only:
            if not (self.support_flags & 4):
                raise OSError('Process security environment enumeration-only support unavailable; no broader read fallback')
            version_query = self._dll.IsProcessSecurityEnvironmentVersionSupported
            version_query.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_ubyte),
                                      ctypes.POINTER(ctypes.c_uint32)]
            version_query.restype = ctypes.c_int32
            available, minor = ctypes.c_ubyte(), ctypes.c_uint32()
            hr = version_query(1, ctypes.byref(available), ctypes.byref(minor))
            if hr < 0 or not available.value or minor.value < 1:
                raise OSError('Process security environment PSEC 1.1 unavailable; no broader read fallback')

        spec = build_psec_spec(policy)
        blob = ctypes.create_string_buffer(spec)
        self._handle = ctypes.c_void_p()
        try:
            hr = create(blob, len(spec), 0, ctypes.byref(self._handle))
            if hr < 0:
                raise OSError('CreateProcessSecurityEnvironment failed: '
                              f'HRESULT 0x{hr & 0xFFFFFFFF:08X}')
            if not self._handle.value:
                raise OSError('CreateProcessSecurityEnvironment returned null handle')
        except BaseException:
            self.close()
            raise

    @property
    def handle(self):
        if not self._handle.value:
            raise RuntimeError('MxcEnvironment is closed')
        return self._handle.value

    def close(self):
        handle = self._handle.value
        if handle:
            self._handle.value = None
            self._close(handle)

    def __enter__(self):
        self.handle
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class WindowsMxcProfile:
    """Explicit native PSEC process factory for the existing WindowsJob supervisor."""
    # A verifier can spawn Git, its native shell and transport helpers below
    # Python. Count the complete tree while preserving per-process/job memory.
    job_limits = (16, 256 * 1024**2, 512 * 1024**2)

    def __init__(self, policy, *, cwd, environment):
        import os
        from pathlib import Path
        from .mediated_tools import validate_command_confinement
        if os.name != 'nt':
            raise OSError('windows_mxc_windows_only')
        if not isinstance(policy, MxcPolicy):
            raise TypeError('mxc_policy_required')
        self._policy = policy
        self._policy_sha256 = policy_sha256(policy)
        self._environment, _ = validate_command_confinement(self, environment, 1)
        self._roots = tuple(self._scope(p) for p in (*policy.read_only, *policy.read_write))
        self._enumerated = tuple(self._directory(p) for p in policy.enumerate_only)
        for path in policy.denied:
            # Denied paths may be absent; the native engine still owns the rule.
            if not ntpath.splitdrive(path)[0].endswith(':'):
                raise ValueError('mxc_local_paths_required')
        self._cwd_source = cwd
        self._cwd = self._directory(cwd)
        if not self._admitted(self._cwd):
            raise ValueError('mxc_cwd_outside_policy')

    @staticmethod
    def _directory(value):
        from pathlib import Path
        if not isinstance(value, str) or '\0' in value or not ntpath.splitdrive(value)[0].endswith(':'):
            raise ValueError('mxc_local_directory_required')
        path = Path(value)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError('mxc_directory_unavailable')
        if any(_path_alias(p) for p in (path, *path.parents)):
            raise ValueError('mxc_directory_alias_refused')
        return path.resolve(strict=True)

    @staticmethod
    def _scope(value):
        from pathlib import Path
        import stat
        value, = _paths((value,), 'scope')
        if not ntpath.splitdrive(value)[0].endswith(':'):
            raise ValueError('mxc_local_paths_required')
        path = Path(value)
        if not path.is_absolute():
            raise ValueError('mxc_scope_unavailable')
        directory = False
        for item in (path, *path.parents):
            try:
                info = item.lstat()
            except FileNotFoundError:
                # Missing roots are exact file grants, never parent grants.
                continue
            if _path_alias(item):
                raise ValueError('mxc_scope_alias_refused')
            if item != path:
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError('mxc_scope_ancestor_unavailable')
            elif stat.S_ISDIR(info.st_mode):
                directory = True
            elif not stat.S_ISREG(info.st_mode):
                raise ValueError('mxc_scope_unavailable')
            elif info.st_nlink != 1:
                raise ValueError('mxc_scope_hardlink_refused')
        return path.resolve(strict=False), directory

    def _revalidate(self):
        if self._directory(self._cwd_source) != self._cwd:
            raise ValueError('mxc_cwd_changed')
        roots = tuple(self._scope(p) for p in
                      (*self._policy.read_only, *self._policy.read_write))
        if roots != self._roots:
            raise ValueError('mxc_scope_changed')
        if tuple(self._directory(p) for p in self._policy.enumerate_only) != self._enumerated:
            raise ValueError('mxc_enumeration_scope_changed')

    def _admitted(self, path):
        return any(path == root or (directory and root in path.parents)
                   for root, directory in self._roots)

    def __call__(self, argv, *, cwd, env, stdin, stdout, stderr, creationflags):
        import ctypes as c
        from ctypes import wintypes as w
        import msvcrt
        from pathlib import Path
        import subprocess
        from .windows_appcontainer import _AppContainerProcess
        if creationflags != 4 or cwd != str(self._cwd) or env != self._environment:
            raise ValueError('mxc_launch_binding_changed')
        if policy_sha256(self._policy) != self._policy_sha256:
            raise ValueError('mxc_policy_changed')
        self._revalidate()
        if (not isinstance(argv, (list, tuple)) or not argv or
                any(not isinstance(a, str) or '\0' in a for a in argv)):
            raise ValueError('mxc_argv_invalid')
        executable = Path(argv[0])
        if (not executable.is_absolute() or not executable.is_file() or
                any(_path_alias(p) for p in (executable, *executable.parents)) or
                not self._admitted(executable.resolve(strict=True))):
            raise ValueError('mxc_executable_outside_policy')
        k = c.WinDLL('kernel32', use_last_error=True, winmode=0x800)

        class StartupInfo(c.Structure):
            _fields_ = [('cb', w.DWORD), ('lpReserved', w.LPWSTR), ('lpDesktop', w.LPWSTR), ('lpTitle', w.LPWSTR),
                ('dwX', w.DWORD), ('dwY', w.DWORD), ('dwXSize', w.DWORD), ('dwYSize', w.DWORD),
                ('dwXCountChars', w.DWORD), ('dwYCountChars', w.DWORD), ('dwFillAttribute', w.DWORD),
                ('dwFlags', w.DWORD), ('wShowWindow', w.WORD), ('cbReserved2', w.WORD),
                ('lpReserved2', c.POINTER(c.c_byte)), ('hStdInput', w.HANDLE), ('hStdOutput', w.HANDLE), ('hStdError', w.HANDLE)]
        class StartupInfoEx(c.Structure):
            _fields_ = [('StartupInfo', StartupInfo), ('lpAttributeList', c.c_void_p)]
        class ProcessInfo(c.Structure):
            _fields_ = [('hProcess', w.HANDLE), ('hThread', w.HANDLE), ('dwProcessId', w.DWORD), ('dwThreadId', w.DWORD)]
        signatures = {
            'GetCurrentProcess': ([], w.HANDLE),
            'DuplicateHandle': ([w.HANDLE,w.HANDLE,w.HANDLE,c.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD],w.BOOL),
            'CloseHandle': ([w.HANDLE],w.BOOL),
            'InitializeProcThreadAttributeList': ([c.c_void_p,w.DWORD,w.DWORD,c.POINTER(c.c_size_t)],w.BOOL),
            'UpdateProcThreadAttribute': ([c.c_void_p,w.DWORD,c.c_size_t,c.c_void_p,c.c_size_t,c.c_void_p,c.c_void_p],w.BOOL),
            'DeleteProcThreadAttributeList': ([c.c_void_p],None),
            'CreateProcessW': ([w.LPCWSTR,w.LPWSTR,c.c_void_p,c.c_void_p,w.BOOL,w.DWORD,c.c_void_p,w.LPCWSTR,c.POINTER(StartupInfoEx),c.POINTER(ProcessInfo)],w.BOOL),
            'GetExitCodeProcess': ([w.HANDLE,c.POINTER(w.DWORD)],w.BOOL),
            'WaitForSingleObject': ([w.HANDLE,w.DWORD],w.DWORD),
            'TerminateProcess': ([w.HANDLE,w.UINT],w.BOOL),
        }
        for name,(arguments,result) in signatures.items():
            function = getattr(k,name); function.argtypes = arguments; function.restype = result
        context = MxcEnvironment(self._policy)
        handles=[]; initialized=False; created=False; transferred=False; info=ProcessInfo()
        try:
            current=k.GetCurrentProcess()
            for stream in (stdin,stdout,stderr):
                handle=w.HANDLE()
                if not k.DuplicateHandle(current,msvcrt.get_osfhandle(stream.fileno()),current,c.byref(handle),0,True,2):
                    raise c.WinError(c.get_last_error())
                handles.append(handle)
            inherited=(w.HANDLE*3)(*(h.value for h in handles))
            size=c.c_size_t()
            sized=k.InitializeProcThreadAttributeList(None,2,0,c.byref(size))
            if sized or c.get_last_error()!=122 or not 0<size.value<=65536:
                raise OSError('mxc_attribute_size_failed')
            storage=c.create_string_buffer(size.value); attributes=c.cast(storage,c.c_void_p)
            if not k.InitializeProcThreadAttributeList(attributes,2,0,c.byref(size)):
                raise c.WinError(c.get_last_error())
            initialized=True
            native_environment=w.HANDLE(context.handle)
            for key,value,length in ((0x20023,c.byref(native_environment),c.sizeof(native_environment)),
                    (0x20002,c.cast(inherited,c.c_void_p),c.sizeof(inherited))):
                if not k.UpdateProcThreadAttribute(attributes,0,key,value,length,None,None):
                    raise c.WinError(c.get_last_error())
            startup=StartupInfoEx(); startup.StartupInfo.cb=c.sizeof(startup)
            startup.StartupInfo.dwFlags=0x100
            startup.StartupInfo.hStdInput,startup.StartupInfo.hStdOutput,startup.StartupInfo.hStdError=inherited
            startup.lpAttributeList=attributes
            command=c.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
            child_environment=c.create_unicode_buffer('\0'.join(name+'='+value for name,value in sorted(env.items(),key=lambda pair:pair[0].upper()))+'\0\0')
            self._revalidate()
            if not k.CreateProcessW(str(executable),command,None,None,True,4|0x400|0x80000|0x08000000,
                    child_environment,str(self._cwd),c.byref(startup),c.byref(info)):
                raise c.WinError(c.get_last_error())
            created=True
            class Child(_AppContainerProcess):
                def close_process(child):
                    try:
                        super().close_process()
                    finally:
                        context.close()
            child=Child(k,c,info.hProcess,info.dwProcessId)
            child.confinement_receipt={'backend':'windows-mxc','policy_sha256':self._policy_sha256,
                'support_flags':context.support_flags,'pid':child.pid,
                'api':'CreateProcessSecurityEnvironment+CreateProcessW'}
            info.hProcess=None; transferred=True
            return child
        except BaseException:
            if created and info.hProcess:
                if not k.TerminateProcess(info.hProcess,1):
                    raise OSError('mxc_partial_child_termination_failed')
                if k.WaitForSingleObject(info.hProcess,5000)!=0:
                    raise OSError('mxc_partial_child_cleanup_unverified')
            raise
        finally:
            try:
                if initialized: k.DeleteProcThreadAttributeList(attributes)
                if info.hThread: k.CloseHandle(info.hThread)
                if info.hProcess: k.CloseHandle(info.hProcess)
                for handle in handles: k.CloseHandle(handle)
            finally:
                if not transferred: context.close()
