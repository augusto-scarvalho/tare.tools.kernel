"""Disposable Windows AppContainer/LPAC process factory.

This constrains process identity and inherited handles; it is not a filesystem
sandbox. The host must provision any required directory ACLs separately.
"""
from __future__ import annotations

import os
import subprocess
import time


class AppContainerProfile:
    """Own a fresh LPAC; registryRead is a separately requested compatibility capability."""

    job_limits = (4, 256 * 1024 * 1024, 512 * 1024 * 1024)

    def __init__(self, name, *, system_registry_read=False):
        if os.name != 'nt':
            raise OSError('appcontainer_windows_only')
        import re
        if not isinstance(name, str) or re.fullmatch(r'tare\.tcp04\.[0-9a-f]{32}', name) is None:
            raise ValueError('invalid_appcontainer_profile_name')
        if type(system_registry_read) is not bool:
            raise ValueError('registry_read_switch_must_be_boolean')
        import ctypes as c
        from ctypes import wintypes as w

        self._c, self._w = c, w
        self._k = c.WinDLL('kernel32', use_last_error=True)
        self._a = c.WinDLL('userenv', use_last_error=True)
        self._advapi = c.WinDLL('advapi32', use_last_error=True)
        self.name = name
        self._sid = c.c_void_p()
        self._closed = False
        self._a.CreateAppContainerProfile.argtypes = [w.LPCWSTR, w.LPCWSTR, w.LPCWSTR, c.c_void_p, w.DWORD, c.POINTER(c.c_void_p)]
        self._a.CreateAppContainerProfile.restype = c.c_long
        self._a.DeleteAppContainerProfile.argtypes = [w.LPCWSTR]
        self._a.DeleteAppContainerProfile.restype = c.c_long
        self._advapi.FreeSid.argtypes = [c.c_void_p]
        self._advapi.FreeSid.restype = c.c_void_p
        self._advapi.ConvertSidToStringSidW.argtypes = [c.c_void_p, c.POINTER(w.LPWSTR)]
        self._advapi.ConvertSidToStringSidW.restype = w.BOOL
        self._k.LocalFree.argtypes = [c.c_void_p]
        self._k.LocalFree.restype = c.c_void_p
        self.capability_names = ('registryRead',) if system_registry_read else ()
        self._capabilities, self._capability_buffers, self.capability_sids = None, [], ()
        if system_registry_read:
            self._derive_registry_read()
        caps = c.cast(self._capabilities, c.c_void_p) if self._capabilities is not None else None
        hr = self._a.CreateAppContainerProfile(name, name, name, caps, len(self.capability_sids), c.byref(self._sid))
        if hr < 0:
            raise OSError(f'CreateAppContainerProfile failed: HRESULT 0x{hr & 0xffffffff:08x}')
        try:
            text = w.LPWSTR()
            if not self._advapi.ConvertSidToStringSidW(self._sid, c.byref(text)):
                raise c.WinError(c.get_last_error())
            try:
                self.sid_text = text.value
            finally:
                self._k.LocalFree(text)
            self._a.GetAppContainerFolderPath.argtypes = [w.LPCWSTR, c.POINTER(w.LPWSTR)]
            self._a.GetAppContainerFolderPath.restype = c.c_long
            folder = w.LPWSTR()
            hr = self._a.GetAppContainerFolderPath(self.sid_text, c.byref(folder))
            if hr < 0:
                raise OSError('native_appcontainer_folder_unavailable')
            try:
                from pathlib import Path
                self.storage_root = Path(folder.value).parent
                if (not self.storage_root.is_absolute() or self.storage_root.name != self.name
                        or self.storage_root.parent.name.casefold() != 'packages'
                        or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                               for p in (self.storage_root, *self.storage_root.parents))):
                    raise OSError('native_appcontainer_storage_path_invalid')
            finally:
                ole = c.WinDLL('ole32')
                ole.CoTaskMemFree.argtypes = [c.c_void_p]; ole.CoTaskMemFree.restype = None
                ole.CoTaskMemFree(folder)
        except BaseException:
            delete_hr = self._a.DeleteAppContainerProfile(name)
            if delete_hr < 0:
                raise OSError(f'constructor cleanup DeleteAppContainerProfile failed: HRESULT 0x{delete_hr & 0xffffffff:08x}')
            freed = self._advapi.FreeSid(self._sid)
            if freed:
                raise OSError('constructor cleanup FreeSid failed')
            self._sid = c.c_void_p()
            self._closed = True
            raise

    @property
    def sid(self):
        if self._closed:
            raise ValueError('appcontainer_profile_closed')
        return self._sid

    def __enter__(self):
        if self._closed:
            raise ValueError('appcontainer_profile_closed')
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._closed:
            return
        hr = self._a.DeleteAppContainerProfile(self.name)
        if hr < 0:
            raise OSError(f'DeleteAppContainerProfile failed: HRESULT 0x{hr & 0xffffffff:08x}')
        freed = self._advapi.FreeSid(self._sid)
        if freed:
            raise OSError('FreeSid failed')
        self._sid = self._c.c_void_p()
        self._closed = True

    def spawn(self, argv, *, cwd, env, stdin, stdout, stderr, creationflags):
        """Create a suspended LPAC child; its owner must assign containment then resume."""
        if self._closed:
            raise ValueError('appcontainer_profile_closed')
        if (not isinstance(argv, (list, tuple)) or not argv or
                any(not isinstance(arg, str) or '\0' in arg for arg in argv)):
            raise ValueError('argv_must_be_nonempty_strings_without_nul')
        executable = argv[0]
        if not os.path.isabs(executable) or not os.path.isfile(executable):
            raise FileNotFoundError('executable_must_be_absolute_and_exist')
        if not isinstance(cwd, str) or not os.path.isabs(cwd) or not os.path.isdir(cwd):
            raise FileNotFoundError('cwd_must_be_absolute_and_exist')
        from .mediated_tools import validate_command_confinement
        env, _ = validate_command_confinement(self.spawn, env, 1)
        if type(creationflags) is not int or not creationflags & 0x4:
            raise ValueError('CREATE_SUSPENDED_required')
        streams = (stdin, stdout, stderr)
        if any(stream is None or not callable(getattr(stream, 'fileno', None)) for stream in streams):
            raise ValueError('stdio_must_be_files_with_fileno')
        c, w, k = self._c, self._w, self._k

        class SecurityCapabilities(c.Structure):
            _fields_ = [('AppContainerSid', c.c_void_p), ('Capabilities', c.c_void_p), ('CapabilityCount', w.DWORD), ('Reserved', w.DWORD)]
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

        k.GetCurrentProcess.restype = w.HANDLE
        k.GetCurrentProcess.argtypes = []
        k.DuplicateHandle.argtypes = [w.HANDLE, w.HANDLE, w.HANDLE, c.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD]
        k.DuplicateHandle.restype = w.BOOL
        k.CloseHandle.argtypes = [w.HANDLE]
        k.CloseHandle.restype = w.BOOL
        k.InitializeProcThreadAttributeList.argtypes = [c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t)]
        k.InitializeProcThreadAttributeList.restype = w.BOOL
        k.UpdateProcThreadAttribute.argtypes = [c.c_void_p, w.DWORD, c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p, c.c_void_p]
        k.UpdateProcThreadAttribute.restype = w.BOOL
        k.DeleteProcThreadAttributeList.argtypes = [c.c_void_p]
        k.DeleteProcThreadAttributeList.restype = None
        k.CreateProcessW.argtypes = [w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p, w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR, c.POINTER(StartupInfoEx), c.POINTER(ProcessInfo)]
        k.CreateProcessW.restype = w.BOOL
        k.GetExitCodeProcess.argtypes = [w.HANDLE, c.POINTER(w.DWORD)]
        k.GetExitCodeProcess.restype = w.BOOL
        k.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        k.WaitForSingleObject.restype = w.DWORD
        k.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
        k.TerminateProcess.restype = w.BOOL

        inherited = []
        attr_mem = None
        initialized = False
        info = ProcessInfo()
        created = False
        try:
            current = k.GetCurrentProcess()
            import msvcrt
            for stream in streams:
                source = w.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
                duplicate = w.HANDLE()
                if not k.DuplicateHandle(current, source, current, c.byref(duplicate), 0, True, 2):
                    raise c.WinError(c.get_last_error())
                inherited.append(duplicate)
            handles = (w.HANDLE * 3)(*(h.value for h in inherited))
            caps = SecurityCapabilities(self._sid,
                c.cast(self._capabilities, c.c_void_p) if self._capabilities is not None else None,
                len(self.capability_sids), 0)
            size = c.c_size_t()
            k.InitializeProcThreadAttributeList(None, 3, 0, c.byref(size))
            attr_mem = c.create_string_buffer(size.value)
            attr = c.cast(attr_mem, c.c_void_p)
            if not k.InitializeProcThreadAttributeList(attr, 3, 0, c.byref(size)):
                raise c.WinError(c.get_last_error())
            initialized = True
            def update(key, value, length):
                if not k.UpdateProcThreadAttribute(attr, 0, key, value, length, None, None):
                    raise c.WinError(c.get_last_error())
            update(0x20009, c.byref(caps), c.sizeof(caps))
            lpac = w.DWORD(1)
            update(0x2000f, c.byref(lpac), c.sizeof(lpac))
            update(0x20002, c.cast(handles, c.c_void_p), c.sizeof(handles))
            startup = StartupInfoEx()
            startup.StartupInfo.cb = c.sizeof(startup)
            startup.StartupInfo.dwFlags = 0x100
            startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = handles[0], handles[1], handles[2]
            startup.lpAttributeList = attr
            command = c.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
            environment = c.create_unicode_buffer('\0'.join(k0 + '=' + v for k0, v in sorted(env.items(), key=lambda item: item[0].upper())) + '\0\0')
            flags = 0x4 | 0x400 | 0x80000 | 0x08000000
            if not k.CreateProcessW(executable, command, None, None, True, flags, environment, cwd, c.byref(startup), c.byref(info)):
                raise c.WinError(c.get_last_error())
            created = True
            identity = self.verify_suspended_token(info.hProcess)
            process = _AppContainerProcess(k, c, info.hProcess, info.dwProcessId)
            process.confinement_receipt = identity
            info.hProcess = None
            return process
        except BaseException:
            if created and info.hProcess:
                k.TerminateProcess(info.hProcess, 1)
                k.WaitForSingleObject(info.hProcess, 0xffffffff)
            raise
        finally:
            if initialized:
                k.DeleteProcThreadAttributeList(attr)
            if info.hThread:
                k.CloseHandle(info.hThread)
            if info.hProcess:
                k.CloseHandle(info.hProcess)
            for handle in inherited:
                k.CloseHandle(handle)

    def __call__(self, *args, **kwargs):
        return self.spawn(*args, **kwargs)

    def _sid_string(self, sid):
        text = self._w.LPWSTR()
        if not self._advapi.ConvertSidToStringSidW(sid, self._c.byref(text)):
            raise self._c.WinError(self._c.get_last_error())
        try:
            return text.value
        finally:
            self._k.LocalFree(text)

    def _derive_registry_read(self):
        c, w = self._c, self._w
        api = c.WinDLL('kernelbase', use_last_error=True)
        derive = api.DeriveCapabilitySidsFromName
        derive.argtypes = [w.LPCWSTR, c.POINTER(c.POINTER(c.c_void_p)), c.POINTER(w.DWORD),
                           c.POINTER(c.POINTER(c.c_void_p)), c.POINTER(w.DWORD)]
        derive.restype = w.BOOL
        self._advapi.GetLengthSid.argtypes = [c.c_void_p]
        self._advapi.GetLengthSid.restype = w.DWORD
        groups, sids = c.POINTER(c.c_void_p)(), c.POINTER(c.c_void_p)()
        group_count, sid_count = w.DWORD(), w.DWORD()
        try:
            if not derive('registryRead', c.byref(groups), c.byref(group_count), c.byref(sids), c.byref(sid_count)):
                raise c.WinError(c.get_last_error())
            if sid_count.value != 1 or group_count.value > 16:
                raise OSError('registry_read_capability_derivation_invalid')
            length = self._advapi.GetLengthSid(sids[0])
            if not 8 <= length <= 128:
                raise OSError('registry_read_capability_sid_invalid')
            buffer = c.create_string_buffer(length)
            c.memmove(buffer, sids[0], length)
            class SidAndAttributes(c.Structure):
                _fields_ = [('sid', c.c_void_p), ('attributes', w.DWORD)]
            self._capability_buffers = [buffer]
            self._capabilities = (SidAndAttributes * 1)(SidAndAttributes(c.cast(buffer, c.c_void_p), 4))
            self.capability_sids = (self._sid_string(sids[0]),)
        finally:
            for array, count in ((groups, group_count.value), (sids, sid_count.value)):
                if array:
                    for index in range(count): self._k.LocalFree(array[index])
                    self._k.LocalFree(array)

    def verify_suspended_token(self, handle):
        """Check native identity before resume; no claim based on child-supplied JSON.

        Class 46 is not a supported Win32 query on qualified builds. The LPAC
        policy must succeed at creation; AccessCheck independently confirms its
        documented ALL_APPLICATION_PACKAGES exclusion and a positive own-SID grant.
        """
        c, w, api, kernel = self._c, self._w, self._advapi, self._k
        api.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)]
        api.OpenProcessToken.restype = w.BOOL
        api.GetTokenInformation.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.POINTER(w.DWORD)]
        api.GetTokenInformation.restype = w.BOOL
        api.DuplicateToken.argtypes = [w.HANDLE, c.c_int, c.POINTER(w.HANDLE)]
        api.DuplicateToken.restype = w.BOOL
        api.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [w.LPCWSTR, w.DWORD, c.POINTER(c.c_void_p), c.POINTER(w.DWORD)]
        api.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = w.BOOL
        class Mapping(c.Structure):
            _fields_ = [('read', w.DWORD), ('write', w.DWORD), ('execute', w.DWORD), ('all', w.DWORD)]
        api.AccessCheck.argtypes = [c.c_void_p, w.HANDLE, w.DWORD, c.POINTER(Mapping), c.c_void_p,
                                   c.POINTER(w.DWORD), c.POINTER(w.DWORD), c.POINTER(w.BOOL)]
        api.AccessCheck.restype = w.BOOL
        token, duplicate = w.HANDLE(), w.HANDLE()
        if not api.OpenProcessToken(handle, 10, c.byref(token)):
            raise c.WinError(c.get_last_error())
        try:
            def information(kind):
                size = w.DWORD()
                if api.GetTokenInformation(token, kind, None, 0, c.byref(size)) or c.get_last_error() != 122:
                    raise OSError('native_token_query_size_failed')
                if not 4 <= size.value <= 65536:
                    raise OSError('native_token_query_size_invalid')
                data = c.create_string_buffer(size.value)
                if not api.GetTokenInformation(token, kind, data, size, c.byref(size)):
                    raise c.WinError(c.get_last_error())
                return data
            appcontainer = w.DWORD.from_buffer(information(29)).value
            capability_data = information(30)
            count = w.DWORD.from_buffer(capability_data).value
            sid_data = information(31)
            actual_sid = c.c_void_p.from_buffer(sid_data)
            text = w.LPWSTR()
            if not actual_sid.value or not api.ConvertSidToStringSidW(actual_sid, c.byref(text)):
                raise OSError('native_token_package_sid_unavailable')
            try:
                observed_sid = text.value
            finally:
                kernel.LocalFree(text)
            if not appcontainer or count != len(self.capability_sids) or observed_sid != self.sid_text:
                raise OSError('native_token_identity_mismatch')
            observed_caps = ()
            if count:
                class SidAndAttributes(c.Structure):
                    _fields_ = [('sid', c.c_void_p), ('attributes', w.DWORD)]
                class Groups(c.Structure):
                    _fields_ = [('count', w.DWORD), ('groups', SidAndAttributes * 1)]
                groups = Groups.from_buffer(capability_data)
                if not groups.groups[0].attributes & 4 or groups.groups[0].attributes & 16:
                    raise OSError('native_token_capability_disabled')
                observed_caps = (self._sid_string(groups.groups[0].sid),)
            if observed_caps != self.capability_sids:
                raise OSError('native_token_capabilities_mismatch')
            if not api.DuplicateToken(token, 2, c.byref(duplicate)):
                raise c.WinError(c.get_last_error())
            def access(grant):
                descriptor, size = c.c_void_p(), w.DWORD()
                if not api.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                        'O:WDG:WDD:(A;;0x1;;;WD)(A;;0x1;;;' + grant + ')', 1, c.byref(descriptor), c.byref(size)):
                    raise c.WinError(c.get_last_error())
                try:
                    mapping = Mapping(1, 2, 4, 7)
                    privileges, size = c.create_string_buffer(1024), w.DWORD(1024)
                    rights, allowed = w.DWORD(), w.BOOL()
                    if not api.AccessCheck(descriptor, duplicate, 1, c.byref(mapping), privileges,
                                           c.byref(size), c.byref(rights), c.byref(allowed)):
                        raise c.WinError(c.get_last_error())
                    return bool(allowed.value)
                finally:
                    kernel.LocalFree(descriptor)
            if access('AC') or not access(self.sid_text):
                raise OSError('native_lpac_access_check_failed')
            return {'profile': 'windows-lpac/1', 'appcontainer_sid': observed_sid,
                    'capability_count': count, 'all_packages_access': False, 'own_package_access': True,
                    'capability_names': list(self.capability_names), 'capability_sids': list(observed_caps),
                    'lpac_validation': 'startup-policy-and-access-check'}
        finally:
            if duplicate.value: kernel.CloseHandle(duplicate)
            kernel.CloseHandle(token)


class _AppContainerProcess:
    def __init__(self, kernel, ctypes, handle, pid):
        self._k, self._c, self._handle, self.pid = kernel, ctypes, handle, int(pid)
        self.returncode = None
        self._closed = False

    def _collect(self):
        code = self._w.DWORD() if hasattr(self, '_w') else self._c.c_uint32()
        if not self._k.GetExitCodeProcess(self._handle, self._c.byref(code)):
            raise self._c.WinError(self._c.get_last_error())
        self.returncode = int(code.value)
        return self.returncode

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        result = self._k.WaitForSingleObject(self._handle, 0)
        if result == 0x102:
            return None
        if result != 0:
            raise self._c.WinError(self._c.get_last_error())
        return self._collect()

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        ms = 0xffffffff if timeout is None else max(0, min(0xfffffffe, int(timeout * 1000)))
        started = time.monotonic()
        while True:
            remaining = ms if timeout is None else max(0, ms - int((time.monotonic() - started) * 1000))
            result = self._k.WaitForSingleObject(self._handle, min(remaining, 50) if timeout is not None else 50)
            if result == 0:
                return self._collect()
            if result not in (0x102,):
                raise self._c.WinError(self._c.get_last_error())
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired('process', timeout)

    def kill(self):
        if self.poll() is None and not self._k.TerminateProcess(self._handle, 1):
            raise self._c.WinError(self._c.get_last_error())

    def close_process(self):
        if self._handle and not self._closed:
            if not self._k.CloseHandle(self._handle):
                raise self._c.WinError(self._c.get_last_error())
            self._closed = True
            self._handle = None
