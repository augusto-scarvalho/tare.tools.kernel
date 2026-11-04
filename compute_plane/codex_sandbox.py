"""Explicit public Codex MXC adapter for the existing managed_process supervisor."""
import hashlib
import json
from pathlib import Path
import subprocess

from .windows_mxc import MxcPolicy, WindowsMxcProfile, policy_sha256


def _alias(path):
    return path.is_symlink() or bool(getattr(path.stat(follow_symlinks=False),'st_file_attributes',0)&0x400)


class CodexSandboxProfile:
    job_limits=WindowsMxcProfile.job_limits

    def __init__(self,policy,*,executable,cwd,environment,expected_executable_sha256):
        if not isinstance(policy,MxcPolicy):
            raise TypeError('codex_mxc_policy_required')
        if policy.registry_read is not True:
            raise ValueError('codex_mxc_requires_explicit_registry_read')
        if (not isinstance(expected_executable_sha256,str) or len(expected_executable_sha256)!=64 or
                any(c not in '0123456789abcdef' for c in expected_executable_sha256)):
            raise ValueError('codex_admitted_executable_digest_required')
        self._binding=WindowsMxcProfile(policy,cwd=cwd,environment=environment)
        self._policy=policy
        self._policy_sha256=policy_sha256(policy)
        self._requested_executable=Path(executable)
        if not self._requested_executable.is_absolute():
            raise ValueError('codex_absolute_admitted_executable_required')
        try:
            self._executable=self._requested_executable.resolve(strict=True)
        except (OSError,RuntimeError):
            raise ValueError('codex_absolute_admitted_executable_required') from None
        self._executable_sha256=expected_executable_sha256
        self._check_executable()
        entries=[]
        for access,paths in (('read',policy.read_only),('write',policy.read_write),('deny',policy.denied)):
            entries.extend({'path':{'type':'path','path':p},'access':access} for p in paths)
        state={'codexExecutable':str(self._executable),'permissionProfile':{'type':'managed',
            'file_system':{'type':'restricted','entries':entries},'network':'restricted'},
            'codexLinuxSandboxExe':None,'sandboxCwd':self._binding._cwd.as_uri(),
            'useLegacyLandlock':False,'useMxc':True}
        self._state_json=json.dumps(state,separators=(',',':'),ensure_ascii=False)
        self._state_sha256=hashlib.sha256(self._state_json.encode('utf-8')).hexdigest()

    def _check_executable(self):
        path=self._executable
        try:
            selected=self._requested_executable.resolve(strict=True)
        except (OSError,RuntimeError):
            raise ValueError('codex_executable_changed') from None
        if selected!=path:
            raise ValueError('codex_executable_changed')
        if (not path.is_absolute() or not path.is_file() or
                any(_alias(p) for p in (path,*path.parents))):
            raise ValueError('codex_absolute_admitted_executable_required')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self._executable_sha256:
            raise ValueError('codex_executable_changed')

    def __call__(self,argv,*,cwd,env,stdin,stdout,stderr,creationflags):
        if creationflags!=4 or cwd!=str(self._binding._cwd) or env!=self._binding._environment:
            raise ValueError('codex_mxc_launch_binding_changed')
        if policy_sha256(self._policy)!=self._policy_sha256:
            raise ValueError('codex_mxc_policy_changed')
        if (not isinstance(argv,(list,tuple)) or not argv or
                any(not isinstance(a,str) or '\0' in a for a in argv)):
            raise ValueError('codex_mxc_argv_invalid')
        workload=Path(argv[0])
        if (not workload.is_absolute() or not workload.is_file() or
                any(_alias(p) for p in (workload,*workload.parents)) or
                not self._binding._admitted(workload.resolve(strict=True))):
            raise ValueError('codex_mxc_workload_outside_policy')
        self._check_executable()
        command=[str(self._executable),'-c','windows.sandbox=mxc','sandbox','--sandbox-state-json',
            self._state_json,'--sandbox-state-disable-network','--',*argv]
        process=subprocess.Popen(command,cwd=cwd,env=dict(env),stdin=stdin,stdout=stdout,stderr=stderr,
            creationflags=4|0x08000000)
        process.confinement_receipt={'backend':'codex-mxc','launcher_pid':process.pid,
            'requested_permission_sha256':self._state_sha256,'policy_sha256':self._policy_sha256,
            'executable_sha256':self._executable_sha256}
        return process
