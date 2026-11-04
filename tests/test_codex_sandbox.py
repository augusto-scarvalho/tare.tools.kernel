import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import compute_plane.codex_sandbox as module
from compute_plane.windows_mxc import MxcPolicy


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    calls=[]
    class Binding:
        job_limits=(4,256*1024**2,512*1024**2)
        def __init__(self,policy,*,cwd,environment):
            self._cwd=Path(cwd);self._environment=dict(environment)
        def _admitted(self,path):
            return path==tmp_path/'workload.exe'
    monkeypatch.setattr(module,'WindowsMxcProfile',Binding)
    monkeypatch.setattr(module.subprocess,'Popen',lambda argv,**kw:(calls.append((argv,kw)) or SimpleNamespace(pid=41)))
    executable=tmp_path/'codex.exe';executable.write_bytes(b'fake admitted Codex')
    workload=tmp_path/'workload.exe';workload.write_bytes(b'fake admitted workload')
    policy=MxcPolicy(('C:/inputs',),('C:/scratch',),('C:/scratch/protected',),registry_read=True)
    env={'SystemRoot':'C:/Windows'}
    factory=module.CodexSandboxProfile(policy,executable=str(executable),cwd=str(tmp_path),environment=env,
        expected_executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
    return factory,calls,executable,workload,policy,env,tmp_path


def invoke(fixture,**changes):
    factory,calls,executable,workload,policy,env,root=fixture
    args=dict(argv=[str(workload),'spaced argument','literal; text'],cwd=str(root),env=env,
              stdin=None,stdout=None,stderr=None,creationflags=4)
    args.update(changes)
    return factory(**args)


def test_exact_state_and_original_argv(fixture):
    process=invoke(fixture);factory,calls,exe,workload,policy,env,root=fixture
    argv,kwargs=calls[0]
    assert argv[:4]==[str(exe),'-c','windows.sandbox=mxc','sandbox']
    state=json.loads(argv[5]);profile=state['permissionProfile']
    assert state['useMxc'] is True and state['useLegacyLandlock'] is False
    assert state['sandboxCwd']==root.as_uri() and state['codexExecutable']==str(exe)
    assert profile['type']=='managed' and profile['network']=='restricted'
    assert profile['file_system']['type']=='restricted'
    assert [(e['access'],e['path']['type'],e['path']['path']) for e in profile['file_system']['entries']]==[
        ('read','path',policy.read_only[0]),('write','path',policy.read_write[0]),('deny','path',policy.denied[0])]
    assert argv[6:]==['--sandbox-state-disable-network','--',str(workload),'spaced argument','literal; text']
    assert kwargs['creationflags']==4|0x08000000 and kwargs['env']==env
    assert process.confinement_receipt['launcher_pid']==41
    assert process.confinement_receipt['requested_permission_sha256']==hashlib.sha256(argv[5].encode()).hexdigest()


@pytest.mark.parametrize('change',[{'cwd':'C:/other'},{'env':{'secret':'not admitted'}},{'creationflags':0},
    {'argv':['relative.exe']},{'argv':['not-a-file']},{'argv':[]},{'argv':[42]}])
def test_changed_binding_or_invalid_workload_refuses(fixture,change):
    with pytest.raises(ValueError):invoke(fixture,**change)
    assert fixture[1]==[]


def test_pinned_launcher_change_refuses_without_fallback(fixture):
    fixture[2].write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):invoke(fixture)
    assert fixture[1]==[]


def test_registry_profile_and_missing_launcher_refuse(fixture):
    factory,calls,exe,workload,policy,env,root=fixture
    digest=hashlib.sha256(exe.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='registry'):
        module.CodexSandboxProfile(MxcPolicy(('C:/inputs',),()),executable=str(exe),cwd=str(root),
            environment=env,expected_executable_sha256=digest)
    with pytest.raises(ValueError,match='executable'):
        module.CodexSandboxProfile(policy,executable=str(root/'missing.exe'),cwd=str(root),
            environment=env,expected_executable_sha256=digest)
    assert calls==[]


def _link_or_skip(link,target):
    try:
        link.symlink_to(target)
    except (OSError,NotImplementedError) as error:
        pytest.skip('Owned link fixture unavailable: '+type(error).__name__)


def test_managed_link_is_pinned_to_its_canonical_target(fixture):
    _,calls,exe,workload,policy,env,root=fixture
    link=root/'managed-codex.exe'; _link_or_skip(link,exe)
    factory=module.CodexSandboxProfile(policy,executable=str(link),cwd=str(root),environment=env,
        expected_executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    changed=(factory,calls,exe,workload,policy,env,root)
    invoke(changed)
    argv=calls[0][0]
    assert argv[0]==str(exe.resolve())
    assert json.loads(argv[5])['codexExecutable']==str(exe.resolve())


def test_retargeted_managed_link_refuses_without_dispatch(fixture):
    _,calls,exe,workload,policy,env,root=fixture
    other=root/'other-codex.exe'; other.write_bytes(exe.read_bytes())
    link=root/'managed-codex.exe'; _link_or_skip(link,exe)
    factory=module.CodexSandboxProfile(policy,executable=str(link),cwd=str(root),environment=env,
        expected_executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    link.unlink(); link.symlink_to(other)
    with pytest.raises(ValueError,match='changed'):
        invoke((factory,calls,exe,workload,policy,env,root))
    assert calls==[]
