"""Native controls use copied runtimes and fresh fixtures, never operator data."""
import json
import os
from pathlib import Path
import socket
import sys
from uuid import uuid4
import pytest
from compute_plane.confined_workspace import ConfinedWorkspace, _names
from compute_plane.managed_process import run_process


@pytest.fixture(scope='module')
def python_runtime():
    if os.name != 'nt': pytest.skip('Windows physical profile qualification')
    base = Path(sys.base_prefix)
    files = {}
    for name in ('python.exe','python3.dll',f'python{sys.version_info.major}{sys.version_info.minor}.dll',
                 'vcruntime140.dll','vcruntime140_1.dll'):
        path = base/name
        if path.is_file(): files[name] = path.read_bytes()
    for path in (base/'Lib').glob('*.py'):
        files['Lib/'+path.name] = path.read_bytes()
    for folder in ('encodings','collections','json','re','importlib','urllib'):
        for path in (base/'Lib'/folder).rglob('*.py'):
            files['Lib/'+path.relative_to(base/'Lib').as_posix()] = path.read_bytes()
    for name in ('_socket.pyd','select.pyd','_json.pyd','unicodedata.pyd'):
        path = base/'DLLs'/name
        if path.is_file(): files['DLLs/'+name] = path.read_bytes()
    return files


@pytest.mark.skipif(os.name != 'nt', reason='Windows native confinement')
def test_python_positive_and_outside_file_registry_network_denials(tmp_path, python_runtime, monkeypatch):
    import winreg
    canary = tmp_path/'host-key-canary.txt'; canary.write_text('fake host signing key')
    outside_write = tmp_path/'outside-write.txt'
    registry = 'Software\\TareTcp04Fixture-'+uuid4().hex
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, registry) as key:
        winreg.SetValueEx(key,'canary',0,winreg.REG_SZ,'fake registry secret')
    monkeypatch.setenv('TARE_FAKE_PRIVATE_CREDENTIAL','must not inherit')
    script = b'''import json, pathlib, socket, sys, os, winreg
result={"inside":pathlib.Path("input.txt").read_text(),"credential":os.getenv("TARE_FAKE_PRIVATE_CREDENTIAL")}
for name,path,write in [("outside_read",sys.argv[1],False),("outside_write",sys.argv[2],True),("input_write","input.txt",True)]:
    try:
        if write:pathlib.Path(path).write_text("escape")
        else:pathlib.Path(path).read_text()
        result[name]="ESCAPE"
    except OSError as exc:result[name]=type(exc).__name__
try:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,sys.argv[4]) as key:winreg.QueryValueEx(key,"canary")
    result["registry"]="ESCAPE"
except OSError as exc:result["registry"]=type(exc).__name__
try:
    socket.create_connection(("127.0.0.1",int(sys.argv[3])),timeout=1).close()
    result["network"]="ESCAPE"
except OSError as exc:result["network"]={"type":type(exc).__name__,"errno":exc.errno,"winerror":getattr(exc,"winerror",None)}
pathlib.Path(os.environ["TMP"],"result.txt").write_text("allowed scratch output")
print(json.dumps(result),flush=True)
'''
    listener = socket.socket(); listener.bind(('127.0.0.1',0)); listener.listen(4)
    try:
        port = listener.getsockname()[1]
        with socket.create_connection(('127.0.0.1',port),timeout=1): pass
        connection,_ = listener.accept(); connection.close()
        with ConfinedWorkspace(tmp_path/('tare-confined.'+uuid4().hex), python_runtime,
                               {'main.py':script,'input.txt':b'allowed input'}, system_registry_read=True) as profile:
            observed = run_process([str(profile.runtime_root/'python.exe'),'-I','-S','-B','main.py',
                                   str(canary),str(outside_write),str(port),registry],
                cwd=str(profile.input_root),env=profile.environment,dest=tmp_path/'stdio',timeout=5,process_factory=profile)
            assert observed['cleanup_verified'] and observed['exit_code'] == 0
            assert observed['resource_limits'] == [4,256*1024**2,512*1024**2]
            assert observed['confinement']['capability_names'] == ['registryRead']
            artifacts = profile.collect(observed)
            report = json.loads(artifacts['stdout.txt'])
            assert report['inside'] == 'allowed input' and report['credential'] is None
            assert all(report[k] == 'PermissionError' for k in ('outside_read','outside_write','input_write'))
            assert report['registry'] in ('PermissionError','FileNotFoundError')
            assert report['network']['winerror'] == 10013
            assert any(data == b'allowed scratch output' for name,data in artifacts.items() if name.startswith('scratch/'))
            assert canary.read_text() == 'fake host signing key' and not outside_write.exists()
        listener.settimeout(.1)
        with pytest.raises(socket.timeout): listener.accept()
    finally:
        listener.close(); winreg.DeleteKey(winreg.HKEY_CURRENT_USER,registry)


@pytest.mark.skipif(os.name != 'nt', reason='Windows native confinement')
@pytest.mark.parametrize('stop',['timeout','cancel'])
def test_stopped_process_is_cleaned_and_cannot_be_collected_as_success(tmp_path, stop):
    import time
    runtime = {'cmd.exe':(Path(os.environ['SystemRoot'])/'System32/cmd.exe').read_bytes()}
    with ConfinedWorkspace(tmp_path/('tare-confined.'+uuid4().hex),runtime,{}) as profile:
        started = time.monotonic()
        observed = run_process([str(profile.runtime_root/'cmd.exe'),'/d','/c','for /l %i in (1,0,2) do @rem'],
            cwd=str(profile.input_root),env=profile.environment,dest=tmp_path/'stdio',timeout=.15 if stop=='timeout' else 2,
            cancelled=lambda: stop=='cancel' and time.monotonic()-started>.15,process_factory=profile)
        assert observed['cleanup_verified'] and observed['timeout' if stop=='timeout' else 'cancelled']
        with pytest.raises(ValueError,match='receipt_invalid'): profile.collect(observed)


@pytest.mark.parametrize('files',[{'../x':b'x'},{'a':b'x','A/b':b'y'},{'CONIN$':b'x'},{'x'*241:b'x'},{'x':bytearray(b'x')}])
def test_unsafe_package_is_rejected_before_any_provisioning(files):
    with pytest.raises(ValueError): _names(files,64,1024)


@pytest.mark.skipif(os.name != 'nt', reason='Windows native confinement')
def test_changed_input_and_hardlinked_output_are_refused(tmp_path):
    runtime = {'cmd.exe':(Path(os.environ['SystemRoot'])/'System32/cmd.exe').read_bytes()}
    with ConfinedWorkspace(tmp_path/('tare-confined.'+uuid4().hex),runtime,{'input.txt':b'original'}) as profile:
        (profile.input_root/'input.txt').write_bytes(b'tampered')
        with pytest.raises(ValueError,match='manifest'): profile.validate()
        (profile.input_root/'input.txt').write_bytes(b'original')
        extra=profile.input_root/'unexpected';extra.mkdir()
        with pytest.raises(ValueError,match='extra_directory'):profile.validate()
        extra.rmdir()
        observed = run_process([str(profile.runtime_root/'cmd.exe'),'/d','/c','echo done'],
            cwd=str(profile.input_root),env=profile.environment,dest=tmp_path/'stdio',timeout=2,process_factory=profile)
        assert observed['cleanup_verified'] and observed['exit_code'] == 0
        canary=tmp_path/'host-only.txt';canary.write_bytes(b'host-only bytes')
        os.link(canary,profile.scratch_root/'alias.txt')
        with pytest.raises(ValueError,match='resource'): profile.collect(observed)


@pytest.mark.skipif(os.name != 'nt', reason='Windows cleanup control')
def test_failed_acl_cleanup_remains_visible_and_retryable(tmp_path,monkeypatch):
    profile=ConfinedWorkspace(tmp_path/('tare-confined.'+uuid4().hex),{}, {})
    original=profile._acl
    failed=[False]
    def partial(exe,folder,command,value):
        if command=='/remove:g' and not failed[0]:
            failed[0]=True;raise OSError('fixture cleanup failure')
        return original(exe,folder,command,value)
    monkeypatch.setattr(profile,'_acl',partial)
    try:
        with pytest.raises(OSError,match='cleanup_unconfirmed'):profile.close()
        assert not profile._closed and len(profile._granted)==1
        profile.close();assert profile._closed
    finally:
        monkeypatch.setattr(profile,'_acl',original);profile.close()


@pytest.mark.skipif(os.name != 'nt', reason='Windows output race control')
def test_resource_monitor_does_not_snapshot_bytes_while_guest_writes(tmp_path,monkeypatch):
    with ConfinedWorkspace(tmp_path/('tare-confined.'+uuid4().hex),{}, {}) as profile:
        (profile.scratch_root/'active.txt').write_bytes(b'output in progress')
        def unavailable(*args,**kwargs):raise OSError('active output cannot be snapshotted yet')
        monkeypatch.setattr('compute_plane.confined_workspace._snapshot',unavailable)
        assert profile.resource_check() is True


@pytest.mark.skipif(os.name == 'nt', reason='Unsupported-platform control')
def test_unavailable_platform_never_creates_a_workspace(tmp_path):
    root = tmp_path/('tare-confined.'+uuid4().hex)
    with pytest.raises(OSError,match='platform_unavailable'): ConfinedWorkspace(root,{}, {})
    assert not root.exists()
