"""Offline ABI checks using the independent FlatBuffers reader."""

from dataclasses import FrozenInstanceError
import hashlib
import struct

from flatbuffers import number_types
from flatbuffers.table import Table
import pytest

from compute_plane.windows_mxc import MxcPolicy, build_psec_spec, policy_sha256


def _root(data):
    assert isinstance(data, bytes)
    assert data[4:8] == b'PSEC'
    return Table(data, struct.unpack_from('<I', data, 0)[0])


def _offset(table, slot):
    return table.Offset(4 + 2 * slot)


def _child(table, slot):
    offset = _offset(table, slot)
    assert offset, f'missing table in slot {slot}'
    return Table(table.Bytes, table.Indirect(table.Pos + offset))


def _strings(table, slot):
    offset = _offset(table, slot)
    assert offset, f'missing vector in slot {slot}'
    start = table.Vector(offset)
    return [table.String(start + 4 * i).decode('utf-8')
            for i in range(table.VectorLen(offset))]


def test_root_version_and_distinct_filesystem_slots():
    policy = MxcPolicy(
        ['C:/Inputs/./One/', 'c:/Inputs/Two', 'C:/Inputs/Two'],
        ['D:/Output/B', 'D:/Output/A'],
        ['//Server/Share/Secret', 'C:/Inputs/Blocked'],
    )
    root = _root(build_psec_spec(policy))
    version = _offset(root, 0)
    assert version
    assert struct.unpack_from('<HH', root.Bytes, root.Pos + version) == (1, 0)
    assert not _offset(root, 1)  # No capabilities by default.
    win32k = _offset(root, 2)
    assert not win32k or not root.Get(number_types.BoolFlags, root.Pos + win32k)
    restrictions = _offset(root, 3)
    assert restrictions
    assert root.Get(number_types.Uint64Flags, root.Pos + restrictions) == 0xFE
    assert _strings(root, 4) == ['D:\\Output\\B', 'D:\\Output\\A']
    assert _strings(root, 5) == ['C:\\Inputs\\One', 'c:\\Inputs\\Two', 'C:\\Inputs\\Two']
    assert _strings(root, 6) == ['\\\\Server\\Share\\Secret', 'C:\\Inputs\\Blocked']


def test_network_has_only_deny_default_egress():
    root = _root(build_psec_spec(MxcPolicy(['C:/Input'], [])))
    network = _child(root, 7)
    assert not _offset(network, 0)  # No proxy.
    assert not _offset(network, 2)  # No peer.
    egress = _child(network, 1)
    action = _offset(egress, 0)
    assert not action or egress.Get(number_types.Int8Flags, egress.Pos + action) == 0
    assert not _offset(egress, 1)  # No allow rules, including loopback.
    assert not _offset(egress, 2)
    assert _strings(root, 4) == []
    assert _strings(root, 6) == []


def test_registry_is_exact_opt_in_and_digest_is_finished_bytes():
    ordinary = MxcPolicy([], ['C:/Output'])
    policy = MxcPolicy([], ['C:/Output'], registry_read=True)
    data = build_psec_spec(policy)
    root = _root(data)
    offset = _offset(root, 1)
    assert offset
    assert root.String(root.Pos + offset) == b'registryRead'
    assert data == build_psec_spec(policy)
    assert data == build_psec_spec(MxcPolicy((), ('C:/Output',), registry_read=True))
    assert policy_sha256(policy) == hashlib.sha256(data).hexdigest()
    assert len(policy_sha256(policy)) == 64
    assert data != build_psec_spec(ordinary)
    assert not _offset(_root(build_psec_spec(ordinary)), 1)


def test_enumeration_only_uses_v1_1_without_promoting_volume_to_content_read():
    policy=MxcPolicy(['C:/Input'],['C:/Output'],enumerate_only=['C:/'])
    root=_root(build_psec_spec(policy))
    assert struct.unpack_from('<HH',root.Bytes,root.Pos+_offset(root,0))==(1,1)
    assert _strings(root,8)==['C:\\']
    assert _strings(root,4)==['C:\\Output'] and _strings(root,5)==['C:\\Input']
    assert policy_sha256(policy)!=policy_sha256(MxcPolicy(['C:/Input'],['C:/Output']))


def test_caller_sequences_are_copied_and_policy_is_frozen():
    read_only, read_write, denied = ['C:/Input'], ['D:/Output'], ['C:/Secret']
    policy = MxcPolicy(read_only, read_write, denied)
    before = build_psec_spec(policy)
    read_only.append('C:/Other')
    read_write.clear()
    denied[0] = 'C:/Different'
    assert policy.read_only == ('C:\\Input',)
    assert policy.read_write == ('D:\\Output',)
    assert policy.denied == ('C:\\Secret',)
    assert build_psec_spec(policy) == before
    with pytest.raises(FrozenInstanceError):
        policy.registry_read = True


@pytest.mark.parametrize('path', [
    '', 'relative', 'C:relative', '/rooted', '\\rooted',
    'C:/Input/../Other', 'C:\\Input\\..\\Other',
    '//server/share/../Other', 'C:/Input/*', 'C:/Input/?',
    'C:/', 'C:\\', 'C:/./', '//server/share', '//server/share/',
    'C:/Input\x00suffix', '1:/Input', '\\\\server',
])
def test_invalid_paths_rejected_in_every_group(path):
    for group in ('read_only', 'read_write', 'denied'):
        kwargs = {'read_only': ['C:/Input'], 'read_write': []}
        kwargs[group] = [path]
        with pytest.raises(ValueError):
            MxcPolicy(**kwargs)


@pytest.mark.parametrize('group', ['read_only', 'read_write', 'denied'])
def test_duplicates_and_overlong_lists(group):
    kwargs = {'read_only': ['C:/Input'], 'read_write': []}
    kwargs[group] = ['D:/Output/./', 'D:\\Output']
    with pytest.raises(ValueError, match='duplicate'):
        MxcPolicy(**kwargs)
    kwargs[group] = [f'D:/Output/{i}' for i in range(33)]
    with pytest.raises(ValueError, match='32'):
        MxcPolicy(**kwargs)
    kwargs[group] = kwargs[group][:32]
    assert len(getattr(MxcPolicy(**kwargs), group)) == 32


@pytest.mark.parametrize('value', [0, 1, None, 'false', []])
def test_registry_requires_real_bool(value):
    with pytest.raises(TypeError, match='bool'):
        MxcPolicy(['C:/Input'], [], registry_read=value)


def test_nonempty_scope_and_string_types():
    with pytest.raises(ValueError, match='nonempty'):
        MxcPolicy([], [], ['C:/Secret'])
    with pytest.raises(TypeError):
        MxcPolicy('C:/Input', [])
    for group in ('read_only', 'read_write', 'denied'):
        kwargs = {'read_only': ['C:/Input'], 'read_write': []}
        kwargs[group] = [123]
        with pytest.raises(TypeError, match='strings'):
            MxcPolicy(**kwargs)


@pytest.mark.parametrize('value', [None, {}, 'policy'])
def test_codec_functions_refuse_wrong_policy_type(value):
    for function in (build_psec_spec, policy_sha256):
        with pytest.raises(TypeError, match='MxcPolicy'):
            function(value)


@pytest.fixture
def native_api(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    import compute_plane.windows_mxc as module
    calls = []
    state = dict(query_hr=0, support=15, create_hr=0, handle=81, version_hr=0, available=1, minor=1)

    class Function:
        def __init__(self, callback): self.callback = callback
        def __call__(self, *args): return self.callback(*args)

    def query(pointer):
        calls.append('query')
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint64))[0] = state['support']
        return state['query_hr']

    def create(blob, length, flags, pointer):
        calls.append(('create', ctypes.string_at(blob, length), flags))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = state['handle']
        return state['create_hr']

    def version(major,available,minor):
        assert major==1
        ctypes.cast(available,ctypes.POINTER(ctypes.c_ubyte))[0]=state['available']
        ctypes.cast(minor,ctypes.POINTER(ctypes.c_uint32))[0]=state['minor']
        return state['version_hr']

    dll = SimpleNamespace(CreateProcessSecurityEnvironment=Function(create),
        QueryProcessSecurityEnvironmentSupport=Function(query),
        IsProcessSecurityEnvironmentVersionSupported=Function(version),
        CloseProcessSecurityEnvironment=Function(lambda handle: calls.append(('close', handle))))
    monkeypatch.setattr(module, '_load_mxc_library', lambda: dll)
    return module, dll, state, calls


@pytest.mark.parametrize('failure',['none','flag','version','export','query'])
def test_enumeration_requires_native_flag_and_version_without_read_fallback(native_api,failure):
    module,dll,state,calls=native_api
    if failure=='flag':state['support'] &= ~4
    if failure=='version':state['minor']=0
    if failure=='export':del dll.IsProcessSecurityEnvironmentVersionSupported
    if failure=='query':state['version_hr']=-1
    policy=MxcPolicy(['C:/input'],[],enumerate_only=['C:/'])
    if failure=='none':
        with module.MxcEnvironment(policy):pass
        assert [x for x in calls if isinstance(x,tuple) and x[0]=='create']==[('create',build_psec_spec(policy),0)]
    else:
        with pytest.raises((OSError,AttributeError)):module.MxcEnvironment(policy)
        assert calls==['query']


def test_native_context_uses_exact_bytes_and_closes_once(native_api):
    module, dll, state, calls = native_api
    policy = MxcPolicy(('C:/input',), (), ('C:/input/private',))
    with module.MxcEnvironment(policy) as context:
        assert context.handle == 81 and context.support_flags == 15
        assert calls == ['query', ('create', build_psec_spec(policy), 0)]
        import ctypes
        assert dll.CreateProcessSecurityEnvironment.restype is ctypes.c_int32
    context.close()
    assert calls[-1] == ('close', 81) and calls.count(('close', 81)) == 1
    with pytest.raises(RuntimeError, match='closed'): context.handle


@pytest.mark.parametrize('phase', ['query', 'deny', 'create', 'null', 'missing'])
def test_native_context_refuses_failure_without_launch_or_leak(native_api, phase):
    module, dll, state, calls = native_api
    if phase == 'query': state['query_hr'] = -2147467259
    if phase == 'deny': state['support'] = 0
    if phase == 'create': state['create_hr'] = -2147467259
    if phase == 'null': state['handle'] = 0
    if phase == 'missing': del dll.CreateProcessSecurityEnvironment
    with pytest.raises((OSError, AttributeError)):
        module.MxcEnvironment(MxcPolicy(('C:/input',), (), ('C:/input/private',)))
    creates = [item for item in calls if isinstance(item, tuple) and item[0] == 'create']
    closes = [item for item in calls if isinstance(item, tuple) and item[0] == 'close']
    assert len(creates) == (1 if phase in ('create', 'null') else 0)
    assert closes == ([('close', 81)] if phase == 'create' else [])


def test_reparse_attribute_rejected_without_python312_method():
    from types import SimpleNamespace
    from compute_plane.windows_mxc import _path_alias
    class PathFixture:
        def is_symlink(self): return False
        def lstat(self): return SimpleNamespace(st_file_attributes=0x400)
    assert _path_alias(PathFixture()) is True


@pytest.fixture
def local_mxc(tmp_path):
    import os
    if os.name != 'nt':
        pytest.skip('local MXC path validation requires Windows')
    from compute_plane.windows_mxc import WindowsMxcProfile
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    data = tmp_path / 'data'
    data.mkdir()

    def profile(*paths):
        return WindowsMxcProfile(
            MxcPolicy((str(cwd),), tuple(str(p) for p in paths)),
            cwd=str(cwd), environment={})

    return WindowsMxcProfile, cwd, data, profile


@pytest.mark.parametrize('existing', [False, True])
def test_file_scope_is_exact_and_never_grants_ancestors(local_mxc, existing):
    cls, cwd, data, profile = local_mxc
    target = data / 'nested' / 'tracked.txt'
    if existing:
        target.parent.mkdir()
        target.write_text('tracked', encoding='utf-8')
    factory = profile(target)
    assert factory._admitted(target)
    assert not factory._admitted(target.parent)
    assert not factory._admitted(target.with_name('sibling.txt'))
    assert not factory._admitted(target / 'child')
    factory._revalidate()
    if not existing:
        target.parent.mkdir()
        target.write_text('created', encoding='utf-8')
        factory._revalidate()
        assert factory._admitted(target)


@pytest.mark.parametrize('existing', [False, True])
def test_cwd_still_requires_real_directory(local_mxc, existing):
    cls, cwd, data, profile = local_mxc
    target = data / 'file.txt'
    if existing:
        target.write_text('file', encoding='utf-8')
    with pytest.raises(ValueError, match='mxc_directory_unavailable'):
        cls(MxcPolicy((str(data),), ()), cwd=str(target), environment={})


def test_existing_directory_scopes_still_allow_descendants(local_mxc):
    cls, cwd, data, profile = local_mxc
    factory = profile(data)
    assert factory._admitted(data)
    assert factory._admitted(data / 'nested' / 'file.txt')
    assert not factory._admitted(data.with_name('sibling'))
    factory._revalidate()


@pytest.mark.parametrize('existing', [False, True])
def test_file_to_directory_scope_widening_refused(local_mxc, existing):
    cls, cwd, data, profile = local_mxc
    target = data / 'tracked'
    if existing:
        target.write_text('file', encoding='utf-8')
    factory = profile(target)
    if existing:
        target.unlink()
    target.mkdir()
    with pytest.raises(ValueError, match='mxc_scope_changed'):
        factory._revalidate()


def test_scope_hardlinks_refused_at_creation_and_revalidation(local_mxc):
    import os
    cls, cwd, data, profile = local_mxc
    target = data / 'tracked.txt'
    target.write_text('file', encoding='utf-8')
    factory = profile(target)
    os.link(target, data / 'other.txt')
    with pytest.raises(ValueError, match='mxc_scope_hardlink_refused'):
        profile(target)
    with pytest.raises(ValueError, match='mxc_scope_hardlink_refused'):
        factory._revalidate()


@pytest.mark.parametrize('location', ['root', 'ancestor', 'cwd'])
def test_reparse_scopes_and_cwd_refused_at_creation_and_revalidation(
        local_mxc, monkeypatch, location):
    import compute_plane.windows_mxc as module
    cls, cwd, data, profile = local_mxc
    target = data / 'tracked.txt'
    target.write_text('file', encoding='utf-8')
    factory = profile(target)
    alias = {'root': target, 'ancestor': data, 'cwd': cwd}[location]
    original = module._path_alias
    monkeypatch.setattr(module, '_path_alias',
                        lambda path: path == alias or original(path))
    with pytest.raises(ValueError, match='alias_refused'):
        profile(target)
    with pytest.raises(ValueError, match='alias_refused'):
        factory._revalidate()


def test_dangling_symlink_scope_refused(local_mxc):
    cls, cwd, data, profile = local_mxc
    alias = data / 'alias.txt'
    try:
        alias.symlink_to(data / 'missing.txt')
    except OSError:
        pytest.skip('Windows symlink privilege unavailable')
    with pytest.raises(ValueError, match='mxc_scope_alias_refused'):
        profile(alias)


def test_launch_revalidates_before_native_api(local_mxc, monkeypatch):
    import ctypes
    cls, cwd, data, profile = local_mxc
    target = data / 'tracked.txt'
    factory = profile(target)
    target.mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail('native API reached with changed scope')

    monkeypatch.setattr(ctypes, 'WinDLL', forbidden)
    with pytest.raises(ValueError, match='mxc_scope_changed'):
        factory([str(cwd / 'tool.exe')], cwd=str(cwd), env={},
                stdin=None, stdout=None, stderr=None, creationflags=4)
def test_native_tool_tree_budget_keeps_the_memory_ceilings():
    from compute_plane.windows_mxc import WindowsMxcProfile
    from compute_plane.codex_sandbox import CodexSandboxProfile
    assert WindowsMxcProfile.job_limits == (16, 256 * 1024**2, 512 * 1024**2)
    assert CodexSandboxProfile.job_limits == WindowsMxcProfile.job_limits
