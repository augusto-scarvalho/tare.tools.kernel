import os

import pytest

from compute_plane.managed_process import host_executable


def tool(directory, name):
    path = directory/(name+'.exe' if os.name == 'nt' else name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'fixture executable; never run')
    path.chmod(0o755)
    return path


def test_passive_lookup_skips_cwd_and_relative_path_entries(tmp_path, monkeypatch):
    project, host = tmp_path/'project', tmp_path/'host tools'
    tool(project, 'git'); tool(project/'relative', 'git')
    expected = tool(host, 'git')
    monkeypatch.chdir(project)
    monkeypatch.setenv('PATH', os.pathsep.join(['', '.', 'relative', str(project), str(host)]))
    assert host_executable('git') == str(expected)
    assert host_executable('git', required=True) == str(expected)


def test_cwd_tool_is_not_a_fallback_when_the_host_lacks_it(tmp_path, monkeypatch):
    tool(tmp_path, 'rg')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PATH', os.pathsep.join(['.', str(tmp_path)]))
    assert host_executable('rg') is None
    with pytest.raises(FileNotFoundError, match='absolute PATH'):
        host_executable('rg', required=True)
    with pytest.raises(ValueError, match='basename'):
        host_executable('../rg')


def test_search_uses_host_binary_without_executing_project_programs(tmp_path, monkeypatch):
    from compute_plane.native_search import search
    project, host = tmp_path/'project', tmp_path/'host'
    tool(project, 'rg')
    expected = tool(host, 'rg')
    monkeypatch.chdir(project)
    monkeypatch.setenv('PATH', os.pathsep.join(['.', str(host)]))
    calls = []
    class Tools:
        roots = {'repo': project}
        def path(self, *a, **k): return project
        def command(self, argv, *a, **k):
            calls.append(argv)
            return {'returncode': 0, 'output': '', 'output_truncated': False,
                    'process': {'cleanup_verified': True, 'wall_ms': 0}}
    result = search(Tools(), repository='repo', query='needle')
    assert result['status'] != 'ERROR'
    assert len(calls) == 1 and calls[0][0] == str(expected)


@pytest.mark.parametrize('operation', ['workspace', 'candidate', 'audit'])
def test_git_inspection_ports_cannot_select_a_current_directory_program(tmp_path, monkeypatch, operation):
    import subprocess
    from compute_plane import audit_sources, candidate_git, workspace
    project, host = tmp_path/'project', tmp_path/'host'
    tool(project, 'git')
    expected = tool(host, 'git')
    monkeypatch.chdir(project)
    monkeypatch.setenv('PATH', os.pathsep.join(['.', str(host)]))
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, '' if kwargs.get('text') else b'', '')
    if operation == 'workspace':
        workspace._run_git(project, 'rev-parse', 'HEAD', runner=run)
    elif operation == 'candidate':
        candidate_git._git_bytes(project, 'diff', runner=run)
    else:
        monkeypatch.setattr(audit_sources.subprocess, 'run', run)
        audit_sources.audit_git(project, 'rev-parse', 'HEAD')
    assert len(calls) == 1 and calls[0][0] == str(expected)
