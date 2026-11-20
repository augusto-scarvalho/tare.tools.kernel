"""The pre-push hook checks (and watches) what a push updates, whatever the local ref is called."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SH = shutil.which('sh') or next(
    (str(p) for p in [Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/usr/bin/sh.exe'] if p.is_file()),
    None,
)
PERSONAL = 't@example.org'
NOREPLY = '21094776+me@users.noreply.github.com'


def _load_preflight():
    path = ROOT / 'tools' / 'push_preflight.py'
    spec = importlib.util.spec_from_file_location('push_preflight', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args, email='t@example.invalid'):
    return subprocess.run(
        ['git', '-c', 'user.name=T', '-c', f'user.email={email}', '-c', 'core.hooksPath=/dev/null', *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.skipif(SH is None, reason='no POSIX shell to run the hook')
@pytest.mark.parametrize('local_ref', ['refs/heads/main', 'HEAD'])  # HEAD: `git push origin HEAD:main`, detached
def test_the_changelog_guard_checks_a_push_from_a_branch_or_a_detached_head(tmp_path, local_ref):
    for path in ('.githooks/pre-push', 'tools/changelog_guard.py', 'CHANGELOG.md', 'tools/push_preflight.py'):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, tmp_path / path)
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'base')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    (tmp_path / 'tools' / 'feature.py').write_text('VALUE = 1\n', encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'code without a changelog entry')
    head = git(tmp_path, 'rev-parse', 'HEAD')
    env = {key: value for key, value in os.environ.items() if key != 'LOCALAPPDATA'}  # no CI watcher
    env['PYTHON'] = sys.executable
    result = subprocess.run(
        [SH, '.githooks/pre-push', 'origin', 'unused-url'],
        cwd=tmp_path,
        env=env,
        input=f'{local_ref} {head} refs/heads/main {base}\n',
        capture_output=True,
        text=True,
    )
    assert 'CHANGELOG_GUARD=' in result.stdout + result.stderr, result


@pytest.mark.skipif(SH is None, reason='no POSIX shell to run the hook')
@pytest.mark.parametrize(
    'email, remote_ref, refused',
    [
        (PERSONAL, 'refs/heads/feature', True),
        (PERSONAL, 'refs/tags/v1', True),  # a tag publishes commits too
        (NOREPLY, 'refs/heads/feature', False),
        (NOREPLY, 'refs/tags/v1', False),
    ],
)
def test_the_hook_refuses_a_pushed_commit_with_a_personal_email(tmp_path, email, remote_ref, refused):
    for path in ('.githooks/pre-push', 'tools/changelog_guard.py', 'tools/push_preflight.py', 'CHANGELOG.md'):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, tmp_path / path)
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'base', email='noreply@github.com')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    changelog = (tmp_path / 'CHANGELOG.md').read_bytes()
    line_end = b'\r\n' if b'\r\n' in changelog else b'\n'
    heading = b'## Unreleased' + line_end + line_end
    assert heading in changelog
    (tmp_path / 'CHANGELOG.md').write_bytes(
        changelog.replace(heading, heading + b'- The notes file records what this hook test pushes.' + line_end, 1)
    )
    (tmp_path / 'notes.txt').write_text('x\n', encoding='utf-8')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'notes', email=email)
    head = git(tmp_path, 'rev-parse', 'HEAD')
    env = {key: value for key, value in os.environ.items() if key != 'LOCALAPPDATA'}  # no CI watcher
    env['PYTHON'] = sys.executable
    result = subprocess.run(
        [SH, '.githooks/pre-push', 'origin', 'unused-url'],
        cwd=tmp_path,
        env=env,
        input=f'refs/heads/feature {head} {remote_ref} {base}\n',
        capture_output=True,
        text=True,
    )
    if refused:
        assert result.returncode != 0 and f'commit {head[:8]}: author email {email}' in result.stderr, result
    else:
        assert result.returncode == 0 and 'noreply' not in result.stderr, result


@pytest.mark.skipif(SH is None, reason='no POSIX shell to run the hook')
def test_the_hook_passes_an_empty_line_from_a_wrapper_with_nothing_to_push(tmp_path):
    for path in ('.githooks/pre-push', 'tools/changelog_guard.py', 'tools/push_preflight.py', 'CHANGELOG.md'):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, tmp_path / path)
    git(tmp_path, 'init', '-q')
    env = {key: value for key, value in os.environ.items() if key != 'LOCALAPPDATA'}  # no CI watcher
    env['PYTHON'] = sys.executable
    result = subprocess.run(
        [SH, '.githooks/pre-push', 'origin', 'unused-url'],
        cwd=tmp_path,
        env=env,
        input='\n',
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and not result.stderr, result


def test_push_preflight_refuses_personal_author_or_committer_and_ignores_published_commits(tmp_path):
    preflight = _load_preflight()
    assert preflight.identity_refusals([{'sha': '1a2b3c4d5e6f', 'author': NOREPLY, 'committer': 'noreply@github.com'}]) == []
    assert preflight.identity_refusals([{'sha': '1a2b3c4d5e6f', 'author': PERSONAL, 'committer': NOREPLY}]) == [
        f'commit 1a2b3c4d: author email {PERSONAL} is not a GitHub noreply address'
    ]
    assert preflight.identity_refusals([{'sha': '1a2b3c4d5e6f', 'author': NOREPLY, 'committer': PERSONAL}]) == [
        f'commit 1a2b3c4d: committer email {PERSONAL} is not a GitHub noreply address'
    ]

    remote, work = tmp_path / 'remote.git', tmp_path / 'work'
    subprocess.run(['git', 'init', '-q', '--bare', '-b', 'main', str(remote)], check=True)
    subprocess.run(['git', 'clone', '-q', str(remote), str(work)], check=True, capture_output=True)
    (work / 'a.txt').write_text('1\n', encoding='utf-8')
    git(work, 'add', '.')
    git(work, 'commit', '-qm', 'published personal commit', email=PERSONAL)
    git(work, 'push', '-q', 'origin', 'HEAD:main')

    git(work, 'checkout', '-qb', 'feature')
    (work / 'b.txt').write_text('2\n', encoding='utf-8')
    git(work, 'add', '.')
    git(work, 'commit', '-qm', 'unpublished personal commit', email=PERSONAL)
    head = git(work, 'rev-parse', 'HEAD')
    script = ROOT / 'tools' / 'push_preflight.py'
    refused = subprocess.run([sys.executable, str(script), 'origin', 'HEAD', '--repo', str(work)],
                             capture_output=True, text=True)
    assert refused.returncode == 1 and head[:8] in refused.stderr and 'author email' in refused.stderr

    git(work, 'reset', '--hard', 'origin/main')
    (work / 'c.txt').write_text('3\n', encoding='utf-8')
    git(work, 'add', '.')
    git(work, 'commit', '-qm', 'unpublished noreply commit', email=NOREPLY)
    accepted = subprocess.run([sys.executable, str(script), 'origin', 'HEAD', '--repo', str(work)],
                              capture_output=True, text=True)
    assert accepted.returncode == 0 and not accepted.stderr
