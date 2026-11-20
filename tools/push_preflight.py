"""What a push would publish, and whether it may: every author and committer email of the unpublished commits must
be a GitHub noreply address (or one a project allows). Published commits cannot be taken back from a public
repository (ADR-072), so this refuses before the push; it never rewrites anything.

The pre-push hook runs this file by path with a bare Python, and `relay/ci_watch.py` imports it: standard library
only, no import from `relay/`.
  push_preflight.py REMOTE SHA [--repo DIR]   exit 1 and one line per offending commit on stderr
"""
import argparse
import re
import subprocess
import sys

NOREPLY = re.compile(r'[^@\s]+@users\.noreply\.github\.com|noreply@github\.com', re.IGNORECASE)
FIX = ('Set `git config user.email` to your GitHub noreply address and recreate these commits before pushing '
       '(they are not published yet).')


def _git(git, repo_dir, *args):
    return subprocess.run([git, '-C', str(repo_dir), *args], capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=300, check=True).stdout


def unpublished(repo_dir, tip, remote='origin', git='git'):
    """The commits reachable from `tip` that no branch of `remote` has (all of them without a remote-tracking ref:
    all of it would be published)."""
    out = _git(git, repo_dir, 'log', '--format=%H%x00%ae%x00%ce', tip, '--not', f'--remotes={remote}')
    return [dict(zip(('sha', 'author', 'committer'), line.split('\0'))) for line in out.splitlines() if line]


def unsigned(repo_dir, tip, remote='origin', git='git'):
    """The unpublished commits that carry no signature. Read from the commit's own headers: `%G?` would run gpg or
    ssh on every signed commit and calls an ssh signature it cannot verify unsigned."""
    records = _git(git, repo_dir, 'rev-list', '--header', tip, '--not', f'--remotes={remote}').split('\0')
    return [record.split('\n', 1)[0] for record in records if record.strip()
            and not any(line.startswith(('gpgsig ', 'gpgsig-sha256 '))
                        for line in record.partition('\n\n')[0].splitlines())]


def identity_refusals(commits, allow=()):
    allowed = {address.lower() for address in allow}
    return [f"commit {commit['sha'][:8]}: {role} email {commit[role]} is not a GitHub noreply address"
            for commit in commits for role in ('author', 'committer')
            if not NOREPLY.fullmatch(commit[role]) and commit[role].lower() not in allowed]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('remote')
    parser.add_argument('sha')
    parser.add_argument('--repo', default='.')
    args = parser.parse_args(argv)
    refusals = identity_refusals(unpublished(args.repo, args.sha, args.remote))
    for refusal in refusals:
        print(refusal, file=sys.stderr)
    if refusals:
        print(FIX, file=sys.stderr)
    return 1 if refusals else 0


if __name__ == '__main__':
    raise SystemExit(main())
