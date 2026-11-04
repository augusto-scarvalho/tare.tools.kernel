"""Bounded read-only audit sources, independent of model and host drivers."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from .managed_process import host_executable
import tempfile

from .workspace import isolated_child_environment


def audit_git(root, *args, limit=2 * 1024 * 1024):
    """Read-only Git plumbing for explicitly selected audit snapshots."""
    with tempfile.TemporaryFile() as output:
        result = subprocess.run([host_executable("git", required=True), "-c", "core.fsmonitor=false", "-C", str(root), *args], stdout=output,
            stderr=subprocess.PIPE, timeout=15, env=isolated_child_environment(),
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        if result.returncode:
            raise ValueError("audit_git_failed")
        output.seek(0)
        raw = output.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("audit_git_output_over_budget")
    return raw


def audit_changes(root, base, target):
    """Renames intentionally appear as deletion+addition: neither side is lost."""
    root = Path(root).resolve()
    if Path(audit_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("explicit_repository_root_required")
    if not re.fullmatch("[a-f0-9]{40}", base) or not (
        target in {"WORKTREE", "INDEX"} or re.fullmatch("[a-f0-9]{40}", target)
    ):
        raise ValueError("explicit_snapshot_required")
    args = ["diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--name-only", "-z", base]
    if target == "INDEX":
        args.append("--cached")
    elif target != "WORKTREE":
        args.append(target)
    names = set(audit_git(root, *args, "--").decode("utf-8").split("\0")) - {""}
    if target == "WORKTREE":
        names.update(audit_git(root, "ls-files", "--others", "--exclude-standard", "-z")
                     .decode("utf-8").split("\0"))
    return sorted(names - {""})


def audit_source(root, revision, name, limit=24000):
    """Return exact bytes or None for absence, refusing symlinks/submodules."""
    from pathlib import PurePosixPath
    root = Path(root).resolve()
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or ":" in name or ".git" in path.parts):
        raise ValueError("invalid_audit_source_path")
    if revision == "WORKTREE":
        source = root / name
        if not source.resolve().is_relative_to(root) or any(
            parent.is_symlink() for parent in [source, *source.parents] if parent != root
        ):
            raise ValueError("audit_source_symlink")
        if not source.exists():
            return None
        if not source.is_file():
            raise ValueError("audit_source_not_file")
        with source.open("rb") as stream:
            raw = stream.read(limit + 1)
    else:
        if revision != "INDEX" and not re.fullmatch("[a-f0-9]{40}", revision):
            raise ValueError("explicit_snapshot_required")
        if revision == "INDEX":
            rows = audit_git(root, "ls-files", "--stage", "-z", "--", ":(literal)" + name)
        else:  # --long carries the blob size, sparing a `cat-file -s` process
            rows = audit_git(root, "ls-tree", "--long", "-z", revision, "--", ":(literal)" + name)
        if not rows:
            return None
        records = rows.rstrip(b"\0").split(b"\0")
        if len(records) != 1:
            raise ValueError("ambiguous_audit_source")
        metadata, actual = records[0].split(b"\t", 1)
        fields = metadata.split()
        if actual.decode("utf-8") != name or fields[0] not in {b"100644", b"100755"}:
            raise ValueError("audit_source_not_regular_blob")
        if revision == "INDEX" and fields[2] != b"0":
            raise ValueError("unmerged_audit_source")
        blob = fields[1] if revision == "INDEX" else fields[2]
        size = audit_git(root, "cat-file", "-s", blob.decode("ascii"), limit=128) if revision == "INDEX" else fields[3]
        if int(size) > limit:
            raise ValueError("audit_source_over_budget")
        raw = audit_git(root, "cat-file", "blob", blob.decode("ascii"), limit=limit)
    if len(raw) > limit:
        raise ValueError("audit_source_over_budget")
    return raw


def repository_refs(repositories):
    return [{'id': repo['id'], 'root': str(Path(repo['root']).resolve()),
             'base': repo.get('base'), 'reference': repo['target'],
             'head': audit_git(repo['root'], 'rev-parse', 'HEAD').decode().strip()}
            for repo in repositories]


def verify_repositories(refs):
    for repo in refs:
        if audit_git(repo['root'], 'rev-parse', 'HEAD').decode().strip() != repo['head']:
            raise ValueError('audit_head_changed')


def evidence_source(repo, path, start_line, end_line, revision='WORKTREE', *, obligation_ids=()):
    """Preserve existing X IDs and exact source text without writing a receipt."""
    if (type(start_line) is not int or type(end_line) is not int
            or not 1 <= start_line <= end_line or end_line - start_line >= 200):
        raise ValueError('request_precise_range_up_to_200_lines')
    raw = audit_source(repo['root'], revision, path, limit=2 * 1024 * 1024)
    if raw is None:
        raise ValueError('repository_file_missing')
    digest = hashlib.sha256(raw).hexdigest()
    lines = raw.decode('utf-8').splitlines(keepends=True)
    if start_line > len(lines):
        raise ValueError('range_outside_file')
    end_line = min(end_line, len(lines))
    identity = json.dumps([repo['id'], path, revision, start_line, end_line, digest],
                          ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return {'id': 'X-' + hashlib.sha256(identity).hexdigest()[:24], 'kind': 'source',
            'repository': repo['id'], 'path': path, 'revision': revision, 'start_line': start_line,
            'text': ''.join(lines[start_line - 1:end_line]),
            'obligation_ids': list(obligation_ids), 'sha256': digest}
