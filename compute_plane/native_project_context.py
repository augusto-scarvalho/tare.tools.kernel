"""Native lookup over SpecGraph's public bounded, snapshot-only context API."""
from pathlib import Path

from .native_models import encode, object_json


class NativeProjectContext:
    def __init__(self, *, repositories, work_id, state_dirs=None, max_bytes=4096, timeout_ms=250,
                 role='implementer'):
        if role not in {'implementer', 'explorer', 'planner', 'auditor'}:
            raise ValueError('invalid_project_context_role')
        self.role = role
        if type(max_bytes) is not int or not 512 <= max_bytes <= 16384:
            raise ValueError('invalid_project_context_bytes')
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= 1000:
            raise ValueError('invalid_project_context_timeout')
        self.roots = {k: Path(v).resolve() for k, v in repositories.items()}
        self.state_dirs = {k: Path(v).resolve() for k, v in (state_dirs or {}).items()}
        if self.state_dirs.keys() - self.roots.keys():
            raise ValueError('index_repository_not_selected')
        self.work_id, self.max_bytes, self.timeout_ms = work_id, max_bytes, timeout_ms

    def lookup(self, repository, query, control, *, dirty_paths=()):
        root = self.roots[repository]
        if not isinstance(query, str) or not 1 <= len(query.encode()) <= 1024:
            raise ValueError('bounded_context_query_required')
        base = {'repository': repository, 'root': str(root), 'query': query,
                'authority': 'NONE', 'coverage': {'status': 'UNKNOWN'},
                'absence_proven': False}
        if control.should_stop():
            return base | {'status': 'CANCELLED'}
        try:
            from specgraph.hook_context import context_hint
            report = context_hint(root, role=self.role, event='pre', query=query,
                work_id=self.work_id, dirty_paths=tuple(dirty_paths)[:64],
                state_dir=self.state_dirs.get(repository), max_bytes=self.max_bytes,
                timeout_ms=max(1, int(min(self.timeout_ms, control.remaining_seconds()*1000))))
        except (ImportError, OSError) as exc:
            report = {'status': 'UNAVAILABLE', 'reason': type(exc).__name__}
        if control.should_stop():
            return base | {'status': 'CANCELLED', 'reason': 'control_stopped_during_lookup'}
        # Detach provider output; lookup never expands filesystem/write authority.
        report = object_json(encode(report))
        return base | report | {'repository': repository, 'root': str(root),
            'absence_proven': False, 'dirty_paths_truncated': len(dirty_paths) > 64 or report.get('dirty_paths_truncated', False)}
