"""Bounded source collection for exploration; no model, Aider or audit driver."""
import hashlib
import time

from .mediated_tools import MediatedTools
from .native_models import encode
from .native_project_context import NativeProjectContext


class ExplorationSources:
    def __init__(self, root, repository, control, *, index_state=None):
        self.repository, self.control = repository, control
        self.tools = MediatedTools({'role': 'auditor', 'repositories': {repository: str(root)},
            'admitted_capabilities': ['read', 'list'], 'specgraph_edit_context': False},
            cancelled=control.should_stop)
        self.context = NativeProjectContext(repositories={repository: root}, work_id=None,
            state_dirs={repository: index_state} if index_state else {},
            max_bytes=8192, timeout_ms=1000, role='explorer')
        self.observations = []

    def read(self, path, start, end, expected=None):
        started = time.monotonic()
        args = {'repository': self.repository, 'path': path, 'start': start, 'end': end}
        value = self.tools.perform('read', args)
        self.observations.append({'operation': 'read', 'path': path,
            'returned_bytes': len(encode(value).encode()), 'duration_ms': (time.monotonic()-started)*1000,
            'tokens': None})
        if self.control.should_stop():
            raise InterruptedError('exploration_control_stopped')
        if not value.get('sha256'):
            raise ValueError('exploration_source_missing_or_range_outside_file')
        if expected is not None and value['sha256'] != expected:
            raise ValueError('exploration_source_changed')
        if start > value.get('lines', 0):
            raise ValueError('exploration_source_missing_or_range_outside_file')
        source = {'repository': self.repository, 'path': path, 'revision': 'WORKTREE',
                  'start_line': start, 'end_line': min(end, value['lines']),
                  'sha256': value['sha256'], 'text': value['text']}
        source['id'] = 'X-' + hashlib.sha256(encode({k:v for k,v in source.items() if k != 'text'}).encode()).hexdigest()[:24]
        return source

    def collect(self, query, focus=()):
        if not isinstance(focus, (list, tuple)) or len(focus) > 8:
            raise ValueError('exploration_requires_at_most_eight_focus_ranges')
        for item in focus:
            if (not isinstance(item, dict) or set(item) != {'path', 'start_line', 'end_line'}
                    or type(item['start_line']) is not int or type(item['end_line']) is not int
                    or not 1 <= item['start_line'] <= item['end_line'] < item['start_line'] + 200):
                raise ValueError('exploration_focus_requires_path_and_up_to_200_lines')
        started = time.monotonic()
        context = self.context.lookup(self.repository, query, self.control)
        self.observations.append({'operation': 'context', 'status': context['status'],
            'duration_ms': (time.monotonic()-started)*1000, 'returned_bytes': len(encode(context).encode()), 'tokens': None})
        if self.control.should_stop():
            raise InterruptedError('exploration_control_stopped')
        sources, missing, seen = [], [], set()
        selections = [(item, None, True) for item in focus]
        # Expand only hash-pinned snapshot ranges, never a guessed path or whole repository.
        if not focus:
            for symbol in context.get('symbols', [])[:4]:
                if not symbol.get('source_sha256'):
                    missing.append({'symbol': symbol['id'], 'reason': 'snapshot_source_hash_missing'})
                    continue
                start, end = symbol['range']
                selections.append(({'path': symbol['id'].split(':', 1)[0], 'start_line': start,
                                    'end_line': min(end, start+199)}, symbol['source_sha256'], False))
        for item, expected, required in selections:
            key = (item['path'], item['start_line'], item['end_line'])
            if key in seen:
                continue
            seen.add(key)
            try:
                source = self.read(*key, expected=expected)
                if len(encode(sources + [source]).encode()) > 24000:
                    raise ValueError('exploration_source_packet_over_budget')
                sources.append(source)
            except (OSError, ValueError) as exc:
                if isinstance(exc, InterruptedError) or required:
                    raise
                missing.append({'path': item['path'], 'reason': str(exc)[:160]})
        packet = {'schema': 'tare.tools/exploration-sources/1', 'context': context,
                  'sources': sources, 'missing': missing, 'authority': 'NONE', 'absence_proven': False,
                  'selection': 'explicit_focus' if focus else 'up_to_four_snapshot_symbols',
                  'relationships': 'PUBLISHED_SNAPSHOT_NOT_LIVE_ATTESTED',
                  'observations': self.observations}
        if not sources:
            packet['directory'] = self.tools.perform('list', {'repository': self.repository})
        return packet

    def verify(self, sources):
        for source in sources:
            self.read(source['path'], source['start_line'], source['end_line'], source['sha256'])
