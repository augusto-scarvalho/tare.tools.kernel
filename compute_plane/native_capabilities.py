"""Runtime capability port over mediated tools; no Aider session or Work authority."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from threading import Lock
from dataclasses import replace
import time

from tare_tools_agent_runtime import EffectResult
from .mediated_tools import MediatedTools, CommandInterrupted
from .native_models import encode, object_json
from .native_terminal import TERMINAL_SPECS, MUTATIONS
from .native_arguments import normalize_booleans


def canonical(args):
    """What a model meant, where it is unambiguous; each of these was refused and its model call wasted.
    './x' and '. /x' name x (the instruction scope check refuses '.' segments); a search's mode
    'insensitive' or 'sensitive' is its case; a file or filename argument is the path."""
    args = dict(args)
    for alias in ('file', 'filename'):
        if 'path' not in args and isinstance(args.get(alias), str):
            args['path'] = args.pop(alias)
    path = args.get('path')
    if isinstance(path, str):
        args['path'] = re.sub(r'^(?:\.\s*/+\s*)+', '', path) or '.'
    if args.get('mode') in ('sensitive', 'insensitive') and 'case' not in args:
        args['case'] = args.pop('mode')
    return args


def _schema(description, properties, required):
    return {'description': description, 'parameters': {'type': 'object', 'properties': properties,
            'required': required, 'additionalProperties': False}}


TEXT = {'type': 'string'}
SHA256 = {'type': 'string', 'minLength': 64, 'maxLength': 64, 'pattern': '^[0-9a-f]{64}$'}
TOOL_SPECS = {
    'search': _schema('Search current repository text with ripgrep (required on PATH). Literal, case-sensitive by default; '
        'uses ripgrep ignore/glob rules and excludes links, .git and files over 2 MiB. '
        'Returns paths and line numbers. PARTIAL is successful, incomplete coverage: use the returned matches '
        'and narrow path/glob for the remaining scope, then read before editing. '
        'Recovered output limits are historical receipts; ERROR or a refused effect requires correction.',
        {'repository': TEXT, 'query': {'type': 'string', 'minLength': 1,
            'description': 'Nonempty single-line content query, up to 4096 UTF-8 bytes. '
                           'glob filters candidate files; it does not replace the content query.'},
         'path': {'type': 'string', 'minLength': 1, 'default': '.',
                  'description': 'Relative file or directory scope; omit or use . for the repository root. '
                                 'Prefer a known subtree for large repositories.'}, 'glob': TEXT,
         'mode': {'type': 'string', 'enum': ['literal', 'regex'], 'default': 'literal',
                  'description': 'Use regex explicitly for patterns such as ^class or first|second; omitted means literal text.'},
         'case': {'type': 'string', 'enum': ['sensitive', 'insensitive'], 'default': 'sensitive'},
         'limit': {'type': 'integer', 'minimum': 1, 'maximum': 200, 'default': 50}}, ['repository', 'query']),
    'list': _schema('List up to 200 immediate entries in a selected repository directory; useful when its index is unavailable.',
        {'repository': TEXT, 'path': TEXT}, ['repository']),
    'read': _schema('Read a bounded source excerpt and the current file SHA-256. Line numbers are 1-based and inclusive; '
        'Each text line is number|source with no added space after |; copy only source after the first | verbatim for edits. '
        'Pages contain at most 200 lines and 24000 bytes; larger ranges return a partial page. '
        'Omit end for a page of up to 200 lines from start. '
        'Use returned start/end and next_start to see what was actually read; a page is not the whole file. '
        'Long lines return partial_line and 1-based Unicode character columns. Continue at next_start '
        'with next_start_column as start_column; a partial line is not whole-line coverage.',
        {'repository': TEXT, 'path': TEXT,
         'start': {'type': 'integer', 'minimum': 1, 'default': 1,
                   'description': 'First line to read, inclusive.'},
         'end': {'type': 'integer', 'minimum': 1,
                  'description': 'Requested last line, inclusive. Large ranges are paginated; follow next_start if needed.'},
         'start_column': {'type': 'integer', 'minimum': 1,
                          'description': 'Continue a long line at the returned next_start_column; reads one line fragment.'},
         'sha256': SHA256}, ['repository', 'path']),
    'edit': _schema('Apply an exact unique replacement using the 64 lowercase hexadecimal SHA-256 returned by read. '
        'For a missing file pass JSON null. Empty search appends. '
        'To delete an existing file, pass delete=true, changes=[], and its current non-null SHA-256; '
        'deletion returns sha256=null. Omit delete or pass false for replacements. '
        'Supply changes even for one replacement: a list of 1 to 16 search/replace pairs applied in order '
        'in memory, then committed once with one SHA check. A file and hash alone are not an edit; use read to inspect. '
        'Wait for the resulting SHA before the next edit; never reuse a stale SHA. '
        'Finish other edits and checks before changing pinned AGENTS.md instructions; a new conversation is needed to adopt them.',
        {'repository': TEXT, 'path': TEXT, 'sha256': {**SHA256, 'type': ['string', 'null']},
         'delete': {'type': 'boolean', 'default': False},
         'changes': {'type': 'array', 'minItems': 0, 'maxItems': 16,
                     'items': {'type': 'object', 'properties': {'search': TEXT, 'replace': TEXT},
                               'required': ['search', 'replace'], 'additionalProperties': False}}},
        ['repository', 'path', 'sha256', 'changes']),
    'verify': _schema('Execute one host-configured verifier by name.', {'name': TEXT}, ['name']),
    'command': _schema('Run one noninteractive argv command under the configured host review and OS isolation. '
        'No shell session or stdin. The host selects executable identity and permissions; refusal never bypasses review.',
        {'repository': TEXT, 'argv': {'type': 'array', 'minItems': 1, 'maxItems': 64,
                                    'items': {'type': 'string'}}}, ['repository', 'argv']),
    'context': _schema('Query SpecGraph structural context in a selected repository snapshot before text search. '
        'MISS does not prove absence. Read returned paths to verify current bytes.',
        {'repository': TEXT, 'query': TEXT}, ['repository', 'query']),
}
TOOL_SPECS.update(TERMINAL_SPECS)


class NativeCapabilities:
    """One session/binding and explicit roots, write scopes, verifiers and scratch.

    The owning host must provide exclusive workspace custody. No resumption of
    the in-memory reads/verifications state is inferred from a session ID.
    """
    names = ('read', 'edit', 'verify', 'list')

    def __init__(self, *, session_id, binding, repositories, writable, verifiers, scratch_root,
                 specgraph_edit_context=True, project_context=None, role='implementer', capabilities=None, review_root=None,
                 terminal=None, scoped_instructions=False, structured_reads=False, command_mediation=None):
        scratch = Path(scratch_root)
        if not scratch.is_dir() or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                                      for p in (scratch, *scratch.parents)):
            raise ValueError('explicit_regular_scratch_directory_required')
        if role not in {'implementer', 'planner', 'plan-auditor', 'auditor', 'explorer'}:
            raise ValueError('native_role_invalid')
        if role != 'implementer' and any(writable.values()):
            raise ValueError('native_role_cannot_write')
        if role not in {'implementer', 'auditor'} and verifiers:
            raise ValueError('native_role_cannot_verify')
        if terminal is not None and role != 'implementer':
            raise ValueError('native_role_cannot_execute_terminal')
        if type(scoped_instructions) is not bool:
            raise ValueError("scoped_instructions_boolean_required")
        self.instructions = None
        self.terminal = terminal
        self.session_id, self.binding = session_id, binding
        self._lock, self._control, self._started = Lock(), None, False
        self._seen = set()
        self._refresh_reads = set()
        self.project_context = project_context
        names = type(self).names if role == 'implementer' else ('read', 'list')
        if role == 'auditor' and verifiers:
            names += ('verify',)
        names += (('context',) if project_context is not None else ())
        if capabilities is not None:
            if (not isinstance(capabilities, list) or not capabilities
                    or any(not isinstance(n, str) or n not in (*names, 'search', *(('command',) if role == 'implementer' and terminal is None else ()), *(TERMINAL_SPECS if terminal is not None else ())) for n in capabilities)
                    or len(capabilities) != len(set(capabilities))):
                raise ValueError('native_capabilities_invalid_for_role_or_context')
            names = tuple(n for n in TOOL_SPECS if n in capabilities)
        self.names = names
        if type(structured_reads) is not bool:
            raise ValueError('structured_reads_boolean_required')
        self._specs = object_json(encode(TOOL_SPECS))
        if structured_reads:
            read = self._specs['read']
            read['description'] += (
                ' For JSON, set json_pointer (empty string for root) to inspect a bounded structural view. '
                'Objects list their keys, not their values; arrays return paginated items. '
                'When the task names fields whose values are needed, select them in the first read with '
                'json_fields instead of spending a call on the directory. Use the directory to discover unknown keys. '
                'Select fields with comma-separated '
                'relative JSON pointers, e.g. json_fields="/id,/title,/completion/status". '
                'Each selected field includes its whole value; fields do not paginate nested collections. '
                'Point json_pointer at a nested array to page its items. An oversized object projection '
                'returns directory metadata with projection_complete=false; values remain uninspected. '
                'Follow next_offset, pin sha256 across pages, and use deeper pointers for large items. '
                'JSON views are projections, not source lines; never use their formatting as edit search text.')
            read['parameters']['properties'].update({
                'json_pointer': TEXT, 'json_fields': TEXT,
                'offset': {'type': 'integer', 'minimum': 0},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100}})
        self._repository_names = tuple(sorted(repositories))
        self.context_observations = []
        self.review_root = Path(review_root) if review_root is not None else None
        self.review_errors = []
        self.command_mediation = command_mediation
        from .command_mediation import CallbackCommandMediation
        config = object_json(encode({'host': 'native', 'role': role, 'repositories': repositories,
            'writable': writable, 'verifiers': verifiers, 'allow_exec': False,
            # The unused file adapter still requires a nonempty schema for a
            # terminal-only selection. Public admission always checks self.names.
            'admitted_capabilities': [n for n in self.names if n not in TERMINAL_SPECS and n != 'command'] or ['list'],
            'specgraph_edit_context': specgraph_edit_context}))
        self._tools = MediatedTools(config, cancelled=lambda: self._control is None or self._control.should_stop(),
            before_effect=self._before_effect, scratch_root=scratch, before_edit=self._before_edit,
            command_mediation=CallbackCommandMediation(self.mediate_command) if command_mediation is not None else None)
        from .native_effect_recovery import EffectReceipts
        self._receipts = EffectReceipts(scratch, session_id, config)
        self._request = None
        if scoped_instructions:
            from .native_instructions import ScopedInstructions
            self.instructions = ScopedInstructions(self._tools.roots, self._tools.path)

    @property
    def tool_specs_json(self):
        specs = object_json(encode({k: self._specs[k] for k in self.names}))
        for spec in specs.values():
            properties = spec['parameters']['properties']
            if 'repository' in properties:
                properties['repository'] = {
                    'type': 'string', 'enum': list(self._repository_names),
                    'description': 'Repository identifier from this enum, not a filesystem path.',
                }
        return encode(specs)

    def observations(self):
        return object_json(encode({'effects': self._tools.effects, 'reads': self._tools.reads,
                                  'verifications': self._tools.verifications,
                                  'context': self.context_observations, 'review_errors': self.review_errors,
                                  **({'scoped_project_instructions': self.instructions.records()} if self.instructions is not None else {}),
                                  **({'terminal': self.terminal.observations()} if self.terminal is not None else {})}))

    def restore(self, journal, *, refresh=False, accepted_sources=None):
        """Recover completed observations, never re-execute effects or trust old tests."""
        pending = {}
        unfinished = set()
        expected = {}
        for record in journal.snapshot().records:
            event = record.event()
            if event['kind'] == 'effect_requested':
                pending[event['effect_id']] = event['tool_call']
                unfinished.add(event['effect_id'])
            elif event['kind'] == 'invocation_not_started':
                unfinished.discard(event['invocation_id'])
            elif event['kind'] == 'effect_returned':
                result = event['result']
                if result['status'] != 'unknown':
                    unfinished.discard(result['effect_id'])
                call = pending.get(result['effect_id'])
                if call is not None and call['name'] in MUTATIONS and result['status'] in {'succeeded', 'failed', 'unknown'}:
                    expected.clear()
                    self._tools.reads.clear()
                    self._tools.verifications.clear()
                if self.instructions is not None and call is not None:
                    saved_output = object_json(result['output_json'])
                    if 'scoped_project_instructions' in saved_output:
                        self.instructions.restore(saved_output['scoped_project_instructions'])
                if call is None or result['status'] != 'succeeded':
                    continue
                name = call['name']
                args = self._intended(name, object_json(call['arguments_json']))
                output = object_json(result['output_json'])
                if name == 'read' and output.get('lines', 0) > 0 and not output.get('text'):
                    continue  # Old successful EOF reads did not deliver source text.
                if name == 'edit':
                    self._tools.effects = output['effects']
                    self._tools.written.add(args['repository']+':'+args['path'])
                if name in {'read', 'edit'}:
                    key = args['repository']+':'+args['path']
                    expected[key] = (args['repository'], args['path'], output.get('sha256'), record.sequence)
                self._seen.add(encode({'name': name, 'args': args, 'effects': self._tools.effects,
                    'output': {k: v for k, v in output.items() if k != 'process'}}))
        if unfinished:
            return  # Runtime reports reconciliation before any next invocation.
        changes = []
        for key, (repository, path, sha, sequence) in expected.items():
            current = self._tools.observed(repository, path)
            accepted = (accepted_sources or {}).get(key)
            acknowledged = (accepted is not None and accepted['baseline_sha256'] == sha
                            and accepted['observation_sequence'] == sequence)
            previous = accepted['current_sha256'] if acknowledged else sha
            if current != previous:
                if not refresh:
                    raise ValueError('native_resume_source_changed:'+key)
                changes.append({'repository': repository, 'path': path,
                    'baseline_sha256': sha, 'previous_sha256': previous, 'current_sha256': current,
                    'observation_sequence': sequence})
                self._refresh_reads.add(key)
            elif acknowledged:
                self._refresh_reads.add(key)
            elif not acknowledged:
                self._tools.reads[key] = sha
        # Existing candidate verification must run on exact candidate bytes.
        self._tools.verifications.clear()
        return changes

    def _before_effect(self):
        if self._control is None or self._control.should_stop():
            raise InterruptedError('execution_control_stopped')
        self._started = True

    def mediate_command(self, facts, effect):
        if self._request is None or self._control is None:
            raise ValueError('command_effect_context_unavailable')
        request=self._request
        if request.tool_call.name=='search':
            # One closed read-only search inventories files and then matches.
            # Each physical command needs its own exact once-only claim while
            # the parent retains one search result/recovery identity.
            self._command_ordinal+=1
            request=replace(request,effect_id=request.effect_id+'/command/'+str(self._command_ordinal))
        facts = replace(facts, owner=self.session_id, session_id=self.session_id,
            effect_id=request.effect_id, work_id=self.binding.work_id,
            binding_id=self.binding.binding_id, fence_seq=self.binding.fence)
        self._action_request=request
        try:
            return self.command_mediation.execute(facts, effect)
        finally:
            self._action_request=None

    def bind_command_mediation(self, port):
        """Host-only installation before the first effect; never an admitted model tool."""
        if self._request is not None or self.command_mediation is not None or not callable(getattr(port, 'execute', None)):
            raise ValueError('command_mediation_binding_invalid')
        from .command_mediation import CallbackCommandMediation
        self.command_mediation = port
        self._tools.command_mediation = CallbackCommandMediation(self.mediate_command)

    def bind_command_confinement(self, process_factory, environment, seconds, *, verification_seconds=None,
                                 staged_effects=False, unbounded_verification=False):
        """Host-only installation before command execution; never a model tool."""
        if (self._started or self._request is not None
                or getattr(self._tools, 'process_factory', None) is not None
                or self.terminal is not None):
            raise ValueError('command_confinement_binding_invalid')
        from .mediated_tools import validate_command_confinement
        if type(staged_effects) is not bool:
            raise ValueError('staged_effects_must_be_bool')
        if type(unbounded_verification) is not bool:
            raise ValueError('unbounded_verification_must_be_bool')
        copied, seconds = validate_command_confinement(process_factory, environment, seconds)
        if verification_seconds is not None:
            _,verification_seconds=validate_command_confinement(process_factory,environment,verification_seconds)
        self._tools.process_factory = process_factory
        self._tools.command_environment = copied
        self._tools.command_seconds = seconds
        self._tools.verification_seconds = verification_seconds
        self._tools.unbounded_verification = unbounded_verification
        self._staged_effects = staged_effects

    def claim_action(self, action, authorization):
        if self._request is None or self._control is None or self._control.should_stop():
            raise ValueError('command_effect_context_unavailable')
        return self._receipts.claim_action(getattr(self,'_action_request',None) or self._request, action, authorization)

    def execute(self, request, control):
        started = time.monotonic()
        result = self._execute(request, control)
        if result.status != 'unknown' and request.session_id == self.session_id and control.binding == self.binding:
            self._save_result(request, result)
        if request.tool_call.name in {'read', 'context', 'list', 'search'}:
            self.context_observations.append({'effect_id': request.effect_id,
                'operation': request.tool_call.name, 'status': result.status,
                'duration_ms': round((time.monotonic()-started)*1000, 3),
                'returned_bytes': len(result.output_json.encode()),
                'tokens': None, 'token_accounting': 'included_in_subsequent_model_request_not_separately_attributed'})
        return result

    def _before_edit(self, args, new, pre_context, effects):
        from .native_review import digest
        self._receipts.save(self._request, edit={
            'after': digest(None if new is None else new.encode()), 'effects': effects, 'pre_context': pre_context})

    def recover_effect(self, journal):
        from .native_effect_recovery import pending
        request = pending(journal, self.session_id)
        if request is None:
            return None
        if request.tool_call.name not in self.names:
            raise ValueError('Recovery capability is not admitted.')
        result = self._receipts.inspect(request, self._tools, self.review_root)
        return (request, result) if result is not None else None

    def _intended(self, name, args):
        """The arguments as executed and as recorded on reopen: canonical, and a file path written as the
        repository, with one repository in the session, is that path there."""
        if name == 'edit' and 'changes' not in args and {'search', 'replace'} <= args.keys():
            args = dict(args)
            args['changes'] = [{'search': args.pop('search'), 'replace': args.pop('replace')}]
        if (name in ('read', 'list', 'search') and 'path' not in args and len(self._tools.roots) == 1
                and isinstance(args.get('repository'), str) and args['repository'] not in self._tools.roots):
            args = dict(args, path=args['repository'], repository=next(iter(self._tools.roots)))
        return canonical(args)

    def _execute(self, request, control):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('capability_execution_in_progress')
        scoped = None
        try:
            self._control, self._started = control, False
            self._request = request
            self._command_ordinal=0
            if request.session_id != self.session_id or control.binding != self.binding or control.should_stop():
                return EffectResult(request.effect_id, 'refused', '{"reason":"CONTROL_OR_SESSION_MISMATCH"}')
            name = request.tool_call.name
            args = self._intended(name, object_json(request.tool_call.arguments_json))
            if name == 'edit' and args.get('repository', '')+':'+args.get('path', '') in self._refresh_reads:
                raise ValueError('source_changed_read_again')
            if name not in self.names:
                raise ValueError('capability_not_admitted')
            if name in TERMINAL_SPECS:
                result = self.terminal.execute(request, control)
                if name in MUTATIONS and result.status in {'succeeded', 'failed', 'unknown'}:
                    self._tools.reads.clear()
                    self._tools.verifications.clear()
                return result
            schema = self._specs[name]['parameters']
            args = normalize_booleans(args, schema)
            unexpected, missing = set(args) - schema['properties'].keys(), set(schema['required']) - args.keys()
            if unexpected or missing:
                # The code stays first; the fields let a model correct its next call.
                raise ValueError('capability_arguments_mismatch: ' + '; '.join(part for part in (
                    'missing ' + ', '.join(sorted(missing)) if missing else '',
                    'unexpected ' + ', '.join(sorted(unexpected)) if unexpected else '',
                    'accepted ' + ', '.join(sorted(schema['properties']))) if part))
            for key, value in args.items():
                kind = schema['properties'][key]['type']
                if (kind == 'boolean' and type(value) is not bool or
                    kind == 'integer' and type(value) is not int or
                    kind == 'string' and not isinstance(value, str) or
                    isinstance(kind, list) and value is not None and not isinstance(value, str)):
                    raise ValueError('capability_argument_type')
            widened = None
            if name == 'search' and isinstance(args.get('path'), str) and args['path'] != '.':
                # A search under a folder that does not exist searches its nearest existing parent below the
                # repository root instead of being refused; the output says so.
                from .native_journal import io_path
                parent = PurePosixPath(args['path'])
                while str(parent) != '.' and not io_path(self._tools.path(args['repository'], str(parent),
                                                                            directory=True)).exists():
                    parent = parent.parent
                if str(parent) not in (args['path'], '.'):  # never widened to the whole repository
                    widened, args = args['path'], dict(args, path=str(parent))
            if self.instructions is not None and name in {'read', 'edit', 'list', 'search'}:
                from .native_journal import io_path
                repository, target = args['repository'], args.get('path', '.')
                directory = name == 'list' or (name == 'search' and io_path(
                    self._tools.path(repository, target, directory=True)).is_dir())
                known = {(row['repository'], row['path']) for row in self.instructions.records()}
                scoped = self.instructions.observe(repository, target, directory=directory)
                if name == 'edit' and any(row['text'] is not None and
                        (row['repository'], row['path']) not in known for row in scoped):
                    return EffectResult(request.effect_id, 'refused', encode({
                        'reason': 'Read the scoped project instructions before retrying this edit.',
                        'scoped_project_instructions': scoped}))
            self._tools.config['operation_timeout'] = control.remaining_seconds()
            if name == 'command':
                if self.command_mediation is None or self._tools.process_factory is None or self.terminal is not None:
                    raise ValueError('protected_command_host_binding_required')
                argv = args['argv']
                if (not isinstance(argv, list) or not 1 <= len(argv) <= 64
                        or any(not isinstance(a, str) or not a or '\0' in a or len(a) > 8192 for a in argv)
                        or len(encode(argv).encode('utf-8')) > 16384):
                    raise ValueError('protected_command_argv_invalid')
                output = self._tools.command(argv, self._tools.path(args['repository'], '.', directory=True),
                                             timeout=control.remaining_seconds())
                self._tools.reads.clear(); self._tools.verifications.clear()
            elif name == 'context':
                repository = args['repository']
                if repository not in self._tools.roots:
                    raise ValueError('repository_not_selected')
                dirty = [p[len(repository)+1:] for p in self._tools.written if p.startswith(repository+':')]
                output = self.project_context.lookup(repository, args['query'], control, dirty_paths=sorted(dirty))
            else:
                if name == 'edit':
                    from .native_review import edit_args
                    args = edit_args(args)
                admitted = {k: v for k, v in args.items() if name != 'read' or k != 'sha256'}
                before = None
                captured = False
                if name == 'edit' and self.review_root is not None:
                    try:
                        from .native_review import capture
                        before = capture(self._tools.path(args['repository'], args['path'], write=True))
                        captured = True
                    except (OSError, ValueError) as exc:
                        self.review_errors.append({'effect_id': request.effect_id, 'reason': type(exc).__name__})
                output = self._tools.perform(name, admitted)
                if captured:
                    try:
                        from .native_review import save_edit
                        after = capture(self._tools.path(args['repository'], args['path']))
                        save_edit(self.review_root, request.effect_id, args, before, after, output)
                    except (OSError, ValueError) as exc:
                        # Review is optional evidence, never turn an applied edit into a retry.
                        self.review_errors.append({'effect_id': request.effect_id, 'reason': str(exc)[:128]})
            status = 'failed' if name in {'verify', 'command'} and output['returncode'] != 0 else 'succeeded'
            if widened is not None:
                output = {**output, 'scope_widened': f"{widened} does not exist; searched {args['path']} instead"}
            if name == 'search' and output['status'] == 'ERROR':
                status = 'failed'
            if name == 'context' and output['status'] not in {'HIT', 'AVAILABLE', 'MISS', 'SILENT'}:
                status = 'failed'
            if name == 'read' and 'sha256' in args and args['sha256'] != output.get('sha256'):
                self._tools.reads.pop(args['repository']+':'+args['path'], None)
                status = 'failed'
                output = {'status': 'STALE', 'repository': args['repository'], 'path': args['path'],
                          'expected_sha256': args['sha256'], 'current_sha256': output.get('sha256'),
                          'reason': 'pinned_source_changed; current content not substituted'}
            if name == 'read' and status == 'succeeded':
                self._refresh_reads.discard(args['repository']+':'+args['path'])
            if scoped is not None:
                output['scoped_project_instructions'] = scoped
                if name == 'edit' and any(Path(row['path']) == Path(args['path'])
                        and row['sha256'] != output.get('sha256') for row in scoped):
                    output['instruction_update'] = {
                        'path': args['path'], 'applied': True, 'guidance_adopted': False,
                        'requires_new_conversation': True,
                        'next_action': 'The edit is applied. Report changes and pending checks, then start a new '
                                       'conversation before further project work. Do not reapply the edit or '
                                       'silently replace the pinned instructions.'}
            observation = encode({'name': name, 'args': args, 'effects': self._tools.effects,
                                  'output': {k: v for k, v in output.items() if k != 'process'}})
            # A search or context lookup that finds nothing is news the first time: a session that starts on red
            # tests and looks for what does not exist yet is working; only repeating what it saw is not.
            progressed = status == 'succeeded' and observation not in self._seen
            self._seen.add(observation)
            return EffectResult(request.effect_id, status, encode(output), progressed)
        except Exception as exc:
            # A post-write hook or interrupted verifier cannot become a safe refusal.
            status = 'unknown' if self._started else 'refused'
            output = {'error_type': type(exc).__name__}
            if scoped is not None:
                output['scoped_project_instructions'] = scoped
            if isinstance(exc, (ValueError, KeyError)) and not self._started:
                output['reason'] = str(exc)[:256]
                if str(exc).startswith('instruction_pinned_row_changed:'):
                    output['next_action'] = (
                        'Project instructions changed after this conversation pinned them. Retain applied changes '
                        'and pending checks, then start a new conversation to load the current instructions. '
                        'Do not retry project tools or silently replace the pinned instructions.')
                if isinstance(getattr(exc, 'detail', None), str):
                    output['detail'] = exc.detail[:600]
            if isinstance(exc, CommandInterrupted):
                output['process'] = exc.receipt
                if request.tool_call.name in {'command','verify'}:
                    self._tools.reads.clear(); self._tools.verifications.clear()
                if (getattr(self,'_staged_effects',False) and request.tool_call.name in {'command','verify'}
                        and exc.receipt.get('cleanup_verified') is True and not exc.receipt.get('cancelled')
                        and not exc.receipt.get('isolation_violation') and not control.should_stop()
                        and (exc.receipt.get('output_limit') or exc.receipt.get('timeout'))):
                    # The host confines all writes to a disposable staged area
                    # and denies network. Termination is proved, but staged
                    # changes may be partial: inspect them, never replay here.
                    status='failed'
                    output.update(reason='Staged command stopped at its output/time bound; partial changes may remain.',
                        effects_may_be_partial=True,
                        next_action='Inspect the candidate before any correction. Do not replay the command automatically. '
                            'Use bounded output or capture/filter large data; checks and independent review still gate application.')
            return EffectResult(request.effect_id, status, encode(output))
        finally:
            self._control = None
            self._lock.release()

    def _save_result(self, request, result):
        from dataclasses import asdict
        self._receipts.save(request, result=asdict(result))
