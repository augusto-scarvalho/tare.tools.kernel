"""Scoped, read-only ripgrep capability; no shell or structural-search prerequisite.

The host retains exclusive cooperative workspace custody, as for read/edit.
Validated file operands prevent link traversal; this is not an OS sandbox for
arbitrary code or protection against a concurrently hostile filesystem writer.
"""
import json
from pathlib import Path
from .managed_process import host_executable


def search(tools, *, repository, query, path='.', glob=None, mode='literal',
           case='sensitive', limit=50):
    if (not isinstance(query, str) or not query or len(query.encode()) > 4096
            or '\0' in query or '\n' in query or '\r' in query):
        raise ValueError('search_requires_single_line_query_up_to_4096_bytes')
    if mode not in {'literal', 'regex'} or case not in {'sensitive', 'insensitive'}:
        raise ValueError('search_mode_or_case_invalid')
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('search_limit_requires_1_to_200')
    if glob is not None and (not isinstance(glob, str) or not glob or len(glob) > 1024 or '\0' in glob):
        raise ValueError('search_glob_invalid')
    scope = tools.path(repository, path, directory=True)
    if not scope.exists():
        raise ValueError('search_scope_missing')
    from .mediated_tools import MAX_READ_BYTES
    explicit_file = scope.is_file()
    file_limit = MAX_READ_BYTES if explicit_file else 2 * 1024 * 1024
    executable = host_executable('rg')
    result = {'repository': repository, 'path': path, 'query': query, 'mode': mode,
              'case': case, 'glob': glob, 'matches': [], 'truncated': False,
              'coverage': {'files_selected': 0, 'files_searched': 0, 'files_skipped': 0,
                           'policy': 'ripgrep repository ignore/glob rules; no parent/global ignore or config; '
                                     f'no links, .git, binary content or files over {file_limit // (1024 * 1024)} MiB'},
              'process': {'count': 0, 'wall_ms': 0, 'cleanup_verified': True}}
    def error(reason):
        return {**result, 'status': 'ERROR', 'reason': reason, 'truncated': True}
    if executable is None:
        return error('ripgrep_not_available_on_host_PATH')
    root = tools.roots[repository]
    def run(args, *, partial_inventory=False):
        # Only fixed ripgrep options and validated file operands enter this path.
        # No caller can pass arbitrary argv, config, preprocessor or follow flags.
        from .mediated_tools import CommandInterrupted
        try:
            receipt = tools.command([executable, '--no-config', *args], root,
                                    gate=False, read_only=True, output_bytes=1024*1024,
                                    **({'partial_read':True} if partial_inventory else {}))
        except CommandInterrupted as exc:
            process = exc.receipt
            if (process.get('output_limit') is not True or process.get('cleanup_verified') is not True
                    or process.get('cancelled') is not False or process.get('timeout') is not False
                    or process.get('isolation_violation')):
                raise
            # A bounded read-only inventory/search overflow is recoverable by
            # narrowing scope. It proves neither absence nor complete coverage.
            receipt = {'process': process, 'output_limit': True}
            result['process']['interruption'] = process
        if receipt.get('output_limit'):
            result['process']['interruption'] = receipt['process']
        result['process']['count'] += 1
        result['process']['wall_ms'] += receipt['process'].get('wall_ms', 0)
        result['process']['cleanup_verified'] &= receipt['process']['cleanup_verified']
        return receipt
    def recovered_output_limit(stage):
        # Keep the exact receipt as history, without presenting a recovered
        # bounded read as an interruption of the completed search capability.
        process=result['process'].pop('interruption')
        result['process'].setdefault('recovered_output_limits',[]).append(
            {'stage':stage,'process':process})
    inventory = run(['--files', '--null', '--sort', 'path', '--no-ignore-parent',
                     '--no-ignore-global', '--no-require-git',
                     *(['--glob', glob] if glob is not None else []), '--', './'+path],partial_inventory=True)
    partial_inventory = inventory.get('output_limit') is True
    if partial_inventory and ('output' not in inventory or '\0' not in inventory['output']):
        return error('search_inventory_output_limit: narrow path to a known directory/file or add glob; '
                     'no matching search completed. Use list to choose a subtree.')
    if not partial_inventory and (inventory['returncode'] not in (0, 1) or inventory['output_truncated']):
        return error('search_inventory_failed: '+inventory['output'][:1000])
    if partial_inventory:
        # Keep complete NUL-delimited names only; the clipped tail is not a path.
        inventory['output'] = inventory['output'].rsplit('\0',1)[0]+'\0'
        result.update(truncated=True,inventory_complete=False,
            inventory_notice='Incomplete inventory: search only the captured prefix, up to 4096 files. '
                             'Narrow path/glob for remaining scope; no absence is proven.')
        recovered_output_limit('inventory')
    names = inventory['output'].split('\0')
    if names[-1]:
        return error('search_inventory_incomplete')
    files, proven = [], set()
    for name in (names[:-1][:4096] if partial_inventory else names[:-1]):
        name = Path(name).as_posix()
        try:
            target = tools.path(repository, name, proven=proven)
            if not target.is_file() or target.stat().st_size > file_limit:
                raise ValueError('search_file_excluded')
        except (ValueError, OSError):
            result['coverage']['files_skipped'] += 1
            continue
        files.append(name)
    result['coverage']['files_selected'] = len(files)
    if explicit_file and not files:
        return {**result, 'status': 'PARTIAL', 'truncated': True,
                'reason': 'search_source_not_inspected: the named file was excluded by ignore, path or size policy; '
                          'no absence is proven. Read a bounded excerpt or request a smaller source.'}
    # Small batches also stay below Windows' command-line length limit.
    batches, batch, size = [], [], 0
    for name in files:
        if batch and (len(batch) == 32 or size+len(name) > 12000):
            batches.append(batch); batch, size = [], 0
        batch.append(name); size += len(name)+4
    if batch:
        batches.append(batch)
    returned_bytes = 0
    def append_match(match):
        nonlocal returned_bytes
        cost = len(json.dumps(match, ensure_ascii=False).encode())
        if len(result['matches']) >= limit or returned_bytes+cost > 24000:
            result['truncated'] = True
            return False
        result['matches'].append(match)
        returned_bytes += cost
        return True
    for batch in batches:
        for name in batch:
            tools.path(repository, name, proven=proven)  # Recheck each file before opening current bytes.
        pattern_args = ['--max-count', str(limit+1),
                     '--fixed-strings' if mode == 'literal' else '--no-fixed-strings',
                     '--case-sensitive' if case == 'sensitive' else '--ignore-case',
                     '-e', query, '--', *('./'+name for name in batch)]
        found = run(['--json', '--sort', 'path', *pattern_args])
        if found.get('output_limit'):
            # JSON output ignores rg's column cap and serializes entire matching
            # lines. After proved cleanup, repeat this read-only batch in bounded
            # location format. Discard its display text; it is not source content.
            for name in batch:
                tools.path(repository, name, proven=proven)
            locations = run(['--sort', 'path', '--no-heading', '--with-filename', '--null',
                '--line-number', '--color', 'never', '--max-columns', '1', '--no-max-columns-preview',
                *pattern_args])
            if (locations.get('output_limit') or locations.get('returncode') not in (0,1)
                    or locations.get('output_truncated')):
                return error('search_match_output_limit: narrow path/glob or lower limit; '
                             'returned matches are incomplete and do not prove absence.')
            pending = locations['output']
            while pending:
                filename, separator, rest = pending.partition('\0')
                row, newline, pending = rest.partition('\n')
                number, colon, _ = row.partition(':')
                name = Path(filename).as_posix()
                if (not separator or not newline or not colon or not number.isascii()
                        or not number.isdigit() or int(number) < 1 or name not in batch):
                    return error('search_location_response_invalid: returned locations are incomplete')
                if not append_match({'path': name, 'line': int(number), 'text': '', 'text_truncated': True}):
                    break
            result['coverage']['files_searched'] += len(batch)
            recovered_output_limit('match')
            return {**result, 'status': 'PARTIAL', 'truncated': True, 'includes_location_only': True,
                'reason': 'search_match_locations_only: some matches contain path/line only after the output '
                          'bound; source text was not inspected. Read bounded excerpts; absence is unproven.'}
        if found['returncode'] not in (0, 1) or found['output_truncated']:
            return error('search_failed: '+found['output'][:1000])
        result['coverage']['files_searched'] += len(batch)
        # rg JSON may emit match rows before reporting a binary offset in its
        # end record. Inspect those bounded end records before returning even
        # the first page; an earlier match is not proof that a file is text.
        binary_files=set()
        for line in found['output'].splitlines():
            try:
                event=json.loads(line)
            except (ValueError,TypeError):
                return error('search_response_invalid')
            if event['type']=='end' and event['data'].get('binary_offset') is not None:
                name=event['data']['path'].get('text')
                if name is not None:
                    binary_files.add(Path(name).as_posix())
        if binary_files:
            result['coverage']['files_skipped'] += len(binary_files)
            result['coverage']['binary_files_skipped']=result['coverage'].get('binary_files_skipped',0)+len(binary_files)
        for line in found['output'].splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                return error('search_response_invalid')
            if event['type'] != 'match':
                continue
            data = event['data']
            if ('text' not in data['path'] or 'text' not in data['lines']
                    or '\0' in data['lines']['text']):
                result['truncated'] = True
                continue  # Do not turn binary/base64 content into source text.
            name = Path(data['path']['text']).as_posix()
            if name in binary_files:
                continue
            text = data['lines']['text'].rstrip('\r\n')
            clipped = text.encode()[:2048].decode('utf-8', errors='ignore')
            match = {'path': name, 'line': data['line_number'], 'text': clipped,
                     'text_truncated': clipped != text}
            if not append_match(match):
                return {**result, 'status': 'PARTIAL'}
    result['status'] = ('PARTIAL' if result['truncated'] else
                        'HIT' if result['matches'] else 'MISS')
    return result
