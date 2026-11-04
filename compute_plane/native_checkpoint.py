"""Small journal-derived tool facts, independent of a model's history summary."""
import json
from pathlib import PurePosixPath


def operational_checkpoint(snapshot, *, writable=None):
    from .native_terminal import MUTATIONS
    pending, edits, checks, hashes = {}, {}, {}, {}
    terminals = {}
    instructions = {}
    read_sources = {}
    working_reads = {}
    last_proposal = None
    last_change = 0
    failures = []
    for record in snapshot.records:
        event = record.event()
        if event['kind'] == 'model_returned':
            reply = event['reply']
            if reply.get('complete') and reply.get('proposes_delivery') and not reply.get('tool_calls'):
                text = reply.get('text', '')
                last_proposal = {'basis': 'Previous proposed answer copied from the journal; '
                    'a model claim, not independent verification or new authority.',
                    'text': text, 'text_truncated': False,
                    'receipt': {'sequence': record.sequence, 'sha256': record.sha256}}
                while len(json.dumps(last_proposal, ensure_ascii=False).encode()) > 4096:
                    last_proposal['text'] = last_proposal['text'][:len(last_proposal['text'])//2]
                    last_proposal['text_truncated'] = True
        if event['kind'] == 'effect_requested':
            pending[event['effect_id']] = event['tool_call']
        elif event['kind'] == 'effect_returned':
            result = event['result']
            call = pending.get(result['effect_id'])
            if call is None:
                continue
            args = json.loads(call['arguments_json'])
            output = json.loads(result['output_json'])
            for row in output.get('scoped_project_instructions', []):
                if row['text'] is not None:
                    instructions[(row['repository'], row['path'])] = row
            name = call['name']
            if name in MUTATIONS and result['status'] in {'succeeded', 'failed', 'unknown'}:
                working_reads.clear()
                # Arbitrary terminal effects may alter files without an edit receipt.
                last_change = record.sequence
            ref = {'sequence': record.sequence, 'sha256': record.sha256}
            if name in {'terminal_start', 'terminal_read', 'terminal_stop', 'terminal_eof', 'terminal_input'}:
                handle = output.get('handle') or args.get('handle')
                if isinstance(handle, str) and handle:
                    prior = terminals.get(handle, {})
                    streams = output.get('output', {})
                    texts = [streams[channel]['text'][-350:] for channel in ('stdout', 'stderr')
                             if isinstance(streams, dict) and isinstance(streams.get(channel), dict)
                             and isinstance(streams[channel].get('text'), str) and streams[channel]['text']]
                    excerpt = '\n'.join(texts)[-700:]
                    # An empty poll does not erase the last observed output, nor
                    # does a failed read without a process view restate its state.
                    if 'status' in output or 'exit_code' in output:
                        state = {'status': output.get('status', 'unknown'),
                                 'exit_code': output.get('exit_code'), 'receipt': ref,
                                 'tool': name}
                    else:
                        state = {k: prior[k] for k in ('status', 'exit_code', 'receipt', 'tool') if k in prior}
                    if 'receipt' in state:
                        terminals[handle] = {**state, 'handle': handle[:128],
                            'output_excerpt': excerpt or prior.get('output_excerpt', ''),
                            'output_receipt': ref if excerpt else prior.get('output_receipt'),
                            'excerpt_only': True}
            key = (args.get('repository'), args.get('path'))
            if name in {'edit', 'read'} and result['status'] == 'succeeded':
                current = output.get('sha256')
                if name == 'edit' or key in hashes and hashes[key] != current:
                    last_change = record.sequence
                    working_reads.pop(key, None)
                hashes[key] = current
                if name == 'read':
                    read_sources[key] = {'repository': key[0], 'path': key[1],
                        'file_sha256': current, 'receipt': ref}
                    if type(output.get('start')) is int and type(output.get('end')) is int:
                        read_sources[key].update(start=output['start'], end=output['end'],
                                                 next_start=output.get('next_start'))
                    if (writable and isinstance(output.get('text'), str)
                            and type(output.get('start')) is int and type(output.get('end')) is int
                            and any(PurePosixPath(key[1]).is_relative_to(scope)
                                    for scope in writable.get(key[0], []))):
                        # Keep only complete, numbered source lines from actual reads.
                        # The journal retains the full response; excerpts are historical.
                        text = ''
                        for line in output['text'].splitlines(keepends=True):
                            if len((text + line).encode('utf-8')) > 6000:
                                break
                            text += line
                        if text:
                            working_reads.setdefault(key, {})[(output['start'], output['end'])] = {
                                **read_sources[key], 'text': text,
                                'excerpt_only': text != output['text']}
                if name == 'edit':
                    read_sources.pop(key, None)
                    edits[key] = {'tool': name, 'repository': key[0], 'path': key[1],
                        'status': 'succeeded', 'file_sha256': current, 'receipt': ref}
            if name == 'verify':
                checks[args.get('name')] = {'tool': name, 'name': args.get('name'),
                    'status': result['status'], 'returncode': output.get('returncode'),
                    'output_excerpt': str(output.get('output', ''))[-700:],
                    'excerpt_only': True, 'receipt': ref}
            elif result['status'] != 'succeeded':
                failures.append({'tool': name, 'status': result['status'], 'receipt': ref,
                    'reason': str(output.get('reason', output.get('error_type', 'unknown')))[:256]})
    for check in checks.values():
        check['after_latest_recorded_change'] = check['receipt']['sequence'] > last_change
    rows = sorted([*edits.values(), *checks.values(), *terminals.values(), *failures[-2:]],
                  key=lambda row: row['receipt']['sequence'])
    total = len(edits) + len(checks) + len(terminals) + len(failures)
    rows = rows[-12:]
    sources = sorted(read_sources.values(), key=lambda row: row['receipt']['sequence'])[-12:]
    excerpts = sorted([r for ranges in working_reads.values() for r in ranges.values()],
                      key=lambda row: row['receipt']['sequence'])[-4:]
    while True:
        value = {'basis': 'Recorded tool receipts, not model claims or a fresh filesystem check. '
            'Successful edits attest application, not correctness. Verification applies only to its '
            'recorded invocation; later changes can invalidate it. Full output remains in the journal.',
            'observations': rows, 'omitted_observations': total - len(rows)}
        if read_sources:
            value.update(read_sources=sources, omitted_read_sources=len(read_sources)-len(sources))
        if last_proposal is not None:
            value['previous_proposed_answer'] = last_proposal
        if excerpts:
            value['working_source_excerpts'] = excerpts
            value['working_source_notice'] = (
                'Historical numbered source lines for authorized edit paths, not fresh filesystem checks. '
                'Reuse matching source context; read only missing ranges needed for the next edit. '
                'Do not restart repository discovery after compaction. File hashes still guard edits.')
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        if len(text.encode()) <= 8192:
            if instructions:
                value['scoped_project_instructions'] = [instructions[key] for key in sorted(instructions)]
                value['instruction_scope'] = ('Project guidance applies only inside each AGENTS.md directory; '
                    'deeper files take precedence within their subtree. Preserve these exact instructions across compaction.')
                text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            return text
        if len(excerpts) == 1 and len(excerpts[0]['text'].splitlines(keepends=True)) > 1:
            lines = excerpts[0]['text'].splitlines(keepends=True)
            excerpts[0] = {**excerpts[0], 'text': ''.join(lines[:len(lines)//2]), 'excerpt_only': True}
        elif excerpts:
            excerpts.pop(0)
        elif sources:
            sources.pop(0)
        elif rows:
            rows.pop(0)
