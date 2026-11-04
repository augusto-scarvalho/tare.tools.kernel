import json

from compute_plane.native_checkpoint import operational_checkpoint
from tare_tools_agent_runtime import ConversationLog


def record(log, tool, args, output, status='succeeded'):
    effect_id = str(log.snapshot().revision)
    log.append({'kind': 'effect_requested', 'effect_id': effect_id,
        'tool_call': {'name': tool, 'arguments_json': json.dumps(args)}})
    log.append({'kind': 'effect_returned', 'result': {'effect_id': effect_id,
        'status': status, 'output_json': json.dumps(output)}})


def test_checkpoint_keeps_real_failure_and_detects_edits_after_verification():
    log = ConversationLog()
    record(log, 'edit', {'repository':'r','path':'test.py'}, {'sha256':'a'*64})
    record(log, 'verify', {'name':'tests'}, {'returncode':2, 'output':'SyntaxError: unterminated string'}, 'failed')
    log.append({'kind':'model_returned', 'reply': {'text':'The editor corrupted my correct code'}})
    value = json.loads(operational_checkpoint(log.snapshot()))
    edit, check = value['observations']
    assert check['returncode'] == 2 and 'SyntaxError' in check['output_excerpt']
    assert check['after_latest_recorded_change']
    assert edit['file_sha256'] == 'a'*64
    assert 'corrupted' not in json.dumps(value)
    record(log, 'edit', {'repository':'r','path':'test.py'}, {'sha256':'b'*64})
    value = json.loads(operational_checkpoint(log.snapshot()))
    check = next(row for row in value['observations'] if row['tool']=='verify')
    assert not check['after_latest_recorded_change']
    assert check['receipt']['sha256'] == log.snapshot().records[check['receipt']['sequence'] - 1].sha256


def test_checkpoint_reports_omissions_and_does_not_copy_source_or_reasoning():
    log = ConversationLog()
    for i in range(30):
        record(log, 'edit', {'repository':'r', 'path':str(i), 'replace':'secret-source-dump'*2000}, {'sha256':'a'*64})
    raw = operational_checkpoint(log.snapshot())
    value = json.loads(raw)
    assert len(raw.encode()) <= 8192 and value['omitted_observations'] == 18
    assert len(value['observations']) == 12 and 'secret-source-dump' not in raw


def test_checkpoint_preserves_delivered_findings_and_source_identity_as_claims():
    log = ConversationLog()
    record(log, 'read', {'repository':'r', 'path':'graph.json'}, {'sha256':'a'*64, 'text':'private source body'})
    answer = 'FLOW: 24 nodes, 2 DONE, 22 NOT_DONE. Deferred work does not block daily use.'
    delivery = log.append({'kind':'model_returned', 'reply': {'text':answer, 'complete':True,
        'proposes_delivery':True, 'tool_calls':[], 'metadata_json':'{"reasoning":"private reasoning"}'}})
    log.append({'kind':'context_activated', 'summary':'Incorrect compressed claim: 113 FLOW nodes'})
    raw = operational_checkpoint(log.snapshot())
    value = json.loads(raw)
    assert value['previous_proposed_answer']['text'] == answer
    assert not value['previous_proposed_answer']['text_truncated']
    assert value['previous_proposed_answer']['receipt']['sha256'] == delivery.sha256
    assert 'model claim' in value['previous_proposed_answer']['basis']
    assert value['read_sources'][0]['file_sha256'] == 'a'*64
    assert all(s not in raw for s in ('113 FLOW', 'private source body', 'private reasoning'))
    assert len(raw.encode()) <= 8192


def test_checkpoint_bounds_large_answer_and_long_source_paths():
    log = ConversationLog()
    for i in range(20):
        record(log, 'read', {'repository':'r', 'path':str(i)+'x'*3500}, {'sha256':'a'*64})
    log.append({'kind':'model_returned', 'reply': {'text':'large answer '*2000, 'complete':True,
        'proposes_delivery':True, 'tool_calls':[]}})
    raw = operational_checkpoint(log.snapshot())
    value = json.loads(raw)
    assert len(raw.encode()) <= 8192 and value['previous_proposed_answer']['text_truncated']
    assert value['omitted_read_sources'] > 0


def test_opt_in_working_sources_preserve_ranges_and_invalidate_changed_files():
    log = ConversationLog()
    def read(path, start, end, sha='a'*64):
        record(log, 'read', {'repository':'r', 'path':path},
               {'sha256':sha, 'start':start, 'end':end, 'next_start':end+1,
                'text':f'{start}: source for {path}\n'})
    read('src/a.py', 1, 10); read('src/a.py', 11, 20); read('tests/helper.py', 1, 10)
    view = lambda: json.loads(operational_checkpoint(log.snapshot(), writable={'r':['src']}))
    assert [r['start'] for r in view()['working_source_excerpts']] == [1, 11]
    assert 'source for tests/helper.py' not in json.dumps(view())
    read('src/a.py', 11, 20, 'b'*64)
    assert len(view()['working_source_excerpts']) == 1
    assert view()['working_source_excerpts'][0]['file_sha256'] == 'b'*64
    record(log, 'edit', {'repository':'r','path':'src/a.py'}, {'sha256':'c'*64})
    assert 'working_source_excerpts' not in view()
    read('src/a.py', 1, 10, 'c'*64)
    record(log, 'terminal_start', {}, {'returncode':0})
    assert 'working_source_excerpts' not in view()


def test_working_source_payload_stays_bounded_and_default_keeps_metadata_only():
    log = ConversationLog()
    for i in range(10):
        record(log, 'read', {'repository':'r','path':f'src/{i}.py'},
            {'sha256':'a'*64, 'start':1, 'end':200, 'text':'1: source line\n'*2000})
    assert 'working_source_excerpts' not in json.loads(operational_checkpoint(log.snapshot()))
    raw = operational_checkpoint(log.snapshot(), writable={'r':['src']})
    value = json.loads(raw)
    assert len(raw.encode()) <= 8192
    assert value['working_source_excerpts']
    assert all(r['excerpt_only'] for r in value['working_source_excerpts'])
    assert value['working_source_excerpts'][-1]['path'] == 'src/9.py'


def test_terminal_receipts_keep_real_exit_and_invalidate_named_verification():
    log = ConversationLog()
    record(log, 'verify', {'name': 'unit'}, {'returncode': 0, 'output': 'passed'})
    record(log, 'terminal_start', {'argv': ['python', '-m', 'pytest']},
           {'handle': 'failed', 'status': 'running', 'exit_code': None})
    record(log, 'terminal_read', {'handle': 'failed'},
           {'handle': 'failed', 'status': 'exited', 'exit_code': 1,
            'output': {'stdout': {'text': '1 failed'}, 'stderr': {'text': 'SyntaxError'}}}, 'failed')
    record(log, 'terminal_start', {'argv': ['python', '-m', 'pytest']},
           {'handle': 'successful', 'status': 'running', 'exit_code': None})
    record(log, 'terminal_read', {'handle': 'successful'},
           {'handle': 'successful', 'status': 'exited', 'exit_code': 0,
            'output': {'stdout': {'text': '8 passed'}}})
    record(log, 'terminal_start', {'argv': ['python']},
           {'handle': 'still-running', 'status': 'running', 'exit_code': None})
    rows = json.loads(operational_checkpoint(log.snapshot()))['observations']
    by_handle = {row['handle']: row for row in rows if 'handle' in row}
    assert by_handle['failed']['exit_code'] == 1
    assert '1 failed' in by_handle['failed']['output_excerpt']
    assert 'SyntaxError' in by_handle['failed']['output_excerpt']
    assert by_handle['successful']['exit_code'] == 0 and by_handle['successful']['status'] == 'exited'
    assert by_handle['still-running']['status'] == 'running' and by_handle['still-running']['exit_code'] is None
    assert all(row['receipt']['sha256'] for row in by_handle.values())
    assert not next(row for row in rows if row['tool'] == 'verify')['after_latest_recorded_change']
    assert not any(row['tool'] == 'verify' and row['name'] != 'unit' for row in rows)
    record(log, 'edit', {'repository': 'r', 'path': 'changed.py'}, {'sha256': 'b'*64})
    assert not next(row for row in json.loads(operational_checkpoint(log.snapshot()))['observations']
                    if row['tool'] == 'verify')['after_latest_recorded_change']


def test_terminal_output_is_bounded_and_omissions_counted():
    log = ConversationLog()
    for index in range(30):
        record(log, 'terminal_start', {}, {'handle': str(index), 'status': 'running', 'exit_code': None})
        record(log, 'terminal_read', {'handle': str(index)},
               {'handle': str(index), 'status': 'exited', 'exit_code': 1,
                'output': {'stdout': {'text': 'X'*20000}}}, 'failed')
    raw = operational_checkpoint(log.snapshot())
    value = json.loads(raw)
    assert len(raw.encode()) <= 8192
    assert value['omitted_observations'] > 0
    assert all(len(row['output_excerpt']) <= 700 for row in value['observations'] if 'handle' in row)
    assert 'X'*1000 not in raw


def test_terminal_empty_final_poll_keeps_output_identity_and_failed_poll_keeps_state():
    log = ConversationLog()
    record(log, 'terminal_start', {}, {'handle': 'h', 'status': 'running', 'exit_code': None,
           'output': {'stdout': {'text': '3 passed'}}})
    first = next(row for row in json.loads(operational_checkpoint(log.snapshot()))['observations']
                 if row.get('handle') == 'h')
    record(log, 'terminal_read', {'handle': 'h'}, {'handle': 'h', 'status': 'exited',
           'exit_code': 0, 'output': {'stdout': {'text': ''}, 'stderr': {'text': ''}}})
    second = next(row for row in json.loads(operational_checkpoint(log.snapshot()))['observations']
                  if row.get('handle') == 'h')
    assert second['status'] == 'exited' and second['exit_code'] == 0
    assert second['output_excerpt'] == '3 passed'
    assert second['output_receipt'] == first['output_receipt']
    assert second['receipt'] != first['receipt']
    record(log, 'terminal_read', {'handle': 'h'}, {'handle': 'h', 'error_type': 'IOError'}, 'failed')
    third = next(row for row in json.loads(operational_checkpoint(log.snapshot()))['observations']
                 if row.get('handle') == 'h')
    assert third['status'] == 'exited' and third['exit_code'] == 0
    assert third['receipt'] == second['receipt'] and third['output_receipt'] == first['output_receipt']


def test_first_failed_terminal_poll_without_process_view_keeps_generic_failure():
    log = ConversationLog()
    record(log, 'terminal_read', {'handle': 'missing'},
           {'error_type': 'TerminalRefused'}, 'failed')
    value = json.loads(operational_checkpoint(log.snapshot()))
    assert not any(row.get('handle') == 'missing' for row in value['observations'])
    assert any(row['tool'] == 'terminal_read' and row['status'] == 'failed'
               for row in value['observations'])
