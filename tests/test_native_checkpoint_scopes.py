"""Working-source checkpoint scope and UTF-8 regressions."""
import json

import pytest

from compute_plane.native_checkpoint import operational_checkpoint
from tare_tools_agent_runtime import ConversationLog
from tests.test_native_checkpoint import record


def test_writable_scope_filters_excerpts_to_authorized_paths():
    """Only paths under a writable scope appear in working_source_excerpts."""
    log = ConversationLog()
    record(log, 'read', {'repository': 'r', 'path': 'src/a.py'},
           {'sha256': 'a' * 64, 'start': 1, 'end': 10, 'next_start': 11,
            'text': '1: def alpha():\n2:     return 1\n3: \n4: def beta():\n5:     return 2\n'
                    '6: \n7: def gamma():\n8:     return 3\n9: \n10: X = 42\n'})
    record(log, 'read', {'repository': 'r', 'path': 'src_extra/b.py'},
           {'sha256': 'b' * 64, 'start': 1, 'end': 10, 'next_start': 11,
            'text': '1: import os\n2: \n3: def helper():\n4:     pass\n5: \n'
                    '6: END = True\n7: \n8: Y = 99\n9: Z = 100\n10: W = 101\n'})
    record(log, 'read', {'repository': 'r', 'path': 'private/c.py'},
           {'sha256': 'c' * 64, 'start': 1, 'end': 10, 'next_start': 11,
            'text': '1: SECRET = "top-secret"\n2: TOKEN = "abc"\n3: \n4: def leak():\n5:     return SECRET\n'
                    '6: \n7: END = True\n8: Y = 99\n9: Z = 100\n10: W = 101\n'})
    value = json.loads(operational_checkpoint(log.snapshot(), writable={'r': ['src']}))
    excerpts = value.get('working_source_excerpts', [])
    assert [e['path'] for e in excerpts] == ['src/a.py']
    raw = json.dumps(value)
    assert 'top-secret' not in raw
    assert 'import os' not in raw
    # Other source identities remain as read_sources metadata
    rs_paths = [s['path'] for s in value['read_sources']]
    assert 'src/a.py' in rs_paths
    assert 'src_extra/b.py' in rs_paths
    assert 'private/c.py' in rs_paths


def test_multibyte_source_body_stays_bounded():
    """A long multibyte source body plus metadata receipts fits within 8192 bytes."""
    log = ConversationLog()
    # Several read metadata receipts from non-writable paths
    for i in range(5):
        record(log, 'read', {'repository': 'r', 'path': f'meta/{i}.json'},
               {'sha256': f'{i:02x}' * 32, 'start': 1, 'end': 50, 'next_start': 51,
                'text': f'{{"key_{i}": "value_{i}"}}\n'})
    # Long multibyte source body (accented + CJK characters, 200 numbered lines)
    lines = []
    for i in range(1, 201):
        lines.append(f'{i}: accented éèêë ñ ü ö 中文テキスト日本語测试 line {i}\n')
    body = ''.join(lines)
    record(log, 'read', {'repository': 'r', 'path': 'src/multibyte.py'},
           {'sha256': 'd' * 64, 'start': 1, 'end': 200, 'next_start': 201,
            'text': body})
    raw = operational_checkpoint(log.snapshot(), writable={'r': ['src']})
    assert len(raw.encode('utf-8')) <= 8192
    value = json.loads(raw)
    excerpts = value.get('working_source_excerpts', [])
    assert len(excerpts) == 1
    latest = excerpts[0]
    assert latest['path'] == 'src/multibyte.py'
    assert latest['excerpt_only'] is True
    # Nonempty whole-line prefix of the latest writable source
    assert latest['text']
    assert body.startswith(latest['text'])
    excerpt_lines = latest['text'].splitlines(keepends=True)
    assert len(excerpt_lines) >= 1
    for line in excerpt_lines:
        assert line.endswith('\n')
    # Preserves the actual first line from the source
    assert excerpt_lines[0].startswith('1: accented ')


@pytest.mark.parametrize('status', ['succeeded', 'failed', 'unknown'])
def test_terminal_mutation_clears_excerpts_and_read_restores(status):
    """A terminal mutation clears all working excerpts; a subsequent read restores them."""
    log = ConversationLog()
    record(log, 'read', {'repository': 'r', 'path': 'src/a.py'},
           {'sha256': 'a' * 64, 'start': 1, 'end': 10, 'next_start': 11,
            'text': '1: def hello():\n2:     print("hi")\n3: \n4: world = "there"\n5: \n'
                    '6: X = 1\n7: Y = 2\n8: Z = 3\n9: W = 4\n10: END = True\n'})
    view = lambda: json.loads(operational_checkpoint(log.snapshot(), writable={'r': ['src']}))
    assert view()['working_source_excerpts']
    record(log, 'terminal_start', {}, {'returncode': 0}, status=status)
    assert 'working_source_excerpts' not in view()
    # A subsequent real read restores them
    record(log, 'read', {'repository': 'r', 'path': 'src/a.py'},
           {'sha256': 'a' * 64, 'start': 1, 'end': 10, 'next_start': 11,
            'text': '1: def hello():\n2:     print("hi")\n3: \n4: world = "there"\n5: \n'
                    '6: X = 1\n7: Y = 2\n8: Z = 3\n9: W = 4\n10: END = True\n'})
    assert view()['working_source_excerpts']
