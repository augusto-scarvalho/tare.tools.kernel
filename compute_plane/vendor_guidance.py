"""What each vendor CLI adds to tare's requests on its own, read from the installed CLI, and the owner's choices.

Tare replaces each CLI's main prompt, yet the CLI still adds blocks of its own. `read()` asks the installed CLI what
it adds now, with zero tokens: Claude Code runs with tare's own flags against a recorder on this machine
(127.0.0.1) that answers every request with an error, so the login token never leaves the machine and nothing
reaches the provider; Codex renders the model's input with its own `debug prompt-input`, which sends nothing.
`read_real()` makes one minimal real call, only when asked by name: Claude Code with tare's mod recording every
piece, AGY from its own record of that call. Machine and day values become placeholders, so two readings of one CLI
give the same texts. A choice maps an item
id to None (leave the block out) or a text sent in its place; VendorCliClient applies it and refuses, before
anything is sent, one its CLI cannot honour. `transport_texts()` shows the protocol texts tare itself sends.
"""
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .managed_process import run_pipe_process
from .native_models import encode, sha

# Claude Code runs this hooks module from --plugin-dir (2.1.322: `prompt.context` and `prompt.attachment`). A hook
# that fails passes the vendor text through, so a removal is not proved per request (docs/VENDOR_GUIDANCE.md).
CLAUDE_MOD = '''import type { Register } from 'claude-code';

// Written by tare for one call. Context blocks to leave out, and attachment type -> null (leave it out) or the
// text to send instead. A reading also marks every block with its name so tare can tell them apart.
const dropContext: string[] = __TARE_DROP__;
const attachments: Record<string, string | null> = __TARE_ATTACHMENTS__;
const reading: boolean = __TARE_READING__;
// A real reading: every piece, as sent, goes to its own file (hooks run concurrently; writes to one file interleave).
const recording: boolean = __TARE_RECORDING__;
let pieces = 0;
// Placeholder -> pattern whose first group is that value in the CLI's own text for the same block.
const fill: [string, string][] = __TARE_FILL__;
// An empty block stays empty: a mark would make Claude Code send a block it otherwise leaves out.
const mark = (kind: string, name: string, text: string) =>
  reading && text ? `<<tare ${kind} ${name}>>${text}<</tare ${kind}>>` : text;

export const register: Register = (on) => {
  on('prompt.context', async ($, e, next) => {
    const blocks = (await next(e)).blocks.filter((b) => !dropContext.includes(b.name));
    if (recording) await $.fs.write(`guidance-record/${++pieces}.json`, JSON.stringify({ kind: 'context', blocks }));
    return { blocks: blocks.map((b) => ({ ...b, text: mark('context', b.name, b.text) })) };
  });
  on('prompt.attachment', async ($, e, next) => {
    const own = Object.hasOwn(attachments, e.type) && attachments[e.type] === null ? null : (await next(e)).text;
    let text = Object.hasOwn(attachments, e.type) ? attachments[e.type] : own;
    if (text !== null && text !== own)  // the owner's text: its placeholders take the values the CLI wrote here
      for (const [name, pattern] of fill) {
        const found = (own ?? '').match(new RegExp(pattern));
        if (found) text = text.split(name).join(found[1]);
      }
    if (recording) await $.fs.write(`guidance-record/${++pieces}.json`,
                                    JSON.stringify({ kind: 'attachment', type: e.type, text }));
    return { text: text === null ? null : mark('attachment', e.type, text) };
  });
};
'''
# The owner edits a block as tare shows it, with placeholders (`normalized`). When the edited text is sent, the mod
# puts back the values Claude Code wrote in that same block; a placeholder with no value there goes out as written.
FILL = (('<date>', r'(\d{4}-\d{2}-\d{2})'), ('<OS version>', r'OS Version: (.+)'),
        ('<working directory>', r'Primary working directory: (.+)'),
        ('<model name>', r'named (.+?)\. The exact model ID'), ('<model id>', r'The exact model ID is (.+?)\. Assistant'),
        ('<knowledge cutoff>', r'Assistant knowledge cutoff is (.+?)\.'),
        ('<account email>', r'([\w.+-]+@[\w-]+(?:\.[\w-]+)+)'), ('<tokens>', r'(\d[\d,]*) tokens left'))
# Context blocks render inside an attachment (2.1.322: `session_context`), so each kind has its own end mark.
MARKS = {kind: re.compile(rf'<<tare {kind} ([^>\s]+)>>(.*?)<</tare {kind}>>', re.S)
         for kind in ('context', 'attachment')}
# The first release measured. An older one may load the folder without running its hooks module, which would
# leave a choice silently unapplied, so the call is refused instead.
CLAUDE_GUIDANCE_MINIMUM = '2.1.322'
# Codex can leave these blocks out (`-c <key>=false`), never rewrite them; a block it shows under another leading
# tag can only be read. The tag is how `codex debug prompt-input` shows the block.
CODEX_SWITCHES = {'skills_instructions': 'skills.include_instructions',
                  'permissions instructions': 'include_permissions_instructions',
                  'collaboration_mode': 'include_collaboration_mode_instructions',
                  'environment_context': 'include_environment_context'}
NOTES = {
    'cli://claude': 'Read by starting Claude Code with tare\'s own settings, pointed at a recorder on this machine '
                    'that refuses every request: the login token stays inside the machine and nothing reaches the '
                    'provider. Removing and replacing work through a small mod tare loads for the call; it needs '
                    'Claude Code\'s hooks on, so while a Claude choice is saved, calls that tare normally runs with '
                    'hooks off also run the hooks of the plugins built into Claude Code. If the mod fails, Claude '
                    'Code sends its own text anyway.',
    'cli://codex': 'Read with Codex\'s own `debug prompt-input`, which sends nothing, as tare\'s real call would '
                   'show it: the skills installed here, but neither your login nor your config.toml. Codex can '
                   'leave a block out but not rewrite it; tare proves each removal once per Codex version.',
    'cli://agy': 'Read from the record AGY keeps of one small real call tare makes when a new AGY version appears. '
                 'Nothing here can be changed.'}
READING_PROMPT = 'Tare reads what this CLI adds on its own.'
REAL_PROMPT = 'Reply with the text OK and nothing else.'
# What one real AGY 1.5.7 call showed on 2026-11-12 (read_real), until a reading of the installed AGY replaces it.
AGY_STARTING_POINT = {'cli': 'AGY', 'cli_version': '1.5.7', 'read_at': '2026-11-12', 'items': [
    {'id': 'system:frame',
     'title': "Frame around tare's system text",
     'sent_as': 'system text',
     'text': "<System Prompt>\n<tare's system text>\n</System Prompt>"},
    {'id': 'request:frame',
     'title': "Frame around tare's request",
     'sent_as': 'user message',
     'text': "<USER_REQUEST>\n<tare's request>\n</USER_REQUEST>"},
    {'id': 'request:ADDITIONAL_METADATA',
     'title': "Block <ADDITIONAL_METADATA> after tare's request",
     'sent_as': "user message, after tare's request",
     'text': 'The current local time is: <local time>.'},
    {'id': 'request:USER_SETTINGS_CHANGE',
     'title': "Block <USER_SETTINGS_CHANGE> after tare's request",
     'sent_as': "user message, after tare's request",
     'text': 'The user changed setting `Model Selection` from None to <model name>. No need to comment on this change '
             "if the user doesn't ask about it. If reporting what model you are, please use a human readable name "
             'instead of the exact string.'},
    {'id': 'tool:finish',
     'title': 'Tool "finish"',
     'sent_as': 'tool definition',
     'text': 'End your task and return its result. Only call this once you are confident the task is complete and '
             'correct.'},
    {'id': 'tool:manage_task',
     'title': 'Tool "manage_task"',
     'sent_as': 'tool definition',
     'text': "Manage the background tasks of this conversation: 'list' the running tasks, check a task's 'status' and "
             "recent log output, 'send_input' to a running task, or 'kill' one. Refer to tasks by the TaskId that "
             "'list' reports; keep task references human-readable when talking to the user.\n"
             '\n'
             'Parameters: '
             '{"$schema":"https://json-schema.org/draft/2020-12/schema","properties":{"Action":{"type":"string","enum":["list","kill","status","send_input"],"description":"\'list\' '
             'reports the running tasks; the other actions operate on the task named in '
             'TaskId."},"TaskId":{"type":"string","description":"Task to manage, as reported by \'list\'. Required '
             'for \'kill\', \'status\' and \'send_input\'."},"Input":{"type":"string","description":"Input to deliver '
             'to the task. Required for '
             '\'send_input\'."}},"additionalProperties":false,"type":"object","required":["Action"]}'}]}
EMAIL = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')


def controls(endpoint, item_id):
    """What the owner may do with an item, from the shape of its id: Claude context blocks can be left out,
    attachments left out or rewritten; Codex blocks with a known switch left out; everything else only read."""
    if endpoint == 'cli://claude':
        kind, _, name = item_id.partition(':')
        return {'context': ('remove',), 'attachment': ('remove', 'replace')}.get(kind, ()) if name else ()
    if endpoint == 'cli://codex':
        return ('remove',) if item_id in CODEX_SWITCHES else ()
    if endpoint == 'cli://agy':
        return ()
    raise ValueError(f'no vendor guidance for {endpoint!r}')


def validate(endpoint, choices):
    """The owner's choices for one CLI as {item id: None (leave out) | text (send instead)}; ValueError for a
    control the item does not offer or an empty replacement."""
    for key, value in (choices or {}).items():
        control = 'remove' if value is None else 'replace' if isinstance(value, str) and value.strip() else None
        if control not in controls(endpoint, key):
            raise ValueError(f'{endpoint} cannot {control or "send an empty"} {key!r}')
    controls(endpoint, '')  # an unknown endpoint is refused even without a choice
    return dict(choices or {})


def receipt(choices):
    """What a call applied, for its transport receipt: each item's control and the digest of the exact choices."""
    return {'choices': {k: 'remove' if v is None else 'replace' for k, v in sorted(choices.items())},
            'sha256': sha(choices)}


def claude_mod(root, choices, *, tag=False, record=False):
    """Write the mod that applies validated Claude choices to one call (for a local reading it also marks every
    block, for a real reading it records every piece as sent); returns the folder for --plugin-dir."""
    drop = sorted(k.partition(':')[2] for k in choices if k.startswith('context:'))
    attachments = {k.partition(':')[2]: v for k, v in choices.items() if k.startswith('attachment:')}
    folder = root/'tare-guidance'
    (folder/'.claude-plugin').mkdir(parents=True)
    (folder/'hooks').mkdir()
    (folder/'.claude-plugin/plugin.json').write_text(encode({
        'name': 'tare-guidance', 'version': '1.0.0',
        'description': 'Applies the owner\'s choices about the blocks Claude Code adds on its own.'}), encoding='utf-8')
    (folder/'hooks/hooks.json').write_text(encode({'modules': ['./register.ts']}), encoding='utf-8')
    (folder/'hooks/register.ts').write_text(
        CLAUDE_MOD.replace('__TARE_DROP__', json.dumps(drop)).replace('__TARE_READING__', json.dumps(tag))
        .replace('__TARE_RECORDING__', json.dumps(record))
        .replace('__TARE_FILL__', json.dumps(FILL))
        .replace('__TARE_ATTACHMENTS__', json.dumps(attachments, sort_keys=True)), encoding='utf-8')
    return folder


def codex_switches(choices):
    """{item id: (config key that leaves the block out, the tag that shows it in `codex debug prompt-input`)}."""
    return {k: (CODEX_SWITCHES[k], '<' + k + '>') for k in choices}


def normalized(text, roots=()):
    """Machine and day values as placeholders, so two readings of the same CLI give the same text."""
    text = text.replace('\r\n', '\n')
    for root in roots:
        for form in {str(root), Path(root).as_posix()}:
            text = re.sub(re.escape(form) + r'[^\s<>`"]*', '<working directory>', text)
    text = EMAIL.sub('<account email>', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}T[\d:.]+(?:[+-]\d{2}:\d{2}|Z)?', '<local time>', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '<date>', text)
    text = re.sub(r'\d[\d,]* tokens left', '<tokens> tokens left', text)
    text = re.sub(r'(OS Version: ).+', r'\1<OS version>', text)
    text = re.sub(r'named .+?\. The exact model ID is .+?\. Assistant knowledge cutoff is .+?\.',
                  'named <model name>. The exact model ID is <model id>. Assistant knowledge cutoff is '
                  '<knowledge cutoff>.', text)
    text = re.sub(r'cc_version=[^;\s]+;', 'cc_version=<CLI version>;', text)
    text = re.sub(r'<timezone>[^<]*</timezone>', '<timezone><time zone></timezone>', text)
    return re.sub(r'(`r\d+` = `)[^`]*`', r'\1<skills folder>`', text)


def claude_items(recorded, body, roots=()):
    """The blocks of one recorded Claude request that tare did not write: context blocks and attachments marked by
    the reading mod (ids `context:<name>`, `attachment:<type>`), other system blocks (`system:<n>`), text around the
    marked blocks (`message:<n>`) and tool descriptions other than tare's (`tool:<name>`)."""
    from .native_cli_claude import input_record
    own = {body['system']} | {b['text'] for b in input_record(body)['message']['content']}
    items = []

    def add(item_id, title, sent_as, text):
        items.append({'id': item_id, 'title': title, 'sent_as': sent_as,
                      'controls': controls('cli://claude', item_id), 'text': normalized(text, roots)})

    def unmarked(text, kinds=('context', 'attachment')):
        for kind in kinds:
            text = MARKS[kind].sub(lambda m, kind=kind: f'<{kind} {m[1]}>', text)
        return text
    system = [b.get('text', '') for b in recorded.get('system') or [] if b.get('text') not in own]
    for number, text in enumerate(system, 1):
        add(f'system:{number}', 'System text: ' + normalized(text, roots)[:48],
            f'system block {number}, before tare\'s system text', text)
    around, after = 0, False
    for message in recorded.get('messages') or []:
        content = message['content'] if isinstance(message['content'], list) else [{'text': message['content']}]
        for part in content:
            text = part.get('text')
            if not isinstance(text, str) or text in own:
                after = after or text in own
                continue
            where = f'{message["role"]} message {"after" if after else "before"} tare\'s records'
            for name, inner in MARKS['context'].findall(text):
                add(f'context:{name}', f'Context block "{name}"', where, inner)
            for name, inner in MARKS['attachment'].findall(text):
                add(f'attachment:{name}', f'Attachment "{name}"', where, unmarked(inner, ('context',)))
            if re.sub(r'<(context|attachment) [^>\s]+>', '', unmarked(text)).strip():
                around += 1
                add(f'message:{around}', 'Text around Claude Code\'s blocks', where, unmarked(text))
    for tool in recorded.get('tools') or []:
        if not str(tool.get('name', '')).startswith('mcp__tare__'):
            add(f'tool:{tool.get("name")}', f'Tool description "{tool.get("name")}"', 'tool description',
                tool.get('description', ''))
    return items


def codex_items(rendered, roots=()):
    """The blocks of `codex debug prompt-input` other than tare's own message; each id is the block's leading tag."""
    items = []
    for entry in json.loads(rendered):
        for part in entry.get('content') if isinstance(entry.get('content'), list) else []:
            text = (part.get('text') or '').replace('\r\n', '\n')
            if text == READING_PROMPT:
                continue
            tag = re.match(r'<([a-z][a-z_ ]*)>', text)
            item_id = tag[1] if tag else f'block:{len(items) + 1}'
            items.append({'id': item_id, 'title': f'Block <{item_id}>' if tag else f'Untagged block {len(items) + 1}',
                          'sent_as': f'{entry.get("role")} message', 'controls': controls('cli://codex', item_id),
                          'text': normalized(text, roots)})
    return items


def _control(seconds):
    from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl
    return ExecutionControl(BindingRef('vendor-guidance-reading', 'local', 'diagnostic-no-authority', 1, 'fixture'),
                            CancellationToken(), time.monotonic() + seconds)


def _version(executable):
    physical, raw, _ = run_pipe_process([executable, '--version'], cwd=Path.cwd(), timeout=10, max_output=8192)
    found = re.findall(r'(?<![\w.])\d+\.\d+\.\d+(?![\w.+-])', raw.decode('utf-8', 'replace'))
    if physical.get('exit_code') != 0 or len(found) != 1:
        raise ValueError(f'{executable} did not report its version')
    return found[0]


def read(endpoint, executable, *, choices=None, seconds=120):
    """What the installed CLI adds to tare's requests now: {'cli', 'cli_version', 'note', 'items': [{'id', 'title',
    'sent_as', 'controls', 'text'}]}. With choices, what it adds with them applied. ValueError when it cannot be read."""
    if endpoint == 'cli://claude':
        return _read_claude(executable, choices or {}, seconds)
    if endpoint == 'cli://codex':
        return _read_codex(executable, choices or {}, seconds)
    controls(endpoint, '')
    return {'cli': 'AGY', 'cli_version': None, 'note': NOTES[endpoint], 'items': []}


def _read_claude(executable, choices, seconds):
    from .native_cli import claude_invocation, cli_env, cli_request, version_tuple
    version = _version(executable)
    if version_tuple(version) < version_tuple(CLAUDE_GUIDANCE_MINIMUM):
        raise ValueError(f'Claude Code {version} cannot be read; tare reads {CLAUDE_GUIDANCE_MINIMUM} or newer.')
    bodies = []

    class Recorder(BaseHTTPRequestHandler):
        def do_POST(self):
            # Content-Length only frames the body: no other header is read, and none is kept.
            raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
            if self.path.split('?')[0].endswith('/v1/messages'):
                try:
                    bodies.append(json.loads(raw))
                except ValueError:
                    pass
            reply = b'{"type":"error","error":{"type":"invalid_request_error","message":"tare reading only"}}'
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix='tare-reading-') as folder:
            root = Path(folder)
            # A call without tools, like heads-up notes and plain answers; on 2.1.322 the blocks Claude Code adds
            # were the same in every request shape tare uses.
            body = cli_request({'model': 'claude-sonnet-5-5', 'tools': [],
                                'messages': [{'role': 'user', 'content': READING_PROMPT}]}, native=True)
            env = cli_env()
            env['ANTHROPIC_BASE_URL'] = f'http://127.0.0.1:{server.server_address[1]}'
            argv, source, _ = claude_invocation(executable, root, body, env, validate('cli://claude', choices),
                                                tag=True)
            run_pipe_process(argv, cwd=root, env=env, timeout=seconds, stdin=source.encode('utf-8'),
                             max_output=4*1024*1024, on_stdout=lambda chunk: None, completed=lambda: bool(bodies))
            if not bodies:
                raise ValueError('Claude Code sent no request to read.')
            items = claude_items(bodies[0], body, [root])
    finally:
        server.shutdown()
        server.server_close()
    return {'cli': 'Claude Code', 'cli_version': version, 'note': NOTES['cli://claude'], 'items': items}


def _read_codex(executable, choices, seconds):
    from .native_cli import cli_request
    from .native_cli_codex import _inspect, discover, prompt_input
    version, control, receipt = _version(executable), _control(seconds), {}
    with tempfile.TemporaryDirectory(prefix='tare-reading-') as folder:
        root = Path(folder)
        features = discover(executable, root, control, run_pipe_process, receipt)
        home = root/'models'
        home.mkdir()
        models = json.loads(_inspect(executable, ['debug', 'models'], home, dict(os.environ, CODEX_HOME=str(home)),
                                     control, run_pipe_process, receipt, seconds=30, max_output=8*1024*1024))
        model = next(m for m in models['models'] if str(m.get('slug', '')).startswith('gpt'))
        body = cli_request({'model': model['slug'], 'messages': [{'role': 'user', 'content': 'reading'}]},
                           structured=True, native_arguments=True)
        rendered = prompt_input(executable, root, body, model, features, validate('cli://codex', choices), control,
                                run_pipe_process, receipt)
        items = codex_items(rendered, [root])
    return {'cli': 'Codex', 'cli_version': version, 'note': NOTES['cli://codex'], 'items': items}


def read_real(endpoint, executable, *, model, choices=None, seconds=180):
    """What the CLI adds in one minimal real call. It spends tokens, so it is never part of `read`: the caller names
    the cheapest model and makes it once per CLI version. Returns a reading as `read` does, plus the call's
    'usage'. Claude Code: tare's own command and mod, which records every piece as sent; each item says whether the
    zero-token reading found it (`local`: same, differs, absent) and `left_out` names what a choice removed. AGY: its
    own record of that call. ValueError when the CLI cannot be read."""
    if endpoint == 'cli://claude':
        return _real_claude(executable, model, validate(endpoint, choices), seconds)
    if endpoint == 'cli://agy':
        return _real_agy(executable, model, seconds)
    raise ValueError(f'{endpoint} has no real reading: its zero-token reading matched a real call.')


def _real_claude(executable, model, choices, seconds):
    from .native_cli import claude_invocation, cli_env, cli_request
    local = _read_claude(executable, choices, seconds)  # also refuses a CLI too old to be read
    known = {item['id']: item for item in local['items']}
    with tempfile.TemporaryDirectory(prefix='tare-reading-') as folder:
        root = Path(folder)
        body = cli_request({'model': model, 'effort': 'low', 'tools': [],
                            'messages': [{'role': 'user', 'content': REAL_PROMPT}]}, native=True)
        env = cli_env()
        argv, source, _ = claude_invocation(executable, root, body, env, choices, record=True)
        _, raw, _ = run_pipe_process(argv, cwd=root, env=env, timeout=seconds, stdin=source.encode('utf-8'),
                                     max_output=4*1024*1024)
        result = {}
        for line in raw.decode('utf-8', 'replace').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get('type') == 'result':
                result = row
        files = sorted((root/'guidance-record').glob('*.json'), key=lambda f: int(f.stem))
        records = [json.loads(f.read_text(encoding='utf-8')) for f in files]
    if result.get('is_error') is not False or not records:
        raise ValueError(f'The real Claude Code call failed ({result.get("subtype") or "no result"}).')
    # A context block is sent inside an attachment that frames it (2.1.322: session_context). One the engine replaces
    # with its own attachment (currentDate: the date attachment, same text) is not sent as a block.
    sent = [r['text'] for r in records if r['kind'] == 'attachment' and r['text']]
    blocks = {b['name']: b['text'] for r in records if r['kind'] == 'context' for b in r['blocks']
              if b['text'] and any(b['text'] in text and b['text'] != text for text in sent)}
    items, left_out = [], []

    def add(item_id, title, text):
        if any(i['id'] == item_id for i in items):
            return
        text = normalized(text, [root])
        mine = known.get(item_id)
        items.append({'id': item_id, 'title': title, 'controls': controls('cli://claude', item_id), 'text': text,
                      'sent_as': mine['sent_as'] if mine else 'sent with the request; seen only in a real call',
                      'local': 'absent' if mine is None else 'same' if mine['text'] == text else 'differs'})
    for record in records:
        if record['kind'] == 'context':
            for name in (b['name'] for b in record['blocks'] if b['name'] in blocks):
                add(f'context:{name}', f'Context block "{name}"', blocks[name])
        elif record['text'] is None:
            left_out.append(f'attachment:{record["type"]}')
        else:  # a context block renders inside an attachment: shown there by name, as the local reading does
            text = record['text']
            for name, block in blocks.items():
                text = text.replace(block, f'<context {name}>')
            add(f'attachment:{record["type"]}', f'Attachment "{record["type"]}"', text)
    return {'cli': 'Claude Code', 'cli_version': local['cli_version'], 'note': NOTES['cli://claude'], 'items': items,
            'left_out': left_out, 'usage': {**(result.get('usage') or {}),
                                            'total_cost_usd': result.get('total_cost_usd')}}


def _protobuf(data):
    """[(field number, value)] of a protobuf message (length-delimited values as bytes), or None when it is not one.
    AGY publishes no schema, so its record is read by the marks in its text, never by field numbers."""
    fields, at = [], 0
    try:
        while at < len(data):
            key, shift = 0, 0
            while True:
                key |= (data[at] & 0x7F) << shift
                at, shift = at + 1, shift + 7
                if not data[at - 1] & 0x80:
                    break
            number, kind = key >> 3, key & 7
            if not 0 < number < 5000:
                return None
            if kind == 0:
                while data[at] & 0x80:
                    at += 1
                at += 1
                continue
            if kind in (1, 5):
                at += 8 if kind == 1 else 4
                continue
            if kind != 2:
                return None
            size, shift = 0, 0
            while True:
                size |= (data[at] & 0x7F) << shift
                at, shift = at + 1, shift + 7
                if not data[at - 1] & 0x80:
                    break
            if at + size > len(data):
                return None
            fields.append((number, data[at:at + size]))
            at += size
        return fields
    except IndexError:
        return None


def agy_items(record, system, roots=()):
    """The blocks AGY added in one generation record (gen_metadata.data of its conversation store): the frame
    around tare's system text, the blocks after tare's request (`request:<TAG>`) and its tools (`tool:<name>`)."""
    texts, tools = [], []

    def walk(data):
        fields = _protobuf(data) or []
        strings = {}
        for number, value in fields:
            try:
                strings.setdefault(number, value.decode('utf-8'))
            except UnicodeDecodeError:
                pass
        if (re.fullmatch(r'[a-z_]+', strings.get(1, '')) and 2 in strings
                and strings.get(3, '').startswith('{')):  # a tool: name, description, JSON schema
            tools.append((strings[1], strings[2], strings[3]))
        for number, value in fields:
            text = strings.get(number)
            if text is not None and ('<System Prompt>' in text or '<USER_REQUEST>' in text):
                texts.append(text.replace('\r\n', '\n'))
            elif len(value) > 2:
                walk(value)
    walk(record)
    framed = next((t for t in texts if '<System Prompt>' in t and system.replace('\r\n', '\n') in t), None)
    request = next((t for t in texts if '</USER_REQUEST>' in t), None)
    if framed is None or request is None:
        raise ValueError('AGY could not be read: its record of the call lacks tare\'s system text or request.')
    asked, added = request.split('</USER_REQUEST>', 1)
    items = [{'id': 'system:frame', 'title': 'Frame around tare\'s system text', 'sent_as': 'system text',
              'text': framed.replace(system.replace('\r\n', '\n'), '<tare\'s system text>')},
             {'id': 'request:frame', 'title': 'Frame around tare\'s request', 'sent_as': 'user message',
              'text': asked.split('<USER_REQUEST>', 1)[0] + '<USER_REQUEST>\n<tare\'s request>\n</USER_REQUEST>'}]
    # The model the call chose is named in the settings block: a placeholder, so readings with any model agree.
    added = re.sub(r'(`Model Selection` from .+? to ).+?(\. )', r'\1<model name>\2', added)
    for tag, inner in re.findall(r'<([A-Z_]+)>\n?(.*?)\n?</\1>', added, re.S):
        items.append({'id': f'request:{tag}', 'title': f'Block <{tag}> after tare\'s request',
                      'sent_as': 'user message, after tare\'s request', 'text': inner})
    for name, description, schema in dict.fromkeys(tools):
        items.append({'id': f'tool:{name}', 'title': f'Tool "{name}"', 'sent_as': 'tool definition',
                      # finish carries tare's response schema; only its description is AGY's
                      'text': description if name == 'finish' else f'{description}\n\nParameters: {schema}'})
    return [{**item, 'controls': (), 'text': normalized(item['text'], roots)} for item in items]


def _real_agy(executable, model, seconds):
    import shutil
    import sqlite3
    from .native_cli import AgyCompletion, agy_invocation, cli_env, cli_request
    version = _version(executable)
    with tempfile.TemporaryDirectory(prefix='tare-reading-') as folder:
        root = Path(folder)
        body = cli_request({'model': model, 'effort': 'low',
                            'messages': [{'role': 'user', 'content': REAL_PROMPT}]}, structured=True)
        argv, source = agy_invocation(executable, root/'call', body, seconds)
        completion = AgyCompletion(root/'call', model, True)
        _, raw, _ = run_pipe_process(argv, cwd=root/'call', env=cli_env(), timeout=seconds + 30,
                                     stdin=source.encode('utf-8'), max_output=4*1024*1024,
                                     on_stdout=completion.feed, completed=completion.complete)
        rows = []
        for line in raw.decode('utf-8', 'replace').splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        conversation = next((r.get('conversation_id') for r in rows if isinstance(r, dict)
                             and re.fullmatch(r'[0-9a-f-]{36}', str(r.get('conversation_id')))), None)
        result = next((r['result'] for r in rows if isinstance(r, dict) and isinstance(r.get('result'), dict)), {})
        # Only the conversation this call created, named by the id the call reported; never another one.
        store = Path.home()/'.gemini'/'antigravity-cli'/'conversations'/f'{conversation}.db'
        if conversation is None or not store.is_file():
            raise ValueError('AGY could not be read: the call left no record of its own.')
        shutil.copy(store, root/'conversation.db')  # AGY may still hold its own file
        database = sqlite3.connect(root/'conversation.db')
        try:
            found = database.execute('select data from gen_metadata order by idx limit 1').fetchone()
        finally:
            database.close()
        if not found:
            raise ValueError('AGY could not be read: the call left no record of its own.')
        items = agy_items(found[0], body['system'], [root])
    return {'cli': 'AGY', 'cli_version': version, 'note': NOTES['cli://agy'], 'items': items,
            'usage': result.get('usage')}


def transport_texts():
    """The protocol texts tare itself sends to vendor CLIs, read-only: each codec's system text, built by
    `cli_request` from a synthetic chat (placeholders in angle brackets), and the fixed reminders."""
    from .native_cli import AGY_ACTION_GUIDANCE, HAIKU_FORMAT_GUIDANCE, HAIKU_SEARCH_GUIDANCE, cli_request

    def chat():
        return {'model': '<model>', 'tools': [{'type': 'function', 'function': {
                    'name': 'read', 'description': '<tool description>', 'parameters': {'type': 'object'}}}],
                'messages': [{'role': 'system', 'content': '<role guidance>'}, {'role': 'user', 'content': '<task>'}]}
    codecs = (('claude-native', 'cli://claude', 'Claude: decisions as native tool calls', {'native': True}),
              ('claude-summary', 'cli://claude', 'Claude: conversation summaries', {'summary': True}),
              ('claude-toolless', 'cli://claude', 'Claude: toolless JSON decisions (rollback only)', {}),
              ('claude-structured', 'cli://claude', 'Claude: StructuredOutput decisions (rollback only)',
               {'structured': True, 'formatter': True}),
              ('agy-structured', 'cli://agy', 'AGY: decisions through its response schema', {'structured': True}),
              ('codex-structured', 'cli://codex', 'Codex: decisions through its output schema',
               {'structured': True, 'native_arguments': True}))
    return [{'id': key, 'title': title, 'endpoint': endpoint, 'text': cli_request(chat(), **flags)['system']}
            for key, endpoint, title, flags in codecs] + [
        {'id': 'agy-action-guidance', 'endpoint': 'cli://agy', 'text': AGY_ACTION_GUIDANCE,
         'title': 'AGY: reminder after the records when read is offered'},
        {'id': 'claude-haiku-format', 'endpoint': 'cli://claude', 'text': HAIKU_FORMAT_GUIDANCE,
         'title': 'Claude Haiku: StructuredOutput format guidance (rollback only)'},
        {'id': 'claude-haiku-search', 'endpoint': 'cli://claude', 'text': HAIKU_SEARCH_GUIDANCE,
         'title': 'Claude Haiku: search guidance (rollback only)'}]
