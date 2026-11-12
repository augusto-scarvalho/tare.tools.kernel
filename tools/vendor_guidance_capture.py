"""Re-measure what a vendor CLI adds to tare's requests, spending no tokens (docs/VENDOR_GUIDANCE.md).

  python tools/vendor_guidance_capture.py claude <claude executable> <native|plain|summary|toolless|structured> [choices]
  python tools/vendor_guidance_capture.py codex <codex executable> [choices]

`choices` is a JSON object of the owner's choices for that CLI, e.g. '{"user_email": null}'.

Claude: tare's own VendorCliClient runs the installed CLI with tare's flags, input and mod, but aimed at a loopback
recorder that answers 400, so nothing reaches the provider. The recorder reads only Content-Length, to frame the
body, and keeps only the path and JSON body: no request header is stored or printed. Codex: its own
`debug prompt-input` renders what the model would see; nothing is sent.

Prints, for each catalog item, whether the request carries it as catalogued (placeholders match any value), then
every block of the request with the account email replaced. Nothing is written to disk.
"""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute_plane import native_cli  # noqa: E402
from compute_plane.managed_process import run_pipe_process  # noqa: E402
from compute_plane.native_cli_codex import _inspect, discover, prompt_input, unproved_removals  # noqa: E402
from compute_plane.native_models import NativeTransportInterrupted  # noqa: E402
from compute_plane.vendor_guidance import catalog  # noqa: E402
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl, ModelFailure  # noqa: E402

PLACEHOLDER = re.compile(r'<(?:account email|working directory|OS version|date|model name|model id|knowledge cutoff'
                         r'|CLI version|short hash|skills folder|time zone)>')
EMAIL = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
CHAT = {'model': 'claude-sonnet-5-5', 'effort': 'high',
        'tools': [{'type': 'function', 'function': {'name': 'read', 'description': 'Read a file range.', 'parameters': {
            'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'],
            'additionalProperties': False}}}],
        'messages': [{'role': 'system', 'content': 'TARE-ROLE-GUIDANCE synthetic role text.'},
                     {'role': 'user', 'content': 'TARE-USER-MESSAGE synthetic request.'}]}
# plain: a Claude call without tools (heads-up notes, plain answers), the native codec with an empty catalog.
CODECS = {'native': {'native': True}, 'plain': {'native': True}, 'summary': {'summary': True}, 'toolless': {},
          'structured': {'structured': True, 'formatter': True}}


def control():
    return ExecutionControl(BindingRef('vendor-guidance-capture', 'local', 'diagnostic-no-authority', 1, 'fixture'),
                            CancellationToken(), time.monotonic() + 180)


def report(endpoint, text, choices, blocks):
    text = text.replace('\r\n', '\n')  # Codex on Windows sends part of its rules with CRLF; the catalog keeps LF
    for item in catalog()[endpoint]['items']:
        parts = re.split(f'({PLACEHOLDER.pattern})', item['text'])
        pattern = ''.join('.*?' if PLACEHOLDER.fullmatch(part) else re.escape(part) for part in parts)
        state = 'present as catalogued' if re.search(pattern, text) else 'absent or changed'
        chosen = ('' if item['id'] not in choices else '  [chosen: remove]' if choices[item['id']] is None else
                  '  [chosen: replace, replacement ' + ('present' if choices[item['id']] in text else 'ABSENT') + ']')
        print(f'{item["id"]:28} {state}{chosen}')
    print()
    for where, block in blocks:
        print(f'--- {where} ({len(block)} characters)\n{EMAIL.sub("<account email>", block)}')


def claude(executable, codec, choices):
    requests = []

    class Recorder(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length') or 0))  # the only header read
            try:
                requests.append((self.path.split('?')[0], json.loads(body)))
            except ValueError:
                requests.append((self.path.split('?')[0], None))
            reply = b'{"type":"error","error":{"type":"invalid_request_error","message":"capture only"}}'
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    streams = []

    def aimed(argv, **options):
        if '-p' not in argv:
            return run_pipe_process(argv, **options)
        # The inference call, unchanged but for its destination; a surrounding Claude Code session's variables go.
        options['env'] = {k: v for k, v in options['env'].items()
                          if k != 'CLAUDECODE' and not k.startswith(('CLAUDE_CODE_', 'ANTHROPIC_'))}
        options['env']['ANTHROPIC_BASE_URL'] = f'http://127.0.0.1:{server.server_address[1]}'
        result = run_pipe_process(argv, **options)
        streams.append(result[1])
        return result
    native_cli.run_pipe_process = aimed
    version = re.search(r'\d+\.\d+\.\d+', run_pipe_process([executable, '--version'], cwd=Path.cwd(),
                                                           timeout=10)[1].decode()).group()

    class Qualified(dict):  # the capture is the measurement: no qualification probe first
        def get(self, key, default=None):
            return {'status': 'ok'}
    client = native_cli.VendorCliClient('cli://claude', executable=executable, version=version,
                                        qualification=Qualified(), vendor_guidance=choices)
    try:
        chat = json.loads(json.dumps({**CHAT, 'tools': []} if codec == 'plain' else CHAT))
        client.post_observed('', native_cli.cli_request(chat, **CODECS[codec]), control())
    except (ModelFailure, NativeTransportInterrupted) as ended:  # the recorder refused the request
        receipt = getattr(ended, 'transport_observation', None) or {}
        print('Claude Code', version, codec, '| call ended with', getattr(ended, 'code', ended),
              '| receipt vendor_guidance:', receipt.get('vendor_guidance'))
    server.shutdown()
    rows = [json.loads(line) for raw in streams for line in raw.splitlines() if line.strip().startswith(b'{')]
    init = next((r for r in rows if r.get('type') == 'system' and r.get('subtype') == 'init'), {})
    print('plugins listed at init:', [p.get('name') for p in init.get('plugins') or []])
    body = next((b for path, b in requests if path.endswith('/v1/messages') and b), None)
    if body is None:
        sys.exit('no request was recorded (refused before dispatch?)')
    blocks = [('system', s.get('text', '')) for s in body.get('system', [])]
    blocks += [(f'{m["role"]} message', b.get('text', '')) for m in body['messages']
               for b in (m['content'] if isinstance(m['content'], list) else [{'text': m['content']}])]
    blocks += [(f'tool {t.get("name")}', t.get('description', '')) for t in body.get('tools', [])]
    report('cli://claude', '\n'.join(text for _, text in blocks), choices, blocks)
    return body  # for a maintainer's own comparison in memory; nothing is written to disk


def codex(executable, choices):
    live = control()
    with tempfile.TemporaryDirectory(prefix='tare-guidance-') as folder:
        root = Path(folder)
        receipt = {}
        features = discover(executable, root, live, run_pipe_process, receipt)
        home = root/'models'
        home.mkdir()
        models = json.loads(_inspect(executable, ['debug', 'models'], home, dict(os.environ, CODEX_HOME=str(home)),
                                     live, run_pipe_process, receipt, seconds=30, max_output=8*1024*1024))['models']
        model = next(m for m in models if m.get('slug', '').startswith('gpt'))
        body = native_cli.cli_request({**json.loads(json.dumps(CHAT)), 'model': model['slug']}, structured=True,
                                      native_arguments=True)
        if choices:
            print('unproved removals:', unproved_removals(executable, root, body, model, features, choices, live,
                                                         run_pipe_process, receipt))
        shown = json.loads(prompt_input(executable, root, body, model, features, choices, live, run_pipe_process,
                                        receipt))
    blocks = [(f'{item.get("role")} message', part.get('text', '')) for item in shown
              for part in (item['content'] if isinstance(item.get('content'), list) else [])]
    print('Codex', model['slug'])
    report('cli://codex', '\n'.join(text for _, text in blocks), choices, blocks)


if __name__ == '__main__':
    vendor, executable, *rest = sys.argv[1:]
    if vendor == 'claude':
        claude(executable, rest[0], json.loads(rest[1]) if len(rest) > 1 else {})
    else:
        codex(executable, json.loads(rest[0]) if rest else {})
