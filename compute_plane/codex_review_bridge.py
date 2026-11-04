"""Private stdio MCP channel to the host's already-claimed command effect."""
import json
import math
import os
from pathlib import Path
import sys
import time


def loads(data, *, max_bytes=65536):
    if type(max_bytes) is not int or not 0 < max_bytes <= 131072:
        raise ValueError('bounded_json_required')
    size = len(data.encode('utf-8')) if isinstance(data, str) else len(data)
    if size > max_bytes:
        raise ValueError('bounded_json_required')
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise ValueError('duplicate_key')
            value[key] = item
        return value
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def main(root):
    root = Path(root)
    packet = loads((root/'packet.json').read_bytes())
    arguments, deadline = packet['arguments'], packet['deadline']
    if type(arguments) is not dict or type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('invalid_bridge_packet')
    expected = json.dumps(arguments, sort_keys=True, allow_nan=False)
    tool = {'name': 'execute_exact_command',
        'description': 'Execute only this host-bound command, once. It cannot be changed through tool arguments. '
                       'Exact command and cwd for approval review: '+expected,
        'inputSchema': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False},
        'annotations': {'readOnlyHint': False, 'destructiveHint': True, 'openWorldHint': False}}
    used = False
    while line := sys.stdin.buffer.readline(65537):
        row = loads(line)
        if 'id' not in row: continue
        result = None
        try:
            method = row.get('method')
            if method == 'initialize':
                result = {'protocolVersion': row['params']['protocolVersion'], 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'tare-exact-command', 'version': '0.1'}}
            elif method == 'tools/list': result = {'tools': [tool]}
            elif method == 'ping': result = {}
            elif method == 'tools/call':
                params = row['params']
                if (used or time.monotonic() >= deadline or params['name'] != tool['name']
                        or type(params.get('arguments')) is not dict or params['arguments']):
                    raise ValueError('exact_bridge_call_required')
                used = True
                with (root/'dispatch.tmp').open('x', encoding='utf-8') as file:
                    file.write(expected)
                os.link(root/'dispatch.tmp', root/'dispatch.json')
                while time.monotonic() < deadline:
                    if (root/'response.json').is_file():
                        result = loads((root/'response.json').read_bytes()); break
                    time.sleep(.02)
                if result is None: raise ValueError('host_command_deadline')
            else: raise ValueError('unsupported_method')
            reply = {'jsonrpc': '2.0', 'id': row['id'], 'result': result}
        except (ValueError, KeyError, TypeError, OSError):
            reply = {'jsonrpc': '2.0', 'id': row['id'], 'error': {'code': -32000, 'message': 'Host command refused'}}
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
