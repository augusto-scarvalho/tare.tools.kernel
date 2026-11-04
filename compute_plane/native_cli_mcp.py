"""Stdio MCP server that only declares Tare's tools to the Claude CLI. It executes nothing.

Claude Code calls the declared tools natively; Tare's PreToolUse hook records and refuses every call before
it reaches this server, and Tare executes the recorded calls after the CLI returns.
"""
import json
import sys


def main():
    # Bytes, not text: under -I the locale codec (cp1252 on Windows) would corrupt UTF-8.
    with open(sys.argv[1], 'rb') as source:
        tools = [{'name': t['name'], 'description': t['description'], 'inputSchema': t['input_schema']}
                 for t in json.loads(source.read())]
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        message = json.loads(line)
        if message.get('id') is None:
            continue  # notifications need no reply
        method = message.get('method')
        if method == 'initialize':
            reply = {'result': {'protocolVersion': (message.get('params') or {}).get('protocolVersion', '2025-06-18'),
                                'capabilities': {'tools': {}}, 'serverInfo': {'name': 'tare', 'version': '1'}}}
        elif method == 'tools/list':
            reply = {'result': {'tools': tools}}
        elif method == 'tools/call':
            reply = {'result': {'content': [{'type': 'text', 'text': 'Tare executes this action.'}], 'isError': True}}
        else:
            reply = {'error': {'code': -32601, 'message': 'Method not found'}}
        sys.stdout.buffer.write((json.dumps({'jsonrpc': '2.0', 'id': message['id'], **reply}) + '\n').encode('utf-8'))
        sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
