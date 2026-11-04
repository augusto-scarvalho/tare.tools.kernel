"""Loopback and fresh consumer processes; no user projects or model provider."""
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

import pytest

from compute_plane.terminal_host import TerminalClient, TerminalHost, TerminalRefused, TerminalTransportError


def grant(host, root, owner='conversation'):
    return host.bind(owner=owner, repositories={'project':str(root)}, env=dict(os.environ))


def start_args(code, request_id='start'):
    return dict(repository='project', request_id=request_id,
                argv=[sys.executable, '-I', '-u', '-c', code])


def until(client, handle, predicate):
    deadline = time.monotonic()+8
    while time.monotonic() < deadline:
        value = client.request('read', handle=handle)
        if predicate(value):
            return value
        time.sleep(.02)
    pytest.fail('terminal state: '+repr(value))


def consumer(descriptor, operation, arguments):
    code = '''import sys,json
sys.path.insert(0,sys.argv[1])
from compute_plane.terminal_host import TerminalClient
value=json.loads(sys.stdin.read())
print(json.dumps(TerminalClient(value['descriptor']).request(value['operation'],**value['arguments'])))
'''
    result = subprocess.run([sys.executable,'-B','-c',code,str(Path(__file__).resolve().parents[1])],
        input=json.dumps(dict(descriptor=descriptor, operation=operation, arguments=arguments)),
        capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_process_survives_two_consumer_turns(tmp_path):
    with TerminalHost() as host:
        descriptor = grant(host, tmp_path)
        args = start_args('import sys; print("READY",flush=True); print("ANSWER="+sys.stdin.readline(),flush=True)')
        first = consumer(descriptor, 'start', args)
        client = TerminalClient(descriptor)
        handle = first['handle']
        ready = until(client, handle, lambda r: 'READY' in r['output']['stdout']['text'])
        assert ready['status'] == 'running'  # first consumer has already exited
        consumer(descriptor, 'send', dict(handle=handle, request_id='message', text='second-turn\n'))
        done = until(client, handle, lambda r: r['cleanup_verified'])
        assert done['exit_code'] == 0 and 'ANSWER=second-turn' in done['output']['stdout']['text']
        assert consumer(descriptor, 'start', args)['handle'] == handle


def test_scope_and_credentials_cannot_be_selected_by_request(tmp_path):
    second = tmp_path/'second'; second.mkdir()
    with TerminalHost() as host:
        a, b = grant(host, tmp_path, 'a'), grant(host, second, 'b')
        ca, cb = TerminalClient(a), TerminalClient(b)
        running = ca.request('start', **start_args('import time; time.sleep(60)'))
        handle = running['handle']
        with pytest.raises(TerminalRefused, match='handle_unavailable'):
            cb.request('stop', handle=handle)
        for extra in ({'owner':'b'}, {'cwd':str(second)}, {'env':dict(os.environ)}):
            with pytest.raises(TerminalRefused, match='scope_selected_by_host'):
                ca.request('start', **start_args('print("not run")'), **extra)
        with pytest.raises(TerminalRefused, match='repository_not_selected'):
            ca.request('start', **{**start_args('print("not run")'), 'repository':str(second)})
        with pytest.raises(TerminalRefused, match='credential_unavailable'):
            TerminalClient({**a, 'token':'0'*64}).request('list')
        assert a['token'] not in repr(ca)
        assert ca.request('read', handle=handle)['status'] == 'running'
        assert cb.request('list') == []


def test_grants_are_stable_then_revocable_without_retargeting(tmp_path):
    second = tmp_path/'second'; second.mkdir()
    with TerminalHost() as host:
        a = grant(host, tmp_path, 'a')
        assert grant(host, tmp_path, 'a') == a
        with pytest.raises(ValueError, match='cannot_be_rebound'):
            grant(host, second, 'a')
        b = grant(host, second, 'b')
        ca, cb = TerminalClient(a), TerminalClient(b)
        handle = ca.request('start', **start_args('import time; time.sleep(60)'))['handle']
        other = cb.request('start', **start_args('import time; time.sleep(60)'))['handle']
        closed = host.revoke('a')
        assert closed[0]['handle'] == handle and closed[0]['cleanup_verified']
        with pytest.raises(TerminalRefused, match='credential_unavailable'):
            ca.request('list')
        with pytest.raises(ValueError, match='cannot_be_rebound'):
            grant(host, tmp_path, 'a')
        assert cb.request('read', handle=other)['status'] == 'running'


def test_lost_start_response_does_not_duplicate_effect(tmp_path):
    with TerminalHost() as host:
        descriptor = grant(host, tmp_path)
        args = start_args('from pathlib import Path; p=Path("count"); p.write_text(p.read_text()+"x" if p.exists() else "x")')
        request = json.dumps(dict(version=1, token=descriptor['token'], operation='start', arguments=args)).encode()
        with socket.create_connection(('127.0.0.1',descriptor['port']), 2) as stream:
            stream.sendall(struct.pack('!I',len(request))+request)
            # Intentionally discard the response after sending the full request.
        client = TerminalClient(descriptor)
        result = client.request('start', **args)
        until(client, result['handle'], lambda r: r['cleanup_verified'])
        assert (tmp_path/'count').read_text() == 'x'
        assert len(client.request('list', include_completed=True)) == 1


def test_host_shutdown_closes_idle_connections_and_owned_processes(tmp_path):
    host = TerminalHost()
    descriptor = grant(host, tmp_path)
    client = TerminalClient(descriptor, timeout=1)
    handle = client.request('start', **start_args('import time; time.sleep(60)'))['handle']
    idle = socket.create_connection(('127.0.0.1',descriptor['port']), 2)
    idle.settimeout(3)
    try:
        idle.sendall(b'\0')
        began = time.monotonic()
        host.close()
        assert time.monotonic()-began < 3
        try:
            assert idle.recv(1) == b''
        except ConnectionResetError:
            pass  # Windows may reset a connection with an incomplete frame.
        assert host.sessions.read(owner='conversation',handle=handle)['cleanup_verified']
        with pytest.raises(TerminalTransportError):
            client.request('list')
        host.close()
    finally:
        idle.close()
        host.close()


def test_invalid_frames_are_bounded_and_do_not_poison_next_request(tmp_path):
    with TerminalHost() as host:
        descriptor = grant(host, tmp_path)
        for payload in (struct.pack('!I',2**32-1), struct.pack('!I',4)+b'null'):
            with socket.create_connection(('127.0.0.1',descriptor['port']),2) as stream:
                stream.settimeout(3)
                stream.sendall(payload)
                response = stream.recv(2048)
                assert not response or b'terminal_request_invalid' in response
        assert TerminalClient(descriptor).request('list') == []


def test_fresh_host_never_accepts_previous_host_credential(tmp_path):
    with TerminalHost() as original:
        old = grant(original, tmp_path)
    with TerminalHost() as fresh:
        new = grant(fresh, tmp_path)
        with pytest.raises(TerminalRefused, match='credential_unavailable'):
            TerminalClient({**old,'port':new['port']}).request('list')
        assert TerminalClient(new).request('list') == []


@pytest.mark.parametrize('changes', [{'host':'example.com'}, {'version':True}, {'token':'short'}, {'port':False}])
def test_descriptor_cannot_redirect_to_an_arbitrary_host(changes):
    with pytest.raises(ValueError, match='descriptor_invalid'):
        TerminalClient({**dict(version=1,port=1234,token='0'*64),**changes})
