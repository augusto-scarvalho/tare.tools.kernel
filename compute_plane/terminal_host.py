"""Bounded loopback transport to a caller-owned terminal supervisor.

The OS explicitly grants an owner roots and a child environment. A bearer
descriptor selects that grant; requests cannot choose owner, cwd or environment.
The caller owns shutdown. No daemon, discovery, disk credentials or retry loop.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import secrets
import socket
import socketserver
import struct
import threading
import time

from .terminal_sessions import TerminalSessions

REQUEST_LIMIT = 262144
RESPONSE_LIMIT = 2097152


class TerminalTransportError(RuntimeError):
    """The exchange is inconclusive; an effect may already have been accepted."""


class TerminalRefused(ValueError):
    """The host returned a definite refusal before dispatching the operation."""


def _encode(value, limit):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()
    if not 0 < len(raw) <= limit:
        raise ValueError('terminal_message_size_invalid')
    return struct.pack('!I', len(raw)) + raw


def _read_exact(stream, length, deadline):
    data = bytearray()
    while len(data) < length:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('terminal_message_deadline')
        stream.settimeout(remaining)
        chunk = stream.recv(length-len(data))
        if not chunk:
            raise EOFError('terminal_transport_closed')
        data.extend(chunk)
    return bytes(data)


def _invalid_constant(_):
    raise ValueError('terminal_nonfinite_json')


def _receive(stream, limit, timeout):
    deadline = time.monotonic()+timeout
    size = struct.unpack('!I', _read_exact(stream, 4, deadline))[0]
    if not 0 < size <= limit:
        raise ValueError('terminal_message_size_invalid')
    return json.loads(_read_exact(stream, size, deadline), parse_constant=_invalid_constant)


class TerminalClient:
    """A connection per request, with no implicit retry after transport loss."""

    def __init__(self, descriptor, *, timeout=20):
        if (not isinstance(descriptor, dict) or set(descriptor) != {'version', 'port', 'token'}
                or type(descriptor['version']) is not int or descriptor['version'] != 1 or type(descriptor['port']) is not int
                or not 1 <= descriptor['port'] <= 65535
                or not isinstance(descriptor['token'], str) or len(descriptor['token']) != 64
                or any(c not in '0123456789abcdef' for c in descriptor['token'])):
            raise ValueError('terminal_descriptor_invalid')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('terminal_transport_timeout_invalid')
        self._descriptor, self.timeout = dict(descriptor), timeout

    def __repr__(self):
        return 'TerminalClient(<session credential redacted>)'

    def request(self, operation, *, transport_timeout=None, **arguments):
        timeout = self.timeout if transport_timeout is None else transport_timeout
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('terminal_transport_timeout_invalid')
        # Encoding/size rejection happens before connecting or sending effects.
        message = _encode(dict(version=1, token=self._descriptor['token'],
                               operation=operation, arguments=arguments), REQUEST_LIMIT)
        try:
            with socket.create_connection(('127.0.0.1', self._descriptor['port']), timeout) as stream:
                stream.settimeout(timeout)
                stream.sendall(message)
                response = _receive(stream, RESPONSE_LIMIT, timeout)
        except (OSError, EOFError, ValueError, RecursionError):
            raise TerminalTransportError('terminal_transport_inconclusive') from None
        if not isinstance(response, dict) or set(response) not in ({'ok', 'result'}, {'ok', 'reason'}):
            raise TerminalTransportError('terminal_response_invalid')
        if response.get('ok') is False and isinstance(response.get('reason'), str):
            raise TerminalRefused(response['reason'])
        if response.get('ok') is not True or 'result' not in response:
            raise TerminalTransportError('terminal_response_invalid')
        return response['result']


class _Grant:
    def __init__(self, owner, roots, env, signature, command_mediation=None):
        self.owner, self.roots, self.env, self.signature = owner, roots, env, signature
        self.command_mediation = command_mediation
        self.process_roots = {}
        from .command_environment import describe
        self.execution_environment = describe(env)
        self.token = secrets.token_hex(32)
        self.lock, self.revoked = threading.Lock(), False


class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False

    def __init__(self, host, max_connections):
        self.host = host
        self.slots = threading.BoundedSemaphore(max_connections)
        self.connections, self.connection_lock = set(), threading.Lock()
        super().__init__(('127.0.0.1', 0), _Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        with self.connection_lock:
            self.connections.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self.connection_lock:
                self.connections.discard(request)
            self.slots.release()
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.connection_lock:
                self.connections.discard(request)
            self.slots.release()

    def handle_error(self, request, client_address):
        pass  # Never emit request bodies, credentials or subprocess output.

    def close_connections(self):
        with self.connection_lock:
            for stream in self.connections:
                try:
                    stream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(2)
        try:
            message = _receive(self.request, REQUEST_LIMIT, 2)
            response = self.server.host._dispatch(message)
            self.request.sendall(_encode(response, RESPONSE_LIMIT))
        except (OSError, EOFError, ValueError, RecursionError):
            return  # Transport loss does not undo or retry an accepted effect.


class TerminalHost:
    """Own alongside the TUI; bind identities from trusted composition code only.

    Project roots are working-directory selection, not a command sandbox.
    Grants authorize host commands with the supplied environment. They cannot be
    retargeted within this host lifetime, including after revocation.
    """

    def __init__(self, *, sessions=None, max_connections=16, max_grants=128):
        if any(type(v) is not int or v < 1 for v in (max_connections, max_grants)):
            raise ValueError('terminal_host_positive_limits_required')
        self.sessions = sessions if sessions is not None else TerminalSessions()
        self._lock, self._closed = threading.Lock(), False
        self._owners, self._tokens = {}, {}
        self.max_grants = max_grants
        self._server = _Server(self, max_connections)
        self._thread = threading.Thread(target=self._server.serve_forever,
            kwargs={'poll_interval': .05}, daemon=True, name='tare-terminal-host')
        try:
            self._thread.start()
        except BaseException:
            self._server.server_close()
            raise

    def bind(self, *, owner, repositories, env, command_mediation=None):
        if not isinstance(owner, str) or not owner or len(owner) > 256 or '\0' in owner:
            raise ValueError('terminal_owner_invalid')
        if not isinstance(repositories, dict) or not repositories or len(repositories) > 32:
            raise ValueError('terminal_repositories_required')
        roots = {}
        for name, directory in repositories.items():
            if not isinstance(name, str) or not name or len(name) > 128 or '\0' in name:
                raise ValueError('terminal_repository_name_invalid')
            path = Path(directory)
            if not path.is_absolute() or not path.is_dir():
                raise ValueError('terminal_absolute_repository_required')
            roots[name] = str(path.resolve())
        if (not isinstance(env, dict) or any(not isinstance(k, str) or not k or '=' in k or '\0' in k
                or not isinstance(v, str) or '\0' in v for k, v in env.items())):
            raise ValueError('terminal_environment_invalid')
        signature = hashlib.sha256(_encode([roots, sorted(env.items())], RESPONSE_LIMIT)).hexdigest()
        with self._lock:
            if self._closed:
                raise ValueError('terminal_host_closed')
            grant = self._owners.get(owner)
            if grant is not None:
                if grant.revoked or grant.signature != signature or grant.command_mediation is not command_mediation:
                    raise ValueError('terminal_grant_cannot_be_rebound')
            else:
                if len(self._owners) >= self.max_grants:
                    raise ValueError('terminal_grant_limit')
                grant = _Grant(owner, roots, dict(env), signature, command_mediation)
                self._owners[owner] = self._tokens[grant.token] = grant
            return dict(version=1, port=self._server.server_address[1], token=grant.token)

    def _dispatch(self, message):
        if (not isinstance(message, dict) or set(message) != {'version', 'token', 'operation', 'arguments'}
                or type(message.get('version')) is not int or message.get('version') != 1 or not isinstance(message.get('token'), str)
                or not isinstance(message.get('operation'), str) or not isinstance(message.get('arguments'), dict)):
            return dict(ok=False, reason='terminal_request_invalid')
        with self._lock:
            grant = self._tokens.get(message['token'])
        if grant is None:
            return dict(ok=False, reason='terminal_credential_unavailable')
        operation, arguments = message['operation'], dict(message['arguments'])
        with grant.lock:
            if self._closed or grant.revoked:
                return dict(ok=False, reason='terminal_credential_unavailable')
            try:
                if set(arguments) & {'owner', 'cwd', 'env'}:
                    raise ValueError('terminal_scope_selected_by_host')
                if operation == 'describe':
                    if arguments:
                        raise ValueError('terminal_arguments_invalid')
                    result = dict(owner=grant.owner, repositories=dict(grant.roots), permission='host_commands',
                                  execution_environment=grant.execution_environment)
                elif operation == 'start':
                    repository = arguments.pop('repository', None)
                    if not isinstance(repository, str) or repository not in grant.roots:
                        raise ValueError('terminal_repository_not_selected')
                    if grant.command_mediation is not None and len(grant.process_roots) >= 4096:
                        raise ValueError('terminal_command_scope_limit')
                    from .command_mediation import CommandFacts, environment_digest, invoke
                    frozen = json.dumps(arguments, sort_keys=True, allow_nan=False, separators=(',', ':'))
                    prepared = json.loads(frozen)
                    for key, value in {'timeout': 300, 'mode': 'pipes', 'columns': 80, 'rows': 24}.items():
                        prepared.setdefault(key, value)
                    frozen = json.dumps(prepared, sort_keys=True, allow_nan=False, separators=(',', ':'))
                    facts = CommandFacts('terminal_start', grant.owner, repository, grant.roots[repository],
                        environment_digest(grant.env), grant.signature, tuple(prepared.get('argv', ())), frozen)
                    result = invoke(grant.command_mediation, facts, lambda: self.sessions.start(
                        owner=grant.owner, cwd=grant.roots[repository], env=grant.env, **json.loads(frozen)))
                    if grant.command_mediation is not None and isinstance(result, dict) and isinstance(result.get('handle'), str):
                        grant.process_roots[result['handle']] = repository
                elif operation == 'send' and grant.command_mediation is not None:
                    from .command_mediation import CommandFacts, environment_digest, invoke
                    repository = grant.process_roots.get(arguments.get('handle'))
                    if repository is None:
                        raise ValueError('terminal_command_scope_unavailable')
                    frozen = json.dumps(arguments, sort_keys=True, allow_nan=False, separators=(',', ':'))
                    # cwd is the launch root, not a claim about a shell's current
                    # directory. A strict argv-only profile must refuse PTY input.
                    facts = CommandFacts('terminal_input', grant.owner, repository, grant.roots[repository],
                        environment_digest(grant.env), grant.signature, None, frozen)
                    result = invoke(grant.command_mediation, facts, lambda: self.sessions.send(
                        owner=grant.owner, **json.loads(frozen)))
                elif operation in {'read', 'send', 'resize', 'close_input', 'stop', 'list'}:
                    result = getattr(self.sessions, operation)(owner=grant.owner, **arguments)
                else:
                    raise ValueError('terminal_operation_not_admitted')
            except (ValueError, TypeError) as exc:
                return dict(ok=False, reason=str(exc)[:160] if isinstance(exc, ValueError)
                            else 'terminal_arguments_invalid')
            return dict(ok=True, result=result)

    def revoke(self, owner):
        with self._lock:
            grant = self._owners.get(owner)
        if grant is None:
            return []
        with grant.lock:
            grant.revoked = True
            with self._lock:
                self._tokens.pop(grant.token, None)
            return self.sessions.close_owner(owner)

    def close(self):
        with self._lock:
            first_close = not self._closed
            self._closed = True
        if first_close:
            self._server.shutdown()
            self._server.close_connections()
            self._server.server_close()
            self._thread.join(timeout=2)
        self.sessions.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
