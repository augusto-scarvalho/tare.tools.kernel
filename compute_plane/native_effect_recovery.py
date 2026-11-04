"""Local effect evidence. The host holds custody; no effect is replayed here."""
from dataclasses import asdict
import hashlib
import re
import os

from tare_tools_agent_runtime import EffectRequest, EffectResult, ToolCall
from .native_journal import io_path, regular, read_json, save_json
from .native_review import capture, digest, edit_args, receipt_path
from .native_models import encode, object_json


def pending(journal, session_id):
    request = None
    for record in journal.snapshot().records:
        event = record.event()
        if event['kind'] == 'effect_requested':
            request = EffectRequest(session_id, event['effect_id'], ToolCall(**event['tool_call']))
        elif request is not None and (
                event['kind'] == 'effect_returned' and event['result']['effect_id'] == request.effect_id
                and event['result']['status'] != 'unknown'
                or event['kind'] == 'invocation_not_started' and event['invocation_id'] == request.effect_id):
            request = None
    return request


class EffectReceipts:
    def __init__(self, scratch, session_id, config):
        self.root = regular(io_path(scratch / 'effect-receipts'))
        self.session_id = session_id
        self.config_sha = hashlib.sha256(encode(config).encode()).hexdigest()

    def identity(self, request):
        return {'schema': 'tare.tools/native-effect/1', 'session_id': self.session_id,
                'config_sha256': self.config_sha, 'request': asdict(request)}

    def save(self, request, **evidence):
        self.root.mkdir(exist_ok=True)
        path = receipt_path(self.root, request.effect_id)
        prior = read_json(path) if path.exists() else {}
        claimed = 'action_control' in prior
        if claimed and any(prior.get(key) != value for key, value in self.identity(request).items()):
            raise ValueError('Effect receipt identity changed before recording the result.')
        if claimed and prior.get('result', {}).get('status') in {'succeeded', 'refused', 'failed'}:
            return  # A later denial/replay cannot erase the original proved result.
        save_json(path, (prior if claimed else {}) | self.identity(request) | evidence)

    def claim_action(self, request, action, authorization):
        """Persist one exact intent atomically; a crash/partial file never enables replay."""
        from .action_control import ActionSnapshot, ActionAuthorization
        if (not isinstance(action, ActionSnapshot) or not isinstance(authorization, ActionAuthorization)
                or request.session_id != self.session_id or action.session_id != self.session_id
                or request.effect_id != action.effect_id or authorization.effect_id != action.effect_id
                or authorization.session_id != action.session_id or authorization.binding_id != action.binding_id
                or authorization.fence_seq != action.fence_seq
                or authorization.action_sha256 != action.action_sha256):
            raise ValueError('Action claim does not match this effect.')
        self.root.mkdir(exist_ok=True)
        path = receipt_path(self.root, request.effect_id)
        record = self.identity(request) | {'action_control': {'action_sha256': action.action_sha256,
            'issuer': authorization.issuer, 'authorization_nonce_sha256': digest(authorization.nonce.encode()),
            'binding_id': action.binding_id, 'fence_seq': action.fence_seq}}
        try:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(encode(record))
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            return False
        return True

    def inspect(self, request, tools, review_root=None):
        """Return a proved result, or None for an effect with an unknown outcome."""
        name = request.tool_call.name
        args = object_json(request.tool_call.arguments_json)
        if name == 'edit':
            args = edit_args(args)
        path = receipt_path(self.root, request.effect_id)
        if path.exists():
            row = read_json(path)
            if any(row.get(k) != v for k, v in self.identity(request).items()):
                raise ValueError('Effect receipt does not match this session and request.')
            if 'result' in row:
                result = EffectResult(**row['result'])
                if result.effect_id != request.effect_id:
                    raise ValueError('Effect receipt identity changed.')
                if result.status != 'unknown':
                    if name == 'edit' and result.status == 'succeeded':
                        self.check_file(tools, args, object_json(result.output_json)['sha256'])
                    return result
            if name == 'edit' and 'edit' in row:
                edit = row['edit']
                tools.path(args['repository'], args['path'], write=True)
                current = tools.observed(args['repository'], args['path'])
                if current == edit['after']:
                    return EffectResult(request.effect_id, 'succeeded', encode({
                        'sha256': current, 'effects': edit['effects'],
                        'pre_context': edit['pre_context'], 'post_context': None,
                        'recovery': 'Atomic edit confirmed from current bytes; post-hook not replayed.'}), True)
                if current == args['sha256']:
                    return EffectResult(request.effect_id, 'refused', encode({
                        'reason': 'Interrupted edit is not present. Read current bytes before deciding to edit again.'}))
                raise ValueError('File changed after the interrupted edit; newer changes were preserved.')
        # Older sessions already saved exact before/after snapshots for review.
        if name == 'edit' and review_root is not None:
            path = receipt_path(review_root, request.effect_id)
            if path.exists():
                row = read_json(path)
                if (row.get('schema') != 'tare.tools/edit-review/1' or row.get('effect_id') != request.effect_id
                        or any(row.get(k) != args[k] for k in ('repository', 'path'))
                        or row.get('before') != args['sha256']
                        or not isinstance(row.get('after'), str)
                        or re.fullmatch('[0-9a-f]{64}', row['after']) is None
                        or row['before'] is not None and (not isinstance(row['before'], str)
                            or re.fullmatch('[0-9a-f]{64}', row['before']) is None)):
                    raise ValueError('Edit review receipt does not match the interrupted request.')
                before = capture(review_root / 'blobs' / row['before']) if row['before'] else None
                after = capture(review_root / 'blobs' / row['after'])
                if (digest(before) != row['before'] or digest(after) != row['after']
                        or tools.edited_text((before or b'').decode(), args).encode() != after):
                    raise ValueError('Edit review snapshots do not prove this replacement.')
                self.check_file(tools, args, row['after'])
                return EffectResult(request.effect_id, 'succeeded', encode({
                    'sha256': row['after'], 'effects': tools.effects + 1,
                    'pre_context': None, 'post_context': None, 'recovery': 'Confirmed from saved edit snapshots.'}), True)
        if name in {'read', 'list', 'search', 'context', 'terminal_read'}:
            return EffectResult(request.effect_id, 'refused', encode({
                'reason': 'Interrupted observation has no saved result. Request a fresh observation if needed.'}))
        return None

    @staticmethod
    def check_file(tools, args, expected):
        tools.path(args['repository'], args['path'], write=True)
        if tools.observed(args['repository'], args['path']) != expected:
            raise ValueError('File changed after the completed edit; newer changes were preserved.')
