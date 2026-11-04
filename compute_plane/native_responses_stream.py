"""Collect a single Responses stream. Deltas never authorize tool execution."""
import copy

KINDS = {'message', 'function_call', 'reasoning'}


def _same_snapshot(completed, terminal):
    if len(completed) != len(terminal):
        return False
    for left, right in zip(completed, terminal):
        if left == right:
            continue
        if (not isinstance(right, dict) or left.get('type') != 'reasoning'
                or right.get('type') != 'reasoning'
                or any(not isinstance(item.get('encrypted_content'), str) or not item['encrypted_content']
                       for item in (left, right))):
            return False
        if ({k: v for k, v in left.items() if k != 'encrypted_content'}
                != {k: v for k, v in right.items() if k != 'encrypted_content'}):
            return False
    return True


def decode(events, emit):
    started, completed, identities = {}, {}, {}
    response_id = None

    def check_item(event, *, done):
        index, item = event.get('output_index'), event.get('item')
        if type(index) is not int or not 0 <= index < 128 or not isinstance(item, dict):
            raise ValueError('responses_item_shape')
        identity, kind = item.get('id'), item.get('type')
        if not isinstance(identity, str) or not identity or kind not in KINDS:
            raise ValueError('responses_item_identity')
        if identity in identities and identities[identity] != index:
            raise ValueError('responses_duplicate_item_id')
        identities[identity] = index
        if index in completed or not done and index in started:
            raise ValueError('responses_duplicate_item_index')
        if index in started and started[index] != (identity, kind):
            raise ValueError('responses_item_changed')
        if done:
            if item.get('status') not in {None, 'completed'}:
                raise ValueError('responses_unfinished_item')
            if kind == 'function_call' and any(not isinstance(item.get(key), str) or not item[key]
                    for key in ('call_id', 'name', 'arguments')):
                raise ValueError('responses_unfinished_function')
            completed[index] = copy.deepcopy(item)
        else:
            started[index] = (identity, kind)

    for number, event in enumerate(events):
        if number >= 32768 or event is None or not isinstance(event, dict):
            raise ValueError('responses_missing_terminal')
        kind = event.get('type')
        response = event.get('response')
        if isinstance(response, dict):
            identity = response.get('id')
            if not isinstance(identity, str) or not identity or (
                    response_id is not None and response_id != identity):
                raise ValueError('responses_response_identity')
            response_id = identity
        if kind == 'response.output_item.added':
            check_item(event, done=False)
        elif kind == 'response.output_item.done':
            check_item(event, done=True)
        elif kind in {'response.output_text.delta', 'response.function_call_arguments.delta',
                      'response.reasoning_summary_text.delta', 'response.reasoning_text.delta'}:
            fragment = event.get('delta')
            if not isinstance(fragment, str):
                raise ValueError('responses_delta_text')
            channel = ('text' if kind == 'response.output_text.delta' else 'tool_arguments'
                       if kind == 'response.function_call_arguments.delta' else 'reasoning_content')
            emit({'kind': 'delta', 'channel': channel, 'text': fragment})
        elif kind in {'response.completed', 'response.incomplete'}:
            if not isinstance(response, dict) or response.get('status') != kind.split('.')[1]:
                raise ValueError('responses_terminal_status')
            if set(started) - set(completed):
                raise ValueError('responses_unclosed_items')
            items = [completed[index] for index in sorted(completed)]
            if completed and sorted(completed) != list(range(len(completed))):
                raise ValueError('responses_missing_output_index')
            supplied = response.get('output')
            if supplied is not None and not isinstance(supplied, list):
                raise ValueError('responses_output_shape')
            if supplied and items and not _same_snapshot(items, supplied):
                raise ValueError('responses_final_snapshot_changed')
            # A terminal snapshot can re-encrypt the same opaque reasoning item.
            # The streaming API specifies replaying the output_item.done value.
            output = items or supplied or []
            if len(output) > 128 or any(not isinstance(item, dict) or item.get('type') not in KINDS
                                      for item in output):
                raise ValueError('responses_output_kind')
            if response.get('usage') is not None:
                emit({'kind': 'usage', 'usage': response['usage']})
            return {**response, 'output': copy.deepcopy(output)}
        elif kind == 'response.failed':
            # A terminal provider failure closes this inference, including any
            # unfinished output items. Never promote those items to tool calls.
            if not isinstance(response, dict) or response.get('status') != 'failed':
                raise ValueError('responses_terminal_status')
            if response.get('usage') is not None:
                emit({'kind': 'usage', 'usage': response['usage']})
            return {'id': response_id, 'status': 'failed', 'output': [],
                    'model': response.get('model'), 'usage': response.get('usage')}
        elif kind in {'error', 'response.error'}:
            raise ValueError('responses_provider_error')
        # Any other event is progress (a keepalive, an event a newer API adds) and is ignored, as the Codex CLI does:
        # output comes only from the items and the terminal event above, and a stream without one still fails.
        # Refusing them cost a 30 s gpt-6-luna call and its session (RECONCILIATION_REQUIRED) on 2026-10-07.
    raise ValueError('responses_missing_terminal')
