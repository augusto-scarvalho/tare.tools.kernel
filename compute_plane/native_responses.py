"""Responses wire conversion only. NativeSession owns every tool and next turn."""
import copy


def to_responses(chat_wire, *, codex=True):
    """Convert the adapter's internal history into the exact submitted payload."""
    wire = copy.deepcopy(chat_wire)
    messages = wire.pop('messages')
    system = messages.pop(0)
    if system['role'] != 'system':
        raise ValueError('responses_system_required')
    items = []
    for message in messages:
        role = message['role']
        if role == 'tool':
            items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                          'output': message['content']})
        elif role == 'assistant':
            original = message.get('_responses_output')
            if not isinstance(original, list):
                raise ValueError('responses_original_output_required')
            items.extend(original)
        elif role == 'user':
            content = message['content']
            if isinstance(content, list):
                converted = []
                for part in content:
                    if part.get('type') == 'text':
                        converted.append({'type': 'input_text', 'text': part['text']})
                    elif part.get('type') == 'image_url':
                        converted.append({'type': 'input_image', 'image_url': part['image_url']['url']})
                    else:
                        raise ValueError('responses_input_type_unsupported')
                content = converted
            items.append({'role': 'user', 'content': content})
        else:
            raise ValueError('responses_history_role_unsupported')
    # Codex uses its service output limit. The profile output_tokens reserves
    # context capacity; it is not represented as a server-enforced generation cap.
    limit = wire.pop('max_tokens')
    if not codex:
        wire['max_output_tokens'] = limit
    wire.pop('stream_options', None)
    tools = wire.pop('tools', None)
    wire.update(store=False, stream=True if codex else wire['stream'], instructions=system['content'], input=items,
                include=['reasoning.encrypted_content'])
    if isinstance(wire.get('tool_choice'), dict):  # a named function, in the Responses form
        wire['tool_choice'] = {'type': 'function', 'name': wire['tool_choice']['function']['name']}
    if tools:
        # Responses otherwise normalizes omitted strict to strict mode, making
        # optional mutually exclusive tool arguments required on the server.
        wire['tools'] = [{'type': 'function', **tool['function'], 'strict': False} for tool in tools]
    return wire


def to_chat(response):
    """Keep complete provider items for replay while sharing native validation."""
    output = response.get('output')
    if not isinstance(output, list) or len(output) > 128:
        raise ValueError('responses_output_required')
    calls, text, item_ids, call_ids = [], [], set(), set()
    complete = response.get('status') == 'completed'
    if response.get('status') not in {'completed', 'incomplete'}:
        raise ValueError('responses_completion_required')
    for item in output:
        identity = item.get('id')
        if not isinstance(identity, str) or not identity or identity in item_ids:
            raise ValueError('responses_unique_item_id_required')
        item_ids.add(identity)
        kind = item.get('type')
        if kind == 'function_call':
            call_id = item.get('call_id')
            if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                raise ValueError('responses_unique_call_id_required')
            call_ids.add(call_id)
            if not isinstance(item.get('name'), str) or not isinstance(item.get('arguments'), str):
                raise ValueError('responses_function_shape_invalid')
            if item.get('status') not in {None, 'completed'}:
                complete = False
            calls.append({'id': call_id, 'type': 'function', 'function': {
                'name': item['name'], 'arguments': item['arguments']}})
        elif kind == 'message':
            if item.get('role') != 'assistant' or not isinstance(item.get('content'), list):
                raise ValueError('responses_assistant_required')
            if item.get('status') not in {None, 'completed'}:
                complete = False
            for part in item['content']:
                key = {'output_text': 'text', 'refusal': 'refusal'}.get(part.get('type'))
                if key is None or not isinstance(part.get(key), str):
                    raise ValueError('responses_content_type_unsupported')
                text.append(part[key])
        elif kind != 'reasoning':
            # Hosted tools would introduce effects outside CapabilityPort.
            raise ValueError('responses_output_type_unsupported')
    assistant = {'role': 'assistant', 'content': ''.join(text),
                 '_responses_output': copy.deepcopy(output)}
    if calls:
        assistant['tool_calls'] = calls
    return {'model': response.get('model'), 'usage': response.get('usage'),
            'choices': [{'message': assistant, 'finish_reason':
                         ('tool_calls' if calls else 'stop') if complete else 'length'}]}
