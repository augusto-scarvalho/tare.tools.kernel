"""Anthropic Messages conversion; common history and effect validation stay native."""
import copy
import re

from .native_models import encode


def to_messages(chat_wire, *, cache_system=False):
    wire = copy.deepcopy(chat_wire)
    source = wire.pop('messages')
    system = source.pop(0)
    if system['role'] != 'system' or wire.get('stream'):
        raise ValueError('anthropic_system_and_nonstream_required')
    wire.pop('stream_options', None)
    wire['system'] = [{'type': 'text', 'text': system['content']}]
    if cache_system:
        wire['system'][0]['cache_control'] = copy.deepcopy(wire.get('cache_control', {'type':'ephemeral'}))
    messages = []
    for message in source:
        role, content = message['role'], message['content']
        if role == 'assistant':
            content = message.get('_anthropic_content')
            if not isinstance(content, list):
                raise ValueError('anthropic_original_content_required')
        elif role == 'tool':
            role = 'user'
            content = [{'type': 'tool_result', 'tool_use_id': message['tool_call_id'],
                        'content': content, 'is_error': message.get('_is_error', False)}]
        elif role == 'user':
            if isinstance(content, str):
                content = [{'type': 'text', 'text': content}]
            else:
                parts = []
                for part in content:
                    if part.get('type') == 'text':
                        parts.append(part)
                    elif part.get('type') == 'image_url':
                        match = re.fullmatch(r'data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=]+)',
                                             part['image_url']['url'])
                        if not match:
                            raise ValueError('anthropic_inline_image_required')
                        parts.append({'type': 'image', 'source': {'type': 'base64',
                                      'media_type': match[1], 'data': match[2]}})
                    else:
                        raise ValueError('anthropic_input_type_unsupported')
                content = parts
        else:
            raise ValueError('anthropic_history_role_unsupported')
        if messages and messages[-1]['role'] == role:
            messages[-1]['content'].extend(content)
        else:
            messages.append({'role': role, 'content': content})
    wire['messages'] = messages
    tools = wire.pop('tools', None)
    choice = wire.pop('tool_choice', None)
    if tools:
        wire['tools'] = [{'name': t['function']['name'], 'description': t['function']['description'],
                          'input_schema': t['function']['parameters']} for t in tools]
        wire['tool_choice'] = ({'type': 'tool', 'name': choice['function']['name']}
                               if isinstance(choice, dict) else {'type': 'auto'})
    if cache_system:
        controls=[block['cache_control'] for message in messages for block in message['content']
                  if 'cache_control' in block]
        existing=len(controls)
        longer=any(control.get('ttl')=='1h' for control in controls)
        if existing >= 4 or (longer and wire['system'][0]['cache_control'].get('ttl')!='1h'):
            wire['system'][0].pop('cache_control',None)  # Keep the caller's explicit controls.
        if not existing:
            for message in reversed(messages):
                selected=next((block for block in reversed(message['content'])
                    if block.get('type') in {'text','image','tool_result','tool_use'}
                    and not (block.get('type')=='text' and block.get('text','').startswith('Runtime budget: '))),None)
                if selected is not None:
                    selected['cache_control']=copy.deepcopy(wire.get('cache_control', {'type':'ephemeral'}))
                    break
    return wire


def to_chat(response):
    if (response.get('type') != 'message' or response.get('role') != 'assistant'
            or not isinstance(response.get('id'), str) or not response['id']):
        raise ValueError('anthropic_message_required')
    blocks = response.get('content')
    if not isinstance(blocks, list) or len(blocks) > 128:
        raise ValueError('anthropic_content_required')
    calls, texts = [], []
    for block in blocks:
        kind = block.get('type')
        if kind == 'text':
            if not isinstance(block.get('text'), str):
                raise ValueError('anthropic_text_required')
            texts.append(block['text'])
        elif kind == 'tool_use':
            if (not isinstance(block.get('input'), dict) or not isinstance(block.get('name'), str)
                    or not isinstance(block.get('id'), str)):
                raise ValueError('anthropic_tool_shape_invalid')
            calls.append({'id': block['id'], 'type': 'function', 'function': {
                'name': block['name'], 'arguments': encode(block['input'])}})
        elif kind == 'thinking':
            if (not isinstance(block.get('thinking'), str) or not isinstance(block.get('signature'), str)
                    or not block['signature']):
                raise ValueError('anthropic_signed_thinking_required')
        elif kind == 'redacted_thinking':
            if not isinstance(block.get('data'), str) or not block['data']:
                raise ValueError('anthropic_redacted_thinking_required')
        else:
            # server_tool_use, web_search_tool_result, containers and all hosted
            # effects are outside CapabilityPort, even alongside valid local calls.
            raise ValueError('anthropic_output_type_unsupported')
    stop = response.get('stop_reason')
    if stop not in {'end_turn', 'tool_use', 'max_tokens', 'stop_sequence', 'pause_turn', 'refusal'}:
        raise ValueError('anthropic_stop_required')
    if stop == 'tool_use' and not calls or calls and stop in {'end_turn', 'stop_sequence'}:
        raise ValueError('anthropic_stop_content_mismatch')
    complete = stop == 'tool_use' if calls else stop in {'end_turn', 'stop_sequence'}
    assistant = {'role': 'assistant', 'content': ''.join(texts),
                 '_anthropic_content': copy.deepcopy(blocks)}
    if calls:
        assistant['tool_calls'] = calls
    return {'model': response.get('model'), 'usage': response.get('usage'),
            'choices': [{'message': assistant, 'finish_reason':
                         ('tool_calls' if calls else 'stop') if complete else 'length'}]}
