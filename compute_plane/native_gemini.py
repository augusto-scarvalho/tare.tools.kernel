"""Gemini generateContent conversion; common history and effect validation stay native."""
import copy
import re

from .native_models import GEMINI_PARAMETERS, encode


def _call_id(call, index):
    # The API may omit call ids; position within the reply is then the only identity it has.
    return call['id'] if isinstance(call.get('id'), str) and call['id'] else 'call_%d' % index


def to_generate_content(chat_wire):
    wire = copy.deepcopy(chat_wire)
    source = wire.pop('messages')
    system = source.pop(0)
    if system['role'] != 'system' or wire.get('stream'):
        raise ValueError('gemini_system_and_nonstream_required')
    contents, calls, previous = [], {}, None
    for message in source:
        role, content = message['role'], message['content']
        if role == 'assistant':
            role, parts = 'model', message.get('_gemini_parts')
            if not isinstance(parts, list):
                raise ValueError('gemini_original_content_required')
            # Original parts carry the opaque thought signatures the API requires back unchanged.
            requested = [part['functionCall'] for part in parts if 'functionCall' in part]
            calls = {_call_id(call, index): call for index, call in enumerate(requested)}
        elif role == 'tool':
            call = calls.get(message['tool_call_id'])
            if call is None:
                raise ValueError('gemini_tool_result_without_call')
            answer = {'name': call['name'],
                      'response': {'error' if message.get('_is_error') else 'output': content}}
            if 'id' in call:
                answer['id'] = call['id']
            role, parts = 'user', [{'functionResponse': answer}]
        elif role == 'user':
            parts = []
            for part in [{'type': 'text', 'text': content}] if isinstance(content, str) else content:
                if part.get('type') == 'text':
                    parts.append({'text': part['text']})
                elif part.get('type') == 'image_url':
                    match = re.fullmatch(r'data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=]+)',
                                         part['image_url']['url'])
                    if not match:
                        raise ValueError('gemini_inline_image_required')
                    parts.append({'inlineData': {'mimeType': match[1], 'data': match[2]}})
                else:
                    raise ValueError('gemini_input_type_unsupported')
        else:
            raise ValueError('gemini_history_role_unsupported')
        if not parts:
            continue  # a reply cut before any output, or an empty message: the API refuses a turn without parts
        # One turn answers every call of a parallel batch, and holds nothing else: a turn that mixed a result
        # with text was refused as "ending with a model turn" (OpenCode, gemini-4-argon, 2026-11-10).
        kind = (role, 'functionResponse' in parts[0])
        if kind == previous:
            contents[-1]['parts'].extend(parts)
        else:
            contents.append({'role': role, 'parts': list(parts)})
        previous = kind
    result = {'systemInstruction': {'parts': [{'text': system['content']}]}, 'contents': contents,
              'generationConfig': {'maxOutputTokens': wire['max_tokens'],
                                   **{key: wire[key] for key in sorted(GEMINI_PARAMETERS & set(wire))}}}
    tools, choice = wire.get('tools'), wire.get('tool_choice')
    if tools:
        result['tools'] = [{'functionDeclarations': [
            {'name': t['function']['name'], 'description': t['function']['description'],
             'parametersJsonSchema': t['function']['parameters']} for t in tools]}]
        result['toolConfig'] = {'functionCallingConfig': (
            {'mode': 'ANY', 'allowedFunctionNames': [choice['function']['name']]}
            if isinstance(choice, dict) else {'mode': 'AUTO'})}
    return result


def to_chat(response):
    candidates = response.get('candidates')
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError('gemini_one_candidate_required')
    content = candidates[0].get('content') or {}
    # A reply cut before any output has no parts.
    parts = content.get('parts', []) if isinstance(content, dict) else None
    if not isinstance(parts, list) or len(parts) > 128 or content.get('role', 'model') != 'model':
        raise ValueError('gemini_content_required')
    calls, texts = [], []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError('gemini_part_required')
        kinds = set(part) - {'thought', 'thoughtSignature'}
        if kinds == {'text'}:
            if not isinstance(part['text'], str):
                raise ValueError('gemini_text_required')
            if part.get('thought') is not True:  # a thought summary is not the answer
                texts.append(part['text'])
        elif kinds == {'functionCall'}:
            call = part['functionCall']
            if (not isinstance(call, dict) or not isinstance(call.get('name'), str)
                    or not isinstance(call.get('args', {}), dict)):
                raise ValueError('gemini_tool_shape_invalid')
            calls.append({'id': _call_id(call, len(calls)), 'type': 'function', 'function': {
                'name': call['name'], 'arguments': encode(call.get('args', {}))}})
        elif kinds:
            # Code execution, hosted search and generated media are outside CapabilityPort.
            raise ValueError('gemini_output_type_unsupported')
    stop = candidates[0].get('finishReason')
    if not isinstance(stop, str) or not stop:
        raise ValueError('gemini_finish_reason_required')
    assistant = {'role': 'assistant', 'content': ''.join(texts), '_gemini_parts': copy.deepcopy(parts)}
    if calls:
        assistant['tool_calls'] = calls
    # Anything but STOP (output limit, safety, a call the provider could not parse) is an incomplete reply.
    finish = (('tool_calls' if calls else 'stop') if stop == 'STOP' else 'length' if stop == 'MAX_TOKENS'
              else 'gemini:' + stop)
    return {'model': response.get('modelVersion'), 'usage': response.get('usageMetadata'),
            'choices': [{'message': assistant, 'finish_reason': finish}]}
