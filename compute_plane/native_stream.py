"""Bounded SSE / AWS EventStream decoding for the isolated HTTP worker.

Only completed protocol messages become finite responses. Deltas are provisional
observations; tool arguments and reasoning are retained but never executed here.
"""
import base64
import copy
import json
import struct
import zlib

MAX_BYTES = 4 * 1024 * 1024
MAX_EVENTS = 32768


def obj(raw):
    value = json.loads(raw)
    if not isinstance(value, dict): raise ValueError('stream_object_required')
    return value


class Reader:
    def __init__(self, stream): self.stream, self.size = stream, 0
    def read(self, count):
        value = self.stream.read(count); self.size += len(value)
        if self.size > MAX_BYTES: raise ValueError('stream_byte_limit')
        return value
    def exact(self, count):
        value = bytearray()
        while len(value) < count:
            part = self.read(count-len(value))
            if not part: raise ValueError('stream_truncated_frame')
            value.extend(part)
        return bytes(value)
    def line(self):
        value = self.stream.readline(MAX_BYTES + 1); self.size += len(value)
        if self.size > MAX_BYTES: raise ValueError('stream_byte_limit')
        return value


def sse(stream):
    reader, data, events = Reader(stream), [], 0
    while line := reader.line():
        line = line.rstrip(b'\r\n')
        if not line:
            if data:
                events += 1
                if events > MAX_EVENTS: raise ValueError('stream_event_limit')
                value = b'\n'.join(data).decode('utf-8'); data = []
                if value == '[DONE]':
                    yield None
                    return
                yield obj(value)
        elif line.startswith(b'data:'):
            data.append(line[5:].removeprefix(b' '))
    raise ValueError('stream_missing_done')


def eventstream(stream):
    reader = Reader(stream)
    for _ in range(MAX_EVENTS):
        first = reader.read(1)
        if not first: return
        prelude = first + reader.exact(11)
        total, headers_len, crc = struct.unpack('!III', prelude)
        if zlib.crc32(prelude[:8]) != crc or not 16 <= total <= MAX_BYTES or headers_len > total-16:
            raise ValueError('stream_bad_prelude')
        rest = reader.exact(total-12)
        if zlib.crc32(prelude+rest[:-4]) != struct.unpack('!I',rest[-4:])[0]:
            raise ValueError('stream_bad_crc')
        headers, at = {}, 0
        while at < headers_len:
            length = rest[at]; at += 1
            name = rest[at:at+length].decode(); at += length
            kind = rest[at]; at += 1
            if kind != 7 or name in headers: raise ValueError('stream_header_type')
            length = struct.unpack('!H',rest[at:at+2])[0]; at += 2
            headers[name] = rest[at:at+length].decode(); at += length
            if at > headers_len: raise ValueError('stream_header_length')
        if headers.get(':message-type') != 'event':
            raise ValueError('stream_provider_exception')
        yield headers.get(':event-type'), obj(rest[headers_len:-4])
    raise ValueError('stream_event_limit')


def text(value):
    if not isinstance(value, str): raise ValueError('stream_text_required')
    return value


def _opaque(target, source, excluded=()):
    """Retain opaque provider extensions exactly; ambiguous fragments fail closed."""
    for key, value in source.items():
        if key in excluded:
            continue
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _opaque(target[key], value)
        elif target[key] != value:
            raise ValueError('stream_opaque_metadata_changed')


def _reasoning_fragments(target, fragments):
    """Join consecutive indexed text deltas with identical opaque metadata.

    Opaque/encrypted blocks and signature or identity changes stay discrete.
    No reasoning character is removed and no provider-specific field rewritten.
    """
    for item in fragments:
        field = {'reasoning.text':'text', 'reasoning.summary':'summary'}.get(item.get('type'))
        previous = target[-1] if target else None
        if (field and type(item.get('index')) is int and isinstance(item.get(field), str)
                and previous is not None and isinstance(previous.get(field), str)
                and {k:v for k,v in item.items() if k != field}
                    == {k:v for k,v in previous.items() if k != field}):
            previous[field] += item[field]
        else:
            target.append(copy.deepcopy(item))


def chat(stream, emit):
    model, finish, usage, fleet, timings = None, None, None, None, None
    accounted = False
    assistant, tools = {'role':'assistant','content':''}, {}
    for value in sse(stream):
        if value is None:
            if model is None or finish is None: raise ValueError('stream_missing_completion')
            if tools: assistant['tool_calls'] = [tools[i] for i in sorted(tools)]
            result = {'model':model,'choices':[{'index':0,'message':assistant,'finish_reason':finish}], 'usage':usage}
            if fleet is not None: result['tare_fleet_observation'] = fleet
            if timings is not None: result['timings'] = timings
            return result
        if value.get('error') is not None: raise ValueError('stream_provider_exception')
        # Keep the latest provider timing snapshot; never add cumulative chunks.
        # Optional telemetry must not make an otherwise valid reply fail.
        if isinstance(value.get('timings'), dict): timings = value['timings']
        if value.get('model') is not None:
            if model is not None and model != value['model']: raise ValueError('stream_model_changed')
            model = text(value['model'])
        if value.get('tare_fleet_observation') is not None:
            if fleet is not None: raise ValueError('stream_duplicate_fleet_receipt')
            fleet = value['tare_fleet_observation']
        if value.get('usage') is not None:
            if usage is not None and usage != value['usage']: raise ValueError('stream_usage_changed')
            usage = value['usage']; emit({'kind':'usage','usage':usage})
        choices = value.get('choices', [])
        if not isinstance(choices,list) or len(choices)>1: raise ValueError('stream_one_choice_required')
        if not choices: continue
        choice = choices[0]
        if choice.get('index') != 0: raise ValueError('stream_choice_order')
        delta = choice.get('delta', {})
        if finish is not None:
            # OpenRouter repeats the terminal choice on its usage-only frame.
            # Accept accounting once, never more content, tools or opaque deltas.
            if (accounted or not isinstance(value.get('usage'), dict) or choice.get('finish_reason') != finish
                    or not isinstance(delta, dict) or set(delta) - {'role', 'content'}
                    or delta.get('role', 'assistant') != 'assistant' or delta.get('content') not in (None, '')):
                raise ValueError('stream_choice_order')
            accounted = True
            continue
        if delta.get('role','assistant') != 'assistant': raise ValueError('stream_role')
        if delta.get('function_call') is not None: raise ValueError('stream_legacy_function_unsupported')
        _opaque(assistant, delta, {'role', 'content', 'reasoning_content', 'reasoning', 'reasoning_details', 'refusal', 'tool_calls'})
        if delta.get('reasoning_details') is not None:
            details = delta['reasoning_details']
            if not isinstance(details, list) or any(not isinstance(item, dict) for item in details):
                raise ValueError('stream_reasoning_details_shape')
            _reasoning_fragments(assistant.setdefault('reasoning_details', []), details)
        for key in ('content','reasoning_content','reasoning','refusal'):
            if delta.get(key) is not None:
                fragment = text(delta[key]); assistant[key] = assistant.get(key,'') + fragment
                if fragment: emit({'kind':'delta','channel':'text' if key=='content' else key,'text':fragment})
        for call in delta.get('tool_calls', []):
            index = call.get('index')
            if type(index) is not int or not 0 <= index < 32: raise ValueError('stream_tool_index')
            target = tools.setdefault(index, {'id':'','type':'function','function':{'name':'','arguments':''}})
            if call.get('type','function') != 'function': raise ValueError('stream_tool_type')
            _opaque(target, call, {'index', 'id', 'type', 'function'})
            _opaque(target['function'], call.get('function', {}), {'name', 'arguments'})
            if call.get('id') is not None: target['id'] += text(call['id'])
            for key in ('name','arguments'):
                if call.get('function',{}).get(key) is not None:
                    fragment = text(call['function'][key]); target['function'][key] += fragment
                    if fragment: emit({'kind':'delta','channel':'tool_'+key,'index':index,'text':fragment})
        if choice.get('finish_reason') is not None: finish = text(choice['finish_reason'])
    raise ValueError('stream_missing_done')


def converse(stream, emit):
    started, stopped, metadata = False, None, None
    blocks, closed = {}, set()
    for kind, value in eventstream(stream):
        if kind == 'messageStart':
            if started or value.get('role') != 'assistant': raise ValueError('stream_message_start')
            started = True; continue
        if not started: raise ValueError('stream_missing_start')
        if kind == 'metadata':
            if stopped is None or metadata is not None: raise ValueError('stream_metadata_order')
            metadata = value
            if value.get('usage') is not None: emit({'kind':'usage','usage':value['usage']})
            continue
        if stopped is not None: raise ValueError('stream_after_message_stop')
        if kind == 'messageStop':
            if set(blocks) != closed: raise ValueError('stream_unclosed_blocks')
            stopped = text(value['stopReason']); continue
        index = value.get('contentBlockIndex')
        if type(index) is not int or not 0 <= index < 128 or index in closed:
            raise ValueError('stream_block_index')
        if kind == 'contentBlockStart':
            if index in blocks: raise ValueError('stream_duplicate_block')
            start = value['start']
            if set(start) != {'toolUse'}: raise ValueError('stream_unsupported_block_start')
            tool = start['toolUse']
            blocks[index] = {'toolUse':{'toolUseId':text(tool['toolUseId']),'name':text(tool['name']),'input':''}}
        elif kind == 'contentBlockDelta':
            delta = value['delta']
            if len(delta) != 1: raise ValueError('stream_delta_union')
            key = next(iter(delta))
            block = blocks.setdefault(index, {key: '' if key=='text' else {}})
            if key not in block: raise ValueError('stream_block_type_changed')
            if key == 'text':
                fragment = text(delta[key]); block[key] += fragment
                if fragment: emit({'kind':'delta','channel':'text','text':fragment})
            elif key == 'toolUse':
                if 'input' not in block[key]: raise ValueError('stream_tool_start_missing')
                fragment = text(delta[key]['input']); block[key]['input'] += fragment
                if fragment: emit({'kind':'delta','channel':'tool_arguments','index':index,'text':fragment})
            elif key == 'reasoningContent':
                content = delta[key]
                if len(content) != 1: raise ValueError('stream_reasoning_union')
                part = next(iter(content)); fragment = text(content[part])
                if part == 'redactedContent':
                    if 'reasoningText' in block[key]: raise ValueError('stream_reasoning_type_changed')
                    previous = base64.b64decode(block[key].get(part,''),validate=True)
                    block[key][part] = base64.b64encode(previous+base64.b64decode(fragment,validate=True)).decode()
                elif part in {'text','signature'}:
                    if 'redactedContent' in block[key]: raise ValueError('stream_reasoning_type_changed')
                    target = block[key].setdefault('reasoningText',{})
                    target[part] = target.get(part,'') + fragment
                    if part=='text' and fragment: emit({'kind':'delta','channel':'reasoning_content','text':fragment})
                else: raise ValueError('stream_unsupported_reasoning')
            else: raise ValueError('stream_unsupported_delta')
        elif kind == 'contentBlockStop':
            if index not in blocks: raise ValueError('stream_unknown_block')
            if 'toolUse' in blocks[index]: blocks[index]['toolUse']['input'] = obj(blocks[index]['toolUse']['input'])
            closed.add(index)
        else: raise ValueError('stream_unsupported_event')
    if not started or stopped is None: raise ValueError('stream_missing_completion')
    return {'output':{'message':{'role':'assistant','content':[blocks[i] for i in sorted(blocks)]}},
            'stopReason':stopped, **(metadata or {})}


def decode(stream, protocol, emit):
    if protocol in {'openai-codex-responses', 'openai-responses'}:
        if __package__:
            from .native_responses_stream import decode as responses
        else:
            from native_responses_stream import decode as responses
        return responses(sse(stream), emit)
    if protocol not in {'openai-chat', 'bedrock-converse'}:
        raise ValueError('unsupported_stream_protocol')
    return (chat if protocol=='openai-chat' else converse)(stream,emit)
