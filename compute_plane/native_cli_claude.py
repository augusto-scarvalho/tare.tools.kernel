"""One quoted CLI input, with stable provider-cache blocks and private-free diagnostics."""
import json
import re

from tare_tools_agent_runtime import ConversationRejected

from .native_models import encode, object_json


def input_record(body):
    """A single user message, never a stream of executable historical turns."""
    prompt = body['prompt']
    try:
        packet = json.loads(prompt)
    except (ValueError, TypeError):
        packet = None
    key = next((k for k in ('messages', 'conversation_records')
                if isinstance(packet, dict) and isinstance(packet.get(k), list)), None)
    if key is None:
        blocks = [{'type': 'text', 'text': prompt}]
        boundary = 0
    else:
        records = packet[key]
        header = {k:v for k,v in packet.items() if k not in {key, 'turn_constraints'}}
        prefix = encode(header)[:-1] + (',' if header else '') + json.dumps(key) + ':['
        blocks = [{'type':'text', 'text':prefix}]
        boundary = 0
        for index, record in enumerate(records):
            blocks.append({'type':'text', 'text':(',' if index else '') + encode(record)})
            # The host appends its changing call allowance after stable history.
            # Excluding it changes cache placement only, never instructions/scope.
            budget = (index == len(records)-1 and isinstance(record, dict)
                      and record.get('role') == 'user' and isinstance(record.get('content'), str)
                      and record['content'].startswith('Runtime budget: '))
            if not budget:
                boundary = len(blocks)-1
        tail = ']' + (',"turn_constraints":' + encode(packet['turn_constraints'])
                      if 'turn_constraints' in packet else '') + '}'
        blocks.append({'type':'text', 'text':tail})
    # Claude Code's observed automatic breakpoints use 1h. A 5m breakpoint
    # before its later 1h breakpoint is rejected by the API's TTL ordering rule.
    blocks[boundary]['cache_control'] = {'type':'ephemeral','ttl':'1h'}
    if body.get('reminder'):
        blocks.append({'type':'text', 'text':body['reminder']})
    return {'type':'user', 'session_id':'', 'parent_tool_use_id':None,
            'message':{'role':'user', 'content':blocks}}


def rejection_reason(text, *, structured=False):
    """Only finite schema categories; never names, arguments or reply text."""
    try:
        value = object_json(text)
    except (ValueError, TypeError, ConversationRejected):
        return 'INVALID_JSON'
    calls = 'actions' if structured else 'tool_calls'
    if set(value) != {'text', calls}:
        return 'ENVELOPE_KEYS'
    if not isinstance(value['text'], str):
        return 'TEXT_TYPE'
    if not isinstance(value[calls], list) or len(value[calls]) > 16:
        return 'CALLS_TYPE_OR_LIMIT'
    for call in value[calls]:
        required = {'name','arguments_json'} if structured else {'id','name','arguments'}
        if not isinstance(call, dict) or set(call) != required:
            return 'CALL_KEYS'
        if not isinstance(call['name'], str) or not call['name']:
            return 'CALL_NAME_TYPE'
        if not structured and (not isinstance(call['id'], str) or not call['id']):
            return 'CALL_ID_TYPE'
        if not structured and not isinstance(call['arguments'], dict):
            return 'ARGUMENTS_TYPE'
    return 'SCHEMA_OTHER'


def json_rejection_details(text):
    """Finite parser diagnostics; no source text, field names or exception messages."""
    try:
        object_json(text)
    except (ValueError,TypeError,ConversationRejected) as error:
        cause=error.__cause__
        if isinstance(cause,json.JSONDecodeError):
            categories={'Expecting value':'EXPECTED_VALUE',
                'Expecting property name enclosed in double quotes':'OBJECT_KEY',
                "Expecting ',' delimiter":'MISSING_COMMA',"Expecting ':' delimiter":'MISSING_COLON',
                'Extra data':'EXTRA_DATA','Unterminated string starting at':'UNTERMINATED_STRING',
                'Invalid \\escape':'INVALID_ESCAPE','Invalid \\uXXXX escape':'INVALID_UNICODE_ESCAPE',
                'Invalid control character at':'CONTROL_CHARACTER'}
            return {'category':categories.get(cause.msg,'JSON_SYNTAX'),
                    'position':cause.pos,'line':cause.lineno,'column':cause.colno}
        return {'category':'EVENT_REJECTED'}
    return {}


def provider_failure(text):
    """Classify synthetic API refusals without retaining their private text."""
    value=str(text).lower()
    status=re.search(r'api error:\s*(\d{3})',value)
    flags=[word for word in ('cache_control','breakpoint','maximum','at most','too many',
        'extra inputs','not permitted','thinking','redacted_thinking','content','session_id',
        'permission','rate limit','invalid_request_error','oauth','credential','credit',
        'ttl','1h','5m','precede') if word in value]
    return {'status':int(status[1]) if status else None,'markers':flags}
