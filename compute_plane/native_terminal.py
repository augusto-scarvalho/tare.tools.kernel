"""Explicit native tool port to an already-authorized local terminal grant."""
import json
import time
import hashlib

from tare_tools_agent_runtime import EffectResult
from .terminal_host import TerminalRefused, TerminalTransportError


def _spec(description, properties, required):
    return dict(description=description, parameters=dict(type='object', properties=properties,
                required=required, additionalProperties=False))


TEXT = {'type':'string'}
HANDLE = {'handle':TEXT}
TERMINAL_SPECS = {
    'terminal_start': _spec('Start an authorized host command in a selected project and return a process handle. '
        'This can change files or external systems. argv runs directly, without an implicit shell. '
        'Use the host execution_environment shell_argv for shell syntax and its tool paths. '
        'PowerShell file cmdlets need absolute -LiteralPath; build paths with Join-Path and (Get-Location).ProviderPath. '
        'After Set-Location, [Environment]::CurrentDirectory may still name the original process directory. '
        'Read UTF-8 files with -Encoding UTF8. '
        'Use terminal_read to observe completion; starting is not successful task completion. '
        'Choose pty for interactive programs, pipes for ordinary tests/builds. '
        'timeout_seconds null keeps a background command alive until exit, stop or host closure.',
        {'repository':TEXT, 'argv':{'type':'array','items':TEXT,'minItems':1,'maxItems':256},
         'mode':{'type':'string','enum':['pipes','pty'],'default':'pipes'},
         'timeout_seconds':{'type':['integer','null'],'minimum':1,'default':300},
         'columns':{'type':'integer','minimum':1,'maximum':1000,'default':80},
         'rows':{'type':'integer','minimum':1,'maximum':1000,'default':24}}, ['repository','argv']),
    'terminal_read': _spec('Read bounded live output and the observed process state. '
        'Cursors count decoded characters separately per stream, not bytes. '
        'Copy next_read for the next request; null means the finished command has no unread output. '
        'Never guess future offsets or batch speculative pages: a cursor beyond the observed end is invalid. '
        'A running command has no final exit result; truncated output explicitly reports lost history. '
         'For ordinary tests/builds, set wait_for_exit=true to wait within wait_ms for completion instead of returning on early progress output.',
        {**HANDLE, 'stdout_cursor':{'type':'integer','minimum':0},
         'stderr_cursor':{'type':'integer','minimum':0},
         'limit':{'type':'integer','minimum':1,'maximum':16384,'default':4096},
         'wait_ms':{'type':'integer','minimum':0,'maximum':30000,'default':1000},
         'wait_for_exit':{'type':'boolean','default':False}}, ['handle']),
    'terminal_input': _spec('Queue text or terminal control characters to the process. '
        'queued/written acknowledges transport, not that the program acted on it.',
        {**HANDLE,'text':TEXT}, ['handle','text']),
    'terminal_stop': _spec('Stop this session-owned command and observe descendant cleanup. '
        'Stopping does not undo files or external effects.', HANDLE, ['handle']),
    'terminal_list': _spec('List this session’s processes without other sessions’ handles or output.',
        {'include_completed':{'type':'boolean','default':False}}, []),
    'terminal_resize': _spec('Resize a live PTY in character columns and rows.',
        {**HANDLE,'columns':{'type':'integer','minimum':1,'maximum':1000},
         'rows':{'type':'integer','minimum':1,'maximum':1000}}, ['handle','columns','rows']),
    'terminal_eof': _spec('Close stdin after queued input in pipe mode. '
        'For a PTY use the program’s exit/control keys or terminal_stop.', HANDLE, ['handle']),
}
OPERATIONS = {'terminal_start':'start', 'terminal_read':'read', 'terminal_input':'send',
              'terminal_stop':'stop', 'terminal_list':'list', 'terminal_resize':'resize', 'terminal_eof':'close_input'}
MUTATIONS = {'terminal_start', 'terminal_input', 'terminal_stop', 'terminal_eof', 'terminal_resize'}


def validate(name, args):
    schema = TERMINAL_SPECS[name]['parameters']
    if not isinstance(args, dict) or set(args)-schema['properties'].keys() or set(schema['required'])-args.keys():
        raise ValueError('terminal_tool_arguments_invalid')
    for key, value in args.items():
        spec = schema['properties'][key]
        kinds = spec['type'] if isinstance(spec['type'], list) else [spec['type']]
        matches = {'string':isinstance(value,str), 'integer':type(value) is int,
                   'null':value is None, 'boolean':type(value) is bool, 'array':isinstance(value,list)}
        if not any(matches.get(kind, False) for kind in kinds):
            raise ValueError('terminal_tool_argument_type')
        if value is None:
            continue
        if 'enum' in spec and value not in spec['enum']:
            raise ValueError('terminal_tool_argument_enum')
        if type(value) is int and (value < spec.get('minimum',value) or value > spec.get('maximum',value)):
            raise ValueError('terminal_tool_argument_bounds')
        if isinstance(value,list) and (not 1 <= len(value) <= 256 or any(not isinstance(v,str) for v in value)):
            raise ValueError('terminal_tool_argv_invalid')


def _stable(value):
    if isinstance(value, dict):
        return {k:_stable(v) for k,v in value.items() if k not in {'elapsed_ms','next_read'}}
    if isinstance(value,list):
        return [_stable(v) for v in value]
    return value


class NativeTerminal:
    """Runtime-compatible port; no discovery, environment reads or host creation."""

    def __init__(self, *, session_id, binding, client, execution_environment=None):
        self.session_id, self.binding, self.client = session_id, binding, client
        self.execution_environment = execution_environment
        self._seen, self._observations = set(), []

    def observations(self):
        return list(self._observations)

    def execute(self, request, control):
        name = request.tool_call.name
        output, status, progressed = {}, 'refused', False
        began = time.monotonic()
        try:
            if request.session_id != self.session_id or control.binding != self.binding or control.should_stop():
                raise ValueError('terminal_control_or_session_mismatch')
            args = json.loads(request.tool_call.arguments_json)
            if name not in TERMINAL_SPECS:
                raise ValueError('terminal_tool_not_admitted')
            validate(name, args)
            if name == 'terminal_start':
                args['request_id'] = request.effect_id
                if 'timeout_seconds' in args:
                    args['timeout'] = args.pop('timeout_seconds')
            elif name == 'terminal_input':
                args['request_id'] = request.effect_id
            wait_ms = 0
            if name == 'terminal_read':
                wait_ms = args.pop('wait_ms',1000)
                wait_for_exit = args.pop('wait_for_exit',False)
                args['limit'] = args.get('limit',4096)
                args['cursors'] = {s:args.pop(s+'_cursor',0) for s in ('stdout','stderr')}
            operation = OPERATIONS[name]
            deadline = time.monotonic()+wait_ms/1000
            while True:
                if control.should_stop():
                    raise ValueError('terminal_control_stopped')
                output = self.client.request(operation, transport_timeout=min(20, control.remaining_seconds()), **args)
                if (name != 'terminal_read' or output.get('status') != 'running'
                        or any(s['more'] or s['truncated'] or len(s['text']) >= args['limit']
                               for s in output['output'].values())
                        or (not wait_for_exit and any(s['text'] for s in output['output'].values()))
                        or time.monotonic() >= deadline):
                    break
                time.sleep(min(.05, max(0,deadline-time.monotonic())))
            output = {'input_closed':True} if output is None else output
            if isinstance(output,list):
                output = {'processes':output}
            if name == 'terminal_read':
                output['read_wait_expired'] = output.get('status') == 'running' and time.monotonic() >= deadline
                pages=output.get('output')
                if isinstance(pages,dict) and all(isinstance(pages.get(s),dict) and 'cursor' in pages[s] for s in ('stdout','stderr')):
                    more=any(page['more'] for page in pages.values())
                    output['next_read']=(dict(handle=args['handle'],stdout_cursor=pages['stdout']['cursor'],
                        stderr_cursor=pages['stderr']['cursor'],limit=args['limit'],wait_ms=wait_ms,
                        wait_for_exit=wait_for_exit) if more or output.get('status')=='running' else None)
                    if more and output.get('status')!='running':
                        output['next_action']='The command finished; consume the remaining output with next_read.'
                if output.get('status') == 'running':
                    output['next_action'] = 'Read this handle again using next_read; the command has not completed. Do not restart it.'
            status = 'succeeded'
            if isinstance(output,dict):
                if output.get('status') == 'unknown' or output.get('state') == 'unknown':
                    status = 'unknown'
                elif output.get('reason') == 'start_failed':
                    status = 'unknown' if output.get('may_have_executed') else 'failed'
                elif name in {'terminal_read','terminal_start'} and output.get('status') == 'exited' and output.get('exit_code') != 0:
                    status = 'failed'
                elif name in {'terminal_read','terminal_start'} and output.get('status') == 'stopped':
                    status = 'failed'
                elif name == 'terminal_stop' and not output.get('cleanup_verified'):
                    status = 'unknown'
            observed = hashlib.sha256(json.dumps([name,_stable(output)],sort_keys=True).encode()).hexdigest()
            progressed = status == 'succeeded' and observed not in self._seen
            if name == 'terminal_read' and output.get('status') == 'running' and time.monotonic()-began >= .1:
                # A bounded, verified wait for a still-owned process is useful
                # continuation; polling without waiting is still deduplicated.
                output['waited_ms'] = round((time.monotonic()-began)*1000)
                progressed = True
            if len(self._seen) >= 4096:
                self._seen.clear()
            self._seen.add(observed)
        except TerminalTransportError:
            status = 'unknown' if name in MUTATIONS else 'refused'
            output = {'reason':'terminal_transport_inconclusive', 'automatic_retry':False}
        except (ValueError, TypeError, KeyError) as exc:
            output = {'reason':str(exc)[:160] if isinstance(exc, (ValueError, TerminalRefused))
                      else 'terminal_tool_arguments_invalid'}
        self._observations.append(dict(effect_id=request.effect_id, operation=name, status=status,
                                       duration_ms=round((time.monotonic()-began)*1000)))
        self._observations = self._observations[-128:]
        return EffectResult(request.effect_id,status,json.dumps(output,ensure_ascii=True),progressed)
