"""AGY and Claude Code official lifecycle hooks. No task tool is executed by this process."""
import json
from pathlib import Path
import sys


def main():
    target, phase = Path(sys.argv[1]), sys.argv[2]
    # Bytes, not text: under -I the locale codec (cp1252 on Windows) would corrupt UTF-8.
    event = json.loads(sys.stdin.buffer.read())
    if phase == 'claude':
        # Claude Code PreToolUse: record the native call for Tare, refuse it and end the turn.
        with target.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': phase, 'tool': event.get('tool_name'),
                                     'input': event.get('tool_input')})+'\n')
        print(json.dumps({'continue': False, 'stopReason': 'Tare executes this action.',
                          'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny',
                                                 'permissionDecisionReason': 'Tare executes this action.'}}))
        return
    row = {'phase': phase, 'model': event.get('modelName'),
           'invocation': event.get('invocationNum'), 'termination': event.get('terminationReason'),
           'tool': event.get('toolCall', {}).get('name') if phase == 'tool' else None}
    if phase == 'tool':
        row['decision'] = 'deny'
        # AGY's schema formatter can emit the final decision through finish.
        # Capture only that completed proposal; task tools remain denied.
        args = event.get('toolCall', {}).get('args')
        if (row['tool'] == 'finish' and isinstance(args, dict)
                and {'text', 'actions'} <= args.keys()
                and not args.keys() - {'text', 'actions', 'toolAction', 'toolSummary'}):
            row['proposal'] = {key: args[key] for key in ('text', 'actions')}
    with target.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    result = ({'terminationBehavior': 'terminate'} if phase == 'post' else
              {'decision': 'deny', 'reason': 'Tare owns tool execution.'} if phase == 'tool' else
              {'decision': 'stop'} if phase == 'stop' else {})
    print(json.dumps(result))


if __name__ == '__main__':
    main()
