"""AGY 1.4.10 can render schema completion as finish without a tool hook."""
import json

import pytest

from compute_plane.native_cli import decode_agy
from tare_tools_agent_runtime import ModelFailure


@pytest.mark.parametrize('damage', [None, 'task_tool', 'missing_stop', 'extra_inference', 'bad_schema', 'error', 'legacy'])
def test_finish_display_requires_the_complete_typed_reply_and_lifecycle(damage):
    hooks = [{'phase':p,'model':'fixture','invocation':0,
              'termination':'TERMINAL_CUSTOM_HOOK' if p == 'stop' else None}
             for p in ('pre','post','stop')]
    value = {'text': 'Complete', 'actions': []}
    rows = [{'event':'step_update','step_update':{'tool_info':{'name':'finish'}}},
            {'event':'result','result':{'status':'SUCCESS','response':json.dumps(value)}}]
    if damage == 'task_tool':
        rows.insert(0, {'event':'step_update','step_update':{'tool_info':{'name':'read'}}})
    if damage == 'missing_stop':hooks.pop()
    if damage == 'extra_inference':hooks.insert(1,hooks[0])
    if damage == 'bad_schema':rows[-1]['result']['response'] = '{}'
    if damage == 'error':rows[-1]['result']['status'] = 'ERROR'
    if damage is None:
        response = decode_agy(rows,hooks,'fixture',structured=True)
        assert response['choices'][0]['message']['content'] == 'Complete'
    else:
        with pytest.raises((ModelFailure,ValueError)):
            decode_agy(rows,hooks,'fixture',structured=damage != 'legacy')
