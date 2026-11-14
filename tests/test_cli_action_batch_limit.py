import json

import pytest

from compute_plane import native_cli
from tare_tools_agent_runtime import ModelFailure


def test_excess_actions_are_refused_as_a_batch_with_a_specific_diagnostic():
    value={'text':'','actions':[{'name':'read','arguments_json':'{}'}]*18}
    with pytest.raises(ModelFailure,match='CLI_ACTION_LIMIT_EXCEEDED') as caught:
        native_cli.action_proposal(value,'fixture',None)
    assert caught.value.proposed_action_count == 18


def test_agy_batch_guidance_is_a_suffix_without_rewriting_cached_records():
    records=[{'role':'user','content':'Audit the current implementation.'}]
    body=native_cli.cli_request({'model':'gemini-4-argon-high','messages':records,'tools':[
        {'type':'function','function':{'name':'read','parameters':{'type':'object'}}}]},structured=True)
    assert 'at most 4 independent actions' in body['reminder']
    assert body['response_schema']['properties']['actions']['maxItems']==4
    assert 'at most 4 independent actions' not in body['system']
    assert json.loads(body['prompt'])['conversation_records']==records


def test_sonnet_on_agy_is_told_which_names_are_not_functions_and_no_other_route_is():
    # 2026-11-14, six real implementer sessions each: 8 of 26 calls repaired without this text, 0 of 30 with it.
    tools=[{'type':'function','function':{'name':name,'parameters':{'type':'object'}}} for name in ('read','tare_deliver')]
    def reminder(model,**options):
        return native_cli.cli_request({'model':model,'messages':[],'tools':tools},structured=True,**options).get('reminder','')
    told=reminder('claude-sonnet-5-5-high')
    assert 'read, tare_deliver are not functions' in told and told.endswith('"{\\"summary\\":\\"The actual answer.\\"}"}]}.')
    assert 'finish is the only function' not in reminder('claude-opus-5-5-high')
    assert 'finish is the only function' not in reminder('gemini-4-hydrogen-high')
    assert 'finish is the only function' not in reminder('claude-sonnet-5-5',formatter=True)  # Claude Code's own path
