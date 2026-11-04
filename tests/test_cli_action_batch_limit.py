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
