import json
import pytest
from compute_plane import native_cli
from tests.test_native_cli import structured_claude_rows
from tare_tools_agent_runtime import ModelFailure


@pytest.mark.parametrize('model,formatter,summary,expected',[
    ('claude-haiku-5-5',True,False,True),('claude-sonnet-5-5',True,False,False),
    ('claude-haiku-5-5',False,False,False),('claude-haiku-5-5',True,True,False)])
def test_search_example_is_only_a_tail_reminder_for_haiku_formatter(model,formatter,summary,expected):
    records=[{'role':'user','content':'Procure o ponto de entrada sem executar ferramentas nativas.'}]
    chat={'model':model,'messages':records,'tools':[{'type':'function','function':{'name':'search',
          'parameters':{'type':'object'}}}]}
    body=native_cli.cli_request(chat,structured=True,formatter=formatter,summary=summary)
    assert (native_cli.HAIKU_SEARCH_GUIDANCE in body.get('reminder','')) is expected
    assert native_cli.HAIKU_SEARCH_GUIDANCE not in body['system']
    assert native_cli.HAIKU_SEARCH_GUIDANCE not in body['prompt']
    assert json.loads(body['prompt'])['messages' if summary else 'conversation_records']==records


def test_native_search_then_valid_formatter_is_still_rejected():
    rows=structured_claude_rows()
    rows[1:1]=[{'type':'assistant','message':{'id':'native-search','model':'fixture','content':[
        {'type':'tool_use','id':'search-1','name':'search','input':{'query':'needle'}}]}},
        {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'search-1',
          'is_error':True,'content':'Unknown tool'}]}}]
    rows[-1]['num_turns']=3
    with pytest.raises(ModelFailure,match='CLI_UNEXPECTED_INVOCATION'):
        native_cli.decode_claude(rows,'fixture',structured=True)
