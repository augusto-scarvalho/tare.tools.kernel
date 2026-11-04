import copy
import pytest
from compute_plane.native_anthropic import to_messages


def test_history_cache_ends_before_the_changing_budget_and_preserves_signed_blocks():
    thinking={'type':'thinking','thinking':'private thought','signature':'signed'}
    original={'model':'fixture','max_tokens':1024,'stream':False,'cache_control':{'type':'ephemeral','ttl':'5m'},
        'messages':[{'role':'system','content':'Stable policy'}, {'role':'user','content':'Stable goal'},
        {'role':'assistant','content':'','_anthropic_content':[thinking,{'type':'tool_use','id':'t','name':'read','input':{}}]},
        {'role':'tool','tool_call_id':'t','content':'Observed source'},
        {'role':'user','content':'Runtime budget: 3 calls remain'}]}
    baseline=copy.deepcopy(original);first=to_messages(original,cache_system=True)
    changed=copy.deepcopy(original);changed['messages'][-1]['content']='Runtime budget: 2 calls remain'
    second=to_messages(changed,cache_system=True)
    assert first['messages'][-1]['content'][0]==second['messages'][-1]['content'][0]
    assert first['messages'][-1]['content'][0]['cache_control']=={'type':'ephemeral','ttl':'5m'}
    assert 'cache_control' not in first['messages'][-1]['content'][-1]
    assert first['messages'][1]['content'][0]==thinking
    assert original==baseline


@pytest.mark.parametrize('count', [1,4])
def test_explicit_history_controls_remain_owned_by_the_caller(count):
    controls=[{'type':'text','text':str(i),'cache_control':{'type':'ephemeral','ttl':'1h'}} for i in range(count)]
    wire=to_messages({'model':'fixture','max_tokens':1024,'messages':[{'role':'system','content':'p'},
        {'role':'user','content':controls}]},cache_system=True)
    assert wire['messages'][0]['content']==controls
    assert 'cache_control' not in wire['system'][0]
