"""Automatic guidance precedes effects and survives journal compaction/reopen."""
import hashlib
import json
import time

import pytest
from compute_plane.native_capabilities import NativeCapabilities
from compute_plane.native_checkpoint import operational_checkpoint
from tare_tools_agent_runtime import BindingRef, CancellationToken, ConversationLog, ExecutionControl, EffectRequest, ToolCall


def host(tmp_path):
    root, scratch = tmp_path/'project', tmp_path/'scratch'
    root.mkdir(); scratch.mkdir()
    (root/'src').mkdir(); (root/'src'/'deep').mkdir()
    (root/'src'/'AGENTS.md').write_text('Use parent convention.', encoding='utf-8')
    (root/'src'/'deep'/'AGENTS.md').write_text('Use child convention here.', encoding='utf-8')
    (root/'src'/'deep'/'value.txt').write_text('one', encoding='utf-8')
    binding = BindingRef('work','binding','owner',1,'token')
    control = ExecutionControl(binding,CancellationToken(),time.monotonic()+30)
    def create():
        return NativeCapabilities(session_id='s',binding=binding,repositories={'repo':str(root)},
            writable={'repo':['src']},verifiers={},scratch_root=scratch,
            specgraph_edit_context=False,scoped_instructions=True)
    return root,create,control


def invoke(tools,control,log,name,**args):
    eid='s/effect/'+str(log.snapshot().revision)
    call=ToolCall('t',name,json.dumps(args))
    log.append({'kind':'effect_requested','effect_id':eid,'tool_call':{'name':name,'arguments_json':call.arguments_json}})
    result=tools.execute(EffectRequest('s',eid,call),control)
    log.append({'kind':'effect_returned','result':{'effect_id':eid,'status':result.status,'output_json':result.output_json}})
    return result,json.loads(result.output_json)


def test_read_delivers_inherited_rules_edit_and_reopen(tmp_path):
    root,create,control=host(tmp_path);tools=create();log=ConversationLog()
    result,data=invoke(tools,control,log,'read',repository='repo',path='src/deep/value.txt')
    assert result.status=='succeeded'
    assert [r['path'] for r in data['scoped_project_instructions']]==['src/AGENTS.md','src/deep/AGENTS.md']
    result,_=invoke(tools,control,log,'edit',repository='repo',path='src/deep/value.txt',sha256=data['sha256'],search='one',replace='two')
    assert result.status=='succeeded' and (root/'src/deep/value.txt').read_text()=='two'
    reopened=create();reopened.restore(log)
    assert reopened.instructions.records()==tools.instructions.records()
    # Rules are retained independently of a model's history summary and the 12-row facts limit.
    for _ in range(18): invoke(tools,control,log,'list',repository='repo',path='.')
    checkpoint=json.loads(operational_checkpoint(log.snapshot()))
    assert [r['text'] for r in checkpoint['scoped_project_instructions']]==['Use parent convention.','Use child convention here.']


def test_first_edit_returns_rules_without_touching_new_file(tmp_path):
    root,create,control=host(tmp_path);tools=create();log=ConversationLog()
    args=dict(repository='repo',path='src/new.txt',sha256=None,search='',replace='new')
    result,data=invoke(tools,control,log,'edit',**args)
    assert result.status=='refused' and not (root/'src/new.txt').exists()
    assert data['scoped_project_instructions'][0]['text']=='Use parent convention.'
    assert tools.observations()['effects']==0
    result,_=invoke(tools,control,log,'edit',**args)
    assert result.status=='succeeded' and (root/'src/new.txt').read_text()=='new'


@pytest.mark.parametrize('change',['change','delete','appear'])
def test_changed_nested_rules_refuse_edit_and_resume(tmp_path,change):
    root,create,control=host(tmp_path);tools=create();log=ConversationLog()
    path=root/'src/deep/AGENTS.md'
    if change=='appear': path.unlink()
    _,data=invoke(tools,control,log,'read',repository='repo',path='src/deep/value.txt')
    if change=='delete': path.unlink()
    else: path.write_text('Different rules',encoding='utf-8')
    result,_=invoke(tools,control,log,'edit',repository='repo',path='src/deep/value.txt',sha256=data['sha256'],search='one',replace='two')
    assert result.status=='refused' and (root/'src/deep/value.txt').read_text()=='one'
    with pytest.raises(ValueError): create().restore(log,refresh=True)


def test_unrelated_instruction_changes_do_not_block_scoped_edit(tmp_path):
    root,create,control=host(tmp_path);tools=create();log=ConversationLog()
    (root/'other').mkdir();(root/'other/AGENTS.md').write_text('Unrelated',encoding='utf-8')
    _,data=invoke(tools,control,log,'read',repository='repo',path='src/deep/value.txt')
    (root/'other/AGENTS.md').write_text('Changed unrelated',encoding='utf-8')
    result,_=invoke(tools,control,log,'edit',repository='repo',path='src/deep/value.txt',sha256=data['sha256'],search='one',replace='two')
    assert result.status=='succeeded'
    create().restore(log)


def test_actual_compaction_keeps_rules_omitted_by_model_summary(tmp_path):
    from tare_tools_agent_runtime import ModelReply, NativeBudget, NativeContext, ContextPolicy, NativeSession
    root,create,control=host(tmp_path);tools=create();log=ConversationLog()
    class Model:
        count=0
        final=None
        def generate(self,request,execution):
            if request.purpose=='summary':
                return ModelReply(text='Earlier file inspected. No instruction text retained in this model summary.')
            self.count+=1
            if self.count==1:
                return ModelReply(tool_calls=(ToolCall('read','read',json.dumps({'repository':'repo','path':'src/deep/value.txt'})),))
            if self.count==2:
                return ModelReply(tool_calls=(ToolCall('list','list',json.dumps({'repository':'repo','path':'.'})),))
            self.final=request.context.events()
            return ModelReply(text='Completed.',proposes_delivery=True)
    model=Model()
    def counter(request,execution):
        events=request.context.events()
        active=any(e['kind']=='context_activated' for e in events)
        count=500 if active or request.purpose=='summary' else 8500 if sum(e['kind']=='model_returned' for e in events)>=2 else 100
        return {'status':'observed','input_tokens':count,'method':'fixture-counter',
            'context_sha256':request.context.sha256,'request_sha256':hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest(),
            'profile_digest':'a'*64,'context_window':10000,'output_tokens':1000,'request_purpose':request.purpose}
    session=NativeSession(session_id='s',instruction='Inspect the source',capabilities=tools.names,
        required_context=('Root rules',),model=model,tools=tools,journal=log,budget=NativeBudget(),
        context=NativeContext(ContextPolicy(10000,1000,20),counter=counter,summarizer=model,checkpoint=operational_checkpoint))
    assert session.run(control).status=='proposed'
    activated=next(e for e in model.final if e['kind']=='context_activated')
    assert 'child convention' not in activated['summary']
    pinned=json.loads(activated['observed_state'])['scoped_project_instructions']
    assert [r['text'] for r in pinned]==['Use parent convention.','Use child convention here.']
    # The retained tool turn is only the later root listing, so protection comes from the checkpoint.
    retained=[json.loads(e['result']['output_json']) for e in model.final if e['kind']=='effect_returned']
    assert len(retained)==1 and retained[0]['scoped_project_instructions']==[]


def test_a_dot_slash_path_reads_the_same_file_and_reopens(tmp_path):
    root,create,control=host(tmp_path);tools=create();log=ConversationLog()
    result,data=invoke(tools,control,log,'read',repository='repo',path='./src/deep/value.txt')
    assert result.status=='succeeded' and data['text'].strip().endswith('one')
    assert tools.observations()['reads']=={'repo:src/deep/value.txt':data['sha256']}
    result,_=invoke(tools,control,log,'list',repository='repo',path='./')
    assert result.status=='succeeded'
    reopened=create();reopened.restore(log)
    assert reopened.observations()['reads']==tools.observations()['reads']
    result,_=invoke(tools,control,log,'read',repository='repo',path='./../outside.txt')
    assert result.status=='refused'


def test_unambiguous_argument_slips_are_corrected_instead_of_refused(tmp_path):
    root,create,control=host(tmp_path);tools=create();log=ConversationLog()
    result,data=invoke(tools,control,log,'read',repository='repo',path='. /src/deep/value.txt')
    assert result.status=='succeeded' and data['text'].strip().endswith('one')
    result,data=invoke(tools,control,log,'read',repository='repo',file='src/deep/value.txt')
    assert result.status=='succeeded'


def test_a_file_path_written_as_the_repository_is_read_there_and_reopens_the_same(tmp_path):
    root,create,control=host(tmp_path);tools=create();log=ConversationLog()
    result,data=invoke(tools,control,log,'read',repository='src/deep/value.txt',start=1,end=1)
    assert result.status=='succeeded' and data['text'].strip().endswith('one')
    assert list(tools.observations()['reads'])==['repo:src/deep/value.txt']
    reopened=create();reopened.restore(log)
    assert reopened.observations()['reads']==tools.observations()['reads']
