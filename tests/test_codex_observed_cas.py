"""Resolving a wire reference never permits overwriting a changed disk version."""
import json

from tare_tools_agent_runtime import EffectRequest, ToolCall
from compute_plane.native_capabilities import NativeCapabilities
from tests.test_native_models import execution
from tests.test_codex_cache_preservation import reply
from tests.test_codex_edit_pins import observed


def test_observed_reference_retains_the_exact_cas_guard(tmp_path):
    root=tmp_path/'repo'; root.mkdir()
    scratch=tmp_path/'scratch'; scratch.mkdir()
    source=root/'a.py'; source.write_bytes(b'value = 1\n')
    control=execution()
    tools=NativeCapabilities(session_id='s',binding=control.binding,
        repositories={'repo':str(root)},writable={'repo':['a.py']},
        verifiers={},scratch_root=scratch,specgraph_edit_context=False)
    read=tools.execute(EffectRequest('s','read',ToolCall('r','read',
        json.dumps({'repository':'repo','path':'a.py'}))),control)
    digest=json.loads(read.output_json)['sha256']
    decision=reply({'repository':'repo','path':'a.py','sha256':'observed','delete':None,
        'changes':[{'search':'value = 1','replace':'value = 2'}]},observed('read',digest))
    call=decision['choices'][0]['message']['tool_calls'][0]
    assert json.loads(call['function']['arguments'])['sha256']==digest
    source.write_bytes(b'value = 99\n')
    result=tools.execute(EffectRequest('s','edit',ToolCall(call['id'],call['function']['name'],
        call['function']['arguments'])),control)
    assert result.status=='refused'
    assert json.loads(result.output_json)['reason']=='source_changed_read_again'
    assert source.read_bytes()==b'value = 99\n'


def test_current_object4_observed_delete_executes(tmp_path):
    root=tmp_path/'repo'; root.mkdir()
    scratch=tmp_path/'scratch'; scratch.mkdir()
    source=root/'a.py'; source.write_bytes(b'value = 1\n')
    control=execution()
    tools=NativeCapabilities(session_id='s',binding=control.binding,
        repositories={'repo':str(root)},writable={'repo':['a.py']},
        verifiers={},scratch_root=scratch,specgraph_edit_context=False)
    read=tools.execute(EffectRequest('s','read',ToolCall('r','read',
        json.dumps({'repository':'repo','path':'a.py'}))),control)
    digest=json.loads(read.output_json)['sha256']
    decision=reply({'repository':'repo','path':'a.py','sha256':'observed',
        'delete':True,'changes':[]},observed('read',digest))
    call=decision['choices'][0]['message']['tool_calls'][0]
    assert call['function']['name']=='edit'
    args=json.loads(call['function']['arguments'])
    assert args['delete'] is True and args['sha256']==digest
    result=tools.execute(EffectRequest('s','delete',ToolCall(call['id'],'edit',
        call['function']['arguments'])),control)
    assert result.status=='succeeded'
    assert json.loads(result.output_json)['sha256'] is None
    assert not source.exists()
    assert tools._tools.effects==1
