"""Native review gates a real host effect; fake public RPC never calls a provider."""
import hashlib
import json
from pathlib import Path
import pytest

from compute_plane.action_control import ActionAuthorization, ActionRefused, ActionSnapshot
from compute_plane.codex_auto_review import CodexNativeCommandReview
from tare_tools_agent_runtime import BindingRef, CancellationToken, ExecutionControl

ENABLED = {'guardian_approval', 'code_mode_host', 'skip_host_skill_discovery'}
FEATURES = ENABLED | {'shell_tool', 'hooks', 'plugins', 'apps', 'multi_agent'}


class FakeSession:
    def __init__(self, mode='approved', change=None, cleanup=True):
        self.mode, self.change, self.cleanup = mode, change, cleanup
        self.queue, self.methods, self.cursor = [], [], 0
        self.closed = False
        self.wire = ''
        self.config_bytes = None
        self.config_line_bytes = []
        self.page_chars = 65536
        self.page_lengths = []
        self.stderr_bytes = 0
        self.truncated = False

    def start(self, **kwargs):
        self.root = Path(kwargs['cwd'])
        self.argv = kwargs['argv']
        catalog=json.loads((self.root/'models.json').read_text())['models'][0]
        assert catalog['shell_type']=='disabled' and catalog['apply_patch_tool_type'] is None
        assert catalog['node_repl_disabled'] and catalog['tool_mode']=='direct'
        assert kwargs['mode'] == 'pipes' and kwargs['timeout'] > 0
        return {'handle': 'handle'}

    def send(self, **kwargs):
        row = json.loads(kwargs['text']); method = row['method']; self.methods.append(method)
        ident = row.get('id')
        if method == 'initialize': reply = {}
        elif method == 'config/read':
            reply = {'config': {'features': {k:k in ENABLED for k in FEATURES},
                'mcp_servers': {'unrelated': {'enabled':True}}, 'model_providers': {}}}
        elif method == 'account/read': reply = {'account': {'type':'chatgpt' if self.mode != 'api' else 'apiKey'}}
        elif method == 'thread/start':
            assert row['params']['config']['mcp_servers.unrelated.enabled'] is False
            assert row['params']['approvalsReviewer'] == 'auto_review'
            reply = {'thread': {'id':'thread'}, 'model':'gpt-6.1-sol',
                     'approvalsReviewer':'auto_review','approvalPolicy':'on-request'}
        elif method == 'turn/start':
            reply = {'turn': {'id':'turn'}}
            self.queue.append({'id':ident,'result':reply})
            packet = json.loads((self.root/'packet.json').read_text())
            args = packet['arguments']
            if self.mode == 'no_event':
                self.queue.append({'method':'turn/completed','params':{}}); return
            item = {'method':'item/started','params':{'threadId':'thread','turnId':'turn','item':{
                'type':'mcpToolCall','id':'item','server':'tare_action','tool':'execute_exact_command','arguments':{}}}}
            self.queue.append(item)
            params = {'threadId':'thread','turnId':'turn','targetItemId':'item','reviewId':'review','startedAtMs':1,
                'action':{'type':'mcpToolCall','server':'tare_action','toolName':'execute_exact_command'},
                'review':{'status':'inProgress'}}
            self.queue.append({'method':'item/autoApprovalReview/started','params':json.loads(json.dumps(params))})
            params.update(completedAtMs=2,decisionSource='agent',review={'status':'denied' if self.mode=='denied' else 'approved'})
            if self.mode == 'mismatch': params['targetItemId']='other'
            self.queue.append({'method':'item/autoApprovalReview/completed','params':params})
            if self.mode not in ('denied','no_dispatch'):
                (self.root/'dispatch.json').write_text(json.dumps(args))
            if self.change:self.change()
            return
        else:return
        if method == 'config/read' and self.config_bytes is not None:
            reply['config']['padding'] = ''
            size = len(json.dumps({'id':ident,'result':reply}).encode('utf-8'))
            reply['config']['padding'] = 'x' * (self.config_bytes - size)
            self.config_line_bytes.append(len(json.dumps({'id':ident,'result':reply}).encode('utf-8')))
        if ident is not None:self.queue.append({'id':ident,'result':reply})

    def read(self, **kwargs):
        self.wire += ''.join(json.dumps(row)+'\n' for row in self.queue)
        self.queue.clear()
        text, self.wire = self.wire[:self.page_chars], self.wire[self.page_chars:]
        self.cursor += len(text)
        self.page_lengths.append(len(text))
        stderr = 'x' * self.stderr_bytes
        self.stderr_bytes = 0
        status='exited' if self.mode=='no_dispatch' and 'turn/start' in self.methods else 'running'
        return {'status':status,'output':{'stdout':{'text':text,'cursor':self.cursor,'truncated':self.truncated},
            'stderr':{'text':stderr,'cursor':len(stderr),'truncated':False}}}

    def close(self):
        self.closed=True
        if not self.cleanup:raise RuntimeError('cleanup_unknown')


def fixture(tmp_path, mode='approved', change=None, cleanup=True, budget=True, record=None):
    executable=tmp_path/'codex-fixture';executable.write_bytes(b'pinned fixture')
    context='Run one owned, offline verifier and return its result.'
    clock=[100.0];token=CancellationToken()
    control=ExecutionControl(BindingRef('work','binding','operator',3,'opaque'),token,200,
                             monotonic=lambda:clock[0])
    h='a'*64
    action=ActionSnapshot('work','session','effect','binding',3,'command','fixture',str(tmp_path),h,
        hashlib.sha256(context.encode()).hexdigest(),h,h,h,'fixture',200,argv=('python','-V'))
    permit=ActionAuthorization('operator','nonce',action.action_sha256,'session','effect','binding',3,190)
    fake=FakeSession(mode,lambda:change(token,clock) if change else None,cleanup)
    receipts=[]; effects=[]; before=[]
    def probe(argv, **kwargs):
        raw=b'codex-cli 0.162.1\n' if argv[-1]=='--version' else '\n'.join(k+' stable true' for k in FEATURES).encode()
        return {'exit_code':0,'cleanup_verified':True},raw,b''
    def sessions_factory(**kwargs):
        assert kwargs == {'output_chars':524288}
        return fake
    port=CodexNativeCommandReview(str(executable),hashlib.sha256(executable.read_bytes()).hexdigest(),
        'gpt-6.1-sol',{'CODEX_HOME':str(tmp_path/'empty-home')},model_catalog={'slug':'gpt-6.1-sol'},disclosure_check=lambda a,c:True,
        reserve_review=lambda a,c:budget,record_receipt=record or receipts.append,
        sessions_factory=sessions_factory,clock=lambda:clock[0],probe=probe)
    def effect():effects.append('effect');return {'exit_code':0,'host':'real-result'}
    def before_effect():before.append('revalidated')
    def run(**overrides):
        options={'public_context':context,'before_effect':before_effect};options.update(overrides)
        return port.run(action,permit,control,effect,**options)
    return port,fake,receipts,effects,before,run


def test_exact_native_approval_releases_one_host_effect_and_preserves_result(tmp_path):
    _,fake,receipts,effects,before,run=fixture(tmp_path)
    assert run()=={'exit_code':0,'host':'real-result'}
    assert effects==['effect'] and before==['revalidated'] and fake.closed
    assert fake.methods.count('turn/start')==1 and 'turn/interrupt' in fake.methods
    assert receipts[0]['authority']=='NONE' and receipts[0]['cleanup_verified']
    assert receipts[0]['host_effect_started'] and receipts[0]['usage'] is None
    assert receipts[0]['billing']=='unknown' and not fake.root.exists()


@pytest.mark.parametrize('mode',['denied','no_event','mismatch','no_dispatch','api'])
def test_native_missing_or_invalid_approval_never_executes(tmp_path,mode):
    _,fake,receipts,effects,before,run=fixture(tmp_path,mode)
    with pytest.raises(ActionRefused):run()
    assert not effects and not before and fake.closed
    assert receipts[0]['host_effect_started'] is False
    if mode=='api':assert 'turn/start' not in fake.methods


@pytest.mark.parametrize('change',[lambda token,clock:token.cancel(),lambda token,clock:clock.__setitem__(0,195)])
def test_cancel_or_expired_permission_while_review_is_pending_prevents_effect(tmp_path,change):
    _,fake,_,effects,_,run=fixture(tmp_path,change=change)
    with pytest.raises(ActionRefused):run()
    assert not effects and fake.closed


def test_final_guard_failure_cannot_release_the_command(tmp_path):
    _,fake,_,effects,_,run=fixture(tmp_path)
    def reject():raise ActionRefused('source_changed')
    with pytest.raises(ActionRefused):run(before_effect=reject)
    assert not effects


def test_budget_refusal_stops_before_model_generation(tmp_path):
    _,fake,_,effects,_,run=fixture(tmp_path,budget=False)
    with pytest.raises(ActionRefused):run()
    assert 'turn/start' not in fake.methods and not effects and fake.closed


def test_record_failure_after_effect_never_retries_it(tmp_path):
    def failed_record(_):raise OSError('custody_failed')
    _,fake,_,effects,_,run=fixture(tmp_path,record=failed_record)
    with pytest.raises(RuntimeError,match='custody_unknown'):run()
    assert effects==['effect'] and fake.closed


def test_executable_change_refuses_before_native_session(tmp_path):
    port,fake,_,effects,_,run=fixture(tmp_path)
    port.executable.write_bytes(b'changed')
    with pytest.raises(ActionRefused,match='executable_changed'):run()
    assert not effects and not fake.methods


def test_cleanup_failure_after_effect_is_uncertain_and_never_replayed(tmp_path):
    _,fake,_,effects,_,run=fixture(tmp_path,cleanup=False)
    with pytest.raises(RuntimeError):run()
    assert effects==['effect'] and fake.closed


def test_observed_config_line_across_pages_releases_exactly_one_effect(tmp_path):
    _,fake,receipts,effects,before,run=fixture(tmp_path)
    fake.config_bytes = 87158
    assert run() == {'exit_code':0,'host':'real-result'}
    assert fake.config_line_bytes == [87158]
    assert fake.page_lengths[1] == 65536
    assert 21623 <= fake.page_lengths[2] < 22000
    assert effects == ['effect'] and before == ['revalidated']
    assert fake.methods.count('turn/start') == 1
    assert fake.closed and not fake.root.exists()
    assert receipts[0]['host_effect_started'] and receipts[0]['cleanup_verified']


@pytest.mark.parametrize('size,page_chars', [(131073,65536), (131073,200000),
                                           (196608,65536)],
                         ids=['paged-complete','single-page-complete','partial'])
def test_oversized_rpc_line_refuses_and_closes_without_effect(tmp_path,size,page_chars):
    _,fake,receipts,effects,before,run=fixture(tmp_path)
    fake.config_bytes, fake.page_chars = size, page_chars
    with pytest.raises(ActionRefused, match='native_review_output_limit'):
        run()
    assert not effects and not before and fake.closed
    assert 'turn/start' not in fake.methods and not fake.root.exists()
    assert not receipts[0]['host_effect_started'] and receipts[0]['cleanup_verified']


def test_total_output_cap_remains_512_kib(tmp_path):
    _,fake,receipts,effects,before,run=fixture(tmp_path)
    fake.stderr_bytes = 524289
    with pytest.raises(ActionRefused, match='native_review_output_limit'):
        run()
    assert not effects and not before and fake.closed
    assert not receipts[0]['host_effect_started']


def test_truncated_rpc_output_refuses_without_effect(tmp_path):
    _,fake,_,effects,before,run=fixture(tmp_path)
    fake.truncated = True
    with pytest.raises(ActionRefused, match='native_review_output_truncated'):
        run()
    assert not effects and not before and fake.closed


def test_invalid_rpc_json_refuses_without_effect(tmp_path):
    _,fake,_,effects,before,run=fixture(tmp_path)
    fake.wire = 'not-json\n'
    with pytest.raises(ValueError):
        run()
    assert not effects and not before and fake.closed


@pytest.mark.parametrize('tamper',['tool_arguments','dispatch'])
def test_bound_action_cannot_be_changed_by_the_native_forwarder(tmp_path,tamper):
    _,fake,receipts,effects,before,run=fixture(tmp_path)
    original=fake.send
    def send(**kwargs):
        original(**kwargs)
        if json.loads(kwargs['text'])['method']=='turn/start':
            if tamper=='tool_arguments':
                next(row for row in fake.queue if row.get('method')=='item/started')['params']['item']['arguments']={'argv':['other']}
            else:
                packet=json.loads((fake.root/'dispatch.json').read_text())
                packet['argv']=['other'];(fake.root/'dispatch.json').write_text(json.dumps(packet))
    fake.send=send
    code='native_review_tool_mismatch' if tamper=='tool_arguments' else 'native_review_dispatch_changed'
    with pytest.raises(ActionRefused,match=code):run()
    assert not effects and not before and fake.closed
    assert receipts[-1]['failure_code']==code and not receipts[-1]['host_effect_started']


def test_cancelled_host_result_survives_already_closed_native_interrupt_input(tmp_path):
    port,fake,receipts,effects,before,run=fixture(tmp_path)
    original=fake.send
    def closed(**kwargs):
        if json.loads(kwargs['text'])['method']=='turn/interrupt':
            raise ValueError('terminal_input_closed')
        return original(**kwargs)
    fake.send=closed
    result=run()
    assert result=={'exit_code': 0, 'host': 'real-result'} and effects==['effect']
    assert fake.closed and receipts[-1]['cleanup_verified']
    assert not receipts[-1]['native_interrupt_requested']
    assert receipts[-1]['remote_stop']=='unknown'
