"""Native Codex approval gates one already-claimed host effect; no new authority."""
import hashlib
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from uuid import uuid4

from .action_control import ActionAuthorization, ActionRefused, ActionSnapshot
from .codex_review_protocol import NativeReviewBinding, canonical
from .managed_process import run_pipe_process
from .native_cli_codex import feature_states, user_instructions
from .terminal_sessions import TerminalSessions


class CodexNativeCommandReview:
    def __init__(self, executable, expected_sha256, model, environment, *, model_catalog, disclosure_check,
                 reserve_review, record_receipt, sessions_factory=TerminalSessions,
                 clock=time.monotonic, probe=run_pipe_process):
        self.requested = Path(executable)
        if (not self.requested.is_absolute() or type(expected_sha256) is not str
                or len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256)
                or type(model) is not str or not model or '\0' in model):
            raise ValueError('admitted_native_codex_required')
        if (type(environment) is not dict or any(type(k) is not str or type(v) is not str
                or not k or '=' in k or '\0' in k + v for k, v in environment.items())):
            raise ValueError('fixed_native_codex_environment_required')
        for callback in (disclosure_check, reserve_review, record_receipt, sessions_factory, clock, probe):
            if not callable(callback): raise ValueError('native_review_host_callbacks_required')
        self.executable = self.requested.resolve(strict=True)
        self.expected_sha256, self.model = expected_sha256, model
        if type(model_catalog) is not dict or model_catalog.get('slug') != model:
            raise ValueError('admitted_native_codex_catalog_required')
        self.catalog = json.loads(canonical(model_catalog))
        self.catalog.update(shell_type='disabled', apply_patch_tool_type=None,
            experimental_supported_tools=[], multi_agent_version=None,
            supports_search_tool=False, tool_mode='direct', node_repl_disabled=True)
        self.environment = dict(environment)
        self.environment.pop('OPENAI_API_KEY', None)
        self.disclosure, self.reserve, self.record = disclosure_check, reserve_review, record_receipt
        self.sessions_factory, self.clock, self.probe = sessions_factory, clock, probe
        self.lock = threading.Lock()
        self._check_executable()

    def _check_executable(self):
        if (self.requested.resolve(strict=True) != self.executable or not self.executable.is_file()
                or hashlib.sha256(self.executable.read_bytes()).hexdigest() != self.expected_sha256):
            raise ActionRefused('native_codex_executable_changed')

    def run(self, action, authorization, control, effect, *, public_context, before_effect):
        if (not isinstance(action, ActionSnapshot) or not isinstance(authorization, ActionAuthorization)
                or not action.argv or action.script is not None or action.stdin is not None
                or not callable(effect) or not callable(before_effect)):
            raise ActionRefused('native_review_command_and_host_guard_required')
        if (type(public_context) is not str or hashlib.sha256(public_context.encode()).hexdigest() != action.context_sha256
                or self.disclosure(action, public_context) is not True):
            raise ActionRefused('native_review_disclosure_refused')
        if not self.lock.acquire(blocking=False): raise ActionRefused('native_review_busy')
        sessions = None
        receipt = {'authority': 'NONE', 'action_sha256': action.action_sha256,
                   'cli_sha256': self.expected_sha256, 'requested_model': self.model,
                   'native_reviewer_model': 'unknown', 'billing': 'unknown',
                   'remote_stop': 'unknown', 'hidden_retries': 'unknown',
                   'host_effect_started': False, 'native_interrupt_requested': False,
                   'native_completed': None, 'usage': None, 'cleanup_verified': False}
        result = None; binding = None
        deadline = min(action.deadline, authorization.expires_at, control.deadline_monotonic)
        def live():
            binding = control.binding
            if (control.should_stop() or not math.isfinite(deadline) or self.clock() >= deadline
                    or (binding.work_id, binding.binding_id, binding.fence) != (action.work_id, action.binding_id, action.fence_seq)
                    or (authorization.action_sha256, authorization.session_id, authorization.effect_id,
                        authorization.binding_id, authorization.fence_seq) !=
                       (action.action_sha256, action.session_id, action.effect_id, action.binding_id, action.fence_seq)):
                raise ActionRefused('native_review_control_changed')
        try:
            live(); self._check_executable()
            if user_instructions(self.environment): raise ActionRefused('native_review_global_instructions')
            with ExitStack() as lifetime:
                folder = lifetime.enter_context(tempfile.TemporaryDirectory(prefix='tare-native-review-'))
                root = Path(folder)
                def probe(args):
                    physical, raw, _ = self.probe([str(self.executable), *args], cwd=root,
                        env=self.environment, timeout=min(10, deadline-self.clock()),
                        cancelled=control.should_stop, max_output=131072)
                    if physical.get('cleanup_verified') is not True:
                        receipt['capability_cleanup_unknown'] = True
                        raise RuntimeError('native_codex_capability_cleanup_unknown')
                    if (physical.get('exit_code') != 0
                            or any(physical.get(k) for k in ('cancelled', 'timeout', 'output_limit'))):
                        raise ActionRefused('native_codex_capability_unavailable')
                    return raw
                version = probe(['--version']).decode().strip()
                if version != 'codex-cli 0.162.1': raise ActionRefused('native_codex_protocol_unqualified')
                receipt['cli_version'] = version
                states = feature_states(probe(['features', 'list']))
                enabled = {'guardian_approval', 'code_mode_host', 'skip_host_skill_discovery'}
                if not enabled <= states.keys(): raise ActionRefused('native_review_features_missing')
                features = {key: key in enabled for key, (stage, _) in states.items() if stage not in ('removed', 'deprecated')}
                catalog_json = canonical({'models':[self.catalog]})
                (root/'models.json').write_text(catalog_json, encoding='utf-8')
                receipt['model_catalog_sha256'] = hashlib.sha256(catalog_json.encode()).hexdigest()
                configs = {'features': '{'+','.join(json.dumps(k)+'='+str(v).lower() for k,v in sorted(features.items()))+'}',
                    'model_catalog_json': json.dumps(str(root/'models.json')),
                    'approval_policy': json.dumps('on-request'), 'approvals_reviewer': json.dumps('auto_review'),
                    'sandbox_mode': json.dumps('read-only'), 'model_provider': json.dumps('openai'),
                    'forced_login_method': json.dumps('chatgpt'), 'service_tier': json.dumps('default'),
                    'model_reasoning_effort': json.dumps('low'), 'web_search': json.dumps('disabled'),
                    'project_doc_max_bytes': '0', 'developer_instructions': json.dumps(''), 'notify': '[]',
                    'tools.update_plan.enabled': 'false', 'tools.experimental_request_user_input.enabled': 'false'}
                argv = [str(self.executable), 'app-server', '--stdio']
                for key, val in configs.items(): argv.extend(['-c', key+'='+val])
                arguments = {'action_sha256': action.action_sha256, 'argv': list(action.argv), 'cwd': action.cwd}
                (root/'packet.json').write_text(json.dumps({'arguments': arguments, 'deadline': deadline}), encoding='utf-8')
                sessions = self.sessions_factory(output_chars=524288)
                lifetime.callback(sessions.close)
                owner = 'native-review-'+uuid4().hex
                terminal = sessions.start(owner=owner, request_id=uuid4().hex, argv=argv, cwd=str(root),
                    env=self.environment, timeout=max(.01, deadline-self.clock()), mode='pipes')
                handle, cursors, pending = terminal['handle'], {}, ''
                config = account = thread_id = turn_id = binding = None
                total = 0
                def send(method, params, ident=None):
                    body = {'jsonrpc':'2.0','method':method,'params':params}
                    if ident is not None: body['id']=ident
                    sessions.send(owner=owner, handle=handle, request_id=uuid4().hex, text=json.dumps(body)+'\n')
                send('initialize', {'clientInfo': {'name':'tare-native-review','version':'0.1'},
                                   'capabilities': {'experimentalApi':True}}, 1)
                thread_requested = False
                while True:
                    live()
                    terminal = sessions.read(owner=owner, handle=handle, cursors=cursors, limit=65536)
                    for name, page in terminal['output'].items():
                        if page['truncated']: raise ActionRefused('native_review_output_truncated')
                        cursors[name] = page['cursor']; total += len(page['text'].encode())
                        if name == 'stdout': pending += page['text']
                    if total > 524288:
                        raise ActionRefused('native_review_output_limit')
                    while '\n' in pending:
                        line, pending = pending.split('\n',1)
                        if len(line.encode('utf-8')) > 131072:
                            raise ActionRefused('native_review_output_limit')
                        if not line.strip(): continue
                        from .codex_review_bridge import loads
                        row = loads(line, max_bytes=131072)
                        if type(row) is not dict or 'error' in row: raise ActionRefused('native_review_rpc_failed')
                        if 'id' in row and 'method' in row: raise ActionRefused('native_review_manual_or_unexpected_request')
                        ident = row.get('id')
                        if ident == 1:
                            send('initialized', {}); send('config/read', {'includeLayers':False}, 2)
                            send('account/read', {'refreshToken':False}, 3)
                        elif ident == 2: config = row['result']['config']
                        elif ident == 3: account = row['result']['account']
                        elif ident == 4:
                            reply = row['result']
                            if (reply.get('model') != self.model or reply.get('approvalsReviewer') != 'auto_review'
                                    or reply.get('approvalPolicy') != 'on-request'):
                                raise ActionRefused('native_review_thread_configuration_changed')
                            thread_id = reply['thread']['id']
                            if self.reserve(action,control) is not True: raise ActionRefused('native_review_budget_refused')
                            live(); self._check_executable()
                            send('turn/start', {'threadId':thread_id, 'input':[{'type':'text',
                                'text':'Operator request and public context:\n'+public_context+'\nExact next command:\n'+canonical(arguments)+'\nUse only execute_exact_command once with {} arguments. Its host-bound description contains this exact command. Do not copy or substitute arguments, actions or tools.'}]}, 5)
                        elif ident == 5:
                            turn_id = row['result']['turn']['id']
                            binding = NativeReviewBinding({},thread_id,turn_id)
                        method = row.get('method','')
                        if method == 'error': raise ActionRefused('native_review_generation_failed')
                        if binding:
                            binding.observe(row)
                            if binding.completed:
                                receipt['native_completed'] = binding.completed
                                if not binding.approved: raise ActionRefused('native_review_'+binding.terminal['status'])
                        if method == 'thread/tokenUsage/updated': receipt['usage'] = row.get('params',{}).get('tokenUsage')
                        if method == 'turn/completed': raise ActionRefused('native_review_no_executable_approval')
                    if len(pending.encode('utf-8')) > 131072:
                        raise ActionRefused('native_review_output_limit')
                    if config is not None and account is not None and not thread_requested:
                        if account.get('type') != 'chatgpt': raise ActionRefused('native_review_chatgpt_login_required')
                        if config.get('chatgpt_base_url') not in (None,'https://chatgpt.com/backend-api','https://chatgpt.com/backend-api/'):
                            raise ActionRefused('native_review_custom_endpoint_refused')
                        providers = config.get('model_providers') or {}
                        if (providers.get('openai') or {}).get('base_url'):
                            raise ActionRefused('native_review_custom_provider_refused')
                        effective = config.get('features') or {}
                        if any(effective.get(k) is not v for k,v in features.items() if k != 'unified_exec'):
                            raise ActionRefused('native_review_features_changed')
                        overrides = {'mcp_servers.'+name+'.enabled':False for name in config.get('mcp_servers',{}) if name != 'tare_action'}
                        overrides['mcp_servers.tare_action'] = {'command':sys.executable,
                            'args':['-I',str(Path(__file__).with_name('codex_review_bridge.py')),str(root)],
                            'required':True,'enabled':True,'default_tools_approval_mode':'prompt'}
                        send('thread/start', {'model':self.model,'modelProvider':'openai',
                            'allowProviderModelFallback':False,'cwd':str(root),'ephemeral':True,
                            'approvalPolicy':'on-request','approvalsReviewer':'auto_review','sandbox':'read-only',
                            'config':overrides,'baseInstructions':'Forward the exact requested command once to the sole tare_action execute_exact_command tool. Do not use other tools or run commands yourself. The Tare host owns execution and returns its result.'}, 4)
                        thread_requested = True
                    dispatch = root/'dispatch.json'
                    if binding and binding.approved and dispatch.is_file():
                        from .codex_review_bridge import loads
                        if canonical(loads(dispatch.read_bytes())) != canonical(arguments):
                            raise ActionRefused('native_review_dispatch_changed')
                        before_effect(); live()
                        receipt['host_effect_started'] = True
                        result = effect()
                        try:
                            send('turn/interrupt', {'threadId':thread_id, 'turnId':turn_id}, 6)
                            receipt['native_interrupt_requested'] = True
                        except ValueError as exc:
                            if str(exc) not in ('terminal_input_closed', 'terminal_session_not_running'):
                                raise
                            receipt['native_interrupt_requested'] = False
                            receipt['native_interrupt_unavailable'] = 'transport_closed_after_host_effect'
                        break
                    if terminal['status'] not in ('starting','running'):
                        raise ActionRefused('native_review_process_closed_without_effect')
                    time.sleep(.02)
        except ActionRefused as exc:
            receipt['failure_code']=str(exc)
            if binding is not None and binding.failure is not None:
                receipt['protocol_failure']=binding.failure
            raise
        finally:
            try:
                if sessions is not None:
                    sessions.close()
                    receipt['native_transport_closed'] = True
                receipt['cleanup_verified'] = not receipt.get('capability_cleanup_unknown', False)
            except Exception:
                raise RuntimeError('native_review_cleanup_unknown') from None
            finally:
                try:
                    self.record(receipt)
                except Exception:
                    raise RuntimeError('native_review_custody_unknown') from None
                finally:
                    self.lock.release()
        return result
