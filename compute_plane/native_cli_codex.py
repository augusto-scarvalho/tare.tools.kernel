"""Discover Codex controls for a toolless, stateless inference call."""
import json
import os
import re
from tare_tools_agent_runtime import ConversationRejected, ModelFailure
from .native_models import encode, object_json, INVALID_CLI_REPLY, NativeTransportInterrupted, MAX_REQUEST_BYTES

ENABLED_FEATURES = frozenset({'skip_host_skill_discovery', 'code_mode_host'})
REQUIRED_FEATURES = ENABLED_FEATURES | {'shell_tool', 'hooks', 'plugins', 'apps', 'multi_agent'}


def feature_states(raw):
    """Parse the public feature table without assuming a fixed feature roster."""
    states = {}
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([a-z][a-z0-9_.]*)\s+([a-z][a-z ]*?)\s+(true|false)\s*', line.strip())
        if not match or match[1] in states:
            raise ModelFailure('CLI_CAPABILITY_UNAVAILABLE', not_started=True)
        states[match[1]] = (match[2], match[3] == 'true')
    if not states or not REQUIRED_FEATURES <= {k for k, v in states.items() if v[0] not in {'removed', 'deprecated'}}:
        raise ModelFailure('CLI_CAPABILITY_UNAVAILABLE', not_started=True)
    return states


def feature_args(features):
    # A single TOML table preserves literal dots in names. The CLI's dotted
    # override parser interprets even quoted path segments as nested tables.
    table=', '.join(json.dumps(name)+' = '+('true' if name in ENABLED_FEATURES else 'false')
                    for name in sorted(features))
    return ['-c', 'features={'+table+'}']


def _inspect(executable, args, cwd, env, control, run_process, receipt, *, seconds=10, max_output=128*1024):
    """One local Codex command that sends no request; any failure is a capability Tare cannot confirm."""
    physical, raw, _ = run_process([executable, *args], cwd=cwd, env=env,
        timeout=min(seconds, control.remaining_seconds()), cancelled=control.should_stop, max_output=max_output)
    if physical.get('cleanup_verified') and (physical.get('cancelled') or control.should_stop()):
        receipt['cancelled'] = True
        raise NativeTransportInterrupted('CLI_CANCELLED', receipt)
    if not physical['cleanup_verified']:
        receipt['cleanup_verified'] = False
        raise NativeTransportInterrupted('CLI_CLEANUP_UNCONFIRMED', receipt)
    if (physical['exit_code'] != 0
            or any(physical.get(k) for k in ('cancelled', 'timeout', 'output_limit'))):
        raise ModelFailure('CLI_CAPABILITY_UNAVAILABLE', not_started=True)
    return raw


def discover(executable, root, control, run_process, receipt):
    # Discovery needs no login and cannot inherit a user's feature defaults.
    home = root/'feature-probe'
    home.mkdir()
    env = dict(os.environ, CODEX_HOME=str(home))
    def probe(args):
        return feature_states(_inspect(executable, ['features', 'list', *args], root, env, control, run_process,
                                       receipt))
    states = probe([])
    # Sending deprecated overrides emits error items even for ignored settings.
    # Omit them, but still require their effective state to remain disabled.
    features = tuple(sorted(k for k, v in states.items() if v[0] not in {'removed', 'deprecated'}))
    effective = probe(feature_args(features))
    # Some releases force the executor implementation on. It exposes no shell
    # when shell_tool=false and the model catalog has shell_type=disabled.
    mismatches = sorted(k for k, v in states.items() if v[0] != 'removed' and k != 'unified_exec'
                        and effective.get(k, (None, None))[1] != (k in ENABLED_FEATURES))
    if (set(effective) != set(states) or any(effective[k][0] != states[k][0] for k in states)
            or mismatches):
        receipt['unconfirmed_cli_features'] = mismatches
        raise ModelFailure('CLI_CAPABILITY_UNAVAILABLE', not_started=True)
    return features


def user_instructions(env):
    """The user's global instruction files Codex would add to every call ({name: bytes}, empty when none). They come
    from CODEX_HOME, which also holds the login, so a call cannot leave them out: --ignore-user-config skips only
    config.toml (shown by `codex debug prompt-input`, 2026-10-07). The role text tare sends must be the only one."""
    from pathlib import Path
    home = Path(env.get('CODEX_HOME') or Path.home()/'.codex')
    found = {}
    for name in ('AGENTS.override.md', 'AGENTS.md'):
        try:
            text = (home/name).read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        if text.strip():
            found[name] = len(text.encode('utf-8'))
    return found


def prompt_input(executable, root, body, catalog, features, guidance, control, run_process, receipt):
    """What Codex shows the model in tare's `exec` call, from its own `debug prompt-input`, which sends nothing.
    Like the real call it lists the skills installed in CODEX_HOME (copied into an empty home, so neither the login
    nor the owner's config.toml is read) under exec's approval policy and read-only sandbox. Codex 0.166.0: every
    block equalled a real call's own session record (docs/VENDOR_GUIDANCE.md)."""
    from pathlib import Path
    from .vendor_guidance import READING_PROMPT
    home = root/'prompt-input'
    home.mkdir(exist_ok=True)
    skills = Path(os.environ.get('CODEX_HOME') or Path.home()/'.codex')/'skills'
    # A skill may link to a whole project: only what decides its listing is copied, its SKILL.md, its
    # agents/*.yaml (`policy.allow_implicit_invocation: false` keeps a skill out of the list) and a plugin manifest
    # (its name prefixes the skill's).
    for pattern in ('SKILL.md', 'agents/*.yaml', '.claude-plugin/plugin.json', '.codex-plugin/plugin.json'):
        for source in [*skills.glob('*/' + pattern), *skills.glob('*/*/' + pattern)]:
            target = home/'skills'/source.relative_to(skills)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    args = prepare(home, body, catalog, features=features, guidance=guidance)
    config = [a for i, a in enumerate(args) if a == '-c' or i and args[i-1] == '-c']
    config += ['-c', 'approval_policy="never"', '-c', 'sandbox_mode="read-only"']  # exec's, with `-s read-only`
    return _inspect(executable, ['debug', 'prompt-input', *config, READING_PROMPT], home,
                    dict(os.environ, CODEX_HOME=str(home)), control, run_process, receipt,
                    seconds=30, max_output=1024*1024).decode('utf-8')


def unproved_removals(executable, root, body, catalog, features, guidance, control, run_process, receipt):
    """Codex can only leave a block out, and an older release may ignore a key it does not know. Each chosen block
    must show in `debug prompt-input` without its switch and not with it; returns the item ids that did not."""
    from .vendor_guidance import codex_switches
    before, after = (prompt_input(executable, root, body, catalog, features, choices, control, run_process, receipt)
                     for choices in ({}, guidance))
    return sorted(k for k, (_, tag) in codex_switches(guidance).items() if tag not in before or tag in after)


def prepare(root, body, catalog, *, features, guidance=None):
    model = json.loads(encode(catalog))
    if model.get('slug') != body['model']:
        raise ModelFailure('CLI_CATALOG_MODEL_MISMATCH', not_started=True)
    model.update(tool_mode='direct', multi_agent_version=None, experimental_supported_tools=[],
                 apply_patch_tool_type=None, shell_type='disabled', supports_search_tool=False)
    (root/'models.json').write_text(encode({'models': [model]}), encoding='utf-8')
    (root/'instructions.md').write_text(body['system'], encoding='utf-8')
    config = {'model_catalog_json': str(root/'models.json'),
              'model_instructions_file': str(root/'instructions.md'),
              # The catalog names priority as some models' default tier, applied when none is set.
              'model_reasoning_effort': body['effort'], 'service_tier': 'default', 'web_search': 'disabled',
              'project_doc_max_bytes': 0, 'developer_instructions': '', 'notify': [],
              'tools.update_plan.enabled': False, 'tools.experimental_request_user_input.enabled': False,
              'suppress_unstable_features_warning': True}
    from .vendor_guidance import codex_switches
    config.update({key: False for key, _ in codex_switches(guidance or {}).values()})  # the owner's removals
    args = ['exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
            '--cd', str(root.parent), '--json', '-m', body['model'], '-s', 'read-only']
    if 'response_schema' in body:
        # Final JSON is data for Tare, never a vendor-side tool invocation.
        schema = root/'response-schema.json'
        schema.write_text(encode(body['response_schema']), encoding='utf-8')
        args += ['--output-schema', str(schema)]
    for key, value in config.items():
        args.extend(['-c', key+'='+json.dumps(value)])
    return args+feature_args(features)+['-']


def codex_usage_limit(rows):
    """Codex reports a spent ChatGPT plan as an error ("You've hit your usage limit. Try again at ..."), with
    codex_error_info (usage_limit_reached, usage_limit_exceeded {limit_id, resets_at}, *_credits_depleted) where
    the CLI passes it through."""
    from .native_cli import USAGE_LIMIT_TEXT, usage_limit

    def find(value, key):
        if isinstance(value, dict):
            return value.get(key, next((found for v in value.values() if (found := find(v, key)) is not None), None))
        return None
    for row in rows:
        if row.get('type') not in {'error', 'turn.failed'}:
            continue
        error = row.get('error') if row.get('type') == 'turn.failed' else row
        if not isinstance(error, dict):
            continue
        info = error.get('codex_error_info')
        coded = info is not None and re.search(r'usage_limit|credits_depleted', json.dumps(info))
        if coded or USAGE_LIMIT_TEXT.search(str(error.get('message') or '')):
            return usage_limit(find(info, 'resets_at'), find(info, 'limit_id') if coded else None)
    return None


def reconnect_notice(row):
    """Recognize a bounded HTTP reconnect notice, never a final/coded failure."""
    if row.get('type') != 'error' or row.get('codex_error_info') is not None:
        return False
    match = re.fullmatch(r'Reconnecting\.\.\. ([1-9][0-9]?)/([1-9][0-9]?) '
                         r'\(unexpected status (?:408|429|500|502|503|504)\b[^\r\n]*\)',
                         str(row.get('message', '')))
    return bool(match and int(match[1]) <= int(match[2]) <= 16)


def _resolved(schema, definitions=None, chain=()):
    value = dict(schema)
    definitions = value.pop('$defs', {}) if definitions is None else definitions
    while '$ref' in value:
        ref = value.pop('$ref')
        if (not isinstance(ref, str) or not ref.startswith('#/$defs/')
                or set(value) - {'title', 'description', 'default'} or ref in chain):
            raise ValueError('unsupported or recursive CLI schema reference')
        name = ref[len('#/$defs/'):].replace('~1', '/').replace('~0', '~')
        if name not in definitions or len(chain) >= 32:
            raise ValueError('unknown or deep CLI schema reference')
        value = {**definitions[name], **value}
        chain = (*chain, ref)
    return value, definitions, chain


def _nullable(schema, definitions=None):
    schema, definitions, _ = _resolved(schema, definitions)
    kinds = schema.get('type', [])
    return ('null' in (kinds if isinstance(kinds, list) else [kinds])
            or any(_nullable(item, definitions) for item in schema.get('anyOf', [])))


def _strict_parameters(schema, definitions=None, chain=()):
    """Inline bounded local definitions; distinguish omission from explicit null."""
    value, definitions, chain = _resolved(schema, definitions, chain)
    value.pop('default', None)
    if value.get('type') == 'object':
        if value.get('additionalProperties', False) is not False:
            raise ValueError('open CLI argument objects are unsupported')
        properties = value.get('properties', {})
        required = set(value.get('required', []))
        converted = {}
        for key, spec in properties.items():
            target = _strict_parameters(spec, definitions, chain)
            if key not in required:
                if _nullable(spec, definitions):
                    target = {'anyOf': [
                        {'type': 'object', 'properties': {'omitted': {'type': 'boolean', 'enum': [True]}},
                         'required': ['omitted'], 'additionalProperties': False},
                        {'type': 'object', 'properties': {'value': target},
                         'required': ['value'], 'additionalProperties': False}]}
                else:
                    target = {'anyOf': [target, {'type': 'null'}]}
            converted[key] = target
        value['properties'] = converted
        value.update(required=list(properties), additionalProperties=False)
    elif value.get('type') == 'array':
        value['items'] = _strict_parameters(value['items'], definitions, chain)
    if 'anyOf' in value:
        value['anyOf'] = [_strict_parameters(item, definitions, chain) for item in value['anyOf']]
    return value


def _restore_optional(value, schema, definitions=None, *, presence=True):
    schema, definitions, _ = _resolved(schema, definitions)
    if 'anyOf' in schema:
        kinds = {'null': value is None, 'object': isinstance(value, dict), 'array': isinstance(value, list),
                 'string': isinstance(value, str), 'boolean': type(value) is bool,
                 'integer': type(value) is int, 'number': type(value) in (int, float)}
        for branch in schema['anyOf']:
            resolved, _, _ = _resolved(branch, definitions)
            kind = resolved.get('type')
            if any(kinds.get(item, False) for item in (kind if isinstance(kind, list) else [kind])):
                try:
                    return _restore_optional(value, branch, definitions, presence=presence)
                except ValueError:
                    pass
        raise ValueError('invalid CLI argument union')
    if isinstance(value, dict) and schema.get('type') == 'object':
        properties = schema.get('properties', {})
        if set(value) != set(properties):
            raise ValueError('invalid structured CLI argument keys')
        required = set(schema.get('required', []))
        result = {}
        for key, item in value.items():
            nullable = _nullable(properties[key], definitions)
            if key not in required:
                if presence and nullable:
                    if isinstance(item, dict) and set(item) == {'omitted'} and item['omitted'] is True:
                        continue
                    if not isinstance(item, dict) or set(item) != {'value'}:
                        raise ValueError('nullable optional argument needs explicit presence')
                    item = item['value']
                elif item is None and not nullable:
                    continue
            result[key] = _restore_optional(item, properties[key], definitions, presence=presence)
        return result
    if isinstance(value, list) and schema.get('type') == 'array':
        return [_restore_optional(item, schema['items'], definitions, presence=presence) for item in value]
    return value


def object_arguments_request(body, actions):
    """Codex validates tool arguments themselves, rather than an opaque JSON string."""
    try:
        variants = [{'type': 'object', 'description': a.get('description', ''),
            'properties': {'action': {'type': 'string', 'enum': [a['name']], 'description': a.get('description', '')},
            'arguments': _strict_parameters(a['input_schema'])},
            'required': ['action', 'arguments'], 'additionalProperties': False} for a in actions]
    except ValueError as error:
        raise ModelFailure('CLI_ACTION_SCHEMA_UNSUPPORTED', not_started=True) from error
    body['response_schema']['properties']['actions']['items'] = (
        {'anyOf': variants} if variants else {'type': 'object', 'properties': {},
                                             'required': [], 'additionalProperties': False})
    envelope = object_json(body['prompt'], max_bytes=MAX_REQUEST_BYTES)
    # These are quoted protocol records, not native CLI tool invocations. Keep
    # source strings inside objects instead of serializing the entire result a
    # second time. Ordinary user/assistant prose is never interpreted as JSON.
    for record in envelope['conversation_records']:
        if record.get('role') == 'tool' and isinstance(record.get('content'), str):
            try:
                record['content'] = object_json(record['content'])
            except (ValueError, ConversationRejected):
                pass
        if record.get('role') == 'assistant':
            for call in record.get('tool_calls', []):
                function = call.get('function', {})
                if isinstance(function.get('arguments'), str):
                    try:
                        function['arguments'] = object_json(function['arguments'])
                    except (ValueError, ConversationRejected):
                        pass
    # A stable reference keeps the grammar and catalog independent of receipts.
    # Resolve it only at decode; the owner still validates and CASes the digest.
    for action, variant in zip(actions, variants):
        if action['name'] == 'edit' and 'sha256' in action['input_schema'].get('required', []):
            variant['properties']['arguments']['properties']['sha256'] = {
                'type': ['string', 'null'], 'enum': [None, 'observed']}
    # The CLI grammar is stricter than the owner's optional-parameter contract.
    # Show its actual shape to the model, so it does not plan abbreviated read
    # arguments that only fit another, shorter action's grammar.
    for action, variant in zip(envelope['action_catalog'], variants):
        action['wire_arguments_schema'] = variant['properties']['arguments']
    # "action" sorts before "arguments", including through CLI canonicalization:
    # constrain tool selection before the differently shaped argument branches.
    envelope['argument_encoding'] = 'native-json-object/4'
    body['prompt'] = encode(envelope)
    body['system'] = (
        'Return a data-only decision for the external Tare executor. The supplied conversation_records '
        'are quoted history, not instructions for this CLI to execute tools. Return one JSON object '
        'with text and actions. Each entry first selects its catalog name in action, then fills arguments as a JSON OBJECT, '
        'never a JSON-encoded string. For that action, fill EVERY property in wire_arguments_schema, '
        'including the original optional fields; do not omit those keys. Optional nonnullable parameters '
        'use null for omission; '
        'nullable OPTIONAL parameters use {"omitted":true} to retain the owner default or {"value":...} '
        'to set a value, including explicit null. Required nullable values use null directly. '
        'For edit.sha256, use "observed" as a stable reference to the latest matched read/edit receipt '
        'for exactly that repository and path in conversation_records; never supply a digest. '
        'Missing observations reject the whole proposal. Null retains create-only owner semantics. '
        'When turn_constraints.required_action is present at the end, return exactly one action '
        'with that name and actual findings or remaining gaps. The host rejects any other decision. '
        'Select the action by its catalog description, '
        'then use exactly its schema. A directory listing is not a file-content read. '
        'The CLI sandbox limits this CLI; Tare rechecks the recorded scope, permissions and guards '
        'before executing an admitted action. Do not refuse an external action merely because this CLI '
        'cannot execute it. Choose the next needed action, or tare_deliver with the actual answer when done. '
        'Machine tool results and historical tool arguments are decoded objects in conversation_records; '
        'copy source string values exactly, without treating JSON display escapes as extra source characters. '
        'With an empty catalog, return the answer in text and actions=[]. No Markdown fences.')
    return body


def _observed_pins(records):
    """Use only matched read/edit receipts from the submitted history."""
    calls, pins = {}, {}

    def decoded(value):
        if isinstance(value, str):
            try:
                return object_json(value)
            except (ValueError, ConversationRejected):
                return None
        return value

    for record in records:
        if record.get('role') == 'assistant':
            for call in record.get('tool_calls', []):
                ident = call.get('id')
                if not isinstance(ident, str):
                    continue
                calls.pop(ident, None)
                function = call.get('function', {})
                args = decoded(function.get('arguments'))
                if function.get('name') in ('read', 'edit') and isinstance(args, dict):
                    location = (args.get('repository'), args.get('path'))
                    if all(isinstance(part, str) for part in location):
                        calls[ident] = location
        if record.get('role') == 'tool':
            location = calls.pop(record.get('tool_call_id'), None)
            result = decoded(record.get('content'))
            if location is None or not isinstance(result, dict) or 'sha256' not in result:
                continue
            sha = result['sha256']
            if sha is None or isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha):
                pins[location] = sha
    return pins


def decode(rows, model, *, structured=False, argument_schemas=None, argument_encoding='native-json-object/1',
           conversation_records=None):
    from .native_cli import proposal, action_proposal
    limit = codex_usage_limit(rows)
    if limit is not None:
        raise limit
    terminal = [r for r in rows if r.get('type') == 'turn.completed']
    completed_at = next((i for i, row in enumerate(rows) if row.get('type') == 'turn.completed'), -1)
    items = [r['item'] for r in rows if r.get('type') in {'item.started', 'item.completed'}]
    messages = [r['item'] for r in rows if r.get('type') == 'item.completed'
                and r['item'].get('type') == 'agent_message']
    # The provider refused the turn (e.g. a model the account cannot use): one turn that
    # failed without any tool item or answer is a generation failure, not a boundary breach.
    if (not terminal and sum(r.get('type') == 'turn.started' for r in rows) <= 1
            and sum(r.get('type') == 'turn.failed' for r in rows) == 1
            and all(item.get('type') in {'reasoning', 'error'} for item in items)):
        raise ModelFailure('CLI_GENERATION_FAILED')
    if any(item.get('type') not in {'agent_message', 'reasoning'} for item in items):
        raise ModelFailure('CLI_BOUNDARY_UNCONFIRMED')
    # Some models send commentary before the answer; exec's final answer is the last message.
    if (len(terminal) != 1 or not messages
            or sum(r.get('type') == 'turn.started' for r in rows) != 1
            or any(item.get('type') not in {'agent_message', 'reasoning'} for item in items)
            or any(r.get('type') == 'turn.failed' or r.get('type') == 'error'
                   and (i >= completed_at or not reconnect_notice(r)) for i, r in enumerate(rows))):
        raise ModelFailure('CLI_BOUNDARY_UNCONFIRMED')
    stage = 'envelope'
    try:
        if structured:
            value = object_json(messages[-1]['text'])
            stage = 'action_shape'
            if argument_schemas is not None:
                name_key = 'action' if argument_encoding in {'native-json-object/3', 'native-json-object/4'} else 'name'
                pins = _observed_pins(conversation_records or []) if argument_encoding == 'native-json-object/4' else {}
                if (set(value) != {'text', 'actions'} or not isinstance(value['text'], str)
                        or not isinstance(value['actions'], list) or len(value['actions']) > 16):
                    raise ValueError('invalid structured CLI response')
                calls = []
                for index, action in enumerate(value['actions'], 1):
                    if (not isinstance(action, dict) or set(action) != {name_key, 'arguments'}
                            or not isinstance(action[name_key], str) or not isinstance(action['arguments'], dict)):
                        raise ValueError('invalid structured CLI action')
                    schema = argument_schemas.get(action[name_key])
                    if schema is None:
                        raise ValueError('CLI action outside the current catalog')
                    args = (_restore_optional(action['arguments'], schema,
                        presence=argument_encoding in {'native-json-object/2','native-json-object/3','native-json-object/4'})
                        if schema is not None else action['arguments'])
                    if argument_encoding == 'native-json-object/4' and action[name_key] == 'edit':
                        pin = args.get('sha256')
                        if pin is not None:
                            if pin != 'observed':
                                raise ValueError('model-authored edit digest')
                            location = (args.get('repository'), args.get('path'))
                            if location not in pins:
                                raise ValueError('missing matched edit observation')
                            args['sha256'] = pins[location]
                    calls.append({'id': f'cli-action-{index}', 'name': action[name_key], 'arguments': args})
                result = proposal(encode({'text': value['text'], 'tool_calls': calls}), model, terminal[0].get('usage'))
            elif isinstance(value.get('actions'), list):
                for action in value['actions']:
                    if isinstance(action, dict) and isinstance(action.get('arguments_json'), str):
                        stage = 'arguments_json'
                        object_json(action['arguments_json'])
            if argument_schemas is None:
                stage = 'action_shape'
                result = action_proposal(value, model, terminal[0].get('usage'))
        else:
            result = proposal(messages[-1]['text'], model, terminal[0].get('usage'))
    except (ValueError, ConversationRejected):
        # A clean turn whose answer is not the JSON object: the model is asked again,
        # nothing is executed and the answer is never reinterpreted as an action.
        result = {'model': model, 'usage': terminal[0].get('usage'), 'choices': [{'finish_reason': 'tool_calls',
            'message': {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'invalid-reply', 'type': 'function',
                'function': {'name': INVALID_CLI_REPLY, 'arguments': '{}'}}]}}]}
        # Record where validation stopped, without saving source or model prose.
        result['_proposal_rejection'] = {'stage': stage,
            'answer_bytes': len(messages[-1]['text'].encode('utf-8'))}
    # Exec JSONL does not attest the backend model. Do not fabricate attestation.
    result['_model_identity'] = 'requested_route_only'
    return result
