"""Codex structured output carries native objects without a second JSON parser."""
import json
import pytest

from compute_plane import native_cli
from compute_plane.native_cli_codex import decode
from compute_plane.native_capabilities import TOOL_SPECS
from compute_plane.native_terminal import TERMINAL_SPECS


def request(specs):
    tools=[{'function': {'name': name, 'description':spec.get('description',''),
                        'parameters': spec['parameters']}} for name,spec in specs.items()]
    return native_cli.cli_request({'model':'fixture','messages':[],'tools':tools},
                                 structured=True,native_arguments=True)


def reply(actions, specs, encoding='native-json-object/3'):
    if encoding=='native-json-object/3':
        actions=[{('action' if key=='name' else key):value for key,value in item.items()} for item in actions]
    rows=[{'type':'turn.started'}, {'type':'item.completed','item':{'type':'agent_message',
        'text':json.dumps({'text':'','actions':actions},ensure_ascii=False)}},
        {'type':'turn.completed','usage':{'input_tokens':5,'output_tokens':2}}]
    return decode(rows,'fixture',structured=True,argument_encoding=encoding,
                  argument_schemas={name:spec['parameters'] for name,spec in specs.items()})


def test_schema_validates_the_arguments_object_and_keeps_owner_schema_unchanged():
    before=json.dumps(TOOL_SPECS,sort_keys=True)
    body=request(TOOL_SPECS)
    assert json.loads(body['prompt'])['argument_encoding']=='native-json-object/4'
    branches=body['response_schema']['properties']['actions']['items']['anyOf']
    catalog={a['name']:a for a in json.loads(body['prompt'])['action_catalog']}
    assert len(branches)==len(TOOL_SPECS)
    for branch in branches:
        assert set(branch['properties'])=={'action','arguments'}
        name=branch['properties']['action']['enum'][0]
        assert branch['description']==TOOL_SPECS[name]['description']
        params=branch['properties']['arguments']
        assert params==catalog[name]['wire_arguments_schema']
        assert params['type']=='object' and params['additionalProperties'] is False
        assert set(params['required'])==set(params['properties'])
    assert json.dumps(TOOL_SPECS,sort_keys=True)==before


@pytest.mark.parametrize('name,args,expected',[
    ('read',dict(repository='projects',path='notas/ação.txt',start=None,end=None,start_column=None,sha256=None),
            dict(repository='projects',path='notas/ação.txt')),
    ('edit',dict(repository='projects',path='new.py',sha256=None,
        changes=[dict(search='',replace='PATH = "C:\\files\\note"\nQUOTE = \'ação\'\n')]),
        dict(repository='projects',path='new.py',sha256=None,
        changes=[dict(search='',replace='PATH = "C:\\files\\note"\nQUOTE = \'ação\'\n')])),
    ('terminal_start',dict(repository='projects',argv=['worker','a b'],mode=None,
         timeout_seconds={'value':None},columns=None,rows=None),
         dict(repository='projects',argv=['worker','a b'],timeout_seconds=None)),
])
def test_roundtrip_preserves_text_and_explicit_nullable_values(name,args,expected):
    spec=(TOOL_SPECS|TERMINAL_SPECS)[name]
    output=reply([dict(name=name,arguments=args)],{name:spec})
    calls=output['choices'][0]['message']['tool_calls']
    assert len(calls)==1 and calls[0]['function']['name']==name
    assert json.loads(calls[0]['function']['arguments'])==expected


@pytest.mark.parametrize('fault',['json-string','legacy-key','missing-required','unknown-key','unknown-action'])
def test_invalid_native_arguments_refuse_the_entire_batch(fault):
    good=dict(name='read',arguments=dict(repository='projects',path='a.py',start=None,end=None,start_column=None,sha256=None))
    bad=json.loads(json.dumps(good))
    if fault=='json-string': bad['arguments']='{"path":"unterminated}'
    elif fault=='legacy-key': bad['arguments_json']=json.dumps(bad.pop('arguments'))
    elif fault=='missing-required': bad['arguments'].pop('path')
    elif fault=='unknown-action': bad['name']='other'
    else: bad['arguments']['unadmitted']='value'
    output=reply([good,bad],{'read':TOOL_SPECS['read']})
    calls=output['choices'][0]['message']['tool_calls']
    assert len(calls)==1 and calls[0]['function']['name']=='invalid_cli_reply'
    assert output['_proposal_rejection']['stage']=='action_shape'


def test_empty_catalog_still_allows_text_and_no_actions():
    body=request({})
    assert body['response_schema']['properties']['actions']['maxItems']==0
    assert reply([], {})['choices'][0]['message']['tool_calls']==[]


@pytest.mark.parametrize('encoding', ['native-json-object/1', 'native-json-object/2', 'native-json-object/3'])
def test_historical_flat_edit_schema_still_decodes(encoding):
    # Historical catalogs are explicit inputs; do not reinterpret them as today's schema.
    legacy = {'type': 'object', 'properties': {
        'repository': {'type': 'string'}, 'path': {'type': 'string'},
        'sha256': {'type': ['string', 'null']}, 'search': {'type': 'string'},
        'replace': {'type': 'string'}, 'changes': {'type': 'array', 'items': {'type': 'object'}}},
        'required': ['repository', 'path', 'sha256'], 'additionalProperties': False}
    args = dict(repository='repo', path='new.py', sha256=None, search='', replace='value = 2\n', changes=None)
    output = reply([dict(name='edit', arguments=args)], {'edit': {'parameters': legacy}}, encoding)
    actual = json.loads(output['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    assert actual == {key: value for key, value in args.items() if key != 'changes'}


def test_current_edit_requires_a_list_and_reaches_the_fenced_executor(tmp_path):
    from compute_plane.native_capabilities import NativeCapabilities
    from tests.test_native_models import execution
    from tare_tools_agent_runtime import EffectRequest, ToolCall
    root = tmp_path/'repo'; root.mkdir()
    scratch = tmp_path/'scratch'; scratch.mkdir()
    source = root/'a.py'; source.write_bytes(b'value = 1\nother = 3\n')
    control = execution()
    tools = NativeCapabilities(session_id='s', binding=control.binding,
        repositories={'repo': str(root)}, writable={'repo': ['a.py']},
        verifiers={}, scratch_root=scratch, specgraph_edit_context=False)
    read = tools.execute(EffectRequest('s', 'read', ToolCall('r', 'read',
        json.dumps(dict(repository='repo', path='a.py')))), control)
    args = dict(repository='repo', path='a.py', sha256=json.loads(read.output_json)['sha256'])
    rejected = reply([dict(name='edit', arguments=args)], {'edit': TOOL_SPECS['edit']})
    assert rejected['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'invalid_cli_reply'
    assert source.read_bytes() == b'value = 1\nother = 3\n'
    args['changes'] = [dict(search='value = 1', replace='value = 2'), dict(search='other = 3', replace='other = 4')]
    output = reply([dict(name='edit', arguments=args)], {'edit': TOOL_SPECS['edit']})
    call = output['choices'][0]['message']['tool_calls'][0]
    result = tools.execute(EffectRequest('s', 'edit', ToolCall(call['id'], call['function']['name'],
        call['function']['arguments'])), control)
    assert result.status == 'succeeded'
    assert source.read_bytes() == b'value = 2\nother = 4\n'
    assert tools.observations()['effects'] == 1
    # Constraining generated pins never authorizes a changed on-disk version.
    args['sha256'] = json.loads(result.output_json)['sha256']
    args['changes'] = [dict(search='value = 2', replace='value = 8')]
    source.write_bytes(b'value = 99\n')
    refused = tools.execute(EffectRequest('s', 'stale-edit', ToolCall('stale', 'edit',
        json.dumps(args))), control)
    assert refused.status == 'refused'
    assert json.loads(refused.output_json)['reason'] == 'source_changed_read_again'
    assert source.read_bytes() == b'value = 99\n'


def test_canonical_schema_selects_the_action_before_generating_its_arguments():
    from compute_plane.native_models import encode
    body=request(TOOL_SPECS)
    schema=json.loads(encode(body['response_schema']))
    for branch in schema['properties']['actions']['items']['anyOf']:
        assert list(branch['properties'])==['action','arguments']
    assert json.loads(body['prompt'])['argument_encoding']=='native-json-object/4'


@pytest.mark.parametrize('encoding',['native-json-object/1','native-json-object/2','native-json-object/3'])
@pytest.mark.parametrize('valid',[True,False])
def test_discriminator_fields_are_versioned_without_accepting_an_ambiguous_old_field(encoding,valid):
    name_key='action' if encoding.endswith('/3') else 'name'
    if not valid: name_key='name' if name_key=='action' else 'action'
    schema={'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}
    decision={'text':'','actions':[{name_key:'inspect','arguments':{'path':'a.py'}}]}
    rows=[{'type':'turn.started'},{'type':'item.completed','item':{'type':'agent_message','text':json.dumps(decision)}},
          {'type':'turn.completed'}]
    result=decode(rows,'fixture',structured=True,argument_encoding=encoding,argument_schemas={'inspect':schema})
    calls=result['choices'][0]['message']['tool_calls']
    assert len(calls)==1 and calls[0]['function']['name']==('inspect' if valid else 'invalid_cli_reply')


def test_search_contract_cannot_generate_the_empty_content_query_the_executor_refuses():
    body=request({'search':TOOL_SPECS['search']})
    query=body['response_schema']['properties']['actions']['items']['anyOf'][0]['properties']['arguments']['properties']['query']
    assert query['minLength']==1
    assert 'nonempty' in query['description'].casefold()


def test_quoted_tool_history_decodes_machine_json_without_interpreting_user_text():
    source='packet = decode(content.split("\\n\\n", 1))\nROOT = "C:\\files"\n'
    machine=dict(text=source,sha256='a'*64)
    user='{"text":"keep this user text verbatim"}'
    records=[dict(role='user',content=user),
        dict(role='assistant',content='Keep prose.',tool_calls=[dict(function=dict(
            name='read',arguments=json.dumps(dict(path='a.py'))))]),
        dict(role='tool',tool_call_id='one',content=json.dumps(machine)),
        dict(role='tool',tool_call_id='two',content='Plain tool text')]
    before=json.dumps(records)
    body=native_cli.cli_request(dict(model='fixture',messages=records,tools=[]),
                                structured=True,native_arguments=True)
    actual=json.loads(body['prompt'])['conversation_records']
    assert actual[0]['content']==user and actual[1]['content']=='Keep prose.'
    assert actual[1]['tool_calls'][0]['function']['arguments']==dict(path='a.py')
    assert actual[2]['content']==machine and actual[2]['content']['text']==source
    assert actual[2]['tool_call_id']=='one' and actual[3]['content']=='Plain tool text'
    assert json.dumps(records)==before


def test_request_preparation_keeps_large_contexts_separate_from_reply_limits():
    from compute_plane.native_models import object_json, MAX_WIRE_BYTES
    from tare_tools_agent_runtime import ConversationRejected
    body=native_cli.cli_request(dict(model='fixture',messages=[dict(role='user',
        content='x'*(MAX_WIRE_BYTES+1000))],tools=[]),structured=True,native_arguments=True)
    assert len(json.loads(body['prompt'])['conversation_records'][0]['content'])>MAX_WIRE_BYTES
    with pytest.raises(ConversationRejected):
        object_json(body['prompt'])  # Untrusted individual replies retain their original bound.


def test_nested_optional_fields_are_omitted_without_dropping_declared_nulls():
    schema={'type':'object','properties':{'rows':{'type':'array','items':{'type':'object',
        'properties':{'name':{'type':'string'},'note':{'type':'string'},
                      'explicit':{'type':['string','null']}},
        'required':['name'],'additionalProperties':False}}},
        'required':['rows'],'additionalProperties':False}
    value=dict(rows=[dict(name='one',note=None,explicit={'value':None})])
    output=reply([dict(name='nested',arguments=value)],{'nested':{'parameters':schema}})
    actual=json.loads(output['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    assert actual==dict(rows=[dict(name='one',explicit=None)])


def test_codex_adapter_selects_object_transport_without_changing_tool_arguments():
    from compute_plane.native_models import ModelProfile, NativeModelAdapter
    from tests.test_native_models import execution, request as model_request
    observed=[]
    specs={name:TOOL_SPECS[name] for name in ('read','edit','verify')}
    class Client:
        endpoint='cli://codex'
        def post_observed(self,path,body,control):
            observed.append(body)
            assert json.loads(body['prompt'])['argument_encoding']=='native-json-object/4'
            args=dict(repository='repo',path='a.py',start=None,end=None,start_column=None,sha256=None)
            return reply([dict(name='read',arguments=args)],specs), {'cleanup_verified':True}
    client=Client()
    model=NativeModelAdapter(ModelProfile('vendor-cli','fixture',32768,4096,'fixture',client.endpoint),
        client,system='Read the requested source.',tool_specs_json=json.dumps(specs),
        counter_protocol='utf8-envelope-estimate-v1')
    result=model.generate(model_request(),execution())
    assert len(observed)==1 and len(result.tool_calls)==1
    assert result.tool_calls[0].name=='read'
    assert json.loads(result.tool_calls[0].arguments_json)==dict(repository='repo',path='a.py')


@pytest.mark.parametrize('timeout,expected',[
    ({'omitted':True},{}), ({'value':None},{'timeout_seconds':None}),
    ({'value':30},{'timeout_seconds':30})])
def test_optional_nullable_timeout_retains_omission_or_explicit_value(timeout,expected):
    spec=TERMINAL_SPECS['terminal_start']
    args=dict(repository='repo',argv=['worker'],mode=None,timeout_seconds=timeout,columns=None,rows=None)
    output=reply([dict(name='terminal_start',arguments=args)],{'terminal_start':spec})
    actual=json.loads(output['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    assert actual==dict(repository='repo',argv=['worker'],**expected)


@pytest.mark.parametrize('wrapper',[None,{}, {'omitted':False},{'omitted':True,'value':None},{'value':None,'extra':1}])
def test_invalid_nullable_presence_does_not_dispatch_any_action(wrapper):
    schema={'type':'object','properties':{'option':{'type':['string','null']}},'additionalProperties':False}
    output=reply([dict(name='task',arguments={'option':{'value':'good'}}),
                  dict(name='task',arguments={'option':wrapper})],{'task':{'parameters':schema}})
    calls=output['choices'][0]['message']['tool_calls']
    assert len(calls)==1 and calls[0]['function']['name']=='invalid_cli_reply'


def test_local_definitions_roundtrip_arrays_optional_members_and_escaped_definition_names():
    schema={'$defs':{'Source/item':{'type':'object','properties':{
        'path':{'type':'string'},'status':{'anyOf':[{'type':'string'},{'type':'null'}]}},
        'required':['path'],'additionalProperties':False}},'type':'object',
        'properties':{'sources':{'type':'array','items':{'$ref':'#/$defs/Source~1item'}}},
        'required':['sources'],'additionalProperties':False}
    before=json.dumps(schema,sort_keys=True)
    specs={'handoff':{'parameters':schema}}
    body=request(specs)
    wire=json.loads(body['prompt'])['action_catalog'][0]['wire_arguments_schema']
    assert '$ref' not in json.dumps(wire) and '$defs' not in wire
    args=dict(sources=[dict(path='one.py',status={'omitted':True}),
                       dict(path='two.py',status={'value':None})])
    output=reply([dict(name='handoff',arguments=args)],specs)
    actual=json.loads(output['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    assert actual==dict(sources=[dict(path='one.py'),dict(path='two.py',status=None)])
    assert json.dumps(schema,sort_keys=True)==before


@pytest.mark.parametrize('ref,definitions',[
    ('https://example.invalid/schema',{}),('#/$defs/Missing',{}),
    ('#/$defs/Recursive',{'Recursive':{'type':'object','properties':{
        'next':{'$ref':'#/$defs/Recursive'}},'additionalProperties':False}})])
def test_unsupported_references_fail_before_inference(ref,definitions):
    from tare_tools_agent_runtime import ModelFailure
    schema={'$defs':definitions,'type':'object','properties':{'item':{'$ref':ref}},
            'required':['item'],'additionalProperties':False}
    with pytest.raises(ModelFailure) as raised:
        request({'task':{'parameters':schema}})
    assert raised.value.code=='CLI_ACTION_SCHEMA_UNSUPPORTED' and raised.value.not_started


def test_union_nullable_type_and_legacy_object_decisions_keep_their_meanings():
    schema={'type':'object','properties':{'option':{'anyOf':[
        {'type':['string','null']},{'type':'integer'}]}},'additionalProperties':False}
    specs={'task':{'parameters':schema}}
    for encoding,args in [('native-json-object/1',{'option':None}),
                          ('native-json-object/2',{'option':{'value':None}}),
                          ('native-json-object/3',{'option':{'value':None}})]:
        output=reply([dict(name='task',arguments=args)],specs,encoding)
        assert json.loads(output['choices'][0]['message']['tool_calls'][0]['function']['arguments'])=={'option':None}


@pytest.mark.parametrize('endpoint',['cli://codex','cli://agy','cli://claude'])
@pytest.mark.parametrize('remaining',[1,2])
def test_explorer_final_delivery_choice_survives_structured_cli_transport(endpoint,remaining):
    from dataclasses import replace
    from types import SimpleNamespace
    from compute_plane.native_models import ModelProfile, NativeModelAdapter
    from tests.test_native_models import request as model_request
    client=SimpleNamespace(endpoint=endpoint,version='2.1.293')
    adapter=NativeModelAdapter(ModelProfile('vendor-cli','fixture',32768,4096,'fixture',endpoint),
        client,system='Inspect sources.',tool_specs_json=json.dumps({'read':TOOL_SPECS['read']}),
        counter_protocol='utf8-envelope-estimate-v1',force_final_delivery=True)
    body=adapter.wire(replace(model_request(),capabilities=('read',),remaining_model_calls=remaining))
    envelope=json.loads(body['prompt'])
    # Claude declares the catalog as native tools; the other CLIs quote it for their response schema.
    native=endpoint=='cli://claude'
    catalog=body['native_tools'] if native else envelope['action_catalog']
    assert {item['name'] for item in catalog}=={'read','tare_deliver'}
    ordinary=adapter.wire(replace(model_request(),capabilities=('read',),remaining_model_calls=3))
    assert body.get('response_schema')==ordinary.get('response_schema')
    assert body['system']==ordinary['system']
    assert catalog==(ordinary['native_tools'] if native else json.loads(ordinary['prompt'])['action_catalog'])
    if remaining==1:
        assert envelope['turn_constraints']=={'required_action':'tare_deliver'}
    else:
        assert 'turn_constraints' not in envelope
