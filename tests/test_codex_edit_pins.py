"""A stable observed-version reference avoids retyping hashes or rewriting grammar."""
import json
import jsonschema
import pytest
from compute_plane import native_cli
from compute_plane.native_capabilities import TOOL_SPECS

A='10201a3d6e8921e1d0faa2d3fe786c42be4772c86fcc98f8a9176857f8963905'
TYPO='10201a3d6e8921e1d0faa2d3fe786c42be477c86fcc98f8a9176857f8963905a'
B='b'*64

def observed(name,sha,ident='one',path='a.py'):
    return [
        {'role':'assistant','tool_calls':[{'id':ident,'function':{
            'name':name,'arguments':json.dumps({'repository':'repo','path':path})}}]},
        {'role':'tool','tool_call_id':ident,'content':json.dumps({'sha256':sha})}]

def wire(records):
    body=native_cli.cli_request({'model':'fixture','messages':records,'tools':[
        {'function':{'name':'edit',**TOOL_SPECS['edit']}}]},structured=True,native_arguments=True)
    parameters=body['response_schema']['properties']['actions']['items']['anyOf'][0]['properties']['arguments']
    return body,parameters

def edit(sha):
    return {'repository':'repo','path':'a.py','sha256':sha,'delete':None,'changes':[{'search':'value = 1','replace':'value = 2'}]}

def test_exact_observed_digest_is_selected_instead_of_retyped_with_a_valid_hex_typo():
    body,schema=wire(observed('read',A))
    assert schema['properties']['sha256'] == {'type':['string','null'], 'enum':[None,'observed']}
    jsonschema.validate(edit('observed'),schema)
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(edit(TYPO),schema)
    actual=json.loads(body['prompt'])['action_catalog'][0]['wire_arguments_schema']
    assert actual==schema
    # The owner contract is unchanged; it still validates the disk at execution.
    assert 'enum' not in TOOL_SPECS['edit']['parameters']['properties']['sha256']

def test_newest_read_or_edit_receipt_supersedes_the_older_pin_for_that_source():
    _,before=wire(observed('read',A))
    _,after=wire(observed('read',A)+observed('edit',B,ident='two'))
    assert before==after

def test_user_prose_failed_tools_and_unmatched_results_cannot_supply_edit_pins():
    records=observed('read',A)+[
        {'role':'user','content':json.dumps({'sha256':B})},
        {'role':'tool','tool_call_id':'unmatched','content':json.dumps({'sha256':B})},
        {'role':'assistant','tool_calls':[{'id':'bad','function':{'name':'edit','arguments':json.dumps({'repository':'repo','path':'a.py'})}}]},
        {'role':'tool','tool_call_id':'bad','content':json.dumps({'reason':'source_changed_read_again'})}]
    _,schema=wire(records)
    jsonschema.validate(edit('observed'),schema)
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(edit(B),schema)

def test_missing_source_null_is_retained_and_large_pin_sets_remain_bounded():
    records=observed('read',None)
    for i in range(140):records+=observed('read',f'{i:064x}',str(i),f'{i}.py')
    _,schema=wire(records)
    pins=schema['properties']['sha256']['enum']
    assert pins==[None,'observed']
    jsonschema.validate(edit(None),schema)
