"""Wire fixtures and a loopback HTTP endpoint, never external model inference."""
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from tare_tools_agent_runtime import (
    BindingRef, CancellationToken, ConversationLog, EffectResult, ExecutionControl,
    ModelFailure, ModelRequest, NativeSession, ProjectionSpec,
)
from compute_plane.native_models import ModelProfile, NativeHttpClient, NativeModelAdapter, DELIVERY_TOOL


SPECS = json.dumps({name: {"description": name + " scoped source", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
    for name in ("read", "edit", "verify")})


@pytest.mark.parametrize('forced,remaining,blocked', [(True,1,True),(True,2,False),(False,1,False)])
def test_ignored_required_final_cli_action_stops_once_with_usage(forced, remaining, blocked):
    raw=response('openai-chat',name='invalid_cli_reply',args={'ignored':True},usage={'input_tokens':10,'output_tokens':2})
    raw['_proposal_rejection']={'stage':'tool_choice'}
    client=Client([raw]);client.endpoint='cli://agy'
    cli=ModelProfile('vendor-cli','fixture-model',8192,1024,'offline fixture','cli://agy',parameters_json='{"effort":"high"}')
    observed=[]
    adapter=NativeModelAdapter(cli,client,system='Stable',tool_specs_json=SPECS,
        force_final_delivery=forced,observe=observed.append,counter_protocol='utf8-envelope-estimate-v1')
    if blocked:
        with pytest.raises(ModelFailure,match='CLI_REQUIRED_ACTION'):
            adapter.generate(replace(request(),remaining_model_calls=remaining),execution())
    else:
        reply=adapter.generate(replace(request(),remaining_model_calls=remaining),execution())
        assert not reply.tool_calls and json.loads(reply.metadata_json)['recovery']['code']=='INVALID_CLI_REPLY'
    assert len(client.calls)==1 and observed[-1]['usage']==raw['usage']


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse'])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('text,partial', [('Hello!', False), ('Hello', True), (' ', False)])
def test_text_completes_only_an_opted_in_complete_nonempty_turn(protocol, enabled, text, partial):
    raw = ({'model': 'fixture-model', 'choices': [{'finish_reason': 'length' if partial else 'stop',
            'message': {'role': 'assistant', 'content': text}}]} if protocol == 'openai-chat' else
           {'stopReason': 'max_tokens' if partial else 'end_turn',
            'output': {'message': {'role': 'assistant', 'content': [{'text': text}]}}})
    adapter = NativeModelAdapter(profile(protocol), Client([raw]), system='Stable',
                                 tool_specs_json=SPECS, text_completes_turn=enabled)
    reply = adapter.generate(request(), execution())
    assert reply.proposes_delivery == (enabled and not partial and bool(text.strip()))
    assert not reply.tool_calls and reply.text == text
    assert json.loads(reply.metadata_json)['provider_stop_reason'] == (
        raw['choices'][0]['finish_reason'] if protocol == 'openai-chat' else raw['stopReason'])
    # Completion policy can repair old conversations without changing their wire identity.
    assert adapter.profile_digest == NativeModelAdapter(profile(protocol), Client([]),
        system='Stable', tool_specs_json=SPECS).profile_digest


@pytest.mark.parametrize('reason', ['length', 'content_filter', 'malformed_function_call', 'other'])
def test_incomplete_provider_stop_is_retained_without_tools_or_a_silent_retry(reason):
    raw = {'model': 'fixture-model', 'choices': [{'finish_reason': reason,
        'message': {'role': 'assistant', 'content': None}}]}
    client = Client([raw])
    adapter = NativeModelAdapter(profile('openai-chat'), client, system='Stable', tool_specs_json=SPECS)
    reply = adapter.generate(request(), execution())
    assert not reply.complete and not reply.tool_calls and not reply.proposes_delivery
    assert json.loads(reply.metadata_json)['provider_stop_reason'] == reason


def test_a_conversation_summary_is_text_not_a_delivery():
    raw = {'model': 'fixture-model', 'choices': [{'finish_reason': 'stop',
           'message': {'role': 'assistant', 'content': 'Objective: ...'}}]}
    adapter = NativeModelAdapter(profile('openai-chat'), Client([raw]), system='Stable',
                                 tool_specs_json=SPECS, text_completes_turn=True)
    reply = adapter.generate(replace(request(), call_id='s/summary/1', purpose='summary'), execution())
    assert reply.text == 'Objective: ...' and not reply.proposes_delivery


@pytest.mark.parametrize('protocol',['openai-chat','bedrock-converse'])
@pytest.mark.parametrize('text,summary,expected',[
    ('AC-01: observed; AC-02: not inspected.','Review complete.',
     'AC-01: observed; AC-02: not inspected.\n\nReview complete.'),
    ('Review complete.','Review complete.','Review complete.'),
    ('  ','Review complete.','Review complete.')])
def test_delivery_retains_the_answer_instead_of_replacing_it_with_its_summary(protocol,text,summary,expected):
    raw=response(protocol,DELIVERY_TOOL,{'summary':summary})
    if protocol=='openai-chat': raw['choices'][0]['message']['content']=text
    else: raw['output']['message']['content'].insert(0,{'text':text})
    adapter=NativeModelAdapter(profile(protocol),Client([raw]),system='Stable',tool_specs_json=SPECS)
    reply=adapter.generate(request(),execution())
    assert reply.proposes_delivery and reply.text==expected and not reply.tool_calls
    original=json.loads(reply.metadata_json)['assistant']
    assert (original['content'] if protocol=='openai-chat' else original['content'][0]['text'])==text


def test_conversational_completion_cannot_bypass_typed_delivery():
    with pytest.raises(ValueError, match='untyped'):
        NativeModelAdapter(profile('openai-chat'), Client([]), system='Stable', tool_specs_json=SPECS,
            text_completes_turn=True, delivery_schema_json='{}', validate_delivery=lambda value: value)


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse'])
@pytest.mark.parametrize('sha', [None, 'null', 'a'*64])
def test_provider_hash_argument_type_is_preserved_without_coercion(protocol, sha):
    args = {'path': 'answer.json', 'sha256': sha, 'search': '', 'replace': '{}'}
    adapter = NativeModelAdapter(profile(protocol), Client([response(protocol, 'edit', args)]),
                                 system='Stable', tool_specs_json=SPECS)
    reply = adapter.generate(request(), execution())
    actual = json.loads(reply.tool_calls[0].arguments_json)
    assert actual == args and type(actual['sha256']) is type(sha)


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse'])
def test_explicit_estimate_keeps_tokens_unknown_and_never_calls_a_counter(protocol):
    client = Client([])
    measured = NativeModelAdapter(profile(protocol), client, system='Stable', tool_specs_json=SPECS)
    adapter = NativeModelAdapter(profile(protocol), client, system='Stable', tool_specs_json=SPECS,
                                 counter_protocol='utf8-envelope-estimate-v1')
    req = replace(request(), instruction='Fix ação 中文 🦜')
    row = adapter.count(req, execution())
    assert row['status'] == 'estimated' and row['input_tokens'] is None
    assert row['estimated_input_tokens'] > len(req.instruction.encode('utf-8'))
    assert row['limitation'] and not client.calls and row['client_invocations'] == 0
    assert adapter.profile_digest != measured.profile_digest
    expanded = adapter.count(replace(req, instruction=req.instruction * 100), execution())
    assert expanded['estimated_input_tokens'] > row['estimated_input_tokens']
    assert expanded['request_sha256'] != row['request_sha256']
    # An unavailable measured counter remains unavailable, even though an
    # estimator exists. No automatic downgrade after provider errors.
    failed = measured.count(req, execution())
    assert failed['status'] == 'unavailable' and 'estimated_input_tokens' not in failed


def profile(protocol):
    params = {"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 512}} if protocol == "openai-chat" else {
        "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 512}}}
    return ModelProfile(protocol, "fixture-model", 8192, 1024, "fixture-deployment", "http://fixture",
                        parameters_json=json.dumps(params))


def execution():
    return ExecutionControl(BindingRef("work", "binding", "owner", 1, "opaque"),
                            CancellationToken(), 100, lambda: 0)


def request():
    context = ConversationLog().project(ProjectionSpec(1, "native-full-v1"))
    return ModelRequest("s", "s/model/1", "Fix and verify app.py", ("read", "edit", "verify"), context)


def response(protocol, name="read", args=None, *, usage=None, partial=False):
    args = args or ({"summary": "changed and verified"} if name == DELIVERY_TOOL else {"path": "app.py"})
    if protocol == "openai-chat":
        return {"model": "fixture-model", "choices": [{"finish_reason": "length" if partial else "tool_calls",
            "message": {"role": "assistant", "content": None, "reasoning_content": "reasoning",
                "tool_calls": [{"id": "provider-id", "type": "function", "function": {
                    "name": name, "arguments": '{"path":' if partial else json.dumps(args)}}]}}], "usage": usage}
    return {"stopReason": "max_tokens" if partial else "tool_use", "output": {"message": {
        "role": "assistant", "content": [{"reasoningContent": {"reasoningText": {
            "text": "reasoning", "signature": "opaque-signature"}}},
            {"toolUse": {"toolUseId": "provider-id", "name": name, "input": args}}]}}, "usage": usage}


class Client:
    endpoint = "http://fixture"
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, path, body, control):
        self.calls.append((path, json.loads(json.dumps(body))))
        reply = next(self.responses)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize('protocol', ['openai-chat','bedrock-converse'])
def test_complete_tool_batch_is_replayed_with_every_provider_result(protocol):
    first = response(protocol)
    if protocol == 'openai-chat':
        assistant=first['choices'][0]['message']
        call=assistant['tool_calls'][0]
        assistant['tool_calls']=[dict(call,id='id-'+str(i)) for i in range(3)]
    else:
        content=first['output']['message']['content']
        call=content.pop()['toolUse']
        content.extend({'toolUse':dict(call,toolUseId='id-'+str(i))} for i in range(3))
    client=Client([first,response(protocol,DELIVERY_TOOL)])
    adapter=NativeModelAdapter(profile(protocol),client,system='Stable instructions',tool_specs_json=SPECS)
    class Tools:
        def execute(self, request, control):
            return EffectResult(request.effect_id,'succeeded','{}',True)
    journal=ConversationLog()
    outcome=NativeSession(session_id='s',instruction='Read and deliver',capabilities=('read','edit','verify'),
        model=adapter,tools=Tools(),journal=journal).run(execution())
    assert outcome.status=='proposed' and outcome.model_calls==2
    history=client.calls[-1][1]['messages']
    if protocol=='openai-chat':
        results=[m['tool_call_id'] for m in history if m['role']=='tool']
    else:
        results=[b['toolResult']['toolUseId'] for m in history for b in m['content'] if 'toolResult' in b]
    assert results==['id-0','id-1','id-2']
    from compute_plane.native_usage import native_usage_report
    usage=native_usage_report(journal,[],contracts={})
    assert usage['calls'][0]['effect_ids']==['s/effect/1/1','s/effect/1/2','s/effect/1/3']
    assert usage['calls'][1]['effect_ids']==[]


@pytest.mark.parametrize("protocol", ["openai-chat", "bedrock-converse"])
def test_native_session_through_codec_read_edit_verify_delivery_and_stable_prefix(protocol):
    client = Client([response(protocol, name) for name in ("read", "edit", "verify", DELIVERY_TOOL)])
    rows = []
    adapter = NativeModelAdapter(profile(protocol), client, system="Stable instructions", tool_specs_json=SPECS, observe=rows.append)

    class Tools:
        calls = []
        def execute(self, req, control):
            self.calls.append(req)
            return EffectResult(req.effect_id, "succeeded", '{"observed":"fixture"}', True)

    tools = Tools()
    session = NativeSession(session_id="s", instruction="Fix and verify app.py", capabilities=("read", "edit", "verify"),
                            model=adapter, tools=tools, journal=ConversationLog())
    result = session.run(execution())
    assert result.status == "proposed" and len(client.calls) == 4
    assert [r.tool_call.name for r in tools.calls] == ["read", "edit", "verify"]
    assert len({r.tool_call.call_id for r in tools.calls}) == 3  # Same provider ID on distinct turns.
    first, second = client.calls[0][1], client.calls[1][1]
    if protocol == "openai-chat":
        # Only the current tail budget is transient; system and initial context stay cached.
        assert second['messages'][:2] == first['messages'][:2]
        assert first["tools"] == second["tools"]
        assert first["chat_template_kwargs"]["enable_thinking"] is True
        assert second["messages"][-3]["reasoning_content"] == "reasoning"
        assert second["messages"][-2]["tool_call_id"] == "provider-id"
        assert '15 model calls remain' in second['messages'][-1]['content']
    else:
        assert first['messages'][0]['content'][0] == second['messages'][0]['content'][0]
        assert first["system"] == second["system"] and first["toolConfig"] == second["toolConfig"]
        assert first["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 512
        assert first["toolConfig"]["toolChoice"] == {"auto": {}}
        assert second["messages"][-2]["content"][0]["reasoningContent"]["reasoningText"]["signature"] == "opaque-signature"
        assert second["messages"][-1]["content"][0]["toolResult"]["toolUseId"] == "provider-id"
        assert '15 model calls remain' in second['messages'][-1]['content'][-1]['text']
    assert all(row["usage"] is None for row in rows)


@pytest.mark.parametrize("protocol", ["openai-chat", "bedrock-converse"])
def test_partial_tool_arguments_are_preserved_but_never_become_action(protocol):
    # A call cut by the output limit is kept as evidence and answered with "use smaller steps".
    client = Client([response(protocol, partial=True)])
    adapter = NativeModelAdapter(profile(protocol), client, system="system", tool_specs_json=SPECS)
    reply = adapter.generate(request(), execution())
    assert not reply.tool_calls and not reply.proposes_delivery
    metadata = json.loads(reply.metadata_json)
    assert metadata["assistant"]["role"] == "assistant" and metadata["recovery"]["code"] == "TRUNCATED_TOOL_CALL"


@pytest.mark.parametrize("protocol", ["openai-chat", "bedrock-converse"])
def test_raw_usage_and_cache_counters_are_not_summed_or_missing_as_zero(protocol):
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 80}} if protocol == "openai-chat" else {
        "inputTokens": 10, "outputTokens": 20, "cacheReadInputTokens": 80}
    rows = []
    adapter = NativeModelAdapter(profile(protocol), Client([response(protocol, usage=usage)]),
                                 system="system", tool_specs_json=SPECS, observe=rows.append)
    reply = adapter.generate(request(), execution())
    assert json.loads(reply.metadata_json)["usage"] == usage == rows[0]["usage"]
    assert "cacheWriteInputTokens" not in rows[0]["usage"]


def test_cache_system_requires_evidence_and_does_not_infer_support_from_model_name():
    with pytest.raises(ValueError, match="support evidence"):
        replace(profile("bedrock-converse"), cache_system=True)
    p = replace(profile("bedrock-converse"), model="opaque-profile", cache_system=True, cache_support_source="fixture-qualified")
    adapter = NativeModelAdapter(p, Client([]), system="stable", tool_specs_json=SPECS)
    wire = adapter.wire(request())
    assert wire["system"] == [{"text": "stable"}, {"cachePoint": {"type": "default"}}]
    assert not any("cachePoint" in x for m in wire["messages"] for x in m["content"])


@pytest.mark.parametrize("key", ["tool_choice", "tools", "messages", "stream", "max_tokens", "model", "n"])
def test_reserved_parameters_cannot_override_profile_or_disable_contract(key):
    with pytest.raises(ValueError, match="reserved"):
        replace(profile("openai-chat"), parameters_json=json.dumps({key: False}))


def test_count_tokens_uses_system_tools_and_history_of_actual_request():
    client = Client([{"inputTokens": 333}])
    adapter = NativeModelAdapter(profile("bedrock-converse"), client, system="system", tool_specs_json=SPECS)
    wire = adapter.wire(request())
    count = adapter.count(request(), execution())
    assert count["input_tokens"] == 333 and count["method"] == "bedrock_count_tokens"
    assert client.calls[0][1] == {"input": {"converse": {k: wire[k] for k in ("system", "messages", "toolConfig")}}}


@pytest.mark.parametrize("failure", [ModelFailure("HTTP_403"), ModelFailure("HTTP_400"), TimeoutError(), {"inputTokens": True}])
def test_count_failure_remains_unavailable_without_hidden_estimate_or_model_call(failure):
    client = Client([failure])
    adapter = NativeModelAdapter(profile("bedrock-converse"), client, system="system", tool_specs_json=SPECS)
    count = adapter.count(request(), execution())
    assert count["status"] == "unavailable" and count["input_tokens"] is None and count["reason"]
    assert len(client.calls) == 1 and client.calls[0][0].endswith("/count-tokens")


def test_local_count_applies_same_wire_template_including_tools_and_thinking():
    counter = Client([{"prompt": "rendered tool schema and prompt"}, {"tokens": [1, 2, 3]}])
    client = Client([])
    adapter = NativeModelAdapter(profile("openai-chat"), client, system="system", tool_specs_json=SPECS)
    result = adapter.count(request(), execution(), local_counter=counter)
    assert result["input_tokens"] == 3
    assert counter.calls[0] == ("/apply-template", adapter.wire(request()))
    assert counter.calls[1] == ("/tokenize", {"content": "rendered tool schema and prompt", "add_special": True, "parse_special": True})
    assert not client.calls


def test_429_is_one_call_no_helper_model_or_implicit_retry():
    client = Client([ModelFailure("HTTP_429", retryable=True)])
    rows = []
    adapter = NativeModelAdapter(profile("bedrock-converse"), client, system="system", tool_specs_json=SPECS, observe=rows.append)
    with pytest.raises(ModelFailure) as caught:
        adapter.generate(request(), execution())
    assert caught.value.retryable and len(client.calls) == 1
    assert rows[0]["error"] == "HTTP_429" and rows[0]["usage"] is None


def test_complete_malformed_provider_response_is_a_known_protocol_failure():
    raw = response("openai-chat")
    raw["model"] = "wrong-model"
    rows = []
    adapter = NativeModelAdapter(profile("openai-chat"), Client([raw]), system="system", tool_specs_json=SPECS, observe=rows.append)
    with pytest.raises(ModelFailure, match="INVALID_PROVIDER_RESPONSE") as caught:
        adapter.generate(request(), execution())
    assert not caught.value.retryable
    assert rows[0]["status"] == "invalid_response" and rows[0]["response_sha256"]


def test_failed_optional_observer_does_not_discard_a_valid_response():
    def fail(row):
        raise OSError("sink full")
    adapter = NativeModelAdapter(profile("openai-chat"), Client([response("openai-chat")]),
                                 system="system", tool_specs_json=SPECS, observe=fail)
    assert adapter.generate(request(), execution()).complete
    assert adapter.observation_failures == 1


def test_no_actual_http_request_after_cancellation():
    control = execution()
    control.cancellation.cancel()
    client = NativeHttpClient("http://127.0.0.1:1", credential=lambda: pytest.fail("credential read after stop"))
    with pytest.raises(InterruptedError):
        client.post("/chat/completions", {}, control)


@pytest.mark.parametrize("protocol", ["openai-chat", "bedrock-converse"])
def test_real_loopback_http_preserves_parameters_auth_and_response(protocol):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))), self.headers.get("Authorization")))
            data = json.dumps(response(protocol, DELIVERY_TOOL)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = []
        client = NativeHttpClient(f"http://127.0.0.1:{server.server_port}/v1", credential=lambda: "fixture-token")
        adapter = NativeModelAdapter(replace(profile(protocol), endpoint=client.endpoint), client,
                                     system="system", tool_specs_json=SPECS, observe=rows.append)
        assert adapter.generate(request(), execution()).proposes_delivery
        assert seen[0][0] == ("/v1/chat/completions" if protocol == "openai-chat" else "/v1/model/fixture-model/converse")
        assert seen[0][2] == "Bearer fixture-token"
        if protocol == "openai-chat":
            assert seen[0][1]["chat_template_kwargs"]["enable_thinking"] is True
        else:
            assert seen[0][1]["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 512
        assert "fixture-token" not in json.dumps(rows)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("protocol", ["openai-chat", "bedrock-converse"])
def test_prose_continuation_has_stable_explicit_user_turn(protocol):
    raw = response(protocol)
    if protocol == "openai-chat":
        raw["choices"][0] = {"finish_reason": "stop", "message": {"role": "assistant", "content": "Let me inspect."}}
    else:
        raw["stopReason"] = "end_turn"
        raw["output"]["message"]["content"] = [{"text": "Let me inspect."}]
    client = Client([raw, response(protocol, DELIVERY_TOOL)])
    adapter = NativeModelAdapter(profile(protocol), client, system="system", tool_specs_json=SPECS)
    class Tools:
        def execute(self, *args):
            pytest.fail("prose must not execute tools")
    session = NativeSession(session_id="s", instruction="task", capabilities=("read", "edit", "verify"),
                            model=adapter, tools=Tools(), journal=ConversationLog())
    assert session.run(execution()).status == "proposed"
    assert client.calls[1][1]["messages"][-1]["role"] == "user"


def test_replay_profile_mismatch_refuses_wire_before_network_call():
    old = NativeModelAdapter(profile("openai-chat"), Client([response("openai-chat", DELIVERY_TOOL)]),
                             system="original", tool_specs_json=SPECS)
    reply = old.generate(request(), execution())
    from dataclasses import asdict
    log = ConversationLog()
    log.append({"kind": "model_returned", "call_id": request().call_id, "reply": asdict(reply)})
    client = Client([])
    new = NativeModelAdapter(profile("openai-chat"), client, system="changed", tool_specs_json=SPECS)
    replay = replace(request(), context=log.project(ProjectionSpec(1, "native-full-v1")))
    with pytest.raises(ValueError, match="profile changed"):
        new.wire(replay)
    assert not client.calls


def mixed_delivery(protocol, delivery_first):
    raw = response(protocol, DELIVERY_TOOL)
    action = response(protocol)
    if protocol == 'openai-chat':
        calls = raw['choices'][0]['message']['tool_calls']
        other = action['choices'][0]['message']['tool_calls'][0]
        other['id'] = 'action-id'
        calls.insert(1 if delivery_first else 0, other)
    else:
        calls = raw['output']['message']['content']
        other = action['output']['message']['content'][-1]
        other['toolUse']['toolUseId'] = 'action-id'
        calls.insert(len(calls) if delivery_first else 0, other)
    return raw


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse'])
@pytest.mark.parametrize('delivery_first', [False, True])
def test_ordinary_delivery_shape_feedback_discards_batch_and_preserves_session(protocol, delivery_first):
    client = Client([mixed_delivery(protocol, delivery_first), response(protocol), response(protocol, DELIVERY_TOOL)])
    observed, effects = [], []
    adapter = NativeModelAdapter(profile(protocol), client, system='Stable', tool_specs_json=SPECS,
                                 observe=observed.append)
    class Tools:
        def execute(self, req, control):
            effects.append(req)
            return EffectResult(req.effect_id, 'succeeded', '{}', True)
    journal = ConversationLog()
    result = NativeSession(session_id='s', instruction='Read then deliver', capabilities=('read', 'edit', 'verify'),
        model=adapter, tools=Tools(), journal=journal).run(execution())
    assert result.status == 'proposed' and result.model_calls == 3
    assert len(effects) == 1 and effects[0].effect_id == 's/effect/2'
    feedback = 'Delivery must be the only call'
    assert feedback in json.dumps(client.calls[1][1])
    assert observed[0]['recovery']['code'] == 'INVALID_ORDINARY_DELIVERY'
    assert len([e for e in journal.project(ProjectionSpec(1, 'native-full-v1')).events()
                if e['kind'] == 'model_returned']) == 3
    key = 'tools' if protocol == 'openai-chat' else 'toolConfig'
    assert client.calls[0][1][key] == client.calls[1][1][key]


@pytest.mark.parametrize('args', [{'summary': ''}, {'summary': 1}, {'summary': 'ok', 'extra': True}])
def test_ordinary_delivery_shape_feedback_rejects_nonempty_or_extra_field_errors(args):
    adapter = NativeModelAdapter(profile('openai-chat'), Client([response('openai-chat', DELIVERY_TOOL, args)]),
                                 system='Stable', tool_specs_json=SPECS)
    reply = adapter.generate(request(), execution())
    assert not reply.proposes_delivery and not reply.tool_calls and not reply.text
    assert json.loads(reply.metadata_json)['recovery']['code'] == 'INVALID_ORDINARY_DELIVERY'


def test_ordinary_delivery_feedback_does_not_renew_the_call_budget():
    from tare_tools_agent_runtime import NativeBudget
    client = Client([mixed_delivery('openai-chat', False)])
    adapter = NativeModelAdapter(profile('openai-chat'), client, system='Stable', tool_specs_json=SPECS)
    class Tools:
        def execute(self, req, control):
            pytest.fail('the rejected batch must not execute')
    result = NativeSession(session_id='s', instruction='Read then deliver', capabilities=('read', 'edit', 'verify'),
        model=adapter, tools=Tools(), journal=ConversationLog(),
        budget=NativeBudget(max_model_calls=1)).run(execution())
    assert result.status != 'proposed' and result.model_calls == 1 and len(client.calls) == 1


def test_mixed_typed_delivery_still_fails_without_an_admissible_typed_repair():
    raw = mixed_delivery('openai-chat', True)
    raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps(
        {'summary': 'Observed', 'path': 'app.py'})
    adapter = typed_adapter('openai-chat', [raw], [])
    with pytest.raises(ModelFailure, match='INVALID_DELIVERY'):
        adapter.generate(replace(request(), remaining_model_calls=4), execution())


def typed_adapter(protocol, replies, observed):
    from pydantic import BaseModel

    class Proposal(BaseModel):
        summary: str
        path: str

    return NativeModelAdapter(profile(protocol), Client(replies), system='Stable', tool_specs_json=SPECS,
        delivery_schema_json=json.dumps(Proposal.model_json_schema()),
        validate_delivery=lambda args: Proposal.model_validate(args).model_dump(), observe=observed.append)


@pytest.mark.parametrize('protocol', ['openai-chat', 'bedrock-converse'])
@pytest.mark.parametrize('remaining', [3, None])
def test_typed_delivery_can_be_corrected_once_within_the_original_budget(protocol, remaining):
    from dataclasses import asdict
    usage = {'input_tokens': 10, 'output_tokens': 2}
    bad = response(protocol, DELIVERY_TOOL, {'summary': 'Observed'}, usage=usage)
    good = response(protocol, DELIVERY_TOOL, {'summary': 'Observed', 'path': 'app.py'})
    observed = []
    adapter = typed_adapter(protocol, [bad, good], observed)
    first = adapter.generate(replace(request(), remaining_model_calls=remaining), execution())
    assert not first.proposes_delivery and first.tool_calls == ()
    meta = json.loads(first.metadata_json)
    assert meta['recovery']['issues'] == [{'field': 'path', 'code': 'missing'}]
    assert 'delivery_payload' not in meta and observed[0]['usage'] == usage
    assert len(adapter.client.calls) == 1  # correction is a normal new turn, never a transport retry
    log = ConversationLog()
    log.append({'kind': 'model_returned', 'call_id': request().call_id, 'reply': asdict(first)})
    followup = replace(request(), call_id='s/model/2', remaining_model_calls=2 if remaining is not None else None,
                       context=log.project(ProjectionSpec(1, 'native-full-v1')))
    second = adapter.generate(followup, execution())
    assert second.proposes_delivery
    assert json.loads(second.metadata_json)['delivery_payload']['path'] == 'app.py'
    assert len(adapter.client.calls) == 2
    # A restarted adapter still sees the recorded repair and must not start another correction.
    restarted = typed_adapter(protocol, [bad], [])
    with pytest.raises(ModelFailure, match='INVALID_DELIVERY'):
        restarted.generate(followup, execution())


def test_invalid_typed_delivery_without_remaining_repair_room_stays_failed():
    observed = []
    adapter = typed_adapter('openai-chat', [response('openai-chat', DELIVERY_TOOL)], observed)
    with pytest.raises(ModelFailure, match='INVALID_DELIVERY'):
        adapter.generate(replace(request(), remaining_model_calls=1), execution())
    assert len(adapter.client.calls) == 1
    assert observed[-1]['delivery_validation']['issues'] == [{'field': 'path', 'code': 'missing'}]


def test_typed_repair_discards_the_whole_batch_and_a_second_invalid_proposal_fails():
    bad = response('openai-chat', DELIVERY_TOOL)
    bad['choices'][0]['message']['tool_calls'].extend(response('openai-chat')['choices'][0]['message']['tool_calls'])
    bad['choices'][0]['message']['tool_calls'][1]['id'] = 'different-effect-id'
    observed = []
    adapter = typed_adapter('openai-chat', [bad, response('openai-chat', DELIVERY_TOOL)], observed)
    first = adapter.generate(replace(request(), remaining_model_calls=4), execution())
    assert not first.proposes_delivery and first.tool_calls == ()
    with pytest.raises(ModelFailure, match='INVALID_DELIVERY'):
        adapter.generate(replace(request(), remaining_model_calls=3), execution())
    assert len(adapter.client.calls) == 2


@pytest.mark.parametrize('code',['native_artifact_source_not_observed','native_artifact_ignored_target','native_artifact_source_changed',
    'native_artifact_relative_source_required','native_artifact_accept_with_open_findings'])
@pytest.mark.parametrize('remaining', [2, None])
def test_only_local_citation_or_target_issues_can_be_corrected_without_relaxing_evidence_checks(code, remaining):
    from dataclasses import asdict
    observed=[]
    bad=response('openai-chat',DELIVERY_TOOL,{'summary':'Claim','path':'unread'})
    good=response('openai-chat',DELIVERY_TOOL,{'summary':'Observed','path':'read'})
    adapter=typed_adapter('openai-chat',[bad,good],observed)
    original=adapter.validate_delivery
    def validate(value):
        if value.get('path')=='unread':raise ValueError(code)
        return original(value)
    adapter.validate_delivery=validate
    if code not in {'native_artifact_source_not_observed','native_artifact_ignored_target'}:
        with pytest.raises(ModelFailure,match='INVALID_DELIVERY'):
            adapter.generate(replace(request(),remaining_model_calls=remaining),execution())
        assert len(adapter.client.calls)==1
        return
    first=adapter.generate(replace(request(),remaining_model_calls=remaining),execution())
    assert not first.proposes_delivery and first.tool_calls==()
    if code=='native_artifact_ignored_target':
        assert 'Git ignores' in json.loads(first.metadata_json)['recovery']['message']
    log=ConversationLog();log.append({'kind':'model_returned','call_id':request().call_id,'reply':asdict(first)})
    second=adapter.generate(replace(request(),call_id='s/model/2',remaining_model_calls=1 if remaining is not None else None,
        context=log.project(ProjectionSpec(1,'native-full-v1'))),execution())
    assert second.proposes_delivery and len(adapter.client.calls)==2
    assert json.loads(second.metadata_json)['delivery_payload']['path']=='read'


def test_endpoint_is_bound_to_profile_and_replay_identity():
    client = Client([])
    with pytest.raises(ValueError, match="endpoint"):
        NativeModelAdapter(replace(profile("openai-chat"), endpoint="http://another-host"),
                           client, system="system", tool_specs_json=SPECS)
    original = NativeModelAdapter(profile("openai-chat"), client, system="system", tool_specs_json=SPECS)
    other = Client([])
    other.endpoint = "http://another-host"
    changed = NativeModelAdapter(replace(profile("openai-chat"), endpoint=other.endpoint),
                                 other, system="system", tool_specs_json=SPECS)
    assert original.profile_digest != changed.profile_digest


@pytest.mark.parametrize("protocol", ["openai-chat", "bedrock-converse"])
def test_count_admit_compact_and_generate_use_identical_wire_with_required_sources(protocol):
    from tare_tools_agent_runtime.native_context import ContextPolicy, NativeContext

    class ContextClient(Client):
        def __init__(self):
            super().__init__([])
            self.main_calls = 0

        @staticmethod
        def input_size(wire):
            raw = json.dumps(wire)
            if "Summarize the earlier task history" in raw:
                return 3000
            if "Derived history summary" in raw:
                return 2500
            return 7000 if sum(m["role"] == "assistant" for m in wire["messages"]) >= 2 else 100

        def post(self, path, body, control):
            self.calls.append((path, json.loads(json.dumps(body))))
            if path.endswith("/count-tokens"):
                return {"inputTokens": self.input_size(body["input"]["converse"])}
            if path == "/apply-template":
                return {"prompt": json.dumps(body)}
            if path == "/tokenize":
                return {"tokens": [1] * self.input_size(json.loads(body["content"]))}
            if "Summarize the earlier task history" in json.dumps(body):
                if protocol == "openai-chat":
                    return {"model": "fixture-model", "choices": [{"finish_reason": "stop", "message": {
                        "role": "assistant", "content": "Earlier source a.py inspected; no verified implementation yet."}}],
                        "usage": {"prompt_tokens": 3000, "completion_tokens": 50}}
                return {"stopReason": "end_turn", "output": {"message": {"role": "assistant", "content": [
                    {"text": "Earlier source a.py inspected; no verified implementation yet."}]}},
                    "usage": {"inputTokens": 3000, "outputTokens": 50}}
            self.main_calls += 1
            return response(protocol, "read" if self.main_calls <= 2 else DELIVERY_TOOL)

    class Tools:
        def execute(self, req, control):
            return EffectResult(req.effect_id, "succeeded", '{"source":"a.py","observed":"old"}', True)

    client, rows, log = ContextClient(), [], ConversationLog()
    p = profile(protocol)
    adapter = NativeModelAdapter(p, client, system="immutable system", tool_specs_json=SPECS,
                                 local_counter=client if protocol == "openai-chat" else None, observe=rows.append)
    policy = ContextPolicy(p.context_window, p.output_tokens, 64)
    session = NativeSession(session_id="s", instruction="immutable task", capabilities=("read", "edit", "verify"),
                            model=adapter, tools=Tools(), journal=log, required_context=("SPEC-1: exact protected bytes",),
                            context=NativeContext(policy, counter=adapter.count, summarizer=adapter))
    result = session.run(execution())
    assert result.status == "proposed" and result.model_calls == 4
    generated = [row for row in rows if row["purpose"] in {"model", "summary"}]
    assert [row["purpose"] for row in generated] == ["model", "model", "summary", "model"]
    counted = {row["request_sha256"] for row in rows if row["purpose"] == "count" and row["status"] == "observed"}
    assert all(row["request_sha256"] in counted for row in generated)
    assert next(row for row in generated if row["purpose"] == "summary")["usage"] is not None
    requests = [body for path,body in client.calls if path.endswith("/converse") or path == "/chat/completions"]
    assert len(requests) == 4
    assert all("SPEC-1: exact protected bytes" in json.dumps(body) and "immutable task" in json.dumps(body) for body in requests)
    assert "Derived history summary" in json.dumps(requests[-1])
    assert sum(e.event()["kind"] == "context_activated" for e in log.snapshot().records) == 1
    from compute_plane.native_usage import UsageContract, native_usage_report
    contract = UsageContract("chat-inclusive" if protocol == "openai-chat" else "converse-exclusive", "fixture qualified")
    report = native_usage_report(log, rows + rows, contracts={adapter.profile_digest: contract},
                                 observation_failures=adapter.observation_failures)
    assert len(report["calls"]) == 4
    assert [c["purpose"] for c in report["calls"]] == ["model", "model", "summary", "model"]
    assert report["calls"][0]["effect_ids"] == ["s/effect/1"]
    assert report["tokens"]["output_tokens"] == {"known_sum": 50, "unknown_calls": 3}
    assert report["coverage"]["client_invocations_observed"] == len(client.calls)
    assert report["coverage"]["count_operations_observed"] == 5
    missing = native_usage_report(log, generated, contracts={adapter.profile_digest: contract}, observation_failures=0)
    assert missing["coverage"]["missing_journaled_count_observations"] == 5
    assert missing["coverage"]["count_coverage"] == "unknown"


def test_counted_request_changed_before_dispatch_is_refused_without_network():
    client = Client([])
    adapter = NativeModelAdapter(profile("openai-chat"), client, system="system", tool_specs_json=SPECS)
    with pytest.raises(ModelFailure, match="ADMITTED_REQUEST_CHANGED"):
        adapter.generate(replace(request(), admitted_request_sha256="0" * 64), execution())
    assert not client.calls


def test_bedrock_counter_can_bind_documented_base_model_without_changing_inference_route():
    p = replace(profile('bedrock-converse'), model='us.route', count_model='base-model', count_model_source='qualification receipt')
    client = Client([{'inputTokens': 123}, response('bedrock-converse', DELIVERY_TOOL)])
    adapter = NativeModelAdapter(p, client, system='system', tool_specs_json=SPECS)
    count = adapter.count(request(), execution())
    assert count['count_model'] == 'base-model' and count['input_tokens'] == 123
    assert adapter.generate(request(), execution()).proposes_delivery
    assert [p for p, _ in client.calls] == ['/model/base-model/count-tokens', '/model/us.route/converse']


@pytest.mark.parametrize('protocol,model,source', [
    ('openai-chat', 'base', 'receipt'), ('bedrock-converse', '', 'receipt'),
    ('bedrock-converse', 'base', None), ('bedrock-converse', None, 'receipt')])
def test_counter_route_is_never_inferred_or_silently_aliased(protocol, model, source):
    with pytest.raises(ValueError):
        replace(profile(protocol), count_model=model, count_model_source=source)
