"""Usage accounting is checked against arithmetic and actual native loop events."""
import json

import pytest

from tare_tools_agent_runtime import ConversationLog, ModelFailure, NativeBudget, NativeSession
from compute_plane.native_models import NativeModelAdapter, DELIVERY_TOOL
from compute_plane.native_usage import UsageContract, UsageTariff, normalize_usage, price_usage, native_usage_report, BILLABLE
from tests.test_native_models import Client, SPECS, execution, profile, response


CHAT = UsageContract("chat-inclusive", "fixture: verified input includes cache; no write billing; reasoning in output")
BEDROCK = UsageContract("converse-exclusive", "fixture: verified disjoint input/cache fields")


GEMINI = UsageContract('gemini-chat', 'Google compatibility totals include thinking; visible completions exclude it')


def test_gemini_thinking_is_derived_once_and_missing_cache_stays_unknown():
    raw = {'prompt_tokens': 2466, 'completion_tokens': 43, 'total_tokens': 2959}
    before = json.dumps(raw)
    value = normalize_usage(raw, GEMINI)
    assert value['issues'] == []
    assert value['tokens'] == dict(input_tokens=2466, uncached_input_tokens=None,
        cache_read_tokens=None, cache_write_tokens=0, output_tokens=493, reasoning_tokens=450, total_tokens=2959)
    assert normalize_usage(raw, CHAT)['issues'] == ['total_mismatch']
    assert json.dumps(raw) == before


def test_gemini_explicit_cache_and_inclusive_output_are_priced_without_double_count():
    raw = {'prompt_tokens': 1000, 'completion_tokens': 100, 'total_tokens': 1300,
           'prompt_tokens_details': {'cached_tokens': 800}}
    value = normalize_usage(raw, GEMINI)
    tariff = UsageTariff('USD', 'fixture', 'fixture', json.dumps(dict(zip(BILLABLE, ['1', '.1', '0', '2']))))
    cost = price_usage(value, tariff)
    assert cost['complete'] and cost['known_amount'] == '0.00088'
    assert value['tokens']['output_tokens'] == 300 and value['tokens']['reasoning_tokens'] == 200
    raw.pop('prompt_tokens_details')
    partial = price_usage(normalize_usage(raw, GEMINI), tariff)
    assert not partial['complete'] and 'uncached_input_tokens' in partial['unpriced_categories']


@pytest.mark.parametrize('updates', [
    {'total_tokens': 900}, {'total_tokens': 1050}, {'prompt_tokens': True},
    {'prompt_tokens_details': {'cached_tokens': 1001}},
    {'completion_tokens_details': {'reasoning_tokens': 99}},
])
def test_gemini_does_not_relax_contradictory_or_invalid_counters(updates):
    raw = {'prompt_tokens': 1000, 'completion_tokens': 100, 'total_tokens': 1300, **updates}
    value = normalize_usage(raw, GEMINI)
    assert value['issues'] and all(v is None for v in value['tokens'].values())


def test_gemini_missing_total_does_not_fabricate_thinking_or_complete_output():
    value = normalize_usage({'prompt_tokens': 100, 'completion_tokens': 20}, GEMINI)
    assert value['tokens']['input_tokens'] == 100
    assert value['tokens']['output_tokens'] is value['tokens']['reasoning_tokens'] is None


def test_qualified_cli_contracts_and_historical_read_are_nonmutating():
    from compute_plane.native_usage import deployment_contract, enrich_provider_costs
    for vendor, version, layout in [('codex', '0.154.0', 'codex-cli-inclusive'),
                                    ('agy', '1.2.7', 'agy-cli-exclusive')]:
        deployment = {'native_profile': {'protocol': 'vendor-cli', 'endpoint': 'cli://'+vendor},
                      'native_cli': {'version': version}}
        assert deployment_contract(deployment)['layout'] == layout
        deployment['native_cli']['version'] = 'future'
        assert deployment_contract(deployment) is None
    deployment = {'native_profile': {'protocol': 'vendor-cli', 'endpoint': 'cli://codex'},
                  'native_cli': {'version': '0.154.0'}}
    report = {'calls': [{'raw_usage': {'input_tokens': 100, 'cached_input_tokens': 80, 'output_tokens': 20},
                        'normalized': {'contract': None}, 'cost': {'complete': False}}]}
    before = json.dumps(report)
    result = enrich_provider_costs(report, deployment)
    assert result['tokens']['uncached_input_tokens'] == {'known_sum': 20, 'unknown_calls': 0}
    assert result['unpriced_or_partial_calls'] == 1
    assert json.dumps(report) == before


def test_codex_cache_writes_and_reasoning_do_not_double_count():
    contract = UsageContract('codex-cli-inclusive', 'qualified CLI')
    raw = {'input_tokens': 100, 'cached_input_tokens': 70, 'cache_write_input_tokens': 10,
           'output_tokens': 20, 'reasoning_output_tokens': 12}
    tokens = normalize_usage(raw, contract)['tokens']
    assert tokens == dict(input_tokens=100, uncached_input_tokens=20, cache_read_tokens=70,
                          cache_write_tokens=10, output_tokens=20, reasoning_tokens=12, total_tokens=120)
    raw['cached_input_tokens'] = 100
    assert normalize_usage(raw, contract)['issues'] == ['cache_exceeds_input']


def test_agy_cumulative_terminal_counts_cache_separately_and_thinking_once():
    raw = {'input_tokens': 30662, 'output_tokens': 8, 'cache_read_tokens': 30214,
           'thinking_tokens': 2, 'total_tokens': 30670}
    tokens = normalize_usage(raw, UsageContract('agy-cli-exclusive', 'official two-turn example'))['tokens']
    assert tokens['input_tokens'] == 60876
    assert tokens['uncached_input_tokens'] == 30662
    assert tokens['total_tokens'] == 60884 and tokens['output_tokens'] == 8


def test_responses_explicit_cache_write_category_is_retained():
    tokens = normalize_usage({'input_tokens': 100, 'output_tokens': 5,
        'input_tokens_details': {'cached_tokens': 60, 'cache_write_tokens': 20}},
        UsageContract('responses-inclusive', 'Responses schema'))['tokens']
    assert tokens['uncached_input_tokens'] == 20 and tokens['cache_write_tokens'] == 20


def chat_usage(**updates):
    return {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 12}, **updates}


def tariff(currency="USD", **overrides):
    return UsageTariff(currency, "fixture tariff, not market prices", "2026-09-10",
                       json.dumps({**dict(zip(BILLABLE, ("2", "0.2", "2.5", "8"))), **overrides}))


def test_inclusive_cache_and_reasoning_are_not_double_counted():
    value = normalize_usage(chat_usage(), CHAT)
    assert value["tokens"] == dict(input_tokens=100, uncached_input_tokens=20, cache_read_tokens=80,
                                  cache_write_tokens=0, output_tokens=20, reasoning_tokens=12, total_tokens=120)
    cost = price_usage(value, tariff())
    assert cost["known_amount"] == "0.000216" and cost["complete"]


@pytest.mark.parametrize('saved_only', [False, True])
def test_openrouter_actual_cost_overrides_tariff_without_double_counting_upstream(saved_only):
    raw = chat_usage(cost=.017, cost_details={'upstream_inference_cost': .01})
    adapter, log, rows, _ = run_session([response('openai-chat', DELIVERY_TOOL, usage=raw)])
    result = native_usage_report(log, [] if saved_only else rows,
        contracts={adapter.profile_digest: UsageContract('openrouter-inclusive', 'provider docs')},
        tariffs={adapter.profile_digest: tariff()})
    assert result['known_cost_by_currency'] == {'USD': '0.017'}
    assert result['unpriced_or_partial_calls'] == 0
    assert result['calls'][0]['cost']['basis'] == 'provider_reported'
    # Missing token detail does not erase an independently supplied charge.
    assert result['calls'][0]['normalized']['tokens']['cache_write_tokens'] is None


def test_old_raw_cost_can_be_read_without_mutating_historical_receipts():
    from compute_plane.native_usage import enrich_provider_costs
    adapter, log, rows, _ = run_session([response('openai-chat', DELIVERY_TOOL, usage=chat_usage(cost=.03))])
    result = report(adapter, log, rows)
    call = result['calls'][0]
    call['normalized']['contract']['layout'] = 'openrouter-inclusive'
    before = json.dumps(result)
    updated = enrich_provider_costs(result)
    assert updated['known_cost_by_currency'] == {'USD': '0.03'}
    assert json.dumps(result) == before
    call['observation']['model_identity'] = 'mismatch'
    assert enrich_provider_costs(result)['calls'][0]['cost'] == call['cost']


@pytest.mark.parametrize('value', [True, -1, 'NaN', 'Infinity', {}, None])
def test_reported_cost_requires_valid_amount_and_explicit_contract(value):
    from compute_plane.native_usage import reported_cost
    contract = UsageContract('openrouter-inclusive', 'provider docs')
    assert reported_cost({'cost': value}, contract) is None
    assert reported_cost({'cost': .1}, CHAT) is None
    assert reported_cost({'cost': 0}, contract)['known_amount'] == '0'


def test_claude_client_estimate_is_separate_from_actual_charge():
    from compute_plane.native_usage import enrich_provider_costs
    result = enrich_provider_costs({'calls': [{'execution': 'possibly_executed',
        'normalized': {'contract': {'layout': 'claude-cli-exclusive', 'source': 'client'}},
        'raw_usage': {'total_cost_usd': .123}, 'cost': {'complete': False}}]})
    assert result['provider_estimates_by_currency'] == {'USD': '0.123'}
    assert result['known_cost_by_currency'] == {} and result['unpriced_or_partial_calls'] == 1


def test_exclusive_cold_and_warm_have_equal_occupancy_and_distinct_charges():
    cold = normalize_usage({"inputTokens": 20, "outputTokens": 20,
                           "cacheReadInputTokens": 0, "cacheWriteInputTokens": 80}, BEDROCK)
    warm = normalize_usage({"inputTokens": 20, "outputTokens": 20,
                           "cacheReadInputTokens": 80, "cacheWriteInputTokens": 0}, BEDROCK)
    assert cold["tokens"]["total_tokens"] == warm["tokens"]["total_tokens"] == 120
    assert price_usage(cold, tariff())["known_amount"] == "0.00040"
    assert price_usage(warm, tariff())["known_amount"] == "0.000216"
    assert warm["tokens"]["reasoning_tokens"] is None


def test_missing_cache_or_price_remains_partial_without_erasing_known_counters():
    value = normalize_usage({"inputTokens": 20, "outputTokens": 5, "cacheReadInputTokens": 80}, BEDROCK)
    assert value["tokens"]["input_tokens"] is None
    assert value["tokens"]["cache_write_tokens"] is None
    cost = price_usage(value, tariff())
    assert not cost["complete"] and cost["unpriced_categories"] == ["cache_write_tokens"]
    assert cost["known_amount"] == "0.000096"
    assert price_usage(value, None)["known_amount"] is None
    cost = price_usage(normalize_usage(chat_usage(), CHAT), tariff(cache_read_tokens=None))
    assert not cost["complete"] and cost["known_amount"] == "0.00020"


@pytest.mark.parametrize("updates", [
    {"prompt_tokens": True}, {"prompt_tokens": -1}, {"completion_tokens": "20"},
    {"prompt_tokens_details": {"cached_tokens": 101}},
    {"completion_tokens_details": {"reasoning_tokens": 21}}, {"total_tokens": 999},
])
def test_malformed_or_contradictory_usage_never_contributes_exact_totals(updates):
    value = normalize_usage(chat_usage(**updates), CHAT)
    assert value["issues"] and all(v is None for v in value["tokens"].values())
    assert not price_usage(value, tariff())["complete"]


def test_compatible_http_does_not_imply_a_usage_contract():
    value = normalize_usage(chat_usage(), None)
    assert value["issues"] == ["usage_contract_missing"]
    assert value["tokens"]["input_tokens"] is None
    with pytest.raises(ValueError):
        UsageContract("cumulative", "unsupported stream")


CLAUDE = UsageContract("claude-cli-exclusive", "fixture: Claude CLI flat input categories are disjoint")


def claude_usage(**updates):
    return {"input_tokens": 2, "cache_creation_input_tokens": 2549,
            "cache_read_input_tokens": 0, "output_tokens": 78, **updates}


def test_claude_cli_exclusive_sums_disjoint_input_categories():
    value = normalize_usage(claude_usage(), CLAUDE)
    assert value["issues"] == []
    assert value["tokens"] == dict(input_tokens=2551, uncached_input_tokens=2, cache_read_tokens=0,
                                  cache_write_tokens=2549, output_tokens=78, reasoning_tokens=None,
                                  total_tokens=2629)
    assert value["contract"]["layout"] == "claude-cli-exclusive"


def test_claude_cli_missing_counter_stays_unknown_without_erasing_known_counters():
    raw = claude_usage()
    del raw["cache_read_input_tokens"]
    value = normalize_usage(raw, CLAUDE)
    assert value["issues"] == []
    tokens = value["tokens"]
    assert tokens["cache_read_tokens"] is None and tokens["input_tokens"] is None
    assert tokens["total_tokens"] is None and tokens["reasoning_tokens"] is None
    assert tokens["uncached_input_tokens"] == 2 and tokens["cache_write_tokens"] == 2549
    assert tokens["output_tokens"] == 78


@pytest.mark.parametrize("updates", [
    {"input_tokens": True}, {"cache_creation_input_tokens": -1},
    {"cache_read_input_tokens": "0"}, {"output_tokens": 78.0},
])
def test_claude_cli_invalid_counters_invalidate_all_tokens(updates):
    value = normalize_usage(claude_usage(**updates), CLAUDE)
    assert value["issues"] and all(v is None for v in value["tokens"].values())


def test_claude_cli_ignores_overlapping_nested_cache_creation_and_iterations():
    raw = claude_usage(
        cache_creation={"ephemeral_5m_input_tokens": 2549, "ephemeral_1h_input_tokens": -7},
        iterations=[{"input_tokens": 2, "cache_creation_input_tokens": 2549, "output_tokens": 78}],
        total_tokens=1)
    value = normalize_usage(raw, CLAUDE)
    assert value["issues"] == []
    assert value["tokens"]["input_tokens"] == 2551 and value["tokens"]["total_tokens"] == 2629


def test_explicit_zero_tariff_costs_zero_without_inventing_usage_counters():
    value=normalize_usage(None,BEDROCK)
    cost=price_usage(value,tariff(**{k:'0' for k in BILLABLE}))
    assert cost['complete'] and cost['known_amount']=='0'
    assert all(v is None for v in value['tokens'].values())


def test_host_denial_is_not_counted_as_an_unknown_provider_charge():
    log=ConversationLog()
    class Denied:
        def generate(self, request, control):raise ModelFailure('SPEND_LIMIT',not_started=True)
    result=NativeSession(session_id='s',instruction='task',capabilities=(),model=Denied(),tools=None,journal=log).run(execution())
    assert result.reason=='SPEND_LIMIT'
    usage=native_usage_report(log,[],contracts={},observation_failures=0)
    assert usage['calls'][0]['execution']=='not_started'
    assert usage['unpriced_or_partial_calls']==0 and usage['coverage']['missing_generation_observations']==0
    assert usage['tokens']['input_tokens']=={'known_sum':0,'unknown_calls':0}


@pytest.mark.parametrize("rate", ["NaN", "Infinity", "-1", 1, True])
def test_tariffs_reject_nonfinite_negative_and_implicit_numeric_values(rate):
    with pytest.raises(ValueError):
        tariff(output_tokens=rate)


def run_session(responses, *, sink=None):
    rows, log = [], ConversationLog()
    adapter = NativeModelAdapter(profile("openai-chat"), Client(responses), system="system",
                                 tool_specs_json=SPECS, observe=rows.append if sink is None else sink)
    session = NativeSession(session_id="s", instruction="task", capabilities=("read", "edit", "verify"),
                            model=adapter, tools=None, journal=log,
                            budget=NativeBudget(max_model_retries=1))
    outcome = session.run(execution())
    return adapter, log, rows, outcome


def report(adapter, log, rows, **kwargs):
    return native_usage_report(log, rows, contracts={adapter.profile_digest: CHAT},
                               tariffs={adapter.profile_digest: tariff()},
                               observation_failures=adapter.observation_failures, **kwargs)


@pytest.mark.parametrize("saved_reply_only", [False, True])
def test_claude_cli_report_accounts_for_usage_and_recovers_without_another_call(saved_reply_only):
    from compute_plane.native_models import ModelProfile
    from compute_plane.native_cli import proposal

    class ClaudeClient:
        endpoint = "cli://claude"
        calls = 0

        def post_observed(self, path, body, control):
            self.calls += 1
            reply = json.dumps({"text": "Done", "tool_calls": [{"id": "delivery",
                "name": DELIVERY_TOOL, "arguments": {"summary": "Done"}}]})
            return proposal(reply, "fixture", claude_usage()), {"cleanup_verified": True}

    client, log, rows = ClaudeClient(), ConversationLog(), []
    adapter = NativeModelAdapter(
        ModelProfile("vendor-cli", "fixture", 32768, 4096, "fixture", client.endpoint),
        client, system="system", tool_specs_json=SPECS, observe=rows.append,
        counter_protocol="utf8-envelope-estimate-v1")
    outcome = NativeSession(session_id="s", instruction="task", capabilities=("read", "edit", "verify"),
                            model=adapter, tools=None, journal=log).run(execution())
    assert outcome.status == "proposed", (outcome, rows)
    result = native_usage_report(log, [] if saved_reply_only else rows,
                                 contracts={adapter.profile_digest: CLAUDE}, observation_failures=0)
    assert result["tokens"]["total_tokens"] == {"known_sum": 2629, "unknown_calls": 0}
    assert result["tokens"]["cache_write_tokens"] == {"known_sum": 2549, "unknown_calls": 0}
    assert result["known_cost_by_currency"] == {} and result["unpriced_or_partial_calls"] == 1
    assert result["calls"][0]["usage_source"] == ("saved_reply" if saved_reply_only else "observation")
    assert client.calls == 1
    with pytest.raises(ValueError, match="usage contract protocol mismatch"):
        native_usage_report(log, rows, contracts={adapter.profile_digest: CHAT})


def test_retry_stays_a_separate_unknown_charge_and_preserves_work_binding():
    adapter, log, rows, outcome = run_session([
        ModelFailure("HTTP_429", retryable=True), response("openai-chat", DELIVERY_TOOL, usage=chat_usage())])
    assert outcome.status == "proposed"
    result = report(adapter, log, rows)
    assert result["binding"]["work_id"] == "work"
    assert result["calls"][1]["retry_of"] == "s/model/1"
    assert result["tokens"]["total_tokens"] == {"known_sum": 120, "unknown_calls": 1}
    assert result["unpriced_or_partial_calls"] == 1
    assert result["coverage"]["client_invocations_observed"] == 2
    assert result["timing"]["preparation_and_task_wall_ms"] is None
    assert result["calls"][0]["raw_usage"] is None


def test_observer_failure_recovers_saved_usage_but_exposes_coverage_loss():
    def broken(row):
        raise OSError("sink unavailable")
    adapter, log, rows, _ = run_session([response("openai-chat", DELIVERY_TOOL, usage=chat_usage())], sink=broken)
    result = report(adapter, log, rows)
    assert result["calls"][0]["usage_source"] == "saved_reply"
    assert result["tokens"]["total_tokens"]["known_sum"] == 120
    assert result["coverage"]["missing_generation_observations"] == 1
    assert result["coverage"]["observation_failures"] == 1
    assert result["coverage"]["count_coverage"] == "unknown"


def test_uncertain_dispatch_has_unknown_usage_and_is_not_retried():
    adapter, log, rows, outcome = run_session([TimeoutError()])
    assert outcome.status == "reconciliation_required"
    result = report(adapter, log, rows)
    assert result["tokens"]["output_tokens"] == {"known_sum": 0, "unknown_calls": 1}
    assert result["known_cost_by_currency"] == {"USD": "0"}
    assert result["unpriced_or_partial_calls"] == 1  # Known subtotal zero is NOT a zero cost claim.


def test_duplicate_replay_is_idempotent_conflict_and_foreign_session_rejected():
    adapter, log, rows, _ = run_session([response("openai-chat", DELIVERY_TOOL, usage=chat_usage())])
    assert report(adapter, log, rows + rows) == report(adapter, log, rows)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        report(adapter, log, rows + [{**rows[0], "usage": None}])
    with pytest.raises(ValueError, match="session mismatch"):
        report(adapter, log, [{**rows[0], "session_id": "other"}])
    with pytest.raises(ValueError, match="conflicts with saved reply"):
        report(adapter, log, [{**rows[0], "usage": None}])


def test_invalid_provider_response_still_accounts_for_returned_usage():
    invalid = response("openai-chat", DELIVERY_TOOL, usage=chat_usage())
    invalid["model"] = "wrong-model"
    adapter, log, rows, outcome = run_session([invalid])
    result = report(adapter, log, rows)
    assert result["calls"][0]["observation"]["status"] == "invalid_response"
    # Profile-priced usage is not safe after an unexpected response model.
    assert result["calls"][0]["raw_usage"] == chat_usage()
    assert result["calls"][0]["cost"]["known_amount"] is None
    assert result["unpriced_or_partial_calls"] == 1


def test_not_dispatched_intent_does_not_invent_usage_or_unknown_remote_charge():
    adapter, log, rows, _ = run_session([response("openai-chat", DELIVERY_TOOL, usage=chat_usage())])
    stopped = ConversationLog()
    stopped.append(log.snapshot().records[0].event())
    stopped.append({"kind": "model_requested", "call_id": "s/model/1"})
    stopped.append({"kind": "invocation_not_started", "invocation_id": "s/model/1"})
    result = report(adapter, stopped, [])
    assert result["coverage"]["not_started"] == 1
    assert result["tokens"]["total_tokens"] == {"known_sum": 0, "unknown_calls": 0}
    assert result["calls"][0]["raw_usage"] is None
    with pytest.raises(ValueError, match="not-started intent"):
        report(adapter, stopped, rows)


OPENCODE = UsageContract('opencode-exclusive', 'https://opencode.ai/docs/console/api/inference')
# The documented usage example, billed to the Go plan within its limits.
OPENCODE_USAGE = {'prompt_tokens': 1200, 'completion_tokens': 300, 'cost': 0,
                  'opencode': {'billing_source': 'go', 'list_cost': 0.00412, 'allowance': 0.00412,
                               'tokens': {'input': 1200, 'output': 300, 'reasoning': 120, 'cache_read': 8000,
                                          'cache_write_5m': 0, 'cache_write_1h': 0}}}


def test_opencode_reads_its_disjoint_counters_not_the_family_counters():
    value = normalize_usage(OPENCODE_USAGE, OPENCODE)
    assert value['issues'] == []
    assert value['tokens'] == dict(input_tokens=9200, uncached_input_tokens=1200, cache_read_tokens=8000,
                                   cache_write_tokens=0, output_tokens=300, reasoning_tokens=120, total_tokens=9500)
    family_only = {'prompt_tokens': 1200, 'completion_tokens': 300}
    assert normalize_usage(family_only, OPENCODE)['tokens'] == dict.fromkeys(value['tokens'])  # unknown, never zero


def test_opencode_charge_keeps_billing_source_list_cost_and_allowance():
    from compute_plane.native_usage import reported_cost
    cost = reported_cost(OPENCODE_USAGE, OPENCODE)
    assert cost['known_amount'] == '0' and cost['basis'] == 'provider_reported' and cost['source'] == 'opencode.usage.cost'
    assert cost['opencode'] == {'billing_source': 'go', 'list_cost': '0.00412', 'allowance': '0.00412'}
    odd = {**OPENCODE_USAGE, 'opencode': {**OPENCODE_USAGE['opencode'], 'billing_source': 'gift', 'allowance': -1}}
    assert reported_cost(odd, OPENCODE)['opencode'] == {'billing_source': None, 'list_cost': '0.00412', 'allowance': None}
    assert reported_cost({**OPENCODE_USAGE, 'cost': 'NaN'}, OPENCODE) is None
    byok = {**OPENCODE_USAGE, 'cost': 0.0041, 'opencode': {**OPENCODE_USAGE['opencode'], 'billing_source': 'byok', 'estimated': True}}
    assert reported_cost(byok, OPENCODE) is None


def test_opencode_report_accepts_its_families_and_rejects_others():
    adapter, log, rows, _ = run_session([response('openai-chat', DELIVERY_TOOL, usage=OPENCODE_USAGE)])
    result = native_usage_report(log, rows, contracts={adapter.profile_digest: OPENCODE})
    assert result['known_cost_by_currency'] == {'USD': '0'}
    assert result['calls'][0]['cost']['opencode']['billing_source'] == 'go'
    rows[0]['protocol'] = 'gemini-generate-content'  # usageMetadata carries the same cost and opencode block
    assert native_usage_report(log, rows, contracts={adapter.profile_digest: OPENCODE})['known_cost_by_currency'] == {'USD': '0'}
    rows[0]['protocol'] = 'bedrock-converse'
    with pytest.raises(ValueError, match='protocol mismatch'):
        native_usage_report(log, rows, contracts={adapter.profile_digest: OPENCODE})
