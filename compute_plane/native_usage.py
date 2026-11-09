"""Derived native usage report over existing journals and adapter observations.

No storage, prices, provider discovery or tactical decisions. Contracts and tariffs
are explicit per-profile facts; HTTP compatibility alone establishes neither.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from decimal import Decimal, InvalidOperation
import json

from .native_models import encode, object_json

# Derived reports retain per-call raw evidence and may exceed either source
# journal. File readers and writers share this finite, report-specific budget.
MAX_USAGE_BYTES = 64 * 1024 * 1024

FIELDS = ("input_tokens", "uncached_input_tokens", "cache_read_tokens",
          "cache_write_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
BILLABLE = ("uncached_input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")


@dataclass(frozen=True)
class UsageContract:
    layout: str  # Per-request provider counters, never cumulative session usage.
    source: str

    def __post_init__(self):
        if self.layout not in {"chat-inclusive", "gemini-chat", "openrouter-inclusive", "openrouter-responses-inclusive", "responses-inclusive", "converse-exclusive",
                               "anthropic-exclusive", "opencode-exclusive", "claude-cli-exclusive", "codex-cli-inclusive", "agy-cli-exclusive"} or not self.source.strip():
            raise ValueError("explicit per-request usage contract and source required")


@dataclass(frozen=True)
class UsageTariff:
    currency: str
    source: str
    effective_at: str
    # Decimal strings per million tokens; null is an unpriced category.
    rates_json: str

    def __post_init__(self):
        if not all(isinstance(v, str) and v.strip() for v in (self.currency, self.source, self.effective_at)):
            raise ValueError("tariff currency, source and effective date required")
        rates = object_json(self.rates_json)
        if set(rates) != set(BILLABLE):
            raise ValueError("explicit rate or null required for every category")
        for value in rates.values():
            try:
                amount = Decimal(value) if isinstance(value, str) else None
            except InvalidOperation:
                amount = None
            if value is not None and (amount is None or not amount.is_finite() or amount < 0):
                raise ValueError("rates must be finite nonnegative decimal strings")
        object.__setattr__(self, "rates_json", encode(rates))


def deployment_contract(deployment):
    """Qualified wire defaults; preserve the recorded deployment identity."""
    if deployment.get('native_usage_contract'):
        return deployment['native_usage_contract']
    profile, cli = deployment.get('native_profile', {}), deployment.get('native_cli', {})
    if profile.get('protocol') in {'openai-codex-responses', 'openai-responses'}:
        return {'layout': 'responses-inclusive', 'source': profile['protocol'] + ' per-request usage'}
    if profile.get('protocol') == 'anthropic-messages':
        return {'layout': 'anthropic-exclusive', 'source': 'https://platform.claude.com/docs/en/api/messages'}
    qualified = {
        ('cli://codex', '0.154.0'): ('codex-cli-inclusive', 'https://developers.openai.com/codex/noninteractive'),
        ('cli://agy', '1.2.7'): ('agy-cli-exclusive', 'https://antigravity.google/docs/cli/headless/'),
        ('cli://claude', '2.1.280'): ('claude-cli-exclusive', 'Claude Code result usage'),
    }
    item = qualified.get((profile.get('endpoint'), cli.get('version')))
    if item is None and profile.get('endpoint') == 'cli://codex':
        from .native_cli import version_tuple
        try:
            if version_tuple(cli.get('version')) >= (0, 154, 0):
                item = qualified[('cli://codex', '0.154.0')]
        except ValueError:
            pass
    if profile.get('protocol') == 'vendor-cli' and item:
        return dict(zip(('layout', 'source'), item))
    return None


def normalize_usage(raw, contract: UsageContract | None):
    values = dict.fromkeys(FIELDS)
    problems = []
    result = {"tokens": values, "issues": problems,
              "contract": asdict(contract) if contract else None}
    if contract is None:
        problems.append("usage_contract_missing")
        return result
    if not isinstance(raw, dict):
        problems.append("usage_missing_or_invalid")
        return result

    def read(*path):
        value = raw
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        if type(value) is not int or value < 0:
            problems.append("invalid_counter:" + ".".join(path))
            return None
        return value

    if contract.layout in {"chat-inclusive", "openrouter-inclusive", "openrouter-responses-inclusive", "responses-inclusive"}:
        input_key, output_key = ('input_tokens', 'output_tokens') if contract.layout in {'responses-inclusive', 'openrouter-responses-inclusive'} else (
            'prompt_tokens', 'completion_tokens')
        inp, out = read(input_key), read(output_key)
        cached = read(input_key + '_details', "cached_tokens")
        # OpenRouter reports disjoint reads/writes within prompt_tokens. Other
        # qualified inclusive layouts have no separate cache creation category.
        write = (read(input_key + '_details', 'cache_write_tokens') if contract.layout in {'openrouter-inclusive', 'openrouter-responses-inclusive'}
                 or 'cache_write_tokens' in (raw.get(input_key + '_details') or {}) else 0)
        reasoning = read(output_key + '_details', "reasoning_tokens")
        uncached = inp - cached - write if None not in (inp, cached, write) else None
        provider_total = read("total_tokens")
    elif contract.layout == 'gemini-chat':
        # Gemini's OpenAI-compatible completion_tokens exclude thinking; total
        # includes it. This is an explicit vendor contract, not a correction
        # inferred for arbitrary OpenAI-compatible servers. Keep raw evidence.
        inp, visible, provider_total = read('prompt_tokens'), read('completion_tokens'), read('total_tokens')
        cached, write = read('prompt_tokens_details', 'cached_tokens'), 0
        uncached = inp-cached if None not in (inp, cached) else None
        out = provider_total-inp if None not in (provider_total, inp) else None
        reasoning = out-visible if None not in (out, visible) else None
        declared = read('completion_tokens_details', 'reasoning_tokens')
        if out is not None and (out < 0 or visible is not None and out < visible):
            problems.append('total_mismatch')
        if declared is not None and reasoning is not None and declared != reasoning:
            problems.append('reasoning_mismatch')
    elif contract.layout == 'opencode-exclusive':
        # OpenCode reports disjoint categories under usage.opencode.tokens for every API family: input excludes
        # cache reads and writes, output includes reasoning. The family's own counters stay raw evidence only.
        uncached, out = read('opencode', 'tokens', 'input'), read('opencode', 'tokens', 'output')
        cached = read('opencode', 'tokens', 'cache_read')
        short, hour = read('opencode', 'tokens', 'cache_write_5m'), read('opencode', 'tokens', 'cache_write_1h')
        write = short + hour if None not in (short, hour) else None
        inp = uncached + cached + write if None not in (uncached, cached, write) else None
        reasoning, provider_total = read('opencode', 'tokens', 'reasoning'), None
    elif contract.layout == 'codex-cli-inclusive':
        inp, out = read('input_tokens'), read('output_tokens')
        cached = read('cached_input_tokens')
        write = read('cache_write_input_tokens') if 'cache_write_input_tokens' in raw else 0
        uncached = inp - cached - write if None not in (inp, cached, write) else None
        reasoning, provider_total = read('reasoning_output_tokens'), None
    elif contract.layout == 'agy-cli-exclusive':
        # AGY's second-turn example reports 278 input + 30214 cache reads.
        # Our qualified transport creates one fresh session per invocation.
        uncached, out = read('input_tokens'), read('output_tokens')
        cached, write = read('cache_read_tokens'), 0
        inp = uncached + cached if None not in (uncached, cached) else None
        reasoning, provider_total = read('thinking_tokens'), None
    elif contract.layout in {"claude-cli-exclusive", "anthropic-exclusive"}:
        # Claude CLI reports disjoint flat input categories. Nested cache_creation
        # and iterations overlap these counters and are deliberately ignored.
        uncached, out = read("input_tokens"), read("output_tokens")
        cached, write = read("cache_read_input_tokens"), read("cache_creation_input_tokens")
        inp = uncached + cached + write if None not in (uncached, cached, write) else None
        reasoning = None  # No portable reasoning usage counter is reported.
        provider_total = None
    else:
        uncached, out = read("inputTokens"), read("outputTokens")
        cached, write = read("cacheReadInputTokens"), read("cacheWriteInputTokens")
        inp = uncached + cached + write if None not in (uncached, cached, write) else None
        reasoning = None  # Converse has no portable reasoning usage field.
        provider_total = read("totalTokens")
    values.update(input_tokens=inp, uncached_input_tokens=uncached,
                  cache_read_tokens=cached, cache_write_tokens=write,
                  output_tokens=out, reasoning_tokens=reasoning)
    if uncached is not None and uncached < 0:
        problems.append("cache_exceeds_input")
    if reasoning is not None and out is not None and reasoning > out:
        problems.append("reasoning_exceeds_output")
    # Check reported totals only in the inclusive contract. Converse totalTokens
    # is retained raw; its cache inclusion is not inferred from the generic name.
    if inp is not None and out is not None:
        values["total_tokens"] = inp + out
        if contract.layout in {"chat-inclusive", "openrouter-inclusive", "openrouter-responses-inclusive", "responses-inclusive"} and provider_total is not None and provider_total != inp + out:
            problems.append("total_mismatch")
    if problems:
        # Contradictory provider counters must not contribute apparently exact totals.
        values.update(dict.fromkeys(FIELDS))
    return result


def price_usage(normalized, tariff: UsageTariff | None):
    if tariff is None:
        return {"currency": None, "known_amount": None, "complete": False,
                "unpriced_categories": list(BILLABLE), "tariff": None}
    rates, amount, missing = json.loads(tariff.rates_json), Decimal(0), []
    for field in BILLABLE:
        tokens, rate = normalized["tokens"][field], rates[field]
        if tokens == 0 or rate is not None and Decimal(rate) == 0:
            continue
        if tokens is None or rate is None:
            missing.append(field)
        else:
            amount += Decimal(tokens) * Decimal(rate) / Decimal(1000000)
    return {"currency": tariff.currency, "known_amount": str(amount), "complete": not missing,
            "unpriced_categories": missing, "tariff": asdict(tariff)}


def reported_cost(raw, contract: UsageContract | None):
    """Use only an explicit provider contract, never a generic field named cost."""
    if (not contract or contract.layout not in {'openrouter-inclusive', 'openrouter-responses-inclusive', 'opencode-exclusive'}
            or not isinstance(raw, dict)):
        return None
    amount = _amount(raw.get('cost'))
    if amount is None:
        return None
    if contract.layout == 'opencode-exclusive':
        gateway = raw.get('opencode') if isinstance(raw.get('opencode'), dict) else {}
        if gateway.get('estimated') is True:
            return None  # with the customer's own provider key, cost estimates that provider's bill: not a charge
        source = gateway.get('billing_source')
        known = {'list_cost': _amount(gateway.get('list_cost')), 'allowance': _amount(gateway.get('allowance'))}
        return {'currency': 'USD', 'known_amount': str(amount), 'complete': True,
                'unpriced_categories': [], 'tariff': None, 'basis': 'provider_reported',
                'source': 'opencode.usage.cost',
                'scope': 'OpenCode balance charge: 0 for Go use within its limits; list_cost and allowance inform only.',
                'opencode': {'billing_source': source if source in {'free', 'balance', 'go', 'go-plus', 'byok'} else None,
                             **{k: None if v is None else str(v) for k, v in known.items()}}}
    return {'currency': 'USD', 'known_amount': str(amount), 'complete': True,
            'unpriced_categories': [], 'tariff': None, 'basis': 'provider_reported',
            'source': 'openrouter.usage.cost', 'scope': 'OpenRouter account charge; external BYOK billing is separate.'}


def _amount(value):
    """A finite non-negative decimal from a provider's JSON number or string, else None."""
    if type(value) not in (int, float, str):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def cli_cost_estimate(raw, contract: UsageContract | None):
    """Claude's bundled price-table estimate is not a subscription/API invoice."""
    if not contract or contract.layout != 'claude-cli-exclusive' or not isinstance(raw, dict):
        return None
    value = raw.get('total_cost_usd')
    if type(value) not in (int, float, str):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return {'currency': 'USD', 'amount': str(amount), 'basis': 'client_estimate',
            'source': 'claude.result.total_cost_usd'}


def native_usage_report(journal, observations, *, contracts, tariffs=None,
                        observation_failures=None):
    """Join one native session snapshot with its complete selected observation set.

    Identical generation observations deduplicate by session/call. Conflicting
    duplicates fail visibly. Counts are distinct operations, never token usage.
    Callers own retention; observations can be replayed from their existing sink.
    """
    if observation_failures is not None and (type(observation_failures) is not int or observation_failures < 0):
        raise ValueError("invalid observation failure count")
    events = [r.event() for r in journal.snapshot().records]
    if not events or events[0].get("kind") != "session_started":
        raise ValueError("native session journal required")
    session = events[0]["session_id"]
    intents, returned, not_started, retry = {}, {}, set(), None
    for event in events[1:]:
        kind = event["kind"]
        if kind in {"model_requested", "context_summary_requested"}:
            call = event["call_id"]
            summary = kind == "context_summary_requested"
            intents[call] = {"call_id": call, "purpose": "summary" if summary else "model",
                             "retry_of": None if summary else retry, "effect_ids": []}
            if not summary:
                retry = None
        elif kind in {"model_returned", "context_summary_returned"}:
            returned[event["call_id"]] = object_json(event["reply"]["metadata_json"])
        elif kind == "model_failed" and event.get("retryable"):
            retry = event["call_id"]
        elif kind == "invocation_not_started":
            not_started.add(event["invocation_id"])
        elif kind == "context_summary_failed" and event.get("reason") == "NOT_STARTED":
            not_started.add(event["call_id"])
        elif kind == "effect_requested":
            prefix = session + '/effect/'
            if not event['effect_id'].startswith(prefix):
                raise ValueError('effect session mismatch')
            model_call = session + "/model/" + event["effect_id"][len(prefix):].split('/', 1)[0]
            if model_call in intents:
                intents[model_call]["effect_ids"].append(event["effect_id"])
        if kind in {'model_failed', 'context_summary_failed'} and event.get('not_started') is True:
            not_started.add(event['call_id'])

    indexed, counts, seen = {}, [], {}
    for original in observations:
        row = object_json(encode(original))  # Detached, bounded, strict JSON.
        if row.get("session_id") != session:
            raise ValueError("observation session mismatch")
        identity = row.get("observation_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("observation identity required")
        if identity in seen:
            if seen[identity] != row:
                raise ValueError("conflicting duplicate observation")
            continue
        seen[identity] = row
        if row.get("purpose") == "count":
            counts.append(row)
            continue
        call = row.get("call_id")
        if call not in intents or row.get("purpose") != intents[call]["purpose"]:
            raise ValueError("observation has no matching native intent")
        if call in indexed and indexed[call] != row:
            raise ValueError("conflicting duplicate observation")
        indexed[call] = row

    calls = []
    for call, intent in intents.items():
        row, receipt = indexed.get(call), returned.get(call, {})
        transport = (row or {}).get('transport_calls', [])
        from .native_http_worker import CONNECT_REASONS
        undispatched = transport and all(t.get('cleanup_verified') is True and (
            t.get('started') is False or (
                t.get('transport') == 'isolated_stdlib_http'
                and t.get('request_transmission') == 'not_started'
                and t.get('connection_failure_reason') in CONNECT_REASONS
                and t.get('remote_stop') == 'not_dispatched'
                and t.get('http_attempts_upper_bound') == 0
                and t.get('exit_code') == 0
                and t.get('cancelled') is False and t.get('timeout') is False
                and not t.get('output_limit') and not t.get('observer_failures')
                and not t.get('response_received') and not t.get('stream_events')
                and t.get('stream_usage') is None
            )) for t in transport)
        if call in not_started and (receipt or row is not None and not undispatched):
            raise ValueError("not-started intent has execution evidence")
        if undispatched:
            if receipt or row.get('usage') is not None:
                raise ValueError('undispatched transport has response evidence')
            not_started.add(call)
        observed_usage = (row or {}).get('usage')
        if (receipt.get('usage_detail_omitted') == ['attribution']
                and receipt.get('usage_semantics') == 'counters_' + str(receipt.get('protocol'))
                and isinstance(observed_usage, dict)):
            observed_usage = {k: v for k, v in observed_usage.items() if k != 'attribution'}
        if row is not None and receipt and (observed_usage != receipt.get("usage")
                                            or row.get("profile_digest") != receipt.get("profile_digest")):
            raise ValueError("usage observation conflicts with saved reply")
        evidence = row if row is not None else receipt
        profile = evidence.get("profile_digest")
        contract = contracts.get(profile)
        protocol = evidence.get("protocol")
        if contract and protocol not in {"chat-inclusive": {"openai-chat"}, "gemini-chat": {"openai-chat"}, "openrouter-inclusive": {"openai-chat"}, "openrouter-responses-inclusive": {"openai-responses"}, "responses-inclusive": {"openai-codex-responses", "openai-responses"},
                                     "converse-exclusive": {"bedrock-converse"},
                                     "anthropic-exclusive": {"anthropic-messages"},
                                     "opencode-exclusive": {"openai-chat", "openai-responses", "anthropic-messages"},
                                     "claude-cli-exclusive": {"vendor-cli"}, "codex-cli-inclusive": {"vendor-cli"},
                                     "agy-cli-exclusive": {"vendor-cli"}}[contract.layout]:
            raise ValueError("usage contract protocol mismatch")
        normalized = normalize_usage(evidence.get("usage"), contract)
        selected_tariff = (tariffs or {}).get(profile)
        provider_cost = reported_cost(evidence.get('usage'), contract)
        estimate = cli_cost_estimate(evidence.get('usage'), contract)
        if estimate is None and len(transport) == 1 and transport[0].get('vendor') == 'claude':
            estimate = cli_cost_estimate(transport[0], contract)
            if estimate is not None:
                estimate['result_subtype'] = transport[0].get('result_subtype')
        if row is not None and row.get("model_identity") == "mismatch":
            selected_tariff = None
            provider_cost = estimate = None
            normalized["issues"].append("response_model_mismatch_unpriced")
        calls.append({**intent, "execution": "not_started" if call in not_started else "possibly_executed",
                      "usage_source": "observation" if row is not None else "saved_reply" if receipt else "missing",
                      "profile_digest": profile, "observation": row,
                      "raw_usage": evidence.get("usage"), "normalized": normalized,
                      "provider_timing": evidence.get("provider_timing"),
                      "provider_estimate": estimate,
                      "cost": provider_cost or price_usage(normalized, selected_tariff)})

    active = [c for c in calls if c["execution"] != "not_started"]
    tokens = {}
    for field in FIELDS:
        known = [c["normalized"]["tokens"][field] for c in active]
        tokens[field] = {"known_sum": sum(v for v in known if v is not None),
                         "unknown_calls": sum(v is None for v in known)}
    currencies = {}
    for c in active:
        cost = c["cost"]
        if cost["currency"] is not None:
            key = cost["currency"]
            currencies[key] = str(Decimal(currencies.get(key, "0")) + Decimal(cost["known_amount"]))
    observed = list(indexed.values()) + counts
    known_duration = [r.get("duration_ms") for r in observed]
    duration_valid = lambda v: type(v) in (int, float) and 0 <= v < float("inf")
    counted_ids = {e["count"].get("observation_id") for e in events if e["kind"] == "context_counted"}
    missing_counts = len(counted_ids - {r["observation_id"] for r in counts})
    physical = [t for row in observed for t in row.get('transport_calls', [])]
    unobserved_transports = sum(max(0, row.get('client_invocations', 0) - len(row.get('transport_calls', []))) for row in observed)
    return {"schema": "tare-native-usage/1", "session_id": session, "binding": events[0]["binding"],
            "calls": calls, "counts": counts, "tokens": tokens,
            "known_cost_by_currency": currencies,
            "unpriced_or_partial_calls": sum(not c["cost"]["complete"] for c in active),
            "coverage": {"generation_intents": len(calls), "not_started": len(not_started & intents.keys()),
                         "missing_generation_observations": sum(c["observation"] is None for c in active),
                         "observation_failures": observation_failures,
                         "count_operations_observed": len(counts),
                         "missing_journaled_count_observations": missing_counts,
                         "client_invocations_observed": sum(r.get("client_invocations", 0) for r in observed),
                         "count_coverage": "unknown" if observation_failures is None or observation_failures or missing_counts else "caller_supplied_observation_set",
                         "physical_attempts": "unknown_for_injected_client"},
            "transport": {"observed_exchanges": len(physical),
                          "client_invocations_without_physical_receipt": unobserved_transports,
                          "http_attempts_upper_bound_for_observed_exchanges": (
                              sum(t['http_attempts_upper_bound'] for t in physical)
                              if all(type(t.get('http_attempts_upper_bound')) is int for t in physical) else None),
                          "http_attempts_unknown_exchanges": sum(type(t.get('http_attempts_upper_bound')) is not int for t in physical),
                          "local_cleanup_confirmed": sum(t.get('cleanup_verified') is True for t in physical),
                          "local_cleanup_unconfirmed": sum(t.get('cleanup_verified') is not True for t in physical),
                          "remote_stop_unknown": sum(t.get('remote_stop') == 'unknown' for t in physical)},
            "timing": {"observed_operation_ms": sum(v for v in known_duration if duration_valid(v)),
                       "invalid_or_missing_observed_durations": sum(not duration_valid(v) for v in known_duration),
                       "preparation_and_task_wall_ms": None},
            "scope": "Provider account charge when reported; otherwise token tariff estimate. External billing, infrastructure and preparation are separate."}


def enrich_provider_costs(report, deployment=None):
    """Derive newly supported provider values from old receipts without rewriting them."""
    result = copy.deepcopy(report)
    inferred = deployment_contract(deployment or {})
    renormalized = False
    totals, estimates, partial = {}, {}, 0
    for call in result['calls']:
        description = (call.get('normalized') or {}).get('contract')
        contract = UsageContract(**description) if description else None
        if contract is None and inferred:
            contract = UsageContract(**inferred)
        mismatch = (call.get('observation') or {}).get('model_identity') == 'mismatch'
        if contract and isinstance(call.get('raw_usage'), dict):
            normalized = normalize_usage(call['raw_usage'], contract)
            if mismatch:
                normalized['issues'].append('response_model_mismatch_unpriced')
            if normalized != call.get('normalized'):
                call['normalized'] = normalized
                renormalized = True
                tariff = (call.get('cost') or {}).get('tariff')
                if tariff and not mismatch:
                    call['cost'] = price_usage(normalized, UsageTariff(**tariff))
        if not mismatch:
            charge = reported_cost(call.get('raw_usage'), contract)
            if charge:
                call['cost'] = charge
            estimate = cli_cost_estimate(call.get('raw_usage'), contract)
            if estimate:
                call['provider_estimate'] = estimate
        if call.get('execution') == 'not_started':
            continue
        cost = call.get('cost') or {}
        currency, amount = cost.get('currency'), cost.get('known_amount')
        complete = cost.get('complete') is True
        try:
            amount = Decimal(str(amount))
            if not isinstance(currency, str) or not currency.strip() or not amount.is_finite() or amount < 0:
                raise ValueError('invalid cost')
            totals[currency] = str(Decimal(totals.get(currency, '0')) + amount)
        except (InvalidOperation, ValueError):
            complete = False
        partial += not complete
        estimate = call.get('provider_estimate')
        if estimate:
            currency = estimate['currency']
            estimates[currency] = str(Decimal(estimates.get(currency, '0')) + Decimal(estimate['amount']))
    result.update(known_cost_by_currency=totals, unpriced_or_partial_calls=partial)
    if estimates:
        result['provider_estimates_by_currency'] = estimates
    if renormalized:
        active = [c for c in result['calls'] if c.get('execution') != 'not_started']
        result['tokens'] = {field: {
            'known_sum': sum(c['normalized']['tokens'][field] or 0 for c in active),
            'unknown_calls': sum(c['normalized']['tokens'][field] is None for c in active)} for field in FIELDS}
    return result
