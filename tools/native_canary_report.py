"""Offline accounting of the bounded M2 lot; never invokes models or invents intents."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

from native_canary import save, ConversationLog
from compute_plane.native_usage import UsageContract, UsageTariff, native_usage_report, normalize_usage, price_usage

PRICE_SOURCE = 'https://www-cdn.anthropic.com/files/4zrzovbb/website/3684c2faafb97418665782cea0001f439f74b1d2.pdf#page=5'


def report(root):
    contracts = {
        'qwen': UsageContract('chat-inclusive', 'protocol.json/local-identity.json; exact full-wire count equals prompt_tokens in all completed local calls; cache_prompt=false; cached_tokens=0; no separate cache creation billing in this local profile'),
        'bedrock': UsageContract('converse-exclusive', 'cache-control/result.json: 350 uncached + 5958 write/read = 6308 input; CountTokens=6325 (+17); functional calls have the same +17 with zero cache'),
    }
    tariff = UsageTariff('USD', PRICE_SOURCE + '; Opus 4.6 AWS Bedrock Standard In-Region & Geo Cross Region, 5m; list-price estimate, not invoice',
        '2026-05-27; consulted 2026-09-10', json.dumps({'uncached_input_tokens': '5.50', 'cache_read_tokens': '0.55',
                                                     'cache_write_tokens': '6.875', 'output_tokens': '27.50'}))
    totals = {}
    for provider in ('qwen', 'bedrock'):
        dest = root / provider
        log = ConversationLog()
        for event in json.loads((dest / 'journal.json').read_text())['events']:
            log.append(event)
        rows = json.loads((dest / 'observations.json').read_text())['rows']
        profile = next(r['profile_digest'] for r in rows if r['purpose'] == 'model')
        initial = json.loads((dest / 'usage-report.json').read_text())
        result = native_usage_report(log, rows, contracts={profile: contracts[provider]},
            tariffs={profile: tariff} if provider == 'bedrock' else {},
            observation_failures=initial['coverage']['observation_failures'])
        save(dest / 'qualified-usage-report.json', result)
        totals[provider] = {'tokens': result['tokens'], 'known_cost_by_currency': result['known_cost_by_currency'],
                            'coverage': result['coverage']}
    for mode, provider in [('local', 'qwen'), ('cache', 'bedrock')]:
        dest = root / (mode + '-control')
        rows = json.loads((dest / 'observations.json').read_text())['rows']
        calls = []
        for row in rows:
            if row['purpose'] == 'count':
                continue
            normalized = normalize_usage(row.get('usage'), contracts[provider])
            calls.append({'call_id': row['call_id'], 'purpose': row['purpose'], 'raw_usage': row.get('usage'),
                'normalized': normalized, 'cost': price_usage(normalized, tariff if provider == 'bedrock' else None)})
        result = {'scope': 'direct adapter/context diagnostics; no full NativeSession journal and no invented intents',
                  'calls': calls, 'observed_count_operations': sum(r['purpose'] == 'count' for r in rows),
                  'observation_failures': None, 'coverage_limitation': 'control scripts did not persist the adapter sink failure counter'}
        save(dest / 'qualified-usage-report.json', result)
        totals[mode + '-control'] = result
    save(root / 'accounting.json', {'contracts': {k: asdict(v) for k, v in contracts.items()},
        'tariff': asdict(tariff), 'results': totals,
        'exclusions': ['actual bill and discounts', 'local power/hardware', 'counting fees', 'preparation, tools and total task wall time',
                       'remote consumption of cancelled request', 'reasoning token breakdown absent; do not add again to output']})


if __name__ == '__main__':
    report(Path(sys.argv[1]).resolve())
