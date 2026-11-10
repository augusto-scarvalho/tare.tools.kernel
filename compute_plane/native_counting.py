"""Usage-anchored estimates; never tokenizer measurements or admission policy."""
from dataclasses import replace
import json
import math

from tare_tools_agent_runtime import ConversationLimits, ConversationLog, ProjectionSpec
from .native_journal import NATIVE_JOURNAL_MAX_BYTES, NATIVE_JOURNAL_MAX_EVENT_BYTES


def input_tokens(protocol, endpoint, usage):
    """Only qualified layouts. Cached input still occupies the context window."""
    if not isinstance(usage, dict):
        return None
    if protocol == 'openai-chat':
        keys = ('prompt_tokens',)
    elif protocol in {'openai-codex-responses', 'openai-responses'} or protocol == 'vendor-cli' and endpoint == 'cli://codex':
        keys = ('input_tokens',)
    elif protocol == 'bedrock-converse':
        keys = ('inputTokens', 'cacheReadInputTokens', 'cacheWriteInputTokens')
    elif protocol == 'anthropic-messages' or protocol == 'vendor-cli' and endpoint == 'cli://claude':
        keys = ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
    elif protocol == 'gemini-generate-content':
        keys = ('promptTokenCount',)  # includes cached content
    elif protocol == 'vendor-cli' and endpoint == 'cli://agy':
        # Qualified AGY layout is exclusive, as in native_usage: cache reads
        # still occupy context even though they are billed separately.
        keys = ('input_tokens', 'cache_read_tokens')
    else:
        return None
    values = [usage.get(key) for key in keys]
    if any(type(value) is not int or value < 0 for value in values):
        return None
    total = sum(values)
    return total if total > 0 else None


def added_bytes(old, new):
    """Charge new/changed content in bytes, without credit for removed content."""
    from .native_models import encode
    if type(old) is type(new) and old == new:
        return 0
    if isinstance(old, dict) and isinstance(new, dict):
        return sum(added_bytes(old[key], value) if key in old else len(encode({key: value}).encode())
                   for key, value in new.items())
    if isinstance(old, list) and isinstance(new, list):
        # Position by position: a changed system message (summary requests) costs its own
        # bytes, not the whole history after it. Shifted elements still count as changed.
        changed = sum(added_bytes(o, n) for o, n in zip(old, new))
        return changed + (len(encode(new[len(old):]).encode()) if len(new) > len(old) else 0)
    if isinstance(old, str) and isinstance(new, str) and new.startswith(old):
        return len(encode(new[len(old):]).encode())
    return len(encode(new).encode())


def observation_anchors(observations, protocol, endpoint, profile_digest):
    """Reduce custody observations to small identities; refuse duplicate evidence."""
    anchors = {}
    for index, row in enumerate(observations):
        if index >= 4096:
            raise ValueError('native_count_observations_over_budget')
        if (row.get('purpose') != 'model' or row.get('protocol') != protocol
                or row.get('profile_digest') != profile_digest):
            continue
        key = (row.get('session_id'), row.get('call_id'))
        tokens = input_tokens(protocol, endpoint, row.get('usage'))
        valid = (row.get('status') in {'complete', 'partial', 'recoverable_tool_arguments'}
                 and row.get('model_identity') in {'matched', 'requested_route_only'} and tokens is not None)
        anchor = {'request_sha256': row.get('request_sha256'), 'input_tokens': tokens} if valid else None
        if anchor is not None and 'remaining_model_calls' in row:
            anchor['remaining_model_calls'] = row['remaining_model_calls']
        anchors[key] = anchor if key not in anchors else None
    return anchors


def estimate(adapter, request, wire, *, has_images):
    """Bind actual prior usage to a reconstructed request before estimating growth.

    Hash mismatch (including changed instructions, tools or compaction) cannot
    reuse an old measurement. Images and unknown vendor layouts stay explicit
    byte estimates. This makes no inference or tokenization calls.
    """
    from .native_models import encode, object_json, sha
    fallback = {'count_basis': 'serialized_utf8_bytes',
                'estimated_input_tokens': len(encode(wire).encode()) + 1024,
                'calibration_reason': 'no_matching_usage_anchor'}
    if adapter.profile.protocol == 'vendor-cli':
        fallback['estimated_input_tokens'] += 4096
    if has_images:
        return fallback | {'calibration_reason': 'image_token_count_not_qualified'}
    events = request.context.events()
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event['kind'] != 'model_returned':
            continue
        metadata = object_json(event['reply']['metadata_json'])
        tokens = input_tokens(adapter.profile.protocol, adapter.profile.endpoint, metadata.get('usage'))
        key = (request.session_id, event['call_id'])
        anchor = metadata.get('count_anchor')
        legacy = adapter._count_anchors.get(key)
        if anchor is not None:
            if (not isinstance(anchor, dict) or anchor.get('session_id') != request.session_id
                    or anchor.get('call_id') != event['call_id']):
                return fallback
        else:
            anchor = legacy
        if (not isinstance(anchor, dict) or tokens is None
                or metadata.get('profile_digest') != adapter.profile_digest
                or anchor.get('input_tokens') != tokens):
            return fallback
        if key in adapter._count_anchors and (legacy is None or any(
                anchor.get(k) != legacy.get(k) for k in ('input_tokens', 'request_sha256'))):
            return fallback | {'calibration_reason': 'conflicting_usage_anchor'}
        # Reconstruct the native journal's accepted event and projection envelope;
        # the general-purpose defaults reject valid history before counting it.
        log = ConversationLog(ConversationLimits(
            max_event_bytes=NATIVE_JOURNAL_MAX_EVENT_BYTES,
            max_projection_bytes=NATIVE_JOURNAL_MAX_BYTES))
        for previous in events[:index]:
            log.append(previous)
        remaining = anchor.get('remaining_model_calls')
        if remaining is not None and (type(remaining) is not int or remaining < 1):
            return fallback | {'calibration_reason': 'invalid_prior_call_budget'}
        previous_request = replace(request, call_id=event['call_id'], purpose='model', suffix='',
                                   admitted_request_sha256=None, remaining_model_calls=remaining,
                                   context=log.project(ProjectionSpec(1, 'count-anchor-v1')))
        previous_wire = adapter.wire(previous_request)
        headroom = 1024 + (4096 if adapter.profile.protocol == 'vendor-cli' else 0)
        if sha(previous_wire) != anchor.get('request_sha256'):
            # A compacted context replaced the history the anchor counted, so its request cannot be rebuilt. The
            # anchor's measured tokens per byte still prices the whole request: a byte per token overstated it
            # ~3x and rejected compactions that fit (COMPACTION_TARGET_NOT_REACHED).
            size = anchor.get('request_bytes')
            if type(size) is not int or size <= 0:
                return fallback | {'calibration_reason': 'prior_request_changed'}
            per_byte = min(1.0, 1.25 * tokens / size)
            return {'count_basis': 'measured_tokens_per_utf8_byte',
                    'estimated_input_tokens': math.ceil(len(encode(wire).encode()) * per_byte) + headroom,
                    'calibration': {'call_id': event['call_id'], 'input_tokens': tokens, 'request_bytes': size,
                                    'headroom': headroom, 'tokens_per_byte': round(per_byte, 4)}}
        # Charge growth at this session's measured tokens per byte, with a 25% margin and
        # never above one token per byte. Counting every byte as a token overstated tool
        # output ~3x and made small local windows compact after a few file reads.
        prior_bytes = len(encode(previous_wire).encode())
        per_byte = min(1.0, 1.25 * tokens / prior_bytes) if prior_bytes else 1.0
        if (adapter.profile.protocol == 'vendor-cli' and adapter.profile.endpoint == 'cli://claude'
                and request.purpose == 'summary' and tokens <= prior_bytes
                and set(json.loads(previous_wire['prompt'])) != set(json.loads(wire['prompt']))):
            # Claude's formatter and plain summary quote the same history under
            # different envelope keys. Structural growth treats that history as
            # entirely new and charges it on top of its measured input again.
            # Price the current complete envelope with the verified anchor's
            # existing 25% margin instead, as for a replaced context above.
            return {'count_basis': 'measured_tokens_per_utf8_byte',
                    'estimated_input_tokens': math.ceil(len(encode(wire).encode()) * per_byte) + headroom,
                    'calibration_reason': 'summary_codec_changed',
                    'calibration': {'call_id': event['call_id'], 'request_sha256': anchor['request_sha256'],
                                    'input_tokens': tokens, 'request_bytes': prior_bytes,
                                    'headroom': headroom, 'tokens_per_byte': round(per_byte, 4)}}
        budget_bytes = 0
        # Replies that recorded their budget notice keep it in later history, so the prior
        # wire is an exact prefix and is compared whole. Older journals dropped the notice.
        retained = type(metadata.get('budget_notice')) is int
        if not retained and (remaining is not None or request.remaining_model_calls is not None):
            # The current budget is a transient suffix. Bedrock merges it with
            # adjacent user content, so compare without that known suffix only
            # AFTER verifying the exact prior wire. Charge its full new bytes;
            # retain all prior tokens, including the previous budget notice.
            previous_wire = adapter.wire(replace(previous_request, remaining_model_calls=None))
            without_budget = adapter.wire(replace(request, remaining_model_calls=None))
            budget_bytes = max(0, len(encode(wire).encode()) - len(encode(without_budget).encode()))
            wire = without_budget
        if adapter.profile.protocol == 'vendor-cli':
            # The CLI codec packs the conversation into a JSON string. Compare
            # its structure so a changed closing suffix does not recharge it all.
            previous_wire = {**previous_wire, 'prompt': json.loads(previous_wire['prompt'])}
            wire = {**wire, 'prompt': json.loads(wire['prompt'])}
        growth = added_bytes(previous_wire, wire) + budget_bytes
        return {'count_basis': 'prior_usage_plus_utf8_growth',
                'estimated_input_tokens': tokens + math.ceil(growth * per_byte) + headroom,
                'calibration': {'call_id': event['call_id'], 'request_sha256': anchor['request_sha256'],
                                'input_tokens': tokens, 'growth_bytes': growth, 'headroom': headroom,
                                'tokens_per_byte': round(per_byte, 4)}}
    return fallback
