"""Pure bounded helper for generic paginated JSON reads."""

from __future__ import annotations

import json
import math
from typing import Any

MAX_TEXT_BYTES = 24_000
MAX_FIELDS = 16
DEFAULT_OFFSET = 0
DEFAULT_LIMIT = 50
MIN_LIMIT = 1
MAX_LIMIT = 100

VIEW_OBJECT_DIRECTORY = "object directory"
VIEW_ARRAY_ITEMS = "array items"
VIEW_PROJECTED_OBJECT = "projected object"
VIEW_SCALAR = "scalar"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate JSON object key: {k!r}")
        res[k] = v
    return res


def _reject_constant(c: str) -> None:
    raise ValueError(f"Invalid JSON constant not permitted: {c!r}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('JSON number exceeds finite float range')
    return result


def _unescape_token(token: str) -> str:
    i = 0
    n = len(token)
    while i < n:
        if token[i] == "~":
            if i + 1 >= n or token[i + 1] not in ("0", "1"):
                raise ValueError(f"Malformed pointer escape in token: {token!r}")
            i += 2
        else:
            i += 1
    return token.replace("~1", "/").replace("~0", "~")


def _parse_pointer(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"Malformed json_pointer: must start with '/': {pointer!r}")
    raw_tokens = pointer.split("/")[1:]
    return [_unescape_token(t) for t in raw_tokens]


def _resolve_pointer(doc: Any, tokens: list[str]) -> Any:
    curr = doc
    for token in tokens:
        if isinstance(curr, dict):
            if token not in curr:
                raise ValueError(f"Pointer key not found: {token!r}")
            curr = curr[token]
        elif isinstance(curr, list):
            if not (token == "0" or (token.isascii() and token.isdigit() and not token.startswith("0"))):
                raise ValueError(f"Malformed or invalid array index in pointer: {token!r}")
            idx = int(token)
            if idx >= len(curr):
                raise ValueError(f"Array index out of bounds: {idx} >= {len(curr)}")
            curr = curr[idx]
        else:
            raise ValueError(f"Cannot traverse into scalar with token: {token!r}")
    return curr


def _resolve_relative_field(obj: Any, field_tokens: list[str]) -> tuple[bool, Any]:
    curr = obj
    for token in field_tokens:
        if isinstance(curr, dict):
            if token not in curr:
                return False, None
            curr = curr[token]
        elif isinstance(curr, list):
            if not (token == "0" or (token.isascii() and token.isdigit() and not token.startswith("0"))):
                raise ValueError(f"Malformed or invalid array index in pointer: {token!r}")
            idx = int(token)
            if idx >= len(curr):
                return False, None
            curr = curr[idx]
        else:
            return False, None
    return True, curr


def _project_record(record: Any, parsed_fields: list[tuple[str, list[str]]]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for raw_ptr, tokens in parsed_fields:
        found, val = _resolve_relative_field(record, tokens)
        if found:
            projected[raw_ptr] = val
    return projected


def _describe_child(val: Any) -> Any:
    if isinstance(val, dict):
        return {"type": "object", "size": len(val)}
    if isinstance(val, list):
        return {"type": "array", "size": len(val)}
    if isinstance(val, str):
        return {"type": "string", "size": len(val)}
    return val


def project_json(raw: bytes, args: dict[str, Any]) -> dict[str, Any]:
    """Pure generic paginated reader and projector for bounded JSON data."""
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("raw must be bytes or bytearray")
    if not isinstance(args, dict):
        raise ValueError("args must be a dict")

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 encoding: {exc}") from exc

    try:
        doc = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if "json_pointer" not in args:
        raise ValueError("Missing required argument: 'json_pointer'")
    json_pointer = args["json_pointer"]
    if not isinstance(json_pointer, str) or isinstance(json_pointer, bool):
        raise ValueError("json_pointer must be a string")

    pointer_tokens = _parse_pointer(json_pointer)
    target = _resolve_pointer(doc, pointer_tokens)

    raw_fields = args.get("json_fields")
    parsed_fields: list[tuple[str, list[str]]] = []
    if raw_fields is not None:
        if not isinstance(raw_fields, str) or isinstance(raw_fields, bool):
            raise ValueError("json_fields must be a string or None")
        stripped = raw_fields.strip()
        if stripped != "":
            field_strs = [f.strip() for f in stripped.split(",")]
            if len(field_strs) > MAX_FIELDS:
                raise ValueError(f"json_fields exceeds maximum of {MAX_FIELDS} fields: {len(field_strs)}")
            seen_fields: set[str] = set()
            for f in field_strs:
                if f == "":
                    raise ValueError("Empty field pointer in json_fields")
                if not f.startswith("/"):
                    raise ValueError(f"Relative field pointer must start with '/': {f!r}")
                if f in seen_fields:
                    raise ValueError(f"Duplicate field pointer in json_fields: {f!r}")
                seen_fields.add(f)
                tokens = [_unescape_token(t) for t in f.split("/")[1:]]
                parsed_fields.append((f, tokens))

    offset = args.get("offset", DEFAULT_OFFSET)
    if offset is None:
        offset = DEFAULT_OFFSET
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset must be an integer, not boolean or other type")
    if offset < 0:
        raise ValueError(f"offset cannot be negative: {offset}")

    limit = args.get("limit", DEFAULT_LIMIT)
    if limit is None:
        limit = DEFAULT_LIMIT
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer, not boolean or other type")
    if limit < MIN_LIMIT or limit > MAX_LIMIT:
        raise ValueError(f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}: {limit}")

    projection_incomplete = False
    if isinstance(target, list):
        view = VIEW_ARRAY_ITEMS
        total = len(target)
        if total == 0:
            if offset > 0:
                raise ValueError(f"offset {offset} is beyond total {total}")
            candidates = []
        else:
            if offset >= total:
                raise ValueError(f"offset {offset} is beyond total {total}")
            slice_items = target[offset : offset + limit]
            if parsed_fields:
                candidates = [_project_record(item, parsed_fields) for item in slice_items]
            else:
                candidates = slice_items

        if len(candidates) == 0:
            text = "[]"
            returned = 0
            next_offset = None
            truncated = False
        else:
            first_text = json.dumps([candidates[0]], indent=2, ensure_ascii=False)
            if len(first_text.encode("utf-8")) > MAX_TEXT_BYTES:
                raise ValueError(
                    "Single record exceeds 24000 UTF-8 byte budget; request a deeper pointer or selected fields"
                )
            kept = [candidates[0]]
            best_text = first_text
            for item in candidates[1:]:
                trial = kept + [item]
                trial_text = json.dumps(trial, indent=2, ensure_ascii=False)
                if len(trial_text.encode("utf-8")) <= MAX_TEXT_BYTES:
                    kept = trial
                    best_text = trial_text
                else:
                    break
            returned = len(kept)
            text = best_text
            if offset + returned < total:
                next_offset = offset + returned
                truncated = True
            else:
                next_offset = None
                truncated = False

    elif isinstance(target, dict):
        if parsed_fields:
            view = VIEW_PROJECTED_OBJECT
            total = 1
            projected = _project_record(target, parsed_fields)
            text = json.dumps(projected, indent=2, ensure_ascii=False)
            if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
                projection_incomplete = True
            else:
                if offset > 0:
                    raise ValueError(f"offset {offset} is beyond total {total}")
                returned = 1
                next_offset = None
                truncated = False
        if not parsed_fields or projection_incomplete:
            view = VIEW_OBJECT_DIRECTORY
            all_keys = list(target.keys())
            total = len(all_keys)
            if total == 0:
                if offset > 0:
                    raise ValueError(f"offset {offset} is beyond total {total}")
                candidate_keys = []
            else:
                if offset >= total:
                    raise ValueError(f"offset {offset} is beyond total {total}")
                candidate_keys = all_keys[offset : offset + limit]

            if len(candidate_keys) == 0:
                text = "{}"
                returned = 0
                next_offset = None
                truncated = False
            else:
                entries = {k: _describe_child(target[k]) for k in candidate_keys}
                first_k = candidate_keys[0]
                first_text = json.dumps({first_k: entries[first_k]}, indent=2, ensure_ascii=False)
                if len(first_text.encode("utf-8")) > MAX_TEXT_BYTES:
                    raise ValueError(
                        "Single directory entry exceeds 24000 UTF-8 byte budget; request a deeper pointer or selected fields"
                    )
                kept_dict = {first_k: entries[first_k]}
                best_text = first_text
                for k in candidate_keys[1:]:
                    trial_dict = dict(kept_dict)
                    trial_dict[k] = entries[k]
                    trial_text = json.dumps(trial_dict, indent=2, ensure_ascii=False)
                    if len(trial_text.encode("utf-8")) <= MAX_TEXT_BYTES:
                        kept_dict = trial_dict
                        best_text = trial_text
                    else:
                        break
                returned = len(kept_dict)
                text = best_text
                if offset + returned < total:
                    next_offset = offset + returned
                    truncated = True
                else:
                    next_offset = None
                    truncated = False

    else:
        # Scalar: int, float, str, bool, None
        if parsed_fields:
            raise ValueError("json_fields cannot be applied to a scalar")
        view = VIEW_SCALAR
        total = 1
        if offset > 0:
            raise ValueError(f"offset {offset} is beyond total {total}")
        text = json.dumps(target, indent=2, ensure_ascii=False)
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError(
                "Single record exceeds 24000 UTF-8 byte budget; request a deeper pointer or selected fields"
            )
        returned = 1
        next_offset = None
        truncated = False

    return {
        "format": "json",
        "view": view,
        "json_pointer": json_pointer,
        "offset": offset,
        "total": total,
        "returned": returned,
        "next_offset": next_offset,
        "truncated": truncated,
        "text": text,
        **({'requested_view': VIEW_PROJECTED_OBJECT, 'projection_complete': False,
            'projection_notice': 'Selected values exceed the 24000-byte text limit. Directory metadata only; '
                'selected values were not inspected. Point json_pointer into a nested collection and '
                'paginate its items; json_fields selects whole values and does not limit nested collections.'}
           if projection_incomplete else {}),
    }
