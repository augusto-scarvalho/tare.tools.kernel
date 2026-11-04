"""Tests for compute_plane.native_json_read."""

from __future__ import annotations

import json
import pytest

from compute_plane.native_json_read import (
    MAX_TEXT_BYTES,
    VIEW_ARRAY_ITEMS,
    VIEW_OBJECT_DIRECTORY,
    VIEW_PROJECTED_OBJECT,
    VIEW_SCALAR,
    project_json,
)


def test_large_array_projection_and_complete_enumeration():
    """Test >1MB one-line JSON array of objects projected and paginated to completion."""
    count = 2500
    records = [
        {
            "id": i,
            "title": f"Task item title number {i} with padding text to expand size",
            "body": "x" * 350,
            "status": "completed" if i % 2 == 0 else "pending",
        }
        for i in range(count)
    ]
    raw = json.dumps(records, separators=(",", ":")).encode("utf-8")
    assert len(raw) > 1024 * 1024  # > 1 MiB
    assert b"\n" not in raw  # one-line

    offset = 0
    all_retrieved: list[dict[str, object]] = []

    while True:
        res = project_json(
            raw,
            {
                "json_pointer": "",
                "json_fields": "/id,/title,/status",
                "offset": offset,
                "limit": 50,
            },
        )
        assert res["format"] == "json"
        assert res["view"] == VIEW_ARRAY_ITEMS
        assert res["json_pointer"] == ""
        assert res["offset"] == offset
        assert res["total"] == count

        text_bytes = res["text"].encode("utf-8")
        assert len(text_bytes) <= MAX_TEXT_BYTES

        page_items = json.loads(res["text"])
        assert isinstance(page_items, list)
        assert len(page_items) == res["returned"]
        all_retrieved.extend(page_items)

        if res["next_offset"] is None:
            assert res["truncated"] is False
            break

        assert res["truncated"] is True
        assert res["next_offset"] == offset + res["returned"]
        offset = res["next_offset"]

    assert len(all_retrieved) == count
    for i, item in enumerate(all_retrieved):
        assert item["/id"] == i
        assert item["/title"] == f"Task item title number {i} with padding text to expand size"
        assert item["/status"] == ("completed" if i % 2 == 0 else "pending")
        assert "body" not in item  # unrequested field omitted


def test_escaped_pointer_keys():
    """Test RFC 6901 escaped characters (~1 for '/', ~0 for '~') in pointer and fields."""
    doc = {
        "a/b": {
            "c~d": "found_escaped",
            "slash/tilde~mix": 99,
        },
        "items": [
            {"path/name": "alpha", "flag~state": True},
            {"path/name": "beta", "flag~state": False},
        ],
    }
    raw = json.dumps(doc).encode("utf-8")

    # Navigate to nested escaped pointer
    res = project_json(raw, {"json_pointer": "/a~1b/c~0d"})
    assert res["view"] == VIEW_SCALAR
    assert res["total"] == 1
    assert res["returned"] == 1
    assert json.loads(res["text"]) == "found_escaped"

    # Deep pointer with both slash and tilde in one token
    res_mix = project_json(raw, {"json_pointer": "/a~1b/slash~1tilde~0mix"})
    assert res_mix["view"] == VIEW_SCALAR
    assert json.loads(res_mix["text"]) == 99

    # Relative fields with escaped keys
    res_arr = project_json(
        raw,
        {
            "json_pointer": "/items",
            "json_fields": "/path~1name,/flag~0state",
            "limit": 10,
        },
    )
    assert res_arr["view"] == VIEW_ARRAY_ITEMS
    items = json.loads(res_arr["text"])
    assert len(items) == 2
    assert items[0] == {"/path~1name": "alpha", "/flag~0state": True}
    assert items[1] == {"/path~1name": "beta", "/flag~0state": False}


def test_unicode_byte_budget_shortens_page():
    """Verify that page budget is enforced on UTF-8 bytes rather than characters."""
    # 4-byte UTF-8 emoji character: 🚀 (\U0001F680)
    # Each entry is ~5,000 characters, but 20,000 UTF-8 bytes.
    # Two entries would be 10,000 chars (< 24,000 chars), but 40,000 bytes (> 24,000 bytes).
    payload = "🚀" * 4500
    records = [{"id": i, "content": payload} for i in range(5)]
    raw = json.dumps(records).encode("utf-8")

    res = project_json(
        raw,
        {
            "json_pointer": "",
            "limit": 5,
        },
    )

    text_bytes = res["text"].encode("utf-8")
    assert len(text_bytes) <= MAX_TEXT_BYTES
    # Only 1 complete entry could fit under the 24,000 UTF-8 bytes budget
    assert res["returned"] == 1
    assert res["truncated"] is True
    assert res["next_offset"] == 1
    data = json.loads(res["text"])
    assert len(data) == 1
    assert data[0]["id"] == 0
    assert data[0]["content"] == payload


def test_object_directory_and_deep_selection():
    """Test object directory of immediate keys and deep pointer selection."""
    doc = {
        "config": {"host": "localhost", "port": 8080},
        "users": [{"name": "alice"}, {"name": "bob"}],
        "readme": "A" * 1500,
        "max_retries": 5,
        "is_active": True,
        "archived_at": None,
    }
    raw = json.dumps(doc).encode("utf-8")

    # Object directory without fields
    res = project_json(raw, {"json_pointer": "", "limit": 10})
    assert res["view"] == VIEW_OBJECT_DIRECTORY
    assert res["total"] == 6
    assert res["returned"] == 6
    assert res["next_offset"] is None
    assert res["truncated"] is False

    directory = json.loads(res["text"])
    assert directory["config"] == {"type": "object", "size": 2}
    assert directory["users"] == {"type": "array", "size": 2}
    assert directory["readme"] == {"type": "string", "size": 1500}
    assert directory["max_retries"] == 5
    assert directory["is_active"] is True
    assert directory["archived_at"] is None

    # Deep pointer selection to nested object
    res_deep = project_json(raw, {"json_pointer": "/config/port"})
    assert res_deep["view"] == VIEW_SCALAR
    assert res_deep["total"] == 1
    assert res_deep["returned"] == 1
    assert json.loads(res_deep["text"]) == 8080


def test_object_directory_pagination():
    """Test pagination of object directory immediate keys."""
    doc = {f"key_{i:02d}": i for i in range(10)}
    raw = json.dumps(doc).encode("utf-8")

    page1 = project_json(raw, {"json_pointer": "", "offset": 0, "limit": 4})
    assert page1["view"] == VIEW_OBJECT_DIRECTORY
    assert page1["total"] == 10
    assert page1["returned"] == 4
    assert page1["next_offset"] == 4
    assert page1["truncated"] is True
    keys_page1 = list(json.loads(page1["text"]).keys())
    assert keys_page1 == [f"key_{i:02d}" for i in range(4)]

    page2 = project_json(raw, {"json_pointer": "", "offset": 4, "limit": 4})
    assert page2["returned"] == 4
    assert page2["next_offset"] == 8
    assert page2["truncated"] is True

    page3 = project_json(raw, {"json_pointer": "", "offset": 8, "limit": 4})
    assert page3["returned"] == 2
    assert page3["next_offset"] is None
    assert page3["truncated"] is False
    keys_page3 = list(json.loads(page3["text"]).keys())
    assert keys_page3 == ["key_08", "key_09"]


def test_projected_object():
    """Test object with json_fields projects fields as a single record."""
    doc = {
        "id": "item_123",
        "title": "Main Project",
        "meta": {"priority": "high", "weight": 10},
        "secret": "do_not_expose",
    }
    raw = json.dumps(doc).encode("utf-8")

    res = project_json(
        raw,
        {
            "json_pointer": "",
            "json_fields": "/id,/meta/priority,/missing_key",
        },
    )
    assert res["view"] == VIEW_PROJECTED_OBJECT
    assert res["total"] == 1
    assert res["returned"] == 1
    assert res["next_offset"] is None
    assert res["truncated"] is False

    projected = json.loads(res["text"])
    assert projected == {
        "/id": "item_123",
        "/meta/priority": "high",
    }
    assert "/missing_key" not in projected
    assert "secret" not in projected


def test_null_retained_vs_missing_omitted():
    """Verify that explicit null values are retained while missing keys are omitted."""
    data = [
        {"id": 1, "assigned_to": None, "extra": "val"},
        {"id": 2, "extra": "val"},  # assigned_to is missing
    ]
    raw = json.dumps(data).encode("utf-8")

    res = project_json(
        raw,
        {
            "json_pointer": "",
            "json_fields": "/id,/assigned_to,/unknown",
        },
    )
    items = json.loads(res["text"])
    assert len(items) == 2
    # First item has explicit null
    assert "/assigned_to" in items[0]
    assert items[0]["/assigned_to"] is None
    assert "/unknown" not in items[0]
    # Second item has missing assigned_to
    assert "/assigned_to" not in items[1]
    assert "/unknown" not in items[1]


def test_invalid_arguments_and_types():
    """Test rejection of invalid types, negative bounds, and boolean integers."""
    raw = b'{"items": [1, 2, 3]}'

    # raw must be bytes
    with pytest.raises(ValueError, match="raw must be bytes"):
        project_json("not bytes", {"json_pointer": ""})  # type: ignore[arg-type]

    # args must be dict
    with pytest.raises(ValueError, match="args must be a dict"):
        project_json(raw, "not a dict")  # type: ignore[arg-type]

    # Missing json_pointer
    with pytest.raises(ValueError, match="Missing required argument"):
        project_json(raw, {})

    # json_pointer must be str, reject bool
    with pytest.raises(ValueError, match="json_pointer must be a string"):
        project_json(raw, {"json_pointer": True})
    with pytest.raises(ValueError, match="json_pointer must be a string"):
        project_json(raw, {"json_pointer": 123})

    # offset cannot be bool
    with pytest.raises(ValueError, match="offset must be an integer, not boolean"):
        project_json(raw, {"json_pointer": "", "offset": True})
    with pytest.raises(ValueError, match="offset must be an integer, not boolean"):
        project_json(raw, {"json_pointer": "", "offset": False})

    # offset cannot be negative
    with pytest.raises(ValueError, match="offset cannot be negative"):
        project_json(raw, {"json_pointer": "", "offset": -1})

    # limit cannot be bool
    with pytest.raises(ValueError, match="limit must be an integer, not boolean"):
        project_json(raw, {"json_pointer": "", "limit": True})

    # limit range 1..100
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        project_json(raw, {"json_pointer": "", "limit": 0})
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        project_json(raw, {"json_pointer": "", "limit": 101})

    # json_fields cannot exceed 16
    too_many_fields = ",".join(f"/field_{i}" for i in range(17))
    with pytest.raises(ValueError, match="exceeds maximum of 16 fields"):
        project_json(raw, {"json_pointer": "", "json_fields": too_many_fields})

    # Duplicate field pointers in json_fields
    with pytest.raises(ValueError, match="Duplicate field pointer"):
        project_json(raw, {"json_pointer": "", "json_fields": "/id,/id"})

    # json_fields applied to scalar
    with pytest.raises(ValueError, match="json_fields cannot be applied to a scalar"):
        project_json(b"42", {"json_pointer": "", "json_fields": "/id"})


def test_invalid_pointers_and_indices():
    """Test malformed pointers, invalid array indices, and out-of-bounds access."""
    raw = b'{"arr": [10, 20, 30], "obj": {"x": 1}}'

    # Pointer must start with '/' when non-empty
    with pytest.raises(ValueError, match="Malformed json_pointer"):
        project_json(raw, {"json_pointer": "arr"})

    # Malformed escape sequence in pointer
    with pytest.raises(ValueError, match="Malformed pointer escape"):
        project_json(raw, {"json_pointer": "/arr~2item"})
    with pytest.raises(ValueError, match="Malformed pointer escape"):
        project_json(raw, {"json_pointer": "/arr~"})

    # Leading zeros in array index (RFC 6901 violation)
    with pytest.raises(ValueError, match="Malformed or invalid array index"):
        project_json(raw, {"json_pointer": "/arr/01"})

    # Negative index in pointer
    with pytest.raises(ValueError, match="Malformed or invalid array index"):
        project_json(raw, {"json_pointer": "/arr/-1"})

    # Non-numeric index for array
    with pytest.raises(ValueError, match="Malformed or invalid array index"):
        project_json(raw, {"json_pointer": "/arr/abc"})

    # Array index out of bounds
    with pytest.raises(ValueError, match="Array index out of bounds"):
        project_json(raw, {"json_pointer": "/arr/99"})

    # Key not found in object
    with pytest.raises(ValueError, match="Pointer key not found"):
        project_json(raw, {"json_pointer": "/nonexistent"})

    # Traversal into scalar
    with pytest.raises(ValueError, match="Cannot traverse into scalar"):
        project_json(raw, {"json_pointer": "/obj/x/child"})


def test_offset_beyond_total():
    """Explicitly test that offsets beyond total raise actionable ValueError."""
    # Array with 3 elements
    raw_arr = b"[1, 2, 3]"
    with pytest.raises(ValueError, match="offset 3 is beyond total 3"):
        project_json(raw_arr, {"json_pointer": "", "offset": 3})
    with pytest.raises(ValueError, match="offset 5 is beyond total 3"):
        project_json(raw_arr, {"json_pointer": "", "offset": 5})

    # Empty array
    raw_empty = b"[]"
    res_empty = project_json(raw_empty, {"json_pointer": "", "offset": 0})
    assert res_empty["total"] == 0
    assert res_empty["returned"] == 0
    assert res_empty["next_offset"] is None
    with pytest.raises(ValueError, match="offset 1 is beyond total 0"):
        project_json(raw_empty, {"json_pointer": "", "offset": 1})

    # Object with 2 keys
    raw_obj = b'{"a": 1, "b": 2}'
    with pytest.raises(ValueError, match="offset 2 is beyond total 2"):
        project_json(raw_obj, {"json_pointer": "", "offset": 2})

    # Scalar (total = 1)
    raw_scalar = b"123"
    with pytest.raises(ValueError, match="offset 1 is beyond total 1"):
        project_json(raw_scalar, {"json_pointer": "", "offset": 1})


def test_invalid_json_utf8_duplicate_keys_constants():
    """Test rejection of invalid UTF-8, malformed JSON, duplicate keys, and NaN/Infinity."""
    # Invalid UTF-8
    with pytest.raises(ValueError, match="Invalid UTF-8 encoding"):
        project_json(b"\xff\xfe\x00\x00", {"json_pointer": ""})

    # Malformed JSON
    with pytest.raises(ValueError, match="Invalid JSON"):
        project_json(b"{key: value}", {"json_pointer": ""})

    # Duplicate JSON object keys
    with pytest.raises(ValueError, match="Duplicate JSON object key: 'dup'"):
        project_json(b'{"dup": 1, "dup": 2}', {"json_pointer": ""})

    # Nested duplicate keys
    with pytest.raises(ValueError, match="Duplicate JSON object key: 'nested'"):
        project_json(b'{"parent": {"nested": "a", "nested": "b"}}', {"json_pointer": ""})

    # Reject NaN
    with pytest.raises(ValueError, match="Invalid JSON constant not permitted: 'NaN'"):
        project_json(b'[1, NaN, 3]', {"json_pointer": ""})

    # Reject Infinity
    with pytest.raises(ValueError, match="Invalid JSON constant not permitted: 'Infinity'"):
        project_json(b'[1, Infinity, 3]', {"json_pointer": ""})

    # Reject -Infinity
    with pytest.raises(ValueError, match="Invalid JSON constant not permitted: '-Infinity'"):
        project_json(b'[1, -Infinity, 3]', {"json_pointer": ""})


def test_oversized_single_record_raises_actionable_error():
    """Verify actionable ValueError when a single record exceeds the 24,000 UTF-8 byte budget."""
    huge_str = "A" * (MAX_TEXT_BYTES + 100)
    raw_array = json.dumps([{"huge": huge_str}]).encode("utf-8")

    with pytest.raises(
        ValueError,
        match="Single record exceeds 24000 UTF-8 byte budget; request a deeper pointer or selected fields",
    ):
        project_json(raw_array, {"json_pointer": ""})

    raw_scalar = json.dumps(huge_str).encode("utf-8")
    with pytest.raises(
        ValueError,
        match="Single record exceeds 24000 UTF-8 byte budget; request a deeper pointer or selected fields",
    ):
        project_json(raw_scalar, {"json_pointer": ""})


def test_oversized_selected_child_returns_explicit_directory_not_uninspected_values():
    doc={'rows':[{'id':str(i),'detail':'PRIVATE-'+'x'*1000} for i in range(60)],'title':'fixture'}
    raw=json.dumps(doc).encode()
    args={'json_pointer':'','json_fields':'/rows'}
    before=dict(args)
    result=project_json(raw,args)
    assert result['view']=='object directory' and result['projection_complete'] is False
    assert result['requested_view']=='projected object' and result['truncated'] is False
    directory=json.loads(result['text'])
    assert set(directory)==set(doc) and 'PRIVATE' not in result['text']
    assert len(result['text'].encode())<=MAX_TEXT_BYTES and args==before
    assert 'not inspected' in result['projection_notice']


def test_oversized_projection_directory_pages_with_same_fields_without_data_substitution():
    raw=json.dumps({'rows':['x'*1000]*60,**{f'key{i}':i for i in range(6)}}).encode()
    first=project_json(raw,{'json_pointer':'','json_fields':'/rows','limit':2})
    second=project_json(raw,{'json_pointer':'','json_fields':'/rows','offset':first['next_offset'],'limit':2})
    assert first['projection_complete'] is second['projection_complete'] is False
    assert not set(json.loads(first['text'])) & set(json.loads(second['text']))
    assert second['next_offset']==4 and first['total']==second['total']==7


def test_no_source_mutation():
    """Verify that source raw bytes and input arguments are never mutated."""
    original_bytes = b'{"items": [10, 20, 30], "meta": {"owner": "test"}}'
    immutable_copy = bytes(original_bytes)
    args = {
        "json_pointer": "/items",
        "offset": 0,
        "limit": 2,
    }
    args_copy = dict(args)

    res = project_json(original_bytes, args)
    assert res["returned"] == 2
    assert original_bytes == immutable_copy
    assert args == args_copy
