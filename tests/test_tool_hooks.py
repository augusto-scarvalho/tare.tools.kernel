from __future__ import annotations

import json
from pathlib import Path

import pytest

from compute_plane.hook_cli import MAX_INPUT_BYTES, dispatch


@pytest.mark.parametrize("command", ["egrep -rn Widget src", "fgrep Widget src/a.py",
    "ugrep Widget src", "ag Widget src", "ack Widget src", "findstr.exe /s /i Widget *.py",
    "Select-String -Path *.py -Pattern Widget", "sls Widget *.py"])
@pytest.mark.parametrize("vendor", ["claude", "agy"])
def test_search_variants_must_use_canonical_resolver(command, vendor):
    payload = ({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                "tool_input": {"command": command}} if vendor == "claude" else
               {"toolCall": {"name": "run_command", "args": {"CommandLine": command}}})
    output = dispatch(vendor, json.dumps(payload).encode(), event_hint="pre")
    assert "deny" in json.dumps(output)
    assert "specgraph resolve" in json.dumps(output)
from compute_plane.tool_hooks import SpecGraphEditPolicy, ToolCall, edited_paths


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("vendor", "payload"),
    [
        (
            "claude",
            {"hook_event_name": "PreToolUse", "tool_name": "Grep", "tool_input": {"pattern": "Widget"}},
        ),
        (
            "codex",
            {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rg Widget src"}},
        ),
        (
            "agy",
            {"toolCall": {"name": "grep_search", "args": {"Query": "Widget", "IsRegex": True}}},
        ),
    ],
)
def test_structural_search_is_denied_with_same_canonical_guidance(vendor, payload):
    output = dispatch(vendor, json.dumps(payload).encode())
    encoded = json.dumps(output)

    assert "deny" in encoded
    assert "specgraph resolve" in encoded
    assert "MISS" in encoded


@pytest.mark.parametrize(
    ("vendor", "payload"),
    [
        (
            "claude",
            {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rg -F Widget src"}},
        ),
        (
            "codex",
            {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "git grep -F Widget"}},
        ),
        (
            "agy",
            {"toolCall": {"name": "grep_search", "args": {"Query": "Widget", "IsRegex": False}}},
        ),
    ],
)
def test_explicit_literal_search_is_also_routed_through_canonical_resolver(vendor, payload):
    output = dispatch(vendor, json.dumps(payload).encode())

    assert "deny" in json.dumps(output)
    assert "specgraph resolve" in json.dumps(output)


@pytest.mark.parametrize(
    ("vendor", "payload"),
    [
        (
            "claude",
            {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "specgraph resolve Widget --format json"}},
        ),
        (
            "codex",
            {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "specgraph resolve Widget --format json"}},
        ),
        (
            "agy",
            {"toolCall": {"name": "run_command", "args": {"CommandLine": "specgraph resolve Widget --format json"}}},
        ),
    ],
)
def test_canonical_resolver_is_allowed(vendor, payload):
    output = dispatch(vendor, json.dumps(payload).encode())

    assert output == ({"decision": "allow", "reason": ""} if vendor == "agy" else {})


def test_compound_shell_search_cannot_bypass_canonical_resolver():
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "rg -F Widget src | grep 'Widget.*Factory'"},
    }

    output = dispatch("codex", json.dumps(payload).encode())

    assert "deny" in json.dumps(output)


@pytest.mark.parametrize("vendor", ["claude", "codex"])
def test_malformed_shell_command_fails_closed(vendor):
    payload = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {}}

    output = dispatch(vendor, json.dumps(payload).encode())

    assert "deny" in json.dumps(output)


@pytest.mark.parametrize("vendor", ["claude", "codex", "agy"])
def test_irrelevant_tool_is_allowed(vendor):
    if vendor == "agy":
        payload = {"toolCall": {"name": "view_file", "args": {"AbsolutePath": "README.md"}}}
    else:
        payload = {"hook_event_name": "PreToolUse", "tool_name": "Read", "tool_input": {"file_path": "README.md"}}

    output = dispatch(vendor, json.dumps(payload).encode())

    assert output == ({"decision": "allow", "reason": ""} if vendor == "agy" else {})


@pytest.mark.parametrize("vendor", ["claude", "codex", "agy"])
@pytest.mark.parametrize("raw", [b"not-json", b"[]", b"{\"toolCall\":{}}"])
def test_malformed_or_wrong_vendor_envelope_fails_closed(vendor, raw):
    output = dispatch(vendor, raw)

    assert "deny" in json.dumps(output)


@pytest.mark.parametrize("vendor", ["claude", "codex", "agy"])
def test_oversized_input_fails_closed(vendor):
    output = dispatch(vendor, b"{" + b"x" * MAX_INPUT_BYTES)

    assert "deny" in json.dumps(output)


def test_kernel_specgraph_config_covers_all_searchable_project_surfaces():
    config = (ROOT / "specgraph.yaml").read_text(encoding="utf-8")

    assert 'project_name: "tare.tools.kernel"' in config
    assert "search_paths:\n" in config
    assert "spec_paths: []\n" in config
    assert "adr_paths: []\n" in config
    for path in (
        "assurance_plane",
        "compute_plane",
        "control_plane",
        "data_plane",
        "experience_plane",
        "scripts",
        "tests",
        "docs",
        "README.md",
        "README.pt-BR.md",
        "CHANGELOG.md",
        "pyproject.toml",
        "specgraph.yaml",
    ):
        assert f"  - {path}\n" in config


@pytest.mark.parametrize(
    ("relative_path", "namespace", "matcher", "command"),
    [
        (".claude/settings.json", "hooks", "Bash|Grep|Edit|Write", "tare-hook claude"),
        (".codex/hooks.json", "hooks", "Bash|Grep|apply_patch|Edit|Write", "tare-hook codex"),
        (
            ".agents/hooks.json",
            "tare-specgraph-first",
            "run_command|grep_search|write_to_file|replace_file_content|multi_replace_file_content",
            "tare-hook agy --event pre",
        ),
    ],
)
def test_checked_in_vendor_manifests_are_thin_and_pinned(
    relative_path, namespace, matcher, command
):
    payload = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    entries = payload[namespace]["PreToolUse"]

    assert entries == [
        {
            "matcher": matcher,
            "hooks": [{"type": "command", "command": command, "timeout": 5}],
        }
    ]


def test_claude_manifest_includes_native_edits_and_post_edit_notice():
    payload = json.loads((ROOT / ".claude/settings.json").read_text(encoding="utf-8"))
    assert payload["hooks"]["PreToolUse"][0]["matcher"] == "Bash|Grep|Edit|Write"
    assert payload["hooks"]["PostToolUse"][0]["matcher"] == "Edit|Write"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (
            ToolCall("claude", "PreToolUse", "Edit", {"file_path": "src/a.py"}),
            ("src/a.py",),
        ),
        (
            ToolCall("agy", "PreToolUse", "replace_file_content", {"TargetFile": "src/b.py"}),
            ("src/b.py",),
        ),
        (
            ToolCall(
                "codex",
                "PreToolUse",
                "apply_patch",
                {
                    "command": "*** Begin Patch\n*** Update File: src/a.py\n*** Add File: src/b.py\n*** End Patch"
                },
            ),
            ("src/a.py", "src/b.py"),
        ),
    ],
)
def test_native_edit_paths_are_extracted_without_transcript(call, expected):
    assert edited_paths(call) == expected


def test_pre_edit_policy_returns_bounded_impact_as_advisory(monkeypatch):
    policy = SpecGraphEditPolicy(max_nodes=7)
    seen = []

    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        return ({"status": "HIT", "impacted_nodes": ["SPEC-1"]}, None)

    monkeypatch.setattr(policy, "_run_json", fake_run)
    result = policy.evaluate(
        ToolCall("claude", "PreToolUse", "Edit", {"file_path": "src/a.py"}, root="C:/repo")
    )

    assert result.decision.value == "allow"
    assert "SpecGraph pre-edit impact" in result.additional_context
    assert seen[0][0][-4:] == ["--max-nodes", "7", "--format", "json"]
    assert 0 < seen[0][1]["timeout_s"] <= 3.0


def test_multi_path_pre_edit_uses_one_total_hook_budget(monkeypatch):
    clock = iter((0.0, 0.0, 3.1))
    policy = SpecGraphEditPolicy(timeout_s=3.0, monotonic=lambda: next(clock))
    seen = []

    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        return ({"status": "HIT", "impacted_nodes": []}, None)

    monkeypatch.setattr(policy, "_run_json", fake_run)
    result = policy.evaluate(
        ToolCall(
            "codex",
            "PreToolUse",
            "apply_patch",
            {
                "command": (
                    "*** Begin Patch\n"
                    "*** Update File: src/a.py\n"
                    "*** Update File: src/b.py\n"
                    "*** End Patch"
                )
            },
            root="C:/repo",
        )
    )

    assert result.decision.value == "allow"
    assert len(seen) == 1
    assert seen[0][1]["timeout_s"] == 3.0
    assert '"query":"src/b.py","status":"TIMEOUT"' in result.additional_context


def test_post_edit_policy_skips_failed_agy_edit_and_reports_success(monkeypatch):
    policy = SpecGraphEditPolicy()
    monkeypatch.setattr(
        policy,
        "_run_json",
        lambda command: ({"status": "STALE", "impacted_nodes": ["SPEC-1"]}, None),
    )
    successful = policy.evaluate(
        ToolCall(
            "agy",
            "PostToolUse",
            "write_to_file",
            {"TargetFile": "src/a.py"},
            root="C:/repo",
        )
    )
    failed = policy.evaluate(
        ToolCall(
            "agy",
            "PostToolUse",
            "write_to_file",
            {"TargetFile": "src/a.py"},
            root="C:/repo",
            succeeded=False,
        )
    )

    assert "STALE" in successful.additional_context
    assert failed.additional_context == ""


def test_agy_successful_post_event_is_explicit_not_inferred(monkeypatch):
    monkeypatch.setattr(
        SpecGraphEditPolicy,
        "_run_json",
        lambda self, command: ({"status": "STALE", "impacted_nodes": []}, None),
    )
    payload = {
        "toolCall": {"name": "write_to_file", "args": {"TargetFile": "src/a.py"}},
        "workspacePaths": ["C:/repo"],
    }

    output = dispatch("agy", json.dumps(payload).encode(), event_hint="post")

    assert output == {}


def test_agy_malformed_post_event_keeps_required_empty_output():
    output = dispatch("agy", b"not-json", event_hint="post")

    assert output == {}


@pytest.mark.parametrize(("vendor", "tool"), [("claude", "Agent"), ("claude", "Task"), ("codex", "spawn_agent")])
def test_frontier_subagents_are_denied_while_local_models_answer(vendor, tool, monkeypatch):
    from compute_plane import tool_hooks
    monkeypatch.setattr(tool_hooks, "local_models_up", lambda *a, **k: True)
    monkeypatch.delenv("TARE_LOCAL_MODELS_BYPASS", raising=False)
    payload = {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": {"prompt": "map the repo"}}
    output = dispatch(vendor, json.dumps(payload).encode())
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "tare-ops explore" in output["hookSpecificOutput"]["permissionDecisionReason"]


def test_subagents_pass_when_local_models_are_down_or_bypassed(monkeypatch):
    from compute_plane import tool_hooks
    payload = {"hook_event_name": "PreToolUse", "tool_name": "Agent", "tool_input": {"prompt": "x"}}
    monkeypatch.setattr(tool_hooks, "local_models_up", lambda *a, **k: False)
    assert dispatch("claude", json.dumps(payload).encode()) == {}
    monkeypatch.setattr(tool_hooks, "local_models_up", lambda *a, **k: True)
    monkeypatch.setenv("TARE_LOCAL_MODELS_BYPASS", "1")
    assert dispatch("claude", json.dumps(payload).encode()) == {}
    # Ordinary tools never trigger the probe.
    monkeypatch.delenv("TARE_LOCAL_MODELS_BYPASS")
    monkeypatch.setattr(tool_hooks, "local_models_up", lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed")))
    assert dispatch("claude", json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Read", "tool_input": {"file_path": "a"}}).encode()) == {}


def test_local_models_probe_is_bounded_and_fail_open():
    from compute_plane.tool_hooks import local_models_up
    assert local_models_up("http://127.0.0.1:9", timeout=0.2) is False


@pytest.mark.parametrize(("explicit", "configured", "expected"), [
    (None, None, "http://127.0.0.1:8080/health"),
    (None, "http://gateway.example.invalid:8080/", "http://gateway.example.invalid:8080/health"),
    ("http://explicit.example.invalid/", "http://other.example.invalid", "http://explicit.example.invalid/health"),
])
def test_gateway_precedence_preserves_live_delegation_denial(explicit, configured, expected, monkeypatch):
    from compute_plane.tool_hooks import LocalModelsFirstGate, ToolDecision, local_models_up
    from unittest.mock import MagicMock
    import urllib.request

    monkeypatch.delenv("TARE_LOCAL_MODELS_BYPASS", raising=False)
    if configured is None:
        monkeypatch.delenv("TARE_LOCAL_MODELS", raising=False)
    else:
        monkeypatch.setenv("TARE_LOCAL_MODELS", configured)
    response = MagicMock()
    response.__enter__.return_value.status = 200
    probe = MagicMock(return_value=response)
    monkeypatch.setattr(urllib.request, "urlopen", probe)
    call = ToolCall("codex", "PreToolUse", "spawn_agent", {"prompt": "inspect source"})
    result = LocalModelsFirstGate(probe=lambda: local_models_up(explicit, timeout=0.25)).evaluate(call)
    assert result.decision is ToolDecision.DENY
    probe.assert_called_once_with(expected, timeout=0.25)
