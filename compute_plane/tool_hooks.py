"""Canonical mediated-tool policy and thin vendor hook adapters."""

from __future__ import annotations

import re
import json
import subprocess
import time
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ToolDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class ToolCall:
    vendor: str
    event: str
    name: str
    arguments: Mapping[str, Any]
    root: str | None = None
    succeeded: bool = True


@dataclass(frozen=True)
class ToolGateResult:
    decision: ToolDecision
    reason: str = ""
    additional_context: str = ""


SPECGRAPH_FIRST_REASON = (
    "Repository search must use `specgraph resolve <query> --format json`. It tries "
    "structural lookup first and permits bounded traditional recovery only for "
    "MISS, STALE, TIMEOUT, or UNAVAILABLE, then validates and reuses the evidence."
)

_SHELL_SEARCH = re.compile(
    r"(?<![\w.-])(?:git\s+grep|rg|ripgrep|u?grep|[ef]grep|ag|ack|findstr|select-string|sls)(?:\.exe)?(?=\s|$)",
    re.IGNORECASE,
)
_SHELL_SEGMENT = re.compile(r"&&|\|\||[;|\r\n]")
LOCAL_MODELS_URL = "http://127.0.0.1:8080"
_SUBAGENT_TOOLS = {"agent", "task", "spawn_agent", "spawn_agents"}
LOCAL_MODELS_FIRST_REASON = (
    "The local model server is up. Repository, documentation and web exploration must be "
    "delegated to it, not to a frontier subagent: `tare-ops explore --root <repo> --question ... "
    "[--focus path:start-end]`, `tare-ops evidence ...`, `tare-ops research --question ... --seed URL "
    "--domain D --max-pages N --max-depth D`. See tare.tools.os/docs/operations/DELEGATION_POLICY.md. "
    "Set TARE_LOCAL_MODELS_BYPASS=1 only on the user's explicit decision."
)


def local_models_up(url: str | None = None, timeout: float = 1.5) -> bool:
    """One bounded probe; an unreachable server means the gate stays open (fail-open by design)."""
    import os
    import urllib.request
    base = (url or os.environ.get("TARE_LOCAL_MODELS") or LOCAL_MODELS_URL).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def cached_local_models_up(root: Path, *, max_age_seconds: float = 30.0) -> bool:
    """Opt-in scoped hooks consume only an explicitly configured health receipt.

    A separate observer may publish schema/status/observed_at outside the repo.
    Missing, stale or malformed observations are UNKNOWN and fail open.
    """
    import os
    configured = os.environ.get("TARE_LOCAL_MODELS_HEALTH_CACHE")
    if not configured:
        return False
    path = Path(configured).expanduser()
    if not path.is_absolute() or path.resolve().is_relative_to(root.resolve()):
        return False
    try:
        with path.open("rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            return False
        value = json.loads(raw)
        observed = value.get("observed_at") if isinstance(value, dict) else None
        if type(observed) not in (float, int) or not math.isfinite(observed):
            return False
        age = time.time() - observed
        return (value.get("schema") == "tare.tools/local-models-health/1"
                and value.get("status") == "UP" and 0 <= age <= max_age_seconds)
    except (OSError, ValueError, TypeError):
        return False


class LocalModelsFirstGate:
    """Deny frontier subagent spawns while the local model server answers its health probe."""

    def __init__(self, probe=None):
        self.probe = probe  # None resolves the module-level probe at call time (patchable)

    def evaluate(self, call: ToolCall) -> ToolGateResult:
        import os
        if call.event != "PreToolUse" or call.name.casefold() not in _SUBAGENT_TOOLS:
            return ToolGateResult(ToolDecision.ALLOW)
        if os.environ.get("TARE_LOCAL_MODELS_BYPASS") == "1":
            return ToolGateResult(ToolDecision.ALLOW)
        if (self.probe or local_models_up)():
            return ToolGateResult(ToolDecision.DENY, LOCAL_MODELS_FIRST_REASON)
        return ToolGateResult(ToolDecision.ALLOW)


class SpecGraphFirstGate:
    """Require structural searches to use the indexed SpecGraph surface."""

    def evaluate(self, call: ToolCall) -> ToolGateResult:
        name = call.name.casefold()

        if name == "grep":
            return ToolGateResult(ToolDecision.DENY, SPECGRAPH_FIRST_REASON)

        if name == "grep_search":
            return ToolGateResult(ToolDecision.DENY, SPECGRAPH_FIRST_REASON)

        command = _shell_command(name, call.arguments)
        if name in {"bash", "run_command"} and command is None:
            return ToolGateResult(ToolDecision.DENY, "Shell tool command must be a string.")
        if command is None:
            return ToolGateResult(ToolDecision.ALLOW)
        for segment in _SHELL_SEGMENT.split(command):
            if _SHELL_SEARCH.search(segment):
                return ToolGateResult(ToolDecision.DENY, SPECGRAPH_FIRST_REASON)
        return ToolGateResult(ToolDecision.ALLOW)


_EDIT_PATH_KEYS: dict[str, str] = {
    "edit": "file_path",
    "write": "file_path",
    "write_to_file": "TargetFile",
    "replace_file_content": "TargetFile",
    "multi_replace_file_content": "TargetFile",
}
_PATCH_PATH = re.compile(
    r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE
)


def edited_paths(call: ToolCall, *, max_paths: int = 16) -> tuple[str, ...]:
    """Extract bounded native-edit paths without inspecting transcripts."""

    name = call.name.casefold()
    key = _EDIT_PATH_KEYS.get(name)
    if key:
        value = call.arguments.get(key)
        return (value,) if isinstance(value, str) and value.strip() else ()
    if name == "apply_patch":
        command = call.arguments.get("command")
        if not isinstance(command, str):
            return ()
        paths = [match.strip() for match in _PATCH_PATH.findall(command) if match.strip()]
        return tuple(dict.fromkeys(paths))[:max_paths]
    return ()


class SpecGraphEditPolicy:
    """Provide pre-edit impact and post-edit staleness from SpecGraph CLI data."""

    def __init__(
        self,
        *,
        timeout_s: float = 3.0,
        max_nodes: int = 50,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_nodes = max_nodes
        self._monotonic = monotonic

    def evaluate(self, call: ToolCall) -> ToolGateResult:
        paths = edited_paths(call)
        if not paths:
            return ToolGateResult(ToolDecision.ALLOW)
        if not call.root:
            return ToolGateResult(
                ToolDecision.ALLOW,
                additional_context="SpecGraph edit preview unavailable: hook payload has no repository root.",
            )
        if call.event == "PreToolUse":
            return self._pre_edit(call.root, paths)
        if call.event == "PostToolUse" and call.succeeded:
            return self._post_edit(call.root, paths)
        return ToolGateResult(ToolDecision.ALLOW)

    def _run_json(
        self,
        command: list[str],
        *,
        timeout_s: float | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=self.timeout_s if timeout_s is None else max(0.001, timeout_s),
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, type(exc).__name__
        if len(completed.stdout.encode("utf-8")) > 65_536:
            return None, "oversized output"
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return None, "invalid JSON"
        return (payload, None) if isinstance(payload, dict) else (None, "non-object JSON")

    def _pre_edit(self, root: str, paths: tuple[str, ...]) -> ToolGateResult:
        reports: list[dict[str, Any]] = []
        deadline = self._monotonic() + self.timeout_s
        for path in paths:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                reports.append(
                    {
                        "query": path,
                        "status": "TIMEOUT",
                        "reason": "hook budget exhausted",
                    }
                )
                continue
            payload, error = self._run_json(
                [
                    "specgraph",
                    "--root",
                    root,
                    "impact",
                    path,
                    "--max-nodes",
                    str(self.max_nodes),
                    "--format",
                    "json",
                ],
                timeout_s=remaining,
            )
            if error:
                reports.append({"query": path, "status": "UNAVAILABLE", "reason": error})
            elif payload is not None:
                reports.append(payload)
        compact = _bounded_context(
            json.dumps(reports, ensure_ascii=False, separators=(",", ":"))
        )
        return ToolGateResult(
            ToolDecision.ALLOW,
            additional_context=f"SpecGraph pre-edit impact: {compact}",
        )

    def _post_edit(self, root: str, paths: tuple[str, ...]) -> ToolGateResult:
        command = ["specgraph", "--root", root, "staleness"]
        for path in paths:
            command.extend(["--changed-path", path])
        command.extend(["--max-nodes", str(self.max_nodes), "--format", "json"])
        payload, error = self._run_json(command)
        if error:
            context = f"SpecGraph post-edit staleness unavailable: {error}."
        else:
            compact = _bounded_context(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
            context = (
                f"SpecGraph post-edit staleness for {json.dumps(paths)}: {compact}. "
                "Reconcile the index after the edit set is complete."
            )
        return ToolGateResult(ToolDecision.ALLOW, additional_context=context)


def _bounded_context(value: str, max_bytes: int = 32_768) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = "...[hook context truncated]"
    budget = max_bytes - len(suffix.encode("utf-8"))
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


def _shell_command(name: str, arguments: Mapping[str, Any]) -> str | None:
    if name == "bash":
        value = arguments.get("command")
    elif name == "run_command":
        value = arguments.get("CommandLine")
    else:
        return None
    return value if isinstance(value, str) else None


def adapt_claude(payload: Mapping[str, Any]) -> ToolCall:
    return _adapt_claude_codex(payload, vendor="claude")


def adapt_codex(payload: Mapping[str, Any]) -> ToolCall:
    return _adapt_claude_codex(payload, vendor="codex")


def adapt_aider(payload: Mapping[str, Any]) -> ToolCall:
    return _adapt_claude_codex(payload, vendor="aider")


def render_aider(result: ToolGateResult) -> dict[str, Any]:
    return {"decision": result.decision.value, "reason": result.reason,
            "context": result.additional_context}


def _adapt_claude_codex(payload: Mapping[str, Any], *, vendor: str) -> ToolCall:
    event = payload.get("hook_event_name")
    if event not in {"PreToolUse", "PostToolUse"}:
        raise ValueError("expected PreToolUse or PostToolUse event")
    name = payload.get("tool_name")
    arguments = payload.get("tool_input")
    if not isinstance(name, str) or not name:
        raise ValueError("tool_name must be a non-empty string")
    leaf = name.rsplit(".", 1)[-1]
    if leaf.casefold() == "apply_patch" and isinstance(arguments, str):
        arguments = {"command": arguments}
    if not isinstance(arguments, Mapping):
        raise ValueError("tool_input must be an object")
    root = payload.get("cwd")
    if leaf.casefold() == "exec_command":
        arguments = dict(arguments)
        arguments["command"] = arguments.get("cmd")
        name = "Bash"
        if isinstance(arguments.get("workdir"), str) and Path(arguments["workdir"]).is_absolute():
            root = arguments["workdir"]
    elif leaf.casefold() == "apply_patch":
        name = "apply_patch"
        arguments = dict(arguments)
        arguments["command"] = arguments.get("command", arguments.get("patch", arguments.get("input")))
    elif leaf.casefold() in _SUBAGENT_TOOLS:
        name = leaf
    response = payload.get("tool_response")
    succeeded = not (isinstance(response, Mapping) and (response.get("isError") or response.get("error")))
    return ToolCall(
        vendor=vendor,
        event=event,
        name=name,
        arguments=arguments,
        root=root if isinstance(root, str) and root else str(Path.cwd()),
        succeeded=succeeded,
    )


def adapt_agy(payload: Mapping[str, Any], *, event: str | None = None) -> ToolCall:
    tool_call = payload.get("toolCall")
    if not isinstance(tool_call, Mapping):
        raise ValueError("toolCall must be an object")
    name = tool_call.get("name")
    arguments = tool_call.get("args")
    if not isinstance(name, str) or not name:
        raise ValueError("toolCall.name must be a non-empty string")
    if not isinstance(arguments, Mapping):
        raise ValueError("toolCall.args must be an object")
    error = payload.get("error")
    if event is None:
        event = "PostToolUse" if "error" in payload else "PreToolUse"
    if event not in {"PreToolUse", "PostToolUse"}:
        raise ValueError("invalid Antigravity lifecycle event")
    workspace_paths = payload.get("workspacePaths")
    root = None
    if isinstance(workspace_paths, list) and len(workspace_paths) == 1:
        candidate = workspace_paths[0]
        root = candidate if isinstance(candidate, str) and candidate else None
    return ToolCall(
        vendor="agy",
        event=event,
        name=name,
        arguments=arguments,
        root=root or str(Path.cwd()),
        succeeded=not isinstance(error, str) or not error,
    )


def render_claude(result: ToolGateResult) -> dict[str, Any]:
    return _render_claude_codex(result)


def render_codex(result: ToolGateResult) -> dict[str, Any]:
    return _render_claude_codex(result)


def _render_claude_codex(result: ToolGateResult) -> dict[str, Any]:
    if result.decision is ToolDecision.ALLOW and not result.additional_context:
        return {}
    if result.decision is ToolDecision.ALLOW:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": result.additional_context,
            }
        }
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": result.reason,
        }
    }


def render_agy(result: ToolGateResult) -> dict[str, Any]:
    return {
        "decision": result.decision.value,
        "reason": result.reason or result.additional_context,
    }
