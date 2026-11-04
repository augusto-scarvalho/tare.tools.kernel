"""Opt-in snapshot advice for native tool hooks, without synchronous index work."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
from typing import Any

from .tool_hooks import ToolCall, ToolDecision, ToolGateResult, edited_paths


def configured_context(path: Path) -> tuple[Path, dict[str, Any]] | None:
    """Use a bounded ancestor walk to a declarative SpecGraph owner marker.

    This checks known config filenames, never scans directories or guesses roots
    from Git/project names. The nearest marker wins, including a disabled one.
    """
    candidate = path.resolve()
    if not candidate.is_dir():
        candidate = candidate.parent
    for root in [candidate, *list(candidate.parents)[:8]]:
        for name in ("specgraph.yaml", ".specgraph.yaml"):
            marker = root / name
            if not marker.is_file():
                continue
            try:
                import yaml
                with marker.open("rb") as stream:
                    raw = stream.read(16385)
                if len(raw) > 16384:
                    return None
                try:
                    configuration = yaml.safe_load(raw)
                except yaml.YAMLError:
                    return None
                setting = configuration.get("hook_context") if isinstance(configuration, dict) else None
                if not isinstance(setting, dict) or setting.get("enabled") is not True:
                    return None
                if not isinstance(setting.get("role", "implementer"), str) or setting.get("role", "implementer") not in {"planner", "implementer", "auditor", "explorer"}:
                    return None
                automatic = setting.get("automatic_roles", ["planner", "implementer", "auditor", "explorer"])
                if not isinstance(automatic, list) or len(automatic) > 4 or any(
                        not isinstance(role, str) or role not in {"planner", "implementer", "auditor", "explorer"}
                        for role in automatic):
                    return None
                return root, setting
            except (ImportError, OSError, UnicodeError, ValueError):
                return None
    return None


def _query(call: ToolCall) -> str | None:
    if call.name.casefold() in {"grep", "grep_search"}:
        query = call.arguments.get("pattern") or call.arguments.get("Query")
        return query if isinstance(query, str) else None
    command = call.arguments.get("command") or call.arguments.get("cmd") or call.arguments.get("CommandLine")
    if not isinstance(command, str) or len(command) > 2048:
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if not words or Path(words[0]).name.lower() not in {"rg", "grep", "ripgrep", "rg.exe"}:
        return None
    skip = False
    for word in words[1:]:
        if skip:
            skip = False
            continue
        if word == "--files":
            return None
        if word in {"-g", "--glob", "-t", "--type", "--max-count", "-m"}:
            skip = True
        elif word in {"-e", "--regexp", "--"}:
            continue
        elif not word.startswith("-"):
            return word
    return None


def _receipt(root: Path, session: str | None) -> Path | None:
    if not session or not isinstance(session, str) or len(session) > 256:
        return None
    state_root = Path(os.environ.get("TARE_HOOK_STATE_DIR", str(Path.home() / ".local/state/tare-hooks")))
    state_root = state_root.expanduser().resolve()
    if state_root.is_relative_to(root):
        return None  # Advisory state never modifies the indexed candidate.
    key = hashlib.sha256((str(root) + "\0" + session).encode()).hexdigest()
    return state_root / "project-context" / (key + ".json")


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            raw = stream.read(16385)
        value = json.loads(raw) if len(raw) <= 16384 else None
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(raw.encode()) > 16384:
            return
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(raw, encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass  # Missing dedup never blocks a tool or changes Work state.


def _event_id(call: ToolCall, payload: dict[str, Any]) -> str | None:
    """Only a host-supplied call identity permits retry deduplication."""
    tool_call = payload.get("toolCall")
    native_id = payload.get("tool_use_id") or payload.get("tool_call_id")
    if native_id is None and isinstance(tool_call, dict):
        native_id = tool_call.get("id")
    if not isinstance(native_id, str) or not native_id:
        return None
    namespace = json.dumps([call.vendor, payload.get("session_id"), native_id], separators=(",", ":"))
    return "kernel:" + hashlib.sha256(namespace.encode("utf-8")).hexdigest()


def _record_changes(root: Path, paths: list[str], *, source: str,
                    event_id: str | None, overflow: bool) -> bool | None:
    """Append bounded observations; indexing is exclusively a worker action."""
    try:
        from specgraph.lifecycle import record_changes
        result = record_changes(root, paths, source=source, event_id=event_id,
                                overflow=overflow, timeout_ms=20)
        return result.get("status") in {"RECORDED", "COALESCED"}
    except ImportError:
        return None  # Older SpecGraph installs retain legacy observations until upgrade.
    except (OSError, ValueError, TypeError, sqlite3.DatabaseError):
        return False


def _delivery_marker(root: Path) -> Path | None:
    receipt = _receipt(root, "kernel-unconfirmed-delivery")
    return receipt.with_suffix(".delivery.json") if receipt else None


def _session_scope(call: ToolCall, payload: dict[str, Any], role: str) -> str | None:
    session = payload.get("session_id")
    if not isinstance(session, str) or not session or len(session) > 256:
        return None
    # Resuming or changing agent/role starts a separate bounded context identity.
    # No role is inferred from prompt text or a vendor/model name.
    agent_id = payload.get("agent_id")
    agent_id = agent_id if isinstance(agent_id, str) and len(agent_id) <= 256 else None
    identity = [call.vendor, session, agent_id, role, payload.get("source") == "resume"]
    return "scope:" + hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def _observe(call: ToolCall, payload: dict[str, Any], root: Path, role: str,
             outcome: str, report: dict[str, Any] | None = None, *, reason: str | None = None) -> None:
    from .hook_observations import record_hook_observation
    report = report or {}
    native_tool = payload.get("tool_name")
    if not isinstance(native_tool, str):
        native = payload.get("toolCall")
        native_tool = native.get("name") if isinstance(native, dict) else call.name
    record_hook_observation(root, host=call.vendor, event=call.event, tool=call.name,
        native_tool=native_tool if isinstance(native_tool, str) else call.name,
        session_id=payload.get("session_id"), agent_id=payload.get("agent_id"), role=role,
        outcome=outcome, snapshot=report.get("snapshot"), digest=report.get("digest"),
        output_bytes=len(json.dumps(report, ensure_ascii=False, separators=(",", ":")).encode()) if report else 0,
        reason=reason)


def scoped_delegation_root(call: ToolCall, payload: dict[str, Any]) -> Path | None:
    """Determine opt-in scope without inspecting prompts or probing a network."""
    if call.name.casefold() not in {"agent", "task", "spawn_agent", "spawn_agents"}:
        return None
    explicit = isinstance(payload.get("cwd"), str) or isinstance(call.arguments.get("workdir"), str)
    roots = payload.get("workspacePaths")
    explicit = explicit or isinstance(roots, list) and len(roots) == 1
    if not explicit or not call.root:
        return None
    scope = configured_context(Path(call.root))
    return scope[0] if scope else None


class ScopedProjectContextPolicy:
    """Return None for legacy policy, or fail-open advice for enabled roots.

    Enabling hook_context explicitly replaces unconditional search blocking and
    synchronous impact/staleness scans for that declarative repository scope.
    The legacy strict impact API remains separate and unchanged.
    """

    def evaluate(self, call: ToolCall, payload: dict[str, Any]) -> ToolGateResult | None:
        extracted_paths = edited_paths(call, max_paths=65)
        path_overflow = len(extracted_paths) > 64
        paths = extracted_paths[:64]
        if not paths:
            value = call.arguments.get("file_path") or call.arguments.get("TargetFile")
            paths = (value,) if isinstance(value, str) and value else ()
        native_roots = payload.get("workspacePaths")
        explicit_root = (isinstance(payload.get("cwd"), str)
                         or isinstance(call.arguments.get("workdir"), str)
                         or isinstance(native_roots, list) and len(native_roots) == 1)
        cwd = Path(call.root).resolve() if call.root and explicit_root else None
        selected: dict[Path, tuple[dict[str, Any], list[str]]] = {}
        scope_cache: dict[Path, tuple[Path, dict[str, Any]] | None] = {}
        for value in paths:
            path = Path(value)
            if not path.is_absolute() and cwd:
                path = cwd / path
            if not path.is_absolute():
                continue
            resolved = path.resolve()
            scope_key = resolved if resolved.is_dir() else resolved.parent
            if scope_key not in scope_cache:
                scope_cache[scope_key] = configured_context(resolved)
            scope = scope_cache[scope_key]
            if scope:
                root, setting = scope
                selected.setdefault(root, (setting, []))[1].append(str(path))
        if not selected and cwd:
            scope = configured_context(cwd)
            if scope:
                if paths:
                    # An absolute path with no configured owner must not be
                    # attached to the host cwd's unrelated project journal.
                    return ToolGateResult(ToolDecision.ALLOW)
                selected[scope[0]] = (scope[1], [])
        if not selected:
            return None
        # The worker performs periodic full reconciliation even when a host
        # drops events or this callback cannot inspect every root/path.
        root_overflow = len(selected) > 4
        selected = dict(list(selected.items())[:4])
        if call.name.casefold() in {"bash", "run_command"}:
            command = call.arguments.get("command") if call.name.casefold() == "bash" else call.arguments.get("CommandLine")
            if not isinstance(command, str):
                return ToolGateResult(ToolDecision.DENY, "Shell tool command must be a string.")
        query = _query(call)
        delegation = call.name.casefold() in {"agent", "task", "spawn_agent", "spawn_agents"}
        if not paths and not query and not delegation:
            return ToolGateResult(ToolDecision.ALLOW)  # No repeated project banner for unrelated tools.
        reports = []
        output_receipts = []
        from .hook_observations import read_context_receipt, write_context_receipt
        for root, (setting, file_paths) in sorted(selected.items(), key=lambda pair: str(pair[0])):
            role = payload.get("agent_role", setting.get("role", "implementer"))
            if not isinstance(role, str) or role not in {"planner", "implementer", "auditor", "explorer"}:
                _observe(call, payload, root, "unknown", "UNAVAILABLE")
                continue
            context_scope = _session_scope(call, payload, role)
            state = read_context_receipt(root, context_scope)
            legacy_path = _receipt(root, payload.get("session_id"))
            legacy = _load(legacy_path)
            if not state.get("pending_paths") and legacy.get("pending_paths"):
                state["pending_paths"] = legacy["pending_paths"]
                state["delta_overflow"] = legacy.get("delta_overflow", False)
            pending = state.get("pending_paths", [])
            pending = pending if isinstance(pending, list) and all(isinstance(v, str) for v in pending) else []
            delivery_marker = _delivery_marker(root)
            if call.event == "PostToolUse":
                if call.succeeded and extracted_paths:
                    recorded = _record_changes(root, file_paths, source=f"{call.vendor}:PostToolUse",
                                               event_id=_event_id(call, payload),
                                               overflow=path_overflow or root_overflow)
                    if recorded is None:
                        legacy_pending = sorted(set(pending + file_paths))
                        _save(legacy_path, legacy | {"pending_paths": legacy_pending[:64],
                            "delta_overflow": path_overflow or root_overflow or len(legacy_pending) > 64})
                    elif not recorded:
                        # Delivery failure is conservatively sticky. Do not
                        # clear it merely because a different snapshot appears;
                        # a concurrent edit may have failed after reconciliation.
                        _save(delivery_marker, {"status": "UNCONFIRMED", "authority": "NONE",
                                                "reason": "hook event could not be durably recorded"})
                    _observe(call, payload, root, role, "EVENT_RECORDED" if recorded else "EVENT_UNCONFIRMED")
                elif not call.succeeded:
                    _observe(call, payload, root, role, "TOOL_FAILED")
                continue  # No scan, indexing, reconciliation or stop loop.
            if pending:
                legacy_id = "kernel-legacy:" + hashlib.sha256(json.dumps(
                    [str(root), payload.get("session_id"), sorted(pending)], separators=(",", ":")
                ).encode()).hexdigest()
                if _record_changes(root, pending[:64], source="kernel:legacy-pending",
                                   event_id=legacy_id, overflow=bool(state.get("delta_overflow")) or len(pending) > 64):
                    # New Post callbacks do not modify this legacy field;
                    # their independent journal sequence cannot be cleared here.
                    pending = []
                    state = {key: value for key, value in state.items()
                             if key not in {"pending_paths", "delta_overflow"}}
                    write_context_receipt(root, context_scope, state)
                    if legacy.get("pending_paths"):
                        _save(legacy_path, {key: value for key, value in legacy.items()
                                           if key not in {"pending_paths", "delta_overflow"}})
            delivery_uncertain = bool(_load(delivery_marker))
            automatic = setting.get("automatic_roles", ["planner", "implementer", "auditor", "explorer"])
            demand = query or payload.get("spec_id") or payload.get("symbol_id")
            if role not in automatic and not demand and not delivery_uncertain:
                _observe(call, payload, root, role, "SILENT", reason="ON_DEMAND_POLICY")
                continue
            try:
                from specgraph.hook_context import context_hint
                requested = setting.get("max_bytes", 4096)
                budget = min(4096, max(512, int(requested)))
                budget = max(512, (budget - 128) // len(selected))
                report = context_hint(
                    root, role=role,
                    event="delegate" if delegation else "pre",
                    files=tuple(file_paths[:16]), spec_id=payload.get("spec_id", setting.get("spec_id")),
                    symbol_id=payload.get("symbol_id", setting.get("symbol_id")), work_id=setting.get("work_id"),
                    source_ref=setting.get("source_ref"), query=query,
                    previous_digest=None if delivery_uncertain else state.get("digest"), dirty_paths=tuple(pending[:64]),
                    max_bytes=budget, timeout_ms=min(1000, max(1, int(setting.get("timeout_ms", 50)))))
            except (ImportError, ValueError, OSError, TypeError):
                _observe(call, payload, root, role, "UNAVAILABLE")
                continue  # Unavailable substitute never denies bounded direct recovery.
            if report.get("status") in {"SILENT", "UNAVAILABLE", "TIMEOUT", "MISS"}:
                _observe(call, payload, root, role, report["status"], report)
                continue
            if delivery_uncertain:
                report["hook_delivery"] = "UNCONFIRMED; durable event recording failed"
                # Keep failure advice deduplicable without folding it into
                # SpecGraph's snapshot/freshness authority.
                digest = hashlib.sha256(json.dumps({
                    "context": report.get("digest", report.get("selection_digest")),
                    "delivery": report["hook_delivery"],
                }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if digest == state.get("delivery_digest"):
                    _observe(call, payload, root, role, "SILENT", report)
                    continue
                state["delivery_digest"] = digest
            if len(file_paths) > 16 or path_overflow or root_overflow:
                report["hook_selection"] = {"selected_files": min(16, len(file_paths)),
                                            "observed_files": len(file_paths), "truncated": True}
            output_receipts.append((root, role, context_scope, state, pending, report))
            reports.append(report)
        if not reports:
            return ToolGateResult(ToolDecision.ALLOW)
        context = json.dumps(reports, ensure_ascii=False, separators=(",", ":"))
        if len(context.encode()) > 4096:
            for root, role, *_ in output_receipts:
                _observe(call, payload, root, role, "BUDGET_REQUIRED")
            return ToolGateResult(ToolDecision.ALLOW)
        for root, role, context_scope, state, pending, report in output_receipts:
            if report.get("digest"):
                write_context_receipt(root, context_scope, state | {"digest": report["digest"],
                    "snapshot": report.get("snapshot"), "pending_paths": pending})
            _observe(call, payload, root, role, "CONTEXT_RETURNED", report)
        return ToolGateResult(ToolDecision.ALLOW, additional_context=context)
