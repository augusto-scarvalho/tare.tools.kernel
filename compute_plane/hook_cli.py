"""Bounded stdin/stdout leaf process for vendor-native PreToolUse hooks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from typing import Any

from .tool_hooks import (
    LocalModelsFirstGate,
    SPECGRAPH_FIRST_REASON,
    SpecGraphFirstGate,
    SpecGraphEditPolicy,
    ToolDecision,
    ToolGateResult,
    adapt_agy,
    adapt_claude,
    adapt_codex,
    adapt_aider,
    render_aider,
    render_agy,
    render_claude,
    render_codex,
    cached_local_models_up,
)

MAX_INPUT_BYTES = 65_536

Adapter = Callable[[Mapping[str, Any]], Any]
Renderer = Callable[[ToolGateResult], dict[str, Any]]

_ADAPTERS: dict[str, tuple[Adapter, Renderer]] = {
    "aider": (adapt_aider, render_aider),
    "agy": (adapt_agy, render_agy),
    "claude": (adapt_claude, render_claude),
    "codex": (adapt_codex, render_codex),
}


def _deny_reason(detail: str) -> str:
    return f"Hook input rejected: {detail}. {SPECGRAPH_FIRST_REASON}"


def dispatch(vendor: str, raw: bytes, *, event_hint: str | None = None) -> dict[str, Any]:
    adapter, renderer = _ADAPTERS[vendor]
    call = None
    try:
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError(f"input exceeds {MAX_INPUT_BYTES} bytes")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("root payload must be an object")
        if vendor == "agy":
            event = {"pre": "PreToolUse", "post": "PostToolUse"}.get(event_hint or "")
            call = adapt_agy(payload, event=event)
        else:
            call = adapter(payload)
        from .project_context_hooks import ScopedProjectContextPolicy, scoped_delegation_root
        scoped_root = scoped_delegation_root(call, dict(payload))
        result = LocalModelsFirstGate(
            probe=(lambda: cached_local_models_up(scoped_root)) if scoped_root else None
        ).evaluate(call)
        if result.decision is ToolDecision.ALLOW:
            scoped = ScopedProjectContextPolicy().evaluate(call, dict(payload))
            if scoped is not None:
                result = scoped
            else:
                result = SpecGraphFirstGate().evaluate(call)
                if result.decision is ToolDecision.ALLOW:
                    result = SpecGraphEditPolicy().evaluate(call)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        result = ToolGateResult(ToolDecision.DENY, _deny_reason(str(exc)))
    if call is None and vendor == "agy" and event_hint == "post":
        return {}
    if call is None:
        return renderer(result)
    if call.event == "PostToolUse" and vendor == "agy":
        return {}
    rendered = renderer(result)
    if call.event == "PostToolUse" and vendor in {"claude", "codex"}:
        specific = rendered.get("hookSpecificOutput")
        if isinstance(specific, dict):
            specific["hookEventName"] = "PostToolUse"
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tare-hook")
    parser.add_argument("vendor", choices=sorted(_ADAPTERS))
    parser.add_argument("--event", choices=["pre", "post"])
    args = parser.parse_args(argv)

    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if args.event and args.vendor != "agy":
        parser.error("--event is only required for Antigravity")
    output = dispatch(args.vendor, raw, event_hint=args.event)
    sys.stdout.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
