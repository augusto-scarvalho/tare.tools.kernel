"""Thin lifecycle codecs; no Work selection, persistence or stopping policy."""
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class SessionEvent:
    vendor: str
    phase: str
    root: str
    session_id: str | None
    resumed: bool = False


def adapt_session(vendor, payload, *, event_hint=None):
    if not isinstance(vendor, str) or vendor not in {"claude", "codex", "agy", "aider", "http"} or not isinstance(payload, Mapping):
        raise ValueError("invalid_session_payload")
    if vendor == "agy":
        if not isinstance(event_hint, str) or event_hint not in {"start", "stop", "end"}:
            raise ValueError("explicit_agy_lifecycle_event_required")
        phase = event_hint
        roots = payload.get("workspacePaths")
        if not isinstance(roots, list) or len(roots) != 1:
            raise ValueError("single_workspace_required")
        root = roots[0]
    else:
        if event_hint is not None:
            raise ValueError("event_hint_only_for_agy")
        name = payload.get('hook_event_name')
        phase = {"SessionStart": "start", "Stop": "stop", "SessionEnd": "end"}.get(name) if isinstance(name, str) else None
        root = payload.get("cwd")
    if phase is None or not isinstance(root, str) or not root.strip():
        raise ValueError("invalid_lifecycle_event_or_root")
    session = payload.get("session_id")
    # IDs are opaque observations, never filesystem paths or authority tokens.
    if session is not None and (not isinstance(session, str) or not session.strip() or len(session) > 256):
        raise ValueError("invalid_session_id")
    return SessionEvent(vendor, phase, root, session, payload.get("source") == "resume")


def render_session(event, context):
    """Stop/end are notifications, never a request to continue or deny exit."""
    if event.phase != "start":
        return {}
    if event.vendor == "agy":
        return {"injectSteps": [{"ephemeralMessage": context}]}
    if event.vendor in {"aider", "http"}:
        return {"context": context}
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}}
