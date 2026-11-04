"""Bounded adapter observations; invocation and output are not host delivery.

Only hashes of opaque session/agent IDs are stored. No tool arguments, prompts,
file selections or transcript text belong in this diagnostic store.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

HOSTS = ("claude", "codex", "agy", "aider")
MAX_ROWS = 128
MAX_COUNT = 1_000_000
MAX_CONFIG_BYTES = 65_536


def _key(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode()).hexdigest()


def _path(root: Path | str) -> Path:
    root = Path(root).resolve()
    state = Path(os.environ.get("TARE_HOOK_STATE_DIR", str(Path.home() / ".local/state/tare-hooks"))).expanduser().resolve()
    if state.is_relative_to(root):
        raise ValueError("Hook observations must remain outside the candidate")
    path = (state / "observations" / (_key(str(root)) + ".sqlite3")).resolve()
    if path.is_relative_to(root):
        raise ValueError("Hook observation target must remain outside the candidate")
    return path


def _owner(connection: sqlite3.Connection, root: Path | str, *, write: bool) -> None:
    if write:
        connection.execute("CREATE TABLE IF NOT EXISTS owner (id INTEGER PRIMARY KEY CHECK(id=1), root TEXT NOT NULL)")
        connection.execute("INSERT OR IGNORE INTO owner VALUES (1, ?)", (str(Path(root).resolve()),))
    row = connection.execute("SELECT root FROM owner WHERE id=1").fetchone()
    if not row or row[0] != str(Path(root).resolve()):
        raise ValueError("Hook observations belong to another repository")


def _budget(timeout_ms: int) -> None:
    if type(timeout_ms) is not int or not 0 <= timeout_ms <= 1000:
        raise ValueError("timeout_ms must be an integer between 0 and 1000")


def read_context_receipt(root: Path | str, scope: str | None) -> dict[str, Any]:
    if scope is None:
        return {}
    connection = None
    try:
        path = _path(root)
        if not path.is_file():
            return {}
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=.02)
        _owner(connection, root, write=False)
        row = connection.execute("SELECT body FROM contexts WHERE key=? AND length(body)<=16384", (scope,)).fetchone()
        value = json.loads(row[0]) if row else {}
        return value if isinstance(value, dict) else {}
    except (OSError, sqlite3.DatabaseError, ValueError):
        return {}
    finally:
        if connection is not None:
            connection.close()


def write_context_receipt(root: Path | str, scope: str | None, value: dict[str, Any]) -> None:
    """At most 128 recent context identities per root, including subagents."""
    if scope is None:
        return
    connection = None
    try:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(raw.encode()) > 16384:
            return
        path = _path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=.02)
        _owner(connection, root, write=True)
        connection.execute("CREATE TABLE IF NOT EXISTS contexts (key TEXT PRIMARY KEY, body TEXT NOT NULL, updated REAL NOT NULL)")
        connection.execute("INSERT OR REPLACE INTO contexts VALUES (?, ?, ?)", (scope, raw, time.time()))
        connection.execute("DELETE FROM contexts WHERE key NOT IN (SELECT key FROM contexts ORDER BY updated DESC,key LIMIT ?)", (MAX_ROWS,))
        connection.commit()
    except (OSError, sqlite3.DatabaseError, ValueError, TypeError):
        pass
    finally:
        if connection is not None:
            connection.close()


def record_hook_observation(root: Path | str, *, host: str, event: str, tool: str,
                            native_tool: str | None = None, session_id: str | None = None,
                            agent_id: str | None = None, role: str = "implementer",
                            outcome: str, snapshot: str | None = None, digest: str | None = None,
                            output_bytes: int = 0, timeout_ms: int = 20, reason: str | None = None) -> bool:
    """Best-effort bounded local observation; never asserts who invoked us."""
    _budget(timeout_ms)
    if host not in HOSTS or not isinstance(event, str) or not isinstance(tool, str):
        return False
    if not isinstance(role, str) or role not in {"planner", "implementer", "auditor", "explorer"}:
        role = "unknown"
    session_id = session_id if isinstance(session_id, str) and len(session_id) <= 256 else None
    agent_id = agent_id if isinstance(agent_id, str) and len(agent_id) <= 256 else None
    native_tool = native_tool if isinstance(native_tool, str) else tool
    if type(output_bytes) is not int:
        return False
    kind = os.environ.get("TARE_HOOK_OBSERVATION_MODE", "unattested")
    if kind not in {"fixture", "manual", "unattested"}:
        kind = "unattested"
    # A caller cannot promote these local observations to HOST_DELIVERED.
    if outcome not in {"CONTEXT_RETURNED", "SILENT", "UNAVAILABLE", "TIMEOUT", "MISS",
                       "BUDGET_REQUIRED", "EVENT_RECORDED", "EVENT_UNCONFIRMED", "TOOL_FAILED"}:
        return False
    body = {"host": host, "event": event[:32], "tool": tool[:128],
            "native_tool": (native_tool or tool)[:128], "role": role,
            "session_hash": _key(session_id), "agent_hash": _key(agent_id),
            "observation_kind": kind, "outcome": outcome,
            "snapshot": str(snapshot)[:128] if snapshot else None,
            "digest": str(digest)[:128] if digest else None,
            "output_bytes": min(65536, max(0, int(output_bytes))),
            "observed_at": time.time(), "authority": "NONE"}
    if reason == "ON_DEMAND_POLICY":
        body["reason"] = reason
    key = _key(json.dumps([body[k] for k in ("host", "event", "tool", "native_tool", "role",
                                           "session_hash", "agent_hash", "observation_kind")]))
    connection = None
    try:
        path = _path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=timeout_ms / 1000)
        _owner(connection, root, write=True)
        connection.execute("CREATE TABLE IF NOT EXISTS observations (key TEXT PRIMARY KEY, body TEXT NOT NULL, count INTEGER NOT NULL, outputs INTEGER NOT NULL, updated REAL NOT NULL)")
        emitted = int(outcome == "CONTEXT_RETURNED")
        connection.execute("""INSERT INTO observations VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(key) DO UPDATE SET body=excluded.body,
            count=min(count+1, ?), outputs=min(outputs+excluded.outputs, ?), updated=excluded.updated""",
            (key, json.dumps(body, separators=(",", ":")), emitted, body["observed_at"], MAX_COUNT, MAX_COUNT))
        connection.execute("DELETE FROM observations WHERE key NOT IN (SELECT key FROM observations ORDER BY updated DESC,key LIMIT ?)", (MAX_ROWS,))
        connection.commit()
        return True
    except (OSError, sqlite3.DatabaseError, ValueError, TypeError):
        return False
    finally:
        if connection is not None:
            connection.close()


def _registration(host: str, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "NOT_INSPECTED", "reason": "No standard native registration path configured"}
    result: dict[str, Any] = {"status": "NOT_FOUND", "scope": "explicit_config_file", "path": str(path)}
    if not path.is_file():
        return result
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            return result | {"status": "OVER_BUDGET"}
        document = json.loads(raw)
        hooks = document.get("hooks", {}) if isinstance(document, dict) else {}
        events = []
        if isinstance(hooks, dict):
            for event, entries in list(hooks.items())[:32]:
                if not isinstance(entries, list):
                    continue
                for entry in entries[:32]:
                    if not isinstance(entry, dict):
                        continue
                    commands = entry.get("hooks", [])
                    if not isinstance(commands, list):
                        continue
                    for command in commands[:16]:
                        value = command.get("command", "") if isinstance(command, dict) else ""
                        if isinstance(value, str) and any(token in value for token in (
                                "tare-hook", "compute_plane.hook_cli", "codex-global.py", "relay.session_hooks")):
                            events.append({"event": str(event)[:32], "matcher": str(entry.get("matcher", ""))[:256],
                                           "command_sha256": _key(value)})
                            if len(events) == 16:
                                break
                    if len(events) == 16:
                        break
                if len(events) == 16:
                    break
        return result | {"status": "DECLARED_UNVERIFIED" if events else "NO_RECOGNIZED_HANDLER",
                         "events": events, "config_sha256": hashlib.sha256(raw).hexdigest()}
    except (OSError, ValueError, UnicodeError):
        return result | {"status": "UNAVAILABLE"}


def observe_codex_delivery(root: Path | str, transcript: Path | str, *, max_bytes: int = 1_048_576) -> dict[str, Any]:
    """Explicit offline inspection of a bounded tail of a local host transcript.

    Never called from a synchronous hook. Only actual developer message blocks
    containing complete SpecGraph payload JSON qualify; quoted tool stdout and
    assistant prose do not. This remains local evidence, not an attestation.
    """
    if type(max_bytes) is not int or not 1024 <= max_bytes <= 1_048_576:
        raise ValueError("max_bytes must be between 1024 and 1048576")
    root = Path(root).resolve()
    transcript = Path(transcript).resolve()
    base = {"status": "NOT_OBSERVED", "host": "codex", "authority": "NONE",
            "source_kind": "LOCAL_HOST_TRANSCRIPT", "source_locator_sha256": _key(str(transcript)),
            "root": str(root), "max_bytes": max_bytes}
    # Restrict this helper to the actual local Codex transcript namespace.
    if not transcript.is_relative_to((Path.home() / ".codex/sessions").resolve()):
        return base | {"status": "UNAVAILABLE", "reason": "outside_local_codex_transcripts"}
    connection = None
    try:
        with transcript.open("rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            start = max(0, end - max_bytes)
            stream.seek(start)
            raw = stream.read(max_bytes)
        if start:
            raw = raw.partition(b"\n")[2]
        selected = None
        for line in raw.splitlines():
            try:
                item = json.loads(line)
                message = item.get("payload", {}) if isinstance(item, dict) else {}
                if item.get("type") != "response_item" or message.get("type") != "message" or message.get("role") != "developer":
                    continue
                content = message.get("content", [])
                for block in content if isinstance(content, list) else []:
                    text = block.get("text", "") if isinstance(block, dict) else ""
                    if not isinstance(text, str) or len(text.encode()) > 65536:
                        continue
                    try:
                        reports = json.loads(text)
                    except ValueError:
                        continue
                    reports = reports if isinstance(reports, list) else [reports]
                    for report in reports[:4]:
                        if (isinstance(report, dict) and report.get("schema") == "tare.tools/specgraph-hook-context/1"
                                and report.get("root") == str(root)):
                            selected = base | {"status": "OBSERVED_IN_HOST_TRANSCRIPT",
                                "payload_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "payload_bytes": len(text.encode()), "snapshot": report.get("snapshot"),
                                "event": report.get("event"), "role": report.get("role"),
                                "timestamp": item.get("timestamp"), "bytes_scanned": len(raw)}
            except (ValueError, UnicodeError, AttributeError, TypeError):
                continue
        if selected is None:
            return base | {"bytes_scanned": len(raw)}
        path = _path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=.02)
        _owner(connection, root, write=True)
        connection.execute("CREATE TABLE IF NOT EXISTS deliveries (host TEXT PRIMARY KEY, body TEXT NOT NULL)")
        connection.execute("INSERT OR REPLACE INTO deliveries VALUES (?, ?)", ("codex", json.dumps(selected, separators=(",", ":"))))
        connection.commit()
        return selected
    except (OSError, sqlite3.DatabaseError, ValueError):
        return base | {"status": "UNAVAILABLE"}
    finally:
        if connection is not None:
            connection.close()


def hook_host_status(root: Path | str, *, timeout_ms: int = 20,
                     registration_paths: dict[str, Path | str] | None = None) -> dict[str, Any]:
    """Read fixed-size diagnostics and known registration files; no source scan.

    Declaration, local adapter invocation/output and actual host delivery are
    separate. The last requires external host evidence, never a callback claim.
    """
    _budget(timeout_ms)
    paths = registration_paths if registration_paths is not None else {
        "codex": Path.home() / ".codex/hooks.json", "claude": Path.home() / ".claude/settings.json"}
    hosts = {host: {"registration": _registration(host, Path(paths[host])) if host in paths else _registration(host, None),
                    "adapter_invocation": "NOT_OBSERVED", "context_output": "NOT_OBSERVED",
                    "host_delivery": "UNATTESTED", "observations": []} for host in HOSTS}
    result = {"schema": "tare.tools/hook-host-observations/1", "root": str(Path(root).resolve()),
              "status": "NOT_INITIALIZED", "hosts": hosts, "max_records": MAX_ROWS,
              "authority": "NONE", "retention": "LATEST_128_SCOPES; COUNTERS_SATURATE"}
    connection = None
    try:
        path = _path(root)
        if not path.is_file():
            return result
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=timeout_ms / 1000)
        connection.execute("PRAGMA query_only=ON")
        _owner(connection, root, write=False)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('observations','deliveries')")}
        rows = connection.execute("SELECT body,count,outputs FROM observations WHERE length(body)<=4096 ORDER BY updated DESC,key LIMIT ?", (MAX_ROWS,)).fetchall() if "observations" in tables else []
        for raw, count, outputs in rows:
            row = json.loads(raw)
            if not isinstance(row, dict) or row.get("host") not in hosts:
                continue
            host = hosts[row["host"]]
            row.update(invocations=count, context_outputs=outputs)
            # Keep doctor output bounded separately from retention capacity.
            if len(host["observations"]) < 8:
                host["observations"].append(row)
            kinds = host.setdefault("observation_kinds", [])
            if row["observation_kind"] not in kinds:
                kinds.append(row["observation_kind"])
            host["adapter_invocation"] = "OBSERVED_LOCALLY"
            if outputs:
                host["context_output"] = "RETURNED_BY_ADAPTER"
        if "deliveries" in tables:
            for host, raw in connection.execute("SELECT host,body FROM deliveries WHERE length(body)<=4096 LIMIT 4"):
                delivery = json.loads(raw)
                if host in hosts and isinstance(delivery, dict) and delivery.get("root") == str(Path(root).resolve()):
                    hosts[host]["host_delivery"] = "OBSERVED_IN_HOST_TRANSCRIPT"
                    hosts[host]["delivery_observation"] = delivery
        result["status"] = "OBSERVED"
        result["retained_records"] = len(rows)
    except (OSError, sqlite3.DatabaseError, ValueError, TypeError, KeyError):
        result["status"] = "UNAVAILABLE"
    finally:
        if connection is not None:
            connection.close()
    return result
