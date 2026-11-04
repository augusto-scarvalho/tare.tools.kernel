"""Disposable live-CLI smoke probe for the canonical tool hook."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REASON_SENTINEL = "specgraph resolve"


def _run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        combined = f"{completed.stdout}\n{completed.stderr}"
        return {
            "exit_code": completed.returncode,
            "deny_reason_observed": REASON_SENTINEL in combined,
            "output_tail": combined[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        combined = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        return {
            "exit_code": None,
            "deny_reason_observed": REASON_SENTINEL in combined,
            "timeout": True,
            "output_tail": combined[-4000:],
        }


def _hook_config(vendor: str) -> dict[str, Any]:
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash|Grep",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"tare-hook {vendor}",
                            "timeout": 5,
                        }
                    ],
                }
            ]
        }
    }


def _agy_hook_config() -> dict[str, Any]:
    return {
        "tare-specgraph-first": {
            "PreToolUse": [
                {
                    "matcher": "run_command|grep_search",
                    "hooks": [
                        {"type": "command", "command": "tare-hook agy", "timeout": 5}
                    ],
                }
            ]
        }
    }


def main() -> int:
    commands = {
        name: shutil.which(name) for name in ("agy", "claude", "codex", "tare-hook")
    }
    missing = [name for name, path in commands.items() if path is None]
    if missing:
        print(json.dumps({"status": "HELD_MISSING_COMMANDS", "missing": missing}, indent=2))
        return 1

    with tempfile.TemporaryDirectory(prefix="tare-tool-hook-e2e-") as raw_root:
        root = Path(raw_root)
        agy_only = os.environ.get("TARE_HOOK_PROBE_AGY_ONLY") == "1"
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "needle.txt").write_text("SpecGraphHookNeedle\n", encoding="utf-8")

        claude_dir = root / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(
            json.dumps(_hook_config("claude")), encoding="utf-8"
        )
        claude = {"skipped": True} if agy_only else _run(
            [
                commands["claude"],
                "-p",
                "Use the Grep tool exactly once to regex-search for SpecGraphHookNeedle, then report the tool result.",
                "--output-format",
                "json",
                "--tools",
                "Grep",
                "--allowedTools",
                "Grep",
                "--permission-mode",
                "bypassPermissions",
            ],
            cwd=root,
        )

        codex_home = root / "codex-home"
        codex_home.mkdir()
        source_auth = Path.home() / ".codex" / "auth.json"
        if source_auth.exists():
            shutil.copy2(source_auth, codex_home / "auth.json")
        (codex_home / "hooks.json").write_text(
            json.dumps(_hook_config("codex")), encoding="utf-8"
        )
        codex_env = dict(os.environ)
        codex_env["CODEX_HOME"] = str(codex_home)
        codex = {"skipped": True} if agy_only else _run(
            [
                commands["codex"],
                "exec",
                "--dangerously-bypass-hook-trust",
                "--ephemeral",
                "--skip-git-repo-check",
                "--json",
                "Use the Bash tool exactly once to run `rg SpecGraphHookNeedle .`, then report the tool result.",
            ],
            cwd=root,
            env=codex_env,
        )

        agy_dir = root / ".agents"
        agy_dir.mkdir()
        (agy_dir / "hooks.json").write_text(
            json.dumps(_agy_hook_config()), encoding="utf-8"
        )
        agy = _run(
            [
                commands["agy"],
                "-p",
                f"Use grep_search exactly once with SearchPath {root}, Query SpecGraphHookNeedle, and IsRegex true; then report the tool result.",
                "--output-format",
                "json",
                "--add-dir",
                str(root),
                "--dangerously-skip-permissions",
            ],
            cwd=root,
        )

        result = {
            "status": "PASS" if (
                agy.get("deny_reason_observed")
                and (
                    agy_only
                    or all(item.get("deny_reason_observed") for item in (claude, codex))
                )
            ) else "HELD",
            "claude": claude,
            "codex": codex,
            "agy": agy,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
