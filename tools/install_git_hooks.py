#!/usr/bin/env python3
"""Install the repository-owned Git hooks for this checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if root_result.returncode != 0:
        print("git hook install failed: not inside a Git repository", file=sys.stderr)
        return 2

    root = Path(root_result.stdout.strip())
    required = [
        root / ".githooks" / "pre-push",
        root / "tools" / "changelog_guard.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(f"git hook install failed: missing {', '.join(missing)}", file=sys.stderr)
        return 2

    configured = subprocess.run(
        ["git", "-C", str(root), "config", "core.hooksPath", ".githooks"],
        capture_output=True,
        text=True,
        check=False,
    )
    if configured.returncode != 0:
        print(configured.stderr.strip(), file=sys.stderr)
        return configured.returncode

    print("GIT_HOOKS=INSTALLED core.hooksPath=.githooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
