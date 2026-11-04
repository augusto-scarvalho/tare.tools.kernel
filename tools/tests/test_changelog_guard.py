from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "repository_changelog_guard", ROOT / "tools" / "changelog_guard.py"
)
assert SPEC is not None and SPEC.loader is not None
GUARD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GUARD
SPEC.loader.exec_module(GUARD)

ChangelogGuardError = GUARD.ChangelogGuardError
check_revisions = GUARD.check_revisions
extract_bullets = GUARD.extract_bullets
is_documentation_path = GUARD.is_documentation_path
parse_changelog = GUARD.parse_changelog
validate_update = GUARD.validate_update

BASE = """# Changelog

Repository history.

## Unreleased

### Added

- Added the existing stable feature with enough descriptive context.

## 2026-08-25

### Fixed

- Fixed the historical issue without rewriting its evidence.
"""


def with_unreleased_entry(entry: str) -> str:
    return BASE.replace(
        "- Added the existing stable feature with enough descriptive context.",
        f"- {entry}\n- Added the existing stable feature with enough descriptive context.",
    )


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


class ChangelogGuardTests(unittest.TestCase):
    def test_repository_changelog_is_valid_for_initial_adoption(self) -> None:
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        parsed = parse_changelog(changelog)
        self.assertTrue(extract_bullets(parsed.unreleased))
        result = validate_update(
            "", changelog, ["tools/changelog_guard.py", "CHANGELOG.md"]
        )
        self.assertTrue(result.ok, result.errors)

    def test_path_classification_keeps_code_and_workflows_material(self) -> None:
        self.assertTrue(is_documentation_path("docs/guide.md"))
        self.assertTrue(is_documentation_path("README.md"))
        self.assertTrue(is_documentation_path("CHANGELOG.md"))
        self.assertFalse(is_documentation_path("src/runtime.py"))
        self.assertFalse(is_documentation_path(".github/workflows/ci.yml"))

    def test_material_change_requires_changelog_update(self) -> None:
        result = validate_update(BASE, BASE, ["src/runtime.py"])
        self.assertFalse(result.ok)
        self.assertIn("material changes require a CHANGELOG.md update", result.errors)

    def test_meaningful_unreleased_entry_is_accepted(self) -> None:
        head = with_unreleased_entry(
            "Added deterministic changelog validation for material repository changes."
        )
        result = validate_update(BASE, head, ["src/runtime.py", "CHANGELOG.md"])
        self.assertTrue(result.ok, result.errors)

    def test_placeholder_entry_is_rejected(self) -> None:
        result = validate_update(
            BASE,
            with_unreleased_entry("TODO misc changes"),
            ["src/runtime.py", "CHANGELOG.md"],
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("placeholder" in error for error in result.errors))

    def test_documentation_only_change_is_exempt(self) -> None:
        result = validate_update(BASE, BASE, ["docs/guide.md"])
        self.assertTrue(result.ok, result.errors)

    def test_historical_sections_cannot_be_deleted_or_rewritten(self) -> None:
        deleted = BASE.split("## 2026-08-25", maxsplit=1)[0]
        result = validate_update(BASE, deleted, ["CHANGELOG.md"])
        self.assertIn("historical section was deleted: ## 2026-08-25", result.errors)

        rewritten = BASE.replace(
            "Fixed the historical issue", "Changed the historical claim"
        )
        result = validate_update(BASE, rewritten, ["CHANGELOG.md"])
        self.assertIn("historical section was rewritten: ## 2026-08-25", result.errors)

    def test_unreleased_entry_must_be_preserved_or_released(self) -> None:
        removed = BASE.replace(
            "- Added the existing stable feature with enough descriptive context.\n", ""
        )
        result = validate_update(BASE, removed, ["CHANGELOG.md"])
        self.assertTrue(
            any("unreleased entry was deleted" in error for error in result.errors)
        )

        moved = removed.replace(
            "## 2026-08-25",
            "## 2026-08-26\n\n### Added\n\n"
            "- Added the existing stable feature with enough descriptive context.\n\n"
            "## 2026-08-25",
        )
        self.assertTrue(validate_update(BASE, moved, ["CHANGELOG.md"]).ok)

    def test_changelog_structure_requires_unreleased_first(self) -> None:
        with self.assertRaisesRegex(ChangelogGuardError, "Unreleased"):
            parse_changelog(BASE.replace("## Unreleased", "## Upcoming"))

    def test_revision_integration_uses_committed_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            git(repo, "init")
            git(repo, "config", "user.email", "guard@example.invalid")
            git(repo, "config", "user.name", "Changelog Guard Test")
            (repo / "CHANGELOG.md").write_text(BASE, encoding="utf-8")
            (repo / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
            git(repo, "add", "CHANGELOG.md", "runtime.py")
            git(repo, "commit", "-m", "base")
            base = git(repo, "rev-parse", "HEAD")

            (repo / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
            (repo / "CHANGELOG.md").write_text(
                with_unreleased_entry(
                    "Changed the runtime value with a deterministic migration note."
                ),
                encoding="utf-8",
            )
            git(repo, "add", "CHANGELOG.md", "runtime.py")
            git(repo, "commit", "-m", "material change")
            head = git(repo, "rev-parse", "HEAD")
            self.assertTrue(check_revisions(str(repo), base, head).ok)

    def test_revision_integration_fails_closed_for_missing_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            git(repo, "init")
            git(repo, "config", "user.email", "guard@example.invalid")
            git(repo, "config", "user.name", "Changelog Guard Test")
            (repo / "CHANGELOG.md").write_text(BASE, encoding="utf-8")
            git(repo, "add", "CHANGELOG.md")
            git(repo, "commit", "-m", "base")
            head = git(repo, "rev-parse", "HEAD")
            with self.assertRaisesRegex(
                ChangelogGuardError, "base commit is unavailable"
            ):
                check_revisions(str(repo), "1" * 40, head)


if __name__ == "__main__":
    unittest.main()
