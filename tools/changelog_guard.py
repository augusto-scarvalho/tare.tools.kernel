#!/usr/bin/env python3
"""Fail closed when material changes lack a useful, append-only changelog entry."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

TOP_LEVEL_HEADING = re.compile(r"^## ([^\r\n]+)\s*$", re.MULTILINE)
PLACEHOLDER = re.compile(
    r"\b(?:todo|tbd|placeholder|update changelog|misc(?:ellaneous)? changes?|various changes?)\b",
    re.IGNORECASE,
)
DOCUMENTATION_SUFFIXES = {".md", ".mdx", ".rst", ".adoc"}
DOCUMENTATION_NAMES = {
    "README",
    "LICENSE",
    "NOTICE",
    "CONTRIBUTING",
    "CODE_OF_CONDUCT",
    "SECURITY",
}


class ChangelogGuardError(RuntimeError):
    """Raised for malformed changelogs or unavailable Git objects."""


@dataclass(frozen=True)
class ParsedChangelog:
    section_order: tuple[str, ...]
    sections: dict[str, str]

    @property
    def unreleased(self) -> str:
        return self.sections["Unreleased"]

    @property
    def historical(self) -> dict[str, str]:
        return {
            name: self.sections[name]
            for name in self.section_order
            if name != "Unreleased"
        }


@dataclass(frozen=True)
class GuardResult:
    material_paths: tuple[str, ...]
    added_entries: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_changelog(text: str) -> ParsedChangelog:
    normalized = _normalize_text(text)
    if not normalized.startswith("# Changelog\n"):
        raise ChangelogGuardError("CHANGELOG.md must start with '# Changelog'")

    matches = list(TOP_LEVEL_HEADING.finditer(normalized))
    if not matches:
        raise ChangelogGuardError("CHANGELOG.md has no level-two sections")

    order: list[str] = []
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        if name in sections:
            raise ChangelogGuardError(f"duplicate changelog section: {name}")
        end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        )
        order.append(name)
        sections[name] = normalized[match.end() : end].strip()

    if "Unreleased" not in sections:
        raise ChangelogGuardError("CHANGELOG.md is missing '## Unreleased'")
    if order[0] != "Unreleased":
        raise ChangelogGuardError("'## Unreleased' must be the first level-two section")
    return ParsedChangelog(tuple(order), sections)


def extract_bullets(section: str) -> tuple[str, ...]:
    """Return normalized top-level bullet blocks from a changelog section."""

    bullets: list[str] = []
    current: list[str] = []
    for raw_line in _normalize_text(section).splitlines():
        if raw_line.startswith("- "):
            if current:
                bullets.append(" ".join(current))
            current = [raw_line[2:].strip()]
        elif current and not raw_line.startswith("#"):
            stripped = raw_line.strip()
            if stripped:
                current.append(stripped)
        elif current:
            bullets.append(" ".join(current))
            current = []
    if current:
        bullets.append(" ".join(current))
    return tuple(bullets)


def is_meaningful_entry(entry: str) -> bool:
    plain = re.sub(r"[`*_\[\]()#]", "", entry).strip()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+/-]*", plain)
    return len(plain) >= 24 and len(words) >= 4 and not PLACEHOLDER.search(plain)


def is_documentation_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    pure = PurePosixPath(normalized)
    upper_name = pure.name.upper()
    if normalized == "CHANGELOG.md" or normalized.startswith("docs/"):
        return True
    if pure.suffix.lower() in DOCUMENTATION_SUFFIXES:
        return True
    return any(
        upper_name == name or upper_name.startswith(f"{name}.")
        for name in DOCUMENTATION_NAMES
    )


def material_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted({path for path in paths if path and not is_documentation_path(path)})
    )


def validate_update(
    base_text: str, head_text: str, changed_paths: Sequence[str]
) -> GuardResult:
    material = material_paths(changed_paths)
    changelog_changed = "CHANGELOG.md" in {
        path.replace("\\", "/") for path in changed_paths
    }
    errors: list[str] = []
    added_entries: tuple[str, ...] = ()

    try:
        head = parse_changelog(head_text)
    except ChangelogGuardError as exc:
        return GuardResult(material, (), (str(exc),))

    base: ParsedChangelog | None = None
    if base_text:
        try:
            base = parse_changelog(base_text)
        except ChangelogGuardError as exc:
            errors.append(f"base CHANGELOG.md is malformed: {exc}")

    if base is not None and changelog_changed:
        for name, content in base.historical.items():
            if name not in head.sections:
                errors.append(f"historical section was deleted: ## {name}")
            elif head.sections[name] != content:
                errors.append(f"historical section was rewritten: ## {name}")

        base_unreleased = set(extract_bullets(base.unreleased))
        head_unreleased = set(extract_bullets(head.unreleased))
        removed = base_unreleased - head_unreleased
        new_sections = [
            name for name in head.section_order if name not in base.sections
        ]
        moved = {
            bullet
            for name in new_sections
            for bullet in extract_bullets(head.sections[name])
        }
        for bullet in sorted(removed - moved):
            errors.append(
                f"unreleased entry was deleted instead of moved to a new release: {bullet[:100]}"
            )

        added_entries = tuple(sorted(head_unreleased - base_unreleased))
    elif base is None:
        added_entries = extract_bullets(head.unreleased)

    if material and not changelog_changed:
        errors.append("material changes require a CHANGELOG.md update")
    elif material:
        meaningful = tuple(
            entry for entry in added_entries if is_meaningful_entry(entry)
        )
        if not meaningful:
            errors.append(
                "material changes require a new meaningful bullet under '## Unreleased'"
            )
        placeholders = tuple(
            entry for entry in added_entries if not is_meaningful_entry(entry)
        )
        if added_entries and len(placeholders) == len(added_entries):
            errors.append("new changelog bullets are placeholders or too vague")

    return GuardResult(material, added_entries, tuple(dict.fromkeys(errors)))


def _git(repo: str, *args: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", repo, *args],
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ChangelogGuardError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def _commit_exists(repo: str, revision: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", repo, "cat-file", "-e", f"{revision}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _is_zero_revision(revision: str) -> bool:
    return bool(revision) and set(revision) == {"0"}


def _empty_tree(repo: str) -> str:
    return (
        _git(repo, "hash-object", "-t", "tree", "--stdin", input_bytes=b"")
        .decode()
        .strip()
    )


def _read_blob(repo: str, revision: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "-C", repo, "show", f"{revision}:{path}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.decode("utf-8", errors="strict")


def check_revisions(repo: str, base_revision: str, head_revision: str) -> GuardResult:
    if not _commit_exists(repo, head_revision):
        raise ChangelogGuardError(f"head commit is unavailable: {head_revision}")
    if _is_zero_revision(base_revision):
        base_revision = _empty_tree(repo)
    elif not _commit_exists(repo, base_revision):
        raise ChangelogGuardError(f"base commit is unavailable: {base_revision}")

    raw_paths = _git(repo, "diff", "--name-only", "-z", base_revision, head_revision)
    paths = tuple(
        part.decode("utf-8", errors="surrogateescape")
        for part in raw_paths.split(b"\0")
        if part
    )
    base_text = _read_blob(repo, base_revision, "CHANGELOG.md")
    head_text = _read_blob(repo, head_revision, "CHANGELOG.md")
    if not head_text:
        return GuardResult(
            material_paths(paths), (), ("head commit has no CHANGELOG.md",)
        )
    return validate_update(base_text, head_text, paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base", help="base commit SHA, or forty zeroes for an empty base"
    )
    parser.add_argument("head", help="head commit SHA")
    parser.add_argument(
        "--repo", default=".", help="repository path (default: current directory)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_revisions(args.repo, args.base, args.head)
    except (ChangelogGuardError, UnicodeDecodeError) as exc:
        print(f"CHANGELOG_GUARD=ERROR: {exc}", file=sys.stderr)
        return 2

    if not result.ok:
        print("CHANGELOG_GUARD=FAIL", file=sys.stderr)
        for error in result.errors:
            print(f"- {error}", file=sys.stderr)
        if result.material_paths:
            print("material paths:", file=sys.stderr)
            for path in result.material_paths:
                print(f"  - {path}", file=sys.stderr)
        return 1

    scope = "material" if result.material_paths else "documentation-only"
    print(f"CHANGELOG_GUARD=PASS scope={scope} new_entries={len(result.added_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
