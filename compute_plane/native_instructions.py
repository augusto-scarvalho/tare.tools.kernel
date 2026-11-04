"""Scoped instruction discovery helper along repository ancestor chains."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Callable, Iterable, Mapping

from .native_journal import io_path

MAX_FILE_BYTES: int = 16 * 1024  # 16 KiB = 16384 bytes
MAX_AGGREGATE_BYTES: int = 24000  # 24000 bytes
MAX_UNIQUE_RECORDS: int = 256
MAX_CHAIN_DEPTH: int = 64

_WINDOWS_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"
})
_WINDOWS_DEVICE_REGEX = re.compile(r"^(COM|LPT)[1-9\u00b9\u00b2\u00b3]$", re.IGNORECASE)
_HEX64_REGEX = re.compile(r"^[0-9a-f]{64}$")


def _check_path_safety(path: str, *, directory: bool = False) -> None:
    """Pre-resolution safety checks for Windows aliases, trailing dots/spaces, traversal, and .git."""
    if type(path) is not str or not path:
        raise ValueError("invalid_repository_path")
    if type(directory) is not bool:
        raise ValueError("directory_must_be_bool")
    if any(c in path for c in "\\:\0\r\n") or len(path) > 4096:
        raise ValueError("invalid_repository_path")
    if path.startswith("/") or path != path.strip():
        raise ValueError("invalid_repository_path")

    if directory and path.endswith("/"):
        path = path.rstrip("/")
        if not path:
            raise ValueError("invalid_repository_path")
    elif not directory and path.endswith("/"):
        raise ValueError("invalid_repository_path")

    if path == ".":
        if not directory:
            raise ValueError("invalid_repository_path")
        return

    parts = path.split("/")
    for part in parts:
        if not part or part == "." or part == "..":
            raise ValueError("invalid_repository_path")
        if part.startswith(" ") or part.endswith(" ") or part.endswith("."):
            raise ValueError("invalid_repository_path")
        if "~" in part:
            raise ValueError("invalid_repository_path")
        lower_part = part.lower()
        if lower_part == ".git" or lower_part.startswith(".git.") or lower_part.startswith(".git "):
            raise ValueError("invalid_repository_path")
        stem = part.split(".")[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES or _WINDOWS_DEVICE_REGEX.match(stem):
            raise ValueError("invalid_repository_path")


def _is_junction(p: Path) -> bool:
    if getattr(p, "is_junction", lambda: False)():
        return True
    try:
        st = os.lstat(p)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        mount_tag = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)
        return bool(st.st_file_attributes & reparse_flag and getattr(st, "st_reparse_tag", 0) == mount_tag)
    except (AttributeError, OSError):
        return False


def _check_links(target_io: Path, root_io: Path) -> None:
    """Ensure no path component between root and target is a symlink or junction."""
    for item in (target_io, *target_io.parents):
        if item == root_io.parent:
            break
        if item.is_symlink() or _is_junction(item) or os.path.islink(item):
            raise ValueError("repository_link_forbidden")


def _read_instruction_disk_state(io_target: Path) -> tuple[str | None, str | None]:
    """Read regular file state using lstat check before open and bounded max+1 read."""
    try:
        st = os.lstat(io_target)
        exists = True
    except FileNotFoundError:
        exists = False
    except OSError as exc:
        raise ValueError(f"unreadable_instruction_file: {exc}") from exc

    if not exists:
        return None, None

    if stat.S_ISLNK(st.st_mode) or _is_junction(io_target):
        raise ValueError("repository_link_forbidden")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("regular_file_required")
    if st.st_size > MAX_FILE_BYTES:
        raise ValueError("instruction_file_too_large")

    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(io_target, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino):
                raise ValueError("instruction_file_changed_during_open")
            raw = stream.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"unreadable_instruction_file: {exc}") from exc

    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("instruction_file_too_large")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("instruction_file_not_utf8") from exc

    sha256 = hashlib.sha256(raw).hexdigest()
    return sha256, text


class ScopedInstructions:
    """Helper for nested AGENTS.md discovery along ancestor chains."""

    def __init__(self, roots: Mapping[str, Path | str], resolve: Callable[..., Path]) -> None:
        if not hasattr(roots, "items"):
            raise ValueError("roots_must_be_mapping")
        self.roots = {k: Path(v).resolve() for k, v in roots.items()}
        self.resolve = resolve
        self._records: dict[tuple[str, str], dict] = {}

    def observe(self, repository: str, path: str, *, directory: bool = False) -> list[dict]:
        """Discover nested AGENTS.md along ancestor chain root-first, excluding root AGENTS.md."""
        if type(repository) is not str or not repository:
            raise ValueError("invalid_repository_name")
        if repository not in self.roots:
            raise KeyError(repository)
        if type(path) is not str or not path:
            raise ValueError("invalid_repository_path")
        if type(directory) is not bool:
            raise ValueError("directory_must_be_bool")

        _check_path_safety(path, directory=directory)

        # Validate requested path through resolve
        resolved_path = self.resolve(repository, path, directory=directory)
        root = self.roots[repository]
        root_io = io_path(root)
        target_io = io_path(resolved_path)

        try:
            resolved_path.resolve().relative_to(root)
        except ValueError:
            raise ValueError("repository_path_escape")

        _check_links(target_io, root_io)

        if not directory and target_io.exists() and not target_io.is_file():
            raise ValueError("regular_file_required")
        if directory and target_io.exists() and not target_io.is_dir():
            raise ValueError("directory_required")

        # Build ancestor chain paths
        if path == ".":
            chain_paths: list[str] = []
        else:
            cleaned = path.rstrip("/") if directory else path
            parts = [p for p in PurePosixPath(cleaned).parts if p and p != "."]
            if directory:
                dir_slices = [parts[:i] for i in range(1, len(parts) + 1)]
            else:
                dir_slices = [parts[:i] for i in range(1, len(parts))]
            chain_paths = ["/".join(s) + "/AGENTS.md" for s in dir_slices]

        if len(chain_paths) > MAX_CHAIN_DEPTH:
            raise ValueError("instruction_chain_depth_exceeded")

        new_records: dict[tuple[str, str], dict] = {}
        chain_records: list[dict] = []

        for instruction_path in chain_paths:
            _check_path_safety(instruction_path, directory=False)
            resolved_inst = self.resolve(repository, instruction_path, directory=False)

            try:
                resolved_inst.resolve().relative_to(root)
            except ValueError:
                raise ValueError("repository_path_escape")

            inst_io = io_path(resolved_inst)
            _check_links(inst_io, root_io)

            sha256, text = _read_instruction_disk_state(inst_io)
            key = (repository, instruction_path)

            if key in self._records:
                pinned = self._records[key]
                if pinned["sha256"] != sha256 or pinned["text"] != text:
                    raise ValueError(f"instruction_pinned_row_changed: {repository}:{instruction_path}")
                chain_records.append(pinned)
            elif key in new_records:
                chain_records.append(new_records[key])
            else:
                rec = dict(repository=repository, path=instruction_path, sha256=sha256, text=text)
                new_records[key] = rec
                chain_records.append(rec)

        if new_records:
            # Guard budget BEFORE mutating cache
            merged = dict(self._records)
            merged.update(new_records)

            if len(merged) > MAX_UNIQUE_RECORDS:
                raise ValueError("instruction_unique_records_limit_exceeded")

            agg_bytes = len(json.dumps([r for r in merged.values() if r["text"] is not None], ensure_ascii=False).encode("utf-8"))
            if agg_bytes > MAX_AGGREGATE_BYTES:
                raise ValueError("instruction_aggregate_budget_exceeded")

            if len(json.dumps(list(merged.values()), ensure_ascii=False).encode("utf-8")) > 65536:
                raise ValueError("instruction_metadata_budget_exceeded")
            self._records.update(new_records)

        return copy.deepcopy(chain_records)

    def records(self) -> list[dict]:
        """All pinned records sorted by repository and path, deep copy."""
        sorted_records = sorted(self._records.values(), key=lambda r: (r["repository"], r["path"]))
        return copy.deepcopy(sorted_records)

    def restore(self, rows: Iterable[Any]) -> None:
        """Validate records against exact current disk bytes and merge pinned records.

        Refuses contradictory snapshots or changes, including an absent file now created.
        No mutation if validation fails.
        """
        if not isinstance(rows, list) or len(rows) > MAX_UNIQUE_RECORDS:
            raise ValueError("instruction_unique_records_limit_exceeded")
        candidate = dict(self._records)
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"repository", "path", "sha256", "text"}:
                raise ValueError("invalid_record_fields")
            repo, path, sha256, text = (row[k] for k in ("repository", "path", "sha256", "text"))
            if type(repo) is not str or not repo or repo not in self.roots:
                raise ValueError("invalid_repository")
            if (type(path) is not str or not path.endswith("/AGENTS.md") or
                    path == "AGENTS.md" or path == "/AGENTS.md"):
                raise ValueError("invalid_instruction_path")

            _check_path_safety(path, directory=False)
            if len(PurePosixPath(path).parts) - 1 > MAX_CHAIN_DEPTH:
                raise ValueError("instruction_chain_depth_exceeded")

            if sha256 is None:
                if text is not None:
                    raise ValueError("absent_record_text_must_be_none")
            else:
                if type(sha256) is not str or not _HEX64_REGEX.match(sha256):
                    raise ValueError("invalid_sha256")
                if type(text) is not str:
                    raise ValueError("present_record_text_must_be_str")
                encoded_text = text.encode("utf-8")
                if len(encoded_text) > MAX_FILE_BYTES:
                    raise ValueError("instruction_file_too_large")
                if hashlib.sha256(encoded_text).hexdigest() != sha256:
                    raise ValueError("sha256_text_mismatch")

            resolved_inst = self.resolve(repo, path, directory=False)
            root = self.roots[repo]
            root_io = io_path(root)

            try:
                resolved_inst.resolve().relative_to(root)
            except ValueError:
                raise ValueError("repository_path_escape")

            inst_io = io_path(resolved_inst)
            _check_links(inst_io, root_io)

            disk_sha, disk_text = _read_instruction_disk_state(inst_io)

            if sha256 is None:
                if disk_sha is not None:
                    raise ValueError(f"absent_file_now_exists: {repo}:{path}")
            else:
                if disk_sha is None:
                    raise ValueError(f"present_file_missing_on_disk: {repo}:{path}")
                if disk_sha != sha256 or disk_text != text:
                    raise ValueError(f"instruction_source_changed: {repo}:{path}")

            key = (repo, path)
            if key in candidate:
                existing = candidate[key]
                if existing["sha256"] != sha256 or existing["text"] != text:
                    raise ValueError(f"contradictory_snapshot: {repo}:{path}")

            rec = dict(repository=repo, path=path, sha256=sha256, text=text)
            candidate[key] = rec

        if len(candidate) > MAX_UNIQUE_RECORDS:
            raise ValueError("instruction_unique_records_limit_exceeded")

        agg_bytes = len(json.dumps([r for r in candidate.values() if r["text"] is not None], ensure_ascii=False).encode("utf-8"))
        if agg_bytes > MAX_AGGREGATE_BYTES:
            raise ValueError("instruction_aggregate_budget_exceeded")

        if len(json.dumps(list(candidate.values()), ensure_ascii=False).encode("utf-8")) > 65536:
            raise ValueError("instruction_metadata_budget_exceeded")
        # No mutation if validation fails: commit candidate now
        self._records = candidate
