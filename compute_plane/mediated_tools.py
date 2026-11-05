"""Mediated repository capabilities, independent of model and host."""
from __future__ import annotations

import os
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
import hashlib
import re

from .tool_hooks import ToolCall, ToolDecision, ToolGateResult, SpecGraphFirstGate, SpecGraphEditPolicy


class SearchMismatch(ValueError):
    """An edit search that is absent or repeated. The code stays exact; `detail` says which and where,
    so a model fixes the right thing (an escaped regex it mistyped is not a missing-context problem)."""

    def __init__(self, text, needle):
        super().__init__("exact_unique_search_required")
        lines = text.split('\n')
        starts, index = [], text.find(needle)
        while index != -1 and len(starts) < 6:
            starts.append(text.count('\n', 0, index) + 1)
            index = text.find(needle, index + 1)
        if starts:
            self.detail = (f"The search text occurs {text.count(needle)} times (lines "
                           f"{', '.join(map(str, starts))}); include neighbouring lines to make it unique.")
            return
        import difflib
        first = next((line for line in needle.split('\n') if line.strip()), needle)
        close = difflib.get_close_matches(first.strip(), [line.strip() for line in lines], n=2, cutoff=0.5)
        shown = [f"line {n}|{lines[n - 1][:200]}" for n in range(1, len(lines) + 1)
                 if lines[n - 1].strip() in close][:2]
        self.detail = ("The search text is not in the file" +
                       (f"; the closest lines, exactly as written: {' | '.join(shown)}" if shown else "") +
                       ". Copy only the source after the first |, preserving indentation, quotes and backslashes, or use a shorter anchor.")
from .workspace import isolated_child_environment
from .native_journal import io_path
from .native_review import edit_args

# One explicitly named source may be larger than the directory search budget.
# Replies remain bounded to 200 lines / 24,000 bytes, including granted-folder reads.
MAX_READ_BYTES = 8 * 1024 * 1024


def validate_command_confinement(factory, environment, seconds):
    if not callable(factory):
        raise ValueError('command_process_factory_required')
    if not isinstance(environment, dict):
        raise ValueError('command_environment_must_be_string_dict')
    copied = dict(environment)
    folded = set()
    for name, value in copied.items():
        if (not isinstance(name, str) or not name or '\0' in name or '=' in name
                or name.casefold() in folded):
            raise ValueError('command_environment_name_invalid')
        if not isinstance(value, str) or '\0' in value:
            raise ValueError('command_environment_value_invalid')
        folded.add(name.casefold())
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('command_seconds_must_be_finite_positive')
    return copied, seconds


SCHEMAS = {
    "search": {"repository": "id", "query": "text", "path": "relative scope", "glob": "file glob",
               "mode": "literal or regex", "case": "sensitive or insensitive", "limit": "up to 200 matches"},
    "list": {"repository": "id", "path": "relative directory, default ."},
    "read": {"repository": "id", "path": "relative file", "start": "optional line", "end": "optional line",
             "start_column": "optional 1-based Unicode column for a long-line continuation",
             "json_pointer": "optional JSON pointer", "json_fields": "comma-separated relative JSON pointers",
             "offset": "JSON page offset", "limit": "up to 100 JSON items"},
    "resolve": {"repository": "id", "query": "literal query"},
    "edit": {"repository": "id", "path": "relative file", "sha256": "latest read hash or null for new", "search": "old text", "replace": "new text", "changes": "1 to 16 search/replace pairs, instead of search/replace", "delete": "boolean: true requires changes=[] and current non-null sha256"},
    "rename": {"repository": "id", "path": "relative source", "sha256": "latest hash", "target": "new relative path"},
    "run": {"repository": "id", "argv": "argument string list; explicit execution permission required"},
    "verify": {"name": "configured verification name"},
    "doctor": {}, "context": {},
    "explore": {"repository": "id", "question": "question", "focus": "optional list; EACH item requires path, start_line and end_line. Lines are integers with 1 <= start_line <= end_line. Example: [{\"path\":\"calc.py\",\"start_line\":1,\"end_line\":6}]. Omit focus entirely if ranges are unknown."},
    "ask_user": {"question": "precise unresolved decision"},
    "submit": {"artifact": "role artifact object"},
}


class MediatedTools:
    def __init__(self, config, *, callbacks=None, cancelled=lambda: False, before_effect=lambda: None, scratch_root=None,
                 before_edit=None, command_mediation=None, process_factory=None, command_environment=None, command_seconds=None):
        self.config = config
        self.roots = {key: Path(root).resolve() for key, root in config["repositories"].items()}
        self.callbacks = callbacks or {}
        self.cancelled = cancelled
        self.before_effect = before_effect
        self.before_edit = before_edit
        self.command_mediation = command_mediation
        if any(value is not None for value in (process_factory, command_environment, command_seconds)):
            command_environment, command_seconds = validate_command_confinement(process_factory, command_environment, command_seconds)
        else:
            process_factory = command_environment = command_seconds = None
        self.process_factory = process_factory
        self.command_environment = command_environment
        self.command_seconds = command_seconds
        self.verification_seconds = None
        self.unbounded_verification = False
        grant = {key: config.get(key) for key in ('host', 'role', 'writable', 'admitted_capabilities', 'allow_exec')}
        grant['repositories'] = {name: str(path) for name, path in self.roots.items()}
        self.command_grant_sha256 = hashlib.sha256(json.dumps(grant, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.scratch_root = scratch_root
        self.reads = {}
        self._source_hashes = {}  # Includes own writes; does not attest a read.
        self.written = set()
        self.verifications = {}
        self.effects = 0
        if type(config.get('specgraph_edit_context', True)) is not bool:
            raise ValueError('specgraph_edit_context_must_be_boolean')
        admitted = config.get('admitted_capabilities')
        if admitted is not None and (not isinstance(admitted, list) or not admitted
                or any(not isinstance(name, str) or name not in SCHEMAS for name in admitted)
                or len(admitted) != len(set(admitted))):
            raise ValueError('invalid_admitted_capabilities')

    def schemas(self):
        names = {"list", "read", "resolve", "doctor", "context", "ask_user", "submit"}
        names |= set(self.config.get("admitted_capabilities") or ()) & {"search"}
        names |= set(self.callbacks) & {"explore"}
        if self.config["role"] == "auditor" and self.config.get("verifiers"):
            names.add("verify")
        if self.config["role"] == "implementer":
            names |= {"edit", "rename", "verify"}
            if self.config.get("allow_exec"):
                names.add("run")
        if self.config.get('admitted_capabilities') is not None:
            names &= set(self.config['admitted_capabilities'])
        return {key: SCHEMAS[key] for key in SCHEMAS if key in names}

    def path(self, repository, name, *, write=False, directory=False, proven=None):
        """`proven`: folders one operation already proved link-free inside the root. A name without '..' under
        such a folder only needs its own last part checked (a search validates thousands of files)."""
        root = self.roots[repository]
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or ".git" in PurePosixPath(name).parts):
            raise ValueError("invalid_repository_path")
        path = root / name
        if proven is not None and path.parent in proven:
            checked = io_path(path)
            if checked.is_symlink() or getattr(checked, "is_junction", lambda: False)():
                raise ValueError("repository_link_forbidden")
        else:
            if not path.resolve().is_relative_to(root):
                raise ValueError("repository_path_escape")
            for item in (path, *path.parents):
                if item == root.parent:
                    break
                checked = io_path(item)
                if checked.is_symlink() or getattr(checked, "is_junction", lambda: False)():
                    raise ValueError("repository_link_forbidden")
            if proven is not None:
                proven.update(item for item in path.parents if item == root or item.is_relative_to(root))
        if write:
            if self.config["role"] != "implementer":
                raise ValueError("role_cannot_edit")
            scopes = self.config.get("writable", {}).get(repository, [])
            relative = path.relative_to(root).as_posix()
            if not any(scope == "." or relative == scope or relative.startswith(scope.rstrip("/") + "/") for scope in scopes):
                raise ValueError("outside_writable_scope")
        checked = io_path(path)
        if not directory and checked.exists() and not checked.is_file():
            raise ValueError("regular_file_required")
        return path

    def observed(self, repository, path):
        target = io_path(self.path(repository, path))
        if not target.exists():
            return None
        if target.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("file_too_large_request_another_source")
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def command(self, argv, root, *, timeout=None, gate=True, read_only=False, output_bytes=16384, partial_read=False,
                verification=False):
        if (type(partial_read) is not bool or partial_read and
                (read_only is not True or type(output_bytes) is not int or not 1 <= output_bytes <= 1024*1024)):
            raise ValueError('partial_output_requires_read_only_and_bounded_bytes')
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v and "\0" not in v for v in argv):
            raise ValueError("argv_list_required")
        if gate:
            call = ToolCall(self.config.get("host", "native"), "PreToolUse", "run_command", {"CommandLine": " ".join(argv)}, str(root))
            result = SpecGraphFirstGate().evaluate(call)
            if result.decision is ToolDecision.DENY:
                raise ValueError(result.reason)
        from .managed_process import run_process
        from .command_mediation import CommandFacts, environment_digest, invoke
        # Freeze the executed argv and environment before consulting advisory ports.
        argv = tuple(argv)
        environment = dict(self.command_environment) if self.command_environment is not None else isolated_child_environment()
        effective_timeout = timeout or self.config.get("operation_timeout", 300)
        command_limit=self.verification_seconds if verification and self.verification_seconds is not None else self.command_seconds
        if verification and self.unbounded_verification:
            command_limit = None  # host opt-in; operation deadline and cancellation still apply
        if command_limit is not None:
            effective_timeout = min(effective_timeout, command_limit)
        with tempfile.TemporaryDirectory(prefix="tare-command-",
                dir=io_path(self.scratch_root) if self.scratch_root is not None else None) as temporary:
            def effect():
                if not read_only:
                    self.before_effect()
                extra_kwargs = {}
                if self.process_factory is not None:
                    extra_kwargs["process_factory"] = self.process_factory
                return run_process(list(argv), cwd=str(root) if self.process_factory is not None else root, dest=temporary,
                    timeout=effective_timeout,
                    env=environment, cancelled=self.cancelled, max_output=1024 * 1024, **extra_kwargs)
            if self.command_mediation is None:
                process = effect()
            else:
                repository = next((name for name, path in self.roots.items() if path == Path(root).resolve()), '')
                recorded_timeout = effective_timeout if effective_timeout is None or math.isfinite(effective_timeout) else None
                facts = CommandFacts('command', self.config.get('host', 'native'), repository, str(root),
                    environment_digest(environment), self.command_grant_sha256, argv, json.dumps({'argv': argv, 'timeout': recorded_timeout,
                        'read_only': read_only, 'output_bytes': output_bytes}, sort_keys=True, allow_nan=False))
                process = invoke(self.command_mediation, facts, effect)
            if (not process["cleanup_verified"] or process["cancelled"] or process["timeout"]
                    or process.get('isolation_violation')):
                raise CommandInterrupted(process)
            if process['output_limit']:
                if not partial_read:
                    raise CommandInterrupted(process)
                # Only a trusted read-only caller may consume a captured prefix,
                # after proved cleanup. Stderr is never filename inventory.
                with (Path(temporary)/'stdout.txt').open('rb') as stream:
                    raw=stream.read(output_bytes)
                return {'returncode':process['exit_code'],'output':raw.decode('utf-8',errors='replace'),
                        'output_truncated':True,'output_limit':True,'process':process}
            raw = (Path(temporary) / "stdout.txt").read_bytes() + (Path(temporary) / "stderr.txt").read_bytes()
            return {"returncode": process["exit_code"], "output": raw[:output_bytes].decode("utf-8", errors="replace"),
                    "output_truncated": len(raw) > output_bytes, "process": process}

    def edited_text(self, old, args):
        if 'changes' in args:
            if 'search' in args or 'replace' in args:
                raise ValueError('edit_choose_changes_or_search_replace')
            rows = args['changes']
            if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
                raise ValueError('edit_changes_requires_1_to_16_pairs')
            if any(not isinstance(row, dict) or set(row) != {'search', 'replace'} for row in rows):
                raise ValueError('edit_changes_requires_search_replace_pairs')
        else:
            rows = [{'search': args.get('search'), 'replace': args.get('replace')}]
        for row in rows:
            search, replacement = row['search'], row['replace']
            if not isinstance(search, str) or not isinstance(replacement, str):
                raise ValueError('edit_text_required')
            old = self.match_edit(old, search, replacement)
            if len(old.encode('utf-8')) > 2 * 1024 * 1024:
                raise ValueError('edited_file_too_large')
        return old

    def match_edit(self, old, search, replacement):
        # Reads expose logical lines, so model-supplied LF text must also match
        # CRLF/CR files. Map the unique match back to the original string rather
        # than normalizing the entire file on write.
        normalized = re.sub(r'\r\n?', '\n', old)
        needle = re.sub(r'\r\n?', '\n', search)
        if search:
            if normalized.count(needle) != 1:
                raise SearchMismatch(normalized, needle)
            begin = normalized.index(needle)
            def offset(index):
                extra = 0
                for match in re.finditer(r'\r\n', old):
                    if match.start() - extra >= index:
                        break
                    extra += 1
                return index + extra
            start, end = offset(begin), offset(begin + len(needle))
        else:
            start = end = len(old)
        # New text follows the edited line's ending, or the last ending when
        # appending. New/empty files retain the supplied replacement exactly.
        ending = re.search(r'\r\n|\r|\n', old[start:])
        if ending is None:
            ending = re.search(r'(\r\n|\r|\n)[^\r\n]*$', old[:start])
            newline = ending.group(1) if ending else None
        else:
            newline = ending.group()
        if newline is not None:
            replacement = re.sub(r'\r\n?', '\n', replacement).replace('\n', newline)
        return old[:start] + replacement + old[end:]

    def perform(self, name, args):
        if name not in self.schemas() or set(args) - set(SCHEMAS[name]):
            raise ValueError("capability_or_arguments_not_admitted:" + name)
        if self.cancelled():
            raise InterruptedError("operation_cancelled")
        if name in self.callbacks:
            return self.callbacks[name](**args)
        if name in {"doctor", "context"}:
            return {"status": "NOT_CONFIGURED"}
        if name == "verify":
            verifiers = self.config.get("verifiers", {})
            if args["name"] not in verifiers:  # a readable refusal, not a KeyError (OS walkthrough, 2026-10-07)
                raise ValueError("unknown_verifier:" + str(args["name"]) + "; known: " + ", ".join(sorted(verifiers)))
            checker = verifiers[args["name"]]
            result = self.command(checker["argv"], self.roots[checker["repository"]],verification=True)
            self.verifications[args["name"]] = {**result, "effects": self.effects}
            return result
        repo = args["repository"]
        root = self.roots[repo]
        if name == "run":
            if not self.config.get("allow_exec") or self.config["role"] != "implementer":
                raise ValueError("execution_not_authorized")
            self.effects += 1
            return self.command(args["argv"], root)
        if name == "resolve":
            return self.command(["specgraph", "--root", str(root), "resolve", args["query"], "--format", "json"], root)
        if name == "search":
            from .native_search import search
            return search(self, **args)
        if name == "list":
            path = io_path(self.path(repo, args.get("path", "."), directory=True))
            names = sorted(item.name + ("/" if item.is_dir() else "") for item in path.iterdir() if item.name != ".git")
            return {"entries": names[:200], "truncated": len(names) > 200}
        path = io_path(self.path(repo, args["path"], write=name in {"edit", "rename"}))
        if name == "read":
            if ('json_pointer' in args and any(k in args for k in ('start', 'end', 'start_column')) or
                    'json_pointer' not in args and any(k in args for k in ('json_fields', 'offset'))):
                raise ValueError('read_choose_source_lines_or_json_pointer: source lines use start/end or limit; '
                             'offset and json_fields need json_pointer')
            start = args.get("start", 1)
            column = args.get('start_column', 1)
            if type(column) is not int or column < 1:
                raise ValueError('read_requires_valid_column: start_column is a 1-based character column')
            line_limit = 200
            if 'json_pointer' not in args and 'limit' in args:
                line_limit = args['limit']
                if type(line_limit) is not int or not 1 <= line_limit <= 100 or 'end' in args:
                    raise ValueError('read_limit_requires_1_to_100_lines_without_end')
            end = args.get("end", start + line_limit - 1 if type(start) is int else None)
            if type(start) is not int or type(end) is not int or not 1 <= start <= end:
                raise ValueError('read_requires_valid_range: start/end are inclusive integers; '
                                 '1 <= start <= end. Pages return at most 200 lines; use next_start to continue.')
            requested_end = end
            end = min(end, start + 199)
            if not path.exists():
                return {"exists": False, "sha256": None}
            with path.open('rb') as stream:
                raw = stream.read(MAX_READ_BYTES + 1)
            if len(raw) > MAX_READ_BYTES:
                raise ValueError('file_too_large_request_another_source')
            # Bind the returned lines and edit hash to the same observed bytes.
            sha = hashlib.sha256(raw).hexdigest()
            key = repo + ":" + args["path"]
            expected = self._source_hashes.get(key, self.reads.get(key))
            if expected is not None and expected != sha:
                self.verifications.clear()
                self.reads.pop(key, None)
            self._source_hashes[key] = sha
            if 'json_pointer' in args:
                from .native_json_read import project_json
                result = project_json(raw, args)
                self.reads[key] = sha
                return {**result, 'sha256': sha}
            lines = raw.decode("utf-8").splitlines()
            if start > len(lines) and (lines or start != 1):
                raise ValueError(f'read_range_outside_file: file has {len(lines)} lines; '
                                 'use line numbers from search or read from start=1')
            selected, size = [], 0
            for i in range(start - 1, min(end, len(lines))):
                if not selected and ('start_column' in args or len((f'{i+1}|'+lines[i]).encode()) > 24000):
                    if column > max(1, len(lines[i])):
                        raise ValueError('read_column_outside_line')
                    prefix = f'{i+1}|'
                    # The source is already valid UTF-8. Ignore only an incomplete
                    # codepoint at the page boundary; its column remains unread.
                    fragment = lines[i][column-1:].encode()[:24000-len(prefix.encode())].decode('utf-8', errors='ignore')
                    last_column = column + len(fragment) - 1
                    unfinished = last_column < len(lines[i])
                    next_line = i+1 if unfinished else (i+2 if i+1 < len(lines) else None)
                    self.reads[key] = sha
                    return {'sha256': sha, 'lines': len(lines), 'text': prefix+fragment,
                        'start': i+1, 'end': i+1, 'requested_end': requested_end,
                        'start_column': column, 'end_column': last_column,
                        'partial_line': column > 1 or unfinished,
                        'truncated': unfinished or i+1 < min(requested_end, len(lines)),
                        'next_start': next_line, 'next_start_column': last_column+1 if unfinished else None}
                line = f'{i + 1}|{lines[i]}'
                length = len(line.encode()) + bool(selected)
                if size + length > 24000:
                    break
                selected.append(line)
                size += length
            actual_end = start + len(selected) - 1
            # Hash observation alone is not evidence that text was delivered.
            self.reads[key] = sha
            return {"sha256": sha, "lines": len(lines), "text": '\n'.join(selected),
                    'start': start, 'end': actual_end,
                    'requested_end': requested_end,
                    'truncated': actual_end < min(requested_end, len(lines)),
                    'next_start': actual_end+1 if actual_end < len(lines) else None}
        if name == 'edit':
            args = edit_args(args)
        deleting = name == 'edit' and args.get('delete', False) is True
        if name == 'edit' and 'delete' in args and type(args['delete']) is not bool:
            raise ValueError('delete_requires_boolean')
        if deleting:
            if args.get('sha256') is None or args.get('sha256') == 'null':
                raise ValueError('delete_requires_sha256')
            if args.get('changes') != [] or 'search' in args or 'replace' in args:
                raise ValueError('delete_requires_empty_changes')
        if args.get("sha256") == "null":
            # Tool-call formats that carry every argument as text (Qwen's XML form) cannot send JSON null;
            # "null" is never a hash, so it can only mean "this file does not exist yet".
            args = {**args, "sha256": None}
        if "sha256" not in args or (args["sha256"] is not None and
                (not isinstance(args["sha256"], str) or
                 re.fullmatch(r"[0-9a-f]{64}", args["sha256"]) is None)):
            raise ValueError('invalid_sha256: use the 64 lowercase hexadecimal characters returned by read; '
                             'to create a missing file use JSON null')
        observed = self.observed(repo, args["path"])
        if observed != args["sha256"]:
            raise ValueError("source_changed_read_again")
        if (name != "edit" or deleting) and observed is None:
            raise ValueError("source_missing")
        target = io_path(self.path(repo, args["target"], write=True)) if name == "rename" else None
        if target is not None and target.exists():
            raise ValueError("rename_target_exists")
        call = ToolCall(self.config.get("host", "aider"), "PreToolUse", "write", {"file_path": args["path"]}, str(root))
        evaluate = (SpecGraphEditPolicy().evaluate if self.config.get('specgraph_edit_context', True)
                    else lambda call: ToolGateResult(ToolDecision.ALLOW))
        pre = evaluate(call)
        if pre.decision is ToolDecision.DENY:
            raise ValueError(pre.reason)
        if target is not None:
            target_pre = evaluate(ToolCall(self.config.get("host", "aider"), "PreToolUse", "write", {"file_path": args["target"]}, str(root)))
            if target_pre.decision is ToolDecision.DENY:
                raise ValueError(target_pre.reason)
        if name == "edit" and not deleting:
            old = path.read_bytes().decode("utf-8") if observed else ""
            new = self.edited_text(old, args)
            if len(new.encode()) > 2 * 1024 * 1024:
                raise ValueError("edited_file_too_large")
            if self.observed(repo, args["path"]) != observed:
                raise ValueError("source_changed_before_write")
            self.before_effect()
            if self.before_edit is not None:
                self.before_edit(args, new, pre.additional_context, self.effects + 1)
            temporary_parent = str(path.parent)
            if os.name == 'nt' and not temporary_parent.startswith('\\\\?\\'):
                # The source may fit MAX_PATH while its longer atomic-edit name
                # does not. Use the native extended spelling for this already
                # validated absolute directory, retaining same-directory replace.
                temporary_parent = ('\\\\?\\UNC\\'+temporary_parent[2:] if temporary_parent.startswith('\\\\')
                                    else '\\\\?\\'+temporary_parent)
            Path(temporary_parent).mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".tare-edit-", dir=temporary_parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                    stream.write(new)
                if path.exists():
                    os.chmod(temporary, path.stat().st_mode)
                if self.observed(repo, args['path']) != observed:
                    raise ValueError('source_changed_before_replace')
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        elif deleting:
            if self.observed(repo, args['path']) != observed:
                raise ValueError('source_changed_before_write')
            self.before_effect()
            if self.before_edit is not None:
                self.before_edit(args, None, pre.additional_context, self.effects + 1)
            if self.observed(repo, args['path']) != observed:
                raise ValueError('source_changed_before_unlink')
            path.unlink()
        elif name == "rename":
            self.before_effect()
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
        else:
            raise ValueError("unknown_capability")
        self.effects += 1
        if deleting:
            self.reads.pop(repo + ':' + args['path'], None)
        self.verifications.clear()
        self._source_hashes[repo + ':' + args['path']] = self.observed(repo, args['path'])
        self.written.add(repo + ":" + args["path"])
        if target is not None:
            self._source_hashes[repo + ':' + args['target']] = self.observed(repo, args['target'])
            self.written.add(repo + ":" + args["target"])
            evaluate(ToolCall(self.config.get("host", "aider"), "PostToolUse", "write", {"file_path": args["target"]}, str(root)))
        post = evaluate(ToolCall(self.config.get("host", "aider"), "PostToolUse", "write", {"file_path": args["path"]}, str(root)))
        return {"sha256": self.observed(repo, args["path"]), "pre_context": pre.additional_context,
                "post_context": post.additional_context, "effects": self.effects}


class CommandInterrupted(OSError):
    def __init__(self, receipt):
        super().__init__("command_interrupted_or_cleanup_unverified")
        self.receipt = receipt
