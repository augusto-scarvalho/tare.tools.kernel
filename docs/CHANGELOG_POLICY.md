# Changelog policy and guard

`CHANGELOG.md` is an append-only operational record. Material repository changes
must add a meaningful bullet under `## Unreleased`; documentation-only changes
do not need filler entries.

## Install the local hook

Run once per checkout:

```powershell
python tools/install_git_hooks.py
```

This sets the repository-local Git configuration to
`core.hooksPath=.githooks`. The tracked `pre-push` hook checks every pushed
branch range against the corresponding remote SHA. GitHub CI runs the same
Python implementation, so bypassing a local hook does not bypass repository
policy.

The hook accepts Python 3.10 or newer through `PYTHON`, `python`, `python3`, or
the Windows `py -3` launcher. It validates each candidate before selecting it,
so a broken Microsoft Store alias cannot mask a working interpreter.

## What requires an entry

Material paths include source, tests, configuration, workflows, scripts,
runtime artifacts, and machine-readable evidence. Documentation paths include
`docs/**`, Markdown/reStructuredText/AsciiDoc files, README, LICENSE, NOTICE,
CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, and `CHANGELOG.md` itself.

A valid material change must:

1. change `CHANGELOG.md` in the compared commit range;
2. add a new top-level bullet inside `## Unreleased`;
3. use at least four words and 24 visible characters; and
4. avoid placeholders such as `TODO`, `TBD`, `misc changes`, or `update
   changelog`.

The guard evaluates committed content, not the worktree. An uncommitted
changelog edit cannot authorize a push.

## Append-only protection

Existing dated release sections are immutable. Existing `Unreleased` bullets
may only disappear when copied unchanged into a newly added release section.
This catches accidental truncation while still allowing a normal release cut.
Corrections to an earlier claim should be a new entry that names what it
supersedes instead of rewriting old evidence.

The guard also validates that `# Changelog` is the document title and that
`## Unreleased` is the first level-two section.

## Manual check

```powershell
python tools/changelog_guard.py <base-sha> <head-sha>
```

Exit codes are stable:

- `0`: policy passed;
- `1`: policy violation;
- `2`: malformed input, missing Git object, or execution error.

The implementation is covered by unit and temporary-repository integration
tests in `tools/tests/test_changelog_guard.py`.
