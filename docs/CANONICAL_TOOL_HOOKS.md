# Canonical tool hooks

Install `tare-kernel` and `tare-tools-specgraph` in the environment that starts
the vendor CLI. The hook command is synchronous and reads exactly one bounded
JSON envelope from stdin. Enable the gate only after the repository's
`specgraph.yaml` covers the source roots that agents are expected to query.

Kernel prepares the first dogfood configuration through its checked-in
`.claude/settings.json`, `.codex/hooks.json`, `.agents/hooks.json` and
`specgraph.yaml`. These are repository-local opt-in files; they do not mutate
any vendor's user-level configuration. Expansion to other repositories remains
an explicit per-repository opt-in.

Set `TARE_LOCAL_MODELS` to the gateway URL in the operator environment before
starting a vendor CLI. The default is `http://127.0.0.1:8080`; remote hosts and
private network addresses belong to that environment. The gate still denies
frontier delegation when `/health` answers, and keeps its existing fail-open
behavior when unreachable. This does not set `TARE_LOCAL_MODELS_BYPASS`.

Repositories can also opt into [scoped snapshot context](SCOPED_PROJECT_CONTEXT_HOOKS.md).
That policy provides bounded on-demand advice and durable Post observations through
the SDK. The search/edit subprocess policy below remains the legacy mode outside
explicitly enabled roots; its registration alone does not prove delivery.

## Claude Code

Add this entry under `hooks` in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Grep|Edit|Write|Agent|Task",
        "hooks": [
          {"type": "command", "command": "tare-hook claude", "timeout": 5}
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {"type": "command", "command": "tare-hook claude", "timeout": 5}
        ]
      }
    ]
  }
}
```

## Codex

Use `.codex/hooks.json` and review the exact hook hash through `/hooks`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Grep|apply_patch|Edit|Write|spawn_agent",
        "hooks": [
          {"type": "command", "command": "tare-hook codex", "timeout": 5}
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "apply_patch|Edit|Write",
        "hooks": [
          {"type": "command", "command": "tare-hook codex", "timeout": 5}
        ]
      }
    ]
  }
}
```

## Google Antigravity

Use `.agents/hooks.json`. In headless runs, pass the project with `--add-dir`
so the CLI admits that workspace customization layer:

```json
{
  "tare-specgraph-first": {
    "PreToolUse": [
      {
        "matcher": "run_command|grep_search|write_to_file|replace_file_content|multi_replace_file_content",
        "hooks": [
          {"type": "command", "command": "tare-hook agy --event pre", "timeout": 5}
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "write_to_file|replace_file_content|multi_replace_file_content",
        "hooks": [
          {"type": "command", "command": "tare-hook agy --event post", "timeout": 5}
        ]
      }
    ]
  }
}
```

The gate covers native `Grep`, Antigravity `grep_search`, and `rg`/`grep`
invocations inside the vendors' shell tools, including fixed-string variants.
It directs every repository lookup to
`specgraph resolve <query> --format json`. That command tries structural lookup
first and uses bounded traditional recovery only after `MISS`, `STALE`,
`TIMEOUT`, or `UNAVAILABLE`; useful fallback evidence is validated and indexed
for reuse by SpecGraph.

The recovery logic deliberately does not depend on vendor `PostToolUse` output.
[Codex](https://developers.openai.com/codex/hooks) and
[Claude Code](https://code.claude.com/docs/en/hooks) expose `tool_response`, but
[Antigravity](https://antigravity.google/docs/hooks) documents only the executed
call and an optional error. Keeping the pipeline in SpecGraph preserves parity
without parsing conversation transcripts. The legacy recovery policy owns no
transcript, cache, daemon or provider call. The scoped policy keeps bounded external
dedup/observation state; its offline transcript helper never runs synchronously. See
[`tare.tools.specgraph/docs/SPECGRAPH_FIRST_RECOVERY_PIPELINE.md`](https://github.com/augusto-scarvalho/tare.tools.specgraph/blob/main/docs/SPECGRAPH_FIRST_RECOVERY_PIPELINE.md)
for the state and telemetry policy.

Recovery state and telemetry live outside target worktrees by default. The
hot path does not call `tare.tools.library`: its vector/Markdown index remains
an optional slow-path destination for reviewed aggregate telemetry reports,
never for raw events, queries or per-call recovery evidence.

Before a recognized edit, the shared policy asks `specgraph impact` for a
bounded preview and returns it as advisory context. After a successful edit it
asks `specgraph staleness` whether the cache is now stale and reports the
bounded blast radius where that vendor supports post-hook context. Neither
path rebuilds the cache or blocks the edit. Antigravity's documented post-hook
contract requires an empty output object, so its post event still performs the
check without inventing a feedback channel.

## Opportunistic implementation queue

This queue records small integrations that may be implemented when their
prerequisites become true. It is not authorization for a hook platform, a
daemon, automatic source mutation, or additional blocking policy.

| ID | Candidate | Lifecycle | Current disposition | Readiness condition |
|---|---|---|---|---|
| `HOOK-SG-01` | Canonical SpecGraph-first search gate | `PreToolUse` | `QUALIFIED` | Keep the existing three adapters and shared policy |
| `HOOK-SG-02` | Neutral documentation/repository search paths | SpecGraph configuration | `QUALIFIED` | Keep `search_paths` retrieval-only and covered by leakage/root-escape tests |
| `HOOK-SG-03` | Staleness and blast-radius notice after a successful edit | `PostToolUse` | `QUALIFIED_ADVISORY` | Keep cache mutation and reindexing outside the hook; Antigravity post output remains `{}` by contract |
| `HOOK-SG-04` | Reviewer Context Bundle for changed files | `Stop` | `HELD_COVERAGE` | Bundle must contain useful mapped specs, symbols and tests; empty mapping remains advisory noise |
| `HOOK-SG-05` | Pre-edit impact preview | `PreToolUse` | `QUALIFIED_ADVISORY` | Require a fresh initialized cache; misses and stale cache never manufacture impact |
| `HOOK-SG-06` | Work-selected repository grounding | `specgraph ground` plus Agent Runtime ingress | `QUALIFIED_SHADOW` | Exact repository/Library IDs are bound by Work, transported canonically, and revalidated by the OS host |
| `HOOK-SG-07` | Formal drift and orphan enforcement | CI or Git pre-commit | `NOT_A_VENDOR_HOOK` | Run `drift-check`/`doctor` only where coverage policy is explicitly adopted |
| `HOOK-BG-01` | Backlog item plus grounding drift fence | Managed Runtime checkpoints | `QUALIFIED_SHADOW` | OS composes independent Backlog and SpecGraph checks before/after dispatch and before claims |
| `HOOK-BG-02` | Backlog fence in unmanaged provider CLI sessions | Vendor-native lifecycle hook | `HELD_AUTHENTICATED_BINDING` | Host must supply a trustworthy dynamic binding; never infer it from prompt, branch or cwd |

### `HOOK-SG-02` acceptance boundary

The Kernel dogfood exposed the immediate prerequisite. General documentation
was placed under `spec_paths`/`adr_paths` only to make it retrievable. SpecGraph
then correctly treated those files as governed specifications or ADRs:

- `doctor` reported six invalid criteria, 121 orphan code symbols and zero
  formal coverage;
- a reviewer bundle for `tests/test_tool_hooks.py` stayed bounded at roughly
  114 tokens but contained no mapped spec, code or test evidence;
- `check-stale` reported the uninitialized graph as stale.

The correction is the retrieval-only `search_paths` field, qualified against
these falsifiers:

1. README, changelog and ordinary documentation are retrievable through
   `specgraph resolve`.
2. The same files do not enter spec, ADR, schema, API or coverage counts.
3. Paths remain root-confined, deduplicated and subject to `excluded_paths`.
4. Existing configurations without `search_paths` retain their behavior.
5. Recovery remains bounded and its derived index and telemetry remain outside
   the target worktree.

Those checks now pass and the Kernel configuration lists ordinary material only
under `search_paths`; `spec_paths` and `adr_paths` are empty. Copying the pattern remains an explicit per-repository
choice. No automatic `sync-cache`, source edit, transcript read, Library
ingestion, or full drift gate belongs in this slice.

### Runtime grounding transport

`compute_plane.specgraph_grounding.materialize_specgraph_grounding` is the
narrow Kernel adapter for the non-hook grounding path. The host supplies an
exact repository ID, Library manifest path, Work-selected Library IDs, command
and budget. Kernel executes `specgraph ground`, validates the canonical envelope
identity, and returns the accepted bytes unchanged for Agent Runtime admission.
`check_specgraph_scope` repeats the same selection against an expected scope
digest and distinguishes current context, selected-context drift, and source
unavailability. It does not parse Library documents, discover manifests, or
merge grounding into prompts.

### Managed backlog fence

`compute_plane.backlog_work.materialize_backlog_work` transports the exact
bounded `tare.tools/work-grounding/1` envelope for an explicit graph and work
ID. `check_backlog_scope` revalidates its execution-scope digest without
mutating the graph. The host maps the result into Agent Runtime's injected
`WorkGuard`, where dispatch and terminal claims can be blocked on drift.

This enforcement is provider-neutral and therefore belongs in the managed
runtime path, not in three duplicated vendor policies. Native hooks remain a
possible fallback only after an authenticated per-session work binding exists.

At the OS composition seam, the Work envelope is the only selector: each
repository scope names one repository and its exact `grounding_refs`.
SpecGraph and Backlog Graph do not import each other. Runtime blocks absent,
duplicate, mismatched or incomplete repository context before the provider,
while the composite Kernel guard revalidates every scope at strict checkpoints.
Native Claude, Codex and Antigravity hooks never infer Work from prompt text,
branch names or current directories.

Pre-edit impact preview and post-edit staleness reporting are advisory. They
consume the narrow SpecGraph contracts, do not rebuild the cache and do not
grant editing authority. The existing SpecGraph-first search gate remains the
blocking search policy outside scoped roots; the independent local-model-first
gate also applies to frontier delegation. Scoped roots use only its explicit
health cache and never probe the network synchronously.

## Live qualification

The disposable probe `py scripts/run_tool_hook_e2e.py` was qualified on
2026-08-25 against Claude Code `2.1.241`, Codex CLI `0.149.1`, and Antigravity
CLI `1.1.20`. Every CLI proposed a real repository-search tool call and surfaced
the same canonical denial before the search executed. The probe creates only a
temporary Git root; Codex receives a temporary `CODEX_HOME`, and no global hook
configuration is modified.
