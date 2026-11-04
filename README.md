<div align="center">

# tare.tools.kernel

**Bounded model and tool execution for the tare.tools ecosystem: provider adapters, mediated tools, Git workspaces, terminal sessions, hooks and append-only session journals.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://python.org)

<p align="center">
  <a href="#what-is-taretoolskernel">What is tare.tools.kernel</a> •
  <a href="#what-exists-today">What exists today</a> •
  <a href="#install-and-test">Install and test</a> •
  <a href="#compute-plane--execution-boundaries">Execution Boundaries</a> •
  <a href="#python-api-quickstart">API Quickstart</a> •
  <a href="#ecosystem-family">Ecosystem Family</a> •
  <a href="CHANGELOG.md">Changelog</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <em>🇧🇷 Para a versão em Português, consulte <a href="README.pt-BR.md">README.pt-BR.md</a>.</em>
</p>

</div>

---

## What is tare.tools.kernel?

`tare.tools.kernel` executes model calls and their effects for the `tare.tools`
orchestrator ([tare.tools.os](https://github.com/augusto-scarvalho/tare.tools.os)),
which owns Work selection, review, landing and the operator interfaces. The OS pins
this repository as a Git submodule at `satellites/tare.tools.kernel` and installs it
from there; on its own, the Kernel is a Python library plus the `tare-hook` command.

[Native model adapters](docs/guides/NATIVE_MODEL_ADAPTERS.md) provide provider
and process ports to [`tare.tools.agent-runtime`](https://github.com/augusto-scarvalho/tare.tools.agent-runtime),
the current tactical engine. The removed Aider and aggregate HTTP-session driver
guides are no longer distributed ([repository content](docs/guides/REPOSITORY_CONTENT.md)).

## What exists today

The implemented code lives in `compute_plane/`, plus producer receipts in `assurance_plane/`:

| Area | Modules |
|---|---|
| Model adapters | `native_models.py` (HTTP codecs, context projection), `native_cli.py` (Claude/Codex/AGY CLIs), `native_stream.py`, `native_usage.py` |
| Mediated tools and effects | `mediated_tools.py`, `native_capabilities.py`, `native_effect_recovery.py`, `tool_hooks.py`, `hook_observations.py` |
| Session journal | `native_journal.py`: append-only JSONL per session, used for replay and recovery |
| Git workspaces | `workspace.py` (linked worktrees with exact leases), `candidate_git.py` |
| Processes and terminals | `managed_process.py`, `terminal_sessions.py`, `terminal_pty.py`, `terminal_host.py` |
| Optional Windows confinement | `windows_mxc.py`, `codex_sandbox.py`, `confined_workspace.py` ([guide](docs/guides/CONFINED_EXECUTION.md)) |
| Producer receipts | `assurance_plane/producer_receipts.py`: Ed25519 receipts binding a producer to output bytes |
| Context | `native_instructions.py`, `project_context_hooks.py` |

Limits stated plainly:

- `control_plane/`, `data_plane/` and `experience_plane/` are empty placeholders
  from the original five-plane plan ([SPEC-KERNEL-001](docs/SPEC-KERNEL-001.md)).
  There is no SQLite WAL store, landing notary or cockpit here; landing, leases,
  the cockpit and the TUI live in tare.tools.os.
- Project roots select working directories; they are **not a command sandbox**.
  Linked worktrees isolate Git changes; the optional confinement profiles above
  are Windows-only and off unless a host injects them.

## Install and test

You need Python 3.11 or newer, Git and [ripgrep](https://github.com/BurntSushi/ripgrep)
(`rg` on `PATH`; the search tests run it). The Kernel depends on
`tare.tools.agent-runtime`, and its hook tests use `tare.tools.specgraph`; neither is
on PyPI, so install them from GitHub first, at the revisions CI tests against
(`RUNTIME_REVISION` and `SPECGRAPH_REVISION` in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml); use the values there if they
have moved):

```bash
git clone https://github.com/augusto-scarvalho/tare.tools.kernel.git
cd tare.tools.kernel
python -m venv .venv
. .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "git+https://github.com/augusto-scarvalho/tare.tools.agent-runtime@ab2cce3b3b7d06c8bc83514b9bae0edc8b2af5c2"
python -m pip install "git+https://github.com/augusto-scarvalho/tare.tools.specgraph@10ae4d21fa69efa7e5903e5b214f17b2953ad359"
python -m pip install -e ".[dev,attachments,live-cache]"
python -m pytest tests/ -q
```

The tests are offline: they call no model provider and download nothing.
Contributor rules are in [AGENTS.md](AGENTS.md).

---

## Compute Plane & Execution Boundaries

### Canonical vendor hooks

The `tare-hook` leaf command applies one shared lifecycle policy across Claude
Code, Codex and Google Antigravity:

```text
tare-hook claude
tare-hook codex
tare-hook agy
```

Each vendor adapter only translates its native JSON envelope. The shared gate
intercepts native `Grep`/`grep_search` and shell `rg`/`grep`, directing all
repository lookup to `specgraph resolve <query> --format json`. That resolver
tries structure first and admits bounded literal recovery only for `MISS`,
`STALE`, `TIMEOUT`, or `UNAVAILABLE`, then indexes validated evidence for reuse.
While the local model server answers `/health`, the same gate denies frontier
subagent spawns (Claude `Agent`/`Task`, Codex `spawn_agent`) and points the
caller to `tare-ops explore`/`evidence`/`research`; an unreachable server
fails open and `TARE_LOCAL_MODELS_BYPASS=1` is the user's explicit override.
Set `TARE_LOCAL_MODELS` in the environment that starts the vendor CLI to the
operator's gateway URL; the portable default is `http://127.0.0.1:8080`.
Invalid or oversized pre-hook input is denied before tool execution. Recognized
edits also receive bounded advisory impact context before editing and a
post-edit staleness assessment after success; these notices never rebuild the
SpecGraph cache or grant authority.

Project-local configuration examples are in
[`docs/CANONICAL_TOOL_HOOKS.md`](docs/CANONICAL_TOOL_HOOKS.md).
Kernel also prepares the three manifests and an explicit `specgraph.yaml` as
the first repository-local dogfood deployment; no user-level vendor settings
are changed. Its neutral `search_paths` makes README, changelog and ordinary
documentation retrievable without classifying them as formal specifications or
ADRs. Expansion remains opt-in and should reuse the same qualified boundary.

Managed executions can additionally use
`compute_plane.backlog_work.materialize_backlog_work` to obtain the exact
bounded `backlog ground` envelope, then `check_backlog_scope` at runtime
checkpoints. Kernel only transports and observes this read-only fence; the host
maps its `CURRENT`, `DRIFTED`, or `UNAVAILABLE` result into Agent Runtime's
injected work guard. It never writes the backlog or grants Authority.

See [scoped project context hooks](docs/SCOPED_PROJECT_CONTEXT_HOOKS.md) for opt-in
SpecGraph advice, durable edit observations, bounded host diagnostics and the
exact receipt bytes exposed by managed Work progress.

`compute_plane.managed_work` provides the opt-in strict-path serializer and
progress observer substrate. It places immutable Work before the operator
instruction, runs only explicitly supplied verification commands in the
ephemeral workspace, hashes bounded tracked files and returns counts plus
digest identities. It does not interpret model prose or settle Evidence.

Managed execution qualifies a clean linked Git worktree and starts children
with a credential-minimal environment. This is an explicit boundary, not
a claim of hard filesystem or network isolation on every supported host.

The P2 prompt-cache laboratory also lives in the Compute Plane. Its fake provider
adapter consumes an admitted, detached Agent Runtime projection and produces
exact request bytes plus an immutable `PromptCachePlan`. The cache is bounded,
in-memory, disposable, and has no network or provider SDK; cache telemetry is
explicitly non-authoritative. Production provider caching remains a separate P3
vertical slice.

The operator-authorized P3a slice adds deliberately narrow live adapters:
OpenRouter Chat Completions with an exact provider pin, Amazon Bedrock
`InvokeModel` with an exact Anthropic body, and Bedrock Mantle Responses with
an explicit GPT-5.6 breakpoint and `store=false`. They preserve cold-cache
correctness, report cache reads/writes, latency and cost estimates, and remain
opt-in qualification paths rather than production defaults.

Inference goes through the native adapters (`native_models.py` for HTTP providers and
`native_cli.py` for official vendor CLIs). The former Aider and LittleCoder drivers were
retired; model selection and validation remain outside the adapters.

---

## Python API Quickstart

The native capability port accepts an explicit `capabilities` list, including
opt-in `search`. Search uses host-installed ripgrep with scoped, validated file
operands and literal/regex queries; it does not require a prior SpecGraph lookup
or grant a shell. It reports paths, line numbers, coverage and truncation. Roles
still bound the selection, and omitted selection preserves previous defaults.
Read/edit path checks are cooperative filesystem boundaries; executing a named
verifier is not a filesystem sandbox for candidate code.

```python
from pathlib import Path

from compute_plane.workspace import qualify_ephemeral_git_worktree

qualification = qualify_ephemeral_git_worktree(Path("/tmp/worktree_01"))
print(qualification.head)
```

---

## Ecosystem Family

| Repository | Role | Technology |
|---|---|---|
| **[`tare.tools.os`](https://github.com/augusto-scarvalho/tare.tools.os)** | Agent Operating System & Central Swarm Orchestrator; GPU node service (`node/`): model gateway, shared GPU lease | Python, AsyncIO, Submodules |
| **[`tare.tools.kernel`](https://github.com/augusto-scarvalho/tare.tools.kernel)** | Bounded model and tool execution (This Repo) | Python, Git Worktrees, ConPTY/PTY |
| **[`tare.tools.agent-runtime`](https://github.com/augusto-scarvalho/tare.tools.agent-runtime)** | Bounded tactical agent loop | Python (Pure Stdlib) |
| **[`tare.tools.specgraph`](https://github.com/augusto-scarvalho/tare.tools.specgraph)** | Spec-Driven Development & Causal Matrix | Python, Tree-Sitter, CPG AST |
| **[`tare.tools.backlog-graph`](https://github.com/augusto-scarvalho/tare.tools.backlog-graph)** | Deterministic DAG Task Backlog Engine | Python (Pure Stdlib), CAS |
| **[`tare.tools.library`](https://github.com/augusto-scarvalho/tare.tools.library)** | Technical library and knowledge records | Markdown, JSON |
| **[`tare.tools.dialog-engine`](https://github.com/augusto-scarvalho/tare.tools.dialog-engine)** | Topological Interaction & AST Dialog Graphs | Python, Statecharts |

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
