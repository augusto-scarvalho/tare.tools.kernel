# Vendor guidance: what a vendor CLI adds on its own

Tare calls Claude Code, Codex and AGY as inference transports and replaces their main prompt. Each CLI still adds
blocks of its own to the request. `compute_plane/vendor_guidance.py` reads those blocks from the installed CLI and
applies the owner's choices about them. Tare's OS reads each CLI when the TUI opens, shows the result in `/guidance`
and passes the saved choices to `VendorCliClient(..., vendor_guidance=...)`.

## Reading what a CLI adds

There are two readings. Both return `{'cli', 'cli_version', 'note', 'items': [{'id', 'title', 'sent_as',
'controls', 'text'}]}`.

- `read(endpoint, executable)` spends no tokens. Tare's OS runs it each time the TUI opens.
- `read_real(endpoint, executable, model=...)` makes one minimal real call and spends a few hundred tokens. It is a
  separate function, so nothing spends tokens by accident. Tare's OS runs it once per CLI version, on that CLI's
  cheapest model at low effort.

### Claude Code (2.1.322 or newer)

- **Zero-token reading.** Tare starts the CLI with its own command, but points it at a recorder that tare runs on
  this machine (127.0.0.1) instead of Anthropic's servers.
  - The CLI sends its login token with that request, so the token only travels inside the machine, and nothing
    reaches the provider.
  - The recorder reads only `Content-Length`, to frame the body, and keeps no request header. It refuses the first
    request, and tare stops the CLI as soon as that request is recorded (about 2 seconds).
  - Tare's mod, with `reading` on, marks every context block with its name and every injected attachment with its
    type. Blocks are therefore discovered, not listed, and a block type a newer CLI adds appears by itself.
  - The reading uses the request shape of a call without tools. On 2.1.322 the blocks Claude Code added were the
    same in every shape tare uses, apart from two tool descriptions of the rollback StructuredOutput codec.
- **Real reading.** One call with tare's command and tare's mod, with `recording` on, on the cheapest model at low
  effort. The mod writes every context block and attachment, as sent, to a file of its own: hooks run
  concurrently, so writes to one file interleave.
  - Each item says whether the zero-token reading found it, under `local`: `same`, `differs` or `absent`.
  - `left_out` names what a saved removal took out, from the mod's own record.
  - A context block the engine replaces with its own attachment is not listed as a block. `currentDate`, for
    example, is sent as the `date` attachment.
- **What only the real reading sees.** On 2.1.322 every piece matched text for text, except one attachment that
  only a real call carries: `total_tokens_reminder` (`<total_tokens>… tokens left</total_tokens>`).
- **Removing it works.** A real reading with `attachment:total_tokens_reminder` removed recorded that attachment as
  left out, and the request was 23 tokens smaller.
- **What neither reading can see.** The mod cannot see the system blocks before tare's system text (the billing
  header and the identity line), so the real reading lists only what the mod sees. The local reading shows those
  system blocks from the request itself.

### Codex

Codex has a zero-token reading only, and on 0.166.0 it was faithful. `codex debug prompt-input` renders the
model's input as tare's `exec` call would, and sends nothing (about 1 second). It runs:

- with an empty `CODEX_HOME` holding a copy of each installed skill's `SKILL.md`, its `agents/*.yaml` and its plugin
  manifest. A skill's `allow_implicit_invocation: false` keeps it out of the list, and a plugin's name prefixes the
  skill's. A skill folder can link to a whole project, so nothing else is copied. Neither the login nor the owner's
  `config.toml` is read.
- with exec's approval policy (`never`) and read-only sandbox.

The removal proof uses the same rendering. Every block other than tare's own message is an item; its id is the
block's leading tag.

Measured against Codex's own session record of a real call (a call made without `--ephemeral`, then that record
removed): every block was equal. Base instructions equalled tare's system text, and the four blocks matched:

| Block | Characters |
| --- | --- |
| skills list | 16,170 |
| permissions | 341 |
| collaboration mode | 920 |
| environment | 427 |

The earlier reading, with an empty `CODEX_HOME` and no approval policy, was wrong: it listed 1,938 characters of
skills and 3,882 of permissions.

### AGY

AGY has a real reading only. One minimal call with tare's AGY command, then tare reads the conversation that call
created, named by the `conversation_id` the call reports, and never another one.

- **Where AGY keeps it.** In `~/.gemini/antigravity-cli/conversations/<id>.db` (SQLite), as `gen_metadata.data`.
- **How it is read.** That record is a protobuf with no published schema, so it is read by the marks in its text,
  never by field numbers. Without those marks the reading raises "could not be read" instead of guessing.
- **What it finds.** Tare's system text inside `<System Prompt>…</System Prompt>`. Tare's request inside
  `<USER_REQUEST>…</USER_REQUEST>`, followed by AGY's blocks (`request:<TAG>`): `ADDITIONAL_METADATA` (the local
  time) and `USER_SETTINGS_CHANGE` (the model selection). The tools: `finish`, whose schema is tare's but whose
  description is AGY's, and `manage_task`, which AGY adds.
- **Controls.** Every AGY item can only be read.
- **Starting point.** `AGY_STARTING_POINT` holds what one AGY 1.5.7 call showed on 2026-11-12. It is shown until a
  reading of the installed AGY replaces it.
- **What it cannot see.** The same record holds the owner's saved permission rules. Nothing shows whether they
  reach the model, so they are neither read nor listed.

Machine and day values become placeholders: `<working directory>`, `<account email>`, `<date>`, `<local time>`,
`<OS version>`, `<model name>`, `<model id>`, `<knowledge cutoff>`, `<tokens>`, `<CLI version>`, `<time zone>`
and `<skills folder>`. Two readings of the same CLI therefore give the same texts. Measured on 2026-11-12: two
zero-token readings each of Claude Code and Codex gave identical item hashes, and two real AGY readings gave items
of identical lengths.

An edited Claude block is written with those placeholders, and when it is sent the mod puts back the values
Claude Code wrote in that same block (`FILL`): "Today is `<date>`." goes out with the real date. A placeholder
whose value is not in that block goes out as written.

### Item ids and controls

The shape of an id decides what the owner may do with it, so a new block needs no code.

| Id | Controls |
| --- | --- |
| Claude `context:<name>`: a context block (e.g. `context:userEmail`) | remove |
| Claude `attachment:<type>`: an injected attachment (e.g. `attachment:environment`) | remove, replace |
| Claude `system:<n>`: a system block other than tare's | read only |
| Claude `message:<n>`: text around the marked blocks | read only |
| Claude `tool:<name>`: a tool description other than tare's | read only |
| Codex `skills_instructions`, `permissions instructions`, `collaboration_mode`, `environment_context` | remove |
| Codex any other leading tag, or `block:<n>` for a block without one | read only |
| AGY `system:frame`, `request:frame`, `request:<TAG>`, `tool:<name>` | read only |

### One reading, as an example (2026-11-12, Windows)

Claude Code 2.1.322:

- `system:1`: the billing header line.
- `system:2`: "You are a Claude agent, built on Anthropic's Claude Agent SDK."
- `context:userEmail`: the account email.
- `attachment:session_context`: the frame around the context blocks. Removing it also drops the email block.
- `message:1`: the `<system-reminder>` wrapper.
- `attachment:environment`: working directory, platform, shell, OS version and the downloaded-files rule.
- `attachment:model`: model name, id and knowledge cutoff.
- `attachment:date`: today's date.
- In a real call only: `attachment:total_tokens_reminder`.

`CLAUDE_CODE_DISABLE_ATTACHMENTS=1` removes none of them.

Codex 0.166.0: `skills_instructions`, `permissions instructions`, `collaboration_mode`, `environment_context`.

AGY 1.5.7 (real reading): `system:frame`, `request:frame`, `request:ADDITIONAL_METADATA`,
`request:USER_SETTINGS_CHANGE`, `tool:finish`, `tool:manage_task`.

## Which model a Claude Code alias names

Claude Code has no command that lists its models, but it resolves a family alias (`--model fable`) to the newest
model of that family the installed CLI knows. `claude_models(executable, aliases, *, seconds=60)` reads that
resolution with zero tokens and returns `{'cli_version', 'models': {alias: model id or None}}`. The caller passes
the aliases; the Kernel holds no family list. Tare's OS runs it once per CLI version.

- **How.** Each alias is one start of tare's command for a call without tools (hooks off, no mod), pointed at the
  recorder of the zero-token reading. The recorder refuses the first request, so nothing reaches Anthropic.
- **What is kept.** Only the `model` field of that request. The recorder drops the rest of the body, which holds
  account details, and reads no header but `Content-Length`.
- **None.** An alias comes back as a model id only when the request names `claude-<alias>-<version>`. Claude Code
  sends an alias it does not know unchanged (measured: `mythos`, `zzz`), and an exact id as given, so neither
  counts as resolved. A start that sends no request is None too.
- **One alias per start.** Tare's command takes one `--model`, so the aliases are read in turn: four took about
  6 seconds on Windows.
- **Too old.** A CLI older than 2.1.322 is refused with the same `ValueError` as the reading.

Measured on 2026-11-14 with Claude Code 2.1.325 on Windows: `fable` named `claude-fable-5-2`, `opus`
`claude-opus-5-5`, `sonnet` `claude-sonnet-5-5` and `haiku` `claude-haiku-5-5`. `Fable`, `opus[1m]` and `default`
were also sent as ids, but not as `claude-<alias>-…`, so they come back as None: pass the family name as the id
spells it.

## How each control works

**Claude Code.** The choices travel as a mod written into the call's temporary folder (`tare-guidance/`) and loaded
with `--plugin-dir`. Its `prompt.context` hook drops the chosen context blocks. Its `prompt.attachment` hook
answers `{text: null}` (leave out) or the owner's text for the chosen attachment types. The call is refused before
anything is sent when:

- the CLI is older than 2.1.322 (`CLI_GUIDANCE_UNSUPPORTED`), the release first measured;
- its `--help` lacks `--plugin-dir` (`CLI_CAPABILITY_UNAVAILABLE`, `missing_cli_switches`).

A mod's hooks do not run while settings carry `disableAllHooks: true`. The native codec with tools (Claude
decisions) already runs with hooks on, for tare's own PreToolUse hook. Other calls set `disableAllHooks: true`:

- calls without tools, which take the same native codec with an empty catalog (heads-up notes, plain answers);
- summaries;
- the rollback toolless and StructuredOutput codecs.

When a Claude choice is saved, tare drops that key from those calls and keeps `enabledPlugins: {}`, so the mod runs
in every shape. A call without a choice is unchanged.

The trade: with hooks on, the plugins built into Claude Code also run their hooks; neither `--setting-sources ''`
nor `enabledPlugins: {}` excludes them. On 2.1.322 the init row lists `cc-plugin-agents-md`, `cc-plugin-telemetry`,
`cc-plugin-privacy` and `cc-plugin-plugin-authoring` in every codec. The native codec with tools already runs their
hooks on every call; the other shapes run them only while the owner has a Claude choice saved.

Measured on 2.1.322 in each of the five shapes, the guided request equals the unguided one except for:

- the chosen blocks;
- the temp folder name;
- the billing header hash;
- the session id Claude Code generates per run.

The receipt's `vendor_guidance` holds:

- `choices`: each chosen item's control;
- `sha256`: the digest of the exact choices;
- `hooks_turned_on`: whether tare turned hooks on for this call;
- `mod_listed_at_init`: whether the stream's init row listed `tare-guidance` among its `plugins` (name, path,
  source, version). The init row says nothing about the plugin's hooks.

**Codex.** Codex can leave a block out but not rewrite it. `prepare` adds the `-c` switch of each chosen block:

- `skills.include_instructions=false`
- `include_permissions_instructions=false`
- `include_collaboration_mode_instructions=false`
- `include_environment_context=false`

An older Codex may ignore a key it does not know. So once per Codex version, before the first call with choices,
tare runs `codex debug prompt-input` without and with the switches. Each chosen block's tag must show without its
switch and not with it; otherwise the call is refused (`CLI_GUIDANCE_UNSUPPORTED`, `unproved`).

**AGY.** Any choice is refused.

## Reading by hand

    python tools/vendor_guidance_capture.py claude <path to claude.exe> ['{"context:userEmail": null}']
    python tools/vendor_guidance_capture.py codex <path to codex.exe> ['{"environment_context": null}']
    python tools/vendor_guidance_capture.py real claude <path to claude.exe> claude-haiku-5-5
    python tools/vendor_guidance_capture.py real agy <path to agy.exe> gemini-3.8-flash-low
    python tools/vendor_guidance_capture.py models <path to claude.exe> fable opus sonnet haiku

The first two and `models` spend no tokens; `real` makes one small real call. The command prints each item's id,
controls, SHA-256 and text (`models`: each alias and its model id), and writes nothing. With choices, the CLI is read with them applied, to see that they take.
When a release changes how hooks or switches work, update `CLAUDE_GUIDANCE_MINIMUM` or `CODEX_SWITCHES`.

## Limits

- Claude mods fail open: a hook that fails passes the vendor's text through, so a removal on Claude is not proved
  per request. `mod_listed_at_init` shows only that the plugin loaded.
- System blocks, the text around the blocks and tool descriptions can only be read.
- While a Claude choice is saved, calls tare normally runs with hooks off also run the hooks of Claude Code's
  built-in plugins. What those hooks do was checked only by comparing the recorded requests.
- The zero-token Claude reading points the CLI at a local address and misses what Claude Code adds only to a
  real call (on 2.1.322, `total_tokens_reminder`); the real reading, once per version, finds it. The real reading
  sees only what tare's mod sees, not the system blocks before tare's system text.
- AGY is read only from its own record of a real call, by the marks in its text; a release that changes those marks
  cannot be read until tare learns them. What AGY sends that its record leaves out cannot be seen, and nothing can
  be changed.
- An alias's model id says only that the installed Claude Code names it; whether the account may use that model
  is not checked. The alias reading was measured only with Claude Code 2.1.325 on Windows.
- What a provider adds on its own servers is invisible to all of this.
