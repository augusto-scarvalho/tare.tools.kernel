# Vendor guidance: what a vendor CLI adds on its own

Tare calls Claude Code, Codex and AGY as inference transports and replaces their main prompt. Each CLI still adds
blocks of its own to the request. `compute_plane/vendor_guidance.py` reads those blocks from the installed CLI and
applies the owner's choices about them. Tare's OS reads each CLI when the TUI opens, shows the result in `/guidance`
and passes the saved choices to `VendorCliClient(..., vendor_guidance=...)`.

## Reading what a CLI adds

`read(endpoint, executable)` returns `{'cli', 'cli_version', 'note', 'items': [{'id', 'title', 'sent_as',
'controls', 'text'}]}`. It spends no tokens.

- **Claude Code** (2.1.322 or newer). Tare starts the CLI with its own command, but points the CLI at a recorder
  that tare runs on this machine (127.0.0.1) instead of Anthropic's servers. The CLI sends its login token with
  that request, so the token only travels inside the machine. Nothing reaches the provider.
  - The recorder reads only `Content-Length`, to frame the body, and keeps no request header. It answers the first
    request with an error, and tare stops the CLI as soon as that request is recorded (about 2 seconds).
  - A reading mod (the same mod as for choices, with `reading` on) marks every context block with its name and
    every injected attachment with its type. Blocks are therefore discovered, not listed, and a block type a newer
    CLI adds appears by itself.
  - The reading uses the request shape of a call without tools. On 2.1.322 the blocks Claude Code added were the
    same in every shape tare uses, apart from two tool descriptions of the rollback StructuredOutput codec.
- **Codex.** `codex debug prompt-input` renders the model's input under tare's configuration with an empty
  `CODEX_HOME` and sends nothing (under a second). Every block other than tare's own message is an item; its id is
  the block's leading tag.
- **AGY.** Nothing can be read; `read` returns no items.

Machine and day values become placeholders: `<working directory>`, `<account email>`, `<date>`, `<OS version>`,
`<model name>`, `<model id>`, `<knowledge cutoff>`, `<CLI version>`, `<time zone>` and `<skills folder>`. Two
readings of the same CLI therefore give the same texts. Measured on 2026-11-12: two readings of each CLI gave
identical item hashes.

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

`CLAUDE_CODE_DISABLE_ATTACHMENTS=1` removes none of them.

Codex 0.166.0: `skills_instructions`, `permissions instructions`, `collaboration_mode`, `environment_context`.

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

The command prints each item's id, controls, SHA-256 and text, and writes nothing. With choices, the CLI is read
with them applied, to see that they take. When a release changes how hooks or switches work, update
`CLAUDE_GUIDANCE_MINIMUM` or `CODEX_SWITCHES`.

## Limits

- Claude mods fail open: a hook that fails passes the vendor's text through, so a removal on Claude is not proved
  per request. `mod_listed_at_init` shows only that the plugin loaded.
- System blocks, the text around the blocks and tool descriptions can only be read.
- While a Claude choice is saved, calls tare normally runs with hooks off also run the hooks of Claude Code's
  built-in plugins. What those hooks do was checked only by comparing the recorded requests.
- The Claude reading points the CLI at a local address. Whether a CLI behaves differently for a custom address
  than for Anthropic's cannot be seen without a real call.
- AGY cannot be read.
- What a provider adds on its own servers is invisible to all of this.
