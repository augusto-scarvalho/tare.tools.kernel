# Vendor guidance: what a vendor CLI adds on its own

Tare calls Claude Code, Codex and AGY as inference transports and replaces their main prompt, but each CLI still adds
blocks of its own to the request. `compute_plane/vendor_guidance.py` lists those blocks as measured and applies the
owner's choices about them. Tare's OS shows the list in the TUI (`/guidance`) and passes the saved choices to
`VendorCliClient(..., vendor_guidance=...)`.

- `catalog()`: per CLI, the measured version, a note and the items: `id`, `title`, `sent_as` (where the block goes),
  `controls` (`remove`, `replace` or none) and `text`, the vendor's text with machine values as placeholders
  (`<account email>`, `<working directory>`, `<OS version>`, `<date>`, `<model name>`, ...).
- `transport_texts()`: the protocol texts tare itself sends, derived from `cli_request`; read-only.
- `validate(endpoint, choices)`: `{item id: None}` leaves a block out, `{item id: text}` sends that text instead;
  an unknown item, a control the item lacks or an empty replacement is a `ValueError`.

With no choice (None or `{}`) every CLI receives exactly the bytes it received before this existed
(`tests/test_vendor_guidance.py` pins them per codec).

## Measured blocks

Measured on 2026-11-12 on Windows with zero tokens.

### Claude Code 2.1.322

| Item | Sent as | Controls |
| --- | --- | --- |
| `billing_header` `x-anthropic-billing-header: cc_version=...` | first system block | none |
| `identity` "You are a Claude agent, built on Anthropic's Claude Agent SDK." | system block | none |
| `user_email` reminder carrying the account email | first user message (context block `userEmail`) | remove |
| `environment` working directory, platform, shell, OS version, downloaded-files rule | trailing system message (attachment `environment`) | remove, replace |
| `model` model name, model id, knowledge cutoff | trailing system message (attachment `model`) | remove, replace |
| `date` today's date | trailing system message (attachment `date`) | remove, replace |
| `structured_output_tool`, `deferred_tool_placeholder` tool descriptions | StructuredOutput codec only (rollback) | none |

`CLAUDE_CODE_DISABLE_ATTACHMENTS=1` removes none of them.

### Codex 0.166.0

| Item | Sent as | Controls |
| --- | --- | --- |
| `skills_instructions` `<skills_instructions>` (plus any skill installed in `CODEX_HOME`) | developer message | remove |
| `permissions_instructions` `<permissions instructions>` | developer message | remove |
| `collaboration_mode` `<collaboration_mode>` | developer message | remove |
| `environment_context` `<environment_context>`: working directory, shell, date, time zone | user message | remove |

### AGY 1.5.6

No official way to see what AGY adds was found, so nothing is listed and any choice is refused.

## How each control works

**Claude Code.** The choices travel as a mod written into the call's temporary folder (`tare-guidance/`:
`.claude-plugin/plugin.json`, `hooks/hooks.json`, `hooks/register.ts` with the choices embedded) and loaded with
`--plugin-dir`. Its `prompt.context` hook drops the `userEmail` block; its `prompt.attachment` hook answers
`{text: null}` (leave out) or the owner's text for `environment`, `model` and `date`. The call is refused before
anything is sent (`CLI_GUIDANCE_UNSUPPORTED`, `not_started`) when the CLI is older than 2.1.322, the release
measured, or when its `--help` lacks `--plugin-dir` (`CLI_CAPABILITY_UNAVAILABLE`, `missing_cli_switches`).

A mod's hooks do not run while settings carry `disableAllHooks: true`. The native codec with tools (Claude decisions)
already runs with hooks on, for tare's own PreToolUse hook. Calls without tools take the same native codec with an
empty catalog (heads-up notes, plain answers); they, summaries and the rollback toolless and StructuredOutput codecs
set `disableAllHooks: true`. When a Claude choice is saved, tare drops that key from those calls (keeping
`enabledPlugins: {}`), so the mod runs in every shape. A call without a choice is unchanged.

The trade: hooks on also run the hooks of the plugins Claude Code builds in, which neither `--setting-sources ''` nor
`enabledPlugins: {}` excludes. On 2.1.322 the init row lists `cc-plugin-agents-md`, `cc-plugin-telemetry`,
`cc-plugin-privacy` and `cc-plugin-plugin-authoring` in every codec. The native codec with tools already runs their
hooks on every call; the other shapes run them only while the owner has a Claude choice saved. Measured on 2.1.322,
the guided request in each shape equals the unguided one except the chosen blocks, the temp folder name, the billing
header hash and the session id Claude Code generates per run.

The receipt's `vendor_guidance` holds each chosen item's control (`choices`), the SHA-256 of the exact choices,
`hooks_turned_on` (whether tare turned hooks on for this call) and `mod_listed_at_init`: whether the stream's init
row listed `tare-guidance` among its `plugins` (name, path, source, version). The init row says nothing about the
plugin's hooks.

**Codex.** Codex can leave a block out but not rewrite it. `prepare` adds `-c` switches
(`skills.include_instructions=false`, `include_permissions_instructions=false`,
`include_collaboration_mode_instructions=false`, `include_environment_context=false`). An older Codex may ignore a
key it does not know, so once per Codex version, before the first call with choices, tare runs
`codex debug prompt-input` (nothing is sent; an empty `CODEX_HOME`, so no login or user configuration is read) under
the call's configuration without and with the switches. Each chosen block's tag must show without its switch and
not with it; otherwise the call is refused (`CLI_GUIDANCE_UNSUPPORTED`, `unproved`).

## Re-measuring for a new CLI version

`tools/vendor_guidance_capture.py` reproduces both measurements, spending no tokens and writing nothing to disk:

    python tools/vendor_guidance_capture.py claude <path to claude.exe> native|plain|summary|toolless|structured ['{"user_email": null}']
    python tools/vendor_guidance_capture.py codex <path to codex.exe> ['{"environment_context": null}']

For Claude it runs tare's own `VendorCliClient` (same flags, input and mod) against a loopback recorder that answers
400, so nothing reaches the provider; the recorder reads only `Content-Length` to frame the body and stores no
request header. For Codex it uses `codex debug prompt-input`. It prints whether each catalog item is present as
catalogued (placeholders match any value), whether each chosen item is gone or replaced, and every block of the
request with the email replaced. When a block changed or a new one appears, update the texts and
`measured_version` in `vendor_guidance.py` and the tables above; when a release changes how hooks or switches work,
update `CLAUDE_GUIDANCE_MINIMUM` or the Codex keys.

`plain` is a call without tools. `claude()` returns the recorded body, so a maintainer can compare an unguided and a
guided request in memory.

Results on 2026-11-12, Claude Code 2.1.322: in the native, plain, summary, toolless and StructuredOutput shapes, with
`user_email`, `environment` and `model` removed and `date` replaced, the three blocks were absent, the trailing
system message was only the replacement, `tare-guidance` was listed at init, and the request otherwise equalled the
unguided one (apart from the temp folder name, billing hash and per-run session id). Codex 0.166.0 with all four
removed: `unproved: []` and `prompt-input` shows only tare's request.

## Limits

- Claude mods fail open: a hook that fails passes the vendor's text through, so a removal on Claude is not proved
  per request. `mod_listed_at_init` shows only that the plugin loaded.
- The identity line, the billing header and the tool descriptions are not controlled.
- While a Claude choice is saved, calls tare normally runs with hooks off (summaries, calls without tools) also run
  the hooks of Claude Code's built-in plugins (see above). What those hooks do was checked only by comparing the
  recorded requests.
- AGY is not inspectable.
- What a provider adds on its own servers is invisible to all of this.
- The texts were measured on Windows; other platforms differ in the environment blocks.
