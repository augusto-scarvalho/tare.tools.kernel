# Native model adapters — M2a through M2d

Gemini's OpenAI-compatible usage has a separate
[qualified counter contract](GEMINI_COMPAT_USAGE.md), including thinking and
unknown cache details. HTTP compatibility does not establish OpenAI token semantics.

This guide describes the native adapter interfaces and their bounded behavior.

Terminal output is paged in decoded characters per stream. Each successful
`terminal_read` supplies `next_read` arguments to copy for the next observation;
`null` means a finished process has no unread captured output. A finished process
can still have unread pages. Do not calculate future offsets or batch speculative
reads beyond the observed end. Continuation hints do not count as new progress,
and invalid cursors remain refused.
The current tactical loop is owned by `tare.tools.agent-runtime`; its dependency
is installed or selected explicitly by the caller and by CI.

The original 2026-09-10 qualification and its environment remain historical
observations in [private record custody](REPOSITORY_CONTENT.md). Current source
validation and live provider qualification are separate operations.

## Host-selected verification bounds

Host confinement binding also accepts `unbounded_verification=True` for an
explicit operator choice to remove a verification process cap. It is a host-only
boolean, absent from the model's tools. Ordinary command bounds remain finite;
verification still uses its enclosing operation deadline, process factory,
mediation and cancellation. The default is false and legacy bindings keep their
existing limits.

## Claude summary count estimates

Claude formatter and plain-summary envelopes name their quoted history differently.
When that codec changes, the context estimator prices the complete summary with
the verified prior usage-per-byte anchor and its existing 25% margin, rather than
adding the same history to the prior measured input again. This remains an
estimate: unknown usage cannot calibrate it, original limits stay in force and
the transmitted requests and cached prefixes are unchanged.

## Ordinary delivery format feedback

An untyped conversational delivery must be the only call and contain exactly a
nonempty string `summary`. A completed tools response with an invalid delivery
shape receives format feedback through the existing recovery turn. The entire
batch is discarded, including otherwise valid actions. Its original provider
response, usage and context remain recorded; the next model turn consumes the
same call, time, spend and stagnation budgets. No HTTP request is replayed and no
budget is renewed. Typed delivery validation retains its separate rules below.

## Bounded typed-delivery correction (0.9.60)

A citation correction was added in 0.9.65; missing-field correction remains available from 0.9.60.

A complete typed proposal missing required fields, citing a source not observed in this session, or naming a locally verified ignored output may receive one correction turn
when at least two calls remain in the original finite budget, or that budget has no call ceiling.
An unbounded call allowance still permits only one format correction, including after resume.
The host returns only field/error
diagnostics and the existing delivery schema; it accepts no artifact or effects from that batch.
A repeated rejection or a rejection on the last call remains a failure. Transport interruptions,
unknown execution, changed-source and authority refusals and provider failures never take this path. A citation correction must use an already observed reference or omit it and preserve the evidence gap; every check runs again before any proposal is accepted. Original usage and context stay recorded.

## HTTP API providers (0.9.50)

Use `ModelProfile` with an exact model, explicit endpoint and limits, and
`NativeHttpClient(endpoint, credential=...)` with an operator-supplied API key.
These API routes do not discover credentials, log in to ChatGPT, retry through
another provider, or create a second runtime. NativeSession still owns tools.

- `openai-responses`: endpoint `https://api.openai.com/v1`, POST `/responses`.
  Reuses `native_responses` and its SSE decoder. Unlike the separate Codex
  subscription protocol, it submits `max_output_tokens=output_tokens` and honors
  the adapter's `streaming` setting. `prompt_cache_key` is session-stable unless
  explicitly supplied. Generated API keys fit its 64-character limit. Cache
  retention is explicit; affinity is not a cache hit.
  Streaming replay retains reasoning from `response.output_item.done`, including
  its opaque `encrypted_content`, if the terminal snapshot re-encrypts that same
  item. Identity, summary, messages and tool arguments must still match exactly.
- The same `openai-responses` codec supports the exact OpenRouter endpoint
  `https://openrouter.ai/api/v1`. Only there, profiles may also specify `provider`,
  `transforms` and `session_id`. Routing identity and `prompt_cache_key` stay
  stable for the conversation, including resume; opaque reasoning is replayed.
  Provider policy is supplied by the host, with no automatic endpoint fallback.
  This does not admit gateway parameters on direct OpenAI or lookalike URLs.
- Gemini: `openai-chat`, endpoint
  `https://generativelanguage.googleapis.com/v1beta/openai`, as documented in
  <https://ai.google.dev/gemini-api/docs/openai>. Original assistant/tool-call
  extensions (including opaque thought signatures) are retained for replay;
  streamed extensions are merged without interpreting them. Conflicting opaque
  fragments fail closed rather than silently discarding a signature. Tool result
  IDs stay linked through the common history validator. No SDK is required.
- `anthropic-messages`: endpoint `https://api.anthropic.com/v1`, POST `/messages`.
  JSON only initially: `streaming=True` is rejected before dispatch. Text, local
  tool use/results, configured inline image media types, signed thinking and
  redacted thinking replay through a small common-history conversion. Hosted
  tools and unknown content blocks are refused as a whole. Exact response model,
  malformed responses and output-limit batches retain the common safety guards.
  Use `request_headers=lambda: {'anthropic-version': '2023-06-01'}`; optional
  `anthropic-workspace-id` is also validated. Authentication remains the common
  Bearer credential, not a new secret-header mechanism. Never put keys in these
  nonsecret headers. `cache_system=True` requires a nonblank
  `cache_support_source` declaring support for the selected API/model (see
  <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>).
  It emits an explicit ephemeral system breakpoint; existing cache controls in
  replayed blocks and explicitly configured top-level cache controls are retained.
  Cache hits, remote cancellation and billing are not inferred.
  `output_config` supports only `effort` (`low`, `medium`, `high`, `xhigh`, `max`),
  following [Anthropic's effort contract](https://platform.claude.com/docs/en/build-with-claude/effort).
  The adapter passes it unchanged; role owners choose a level before creating
  a session. It never enables thinking or changes cache controls implicitly.
  On the last model call with `force_final_delivery=True`, `tool_choice` remains
  `{'type': 'auto'}` when thinking is omitted (including an adaptive provider
  default) or is not explicitly disabled. Only a profile parameter object
  `thinking: {'type': 'disabled'}` permits forcing
  `{'type': 'tool', 'name': 'tare_deliver'}`. Non-object thinking values do not
  count as explicit disabling. The existing last-call budget warning is preserved;
  the adapter never toggles thinking or changes other parameters to force delivery.
  OpenAI Chat still forces its nested named function choice; OpenAI Responses and
  Codex Responses still force `{'type': 'function', 'name': 'tare_deliver'}`.
- `gemini-generate-content`: Gemini's own API, POST `/models/<model>:generateContent` under the profile endpoint
  (OpenCode: `https://opencode.ai/inference/go/google/v1beta`), with the common Bearer credential. JSON only:
  `streaming=True` is rejected before dispatch. The system prompt is `systemInstruction`, the output limit is
  `generationConfig.maxOutputTokens`, and the profile parameters `temperature`, `topP`, `topK`, `stopSequences`
  and `thinkingConfig` go into `generationConfig`; any other parameter is refused. Tools are
  `functionDeclarations` with `parametersJsonSchema`; the choice is `AUTO`, or `ANY` restricted to `tare_deliver`
  on a forced last call. The model's original parts, with their opaque `thoughtSignature`, are replayed unchanged.
  A tool result is a `functionResponse` (`output`, or `error` for a failed effect) carrying the call's id when
  the API gave one; a call without an id is identified by its position in the reply. The results of one batch
  share a user turn that holds nothing else: OpenCode refused a turn mixing a result with text (HTTP 400,
  "Requests ending with a model turn are not supported", gemini-4-argon, 2026-11-10), so text after a result
  starts its own turn. Thought summaries are not answer text. Code execution, hosted search and any other part
  kind refuse the whole reply. `finishReason` other than `STOP` is an incomplete reply without effects
  (`MAX_TOKENS` with calls asks for smaller steps). `modelVersion` is the observed model and `usageMetadata` the
  raw usage; the context anchor is `promptTokenCount`, which includes cached content. Only the
  `opencode-exclusive` usage contract is admitted for it so far; Google's own endpoint, its `x-goog-api-key`
  header and its native usage layout are not qualified. Gemini's implicit cache was read only some 30 to 60
  seconds after a prefix was first sent (same probe), so short sessions read little of it.

Anthropic usage uses the distinct `anthropic-exclusive` contract: `input_tokens`,
`cache_read_input_tokens`, and `cache_creation_input_tokens` are disjoint input
categories. Missing counters remain unknown, including totals that depend on
missing cache counters; no tariff is supplied automatically. `openai-responses`
uses `responses-inclusive`. These contracts are not inferred from HTTP compatibility.
Use `counter_protocol='utf8-envelope-estimate-v1'` where no measured counter is
configured; usage anchors include cached input only when all required counters
are present. Qualification here is offline fixtures, not real API calls.

## Implemented boundary

AGY submissions use a conservative **160 KiB complete-envelope operating bound**.
This is not Gemini's native context window or an exact vendor limit. A 176 KB
input-integrity diagnostic passed on AGY 1.4.12; an ordinary 338 KB audit reported
truncated user context. Larger submissions now fail before CLI startup with
`CLI_INPUT_BYTE_LIMIT_EXCEEDED`, preserving their complete bytes and an unstarted
transport receipt. Do not shorten source evidence silently or treat that refusal
as a completed review. AGY decisions admit at most four external actions and ask
for bounded read ranges; the workflow owner controls compaction and recovery.

Every Claude CLI decision (all models; summaries excepted) uses native tool calls.
The CLI keeps built-in tools disabled and reads no user settings; Tare declares its
catalog through a declaration-only stdio MCP server (`native_cli_mcp.py`) and a
PreToolUse hook (`native_cli_hook.py claude`) records each call, refuses it and
ends the turn. A reply is accepted only when the session declared exactly
`mcp__tare__*`, one API message was produced, and the tool_use blocks, hook records
and the CLI's `permission_denials` name the same calls, all refused. Tare then
executes them through its own tools. Hook commands are quoted with forward slashes:
Claude Code runs them through a shell, and unquoted Windows paths failed silently
into the CLI's own permission refusal. Live replays of a failed Sonnet implementer
turn (2026-11-02): the toolless JSON codec gave 0-2 of 4, with `<invoke>` markup
and runaway loops at medium, high and max effort; StructuredOutput 1 of 6; native
calls 33 of 33 decisions over 1-8 turns with 12 well-formed edits. Opus, Haiku and
Fable 5.1 were 4 of 4 natively. XML written into text is still never an action.
The toolless and formatter codecs below remain only as a one-line rollback.

A later one-turn formatter trial prevented extra CLI inference and recovered one
refused-action batch, but the productive continuation still exhausted bounded
format correction after partial edits. Its patch was preserved; productive
qualification failed again. The experimental codec and repair were not promoted.
A fixture success or repeated retry does not change that outcome.

Malformed JSON diagnostics retain only a finite parser category and location,
never response text, arbitrary keys, arguments or exception messages.

Claude input is exactly one stream-json user event, with the historical records
quoted in separate text blocks. The changing runtime allowance and final-choice
reminder follow the stable cache breakpoint. The observed CLI 2.1.314 uses 1h
cache writes; the explicit breakpoint uses the same TTL so a shorter-lived block
does not precede the CLI's longer-lived automatic breakpoint. The bounded live
Sonnet-medium toolless pilot completed one read and one delivery with 4,140
cache-read tokens on its second call. The separate typed-format pilot observed
7,129 reads, but its productive qualification failed. These qualify only their
observed cache slices, not model quality or every CLI/model version. Saved
journals and budgets remain unchanged. A codec
change can cause one cold request and must not be called a cache hit. Cache reuse
is observed through returned usage, not inferred from retained history.

Malformed Claude proposal data from an otherwise valid single CLI turn returns
the existing `INVALID_CLI_REPLY` feedback to the runtime. The transport executes
no proposed action and performs no inference retry. The runtime retains its
call, time and stagnation limits. Original usage remains attributed; custody
failures, summary failures and invalid enforced final choices still stop.

An unknown tool result remains in the journal until its owner proves a known
outcome. The provider-history renderer keeps that tool call pending across the
unknown observation and emits the later reconciled result once. Unresolved calls,
unmatched/duplicate conclusive results and user input crossing a pending effect
remain blocked. Rendering never repeats an effect, changes the saved session or
renews a budget; the native Responses cache identity stays with the same session.

The search schema requires a nonempty content query, matching its executor.
For OpenRouter Chat, an observed argument-schema refusal in retained history
requests `parallel_tool_calls=false` on subsequent model calls unless the profile
explicitly configures that parameter. This is a host repair policy: messages,
tool schemas, profile identity and session affinity remain unchanged. It grants
no effect authority, makes no retry itself and does not guarantee provider compliance.
`glob` filters candidate files and does not replace the content query. Constrained
decisions therefore cannot silently choose an empty query for filename discovery;
the executor still validates the query and its byte/line limits independently.
If a matching long line overflows JSON output, proved cleanup permits one fixed
read-only repeat of that batch with bounded location output. Returned rows have
path/line, empty text and `text_truncated=true`; the result is explicitly partial
with `includes_location_only=true`. No source text or absence is attested, no
output budget is raised, and cancellation/uncertain cleanup never retries.
An oversized filename inventory may instead search its captured prefix after
proved cleanup: only complete NUL-delimited names, at most 4096 files, with
`inventory_complete=false` and an always-partial result. The trusted read-only
caller consumes bounded stdout only; stderr, clipped names, effects, cancellation,
timeouts, isolation violations and unproved cleanup never become partial success.
Successfully consumed partial reads retain their exact process receipts under
`process.recovered_output_limits`; `process.interruption` is reserved for an
unresolved search failure. `PARTIAL` permits using the returned evidence and
narrowing the remaining scope, without proving absence. Before emitting a text
page, inspect the bounded rg end records and discard matches from any file
reported as binary, even when the requested page contains only one match.

Source reads keep their 24,000-byte text page limit. A longer single line returns
`partial_line`, its 1-based Unicode `start_column`/`end_column` and a continuation
`next_start`/`next_start_column`. Pass the latter as `start`/`start_column` to read
the next fragment; ordinary line pagination is unchanged. The complete file hash
still binds its observed version, not a claim of full content inspection. JSON
pointer and source-line coordinates cannot be mixed. Numbered lines drop the file's
last line break, so a page that reaches the end of the file also returns
`final_newline`, whether the file ends with one; the tool description asks the model
to keep it unless the task changes it. Without it, Claude Haiku 5.5 took
`value = 1` for a file with no final newline and replaced it with `value = 2\n`,
doubling the line break.
An explicitly named source can be up to 8 MiB, including through an approved
external-folder reader. An explicit-file search shares that input bound; broad
directory scans retain their 2 MiB per-file policy. If an explicit source is
excluded, search returns `PARTIAL` with zero inspected files, rather than `MISS`.
Output pages, link/path checks and the existing edit bound remain unchanged.
JSON field selection includes the entire selected value; it does not paginate
nested collections. An object projection larger than the text budget returns
the existing object directory with `projection_complete=false` and an explicit
notice that values were not inspected. Move the pointer into the collection and
page its items. Ordinary projections and oversized array/scalar refusals retain
their existing behavior; metadata pagination never attests the omitted values.

Final delivery preserves assistant prose as well as its delivery summary when
they differ. A summary must not replace the detailed answer the user requested;
identical text is emitted once. Typed artifact payloads and original provider
messages retain their original content and validation.
Codex HTTP reconnect notices preceding a successfully completed single turn are
retained as transport diagnostics instead of discarding the valid final answer.
This does not retry inference in Tare or attest hidden provider attempts/billing.
Final failures, unknown or coded errors, post-completion diagnostics, extra turns
and unexpected tools continue to fail the exchange. The maintained
[CLI JSONL interface](https://learn.chatgpt.com/docs/non-interactive-mode) has
separate error and turn-completion events; the bounded reconnect recognition is
qualified by offline fixtures and the observed 503 recovery receipt.

### Codex CLI object arguments (0.9.28–0.9.29)

The current `native-json-object/4` wire contract retains the v3 envelope below.
Historically, v3 introduced `action` for the catalog name so it sorts before
`arguments`, including when the CLI canonicalizes JSON. Tool selection therefore
precedes the differing argument schemas instead of committing to a shared shape
before choosing its name. Host tool calls and artifact schemas are unchanged;
historical v1/v2 decisions keep their original `name` field and decoding rules.

Codex action decisions carry native argument objects under the existing bounded
output schema. This removes a second model-authored JSON string and its escaping
failures. Each catalog entry shows the actual wire schema, including required
nullable placeholders for optional fields; the owner schema remains unchanged.
Optional nonnullable fields use null as an omission placeholder. Nullable optional
fields instead use `{"omitted":true}` for absence or `{"value":...}` for a value,
including explicit null; required nullable fields retain their raw values. This
preserves defaults such as an omitted command timeout versus explicit background
execution. Bounded local `$defs`/`$ref` definitions are inlined for role handoffs;
unknown, external and recursive references fail before inference. Owner schemas,
source text and action ordering retain their meanings. Historical object decisions
retain their original decoding contract.
Tools/schema and earlier history stay stable across reads, edits, follow-ups
and required final delivery. A required action is a tail turn constraint,
validated before effects, not a rewrite of the catalog or output schema. An
explorer's forced final call reports observed findings and remaining gaps
within the existing budget rather than requesting another inspection.
Quoted machine tool results and historical tool arguments are decoded once in
the CLI input; user prose and the original journals remain unchanged. This
avoids presenting serialized display escapes as additional source characters.
Invalid decisions are rejected as a whole, and catalog membership, scopes and
authority are still checked before effects. Legacy codecs remain available for
their existing formats. Schema normalization follows the official
[Structured Outputs contract](https://developers.openai.com/api/docs/guides/structured-outputs).

Structured CLI decisions retain the trusted initial Tare system message as role
guidance in the CLI instruction file. It governs the proposed external decision;
the CLI remains toolless. Later records, user prose, retrieved sources and tool
results stay quoted and gain no instruction priority. This also preserves the
role's real-edit and delivery rules rather than burying them in quoted history.

In v4, `edit.sha256` uses the stable `"observed"` reference, or null for
create-only. The host resolves it to the latest matching read/edit receipt for
exactly that repository/path; the original CAS still rejects a changed disk
version. Repository/path permissions and exact source matching remain independent
execution guards. Historically, v3 selected pins from a wire enum of at most
128 observed hashes plus null; v1–v3 records retain their original decoding.

The CLI uses a stable cwd without enabling native task tools. Cache hits and
retention remain provider-owned; raw CLI quota has no measured cash conversion.

### Native Codex command review

`compute_plane.codex_auto_review.CodexNativeCommandReview` is an opt-in
reviewed-execution gate, separate from the toolless ModelPort. The host calls
`run(action, authorization, control, effect, public_context=...,
before_effect=...)` inside its already-authorized, durably claimed effect.
The public Codex app-server routes one fixed MCP command through native
auto-review; the bridge only queues that exact command for the host callback.
Matching thread, turn, item, review, action subject and exact arguments are
required before the host revalidates its original guard and invokes the existing
confined effect once. Approval grants no Tare authority or wider permissions.

The host admits an absolute CLI executable/hash, the explicit model and its
catalog, a fixed environment, disclosure and budget callbacks, and receipt
custody. Version 0.162.1 is the qualified protocol baseline. Shell, patch, search,
node and multi-agent task tools are removed from the copied catalog; optional
features, hooks, apps and plugins are disabled and checked. Only the host's
fixed bridge is enabled. Existing official ChatGPT login remains with the CLI;
API-key login, custom endpoints, global instructions, changed executable,
manual approval, missing native events or failed cleanup do not execute.

The MCP command envelope remains bounded at 64 KiB; app-server RPC lines allow
128 KiB for its configuration reply, with a 512 KiB total transport ceiling.
MCP advertises ordinary typed fields; exact hash/argv/cwd enforcement remains
in the bridge and host rather than unsupported const-only tool properties.
The native turn is interrupted without forwarding the command result back to
Codex; the original host journal retains the real result. Native approval,
local cleanup and host dispatch are separate receipt facts. Native reviewer
model, billing, hidden retries and remote stop remain unknown where not emitted.
The forwarding generation uses ordinary Codex quota.

This is not a standalone ActionReviewerPort or `review/start` code review.
An effect-free advisory MCP call can be approved without approving its proposed
effect; it must not authorize that effect. Source tests use offline fake public
RPC and owned bridge processes. Live qualification is recorded separately in
private custody, including exact host execution and replay controls.

The 2026-09-20 vendor increment adds two explicit protocols:

- `vendor-cli`: `native_cli.py` constrains AGY 1.2.7, Codex 0.154.0 and Claude
  2.1.278 to stateless inference with official CLI authentication and no task
  tools. Public inputs are hashed before dispatch; strict JSON proposals return
  through the existing ModelPort for Tare to validate and execute. Cancellation
  uses managed process containment. Hidden framing, remote stop and backend
  retries remain unknown where the CLI cannot attest them. Raw usage is retained;
  normalized CLI usage is not yet qualified. Model identity from Codex JSONL is
  request-pinned only. All three CLIs accept compatible newer releases:
  the saved version is a reference, not an equality lock. New releases must expose
  the required isolation and transport switches; every response still passes the
  existing model, tool, schema and hook checks. Minimum versions are AGY 1.2.7,
  Claude 2.1.278, Claude 2.1.280 for structured output/Opus 5.5, and Claude
  2.1.293 for Haiku 5.5 (including structured output). Receipts record
  both reference and actual versions without rewriting saved deployment identity.
  A release the adapter has not seen first answers one probe per model before the
  real request: the same request with its conversation replaced by one short record
  (low effort, except AGY, whose tier-named models accept only their own effort; at
  most 60 seconds), through the same branch, decoder and one-time repair. A shape
  failure (boundary, invocation, tool or decode) is recorded and returns
  CLI_QUALIFICATION_FAILED, not started, now and on later calls without a new probe;
  a probe cut short returns CLI_QUALIFICATION_INTERRUPTED, not started and
  retryable. Quota and provider failures of the probe keep their own code and are no
  verdict. The host may pass a persisted `qualification` mapping ("version model" ->
  record) so a machine probes a release once per model; a store that cannot be
  written loses the verdict, never the call. The probe's facts (`qualification`)
  go with the real call that paid for it, also when that call fails.
  Opus 5.5 decisions use the existing toolless JSON protocol from the outset: its
  formatter can answer in text instead of invoking StructuredOutput and hit a CLI
  turn limit. Failed results are never accepted or automatically retried. Other
  structured Claude decisions allow two CLI turns for the formatter acknowledgement;
  the response must still prove exactly one model inference and no task tools.
  Haiku 5.5 structured decisions also receive a compact delivery example in the
  transport system prompt and final reminder. It distinguishes action data from
  callable functions, avoids duplicating an answer, and explains JSON escaping.
  Bounded final-decision replays exposed both direct `tare_deliver` calls and
  unparseable formatter input. Claude receipts retain allowlisted tool names,
  an unparsed-input flag and tool-result error status, without arguments or
  vendor error text. Rejected output still grants no effect authority; the
  single-inference boundary and failure codes are unchanged. These observations
  are transport qualification, not general implementation qualification.
  When search is in Haiku's action catalog, the final reminder also includes a
  concrete external-search example. It is appended after the unchanged system
  and history prefix. A native search attempt followed by formatting still fails
  the same boundary checks; the example does not authorize native task tools.
  AGY's structured decisions likewise receive a final external-action example,
  appended after its unchanged cached prefix. Native read/search attempts remain
  rejected even when a later result contains valid structured data.
  Codex requires at least 0.154.0 and discovers the installed feature table using
  `features list` in a disposable home without authentication. It disables all
  active optional features except the two existing inference-isolation controls,
  and checks the effective values before dispatch. New features are included
  automatically; removed and deprecated overrides are omitted, with deprecated
  features still required to be inactive. A forced `unified_exec` implementation
  is harmless only with the required shell feature disabled and the selected
  model's shell capability removed. Missing controls or ineffective overrides
  fail before inference; version changes alone do not. Discovery is reused within
  a client for the same observed version; every call still probes the executable.
  On 2026-09-26, a real 0.157.0 CLI sent no tool catalog to a local fake Responses
  endpoint. Two authenticated TUI turns using the unchanged 0.154.0 profile then
  completed, preserving conversational history and executing no file effects.
- `openai-codex-responses`: the separate Responses codec retains encrypted
  reasoning and tool IDs/results through SSE and history replay. Its auth
  component reads an explicitly selected existing ChatGPT login, pins the account
  and fixed endpoint, and refuses expiry, redirects and API-key fallback. It does
  not renew credentials or read the OS keyring.

  The subscription HTTP client sends `session-id` equal to the submitted
  `prompt_cache_key` on `/responses`. The official
  [Codex 0.162.1 client](https://github.com/openai/codex/blob/rust-v0.162.1/codex-rs/core/src/client.rs#L595-L602)
  uses this header for ChatGPT cache affinity; supplying only the body key is
  insufficient to reproduce that client contract. Each exchange derives its
  headers from its own body, without mutating shared authentication headers,
  conversation identity, input, deadlines or budgets. Explicit shared-prefix
  keys are preserved. This is an affinity hint, not a guarantee of cache reads
  or subscription savings. API-only cache options remain unsupported on this
  qualified subscription route. Offline HTTP tests cover header delivery,
  invalid values, streaming and cancellation; live qualification is separate.

OS composes these adapters into both interactive NativeSession and Work flows;
the Runtime loop is unchanged. Operational profiles, live TUI and flow evidence,
and limits are documented in OS `docs/guides/VENDOR_CLI_INFERENCE.md`. External
Tare CLI/MCP frontends are separate follow-up work.

### AGY action envelope and working context (0.9.12)

AGY requests use `conversation_records`, `action_catalog`, and the official
`--json-schema` output option. The schema and catalog are part of the counted,
hashed request. Responses contain `text` and data-only `actions`, with JSON
arguments in a string. The host assigns call IDs and validates action names
against the admitted catalog; summary requests admit no actions. Other CLI
codecs retain their existing proposal format. No incomplete/error response is
salvaged as an executable action.

The invocation-local agent declares only `finish`; the PreToolUse hook still
denies every invocation, including finish. Accepted responses show exactly
pre/post/stop, optionally with one denied finish carrying the schema proposal.
That final formatting proposal is decoded from typed hook arguments, since the
CLI's text rendering can over-escape nested JSON. Any task tool, extra inference,
missing denial or failed final result remains a refusal. A denied task tool's
name is recorded without its arguments to make failures diagnosable. Official
CLI authentication and hooks remain in use; no credential extraction or vendor
binary modification is involved. General model reliability is not guaranteed
by the schema. See the OS recovery report for the bounded live qualification.

Current source reads render each line as `number|source` without padding after
the delimiter. Copy only the source after the first `|` for exact edits; original
indentation and later `|` characters belong to the source. Range, Unicode column,
byte limits and SHA checks still apply. Historical read receipts retain their
original rendering.

An edit to an observed scoped `AGENTS.md` can succeed while its old instruction
pin remains in force. Its receipt reports `instruction_update`, confirms the
applied edit and requires a new conversation before further project work.
Schedule instruction edits after other edits and checks. A changed-pin refusal
also returns this recovery action; neither path silently adopts new guidance.

`operational_checkpoint(snapshot, writable=...)` can retain historical numbered
source excerpts for explicitly writable paths, with hashes, ranges and journal
receipts. Recent excerpts share the existing 8 KiB tool-fact budget; oversized
payloads reduce complete lines before dropping the last excerpt. Changed hashes,
edits and possibly mutating terminal results invalidate excerpts. Protected
project instructions remain separately retained as before. Metadata-only
behavior remains the default for consumers that do not opt in. The host must
still count the whole candidate context, and edits still require hash validation.

Kernel 0.9.3 adds explicit `NativeModelAdapter(..., streaming=True,
on_progress=...)` and `NativeHttpClient.post_stream_observed`. Chat uses SSE;
Bedrock uses the official [ConverseStream](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html)
binary EventStream framing with prelude/message CRC validation. Only complete
messages pass the existing reply/tool validator. Provisional deltas carry no
effect authority. Tools, reasoning and signatures are reassembled for the same
history contract; the console can show text/phase progress without displaying
reasoning or partial arguments. Cumulative usage is retained once, never summed
per delta. Missing terminal events are uncertain exchanges, not retryable
completed failures. Cancellation preserves unknown remote stop/usage and local
cleanup evidence. A failed optional display observer cannot discard the response.
The callback is a trusted bounded local observer, not an asynchronous plugin port.

OpenRouter Chat compatibility: preserve the ordered `delta.reasoning_details`
objects exactly and replay that sequence with the assistant message, including
opaque signatures and encrypted fields. See the provider's
[reasoning-details streaming contract](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
The [final usage frame](https://openrouter.ai/docs/api_reference/streaming)
can repeat the same terminal choice with an empty delta. Accept one such accounting
frame only; reject later content, tools, opaque deltas, a different finish reason
or a second accounting choice. Keep usage once. Chat shape failures retain fixed
allowlisted diagnostics without returning provider values or exception text.

Responses terminal failure handling (2026-09-21): a `response.failed` event with
a stable, nonempty response ID and explicit `status=failed` closes inference as
`PROVIDER_RESPONSE_FAILED`, without automatic retry. Its known usage is retained;
all provisional output items are discarded. Generic errors, malformed terminal
events and missing completion remain `NativeTransportInterrupted` and never
authorize effects. Worker receipts now retain allowlisted `stream_failure_reason`
and `worker_error_type` diagnostics without provider error text or credentials.
Existing interrupted journals are not rewritten: their original causes cannot be
recovered from older receipts. `stream_events` counts emitted deltas, not all SSE
events.

Connection failure handling (2026-09-21): the isolated HTTP worker records whether
the first socket send has started. A clean worker exit can attest a DNS,
connection or TLS failure before that boundary. The parent then returns
`ModelFailure(not_started=True)` with a fixed diagnostic code, preserving the
fact that the worker started while the HTTP request did not. Transient DNS,
refusal and connect failures may use the Runtime's explicit retry budget;
certificate and TLS failures are not automatically retried. No exception text,
hostname or credential is copied into diagnostics.

Partial sends, response disconnects, proxy CONNECT followed by TLS failure,
timeouts/cancellation without a complete worker receipt, and malformed or
contradictory receipts remain uncertain. No inference about dispatch is made from
zero stream deltas or elapsed time. Undispatched calls are excluded from possible
remote charges without fabricating provider usage. Existing interrupted journals
retain their original uncertainty.

Interrupted transport diagnostics also record an explicit `transport_failure_phase`
(`prepare`, `connect`, `request_send`, `response_headers` or `response_body`) and,
when the exception type is recognized, a fixed `worker_error_cause`. Wrapped
`URLError` causes retain typed distinctions such as timeout, reset and TLS failure;
text-only causes are omitted. These observations do not establish whether the
provider ran or charged for the call, authorize retry, or replace the independent
pre-send proof. A connect phase can include proxy setup that already sent bytes.
Older receipts cannot recover these missing diagnostics retroactively.

The selected streaming mode enters the adapter digest, and Chat's exact stream
options enter its counted request. Default nonstream identities remain unchanged.
The stream decoder supports bounded text, tools and reasoning; new multimodal or
citation block types are explicitly unsupported. SigV4 remains outside this client.

`ModelProfile` records the protocol, endpoint, route/model, effective context and output
limits with their source, parameters, optional template digest, expected response
identity and optional explicit Bedrock cache evidence. Objects are detached on
ingress. These are supplied facts in the default protocol. The opt-in fleet
counter below checks the live represented configuration. It never infers support from a
commercial model name or silently turns thinking off.

`NativeModelAdapter` implements `ModelPort.generate` and an explicit `count`
operation. It translates native conversation projections into OpenAI-compatible
Chat or Bedrock Converse, preserves provider assistant/reasoning blocks for
same-profile replay, pairs tool results and namespaces provider tool IDs by model
call. It rejects replay under a changed model/profile/system/tool schema.

OpenAI Chat summary requests omit tool schemas: disabling tool selection alone
still let Qwen emit raw tool-call markup as summary text. Summaries use an
operation-specific system instruction and the recorded history. Ordinary model
requests retain automatic tool choice. Runtime still rejects tool calls, delivery proposals and
empty or incomplete summary responses, including from nonconforming providers.

Identical scoped project guidance is emitted once within each request projection.
Guidance retained in the operational checkpoint counts as already present;
changed content or scope is emitted again. All other tool output and original
journal records remain intact. No cross-request memory is used for omission, and
counting and generation use the same projection.

The normal tool choice is automatic. The reserved `tare_deliver` tool translates
a nonempty summary into a delivery proposal; it is never executed as a capability
and cannot be combined with an effect. Plain assistant text adds a deterministic
continuation user turn when projected for the next request. Complete malformed
provider responses are explicit non-retryable protocol failures; incomplete tool
arguments never become an effect.

`NativeHttpClient` exposes a synchronous port over one isolated HTTP exchange with
bounded response size and monitored wall budget, an explicit endpoint and an injected bearer credential
callback. Credentials can come from the existing keyring composition without
being copied into profiles, journals or telemetry. Redirects are refused. No SDK,
provider/model discovery, auxiliary GLM call, retry, summary or host-agent loop
is embedded here. Confirmed HTTP 429 and proven transient pre-send connection
failures are classified as retryable; Runtime decides whether to retry. Network
uncertainty after send is not a completed failure.

The Bedrock codec follows the official [Converse request/response contract](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html).
This client supports bearer Bedrock API keys or compatible gateways, not SigV4.
It is deliberately not a universal Bedrock client.

## Counting, caching and observation

For Runtime 0.13.0 operator continuation, projected `user_message` events become
actual provider user input. A prior `tare_deliver` assistant message is preserved
with its reasoning and receives a deterministic protocol tool acknowledgement
before further user input. This is not a repository effect or Work acceptance.
Both Chat and Converse histories were qualified through actual loopback HTTP
CLI subprocesses; one real Qwen canary completed two instructions in the same
session and an idempotent repeat without another HTTP exchange.

The default counting protocol uses the same generated request: Bedrock receives system, messages and
toolConfig through [CountTokens](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html);
the local counter receives the full wire through `/apply-template`, then the
rendered prompt through `/tokenize`. The counter endpoint is explicitly supplied
separately from the generation endpoint. Unsupported, denied or malformed counts
return `unavailable`, with no hidden estimate or generation fallback.
Bedrock can additionally specify a sourced `count_model` independently of its
inference-profile route. No route rewriting is inferred. Live observations show
a +17 counter discrepancy for the selected Opus profile; the larger count is
retained rather than silently corrected.

M2b connects these counts to Runtime `NativeContext`. Count observations bind
the projection, purpose, profile and effective limits; a prepared generation
must match its admitted request digest before transport is invoked. Required
source strings and summary purpose are preserved in both protocols. Summary
activation is owned by Runtime; the codec projects its derived summary before
retained turns and preserves their original provider blocks. The injected local
counter can be supplied explicitly; it does not imply endpoint auto-discovery.

Usage-anchored estimates also recognize the qualified AGY CLI exclusive layout:
`input_tokens + cache_read_tokens` occupies context. Persisted matching receipts
can calibrate growth without another inference call; missing counters remain
unknown. Byte-only fallback is an estimate, not measured provider token usage.

Only an explicitly configured system cache point is currently supported, with
default provider TTL. A live Opus control qualified a 5-minute write/read for one
eligible prefix and disjoint usage categories. Tool/message placement, other
profiles and actual task savings remain unqualified. Stable tools/system and
original assistant blocks preserve a repeatable request prefix.

Raw [usage fields](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html)
are preserved, including optional cache fields. Missing fields stay absent or
null. The adapter does not sum cache into an allegedly complete total or price
the request itself. The M2c derived report below performs qualified normalization.
The observer receives session/call
identity, purpose, request/response digest, raw usage, duration and error category.
Its failure increments `observation_failures` without discarding a valid response;
that counter must be included in any claim of telemetry coverage.

`client_invocations` counts adapter calls to the injected transport, not known
physical requests after SDK retries. `NativeHttpClient` does not retry; another
injected client must declare its behavior before qualification. A failed or
cancelled request may have unknown remote consumption.

### M4e: model-bound public fleet counter

`NativeModelAdapter(..., counter_protocol="slop-fleet-v1")` selects the explicit
Chat-only gateway protocol. It uses the generation client for `/fleet/count`;
a separate counter endpoint is refused. The adapter validates the request SHA,
requested/expected model, live window, optional pinned template and fingerprint,
then retains only the last count binding. Generation requires that admitted
request SHA and sends a transient `tare_fleet_binding` envelope field. A reply
must confirm the binding in `tare_fleet_observation`; returned usage is preserved
even when confirmation fails. Raw counter and generation metadata are retained
in observations and saved response metadata for recovery.

Default ModelProfile fields and digests remain unchanged. The nondefault counter
protocol enters adapter/attempt identity only when selected; this is not a
silent reinterpretation of existing durable sessions.

The gateway checks template/slots/build/registry identity before generation,
under its existing serialization lock. This is stateless drift detection, not
a lease or proof of freshly hashed GGUF bytes. Text/tools are qualified;
multimodal content and bound streaming are unsupported. Client HTTP counts
exclude gateway health/loading polls and internal template/tokenizer/profile
requests; their available operation receipts declare that partial coverage.

123 selected Windows tests across Local Labs/Kernel/OS passed. Four live native
agent canaries used Qwen3.8, HauhauCS, Fable-TC and Qwen3.6 MoE: all eight input
counts matched usage, and all four changed-request probes were refused409.
See Local Labs' `docs/FLEET_COUNT_PROTOCOL_2026-09-10.md` for profiles, storage and
operational scope. No all-template equality or task-economy claim is implied.

## M2d: physical HTTP cancellation

`NativeHttpClient.post_observed` runs the finite `native_http_worker.py` using
the current interpreter, isolated mode (`-I`) and an absolute source path. It
performs one urllib exchange with no SDK retries or auxiliary model calls.
`managed_process.run_pipe_process` reuses the existing POSIX process group and
Windows kill-on-close Job Object containment. No installed host launcher, Aider
session or frozen harness module participates in this path.

The supervisor polls live cancellation/deadline/revocation while the worker can
be blocked in DNS, TLS, headers or body reads. Its operation timeout includes
worker startup and response collection, and still expires if the server keeps
dripping bytes. Termination is followed by pipe collection, worker exit, input
writer join and containment checks. Cleanup has bounded additional waits (up
to five seconds per cleanup step); the operation timeout is not a promise that
all cleanup or OS process creation finishes at that exact instant. An error or
unconfirmed cleanup leaves an uncertain invocation, never an admitted success.
The credential callback must itself be bounded by its host; arbitrary blocking
code inside that callback is outside the worker containment.

Input and bearer credentials travel through a finite pipe; no prompt/credential
temp files, argv secrets or new scratch root are created. An owned input writer
prevents a full unread pipe from blocking supervision on Windows as well as POSIX.
The trusted worker bounds request and response bytes. The supervisor adds an
observed output ceiling; it is not a general memory sandbox for arbitrary
malicious programs. Worker diagnostics retain error categories, not HTTP error
bodies or exception text that could echo secrets.

Generation and count observations carry per-exchange physical receipts: whether
a worker started, containment type, local cleanup, timeout/cancellation and an
upper bound of one HTTP attempt. Local termination cannot establish whether a
remote provider stopped generation or stopped charging. Such remote state stays
unknown. A pre-dispatch cancellation is distinguished from an uncertain request;
the usage report excludes proven undispatched calls without fabricating usage.
The report aggregates physical receipt coverage separately from client invocation
counts, and still declares gaps for injected transports without receipts.

A complete reply racing cancellation is preserved, including usage, while the
Runtime rechecks control before consuming a tool request. An incomplete exchange
requires reconciliation; confirmed 429 is the only automatic-retry candidate,
with retry count owned by Runtime. Complete invalid JSON is a known non-retryable
HTTP response failure. Redirects are not followed. No recorded PID is used to
resume or cancel a prior session: durable custody/rebind remains M3.

The existing `AuditStream` was inspected for terminal/draft separation. Its CLI
event parser remains with that adapter; it was not transplanted into HTTP. These
native codecs still consume finite Chat/Converse responses. SSE/ConverseStream
token-event decoding and streaming UI are unsupported. A body that never reaches
a terminal response is physically interrupted rather than partially executed.

This isolation has a per-exchange process startup and connection setup cost,
including each counting operation. Existing operation durations include it;
the real canary must measure whether it is acceptable. No daemon, extra runtime
dependency or persistent HTTP pool was introduced. An async transport remains
substitutable through the existing port if it can satisfy the chosen cancellation
and observation contract. [HTTPX's documented async lifecycle](https://www.python-httpx.org/async/)
and [operation timeouts](https://www.python-httpx.org/advanced/timeouts/) were reviewed;
the selected increment preserves physical containment around blocking resolver
and library calls rather than adding a new async lifecycle to the Runtime.

## M2c: derived usage report

`compute_plane/native_usage.py` joins the existing session journal with the
adapter observation sink. It creates no storage, service, registry or extra
agent loop. The caller supplies the retained observation set and cumulative
`observation_failures` from every adapter instance used for that session.
Generation and count observations now carry session identity and a unique
observation ID. Exact replay deduplicates; conflicting duplicates are rejected.
Count transport invocations are recorded separately from model-token consumption.

```python
from compute_plane.native_usage import UsageContract, native_usage_report

report = native_usage_report(
    journal,
    observations,
    contracts={adapter.profile_digest: UsageContract(
        "converse-exclusive", "deployment-specific evidence of counter semantics",
    )},
    tariffs={},  # No tariff means unpriced, not free.
    observation_failures=adapter.observation_failures,
)
```

Two explicit per-request layouts are implemented. Neither is inferred from a
model name or HTTP compatibility:

- `chat-inclusive`: `prompt_tokens` includes cached input;
  `prompt_tokens_details.cached_tokens` is a subset. Completion includes the
  optional `completion_tokens_details.reasoning_tokens` subset. This layout is
  qualified only for profiles with no separately billed cache creation category;
  its normalized write count of zero is a contract consequence, not an observed
  cache counter. Other Chat/gateway conventions need explicit qualification.
- `converse-exclusive`: the selected deployment has disjoint `inputTokens`,
  `cacheReadInputTokens` and `cacheWriteInputTokens`. All three are needed to
  derive full input; absent cache counters stay unknown. `totalTokens` is kept
  raw, without guessing whether it includes cache. Reasoning remains unknown
  because this interface has no portable usage field for it.

The generic [Bedrock usage fields](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html)
and [cache billing documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
were checked on 2026-09-10. They establish distinct reported cache categories and
potentially distinct charges, but do not by themselves qualify every deployment's
counter inclusion. Contract sources must record that qualification before real
totals are claimed. The archived OS DeepSeek study likewise documents why adapter
normalized input and raw prompt input cannot be treated as interchangeable.

`UsageTariff` records currency, source, effective date and decimal-string rates
per million uncached input, cache read, cache write and output tokens. Null rates
remain unpriced. The caller pins it to the exact profile digest and must verify
applicability, including region, routing, TTL and pricing tiers. This increment
supports a qualified flat rate per category; mixed TTL/tier and other charge
models are not qualified. No actual market tariffs are embedded. Currency
subtotals remain separate; reasoning is not charged again on top of output.
An unexpected Chat response model preserves raw usage but disables profile pricing.

The report correlates Work binding, session, main/summary call, retry parent and
effect IDs. OS attempt/candidate association remains M3. Failed/partial/invalid
responses retain observed usage; an uncertain call without counters remains
unknown. A saved valid reply can recover usage after observer failure while the
coverage gap remains visible. An explicitly undispatched intent is excluded from
possible remote charges without fabricating a zero usage receipt.

Each token subtotal includes an unknown-call count. Monetary subtotals include
unpriced/partial-call counts. Observed count operations, missing journaled counts,
sink failures, client invocations and operation durations are separate. Counting
occupancy never adds to billed input. Duration sums are not time since task
preparation; injected-client invocations are not verified physical SDK attempts.
Counter reset/delta handling for cumulative vendor CLI telemetry is unsupported,
as are final bills, infrastructure, counting fees, tool charges and task wall time.
This report cannot establish integral savings. Retention remains with the existing
host sink; it does not create a second on-disk ledger or import managed Aider code.

## Strands disposition

Revisited the archived source at revision
`66101c7d5f810dc744b5be18b933af3441e00210`, verified against SHA-256
`cba9dee70757238558b836d8b0ab2baee990aa3f4f74bee0e30fb0ac13c2918f`.
The public `format_request`, `count_tokens` and `stream` interfaces and the
thinking helper were inspected again. `count_tokens` returns an integer even
when it falls back to a heuristic; forced tool choice may remove thinking;
stream cancellation can return before remote activity stops.

For this increment the selected implementation is a direct codec plus a small
bearer HTTP client, reusing the current Runtime contracts and stdlib HTTP. This
keeps counting provenance explicit without private Strands hooks. The Strands
package was not installed or exercised during M2a–d. The subsequent
[public component evaluation EX-01](https://github.com/augusto-scarvalho/tare.tools.kernel/blob/03d3591365b4fe66ae4005b0977a6657253070ea/docs/STRANDS_EX01_20260910.md) now demonstrates
functional substitution and records why the full provider was not adopted for
this contract. The direct implementation remains selected. This is not a claim that Strands cannot be adapted or
that the direct implementation outperforms it. Reassess if SDK auth/streaming or
provider support saves more code than its integration costs.

## Qualification and remaining work

The unit/regression tests use recorded-format fixtures and actual loopback HTTP.
The separate live canaries above also traverse native read/edit/verify/proposal.
Both protocols traverse native read/edit/verify/proposal in contract tests with
fake capabilities; thinking, signatures, stable prefix, ID mapping, partial
responses, telemetry gaps, counter errors and retry boundaries are checked.
These are contract tests, not proof of a current Qwen/slop.cpp or Opus deployment.

Before live adoption:

- M2b: full-request admission and transactional compaction pass local tests
  with Runtime 0.10 and one real Qwen pressure control. General summary quality
  remains unqualified. Direct `generate` without a Runtime admission digest is
  still an isolated primitive, not a window-qualified production entrypoint.
- M2c: normalization, tariff provenance and cold/warm arithmetic pass locally;
  the selected live Opus counter/cache behavior and list-price estimate are
  recorded in the canary report, with exclusions. CLI cumulative usage is deferred
  to its explicit host adapter, not accepted as per-request native usage.
- M2d: local physical cancellation passed under POSIX/WSL and Windows Job Objects.
  The actual Qwen backend stayed busy for ten seconds after local disconnect;
  per-request stop/usage remains unknown despite later idle observation.
  SSE/ConverseStream token events are not a
  supported codec in this finite-response increment.
- Live scripts explicitly resolve endpoint, template, model launch and counting
  capabilities; the adapter itself has no automatic identity probe. Local non-tool/text-JSON codecs
  and additional API variants are unsupported until explicitly qualified.
- Strands public component evaluation is complete in the linked bounded scope;
  M3 still owns managed Work/settlement composition.

No packaging defaults, credentials, federation pins, legacy consumers or deployed
workers changed. No copied legacy harness modules, temporary repo clones or
unbounded disk journals were introduced.

## Historical verification

Dated local runs are preserved as private records at the original owner commit.
See [the records reference](../RECORDS_ARCHIVE.json) for paths and hashes.
They are separate from validation of the current source.

## Optional HTTP deadlines (0.9.6)

`NativeHttpClient(timeout_seconds=None)` disables its total request ceiling and,
when omitted, its inherited socket timeouts. `connect_timeout_seconds` and
`read_timeout_seconds` independently accept positive seconds or explicit None;
omission inherits `timeout_seconds`. Connection/TLS uses the former; socket read
waits (including response headers) use the latter. DNS timing remains subject to
the platform resolver; parent cancellation/deadline still contains that wait.
The parent continues polling cancellation and enclosing ExecutionControl deadlines
without serializing infinity into JSON. HTTP receipts include the selected waits;
local cleanup never claims a confirmed remote stop. Text and streaming share this
transport. No extra HTTP attempt or parser of textual tool calls was introduced.

## OpenRouter session affinity and cache accounting

For `openai-chat` at `https://openrouter.ai/api/v1`, the adapter supplies an
opaque stable `session_id` derived from the native session ID. It survives
turns, summary generation and adapter reconstruction; an explicit profile
parameter takes precedence. No routing field is added to other endpoints.
OpenRouter can use this hint for provider affinity before the first cache hit.

Stable system instructions, sorted tools and append-only history preserve
prefix reuse. A compaction may legitimately invalidate the history prefix.
The `openrouter-inclusive` usage contract reads cache reads and writes as
subsets of prompt input. Missing counters remain unknown and invalid arithmetic
does not contribute exact totals. Cached-token discounts do not imply the same
discount on output or on a whole call. Provider costs remain raw unless an
explicit tariff is supplied. See the [provider policy](https://openrouter.ai/docs/guides/best-practices/prompt-caching).


## Declared session header and OpenCode usage

A profile may declare `session_header`, a provider's per-conversation header. The only one admitted is
`x-opencode-session`, and only on `openai-chat`, `openai-responses`, `anthropic-messages` and
`gemini-generate-content` profiles. The adapter
sends the opaque key behind OpenRouter's `session_id` (the Responses `prompt_cache_key` is its first 64
characters), derived from the native session ID, as that header on every generation exchange, streamed or not. OpenCode uses it for routing
and prompt caching. The HTTP client admits per-exchange headers from that list only, never a credential, and
rejects any other name or a control character before reading any credential or request header. A port without `post_observed` (a test fixture)
receives no header. A profile without the field keeps its earlier digest.

The `opencode-exclusive` usage contract applies to those four protocols. It reads OpenCode's disjoint counters
under `usage.opencode.tokens` (`usageMetadata.opencode.tokens` on `gemini-generate-content`): `input` excludes cache reads and writes, `output` includes `reasoning`, and cache
writes are `cache_write_5m + cache_write_1h`. The family's own `prompt_tokens`/`input_tokens` stay raw evidence. A
missing block leaves every counter unknown. The reported charge is `usage.cost`, which is 0 for Go use within the
plan's limits. It carries `opencode.billing_source` (`free`, `balance`, `go`, `go-plus` or `byok`; anything else
becomes null), `list_cost` and `allowance`, the dollars of the Go plan's monthly limit the request used. A cost
OpenCode marks `estimated: true` (the customer's own provider key) estimates that provider's bill and is not taken
as a charge. See
[OpenCode inference](https://opencode.ai/docs/console/api/inference) and [OpenCode Go](https://opencode.ai/docs/go).

## Billing guard and quota refusals

A profile may declare `billing_sources`, the provider-reported billing sources it may be charged to (lowercase names,
HTTP API protocols only). OpenCode reports one per reply in `usage.opencode.billing_source`. A reply billed to any
other source, such as `balance` for a subscription profile whose allowance ran out with Extra Usage on, keeps its
usage on record, marks the observation `billing_refused` with that source, and ends the session with
`BILLING_SOURCE_REFUSED` before another call is billed the same way. A reply without a reported source passes, and so
does a profile without the field, whose digest stays unchanged.

A refused HTTP 429 keeps its error type in the transport receipt's `provider_error`, a single word such as OpenCode's
`GoUsageLimitError`, so the host can tell a spent plan from a passing rate limit; with `retry_after` (from
Retry-After, up to 24 hours) it knows when the plan's window resets. The 429's message stays out, as before.

## Streaming reasoning custody (0.9.73)

Chat streaming consolidates consecutive indexed `reasoning.text` and
`reasoning.summary` fragments only when all other fields match. Every payload
character survives the next-turn projection. Different identities, signatures,
unknown fields, encrypted blocks and unindexed details remain discrete; no
reasoning truncation or event-size increase is used. This avoids multiplying
metadata until the Runtime rejects an otherwise small response. Validation
receipts retain the Runtime's fixed rejection enum without raw provider text.

See the [OpenRouter reasoning format](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## Protected noninteractive command capability (0.9.26)

An implementer can explicitly admit `command` with repository and argv inputs.
The capability requires both the host mediation port and an OS process factory,
refuses interactive terminal use, and invokes the existing managed process path.
It retains the real exit code and output and invalidates earlier read/check state.
Default capabilities and the file adapter remain unchanged.

Native Codex reviewed execution retains the original host result if the optional
post-effect `turn/interrupt` cannot be sent because its input already closed.
Pre-effect correlation, authorization and revalidation still fail closed. Local
supervisor cleanup and receipt custody remain mandatory; remote stop stays unknown.

AGY receives a forced decision's closing reminder after the unchanged quoted
records, as Claude's formatter already does. System text, action catalogs and
response grammars stay stable. An ignored final delivery requirement returns
`CLI_REQUIRED_ACTION`, with usage retained and no proposed action executed;
the adapter does not describe that choice violation as malformed JSON or spend
further calls repairing an otherwise valid schema.

Explicit corrective conversation input (`interrupts_turn`) closes only unstarted
tool calls in the provider projection with `CANCELLED_BEFORE_EXECUTION`. Prior
assistant messages and completed tool results remain unchanged. Runtime
admission requires reconciled external effects and retains binding and limits.

AGY terminal completion also accepts a fully decoded successful result after the exact
model/hook/stop boundary, followed by verified cleanup of the owned process tree.
The CLI may retain background threads after emitting that terminal result. The
transport drains output again and repeats validation after cleanup; truncated or
duplicate results, unexpected tools, cancellation and output overflow remain
rejections. This does not salvage prior interrupted exchanges or infer remote stop.
