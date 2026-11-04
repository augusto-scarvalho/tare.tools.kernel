# Optional Von CPU choice worker

`compute_plane.von_choice_worker` implements the existing bounded
`choice_assessment` stdin/stdout protocol. It produces observations only; OS
retains routing, resource admission, abstention and fallback authority.

**Experimental / not qualified for automatic routing.** The first Local Labs
screen scored 6/15, failed all six semantic option-order groups and all four
missing-evidence boundary abstentions. The existing worker configuration is
unchanged. Do not install this worker as the default evaluator.

## Reproduce an observation

Use the isolated CPU environment with Torch `2.8.0+cpu` and Transformers
`4.57.6`, set `PYTHONPATH` to this Kernel checkout, and run:

```text
python -B -m compute_plane.von_choice_worker /path/to/checkpoint.json
```

Supply one JSON object with `state`, `question`, and `options` on stdin. Callers
can use `choice_assessment.run_assessment(command, request, timeout_seconds=60)`
with a direct interpreter executable; they remain responsible for process-tree
containment if their interpreter command is a launcher. Do not run concurrently
with another CPU evaluator without a resource admission policy. Allow at least
6 GiB available RAM before launch. No daemon, shared lock or routing activation
is installed by this delivery.

## SDK contract review

This worker targets **Option-Marker**, selected explicitly upstream as `marker`
or `option-marker`. The SDK's default `von-1.0`/`von.decide` path instead uses
`BertaBackend` with separate NLI sequence-classification weights. They are distinct
inference paths and must not share a qualification label.

A subsequent documentation audit compared all 15 original requests with the
unmodified upstream Option-Marker constructor/backend under Transformers 5.17.0.
All token sequences and decisions matched; maximum probability difference was
0.0000471, within SDK rounding. Thus the original result describes this checkpoint
on those requests, rather than a demonstrated tokenizer/config compatibility bug.
It does not measure all Von backends or establish a general ranking against Laya
or OpenJEV. Local Labs records separate natural-language and public-control arms
in `VON_INPUT_CONTRACT_REVIEW_2026-09-21.md`.

The manifest uses `tare.local-labs/von-checkpoint/1`, a local `directory`, the
pinned model/source revisions and SHA-256 hashes for exactly the five files in
`FILES`. The measured manifest is retained by Local Labs under
`docs/research/evidence/von-20260921/checkpoint.json`. Model weights remain in the
operator's external assessment-models directory, outside Git.

## Inference boundary

- CPU FP32 only, two threads; optional dependencies load only at invocation.
- No downloads, remote custom code, SDK auto-device selection or service calls.
- Hash every checkpoint file before loading. Restricted `weights_only=True`
  deserialization, strict state dictionary loading, explicit local tokenizer.
- Preserve the upstream option-marker forward pass and prompt packing. Construct
  the encoder from config before loading the complete trained checkpoint, avoiding
  the SDK's redundant separate encoder download. The vendored source is unchanged.
- Translate explicit full/sliding RoPE frequencies to the pinned Transformers
  config and use CPU SDPA without compilation. This qualification covers that
  CPU implementation, not the upstream GPU/Transformers 5.17 environment.
- Respect the checkpoint's 2048-token window. Reject overflow and input special
  markers; never truncate or inherit the SDK's automatic extension to 8192.
- Report exact packed input tokens and zero generated tokens. Probabilities are
  unrounded restricted softmax at the checkpoint's temperature **2.2**; neither
  they nor their margin is a calibrated probability of task success.

Upstream attribution/license: `compute_plane/_vendor/von/NOTICE.md` and
`LICENSE.md`, both included as package data. Local Labs owns the protocol,
resource measurements, qualification and comparisons.
