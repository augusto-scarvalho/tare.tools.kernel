# Optional OpenJEV choice worker

`compute_plane.openjev_choice_worker` adapts a pinned three-class NLI checkpoint
to the existing Choice contract. It does not train a model or grant routing
authority. Ordinary Kernel users do not import its optional ML dependencies.

## CPU CLI

```text
python -B -m compute_plane.openjev_choice_worker /absolute/checkpoint.json
```

Stdin is one bounded `state`/`question`/`options` JSON request; stdout is one
bounded Choice answer. Diagnostics go to stderr. The CLI uses four CPU threads,
FP32 and no CUDA. It refuses less than 36 GiB available RAM before model load:
the measured original checkpoint needs about 18 GiB process RSS plus operating
headroom. Call through `run_assessment` with a finite deadline. On the measured
i7-13700K, the roughly 700-token three-option request takes about 29 seconds
warm plus about nine seconds of load/verification, so use a 60-second timeout.
Larger requests can legitimately time out and must retain the baseline route.

The checkpoint manifest has schema `tare.local-labs/openjev-checkpoint/1`,
`model`, exact `revision`, `subfolder`, a local `directory`, `torch_version`,
`transformers_version`, and a `files` mapping containing SHA-256 for precisely
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`.
All bytes and the expected architecture, label order and NLI template are
validated before loading. Runtime is offline and rejects remote model code.
The measured setup uses torch 2.14.0+cu126 and transformers 5.17.0; psutil is
also required by the CPU CLI. Accelerator probes use accelerate 1.14.0.

## Meaning of the answer

The premise contains the full state, question and all option descriptions. Each
hypothesis is `The correct answer is: <description>`. Every pair is tokenized
without truncation and checked against 4096 tokens before any inference.
Pairs execute sequentially with no shared prefix cache. `input_tokens` counts
the actual sum across all pairs; `output_tokens` is zero.

The model returns contradiction/entailment/neutral logits. The adapter computes
each hypothesis's entailment probability and normalizes these across choices,
using log-space arithmetic. This preserves the upstream entailment ranking.
Raw entailment logits from different pairs are not directly comparable. The
resulting Choice distribution is a relative preference, **not calibrated Work
success**, and has no authority to admit a model or alter a running session.
The probe retains all original logits and NLI probabilities separately.

## Resource experiments

`OpenJevChoiceModel` exposes explicit CUDA and hybrid placements for Local Labs;
the production CLI exposes CPU only. The lab runner requires a verified parent
GPU lease before a CUDA/hybrid load. Hybrid uses four final text layers and the
score/norm layers resident on CUDA; Accelerate retains other weights on CPU
and transfers them for GPU execution. It is weight offload, not CPU matrix
computation. Inputs must never be sent to an offloaded weight's meta placeholder.

Measured results and unchanged-service recovery are owned by Local Labs:
`docs/research/OPENJEV_FEASIBILITY_2026-09-21.md`. The first hybrid setup failure
is retained alongside the repaired run. Full GPU and hybrid probes do not
authorize concurrent residency with Qwen under today's exclusive lease.
