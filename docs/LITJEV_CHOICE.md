# Resident LitJEV method

`compute_plane.litjev_choice.assess(request, post, identity)` implements one
question with the frozen LitJEV prompt format over a caller-owned native backend.
The caller must retain the GPU lease and verify the resident model throughout
all template, tokenizer and completion operations. It must drain a timed-out
inference before releasing that lease. Kernel never starts or switches models.

The complete input is bounded to 4096 tokens; all candidate codes must be exact
single-token continuations at the answer boundary. Every candidate must appear
in the top128 pre-sampling logprobs. Missing scores, truncation, modified sampling
probabilities or incomplete usage are failures, not zero-valued candidates.
The selected answer is the restricted-softmax maximum, not the sampled token.

The native API produces one transport token. `choice_assessment.validate_answer`
accepts one output token only for identity method `litjev-native-first-token-v1`;
existing NLI workers still report zero. This is truthful usage, not an inference
optimization claim. Normalized probability is not calibrated correctness.

The serving worker and CPU fallback belong to Local Labs. Joint route evidence,
role policy, admission and activation belong to OS. No network endpoint, routing
policy or new dependency is installed by importing this module.

[Upstream attribution and license](third-party/litjev/README.md).
