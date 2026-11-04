# Gemini compatibility usage

`gemini-chat` is an explicit per-request usage contract for Gemini's
OpenAI-compatible endpoint. It does not change request formatting, keys, cache
affinity or the generic `chat-inclusive` contract.

Gemini's `completion_tokens` excludes thinking while `total_tokens` includes
it. Generated output is therefore `total_tokens - prompt_tokens`, with thinking
derived by subtracting the visible completion. Output pricing applies to the
generated total once; thinking is a subset. These semantics are documented in
[Google's compatibility clarification](https://discuss.ai.google.dev/t/gemini-3-6-flash-openai-compatible-token-limits-response-fields-and-auth-keys/183372)
and the [native usage metadata](https://ai.google.dev/api/generate-content).

Raw counters remain unchanged. Missing totals do not establish complete output;
missing `prompt_tokens_details.cached_tokens` remains unknown, including when
the other counters are valid. Implicit caching has no separate cache-creation
category here. Negative differences, cache beyond input, malformed counters and
contradictory explicit reasoning counters invalidate the normalized totals.

Ordinary OpenAI-compatible profiles still use their own explicit contract and
retain strict total arithmetic. An old session keeps the contract saved at its
creation. A new Gemini preset can adopt this contract without rewriting old
usage, spend receipts or provider history. Historical revaluation, if needed,
must be a separately labeled derivation over the retained raw counters.
