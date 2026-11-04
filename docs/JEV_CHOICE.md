# Optional hosted Jev choice worker

`compute_plane.jev_choice_worker` answers the bounded `choice_assessment`
stdin/stdout protocol with TypeSafe's hosted Jev (System One) model: one
`POST https://api.typesafe.ai/v1/systemone` carrying a single Choice question.
It produces observations only; OS keeps routing, abstention and fallback authority.

**The request leaves the machine.** `state`, `question` and options are sent to
TypeSafe. Use it only where sending that text is acceptable.

```text
set TYPESAFE_API_KEY=...            (required)
python -B -m compute_plane.jev_choice_worker < request.json
```

- Model pinned to `jev-1.13.0` so thresholds tuned against it survive alias moves;
  `TYPESAFE_DEFAULT_MODEL` overrides it, `TYPESAFE_BASE_URL` the endpoint.
- One call, no retries. 429/529 → `ASSESSMENT_RESOURCE_BUSY`; 401/403 or no key or no
  network → `ASSESSMENT_BACKEND_UNAVAILABLE`; anything else → `ASSESSMENT_WORKER_FAILED`.
- Usage reports input tokens only: Jev bills input and its output tokens are not text.
- Jev's `confidence` is not passed through; calibrate thresholds per question
  (see the Jev 1.13 jaggedness notes on docs.typesafe.ai).
