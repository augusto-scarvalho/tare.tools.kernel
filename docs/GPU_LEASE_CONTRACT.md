# Shared local GPU: contract `gpu-lease/1`

A node with a GPU shared by text models, image jobs and training serves text through a gateway. The node owns the
GPU lock and its queue (the node service, `node/` in [tare.tools.os](https://github.com/augusto-scarvalho/tare.tools.os)); the kernel owns the client side described here; the OS decides
whether a caller waits for the GPU or goes to another model. Nothing here reserves the GPU from another machine: a
free GPU is an observation, never a reservation.

## What the node says

- `GET /v1/fleet/status` → `gpu_coordination`: `enabled`, `held` (bool), `owner.kind` (`image`, `text` or none),
  `queue` (waiting tickets with `kind`), and optionally `eta_seconds`: the node's estimate, in seconds, until text
  gets the GPU (from the owner's and the queue's learned durations; absent or null without an estimate).
- A text request refused because the GPU is taken: **503** with the body
  `{"error": {"type": "gpu_busy" | "gpu_lease_timeout" | "gpu_lease_cancelled", "message": ..., "model": ...}}`,
  optionally `"retry_after_seconds": N` in `error`, and optionally the header `Retry-After: N` (whole seconds).

## What the kernel does with it

- The HTTP worker passes on only the status, the `Retry-After` seconds (capped at 24 h, so a 12-hour training run
  reads as 12 h) and, for a 503, the error type when it is one of the three above
  (`native_http_worker.busy_hints`). The message and any other text of the reply stay in the worker: they may echo
  credentials or the request.
- A busy GPU is `ModelFailure('HTTP_503', retryable=True)` with `busy_reason` and `retry_after` on the failure and in
  its transport receipt, and costs **no wait**: the fixed 5 s pause before a retry (`UNAVAILABLE_WAIT`) stays for a
  provider briefly unavailable (502 to 504 without a busy type), where it avoids landing in the same outage.
- `model_facts.gpu_availability` keeps `eta_seconds` when the GPU is `BUSY` or `QUEUED`.

## What the kernel does not do

It does not decide to reserve, wait for the GPU, load or switch models: the caller has the policy and the budget, and
the reservation client below acts only when its caller asks. Retries stay within the session's `max_model_retries`.

## `gpu-lease/2`: named reservations

A holder elsewhere (a bench, a long run) can hold the node's GPU for the text models by name, one reservation at a
time, up to 4 h (tare.tools.os `node/docs/SHARED_GPU.md`):

- `POST /v1/gpu/reservations`, `Authorization: Bearer <node token>`, body `{"name", "holder", "max_seconds",
  "ttl_seconds" (15-600), "wait_seconds" (0-120)}` → **201** `{"id", "secret", "fence", "expires_at", "hard_until"}`;
  **409** `gpu_reserved` (another holder) or **503** `gpu_busy` (a job kept the GPU), both with `Retry-After`;
  **401** without the token; **404** `reservations_disabled` on a node without one.
- `POST /v1/gpu/reservations/{id}/renew` and `DELETE /v1/gpu/reservations/{id}`, body `{"secret", "fence"}`: **404**
  when it is over, **409** `stale_fence` for an old holder. Without renewal it ends after `ttl_seconds`.
- While it lasts, requests with the secret in `X-Tare-GPU-Reservation` use the GPU; every other text or image
  request gets **503** `gpu_reserved` with `Retry-After` = the time left. `/v1/fleet/status` shows
  `gpu_coordination.reservation` (`id, name, holder, fence, expires_at, hard_until, seconds_left`, never the secret)
  and `eta_seconds`.

The kernel's client is `compute_plane/gpu_lease_client.py`: `hold(origin, token, name, max_seconds)` reserves,
renews every ttl/3 in the background and releases on exit (`LeaseLost` when the node ended it, `ReservationRefused`
when it would not reserve); `RESERVATION_HEADER` names the header. `gpu_reserved` is a busy reason like the others,
and `model_facts.gpu_availability` reports a reservation as `BUSY` / `GPU_RESERVED` with its time left. The caller
(the OS) decides who reserves and which requests carry the secret.
