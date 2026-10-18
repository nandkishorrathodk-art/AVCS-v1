# AVCS System 1 Platform Integration

**Status:** design only for System 1; the inspected platform has separate
inference routes for other AVCS models.
**Model package:** `avcs-system1` `1.0.0rc1` (preview `1.0.0-preview.1`).

This document separates two repositories:

- **Model/source repository:** `Parth-labs-in/AVCS`, containing the System 1
  Python package and its existing FastAPI inference service.
- **Platform website repository:** `nandkishorrathodk-art/AVCS`, containing the
  Next.js application inspected at commit
  `7fc7a9ce902d6854fd9c3da6bb17d3786ed03b48` on 2026-10-18.

The platform repository was inspected independently. Existing API-key
authentication and other-model inference endpoints are implemented there, but
they do not serve `AVCSSystem1V1`. This proposal is not evidence that a System 1
endpoint or deployment exists.

## What exists today

The inspected platform is a Next.js 16.4.2 Node application with Supabase
dashboard authentication and SQLite-backed API-key and billing stores. It has
authenticated routes including `POST /api/v1/perceive`,
`POST /api/v1/ground`, and an OpenAI-shaped
`POST /api/v1/chat/completions`. The API-key store persists SHA-256 hashes,
checks scope/revocation/expiry, and applies a tier-based in-process
sliding-window rate limit. The existing perception route uses `vision:read`
and calls the separate Python worker's `/v1/perceive`; the worker runs the
existing `HumanEye` component, not System 1. The `/api/v1/models` list likewise
contains `avcs-human-eye` and `avcs-goal-grounder`, not `AVCSSystem1V1`.

The existing public `/api/v1/perceive` route accepts base64 PNG/JPEG with an
8 MiB decoded-image limit, but it parses JSON before enforcing any explicit raw
request-body limit. The Python worker's current image decoder has no explicit
byte or pixel-dimension limit, and no application-level worker rate or
concurrency control was found. The platform request log stores metadata
including client IP, but not image bytes in the inspected route. Hosting and
proxy retention policies remain unknown.

The model/source repository separately has a Python FastAPI server with
authenticated `POST /v1/perceive` and `POST /v1/ground` operations. They use
other AVCS components and the server-side `AVCS_INFERENCE_API_KEY` in
`X-API-Key`; they are not System 1 endpoints. The existing decoder processes
images in memory but has no explicit size cap and can return decoder exception
details. No actual hosting provider, deployed Python service, or deployment
secrets were verified.

No deployment provider or production environment configuration is identified
in the inspected platform checkout. Its documentation expects a separately
reachable Python service; it does not prove PyTorch hosting or a live
deployment.

## Proposed public contract

Implement a dedicated public JSON API route in the platform repository at
`POST /api/v1/perception` (`src/app/api/v1/perception/route.ts`) and a separate
private Python worker route at `POST /v1/perception`. Do not implement it as a
chat-completions compatibility shim: System 1 is a typed visual-observation
API and does not need an LLM.

### Request

```http
POST /api/v1/perception HTTP/1.1
Authorization: Bearer avcs_live_<caller-key>
Content-Type: application/json
```

```json
{
  "image_base64": "<base64-encoded PNG or JPEG bytes>",
  "query": "Save button"
}
```

`query` is optional. The existing platform parser supports PNG/JPEG and an
8 MiB decoded-image limit; it does not currently support WebP. Decode the image
to RGB before inference. Reject malformed base64, unsupported formats, empty
images, oversized raw/decoded payloads, invalid dimensions, and queries over
256 UTF-8 bytes. Reuse the platform's server-side hashed API-key registry and
require `vision:read` (matching its current visual inference routes), including
revocation, expiry, and scope checks.

### Successful response

The platform adapter should normalize the model's typed observation into a
stable API schema. Coordinates remain evidence, not instructions or permission
to act.

```json
{
  "model": {
    "id": "avcs-system1-v1",
    "version": "1.0.0-preview.1"
  },
  "request_id": "req_<server-generated-id>",
  "frame": {
    "width": 1280,
    "height": 800,
    "coordinate_convention": "pixel_origin_top_left"
  },
  "observation": {
    "primary_category": "desktop_canvas",
    "category_confidence": 0.0,
    "target": {
      "query": "Save button",
      "presence": "UNCERTAIN",
      "confidence": 0.0,
      "center_px": null,
      "center_normalized": null,
      "ambiguous": true,
      "margin": 0.0
    },
    "regions": [
      {
        "kind": "button",
        "bbox_px": {
          "x_min": 0.0,
          "y_min": 0.0,
          "x_max": 0.0,
          "y_max": 0.0
        },
        "confidence": 0.0,
        "text": null,
        "text_confidence": 0.0
      }
    ],
    "uncertainty_flags": ["example_uncertainty"],
    "temporal_change": null
  }
}
```

The values above are schema examples, not benchmark results or a sample model
prediction. The implementation must map System 1's normalized `(x, y)` target
coordinates and pixel `(x, y)` coordinates explicitly. Internal region boxes
currently serialize as `[y_min, x_min, y_max, x_max]`; the platform API should
convert them to named `x_min/y_min/x_max/y_max` fields rather than leaking an
ambiguous tuple. Include only predictions supported by the checkpoint's
vocabulary.

Do not expose an `action_authorized` field in this API. A candidate point,
presence score, or prediction is never computer-action authorization. Any
consumer that can act must independently ground and validate a fresh target,
apply its own safety/approval policy, and verify the postcondition.

### Errors

Use a stable envelope and do not return exception text, local paths, stack
traces, submitted pixels, or credentials:

```json
{
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "System 1 inference is not available.",
    "request_id": "req_<server-generated-id>"
  }
}
```

Recommended statuses: `400` malformed image, `401` missing/invalid caller key,
`403` revoked/expired/insufficient scope, `413` payload too large, `415`
unsupported media, `422` invalid query or image dimensions, `429` rate limit,
`503` disabled/unlicensed/unavailable model, and `504` inference timeout.

## Request path and credential boundaries

1. A caller sends the request to `/api/v1/perception` using its own
   `avcs_live_…` or `avcs_test_…` key.
2. The platform validates the key and `vision:read` scope using its existing
   server-side SQLite API-key registry, and applies the configured rate and
   billing policies.
3. The platform server forwards the bounded request over private HTTPS to a
   dedicated Python System 1 inference worker. Do not run PyTorch inference in
   browser JavaScript.
4. The worker loads only an explicitly approved, separately mounted
   checkpoint and returns a typed observation. It does not call an LLM or
   dispatch a motor action.

Suggested server-only configuration placeholders:

```dotenv
# Disabled until checkpoint lineage and redistribution rights are cleared.
AVCS_SYSTEM1_ENABLED=false
AVCS_INFERENCE_ENDPOINT=https://<private-system1-worker>
AVCS_INFERENCE_API_KEY=<server-only-32+-character-service-credential>
AVCS_SYSTEM1_CHECKPOINT_PATH=/run/secrets/avcs_system1.pt
AVCS_SYSTEM1_CHECKPOINT_SHA256=<approved-checkpoint-sha256>
AVCS_LLM_API_KEY=<optional-outbound-llm-credential-not-used-by-system1>
```

`AVCS_INFERENCE_API_KEY` is an internal platform-to-worker credential, never
a browser variable or substitute for each caller's scoped API key. The
inspected platform proxy and existing FastAPI worker use this server-only name
for their shared service secret. Provision it through each runtime's secret
manager; do not log it. `AVCS_LLM_API_KEY` is unrelated and must not be read or
forwarded by System 1.

Keep `AVCS_SYSTEM1_ENABLED=false` and do not mount a checkpoint while rights
remain unresolved. The current checkpoint demonstrably includes the grounder
and word-recognizer candidates; the FPN candidate's initialization is unknown.
The checkpoint and its inherited candidates are **BLOCKED/UNKNOWN** for
redistribution in
[`RELEASE_RIGHTS_AUDIT.md`](RELEASE_RIGHTS_AUDIT.md). Once written permission
and complete lineage are established, mount the approved artifact read-only,
verify its SHA-256 against the separately approved configuration and sidecar,
then load with `load_system1(checkpoint_path=..., device="cpu")`. Do not fall
back to a repository-relative default checkpoint, a candidate weight, or an
unverified download. A mismatch or missing permission must leave the endpoint
disabled and return `503`.

## Security, capacity, and privacy gates

These are proposed initial limits to validate with deployment-specific
load-testing; they are not currently enforced by either repository:

- Enforce a **12 MiB encoded request limit** at the ingress and route before
  fully buffering JSON; also enforce decoded dimensions no larger than 4,096
  pixels per side and 16,777,216 pixels total. These are proposed limits and
  still require worker memory validation.
- Accept only PNG/JPEG after decoding and format verification. Bound
  query length by UTF-8 bytes. Do not trust the client MIME label alone.
- Apply a distributed per-caller-key rate limit (proposed starting point:
  30 requests/minute) and a bounded worker semaphore (proposed maximum two
  concurrent inferences per CPU worker). Return `429` rather than queueing
  without a bound. Adjust only after measured capacity tests.
- Bound upstream inference to 15 seconds initially; enforce cancellation and
  return `504` on timeout. This is a proposed service budget, not a measured
  latency guarantee.
- Restrict the worker to private service networking, authenticate
  platform-to-worker requests, use TLS, and do not permit caller-selected
  worker URLs or filesystem paths. Keep checkpoint paths fixed by deployment
  configuration.
- Never log request bodies, base64 image data, decoded pixels, API keys, or
  full natural-language queries. Log request ID, status, bounded dimensions,
  timing, and a non-secret model version only.
- The inference code path should keep image bytes and decoded pixels in bounded
  process memory only and should not write them to application storage. This is
  not a claim of zero retention: ingress, hosting, crash, tracing, and backup
  retention are provider-specific and were not established for the platform.
  Configure and verify those policies before production use. Tell callers that
  submitted pixels leave their device and are processed by the service; obtain
  any required consent.
- Do not record images in Supabase, analytics, traces, error reports, or
  benchmark fixtures by default. Avoid request-body logging in the reverse
  proxy and Next.js runtime as well as in the Python worker.

The platform has an in-process tier limiter, not a shared distributed
limiter; its README recommends a single instance with durable SQLite storage.
The existing `/api/v1/perceive` handler has no raw JSON body cap before parsing,
and the Python worker has no explicit image byte/pixel or concurrency cap.
Hosting-provider ingress limits and image-log retention are unknown. These
must be fixed or verified in the actual deployment.

## Deployment readiness checklist

- [ ] Rights audit clears the exact checkpoint SHA-256, all possible inherited
      initializers, training data, and third-party training assets for public
      or hosted inference use.
- [ ] The dedicated route reuses caller-key hashes, scope, expiry, revocation,
      rate-limit, and billing checks; it requires `vision:read`, matching the
      existing visual inference routes.
- [ ] The platform route handler and a separate Python worker are implemented,
      reviewed, tested, and documented. No OpenAI-compatible API is claimed
      unless separately implemented and validated.
- [ ] Inference is disabled by default and uses an explicit read-only
      checkpoint path plus a verified SHA-256 allowlist.
- [ ] Service and caller credentials are configured server-side; no secret is
      prefixed `NEXT_PUBLIC_` or bundled into browser assets.
- [ ] Ingress/body limits, image decode limits, distributed rate limiting,
      timeouts, concurrency caps, and generic error responses are verified.
- [ ] No-action-authorization semantics are enforced end-to-end; consumers
      cannot dispatch a GUI action solely from returned coordinates.
- [ ] Hosting-provider access-log, crash-report, tracing, and backup retention
      are reviewed; image/body logging is disabled or explicitly disclosed.
- [ ] TLS/private service networking and abuse monitoring are enabled.
- [ ] Clean deployment smoke tests cover authentication, scope denial, payload
      limits, malformed media, inference timeout, model-unavailable behavior,
      JSON schema, and CPU resource bounds.
- [ ] A real deployed URL and downloadable service/model artifacts are verified
      before anyone describes the integration as live.

**Current status:** platform authentication and existing model routes were
inspected. System 1 contract documented; no System 1 endpoint implemented, no
checkpoint loaded by the platform worker, and no deployment or weight upload
performed. See [`SYSTEM1_MODEL_PROVENANCE_AND_SERVING_GATE.md`](SYSTEM1_MODEL_PROVENANCE_AND_SERVING_GATE.md)
for checkpoint lineage and the publication/deployment gate.
