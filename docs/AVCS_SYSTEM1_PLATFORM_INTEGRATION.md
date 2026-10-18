# AVCS System 1 Platform Integration

**Status:** design only; no platform endpoint or model deployment is enabled.
**Model package:** `avcs-system1` `1.0.0rc1` (preview `1.0.0-preview.1`).

This document separates two repositories:

- **Model/source repository:** `Parth-labs-in/AVCS`, containing the System 1
  Python package and its existing FastAPI inference service.
- **Platform website repository:** `nandkishorrathodk-art/AVCS`, containing the
  Next.js application inspected at commit
  `b917824cc5bef742da194ec77ba2f745b80ffe13` on 2026-10-18.

The platform repository was inspected independently. The proposal below is
not evidence that an endpoint has been implemented, that the platform accepts
OpenAI-compatible requests, or that either repository has been deployed.

## What exists today

The inspected platform is a Next.js 16.4.2 application with Supabase
authentication configuration, a browser-side heuristic affordance playground,
and a browser-only API-key demonstration. It has no `src/app/api` route
handlers, OpenAI chat-completions route, or configured model-provider client.
The playground's connected-component/edge detector is explicitly documented
as a heuristic, not a trained model. The API-key console keeps generated
plaintext and its hash in component state; it does not persist keys or
authenticate server requests. `vision:read` and `vision:write` are currently
scope definitions, not deployed authorization.

The model/source repository separately has a Python FastAPI server with
authenticated `POST /v1/perceive` and `POST /v1/ground` operations. These
currently use other AVCS components, base64 JSON, and the server-side
`AVCS_INFERENCE_API_KEY` read through `X-API-Key`; they are not System 1
endpoints. The Docker Compose service binds to loopback and has CPU/memory
limits, but no application rate limiter or request-body cap was found in the
server route. Do not treat that service as a ready public System 1 deployment.

No platform host, public API base URL, model-artifact host, per-user API-key
store, or server-side retention policy was established by this inspection.
The Next.js package scripts (`dev`, `build`, `start`) do not identify a hosting
provider.

## Proposed public contract

Implement a dedicated JSON API route in the platform repository, preferably
`POST /v1/perception`. In a Next.js App Router deployment, a matching route
handler can be placed at `src/app/v1/perception/route.ts`. Do not implement it
as a chat-completions compatibility shim: System 1 is a typed visual
observation API and does not need an LLM.

### Request

```http
POST /v1/perception HTTP/1.1
Authorization: Bearer avcs_live_<caller-key>
Content-Type: application/json
```

```json
{
  "image_b64": "<base64-encoded PNG, JPEG, or WebP bytes>",
  "query": "Save button"
}
```

`query` is optional. Decode the image to RGB before inference. Reject malformed
base64, unsupported image formats, empty images, oversized payloads, invalid
dimensions, and queries exceeding the documented UTF-8 byte limit. The caller's
API key must be resolved server-side from a persistent hashed-key store and
authorized for `vision:write`; reject missing, revoked, expired, or
under-scoped keys. Browser-side key-generation state is not authentication.

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

1. A caller sends the request to the platform's server route using its own
   `avcs_live_…` or `avcs_test_…` key.
2. The platform validates that key and `vision:write` scope against a durable,
   server-side store. The current key console is only a prototype; persistence
   and route authentication must be implemented before this step is active.
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
AVCS_SYSTEM1_INFERENCE_URL=https://<private-system1-worker>
AVCS_PLATFORM_API_KEY=<server-only-service-credential>
AVCS_SYSTEM1_CHECKPOINT_PATH=/run/secrets/avcs_system1.pt
AVCS_SYSTEM1_CHECKPOINT_SHA256=<approved-checkpoint-sha256>
AVCS_LLM_API_KEY=<optional-outbound-llm-credential-not-used-by-system1>
```

`AVCS_PLATFORM_API_KEY` is an internal platform-to-worker credential, never a
browser variable or substitute for each caller's scoped API key. Configure the
worker's inbound service credential through its server-side secret mechanism
(the existing FastAPI service currently names this `AVCS_INFERENCE_API_KEY`).
If both settings represent the same service credential, provision them from
the same secret-manager entry; do not log either value. `AVCS_LLM_API_KEY` is
unrelated and must not be read or forwarded by System 1.

Keep `AVCS_SYSTEM1_ENABLED=false` and do not mount a checkpoint while rights
remain unresolved. The current checkpoint and possible inherited candidate
weights are **BLOCKED/UNKNOWN** for redistribution in
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
  fully buffering JSON; also enforce decoded dimensions no larger than 8,192
  pixels per side and 32 million pixels total. These align with the inspected
  platform playground constants but still require worker memory validation.
- Accept only PNG/JPEG/WebP after decoding and format verification. Bound
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

The current platform repository does not establish production ingress limits,
distributed rate limiting, a hosting provider, or the provider's image-log
retention. These must be confirmed and tested in the actual deployment rather
than inferred from the local Next.js scripts.

## Deployment readiness checklist

- [ ] Rights audit clears the exact checkpoint SHA-256, all possible inherited
      initializers, training data, and third-party training assets for public
      or hosted inference use.
- [ ] A persistent caller-key store exists; keys are stored hashed, scopes,
      expiry, and revocation are checked server-side, and the route requires
      `vision:write`.
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

**Current status:** platform contract inspected; proposal documented; no
platform files changed, no endpoint implemented, no checkpoint loaded, and no
deployment or upload performed.
