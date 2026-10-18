# AVCS System 1: Provenance Review and API Serving Gate

**Review date:** 2026-10-18

**Checkpoint:** `runs/weights/candidates/avcs_system1_v1.pt`
**Decision:** do not publish or serve this checkpoint yet. Two candidate
checkpoints are demonstrably embedded in it, and their complete data lineage
and distribution rights are not established.

This is an evidence review, not a legal opinion. The local checkpoint was
inspected and exercised on a generated image only. It was not copied, modified,
uploaded, or used by a public service.

## 1. Frozen checkpoint identity and local smoke test

| Property | Observed value | Evidence |
|---|---|---|
| Architecture | `AVCSSystem1V1` | Checkpoint payload and JSON sidecar |
| Schema | 2 | Checkpoint payload and JSON sidecar |
| Parameters | 1,663,564 | Checkpoint payload, sidecar, and instantiated model |
| File size | 6,800,337 bytes | Local checkpoint |
| SHA-256 | `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4` | Recomputed before and after CPU inference |
| Sidecar creation time | `2026-10-18T05:15:50Z` | `avcs_system1_v1.json` |
| Recorded training summary | 3 epochs, batch 16, LR 0.0005, best validation loss 0.8749, 13.45 seconds | Sidecar and embedded `training_metadata` |

A local CPU smoke test loaded the frozen checkpoint through
`load_system1(checkpoint_path=..., device="cpu")` and inferred one generated
256x384 RGB test frame. Observed load time was 158.35 ms and one-frame
inference time was 232.58 ms in that process. The output reported target
presence `PRESENT` with confidence 0.966, a selected pixel point `(71.2, 40.9)`,
two regions, and the uncertainty flag
`multimodal_target_ambiguity`. This is an API/integrity smoke test only, not an
accuracy result or a service benchmark; that single prediction is not evidence
that the proposed target was correct. Process RSS was not measured. The
checkpoint digest was unchanged afterward.

The report does not use a private screenshot. A single generated frame cannot
validate accuracy, generalization, or calibration.

## 2. Initialization and candidate lineage

The checkpoint sidecar does not record an initialization manifest, random
seed, dataset identity, or the candidate tensor counts. The current
`train_system1_v1.py` calls `initialize_from_candidate_weights(model)` but
discards its returned counts. It then freezes both the grounder and word
recognizer while training the proposal branch.

To test actual use rather than infer it from files merely being present, the
final checkpoint tensors were compared with each candidate's saved tensors:

| Candidate | Result of exact tensor comparison | Finding |
|---|---|---|
| `candidate_m3_highres_v2.pt` (SHA-256 `600ce046539c1274dd265166933660f2ab2005e412517eee74c4df26071c673f`) | All 195 of 195 grounder state tensors exactly match the final checkpoint's `grounder` state. The trainer loads this candidate into that branch and freezes the branch before training. | **Used.** This is direct tensor evidence of inherited initialization, not a conclusion from file presence alone. |
| `candidate_m3_word_recognizer_v1.pt` (SHA-256 `2a63534dde9545d0f33d0632534349935e0f190646e2ecfd273caecfa5065639`) | All 26 of 26 word-recognizer state tensors exactly match the final checkpoint's `word_recognizer` state. The trainer loads this candidate and freezes the branch before training. | **Used.** Direct tensor evidence. |
| `candidate_m3_fpn_v1.pt` (SHA-256 `15d5c8002079d9dc7feb772f54e33d72b1f4cf740dceb76d35f215b2d1ed7466`) | None of the 62 compatible tensors remained exactly equal in the final checkpoint. The FPN/proposal branch is trainable, so this cannot establish whether the candidate was loaded before training. It was present locally before the final checkpoint timestamp, and the inspected trainer loads it conditionally. | **Unknown.** Neither use nor non-use is proved by the final tensors. |

The exact matches establish that this checkpoint is not a clean random-init
training result. The grounder candidate's sidecar points to
`candidate_m3_highres_v1.pt`; its parent `runs/weights/goal_grounder_best.pt.prev`
has the same digest as that v1 checkpoint (`74412a750d6e28bbecd15b2dad84ee4110d2ce3b10bcfc4f70a173fcff18e2e1`).
The v2 sidecar reports `recorded_ui_click_and_absence_demonstrations`; the v1
payload reports `recorded_ui_click_demonstrations`. Those records do not grant
redistribution rights.

The word-recognizer candidate sidecar records training metrics and vocabulary
but does not identify a dataset or license. Its training source defaults to
the procedural UI manifest, but there is no run record tying that exact
candidate to that invocation.

The FPN candidate payload records `datasets/ui_grounding/manifest.json` as its
data provenance, with 160 training, 40 validation, and 40 held-out test
samples. That establishes the candidate's declared data source, not whether
the final run loaded it.

No explicit call to a public model hub or named third-party pretrained
checkpoint was found in the inspected System 1 training entry point. This is
not proof that every candidate ancestor was trained only from random
initialization; complete ancestor training records are absent.

## 3. Training record and configuration gaps

The final checkpoint's embedded metadata records loss history and a few
hyperparameters, but omits the training seed, manifest/data hashes,
initialization sources and counts, software revision, and command line. The
current trainer has `seed=42` as a function default, but the checkpoint does
not prove that the final invocation used that code version or default.

The available PowerShell history contains no matching
`train_system1_v1` invocation. The file
`runs/weights/avcs-visual-cortex-training.log` is an unrelated older
AVCS pretraining log (12 epochs, different 633,298-parameter model and output
path); it is not a System 1 run log. No contemporaneous System 1 stdout log or
run manifest was found.

The local checkpoint's modification time is 2026-10-18 10:45 local time. The
candidate files predate it, but timestamps alone are not run provenance.

## 4. Data, annotations, generated assets, and rights

| Material | Observed evidence | Redistribution status |
|---|---|---|
| Default UI grounding manifest | `datasets/ui_grounding/manifest.json`: 270 frames, split as 160 train / 40 validation / 40 held-out / 30 negative; declared sources are 240 `procedural_synthetic` and 30 `procedural_negative`; 1,886 interactive points and 270 affordance masks. | **Unknown.** The manifest declares procedural generation but is not a license, provenance attestation, or proof that every ancestor used only this manifest. |
| Goal-grounder demonstrations | Candidate metadata reports 2,392 training and 682 validation examples from `recorded_ui_click_and_absence_demonstrations`; its ancestor reports recorded UI click demonstrations. The repository contains demonstration manifests, JSONL samples, and captured frames. | **Unknown / blocked from public redistribution.** No participant/owner authorization, image clearance, annotation license, or complete inclusion record was found. Treat the recorded frames as potentially user-sensitive. |
| Word-recognizer training data | Candidate sidecar has metrics but omits source manifest and rights evidence. | **Unknown.** |
| Fonts and synthetic visual assets | The procedural generator uses Pillow drawing and a font lookup helper. The repository does not include the host font files or a per-font asset/license inventory for the generated corpus. | **Unknown pending asset inventory.** Do not distribute generated training frames until the exact font files and generation inputs are cleared. |
| Model code | The repository root `LICENSE.md` is Apache-2.0. | **Source license only.** It does not license checkpoint tensors, captured images, annotations, fonts, or candidate ancestors. |

The existing source-only package is a distinct artifact and excludes weights,
data, masks, screenshots, and evaluation report. Its source-release status
does not clear this checkpoint.

### Distribution-rights decision by artifact

| Artifact | Source and license evidence | Status |
|---|---|---|
| AVCS System 1 source code | Model repository `LICENSE.md` states Apache-2.0. This applies to repository source under its terms, not embedded training materials or weights. | **ALLOWED** as source code under Apache-2.0 terms |
| Final `avcs_system1_v1.pt` checkpoint | No separate weight license or model-weight grant. Exact grounder and word-recognizer state tensors match candidate checkpoints whose training data rights are not documented; FPN initialization is unknown. | **BLOCKED** from public distribution and hosted inference |
| Grounder candidate and ancestors | Sidecars/payload name recorded UI click and absence demonstrations; no consent, license, or redistribution grant found. | **BLOCKED** from distribution |
| Word-recognizer candidate | Candidate sidecar has no identified training-data source or license. | **BLOCKED** from distribution |
| FPN candidate | Candidate payload declares the UI grounding manifest, but no license for that data is recorded; whether this candidate initialized the final checkpoint is unknown. | **UNKNOWN** lineage and rights; excluded from distribution |
| Procedural UI frames, point labels, masks | `datasets/ui_grounding/manifest.json` declares synthetic sources and splits, but has no asset-level license/provenance grant; no complete final-training manifest links this corpus to every checkpoint component. | **UNKNOWN**; do not distribute until cleared |
| Recorded demonstration frames and annotations | Demonstration manifests/samples contain recorded desktop UI evidence; no participant/owner authorization or annotation license was found. | **BLOCKED** from public distribution pending written clearance |
| Fonts and generated third-party assets | Generator uses a font lookup helper, but no exact font file inventory, font licenses, or generation asset record was found. | **UNKNOWN** |
| External package dependencies | The source package declares dependencies for package-manager installation; no dependency wheels or vendor code are bundled in the checkpoint. The exact transitive versions/licenses used for a future deployment image were not audited. | **UNKNOWN** for a bundled deployment image; resolve with a versioned dependency/license inventory before image publication |

No row marked **UNKNOWN** or **BLOCKED** is cleared by the repository's
Apache-2.0 source license. No dataset or screenshot should be bundled to
resolve this gap; obtain written rights evidence or generate a replacement
corpus from assets with documented permissions.

## 5. Clean-lineage training experiment recommendation

Do not train a purportedly clean model from the current default training
function as-is: it unconditionally calls the optional candidate initializer,
which loads local candidate files when found.

Before a clean run:

1. Obtain written rights evidence for the generator, fonts, and any other
   bundled or host-provided assets, or replace them with documented
   project-owned assets.
2. Generate a new dataset using only those cleared assets. Do not use
   `runs/demonstrations`, candidate checkpoints, existing frames/masks, or
   evaluation artifacts with unknown rights.
3. Add a distinct random-initialization mode that never calls
   `initialize_from_candidate_weights`; fail if an inherited checkpoint is
   specified. Verify the initialized model state against a fresh seeded
   initialization before the first optimizer step.
4. Split by independent generator seed/template into train, validation, and
   final test sets; record manifest and per-file hashes.
5. Save a separate checkpoint under a new clean-lineage path and record
   source revision, exact command, seed, library versions, dataset manifest
   hash, asset licenses, initialization mode, metrics, and output SHA-256.
6. Evaluate target presence/absence, grounding precision/recall, region
   proposals, closed-vocabulary word accuracy, CPU latency, and peak memory.
   Compare against the current checkpoint only in a private, rights-compliant
   environment; do not promise similar accuracy.

No clean training was started: the necessary authorized-data and asset record
is not yet available. This avoids falsely labelling a run as clean.

## 6. Separate platform repository: inspected implementation

The platform repository is `nandkishorrathodk-art/AVCS`, distinct from this
model repository. The inspected remote `main` was commit
`7fc7a9ce902d6854fd9c3da6bb17d3786ed03b48` on 2026-10-18.

- It is a Next.js 16.4.2 Node application (`node >=20.9`) with Supabase
  dashboard authentication and SQLite API-key/billing storage. No deployment
  manifest, Python source, PyTorch runtime, hosting-provider declaration, or
  deployment credentials were present in the inspected checkout.
- Public API keys are SHA-256 hashed in the SQLite registry, checked in
  constant time, validated for environment/revocation/scope, and charged
  against usage. The current inference routes require `vision:read`.
- Tier limits are applied by an in-process sliding-window limiter. The
  platform README requires one instance and a persistent volume for SQLite;
  the limiter is not a shared distributed limiter.
- Existing `POST /api/v1/perceive` authenticates callers, accepts base64
  PNG/JPEG (decoded image limit 8 MiB), and calls the separate private Python
  worker at `POST /v1/perceive`. That worker runs the existing `HumanEye`
  perception component, not `AVCSSystem1V1`. `GET /api/v1/models` lists the
  existing `avcs-human-eye` and `avcs-goal-grounder` models. Neither route
  serves the System 1 checkpoint.
- The existing Next.js proxy uses server-only `AVCS_INFERENCE_ENDPOINT` and
  `AVCS_INFERENCE_API_KEY`, sends `X-API-Key`, rejects non-loopback plain HTTP
  in production, rejects redirects, and has a 30-second upstream timeout.
- `/api/v1/perceive` parses `request.json()` before checking decoded image size;
  no explicit raw JSON body cap was found in that route. The Python worker's
  current image decoder has no explicit byte or pixel-dimension cap. The worker
  has no application-level rate/concurrency limit identified in the inspected
  route. These must be addressed for a System 1 serving path.
- Request telemetry stores status, endpoint, elapsed time, key prefix, cost,
  and client IP in a 250-entry in-memory buffer; the inspected route does not
  place image bytes in this application log. The Python decoder processes
  pixels in memory and has no application persistence path. Hosting, reverse
  proxy, crash-dump, and platform logging retention were not available to
  inspect, so cloud retention is **unknown**. Cloud inference transmits the
  submitted pixels to the private worker and hosting infrastructure.
- Platform documentation describes a separate privately reachable Python
  service, but does not identify the real hosting provider or prove that an
  authorized Python/PyTorch runtime is deployed. No secret values or live
  environment configuration were available or inspected.

## 7. Proposed endpoint contract (not implemented or deployed)

Keep the System 1 visual-observation contract separate from the current
OpenAI-compatible chat envelope:

- Platform caller endpoint: `POST /api/v1/perception`.
- Private model worker endpoint: `POST /v1/perception`.
- Caller authentication: existing platform `Authorization: Bearer <issued
  key>` flow with `vision:read`, revoked-key checks, organization rate limits,
  and normal successful-request billing.
- Platform-to-worker authentication: existing server-only
  `AVCS_INFERENCE_API_KEY` in `X-API-Key`; do not use
  `AVCS_LLM_API_KEY` for this hop.
- JSON request: `{"image_base64":"<PNG-or-JPEG-base64>","query":"Save button"}`.
  Query is optional; cap UTF-8 bytes at 256. Reject remote image URLs.
- Enforce a raw body cap before parsing, decoded image cap of 8 MiB, supported
  PNG/JPEG signatures, valid decode, and bounded dimensions/pixel count in
  both tiers. A suggested initial dimension cap is 4096x4096 (subject to
  measured memory limits), with requests that exceed any cap rejected before
  model inference.
- Return request ID, `model.id=avcs-system1-v1`, package/model version
  `1.0.0-preview.1`, width and height, category/confidence, target presence and
  confidence, candidate and selected coordinates, region proposals, supported
  word labels and their confidence, ambiguity, and uncertainty flags. Convert
  region arrays from the model's `[y_min,x_min,y_max,x_max]` convention into
  named coordinate fields. Do not include `action_authorized` as an action
  grant; coordinates remain observations only.
- Load only a checkpoint whose configured path and expected SHA-256 match an
  operator-approved artifact. `AVCS_SYSTEM1_ENABLED=false` must be the default;
  missing clearance, enablement, path, digest, or file returns a stable 503,
  never a success-shaped placeholder.
- Use a bounded concurrency semaphore, explicit worker timeout, health
  readiness, sanitized errors, and structured metadata-only logs. Do not
  persist uploaded image bytes. Keep raw pixel logging disabled and publish
  accurate retention disclosure only after actual host logging policies are
  verified.

No code for this new System 1 endpoint was added to either repository, and no
model endpoint was deployed. This contract is a preparation specification,
not an available API.

## 8. Deployment gate and remaining evidence

| Gate | Status |
|---|---|
| Frozen checkpoint identity and parameter count | **Verified locally** |
| Actual checkpoint inference on a generated RGB frame | **Verified locally on CPU; one smoke sample only** |
| Complete candidate/ancestor lineage | **Not established; confirmed grounder and word-recognizer candidates; FPN initialization unknown** |
| Training-run command and durable stdout | **Not found for the final run** |
| Training data, annotation, font, and captured-image rights | **Unknown; not cleared** |
| Public checkpoint redistribution permission | **Not established; do not upload** |
| Platform caller authentication and usage controls | **Implemented in inspected source, but platform runtime not verified** |
| Dedicated System 1 route | **Specified only; not implemented** |
| Python/PyTorch hosting target and credentials | **Not available / unverified** |
| Authenticated request through a deployed System 1 service | **Not run** |
| Service p50/p95 latency, RSS, and concurrency behavior | **Not measured** |
| Raw-image service retention at hosting layer | **Unknown** |

Deployment remains disabled until rights, hosting, and endpoint implementation
are independently reviewed. Do not publish a public URL or claim that the
existing `avcs-human-eye` endpoint is serving System 1.
