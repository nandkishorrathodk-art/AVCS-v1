# AVCS System 1 v1.0.0-preview.1 — Distribution Rights Audit

**Audit date:** 2026-10-18<br>
**Scope:** proposed source-only developer-preview distribution<br>
**Decision:** the Apache-licensed software subset may be prepared for
publication. The trained checkpoint and evaluation data/report are excluded
pending the evidence identified below. This is a provenance review, not legal
advice or a substitute for approval from the rights holder.

## Status key

- **ALLOWED** — a concrete license or distribution fact supports including the
  named material in this source-only bundle.
- **BLOCKED** — the material is deliberately excluded because required rights
  are not documented.
- **UNKNOWN** — the available records do not establish provenance or
  redistribution permission; the material is excluded until resolved.

## Artifact-by-artifact findings

| Artifact | Source and observed evidence | License / permission evidence | Status | Release treatment and evidence needed |
|---|---|---|---|---|
| AVCS System 1 Python source (`avcs/`, `forge/`) and release documentation | Repository source; focused package is staged from these Python packages by `scripts/build_system1_release.py`. | Root `LICENSE.md` is Apache License 2.0 and identifies Parth Labs, copyright 2026. The release wheel and sdist carry the license. | **ALLOWED** | Included in the source-only wheel/sdist. Confirm that the publisher has authority to grant this license and that all included contributions are covered by it. |
| System 1 checkpoint `avcs_system1_v1.pt` | Local ignored file `runs/weights/candidates/avcs_system1_v1.pt`; SHA-256 `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4`; sidecar gives architecture `AVCSSystem1V1`, schema 2, 1,663,564 parameters and training settings. | The checkpoint is not covered by an individual weight license or redistribution grant in its sidecar. Apache-2.0 on repository software is not evidence of permission for these weights. | **BLOCKED** | Not included in the source-only bundle or public release. Required: written authorization from the checkpoint owner identifying the exact checkpoint/hash and permitted public hosting, copying, and downstream use. |
| Potentially inherited FPN candidate weights | Training function `initialize_from_candidate_weights()` loads `candidate_m3_fpn_v1.pt` if present. The file is present locally and predates the final checkpoint. SHA-256: `15d5c8002079d9dc7feb772f54e33d72b1f4cf740dceb76d35f215b2d1ed7466`. No matching provenance sidecar is present. | No upstream/source checkpoint, training dataset, author, or weight license is recorded for this file. The final checkpoint sidecar does not record whether this initializer loaded. | **UNKNOWN** | Not distributed. Required: training-run evidence confirming whether it was used, plus model/data lineage and rights evidence for this candidate and any ancestors. |
| Potentially inherited goal-grounder candidate weights | Training code loads `candidate_m3_highres_v2.pt` if present. It is present locally and predates the final checkpoint. Its sidecar names `candidate_m3_highres_v1.pt` as its source; the v1 sidecar names `runs/weights/goal_grounder_best.pt.prev`. The referenced parent exists locally and has the same SHA-256 as v1: `74412a750d6e28bbecd15b2dad84ee4110d2ce3b10bcfc4f70a173fcff18e2e1`. V2 SHA-256: `600ce046539c1274dd265166933660f2ab2005e412517eee74c4df26071c673f`. | Available sidecars record parent paths and evaluation fields but do not grant a weight license or document complete ancestor/data rights. The final checkpoint sidecar does not record whether this initializer loaded. | **UNKNOWN** | Not distributed. Required: training-run evidence confirming initializer use, and provenance/redistribution rights for v2, v1, `goal_grounder_best.pt.prev`, and all ancestors. |
| Potentially inherited word-recognizer candidate weights | Training code loads `candidate_m3_word_recognizer_v1.pt` if present. The file is present locally and predates the final checkpoint. Sidecar hash: `2a63534dde9545d0f33d0632534349935e0f190646e2ecfd273caecfa5065639`. | Sidecar includes training metrics but no data source or weight license. The final checkpoint sidecar does not record whether this initializer loaded. | **UNKNOWN** | Not distributed. Required: training-run evidence confirming initializer use, dataset and weight lineage, and a redistribution grant. |
| AVCS System 1 training frames and labels | The actual System 1 training/evaluation input manifest is `datasets/ui_grounding/manifest.json`: 270 frames, reported as 240 `procedural_synthetic` and 30 `procedural_negative`, with 1,886 annotated interactive points. The training entry point defaults to this manifest. No image or mask files are included in this bundle. | Manifest states procedural provenance but contains no dataset license or rights attestation. The separate `avcs/training/dataset_manifest.json` is not the entry point's default and contains one `win32_fast_capture` frame among procedural records. Candidate-weight training lineage is incomplete. | **UNKNOWN** | Dataset frames, masks, and annotations are excluded. Required: owner attestation that all training and validation data (including candidate-checkpoint ancestry) were generated from authorized material and may be used to train and distribute the resulting weights. Identify and clear the captured frame if it contributed to any candidate. |
| Local evaluation corpus and `system1_v1_evaluation.json` | Benchmark command uses `datasets/ui_grounding/manifest.json`; the local JSON report references the frozen checkpoint SHA, architecture, and parameter count. Report SHA-256: `BC99D821845211006BCC383DA48D37282246513D470FFE28B5BF583C38A88A00`. | No separate dataset license or rights attestation was found. The report is derived from evaluation on that corpus; its inclusion is deferred with the corpus rather than treating a benchmark output as a license. | **UNKNOWN** | Report and data are retained locally but excluded from the public source-only bundle. Required: data-rights clearance or a documented determination that the aggregate report may be redistributed independently. |
| Screenshots, image frames, masks, and other visual assets | No screenshots, image frames, or masks are copied into the source-only bundle. System 1 manifests describe synthetic frames; the supplemental manifest also references a Win32-captured frame. | No rights records for any captured images were found. Synthetic generator code refers to host-installed fonts (including Segoe UI, Arial, and DejaVu Sans); these font files are not bundled. | **BLOCKED** for captured/user images; **UNKNOWN** for any captured data used in candidate lineage | No images or masks are included. Required before using any captured frame: provenance, consent/authorization, and a license covering training and redistribution. Do not include private user screenshots. |
| Runtime/build/test dependencies | Runtime: `torch>=2.0.0`, `numpy>=1.24.0,<3.0`, `Pillow>=9.0.0`. Build: `setuptools>=77.0.3`, `wheel`; optional developer tools: `pytest`, `build`; benchmark helper: `psutil`. Wheel and sdist contain project Python source and metadata, not third-party dependency wheels or native libraries. Installation resolves dependencies separately from package indexes. | Each dependency is subject to its own upstream license and distribution terms; those licenses are not relicensed as AVCS. No dependency binaries are bundled in the source-only release ZIP. | **ALLOWED** as declarations only | Dependencies are not vendored or repackaged. Consumers must obtain them from their configured package index and comply with their respective licenses. |
| Other third-party assets (fonts, icons, screenshots, model files) | No external model files, font files, or screenshot/image assets are included in the source-only bundle. Font lookup paths in the synthetic generator refer to fonts installed on the local host at runtime. | No separate third-party asset license files were found for assets bundled with the proposed release; none are bundled. | **ALLOWED** for the inspected bundle contents | Keep host fonts and data assets out of the release. Re-audit before adding any such asset. |

## Checkpoint lineage limitation

The final checkpoint's sidecar records its training parameters and digest, but
not candidate initialization or dataset identity. The training source loads
three optional candidate files when they exist; all three were present locally
and timestamped before the final checkpoint. No training log or recorded
initialization counts establish whether they were loaded in the exact run that
created the final file. Conservatively treat them as potential ancestors until
that is resolved. No claim is made that the candidates contain third-party
pretrained weights; their provenance and rights are **UNKNOWN**.

## Bundle verification and disposition

The versioned source-only bundle is
`dist/avcs-system1-v1.0.0-preview.1-source-only.zip`. It contains the licensed
source distributions, license, package documentation, model card, this audit,
and version/checksum manifest. It excludes the checkpoint, candidate weights,
dataset, annotations, screenshots, and evaluation JSON. `SHA256SUMS` covers
every file inside the archive except the checksum file itself; a separate
`.sha256` sidecar covers the ZIP.

The local checkpoint and evaluation report remain in the developer workspace
for verification only. They are not cleared for publication by this audit.
Any earlier full ZIP containing the checkpoint is a restricted local artifact,
not a publication candidate.

## Publication decision

- **Legally cleared:** only the identified Apache-2.0 source/documentation
  subset, subject to the publisher's authority and contribution review.
- **Source-only package locally built and clean-install tested:** yes.
- **Model weights/data/evaluation publicly cleared:** no.
- **Uploaded or publicly available:** no.

Do not publish the checkpoint, candidate weights, training/evaluation corpus,
or evaluation report until the missing rights and lineage evidence above is
obtained and reviewed. A source-only bundle is not a model-weights release and
does not enable checkpoint inference without independently authorized weights.
