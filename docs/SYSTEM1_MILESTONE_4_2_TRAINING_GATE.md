# AVCS System 1 Milestone 4.2 — Clean-Lineage Training Gate

**Decision: BLOCKED — no new candidate trained.**

The repository currently has no training dataset or generation assets with
sufficiently documented redistribution rights to support the requested clean
lineage. Training was deliberately not started. The frozen baseline and all
candidate checkpoints were left in place.

## Frozen baseline

| Item | Verified value |
|---|---|
| Checkpoint | `runs/weights/candidates/avcs_system1_v1.pt` |
| SHA-256 before audit | `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4` |
| SHA-256 after audit | `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4` |
| Sidecar | `runs/weights/candidates/avcs_system1_v1.json` |
| Sidecar SHA-256 | `f7f82d8b7152d5ec42904cd70ab76478f670aad5653d1bf86fd1daa838d10456` |
| Architecture / parameters | `AVCSSystem1V1` / 1,663,564 |

Direct tensor comparison using the repository's CPU PyTorch environment found:

| Optional candidate | Compatible tensors | Exact matches in baseline | Conclusion |
|---|---:|---:|---|
| `candidate_m3_highres_v2.pt` | 195 | 195 | Grounder candidate was used; the trainer loads this branch and freezes it. |
| `candidate_m3_word_recognizer_v1.pt` | 26 | 26 | Word-recognizer candidate was used; the trainer loads this branch and freezes it. |
| `candidate_m3_fpn_v1.pt` | 62 | 0 | Unknown whether it initialized the trainable FPN branch. Divergence after training cannot distinguish loaded-then-trained from random initialization. |

The current trainer conditionally loads all three fixed candidate paths when
each path exists relative to the working directory. All three checkpoint files
exist in this checkout. This describes what a default invocation from the
repository root would do now; it does not prove which FPN initialization was
used in the historical run. The final checkpoint is not fully trained from
scratch.

## Exact current training inputs

`avcs/training/train_system1_v1.py::train_system1_v1()` defaults to:

- Manifest: `datasets/ui_grounding/manifest.json`
- Data root: `datasets/ui_grounding`
- Training split: 160 `train` frames
- Validation split: 40 `val` frames
- Final held-out and negative test splits: not loaded by this trainer
- Direct image reads: the 200 PNG frames referenced by the train/validation
  records; annotation points and categories are inline in the manifest
- Affordance masks: paths are declared in the manifest, but this trainer never
  opens the `.npy` masks
- Split index JSON files: not read by this trainer
- Current code's function defaults: 15 epochs, batch 16, LR 0.0005, seed 42;
  CLI defaults to 10 epochs and overrides the function epoch default
- Optimizer/scheduler: AdamW and CosineAnnealingLR
- Output default: the **frozen baseline path itself**. Never invoke this
  default for a new run; it can overwrite the baseline.

The manifest declares all 270 records procedural (240 `procedural_synthetic`,
30 `procedural_negative`). That label identifies a generation method; it is
not a license or proof of asset ownership. The generator draws UI shapes and
icons procedurally and does not open a separate icon/image file in the
inspected rendering path. `primitives.py` searches host font paths and records
neither the selected font nor its digest in the dataset manifest. Current
host font hashes in the machine-readable inventory are only candidate font
identities, not proof those exact files rendered the existing PNGs.

The training loop uses category and FPN proposal losses only. It constructs
grounding heatmaps, text crops, and word labels in the dataset item, but the
optimizer loop does not consume those targets. It freezes the grounder and word
recognizer. A clean run of this exact loop would therefore not train all
System 1 heads, regardless of its initialization mode.

The machine-readable inventory is
[`system1_clean_training_input_provenance.json`](system1_clean_training_input_provenance.json).
It includes SHA-256 entries for 270 PNGs, 270 mask files, dataset metadata and
split files, relevant generation/training source files, optional checkpoints
and sidecars (including 15 candidate/checkpoint lineage files), current
candidate host-font files, and all 40 files under `runs/demonstrations/`.
Demonstration files are inventoried by name and digest only; they were not
opened, copied, or added to a proposed dataset. All data/asset/weight records
are marked excluded from a proposed from-scratch training dataset. The
inventory's hashes were rechecked against the current files: no mismatch or
missing-file errors. The FPN candidate JSON sidecar is missing.

The current manifest SHA-256 is
`66e8a3744c313a6e2c1cd1171f1f275cd633ae1c91493cfb086944634e822a90`.

## Unresolved inputs and evidence needed

1. **Recorded desktop demonstrations and annotations:** The grounder candidate
   sidecar reports recorded UI click/absence demonstrations, and the repository
   has demonstration manifests, JSONL records, and frames. No participant or
   content-owner permission, privacy clearance, dataset license, or
   redistribution grant was located. Keep these out of clean training and
   public distribution unless written rights cover both training and model
   weight distribution.
2. **Procedural raster corpus and labels:** The manifest and README describe
   procedurally generated frames and labels but do not provide an asset-level
   rights declaration or a guarantee that all System 1 components were trained
   only from this corpus. Obtain a signed project ownership/license statement
   covering rendered frames, annotations, masks, and derivatives.
3. **Fonts:** The renderer dynamically resolves fonts from Windows Segoe UI,
   Arial, or a Linux DejaVu fallback. The dataset does not capture which path
   or font bytes were selected when it was generated. Record and clear exact
   font files, or replace the font dependency with a documented project-owned
   or licensed font before generating a replacement corpus.
4. **Candidate ancestry:** Two candidate modules are directly proven in the
   baseline. Their training records do not establish complete rights for all
   ancestors. The word-recognizer candidate has no adequate data/license
   record. FPN candidate use in the baseline is unresolved. Do not reuse any
   of these weights in a clean-lineage candidate.
5. **Historical run provenance:** The baseline sidecar omits candidate load
   counts, data hash, run command, software revision, and seed. The trainer's
   current seed default does not establish the seed of the historical run.

The checkpoint/source license distinction and candidate/data rights findings
are also summarized in
[`SYSTEM1_MODEL_PROVENANCE_AND_SERVING_GATE.md`](SYSTEM1_MODEL_PROVENANCE_AND_SERVING_GATE.md).

## From-scratch training and evaluation status

No new from-scratch mode was added and no training command was run. There is
not yet an authorized training manifest to bind to such a mode, so adding a
mode and presenting this dataset as eligible would weaken rather than
establish the requested lineage. Before training, require a distinct explicit
random-initialization path that bypasses and rejects candidate loading, an
approved immutable data manifest, a separate output path, and a saved record
of actual initialized tensor sources, seed, config, source revision, optimizer,
epochs, split sizes, metrics, and checkpoint digest.

No new candidate exists; therefore no candidate-specific localization,
recognition, absence-rejection, latency, throughput, memory, pure-pixel, or
output-contract evaluation was run. Existing baseline/candidate metrics were
not copied or represented as new-run results. Once clearance exists, keep
train/validation/test isolated and report sample counts and measured failures
for every required metric.

## Release gate

- Baseline checkpoint and sidecar: **preserved; digest verified before and
  after**.
- Training input inventory: **created and hash-validated**.
- Rights clearance for a clean training corpus: **not established**.
- Candidate from scratch: **not trained; none produced**.
- Candidate evaluation: **not applicable; no candidate exists**.
- Model distribution or public endpoint: **not authorized by this milestone**.

Reopen the training gate only after the unresolved dataset and font rights are
documented and a revised training plan uses no recorded screenshots or
candidate weights unless their rights are separately cleared.
