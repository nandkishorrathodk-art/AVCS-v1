# Clean-Room System 1 Rights Evidence Matrix

**Audit status: BLOCKED — evidence inventory only, not a legal opinion or rights approval.**
No training, evaluation, publication, deployment, or checkpoint modification was performed for this audit. The attestation in `CLEANROOM_DATA_RIGHTS_ATTESTATION.md` remains `PENDING` and unsigned.

## Audit scope and fixed evidence

- Repository root: `C:\Users\nandk\Documents\antigravity\splendid-babbage`
- Branch: `forge-screen-vision`
- Audited `HEAD`: `4cf794cb577cccf3a7785701eeabc877c0b51942`
- Dataset manifest SHA-256: `abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1`
- Frozen baseline checkpoint SHA-256: `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4`
- Clean-lineage candidate exists: **No** (`runs/weights/clean_lineage_candidates/avcs_system1_clean_v1.pt`)
- Source license SHA-256: `4ba072dc7d327ab5159afbc744619e5d25a798e102740a53fc6fb4a5ed6adc61`
- Project metadata SHA-256: `92c9ab63255fd8831fc9bfca0182db82341755c793842a7a0e35e1dcffd80a46`

The manifest validator confirmed 112 images and 112 annotation files: 64 train, 16 validation, 16 held-out, and 16 negative-test records. The four split groups use disjoint layout-family sets. Validation reported 112 distinct image hashes and 112 annotation hashes and retained `PENDING_OWNER_ATTESTATION`. These checks establish file and annotation integrity, not ownership or permission.

The manifest records `git_commit=4cf794...` and `generator_worktree_dirty=true`. The clean-room generator source files are untracked at that revision: no commit in the available Git history contains them. The recorded commit therefore identifies the repository base at generation time; it does not establish who authored or owns those uncommitted files.

## Contribution, data, and model evidence

“Not established” means the inspected repository records do not prove the permission. It is not a determination that the work is unlicensed or that permission is unavailable.

| Category | Source and ownership evidence found | License / permission evidence | Git or file evidence | Internal training | Commercial inference | Public weight redistribution | Source / corpus redistribution | Remaining uncertainty / evidence needed |
|---|---|---|---|---|---|---|---|---|
| Clean-room renderer | First-party-style Pillow drawing primitives, fixed palettes, geometry, and seeded layout generation in `avcs/dataset/cleanroom_v1/generator.py`; no real application captures or downloaded image assets are identified by the generator. “First-party-style” describes the implementation method only; authorship is not independently established. | Repository declares Apache-2.0. No file-specific grant, contributor assignment, or owner authorization for these uncommitted contributions was found. | Generator SHA-256 `9eeccf7f3b56c0ed949619b83ad22c7d0c1d63fe45e40bdbc7c10c93588d01fb`; path is untracked; no commit history for this path. | Not established | Not established | Not established | Not established | Authorized owner must confirm authorship, authority, and contribution rights, or provide contributor assignments/permissions and identify any non-project contributions. |
| Bitmap glyph table | `avcs/dataset/cleanroom_v1/glyphs.py` defines a small fixed 5×7 bitmap table and errors on unsupported glyphs; it does not request a system-font fallback. | Repository Apache-2.0 declaration is not proof that the table's author could license it. No separate glyph provenance or license notice exists in the inspected evidence. | SHA-256 `881436e0dae57d0a1077cfa0976e205046ba496c96968cf30f8fb0dad1285e73`; untracked, no path-specific commit history. | Not established | Not established | Not established | Not established | Owner/contributor must attest to original authorship or provide the glyph source and permission evidence. |
| Dataset adapter and schema | `avcs/dataset/cleanroom_v1/dataset.py` maps renderer annotations to training labels; `__init__.py` records schema/version. | No distinct data or annotation license. Rights depend on code authorship and permission for the exact generated corpus. | `dataset.py` SHA-256 `28ac2a4d2ffb07359d719e30d71702d0f32238b710e4cccd7ecdead3a9115e21`; `__init__.py` SHA-256 `c9393f17fc87634c918055b1bb44f5a2d9256cfa62b04b4fe0e653433919107f`; both untracked and absent from commit history. | Not established | Not established | Not established | Not established | Owner must establish authorship/contribution rights and explicitly grant the contemplated corpus uses. |
| Generated images and labels | Manifest records each image and annotation path and SHA-256. Labels/coordinates are produced from the renderer, not from an independently labeled real-world dataset. Generator identifies primitive-drawn generic windows, controls, symbols and text. | No independent corpus license or owner permission is present. Renderer-derived labels do not independently prove the legal provenance of the renderer code. | Manifest SHA-256 above; `datasets/cleanroom_system1_v1/manifest.json`. Per-item hashes and provenance declarations are in the manifest. No legacy captures, demonstration frames, masks, or checkpoints are declared as inputs. | Not established | Not established | Not established | Not established | Owner must review the exact manifest and confirm rights in generated pixels/annotations and intended resulting-weight use. Dataset rights must not be inferred from synthetic generation alone. |
| System 1 architecture and training code | The clean trainer constructs `AVCSSystem1V1()` from random initialization, rejects checkpoint loads during construction, and writes to a separate output path. No training has run. The candidate architecture and clean trainer files are untracked; related architecture modules include both tracked/modified and untracked files. | Repository metadata says Apache-2.0; inspected records do not provide independent authorship/assignment evidence for all model code contributors or a specific weights grant. | `system1_v1.py` SHA-256 `eb16b2c7c590ad01decbb7154066379e09978ad21bf441c07666f68470742bca`; trainer SHA-256 `48a43fa81c5c9230aae7541ad51b263e2f190cdff3e98dd231fae97301c29497`; evaluator SHA-256 `e701061ef5a410dff9d88658dc09892e2f2f11acd46806231bfcfcea1d80da6b`. Clean trainer/evaluator and `system1_v1.py` are untracked. | Not established | Not established | Not established | Not established | Owner must establish relevant source contributions and decide/authorize any rights for a future checkpoint. The trainer's random-initialization design is technical evidence, not an ownership grant. |
| Existing baseline and historical candidates | Frozen baseline remains a separate reference. Prior provenance records report matches with earlier candidate tensor weights; no clean-room training consumed those checkpoints. | Existing source license does not resolve the baseline's training data, weight lineage, or redistribution rights. No baseline/candidate weight permission is established by this audit. | Baseline SHA-256 above, size 6,800,337 bytes. No clean candidate checkpoint exists. Prior tensor lineage is documented in `SYSTEM1_MODEL_PROVENANCE_AND_SERVING_GATE.md`; not re-audited or reinterpreted here. | Not established | Not established | Not established | Not established | Any use or release of legacy weights requires its separate lineage/data-rights review. Do not conflate the baseline with a future clean candidate. |
| Repository source license | `LICENSE.md` contains Apache License 2.0 text and says “Copyright (c) 2026 Parth Labs”; `pyproject.toml` declares `license = "Apache-2.0"` and lists Parth Labs as package author. | This is evidence of the repository's stated license, not proof that the stated licensor owns every contribution or that all relevant contributors granted rights. It does not settle generated-data or trained-weight rights. | `LICENSE.md` SHA-256 above; `pyproject.toml` SHA-256 above. License file appears in Git history (including release commit `db7c0b24d82026f0fe07cd17f996e27cdcb95a89`). | Conditional only on valid licensor authority and applicable terms; not established for these uncommitted contributions | Same | Same | Source redistribution is stated by the repository license only to the extent the licensor had authority over the particular source and contributions. Dataset redistribution is not separately established. | Need confirmation of licensor authority, contributor chain-of-title/permissions, and a legal review of whether this license covers the specific corpus and future model artifact. |
| Git authorship and project identity | Git records contain commits by named identities, including the configured local author and a release commit. Repository owner/name and commit author fields are not independent legal identity, employment, agency, assignment, or authorization records. | No contributor license agreement, copyright assignment, signed owner authorization, or organizational delegation record was found in the inspected repository records. No contributor/notice files were found under `.github`. | Git identity and commit history are technically observable; the clean-room source paths have no commits. The whole worktree has many unrelated modified/untracked files, so this audit did not attribute all local changes to one person. | Not established | Not established | Not established | Not established | Authorized owner must identify relevant contributors and supply/confirm the applicable rights and authority evidence. |

## Dependency inventory

The clean-room generator, dataset reader, trainer, and evaluator declare/use the project's direct runtime requirements `torch`, `numpy`, and `Pillow`. The code's generator imports Pillow, NumPy, and Torch; the trainer/evaluator use the same scientific stack. `pytest` is used for tests; `setuptools` and `wheel` are build requirements.

The runtime inventory below was derived from the active `.venv` package metadata and recursively evaluated `Requires-Dist` dependencies for the current Python environment (Python 3.14.7). `uv.lock` is multi-version/marker-based. Its Python >=3.12 pins agree for the listed common packages except that the installed Pillow is 13.0.0 while the lock resolves Pillow 12.3.0; `uv.lock` specifies `wheel==0.48.0`, but wheel is not installed in the active environment. The installed environment therefore is not an exact reproduction of every lockfile package.

License values below are package metadata/classifier indicators, not legal determinations. Paths are relative to `.venv/Lib/site-packages/`; license-text hashes were checked for the principal file shown. Packages with missing SPDX expressions, multi-license declarations, or bundled third-party code require review of the actual notices before redistributing dependencies or artifacts.

| Package | Role / dependency path | Installed version | Metadata license identifier or value | Principal installed license text and SHA-256 | Audit note |
|---|---|---:|---|---|---|
| torch | Direct runtime; model, trainer, evaluator | 2.14.1 (`2.14.1+cpu` runtime) | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | `torch-2.14.1.dist-info/licenses/LICENSE` — `0fc92425d130d9c87a5eb7427082e3afdbff6ef169bc678e15abba4c2d37c934` | Distribution contains numerous third-party license texts under `torch-2.14.1.dist-info/licenses/third_party/`; their complete inventory/obligations have not been legally reviewed. |
| numpy | Direct runtime; generator, dataset, model | 2.5.4 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | `numpy-2.5.4.dist-info/licenses/LICENSE.txt` — `a804dff0ead9fadc5293456410bcbfc32bf024be9c4513459663fb7b442d2341` | Additional vendored license files are present under `numpy-2.5.4.dist-info/licenses/numpy/`; not all have been reviewed. |
| Pillow | Direct runtime; raster drawing and image loading | 13.0.0 | `MIT-CMU` | `pillow-13.0.0.dist-info/licenses/LICENSE` — `988ffe8ab3480b0f4e9a0ab2702bffd73d9fb6b47198043fb3ba101a62fac236` | Installed version is not the Python >=3.12 version pinned by current `uv.lock` (12.3.0); resolve and review the exact release environment before packaging. |
| filelock | Torch transitive | 4.2.2 | MIT | `filelock-4.2.2.dist-info/licenses/LICENSE` — `608c89d5060ae9921adccf3236695bc654a9946e12323ef6c021dfa04e294d48` | Metadata and text present. |
| fsspec | Torch transitive | 2026.9.0 | BSD-3-Clause | `fsspec-2026.9.0.dist-info/licenses/LICENSE` — `2dc35496ce53a7307915c008a844aad53e772b49b34cdd00445067691407ec94` | Metadata and text present. |
| Jinja2 | Torch transitive | 3.1.6 | License expression absent; classifier says BSD License | `jinja2-3.1.6.dist-info/licenses/LICENSE.txt` — `3b49dcee4105eb37bac10faf1be260408fe85d252b8e9df2e0979fc1e094437b` | Identifier is not supplied as SPDX metadata; verify actual license text. |
| MarkupSafe | Jinja2 transitive | 3.0.4 | BSD-3-Clause | `markupsafe-3.0.4.dist-info/licenses/LICENSE.txt` — `4631ec0db5fd90a547e336817264c6798214338146f8ac94b4a57f96ee8c9ec4` | Metadata and text present. |
| mpmath | SymPy transitive | 1.3.0 | `BSD` (non-specific metadata value) | `mpmath-1.3.0.dist-info/LICENSE` — `c26cae81da4508e5e249985777a33821f183223ebb74d7f8cfbf90fe7eef2fb7` | Exact BSD variant is not encoded in the metadata; review the license text. |
| networkx | Torch transitive | 3.7 | BSD-3-Clause | `networkx-3.7.dist-info/licenses/LICENSE.txt` — `6025f323ea29dc2f0ee0abd9523bf910186f55b0e148e6ebf646c55566278e2e` | Metadata and text present. |
| sympy | Torch transitive | 1.14.0 | `BSD` (non-specific metadata value) | `sympy-1.14.0.dist-info/licenses/LICENSE` — `07a5e9819f727b4986ad2829c7a29a6320d42575f720eb24d71b7fef573a0286` | Exact BSD variant and any included notices require review. |
| typing_extensions | Torch transitive | 4.16.0 | PSF-2.0 | `typing_extensions-4.16.0.dist-info/licenses/LICENSE` — `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` | Metadata and text present. |
| setuptools | Build requirement and Torch dependency | 84.0.0 | MIT | `setuptools-84.0.0.dist-info/licenses/LICENSE` — `86da0f01aeae46348a3c3d465195dc1ceccde79f79e87769a64b8da04b2a4741` | Distribution includes vendored libraries with separate license texts; complete notice review remains outstanding. |
| wheel | Build requirement | Not installed; `uv.lock` resolves 0.48.0 | Not observed locally | No installed license text to verify in this environment. | Install/resolve in a clean build environment and verify the exact wheel distribution's license and notices before packaging. |
| pytest | Test dependency | 9.1.1 | MIT | `pytest-9.1.1.dist-info/licenses/LICENSE` — `ca836a5f9ecca3b2f350230faa20a48fb8b145653b5568d784862df864706b9b` | Test-only dependency. |
| pluggy | Pytest transitive | 1.7.0 | MIT | `pluggy-1.7.0.dist-info/licenses/LICENSE` — `d6b65e6c213a5d0b577911d34d6e5949b9f59d76c238c5071a2f3fc16cfb2606` | Test-only dependency. |
| iniconfig | Pytest transitive | 2.3.1 | MIT | `iniconfig-2.3.1.dist-info/licenses/LICENSE` — `3409fa91f7ace557894632676656e32264fe5ef7581535725dc9a23774551bd4` | Test-only dependency. |
| packaging | Pytest transitive | 26.3 | Apache-2.0 OR BSD-2-Clause | `packaging-26.3.dist-info/licenses/LICENSE` and `LICENSE.APACHE` — hashes `cad1ef5bd340d73e074ba614d26f7deaca5c7940c3d8c34852e65c4909686c48` and `0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594` | Test-only dependency; preserve applicable notices. |
| Pygments | Pytest transitive | 2.21.0 | BSD-2-Clause | `pygments-2.21.0.dist-info/licenses/LICENSE` — `a9d66f1d526df02e29dce73436d34e56e8632f46c275bbdffc70569e882f9f17` | Test-only dependency; AUTHORS file is also present. |
| colorama | Pytest transitive | 0.4.6 | BSD License (classifier/metadata) | `colorama-0.4.6.dist-info/licenses/LICENSE.txt` — `cac35c02686e5d04a5a7140bfb3b36e73aed496656e891102e428886d7930318` | Test-only; confirm precise license text before redistribution. |

This is a package metadata and installed-license-file inventory, not a full software bill of materials or legal clearance. In particular, embedded native/vendor components and all bundled notices in Torch, NumPy, and setuptools have not been enumerated and reviewed exhaustively. The available install environment is not itself permission to redistribute package code.

## Rights questions kept separate

| Intended activity | What repository evidence supports | Current decision |
|---|---|---|
| Internal experimentation and training | Synthetic-only generation is reproducible and the manifest validates. Source/data authorship and project-owner authority are not proven; no owner attestation exists. | **Not authorized by available evidence.** |
| Commercial inference service | No specific commercial-service permission or confirmed chain of title for the generator, model code, corpus, and future weights was found. Dependency license inventory alone does not grant rights to the project model. | **Not established.** |
| Public redistribution of trained weights | No clean-lineage weights exist, and no explicit owner permission/terms for the future weights exist. The baseline has separate unresolved lineage. | **Not established; do not distribute.** |
| Redistribution of source and generated corpus | Apache-2.0 is declared for the repository, conditional on the licensor having rights. The corpus has no separate owner approval or distribution terms. | **Source permission is conditional/unverified for the relevant contributions; corpus redistribution is not established.** |

These are separate decisions. Approval for internal research would not automatically authorize commercial operation or public redistribution of model weights, and source-code license metadata does not itself establish permission for the generated corpus or model artifacts.

## Attestation gate: observed behavior and limits

Milestone 4.4 recorded the earlier text-only gate. That implementation has since been hardened; the current contract is documented in [CLEANROOM_MILESTONE_4_5_REPORT.md](./CLEANROOM_MILESTONE_4_5_REPORT.md). Current training/evaluation requires an explicitly approved human record, a signed Ed25519 payload binding the exact manifest digest and all four distinct permission decisions, and a trust configuration pinned by a separately supplied SHA-256. Plain-text assertions and the previous checkboxes are not cryptographic approval.

The current template is still `PENDING`, unsigned, and has no owner public key or trust configuration. The signature verifier establishes only signature validity and membership in the independently pinned trust set; it does **not** authenticate legal authority independently, prove the truth of legal claims, or establish that cited rights evidence is complete. Human owner/legal review and an auditable trust-root approval remain necessary.

## Evidence confirmed

- Manifest bytes still hash to `abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1`.
- Generator's supported `--validate-only` check passes for 112 records, split counts, per-file hashes, and disjoint family assignments.
- Clean-room source describes deterministic primitive drawing and an authored bitmap glyph table; its declared inputs do not include legacy datasets, captures, or checkpoints.
- The Git history contains project license and model-source history, but does not contain the untracked clean-room generator files at the audited `HEAD`.
- The repository declares Apache-2.0 in `LICENSE.md` and `pyproject.toml`.
- Active dependency metadata and principal installed license files were inventoried above; declared build dependency `wheel` is missing from the active environment.
- Frozen baseline SHA-256 remains `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4`.
- No clean-lineage output checkpoint exists.

## Evidence missing and resolution path

### Resolvable from repository records, if the records exist

- Locate contributor agreements, copyright assignments, employer/contractor work-product terms, or signed permissions covering the actual generator, glyph table, model code, and annotations. No such evidence was found in the inspected Git history or `.github` contributor/notice files.
- Resolve whether all relevant worktree changes came from authorized contributors; many unrelated files are also modified/untracked, so Git identity alone cannot establish authorship.
- Complete a reproducible dependency/SBOM review for the exact training/build environment, including `wheel`, exact resolved package versions, Torch/NumPy/setuptools bundled notices, and the Pillow lock-versus-installed mismatch.
- Attach the relevant rights/permission documents and hashes to the attestation evidence list; repository code/license metadata is not a substitute.

### Requires the actual authorized project owner and possibly counsel

- Identify the person or organization legally empowered to attest for Parth Labs and all relevant contributors; provide a genuine signature or approved organizational signing record.
- Decide and explicitly authorize each intended scope separately: internal training, commercial inference, public redistribution of weights, and redistribution of source/generated data.
- Confirm that the exact manifest hash above and the renderer-produced labels are covered, and list all restrictions, attribution, notice, or other obligations.
- Review legal rights in the source contributions and future learned weights. Do not assume synthetic status or Apache-2.0 source metadata decides the result.

## Next action before training

An authorized project owner must review the exact evidence, resolve or explicitly scope all rights questions, complete the attestation for manifest SHA-256 `abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d`, set blockers to `NONE` only if true, mark each exact statement only if supported, and provide a genuine approval record. Then independently validate the attestation and manifest and run the existing gated CPU smoke test before considering training. Until then the candidate remains **BLOCKED**; no checkpoint or evaluation result is authorized by this evidence audit.
