# AVCS Milestone 4.5 — Reproducible Environment and Rights-Gate Hardening

**Status: PARTIAL — lockfile environment and gate hardening verified; rights remain pending.**

No training, model evaluation, checkpoint modification, publication, GitHub push, or deployment was performed.

## Protected artifact hashes

| Artifact | SHA-256 at task start | SHA-256 at task end | Status |
|---|---|---|---|
| Frozen baseline `runs/weights/candidates/avcs_system1_v1.pt` | `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4` | `c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4` | Unchanged |
| Canonical 112-frame manifest `datasets/cleanroom_system1_v1/manifest.json` | `abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1` | `abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1` | Unchanged |
| Rights attestation Markdown | `6c72dee955a34ea25529f3faaa8e5262e0ace48e94056267e2bfb0ffc529a0cf` | Recorded after final edits below | Still pending; documentation updated to explain signed approval |
| Rights evidence matrix | `0b12bc23ce7527588c21dc161e86aa78c811fba3b337604d54637865b357a9de` | Recorded after final edits below | Updated to point to this environment/gate report |

The start hashes were measured before edits. The attestation and matrix were intentionally edited for documentation; they were not replaced with approval. Their final hashes are reported in the final response. An unsigned unresolved [CLEANROOM_RIGHTS_APPROVAL_TEMPLATE.json](./CLEANROOM_RIGHTS_APPROVAL_TEMPLATE.json) was added; it is not accepted as approval.

## Dependency/environment reconciliation

### Existing environment

- Python: `3.14.7` (CPython, MSC v.1944, 64-bit AMD64)
- Platform: `Windows-11-10.0.26300-SP0`, AMD64
- `torch`: `2.14.1+cpu`
- `numpy`: `2.5.4`
- Pillow: `13.0.0`
- `pytest`: `9.1.1`
- `wheel`: missing
- `cryptography`: missing before this milestone

For Python 3.12+, `uv.lock` resolves Pillow `12.3.0`, NumPy `2.5.4`, Torch `2.14.1`, and wheel `0.48.0`. The mismatch was real: the existing environment was not lock-synchronized.

### Separate lock-synchronized environment

The existing `.venv` was not modified. A new environment was created at:

`C:\Users\nandk\AppData\Local\Temp\avcs-m45-b845cd73-uv`

It was created from the checked-in project lock using:

```powershell
uv sync --locked --extra dev --no-install-project --python .venv\Scripts\python.exe
```

Observed locked runtime: Python `3.14.7`, Windows 11 AMD64, `torch 2.14.1+cpu`, NumPy `2.5.4`, Pillow `12.3.0`, and `cryptography 50.0.2`. The dev/build environment also installed `pytest 9.1.1`, `setuptools 84.0.0`, and `wheel 0.48.0`. `uv lock --check` passed after the new declared direct dependency `cryptography>=44.0.0` was added for Ed25519 signature verification.

Direct and relevant transitive packages from the locked environment:

| Package | Version | Metadata license identifier/value | Principal installed license-text path and SHA-256 |
|---|---:|---|---|
| torch | 2.14.1+cpu | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | `torch-2.14.1.dist-info/licenses/LICENSE` — `0fc92425d130d9c87a5eb7427082e3afdbff6ef169bc678e15abba4c2d37c934` |
| numpy | 2.5.4 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | `numpy-2.5.4.dist-info/licenses/LICENSE.txt` — `a804dff0ead9fadc5293456410bcbfc32bf024be9c4513459663fb7b442d2341` |
| Pillow | 12.3.0 | `MIT-CMU` | `pillow-12.3.0.dist-info/licenses/LICENSE` — `4f7866a74802c6326f81faff59a56546b6aec2b10b91973e0e9308de95e79857` |
| cryptography | 50.0.2 | `Apache-2.0 OR BSD-3-Clause` | `cryptography-50.0.2.dist-info/licenses/LICENSE` — `3e0c7c091a948b82533ba98fd7cbb40432d6f1a9acbf85f5922d2f99a93ae6bb`; Apache text `aac73b3148f6d1d7111dbca32099f68d26c644c6813ae1e4f05f6579aa2663fe`; BSD text `602c4c7482de6479dd2e9793cda275e5e63d773dacd1eca689232ab7008fb4fb` |
| cffi (cryptography transitive) | 2.1.1 | MIT-0 | `cffi-2.1.1.dist-info/licenses/LICENSE` — `5ba24ddc57067f9249add644c3afc41a5d6dc37e23433ef759d95df370b0af63` |
| pycparser (cffi transitive) | 3.11 | BSD-3-Clause | `pycparser-3.11.dist-info/licenses/LICENSE` — `0a9e78156ae8b96caf1b4f11ac2df424b83de29736f9e757630548a6272fdf90` |
| filelock (Torch transitive) | 4.2.0 | MIT | `filelock-4.2.0.dist-info/licenses/LICENSE` — `608c89d5060ae9921adccf3236695bc654a9946e12323ef6c021dfa04e294d48` |
| fsspec (Torch transitive) | 2026.9.0 | BSD-3-Clause | `fsspec-2026.9.0.dist-info/licenses/LICENSE` — `2dc35496ce53a7307915c008a844aad53e772b49b34cdd00445067691407ec94` |
| Jinja2 (Torch transitive) | 3.1.6 | BSD License classifier; no SPDX expression | `jinja2-3.1.6.dist-info/licenses/LICENSE.txt` — `3b49dcee4105eb37bac10faf1be260408fe85d252b8e9df2e0979fc1e094437b` |
| MarkupSafe (Jinja2 transitive) | 3.0.4 | BSD-3-Clause | `markupsafe-3.0.4.dist-info/licenses/LICENSE.txt` — `4631ec0db5fd90a547e336817264c6798214338146f8ac94b4a57f96ee8c9ec4` |
| mpmath (SymPy transitive) | 1.3.0 | generic BSD metadata | `mpmath-1.3.0.dist-info/LICENSE` — `c26cae81da4508e5e249985777a33821f183223ebb74d7f8cfbf90fe7eef2fb7` |
| networkx (Torch transitive) | 3.7 | BSD-3-Clause | `networkx-3.7.dist-info/licenses/LICENSE.txt` — `6025f323ea29dc2f0ee0abd9523bf910186f55b0e148e6ebf646c55566278e2e` |
| sympy (Torch transitive) | 1.14.0 | generic BSD metadata | `sympy-1.14.0.dist-info/licenses/LICENSE` — `07a5e9819f727b4986ad2829c7a29a6320d42575f720eb24d71b7fef573a0286` |
| typing_extensions (Torch transitive) | 4.16.0 | PSF-2.0 | `typing_extensions-4.16.0.dist-info/licenses/LICENSE` — `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` |
| setuptools | 84.0.0 | MIT | `setuptools-84.0.0.dist-info/licenses/LICENSE` — `86da0f01aeae46348a3c3d465195dc1ceccde79f79e87769a64b8da04b2a4741` |
| wheel | 0.48.0 | MIT | `wheel-0.48.0.dist-info/licenses/LICENSE.txt` — `30c23618679108f3e8ea1d2a658c7ca417bdfc891c98ef1a89fa4ff0c9828654` |
| pytest | 9.1.1 | MIT | `pytest-9.1.1.dist-info/licenses/LICENSE` — `ca836a5f9ecca3b2f350230faa20a48fb8b145653b5568d784862df864706b9b` |
| pluggy (pytest transitive) | 1.7.0 | MIT | `pluggy-1.7.0.dist-info/licenses/LICENSE` — `d6b65e6c213a5d0b577911d34d6e5949b9f59d76c238c5071a2f3fc16cfb2606` |
| iniconfig (pytest transitive) | 2.3.1 | MIT | `iniconfig-2.3.1.dist-info/licenses/LICENSE` — `3409fa91f7ace557894632676656e32264fe5ef7581535725dc9a23774551bd4` |
| packaging (pytest transitive) | 26.3 | Apache-2.0 OR BSD-2-Clause | `packaging-26.3.dist-info/licenses/LICENSE` — `cad1ef5bd340d73e074ba614d26f7deaca5c7940c3d8c34852e65c4909686c48`; Apache text `0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594` |
| Pygments (pytest transitive) | 2.21.0 | BSD-2-Clause | `pygments-2.21.0.dist-info/licenses/LICENSE` — `a9d66f1d526df02e29dce73436d34e56e8632f46c275bbdffc70569e882f9f17` |
| colorama (pytest transitive) | 0.4.6 | BSD License classifier | `colorama-0.4.6.dist-info/licenses/LICENSE.txt` — `cac35c02686e5d04a5a7140bfb3b36e73aed496656e891102e428886d7930318` |

These license identifiers and texts are package-distribution evidence only, not a legal review. Torch and NumPy bundle many third-party notices; setuptools vendors libraries. Those notice collections were not exhaustively reviewed. The actual owner/legal dependency review remains outstanding.

### Deterministic generation and corpus version comparison

In the lock-synchronized environment, generated the full 112-frame corpus twice, each time into a different new temporary directory, with seed `20261018` and default configuration. The two copies had identical manifest bytes and matching image hashes, annotation hashes, frame IDs, split assignments, layout families, source revision fields, and runtime fields:

- Repository `HEAD`: `4cf794cb577cccf3a7785701eeabc877c0b51942`; clean-room generator worktree was dirty/untracked. The manifest also records the source hashes, which identify the actual renderer inputs:
  - `avcs/dataset/cleanroom_v1/generator.py`: `9eeccf7f3b56c0ed949619b83ad22c7d0c1d63fe45e40bdbc7c10c93588d01fb`
  - `avcs/dataset/cleanroom_v1/glyphs.py`: `881436e0dae57d0a1077cfa0976e205046ba496c96968cf30f8fb0dad1285e73`
  - `avcs/dataset/cleanroom_v1/dataset.py`: `28ac2a4d2ffb07359d719e30d71702d0f32238b710e4cccd7ecdead3a9115e21`
  - `avcs/dataset/cleanroom_v1/__init__.py`: `c9393f17fc87634c918055b1bb44f5a2d9256cfa62b04b4fe0e653433919107f`
- Copy A manifest SHA-256: `c1faf31e39cd5b3acd0a32a4c1b5b2e54e6267225bf4ba774715c7734e72931d`
- Copy B manifest SHA-256: `c1faf31e39cd5b3acd0a32a4c1b5b2e54e6267225bf4ba774715c7734e72931d`
- Both copies passed manifest validation: 112 records; 64 train, 16 validation, 16 held-out, and 16 negative-test; 112 distinct image hashes and 112 annotation hashes.

Compared with the preserved manifest generated under Pillow 13.0.0, the locked Pillow 12.3.0 outputs had **zero** image-hash differences, **zero** annotation-hash differences, **zero** split differences, and **zero** family differences for all 112 records. The runtime metadata differs (`runtime.pillow` and `asset_provenance.dependencies.generator.pillow`), so the locked environment's manifest digest is `c1faf...` instead of the preserved `abde...`. This demonstrates same pixels/annotations for the tested code/configuration across these two Pillow versions, but does not change the historical fact that the existing corpus was generated with Pillow 13.0.0, not the intended lock.

Do not replace the preserved manifest. If release provenance is to designate the lockfile environment as canonical, create a separately versioned reviewed corpus/manifest with its own owner attestation; do not rewrite the existing record simply because raster bytes happened to match.

### Signed-permission gate

Added `cryptography>=44.0.0` as an explicit project runtime dependency and `avcs/training/cleanroom_rights.py` as the verification path. The gate requires:

1. The human-readable rights record to identify the exact manifest digest, show `Rights decision: APPROVED`, and state `Unresolved rights blockers: NONE`.
2. A separate JSON permission payload with the exact dataset manifest SHA-256 and exactly four explicit decisions: `internal_training`, `commercial_inference`, `public_weight_redistribution`, and `source_dataset_redistribution`. Each must be `GRANTED` or `DENIED`; unresolved, blank, missing, malformed, or extra values fail. `internal_training` must be granted before training. A denial for another use does not authorize that use.
3. A valid Ed25519 signature covering the canonical JSON schema, exact dataset digest, complete four-permission mapping, and signer-key fingerprint.
4. A distinct trust configuration mapping the public key and its matching raw-key SHA-256 fingerprint to a signer `principal` and `authority_reference`. Its complete file hash must match `AVCS_RIGHTS_TRUST_CONFIG_SHA256`, loaded only from the protected trainer environment. The pin is not a command-line option and cannot be read from the approval payload itself.

The verifier rejects duplicate JSON keys, unknown/missing fields, malformed Base64 or key fingerprints, keys absent from the pinned trust set, trust-file pin mismatches, invalid signatures, mismatched dataset digests, and unresolved permissions. The separately pinned trust record must associate the matching fingerprint with a non-empty principal and authority-evidence reference. A valid signature proves that the private key associated by that pinned record signed these exact payload bytes. It does **not** prove the signer's legal identity/authority, truth of the claims, or completeness of the underlying rights evidence. No owner key, signature, approval, or trusted config was generated or supplied. The current decision remains pending and no training can proceed.

The verifier expects the trust JSON fields `schema`, `trusted_signers`, and for every signer `key_id`, `principal`, `authority_reference`, `ed25519_public_key_base64`, and `fingerprint_sha256`. The signer fingerprint is SHA-256 of the 32 raw public-key bytes. The approval signature is Base64 Ed25519 over UTF-8 JSON with keys sorted, compact separators `,` and `:`, ASCII escaping enabled, and the `signature_ed25519_base64` field omitted. Trust-config pinning uses SHA-256 of the exact file bytes. These serialization rules are technical verification rules, not a key-issuance or owner-approval process.

### Checks run

| Command/check | Result |
|---|---|
| `uv lock` after declaring `cryptography>=44.0.0` | Passed; resolved 119 packages |
| `uv lock --check` | Passed |
| `uv sync --locked --extra dev --no-install-project --python .venv\Scripts\python.exe` into a new temporary environment | Passed; Pillow 12.3.0 and wheel 0.48.0 installed without modifying existing `.venv` |
| `C:\Users\nandk\AppData\Local\Temp\avcs-m45-b845cd73-uv\Scripts\python.exe -m pytest -q tests\test_cleanroom_system1_v1.py tests\test_cleanroom_rights_hardening.py` | Passed: 28 passed, 1 skipped. The skipped test is the gated CPU training smoke test; rights approval remains pending. |
| Same-seed default corpus generation in two new directories under the locked environment | Passed; full manifest, image, annotation, split, and family comparisons identical |
| Locked corpus manifest validation | Passed; 112 records and all split/hash checks passed |
| Existing `.venv` targeted test collection (diagnostic only) | Failed at import because `cryptography` is intentionally absent from that unchanged environment. This confirms it cannot execute the newly declared rights verifier; the lock-synchronized test run above is the validation result. |

### Remaining blockers

- No authorized project owner has supplied approval or a verifiable trust key.
- The active old `.venv` remains out of sync; use the locked environment for relevant pipeline work.
- The canonical manifest preserves Pillow 13.0.0 generation provenance; do not claim it was created in the lock environment. Pixel-level reruns matched, but selecting a new canonical runtime would require a new reviewed manifest.
- Dependency license texts and third-party notices still require owner/legal review.
- Cryptographic verification does not validate the truth or legal sufficiency of any signed rights claim.

The only next action that could unblock the gate is for an authorized owner, after rights review, to provide a genuine signed permission payload plus the corresponding public key in an independently approved trust configuration and its out-of-band hash pin. This audit deliberately did not train or evaluate any model.
