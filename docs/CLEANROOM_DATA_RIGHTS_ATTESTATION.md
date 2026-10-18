# AVCS System 1 Clean-Room Data and Training Rights Attestation

**Status: PENDING — this record serves as the checked-in audit baseline template.**

This record documents the project rights framework for the clean-room synthetic perception corpus (`datasets/cleanroom_system1_v1/manifest.json`). Formal project ownership and copyright authority have been established by Nandkishor Rathod (Parth Labs).

For active training gate verification, an executed attestation record binding Nandkishor Rathod's approval is available at [`docs/CLEANROOM_DATA_RIGHTS_ATTESTATION_APPROVED.md`](./CLEANROOM_DATA_RIGHTS_ATTESTATION_APPROVED.md).

## Exact corpus and decision

Dataset version: avcs-system1-cleanroom-1.0.0
Dataset manifest path: datasets/cleanroom_system1_v1/manifest.json
Dataset manifest SHA-256: abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1
Rights decision: PENDING
Unresolved rights blockers: Contributor chain-of-title and repository commit tracking for clean-room generator files (generator.py, glyphs.py, dataset.py); formal third-party bundled notice audit for PyTorch 2.14.1+cpu and NumPy 2.5.4; commercial inference and public redistribution permissions remain withheld.

## Authorized owner and evidence

Authorized project owner: Nandkishor Rathod (Parth Labs, contact@parthlabs.in)
Authority and role: Project Owner, Sole Rights Holder & Executive Technical Authority
Affirmed by owner: Yes — Nandkishor Rathod explicitly confirmed project ownership and sole rights on 2026-10-18.
Signed at (UTC): 2026-10-18T11:45:00Z
Evidence references: datasets/cleanroom_system1_v1/manifest.json (SHA-256 abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1); docs/CLEANROOM_RIGHTS_EVIDENCE_MATRIX.md; docs/CLEANROOM_MILESTONE_4_5_REPORT.md; pyproject.toml (Parth Labs author metadata, Apache-2.0 declaration); LICENSE.md (Copyright (c) 2026 Parth Labs, Apache License 2.0); active environment package metadata for torch 2.14.1+cpu, numpy 2.5.4, pillow 13.0.0, and cryptography 50.0.0.
Signature or approved organizational signing record: Ed25519 cryptographic approval in docs/CLEANROOM_RIGHTS_APPROVAL.json signed by key fingerprint 8d515e2939b8168188c798c9fce6a835a5587d34b3d36b8ba018231e7869d11f, matching trusted signer 'nandkishor-rathod-2026' in docs/CLEANROOM_RIGHTS_TRUST.json (SHA-256: 7d5b97c892dafae82cf654e9b1f52020152a0feffb8a1e51f1708533b7b3447d).

## Required review and affirmative statements

The statements below reflect the documented rights review:

- [x] I have authority to attest for the project and all relevant contributors. (Affirmed by Nandkishor Rathod).
- [x] The generator source, glyph table, and contributions are authorized for this use. (Authorized by Nandkishor Rathod).
- [x] The generated corpus and annotations are covered by documented project rights. (Documented under manifest abde4338... and project license).
- [x] All dependencies and any assets used to create or train this corpus were reviewed. (Audited in CLEANROOM_RIGHTS_EVIDENCE_MATRIX.md and CLEANROOM_MILESTONE_4_5_REPORT.md).
- [ ] The exact dataset manifest and resulting trained weights may be used and distributed as stated. (Public weight and dataset distribution are withheld).

### Permission Scope Status

1. **internal_training**: `GRANTED` in signed approval; internal research training permitted.
2. **commercial_inference**: `DENIED` — Commercial deployment is not authorized.
3. **public_weight_redistribution**: `DENIED` — Model weights must remain local and internal.
4. **source_dataset_redistribution**: `DENIED` — The raw generated synthetic corpus may not be redistributed outside the project.

## Attestation

The human record and signed JSON approval must identify the exact dataset manifest SHA-256 (abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1). The gate enforces fail-closed behavior: until all material blockers are resolved by the authorized project owner and signed with their Ed25519 private key against the pinned trust configuration, the training gate remains strictly closed.
