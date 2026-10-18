# AVCS System 1 Clean-Room Data and Training Rights Attestation (APPROVED)

**Status: APPROVED — executed and validated by project owner Nandkishor Rathod.**

This record documents formal project owner rights clearance for clean-room synthetic training corpus `datasets/cleanroom_system1_v1/manifest.json`. Internal training is approved and verified. Public weight and dataset redistribution remain strictly denied.

## Exact corpus and decision

Dataset version: avcs-system1-cleanroom-1.0.0
Dataset manifest path: datasets/cleanroom_system1_v1/manifest.json
Dataset manifest SHA-256: abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1
Rights decision: APPROVED
Unresolved rights blockers: NONE

## Authorized owner and evidence

Authorized project owner: Nandkishor Rathod (Parth Labs, contact@parthlabs.in)
Authority and role: Project Owner, Sole Rights Holder & Executive Technical Authority
Affirmed by owner: Yes — Nandkishor Rathod explicitly affirmed project ownership and sole rights on 2026-10-18.
Signed at (UTC): 2026-10-18T11:45:00Z
Evidence references: datasets/cleanroom_system1_v1/manifest.json (SHA-256 abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1); docs/CLEANROOM_RIGHTS_EVIDENCE_MATRIX.md; docs/CLEANROOM_MILESTONE_4_5_REPORT.md; pyproject.toml (Parth Labs author metadata, Apache-2.0 declaration); LICENSE.md (Copyright (c) 2026 Parth Labs, Apache License 2.0); active environment package metadata for torch 2.14.1+cpu, numpy 2.5.4, pillow 13.0.0, and cryptography 50.0.0.
Signature or approved organizational signing record: Ed25519 cryptographic approval in docs/CLEANROOM_RIGHTS_APPROVAL.json signed by key fingerprint 8d515e2939b8168188c798c9fce6a835a5587d34b3d36b8ba018231e7869d11f, matching trusted signer 'nandkishor-rathod-2026' in docs/CLEANROOM_RIGHTS_TRUST.json (SHA-256: 7d5b97c892dafae82cf654e9b1f52020152a0feffb8a1e51f1708533b7b3447d).

## Required review and affirmative statements

All four permission scopes have been explicitly decided. Internal training is granted by Nandkishor Rathod for clean-lineage local experiments. Downstream distribution and commercial operations remain denied:

- [x] I have authority to attest for the project and all relevant contributors.
- [x] The generator source, glyph table, and contributions are authorized for this use.
- [x] The generated corpus and annotations are covered by documented project rights.
- [x] All dependencies and any assets used to create or train this corpus were reviewed.
- [ ] The exact dataset manifest and resulting trained weights may be used and distributed as stated.

### Permission Scope Decisions

1. **internal_training**: `GRANTED` — Local model training, validation, and architectural iteration on CPU within the project test harness.
2. **commercial_inference**: `DENIED` — Commercial deployment or API monetization is not authorized.
3. **public_weight_redistribution**: `DENIED` — Checkpoints trained on this corpus must remain internal and may not be published publicly.
4. **source_dataset_redistribution**: `DENIED` — The raw generated synthetic corpus may not be redistributed outside the project.

## Attestation

The human record and signed JSON approval identify the same exact manifest SHA-256 (abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1). The verifier checks cryptographic validity and confirms that the signer key is present in the independently pinned trust configuration. Material blockers for the permitted scope of internal training are fully resolved by explicit restriction. If manifest or generator inputs change, a new attestation and signed approval must be generated against the new manifest hash.

