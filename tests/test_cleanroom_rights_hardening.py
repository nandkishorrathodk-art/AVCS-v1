from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from avcs.dataset.cleanroom_v1.generator import generate_dataset
from avcs.dataset.cleanroom_v1.generator import _sha256
from avcs.training.cleanroom_environment import compare_dependency_versions
from avcs.training.cleanroom_rights import (
    REQUIRED_PERMISSIONS,
    RIGHTS_APPROVAL_SCHEMA,
    TRUST_CONFIG_SCHEMA,
    RightsApprovalError,
    validate_signed_rights_approval,
)
from avcs.training.train_system1_clean_v1 import (
    DEFAULT_ATTESTATION,
    RightsGateError,
    assert_safe_output_path,
    validate_rights_attestation,
)

SMALL_COUNTS = {
    "train": 8,
    "validation": 2,
    "test_heldout": 4,
    "test_negative": 2,
}
MANIFEST_HASH = "a" * 64


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _approval_files(
    tmp_path: Path,
    *,
    permissions: dict[str, str] | None = None,
    signing_key: Ed25519PrivateKey | None = None,
    trusted_key: Ed25519PrivateKey | None = None,
    manifest_hash: str = MANIFEST_HASH,
) -> tuple[Path, Path, str, dict]:
    signing_key = signing_key or Ed25519PrivateKey.generate()
    trusted_key = trusted_key or signing_key
    public_key = signing_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    trusted_public_key = trusted_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    signer_fingerprint = hashlib.sha256(public_key).hexdigest()
    trusted_fingerprint = hashlib.sha256(trusted_public_key).hexdigest()
    trust = {
        "schema": TRUST_CONFIG_SCHEMA,
        "trusted_signers": [
            {
                "key_id": "test-only-synthetic-key",
                "principal": "Synthetic test identity, not a real project owner",
                "authority_reference": "Synthetic test-only trust fixture",
                "ed25519_public_key_base64": base64.b64encode(trusted_public_key).decode("ascii"),
                "fingerprint_sha256": trusted_fingerprint,
            }
        ],
    }
    trust_path = tmp_path / "trust.json"
    _write_json(trust_path, trust)
    trust_hash = hashlib.sha256(trust_path.read_bytes()).hexdigest()

    approval = {
        "schema": RIGHTS_APPROVAL_SCHEMA,
        "dataset_manifest_sha256": manifest_hash,
        "permissions": permissions
        or {
            "internal_training": "GRANTED",
            "commercial_inference": "DENIED",
            "public_weight_redistribution": "DENIED",
            "source_dataset_redistribution": "DENIED",
        },
        "signer_key_fingerprint_sha256": signer_fingerprint,
        "signature_ed25519_base64": "",
    }
    signature = signing_key.sign(
        _canonical_json({key: value for key, value in approval.items() if key != "signature_ed25519_base64"})
    )
    approval["signature_ed25519_base64"] = base64.b64encode(signature).decode("ascii")
    approval_path = tmp_path / "approval.json"
    _write_json(approval_path, approval)
    return approval_path, trust_path, trust_hash, approval


def test_signed_approval_binds_manifest_and_each_permission(tmp_path: Path) -> None:
    approval_path, trust_path, trust_hash, approval = _approval_files(tmp_path)
    result = validate_signed_rights_approval(
        approval_path,
        MANIFEST_HASH,
        trust_path,
        trust_hash,
    )
    assert result["signature_verified"] is True
    assert result["legal_claims_independently_verified"] is False
    assert set(result["permissions"]) == set(REQUIRED_PERMISSIONS)
    assert result["permissions"]["internal_training"] == "GRANTED"
    assert result["permissions"]["commercial_inference"] == "DENIED"
    assert result["trusted_signer"]["principal"].startswith("Synthetic test identity")
    assert approval["dataset_manifest_sha256"] == MANIFEST_HASH


def test_full_gate_accepts_only_matching_pinned_signed_permission_payload(tmp_path: Path) -> None:
    approval_path, trust_path, trust_hash, _ = _approval_files(tmp_path)
    attestation_path = tmp_path / "approved-human-record.md"
    attestation_path.write_text(
        "\n".join(
            (
                f"Dataset manifest SHA-256: {MANIFEST_HASH}",
                "Rights decision: APPROVED",
                "Unresolved rights blockers: NONE",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    verified = validate_rights_attestation(
        attestation_path,
        MANIFEST_HASH,
        approval_path=approval_path,
        trust_config_path=trust_path,
        expected_trust_config_sha256=trust_hash,
    )
    assert verified["signature_verified"] is True
    assert verified["permissions"]["internal_training"] == "GRANTED"
    assert verified["trusted_config_sha256"] == trust_hash
    assert verified["legal_claims_independently_verified"] is False


def test_signed_approval_rejects_missing_or_mismatched_manifest_hash(tmp_path: Path) -> None:
    approval_path, trust_path, trust_hash, _ = _approval_files(tmp_path)
    with pytest.raises(RightsApprovalError, match="different dataset manifest"):
        validate_signed_rights_approval(approval_path, "b" * 64, trust_path, trust_hash)
    with pytest.raises(RightsApprovalError, match="64 lowercase hexadecimal"):
        validate_signed_rights_approval(approval_path, "", trust_path, trust_hash)


def test_signed_approval_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _, trust_path, trust_hash, _ = _approval_files(tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        '{"schema":"avcs.cleanroom.rights-approval.v1","schema":"other"}',
        encoding="utf-8",
    )
    with pytest.raises(RightsApprovalError, match="unambiguous UTF-8 JSON"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, trust_hash)


def test_signed_approval_requires_independently_pinned_trust_configuration(tmp_path: Path) -> None:
    approval_path, trust_path, trust_hash, _ = _approval_files(tmp_path)
    with pytest.raises(RightsApprovalError, match="independently supplied"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, None)
    with pytest.raises(RightsApprovalError, match="independent SHA-256 pin"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, "0" * 64)


@pytest.mark.parametrize(
    ("signature", "message"),
    [
        ("", "no Ed25519 signature"),
        ("not-base64!", "not valid Base64"),
    ],
)
def test_signed_approval_rejects_missing_and_malformed_signatures(
    tmp_path: Path,
    signature: str,
    message: str,
) -> None:
    approval_path, trust_path, trust_hash, approval = _approval_files(tmp_path)
    approval["signature_ed25519_base64"] = signature
    _write_json(approval_path, approval)
    with pytest.raises(RightsApprovalError, match=message):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, trust_hash)


def test_signed_approval_rejects_invalid_signature(tmp_path: Path) -> None:
    approval_path, trust_path, trust_hash, approval = _approval_files(tmp_path)
    signature = bytearray(base64.b64decode(approval["signature_ed25519_base64"]))
    signature[0] ^= 0x01
    approval["signature_ed25519_base64"] = base64.b64encode(signature).decode("ascii")
    _write_json(approval_path, approval)
    with pytest.raises(RightsApprovalError, match="signature is invalid"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, trust_hash)


def test_signed_approval_signature_covers_permission_set(tmp_path: Path) -> None:
    approval_path, trust_path, trust_hash, approval = _approval_files(tmp_path)
    approval["permissions"]["commercial_inference"] = "GRANTED"
    _write_json(approval_path, approval)
    with pytest.raises(RightsApprovalError, match="signature is invalid"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, trust_hash)


def test_signed_approval_rejects_untrusted_signing_key(tmp_path: Path) -> None:
    signing_key = Ed25519PrivateKey.generate()
    trust_key = Ed25519PrivateKey.generate()
    approval_path, trust_path, trust_hash, _ = _approval_files(
        tmp_path,
        signing_key=signing_key,
        trusted_key=trust_key,
    )
    with pytest.raises(RightsApprovalError, match="absent from the pinned trust configuration"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, trust_hash)


@pytest.mark.parametrize(
    "permission_update",
    [
        {"commercial_inference": "UNRESOLVED"},
        {"public_weight_redistribution": ""},
        {"source_dataset_redistribution": "MAYBE"},
    ],
)
def test_signed_approval_rejects_unresolved_permission_scopes(
    tmp_path: Path,
    permission_update: dict[str, str],
) -> None:
    permissions = {
        "internal_training": "GRANTED",
        "commercial_inference": "DENIED",
        "public_weight_redistribution": "DENIED",
        "source_dataset_redistribution": "DENIED",
        **permission_update,
    }
    approval_path, trust_path, trust_hash, _ = _approval_files(tmp_path, permissions=permissions)
    with pytest.raises(RightsApprovalError, match="unresolved or has an invalid decision"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, trust_hash)


def test_signed_approval_rejects_missing_permission_or_denied_training(tmp_path: Path) -> None:
    permissions = {
        "internal_training": "GRANTED",
        "commercial_inference": "DENIED",
        "public_weight_redistribution": "DENIED",
    }
    approval_path, trust_path, trust_hash, _ = _approval_files(tmp_path, permissions=permissions)
    with pytest.raises(RightsApprovalError, match="each required permission"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, trust_hash)

    permissions["source_dataset_redistribution"] = "DENIED"
    permissions["internal_training"] = "DENIED"
    approval_path, trust_path, trust_hash, _ = _approval_files(tmp_path, permissions=permissions)
    with pytest.raises(RightsApprovalError, match="Internal training permission is not granted"):
        validate_signed_rights_approval(approval_path, MANIFEST_HASH, trust_path, trust_hash)


def test_pending_human_attestation_and_unsigned_template_never_open_gate(tmp_path: Path) -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "datasets" / "cleanroom_system1_v1" / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    approval_path, trust_path, trust_hash, _ = _approval_files(
        tmp_path,
        manifest_hash=manifest_hash,
        permissions={
            "internal_training": "UNRESOLVED",
            "commercial_inference": "UNRESOLVED",
            "public_weight_redistribution": "UNRESOLVED",
            "source_dataset_redistribution": "UNRESOLVED",
        },
    )
    with pytest.raises(RightsGateError, match="not explicitly APPROVED"):
        validate_rights_attestation(
            DEFAULT_ATTESTATION,
            manifest_hash,
            approval_path=approval_path,
            trust_config_path=trust_path,
            expected_trust_config_sha256=trust_hash,
        )
    with pytest.raises(RightsGateError, match="not explicitly APPROVED"):
        validate_rights_attestation(DEFAULT_ATTESTATION, manifest_hash)


def test_two_fresh_output_directories_have_identical_synthetic_corpus(tmp_path: Path) -> None:
    first = tmp_path / "corpus-a"
    second = tmp_path / "corpus-b"
    manifest_a = generate_dataset(first, seed=99481, counts=SMALL_COUNTS)
    manifest_b = generate_dataset(second, seed=99481, counts=SMALL_COUNTS)
    assert manifest_a == manifest_b
    rows_a = {row["frame_id"]: row for row in manifest_a["files"]}
    rows_b = {row["frame_id"]: row for row in manifest_b["files"]}
    assert rows_a.keys() == rows_b.keys()
    for frame_id, row_a in rows_a.items():
        row_b = rows_b[frame_id]
        assert row_a["split"] == row_b["split"]
        assert row_a["layout_family"] == row_b["layout_family"]
        assert row_a["image"]["sha256"] == row_b["image"]["sha256"]
        assert row_a["annotation"]["sha256"] == row_b["annotation"]["sha256"]
    assert hashlib.sha256((first / "manifest.json").read_bytes()).digest() == hashlib.sha256(
        (second / "manifest.json").read_bytes()
    ).digest()


def test_dependency_version_comparison_detects_mismatch_and_missing_package() -> None:
    mismatches = compare_dependency_versions(
        {"Pillow": "12.3.0", "wheel": "0.48.0", "torch": "2.14.1"},
        {"pillow": "13.0.0", "torch": "2.14.1"},
    )
    assert mismatches == {
        "Pillow": {"expected": "12.3.0", "observed": "13.0.0"},
        "wheel": {"expected": "0.48.0", "observed": None},
    }


def test_baseline_hash_is_unchanged_and_output_guards_remain_active(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    baseline = repository / "runs" / "weights" / "candidates" / "avcs_system1_v1.pt"
    original_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="frozen baseline"):
        assert_safe_output_path(baseline)
    clean_output = tmp_path / "candidate.pt"
    clean_output.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        assert_safe_output_path(clean_output)
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == original_hash
    assert original_hash == "c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4"
