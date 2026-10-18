"""Cryptographic verification for clean-room training rights approvals."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

RIGHTS_APPROVAL_SCHEMA = "avcs.cleanroom.rights-approval.v1"
TRUST_CONFIG_SCHEMA = "avcs.cleanroom.rights-trust.v1"
REQUIRED_PERMISSIONS = (
    "internal_training",
    "commercial_inference",
    "public_weight_redistribution",
    "source_dataset_redistribution",
)
_APPROVAL_FIELDS = {
    "schema",
    "dataset_manifest_sha256",
    "permissions",
    "signer_key_fingerprint_sha256",
    "signature_ed25519_base64",
}


class RightsApprovalError(RuntimeError):
    """Raised when an approval is missing, untrusted, malformed, or invalid."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _load_json(path: str | Path, description: str) -> tuple[dict[str, Any], bytes]:
    file_path = Path(path)
    if not file_path.is_file():
        raise RightsApprovalError(f"{description} is missing: {file_path}")
    raw = file_path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise RightsApprovalError(f"{description} is not valid, unambiguous UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise RightsApprovalError(f"{description} must be a JSON object.")
    return value, raw


def validate_signed_rights_approval(
    approval_path: str | Path,
    manifest_sha256: str,
    trust_config_path: str | Path,
    expected_trust_config_sha256: str | None,
) -> dict[str, Any]:
    """Verify an Ed25519 approval against an independently pinned trust file."""
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise RightsApprovalError("Expected manifest SHA-256 must be 64 lowercase hexadecimal characters.")
    if not expected_trust_config_sha256 or not re.fullmatch(
        r"[0-9a-f]{64}", expected_trust_config_sha256
    ):
        raise RightsApprovalError(
            "An independently supplied trusted-config SHA-256 pin is required."
        )

    trust, trust_raw = _load_json(trust_config_path, "Trusted key configuration")
    actual_trust_hash = hashlib.sha256(trust_raw).hexdigest()
    if actual_trust_hash != expected_trust_config_sha256:
        raise RightsApprovalError("Trusted key configuration does not match the independent SHA-256 pin.")
    if set(trust) != {"schema", "trusted_signers"} or trust.get("schema") != TRUST_CONFIG_SCHEMA:
        raise RightsApprovalError("Trusted key configuration has an unsupported schema or fields.")
    signers = trust.get("trusted_signers")
    if not isinstance(signers, list) or not signers:
        raise RightsApprovalError("Trusted key configuration contains no trusted signers.")

    approval, _ = _load_json(approval_path, "Signed rights approval")
    if set(approval) != _APPROVAL_FIELDS:
        raise RightsApprovalError("Signed rights approval has missing or unexpected fields.")
    if approval.get("schema") != RIGHTS_APPROVAL_SCHEMA:
        raise RightsApprovalError("Signed rights approval has an unsupported schema.")
    if approval.get("dataset_manifest_sha256") != manifest_sha256:
        raise RightsApprovalError("Signed rights approval targets a different dataset manifest.")
    permissions = approval.get("permissions")
    if not isinstance(permissions, dict) or set(permissions) != set(REQUIRED_PERMISSIONS):
        raise RightsApprovalError("Approval must decide each required permission exactly once.")
    for permission in REQUIRED_PERMISSIONS:
        decision = permissions[permission]
        if not isinstance(decision, str) or decision not in {"GRANTED", "DENIED"}:
            raise RightsApprovalError(
                f"Permission {permission!r} is unresolved or has an invalid decision."
            )
    if permissions["internal_training"] != "GRANTED":
        raise RightsApprovalError("Internal training permission is not granted.")

    fingerprint = approval.get("signer_key_fingerprint_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise RightsApprovalError("Approval signer fingerprint is malformed.")
    matched_public_key: bytes | None = None
    matched_signer: dict[str, str] | None = None
    seen_fingerprints: set[str] = set()
    for signer in signers:
        if not isinstance(signer, dict) or set(signer) != {
            "key_id",
            "principal",
            "authority_reference",
            "ed25519_public_key_base64",
            "fingerprint_sha256",
        }:
            raise RightsApprovalError("Trusted signer entry has missing or unexpected fields.")
        key_id = signer.get("key_id")
        principal = signer.get("principal")
        authority_reference = signer.get("authority_reference")
        configured_fingerprint = signer.get("fingerprint_sha256")
        encoded_key = signer.get("ed25519_public_key_base64")
        if (
            not isinstance(key_id, str)
            or not key_id.strip()
            or not isinstance(principal, str)
            or not principal.strip()
            or not isinstance(authority_reference, str)
            or not authority_reference.strip()
            or not isinstance(configured_fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", configured_fingerprint)
            or configured_fingerprint in seen_fingerprints
            or not isinstance(encoded_key, str)
        ):
            raise RightsApprovalError("Trusted signer identity or fingerprint is malformed.")
        seen_fingerprints.add(configured_fingerprint)
        try:
            public_key = base64.b64decode(encoded_key, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise RightsApprovalError("Trusted Ed25519 public key is not valid Base64.") from exc
        if len(public_key) != 32:
            raise RightsApprovalError("Trusted Ed25519 public key must be exactly 32 bytes.")
        actual_fingerprint = hashlib.sha256(public_key).hexdigest()
        if actual_fingerprint != configured_fingerprint:
            raise RightsApprovalError("Trusted key fingerprint does not match its public key.")
        if configured_fingerprint == fingerprint:
            matched_public_key = public_key
            matched_signer = {
                "key_id": key_id,
                "principal": principal,
                "authority_reference": authority_reference,
            }

    if matched_public_key is None or matched_signer is None:
        raise RightsApprovalError("Approval was signed by a key absent from the pinned trust configuration.")
    encoded_signature = approval.get("signature_ed25519_base64")
    if not isinstance(encoded_signature, str) or not encoded_signature:
        raise RightsApprovalError("Signed rights approval has no Ed25519 signature.")
    try:
        signature = base64.b64decode(encoded_signature, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RightsApprovalError("Approval signature is not valid Base64.") from exc
    if len(signature) != 64:
        raise RightsApprovalError("Ed25519 signature must be exactly 64 bytes.")

    signed_payload = {key: value for key, value in approval.items() if key != "signature_ed25519_base64"}
    try:
        Ed25519PublicKey.from_public_bytes(matched_public_key).verify(
            signature,
            _canonical_json(signed_payload),
        )
    except (InvalidSignature, ValueError) as exc:
        raise RightsApprovalError("Ed25519 approval signature is invalid.") from exc

    return {
        "approval_sha256": hashlib.sha256(_canonical_json(approval)).hexdigest(),
        "dataset_manifest_sha256": manifest_sha256,
        "permissions": permissions,
        "signer_key_fingerprint_sha256": fingerprint,
        "trusted_signer": matched_signer,
        "trusted_config_sha256": actual_trust_hash,
        "signature_verified": True,
        "legal_claims_independently_verified": False,
    }
