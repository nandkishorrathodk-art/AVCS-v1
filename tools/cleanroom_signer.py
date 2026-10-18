"""AVCS Clean-Room Rights Signing & Trust Verification Utility.

Provides authorized project owners and security administrators with a local,
offline CLI to manage Ed25519 cryptographic keys, generate signed rights approvals,
and produce pinned trust configurations conforming to avcs.cleanroom.rights-approval.v1.

ZERO private keys or passwords should ever be transmitted to external servers, LLMs, or chats.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.training.cleanroom_rights import (
    REQUIRED_PERMISSIONS,
    RIGHTS_APPROVAL_SCHEMA,
    TRUST_CONFIG_SCHEMA,
    validate_signed_rights_approval,
)
from avcs.training.train_system1_clean_v1 import validate_rights_attestation


def canonical_json(value: Dict[str, Any]) -> bytes:
    """Produces sorted, compact UTF-8 JSON bytes for deterministic signing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def cmd_generate_key(args: argparse.Namespace) -> None:
    """Generates an Ed25519 private key and writes PEM private/public files."""
    private_path = Path(args.output_private)
    public_path = Path(args.output_public)

    if private_path.exists() and not args.force:
        raise FileExistsError(f"Private key file already exists: {private_path}. Use --force to overwrite.")

    key = Ed25519PrivateKey.generate()
    private_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)

    private_path.write_bytes(private_bytes)
    public_path.write_bytes(public_bytes)

    raw_pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    fingerprint = hashlib.sha256(raw_pub).hexdigest()

    print(f"[Key Generated] Private key saved to : {private_path}")
    print(f"[Key Generated] Public key saved to  : {public_path}")
    print(f"[Key Fingerprint (SHA-256)]          : {fingerprint}")
    print("\nWARNING: Keep your private key confidential. Never check it into git or share it.")


def cmd_create_trust_config(args: argparse.Namespace) -> None:
    """Creates a trusted public-key configuration file and prints its SHA-256 digest."""
    pub_path = Path(args.public_key)
    if not pub_path.is_file():
        raise FileNotFoundError(f"Public key file not found: {pub_path}")

    raw_pub_pem = pub_path.read_bytes()
    loaded_pub = serialization.load_pem_public_key(raw_pub_pem)
    if not isinstance(loaded_pub, Ed25519PublicKey):
        raise TypeError("Key must be an Ed25519 public key.")

    raw_pub_bytes = loaded_pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    b64_pub = base64.b64encode(raw_pub_bytes).decode("ascii")
    fingerprint = hashlib.sha256(raw_pub_bytes).hexdigest()

    trust_doc = {
        "schema": TRUST_CONFIG_SCHEMA,
        "trusted_signers": [
            {
                "key_id": args.key_id.strip(),
                "principal": args.principal.strip(),
                "authority_reference": args.authority_reference.strip(),
                "ed25519_public_key_base64": b64_pub,
                "fingerprint_sha256": fingerprint,
            }
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_bytes = json.dumps(trust_doc, indent=2).encode("utf-8") + b"\n"
    out_path.write_bytes(out_bytes)

    trust_sha256 = hashlib.sha256(out_bytes).hexdigest()
    print(f"[Trust Config Created] Saved to: {out_path}")
    print(f"[Trust Config SHA-256]           : {trust_sha256}")
    print(f"\nSet this environment variable before training:\n  $env:AVCS_RIGHTS_TRUST_CONFIG_SHA256 = \"{trust_sha256}\"")


def cmd_sign_approval(args: argparse.Namespace) -> None:
    """Signs an approval payload with an authorized Ed25519 private key."""
    priv_path = Path(args.private_key)
    if not priv_path.is_file():
        raise FileNotFoundError(f"Private key file not found: {priv_path}")

    raw_priv_pem = priv_path.read_bytes()
    signing_key = serialization.load_pem_private_key(raw_priv_pem, password=None)
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise TypeError("Key must be an Ed25519 private key.")

    raw_pub_bytes = signing_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    fingerprint = hashlib.sha256(raw_pub_bytes).hexdigest()

    permissions = {
        "internal_training": args.internal_training,
        "commercial_inference": args.commercial_inference,
        "public_weight_redistribution": args.public_weight_redistribution,
        "source_dataset_redistribution": args.source_dataset_redistribution,
    }

    manifest_hash = args.manifest_hash.strip().lower()
    if len(manifest_hash) != 64:
        raise ValueError("manifest-hash must be 64 hexadecimal characters.")

    approval_payload = {
        "schema": RIGHTS_APPROVAL_SCHEMA,
        "dataset_manifest_sha256": manifest_hash,
        "permissions": permissions,
        "signer_key_fingerprint_sha256": fingerprint,
    }

    # Canonicalize and sign
    to_sign_bytes = canonical_json(approval_payload)
    sig_raw = signing_key.sign(to_sign_bytes)
    sig_b64 = base64.b64encode(sig_raw).decode("ascii")

    full_approval = {
        **approval_payload,
        "signature_ed25519_base64": sig_b64,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_bytes = json.dumps(full_approval, indent=2).encode("utf-8") + b"\n"
    out_path.write_bytes(out_bytes)

    print(f"[Approval Signed] Output written to: {out_path}")
    print(f"[Manifest Bound]  : {manifest_hash}")
    print(f"[Signer Fingerprint]: {fingerprint}")
    print(f"[Decisions]       : {permissions}")


def cmd_verify_gate(args: argparse.Namespace) -> None:
    """Verifies the complete human attestation and cryptographic approval gate."""
    manifest_hash = args.manifest_hash.strip().lower()
    attestation_path = Path(args.attestation)
    approval_path = Path(args.approval)
    trust_path = Path(args.trust_config)

    trust_bytes = trust_path.read_bytes()
    actual_trust_hash = hashlib.sha256(trust_bytes).hexdigest()
    print(f"[Trust File SHA-256] : {actual_trust_hash}")

    expected_trust_hash = args.expected_trust_hash or os.environ.get("AVCS_RIGHTS_TRUST_CONFIG_SHA256")
    if not expected_trust_hash:
        print("Note: No expected trust hash pin supplied via --expected-trust-hash or $env:AVCS_RIGHTS_TRUST_CONFIG_SHA256.")
        print(f"Using actual file hash for verification test: {actual_trust_hash}")
        expected_trust_hash = actual_trust_hash

    result = validate_rights_attestation(
        attestation_path=attestation_path,
        manifest_sha256=manifest_hash,
        approval_path=approval_path,
        trust_config_path=trust_path,
        expected_trust_config_sha256=expected_trust_hash,
    )

    print("\n" + "=" * 60)
    print("   AVCS CLEAN-ROOM RIGHTS GATE VALIDATION RESULT")
    print("=" * 60)
    print(f"Signature Verified         : {result['signature_verified']}")
    print(f"Dataset Manifest SHA-256   : {result['dataset_manifest_sha256']}")
    print(f"Trusted Signer Key ID      : {result['trusted_signer']['key_id']}")
    print(f"Trusted Principal          : {result['trusted_signer']['principal']}")
    print(f"Authority Reference        : {result['trusted_signer']['authority_reference']}")
    print(f"Signer Fingerprint         : {result['signer_key_fingerprint_sha256']}")
    print(f"Permissions Decided        : {result['permissions']}")
    print("=" * 60)
    print("GATE STATUS: PASSED (CRYPTOGRAPHICALLY VERIFIED)")


def main() -> None:
    parser = argparse.ArgumentParser(description="AVCS Clean-Room Rights Signing & Gate Verification Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. generate-key
    p_gen = subparsers.add_parser("generate-key", help="Generate Ed25519 private and public key pair.")
    p_gen.add_argument("--output-private", type=str, default="keys/cleanroom_ed25519.pem")
    p_gen.add_argument("--output-public", type=str, default="keys/cleanroom_ed25519.pub.pem")
    p_gen.add_argument("--force", action="store_true", help="Overwrite existing key files.")
    p_gen.set_defaults(func=cmd_generate_key)

    # 2. create-trust-config
    p_trust = subparsers.add_parser("create-trust-config", help="Create trusted public-key configuration JSON.")
    p_trust.add_argument("--public-key", type=str, required=True, help="Path to PEM public key.")
    p_trust.add_argument("--key-id", type=str, required=True, help="Unique identifier for this key.")
    p_trust.add_argument("--principal", type=str, required=True, help="Name/identity of authorized signer.")
    p_trust.add_argument("--authority-reference", type=str, required=True, help="Legal/organizational authority doc.")
    p_trust.add_argument("--output", type=str, default="docs/CLEANROOM_RIGHTS_TRUST.json")
    p_trust.set_defaults(func=cmd_create_trust_config)

    # 3. sign-approval
    p_sign = subparsers.add_parser("sign-approval", help="Sign approval JSON with Ed25519 private key.")
    p_sign.add_argument("--private-key", type=str, required=True, help="Path to PEM private key.")
    p_sign.add_argument("--manifest-hash", type=str, default="abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1")
    p_sign.add_argument("--internal-training", choices=["GRANTED", "DENIED"], default="GRANTED")
    p_sign.add_argument("--commercial-inference", choices=["GRANTED", "DENIED"], default="DENIED")
    p_sign.add_argument("--public-weight-redistribution", choices=["GRANTED", "DENIED"], default="DENIED")
    p_sign.add_argument("--source-dataset-redistribution", choices=["GRANTED", "DENIED"], default="DENIED")
    p_sign.add_argument("--output", type=str, default="docs/CLEANROOM_RIGHTS_APPROVAL.json")
    p_sign.set_defaults(func=cmd_sign_approval)

    # 4. verify-gate
    p_ver = subparsers.add_parser("verify-gate", help="Validate rights attestation, approval, and trust config.")
    p_ver.add_argument("--manifest-hash", type=str, default="abde4338a1fa4c30d1c349036efa392f144c457150fb1c393ac2b0081f9669d1")
    p_ver.add_argument("--attestation", type=str, default="docs/CLEANROOM_DATA_RIGHTS_ATTESTATION.md")
    p_ver.add_argument("--approval", type=str, default="docs/CLEANROOM_RIGHTS_APPROVAL.json")
    p_ver.add_argument("--trust-config", type=str, default="docs/CLEANROOM_RIGHTS_TRUST.json")
    p_ver.add_argument("--expected-trust-hash", type=str, default=None)
    p_ver.set_defaults(func=cmd_verify_gate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
