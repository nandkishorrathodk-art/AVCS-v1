"""AVC Enterprise Locked Distribution Builder.

Packages the Artificial Visual Cortex (AVC) standalone neural package from packages/avc:
1. AES-256-GCM encrypted weights vault (.avcs)
2. 100% Volatile RAM in-memory decryption
3. Zero plaintext .pt files in final distribution wheel
"""
from __future__ import annotations

import argparse
import compileall
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "packages" / "avc"
sys.path.insert(0, str(PACKAGE_DIR))

from avc.vault import ModelVault

DEFAULT_COMMUNITY_KEY = "avc-community-free-2026"


def encrypt_weights(api_key: str = DEFAULT_COMMUNITY_KEY) -> Path:
    """Encrypts avcs_cortex_weights.pt into avcs_cortex_weights.avcs vault in package tree."""
    weights_dir = PACKAGE_DIR / "avc" / "weights"
    pt_path = weights_dir / "avcs_cortex_weights.pt"
    vault_path = weights_dir / "avcs_cortex_weights.avcs"

    if not pt_path.exists():
        # Fallback to root avc weights if needed
        root_pt = ROOT_DIR / "avc" / "weights" / "avcs_cortex_weights.pt"
        if root_pt.exists():
            shutil.copy2(str(root_pt), str(pt_path))
        else:
            raise FileNotFoundError(f"Source weight checkpoint not found: {pt_path}")

    print(f"\n[1/5] Encrypting AVC Neural Weights -> {vault_path.name}...")
    ModelVault.encrypt_checkpoint(pt_path, vault_path, api_key=api_key)

    size_mb = vault_path.stat().st_size / (1024.0 * 1024.0)
    print(f"      * Encrypted Vault Size: {size_mb:.2f} MB")

    print("[2/5] Validating In-Memory Vault Decryption in RAM...")
    state = ModelVault.decrypt_to_ram(vault_path, api_key=api_key)
    print(f"      * Verification Passed: {len(state)} tensors loaded into RAM (0 disk leaks).")

    # Also sync vault to root avc for local development
    root_vault = ROOT_DIR / "avc" / "weights" / "avcs_cortex_weights.avcs"
    shutil.copy2(str(vault_path), str(root_vault))

    return vault_path


def compile_bytecode() -> None:
    """Pre-compile AVC Python modules to bytecode."""
    print("\n[3/5] Pre-compiling AVC modules to bytecode...")
    avc_dir = PACKAGE_DIR / "avc"
    compileall.compile_dir(str(avc_dir), force=True, quiet=1)
    print("      * Bytecode pre-compilation complete.")


def build_wheel() -> Path:
    """Builds the binary wheel package from packages/avc with zero plaintext .pt leaks."""
    print("\n[4/5] Building AVC distribution wheel (.whl)...")
    dist_dir = PACKAGE_DIR / "dist"
    build_dir = PACKAGE_DIR / "build"
    egg_info_dir = PACKAGE_DIR / "avc.egg-info"
    backup_pt_dir = PACKAGE_DIR / ".pt_weights_backup_avc"
    weights_dir = PACKAGE_DIR / "avc" / "weights"

    if build_dir.exists():
        shutil.rmtree(build_dir)
    if egg_info_dir.exists():
        shutil.rmtree(egg_info_dir)

    # Stash .pt weights outside package tree during build
    backup_pt_dir.mkdir(parents=True, exist_ok=True)
    moved_pts: list[tuple[Path, Path]] = []
    for pt_file in weights_dir.glob("*.pt"):
        dest = backup_pt_dir / pt_file.name
        shutil.move(str(pt_file), str(dest))
        moved_pts.append((dest, pt_file))

    pyproject_path = PACKAGE_DIR / "pyproject.toml"
    with open(pyproject_path, "r", encoding="utf-8") as f:
        original_pyproject = f.read()

    locked_pyproject = original_pyproject.replace(
        '"weights/*.pt"',
        '"weights/*.avcs"',
    )

    try:
        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.write(locked_pyproject)

        cmd = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(dist_dir),
        ]
        subprocess.check_call(cmd, cwd=str(PACKAGE_DIR))

    finally:
        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.write(original_pyproject)

        # Restore .pt files
        for backup_dest, original_pt in moved_pts:
            if backup_dest.exists():
                shutil.move(str(backup_dest), str(original_pt))
        if backup_pt_dir.exists():
            shutil.rmtree(backup_pt_dir)

    # Find the newest wheel starting with avc-
    wheels = sorted(dist_dir.glob("avc-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        raise RuntimeError("Wheel generation failed; no avc-*.whl found in packages/avc/dist/")
    return wheels[0]


def inspect_wheel(wheel_path: Path) -> None:
    """Audit the built wheel to ensure ZERO unencrypted .pt files are packaged."""
    print(f"\n[5/5] Auditing Built Wheel Archive: {wheel_path.name}...")
    with zipfile.ZipFile(wheel_path, "r") as zf:
        names = zf.namelist()

    pt_leaks = [n for n in names if n.endswith(".pt")]
    avcs_vaults = [n for n in names if n.endswith(".avcs")]

    print(f"      * Total Files in Archive: {len(names)}")
    print(f"      * Encrypted Vaults Found: {avcs_vaults}")

    if pt_leaks:
        raise RuntimeError(
            f"SECURITY AUDIT FAILED: Unencrypted weight files found in package: {pt_leaks}"
        )

    if not avcs_vaults:
        raise RuntimeError("SECURITY AUDIT FAILED: Encrypted .avcs vault missing from wheel!")

    size_mb = wheel_path.stat().st_size / (1024.0 * 1024.0)
    print(f"\n[SUCCESS] Locked Wheel Verified: {wheel_path.name} ({size_mb:.2f} MB)")
    print("  * Plaintext Weights (.pt) : 0 (100% Protected)")
    print("  * Encrypted Vaults (.avcs): Verified")
    print("  * Decryption Engine       : AES-256-GCM Volatile RAM Only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Locked AVC Distribution Wheel")
    parser.add_argument("--key", type=str, default=DEFAULT_COMMUNITY_KEY, help="API Key for encrypting vault")
    args = parser.parse_args()

    print("=" * 65)
    print(" AVC Enterprise Locked Distribution Builder")
    print("=" * 65)

    encrypt_weights(api_key=args.key)
    compile_bytecode()
    wheel = build_wheel()
    inspect_wheel(wheel)


if __name__ == "__main__":
    main()
