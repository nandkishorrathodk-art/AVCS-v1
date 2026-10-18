"""Build the local AVCS System 1 source-only developer-preview distribution."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "system1"
DIST = ROOT / "dist"
VERSION = "1.0.0rc1"
ARCHIVE_NAME = "avcs-system1-v1.0.0-preview.1-source-only.zip"
CHECKPOINT_SHA256 = (
    "c30eb0686da96eb6f67981021bcb646b60b3ed6b050927ee97085e59ee6125a4"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_python_package(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file() or (path.suffix != ".py" and path.name != "py.typed"):
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def validate_inputs() -> dict:
    metadata = json.loads((RELEASE / "model_metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or metadata.get("model_architecture") != "AVCSSystem1V1"
        or metadata.get("parameter_count") != 1_663_564
        or metadata.get("distribution_scope") != (
            "Source-only package; no trained checkpoint, candidate weights, data, "
            "annotations, screenshots, or evaluation report."
        )
    ):
        raise ValueError("Release metadata does not match the source-only scope")

    audit = (ROOT / "docs" / "RELEASE_RIGHTS_AUDIT.md").read_text(encoding="utf-8")
    if "BLOCKED" not in audit or "UNKNOWN" not in audit:
        raise ValueError("Rights audit is missing required non-cleared statuses")
    if metadata.get("checkpoint_redistribution_status", "").startswith("BLOCKED") is False:
        raise ValueError("Metadata must mark checkpoint redistribution as BLOCKED")
    if not metadata.get("evaluation_report_redistribution_status", "").startswith(
        "UNKNOWN"
    ):
        raise ValueError("Metadata must mark the evaluation report as UNKNOWN")
    return metadata


def build_package(dist: Path) -> tuple[Path, Path]:
    with tempfile.TemporaryDirectory(prefix="avcs-system1-build-") as temp:
        source_root = Path(temp) / "source"
        source_root.mkdir()
        shutil.copy2(RELEASE / "pyproject.toml", source_root / "pyproject.toml")
        shutil.copy2(RELEASE / "README.md", source_root / "README.md")
        shutil.copy2(ROOT / "LICENSE.md", source_root / "LICENSE")
        copy_python_package(ROOT / "avcs", source_root / "avcs")
        copy_python_package(ROOT / "forge", source_root / "forge")

        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(dist), str(source_root)],
            cwd=ROOT,
            check=True,
        )

    wheel = dist / f"avcs_system1-{VERSION}-py3-none-any.whl"
    sdist = dist / f"avcs_system1-{VERSION}.tar.gz"
    if not wheel.is_file() or not sdist.is_file():
        raise FileNotFoundError("Build backend did not produce the expected wheel and sdist")
    return wheel, sdist


def build_archive(
    archive_path: Path,
    wheel: Path,
    sdist: Path,
    metadata: dict,
) -> None:
    with tempfile.TemporaryDirectory(prefix="avcs-system1-release-") as temp:
        payload = Path(temp) / "avcs-system1-v1.0.0-preview.1-source-only"
        artifacts = payload / "artifacts"
        docs = payload / "docs"
        artifacts.mkdir(parents=True)
        docs.mkdir()

        shutil.copy2(RELEASE / "README.md", payload / "README.md")
        shutil.copy2(ROOT / "LICENSE.md", payload / "LICENSE")
        shutil.copy2(ROOT / "docs" / "SYSTEM1_DEVELOPER_GUIDE.md", docs)
        shutil.copy2(ROOT / "docs" / "AVCS_SYSTEM1_MODEL_CARD.md", docs)
        shutil.copy2(ROOT / "docs" / "RELEASE_RIGHTS_AUDIT.md", docs)
        shutil.copy2(wheel, artifacts)
        shutil.copy2(sdist, artifacts)
        shutil.copy2(RELEASE / "model_metadata.json", artifacts)

        manifest = {
            "release_name": "avcs-system1-v1.0.0-preview.1-source-only",
            "package_version": VERSION,
            "model_architecture": "AVCSSystem1V1",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "parameter_count": 1_663_564,
            "redistribution_scope": "Apache-2.0 source package only",
            "checkpoint_included": False,
            "checkpoint_rights": "BLOCKED",
            "training_data_included": False,
            "training_data_rights": "UNKNOWN",
            "evaluation_report_included": False,
            "evaluation_report_rights": "UNKNOWN; report withheld",
            "publication_status": "Locally packaged; not uploaded or published",
            "files": {},
        }
        for path in sorted(p for p in payload.rglob("*") if p.is_file()):
            manifest["files"][path.relative_to(payload).as_posix()] = sha256_file(path)
        (payload / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        sum_lines = []
        for path in sorted(p for p in payload.rglob("*") if p.is_file()):
            relative = path.relative_to(payload).as_posix()
            sum_lines.append(f"{sha256_file(path)}  {relative}")
        (payload / "SHA256SUMS").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

        with zipfile.ZipFile(
            archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(p for p in payload.rglob("*") if p.is_file()):
                archive.write(path, path.relative_to(payload.parent).as_posix())


def main() -> None:
    metadata = validate_inputs()
    DIST.mkdir(exist_ok=True)
    wheel_path = DIST / f"avcs_system1-{VERSION}-py3-none-any.whl"
    sdist_path = DIST / f"avcs_system1-{VERSION}.tar.gz"
    archive_path = DIST / ARCHIVE_NAME
    hash_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    for path in (wheel_path, sdist_path, archive_path, hash_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing release artifact: {path}"
            )

    wheel, sdist = build_package(DIST)
    build_archive(archive_path, wheel, sdist, metadata)
    archive_hash = sha256_file(archive_path)
    hash_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="ascii")

    print(f"Wheel:   {wheel}")
    print(f"Source:  {sdist}")
    print(f"Archive: {archive_path}")
    print(f"SHA256:  {archive_hash}")


if __name__ == "__main__":
    main()
