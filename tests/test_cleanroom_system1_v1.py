from __future__ import annotations

import hashlib
import json
import ast
import os
import re
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from avcs.dataset.cleanroom_v1.dataset import CleanroomSystem1Dataset
from avcs.dataset.cleanroom_v1.generator import (
    TARGET_LABELS,
    generate_dataset,
    validate_dataset,
)
from avcs.dataset.cleanroom_v1.glyphs import GLYPHS, draw_bitmap_text
from avcs.training.train_system1_clean_v1 import (
    DEFAULT_ATTESTATION,
    DEFAULT_MANIFEST,
    RightsGateError,
    assert_safe_output_path,
    collate_cleanroom_batch,
    compute_cleanroom_losses,
    instantiate_from_random_initialization,
    train_clean_candidate,
    validate_rights_attestation,
)
from avcs.perception.word_recognizer import WORD_TO_ID


SMALL_COUNTS = {
    "train": 8,
    "validation": 2,
    "test_heldout": 4,
    "test_negative": 2,
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    result = tmp_path / "cleanroom"
    generate_dataset(result, seed=7821, counts=SMALL_COUNTS)
    return result


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bitmap_glyph_table_is_closed_and_has_no_font_fallback() -> None:
    assert set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :.-?") == set(GLYPHS)
    assert all(len(rows) == 7 and all(len(row) == 5 for row in rows) for rows in GLYPHS.values())
    assert set(TARGET_LABELS).issubset(WORD_TO_ID)
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (32, 12), "white")
    with pytest.raises(ValueError, match="Unsupported bitmap glyph"):
        draw_bitmap_text(ImageDraw.Draw(image), (0, 0), "SAVE✓", (0, 0, 0), 32)
    assert set(TARGET_LABELS)  # Vocabulary is explicit, not inferred from host fonts.


def test_generator_is_deterministic_and_splits_by_disjoint_families(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, seed=7821, counts=SMALL_COUNTS)
    generate_dataset(second, seed=7821, counts=SMALL_COUNTS)
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    assert validate_dataset(first)["split_counts"] == SMALL_COUNTS
    assert first_manifest["rights"]["status"] == "PENDING_OWNER_ATTESTATION"
    assert not any(first_manifest["legacy_inputs_consumed"].values())
    families = {
        split: set(first_manifest["splits"][split]["layout_families"])
        for split in first_manifest["splits"]
    }
    for index, first_split in enumerate(families):
        for second_split in list(families)[index + 1:]:
            assert families[first_split].isdisjoint(families[second_split])


def test_manifest_hashes_and_annotation_coordinate_integrity(corpus: Path) -> None:
    report = validate_dataset(corpus)
    assert report["frame_count"] == sum(SMALL_COUNTS.values())
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image = corpus / manifest["files"][0]["image"]["relative_path"]
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_dataset(corpus)

    generate_dataset(corpus / "bad-coordinate", seed=7821, counts=SMALL_COUNTS)
    bad_dir = corpus / "bad-coordinate"
    bad_manifest_path = bad_dir / "manifest.json"
    bad_manifest = json.loads(bad_manifest_path.read_text(encoding="utf-8"))
    record = next(row for row in bad_manifest["files"] if row["split"] == "train")
    annotation_path = bad_dir / record["annotation"]["relative_path"]
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["regions"][0]["bbox_xyxy_px"] = [-1, 240, 256, 256]
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    record["annotation"]["sha256"] = _file_digest(annotation_path)
    bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid pixel xyxy box"):
        validate_dataset(bad_dir)


def test_manifest_rejects_empty_split_unknown_vocabulary_and_family_leakage(corpus: Path) -> None:
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [row for row in manifest["files"] if row["split"] != "test_negative"]
    manifest["splits"]["test_negative"]["count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="required train/validation/held-out/negative split"):
        validate_dataset(corpus)

    generate_dataset(corpus / "bad-vocabulary", seed=7821, counts=SMALL_COUNTS)
    bad_vocabulary_dir = corpus / "bad-vocabulary"
    bad_manifest_path = bad_vocabulary_dir / "manifest.json"
    bad_manifest = json.loads(bad_manifest_path.read_text(encoding="utf-8"))
    record = next(row for row in bad_manifest["files"] if row["split"] == "train")
    annotation_path = bad_vocabulary_dir / record["annotation"]["relative_path"]
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["regions"][0]["word_class"] = "NOT_IN_VOCABULARY"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    record["annotation"]["sha256"] = _file_digest(annotation_path)
    bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown word class"):
        validate_dataset(bad_vocabulary_dir)

    generate_dataset(corpus / "bad-family", seed=7821, counts=SMALL_COUNTS)
    bad_family_dir = corpus / "bad-family"
    bad_family_path = bad_family_dir / "manifest.json"
    bad_family_manifest = json.loads(bad_family_path.read_text(encoding="utf-8"))
    record = next(row for row in bad_family_manifest["files"] if row["split"] == "test_heldout")
    record["layout_family"] = "toolbar"
    bad_family_path.write_text(json.dumps(bad_family_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split-incompatible layout family"):
        validate_dataset(bad_family_dir)


def test_generator_refuses_overwrite_and_bad_split_configuration(tmp_path: Path) -> None:
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "keep.txt").write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate_dataset(target, counts=SMALL_COUNTS)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "unchanged"
    with pytest.raises(ValueError, match="positive integer"):
        generate_dataset(
            tmp_path / "invalid",
            counts={**SMALL_COUNTS, "train": 0},
        )


def test_clean_dataset_maps_renderer_labels_into_all_model_supervision(corpus: Path) -> None:
    data = CleanroomSystem1Dataset(corpus, "train")
    assert len(data) == SMALL_COUNTS["train"] * 5
    sample = data[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["word_crops"].shape == (5, 3, 32, 96)
    assert int(sample["p2_aff"].sum() + sample["p3_aff"].sum() + sample["p4_aff"].sum()) == 4
    assert int(sample["p2_aff"].sum()) > 0
    assert int(sample["p3_aff"].sum()) > 0
    assert int(sample["p4_aff"].sum()) > 0
    assert sample["target_box"].shape == (4,)

    loader = DataLoader(data, batch_size=1, shuffle=False)
    model, _ = instantiate_from_random_initialization(19)
    model.eval()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            losses = compute_cleanroom_losses(model, next(iter(loader)))
    finally:
        torch.set_num_threads(previous_threads)
    assert set(losses) == {"category", "proposals", "grounding", "words", "total"}
    assert all(torch.isfinite(loss).item() for loss in losses.values())


def test_cleanroom_collation_preserves_absent_target_metadata(corpus: Path) -> None:
    negatives = CleanroomSystem1Dataset(corpus, "test_negative")
    loader = DataLoader(
        negatives,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_cleanroom_batch,
    )
    batch = next(iter(loader))
    assert batch["target_region_id"] == [None, None]
    assert batch["image"].shape == (2, 3, 256, 256)


def test_random_initialization_is_repeatable_and_checkpoint_loading_is_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load(*args, **kwargs):
        raise AssertionError("Unexpected checkpoint load attempt")

    monkeypatch.setattr(torch, "load", fail_load)
    first, first_hash = instantiate_from_random_initialization(933)
    second, second_hash = instantiate_from_random_initialization(933)
    third, third_hash = instantiate_from_random_initialization(934)
    assert first_hash == second_hash
    assert first_hash != third_hash
    assert all(parameter.requires_grad for parameter in first.parameters())


def test_cleanroom_modules_do_not_import_external_ocr_or_desktop_control() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "avcs" / "dataset" / "cleanroom_v1"
    paths = list(package_dir.glob("*.py")) + [
        Path(__file__).resolve().parents[1] / "avcs" / "training" / "train_system1_clean_v1.py",
        Path(__file__).resolve().parents[1] / "avcs" / "training" / "evaluate_system1_clean_v1.py",
        Path(__file__).resolve().parents[1] / "avcs" / "perception" / "system1_v1.py",
        Path(__file__).resolve().parents[1] / "avcs" / "perception" / "goal_grounder.py",
        Path(__file__).resolve().parents[1] / "avcs" / "perception" / "word_recognizer.py",
        Path(__file__).resolve().parents[1] / "avcs" / "perception" / "human_eye.py",
    ]
    forbidden_roots = {"cv2", "pytesseract", "easyocr", "selenium", "playwright", "pyautogui"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".", 1)[0])
        assert forbidden_roots.isdisjoint(imports), path


def test_random_model_emits_typed_pixel_observation_contract(corpus: Path) -> None:
    data = CleanroomSystem1Dataset(corpus, "train")
    sample = data[0]
    model, _ = instantiate_from_random_initialization(108)
    model.eval()
    pixels = sample["image"].permute(1, 2, 0).mul(255).round().byte().numpy()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        observation = model.observe_frame(pixels, active_query=sample["query"])
    finally:
        torch.set_num_threads(old_threads)
    assert observation.screen_dimensions == (256, 256)
    assert observation.target_presence in {"PRESENT", "ABSENT", "UNCERTAIN"}
    assert 0.0 <= observation.presence_confidence <= 1.0
    assert isinstance(observation.detected_regions, list)
    assert isinstance(observation.uncertainty_flags, list)
    if observation.selected_target_norm is not None:
        assert all(0.0 <= coordinate <= 1.0 for coordinate in observation.selected_target_norm)


def test_rights_gate_blocks_pending_attestation_before_model_construction(
    corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_construction(seed: int):
        raise AssertionError("Model construction must happen only after rights approval")

    monkeypatch.setattr(
        "avcs.training.train_system1_clean_v1.instantiate_from_random_initialization",
        unexpected_construction,
    )
    with pytest.raises(RightsGateError):
        train_clean_candidate(
            corpus / "manifest.json",
            DEFAULT_ATTESTATION,
            tmp_path / "blocked.pt",
            epochs=1,
            batch_size=1,
        )
    assert not (tmp_path / "blocked.pt").exists()


def test_attestation_must_bind_exact_hash_and_fully_clear_rights(tmp_path: Path) -> None:
    attestation = tmp_path / "attestation.md"
    template = DEFAULT_ATTESTATION.read_text(encoding="utf-8")
    template = re.sub(
        r"(?m)^Dataset manifest SHA-256:.*$",
        "Dataset manifest SHA-256: " + "a" * 64,
        template,
    )
    attestation.write_text(template, encoding="utf-8")
    with pytest.raises(RightsGateError, match="exact dataset manifest"):
        validate_rights_attestation(attestation, "b" * 64)
    with pytest.raises(RightsGateError, match="not explicitly APPROVED"):
        validate_rights_attestation(attestation, "a" * 64)
    with pytest.raises(RightsGateError, match="missing"):
        validate_rights_attestation(tmp_path / "not-created.md", "a" * 64)


def test_output_path_protects_legacy_checkpoint_and_candidate_tree() -> None:
    frozen_baseline = Path(__file__).resolve().parents[1] / "runs/weights/candidates/avcs_system1_v1.pt"
    with pytest.raises(ValueError, match="frozen baseline"):
        assert_safe_output_path(frozen_baseline)
    candidate_dir = frozen_baseline.parent / "new-clean.pt"
    with pytest.raises(ValueError, match="legacy candidate directory"):
        assert_safe_output_path(candidate_dir)


def test_rights_cleared_cpu_smoke_training_is_explicitly_gated(tmp_path: Path) -> None:
    if not DEFAULT_MANIFEST.is_file():
        pytest.skip("The canonical clean-room corpus has not been generated.")
    if os.environ.get("AVCS_RUN_CLEANROOM_CPU_SMOKE") != "1":
        pytest.skip("Set AVCS_RUN_CLEANROOM_CPU_SMOKE=1 to explicitly run smoke training.")
    root = Path(__file__).resolve().parents[1]
    approval_path = root / "docs" / "CLEANROOM_RIGHTS_APPROVAL.json"
    trust_config_path = root / "docs" / "CLEANROOM_RIGHTS_TRUST.json"
    approved_attestation = root / "docs" / "CLEANROOM_DATA_RIGHTS_ATTESTATION_APPROVED.md"
    expected_trust_hash = os.environ.get("AVCS_RIGHTS_TRUST_CONFIG_SHA256")
    if not all(path.is_file() for path in (approval_path, trust_config_path, approved_attestation)):
        pytest.skip("The signed owner approval bundle is unavailable.")
    if not expected_trust_hash:
        pytest.skip("The independent trusted-config hash pin is unavailable.")
    manifest_hash = validate_dataset(DEFAULT_MANIFEST.parent)["manifest_sha256"]
    validate_rights_attestation(
        approved_attestation,
        manifest_hash,
        approval_path=approval_path,
        trust_config_path=trust_config_path,
        expected_trust_config_sha256=expected_trust_hash,
    )
    result = train_clean_candidate(
        DEFAULT_MANIFEST,
        approved_attestation,
        tmp_path / "attested-smoke.pt",
        approval_path=approval_path,
        trust_config_path=trust_config_path,
        expected_trust_config_sha256=expected_trust_hash,
        epochs=1,
        batch_size=4,
        seed=20261018,
    )
    assert Path(result["checkpoint_path"]).is_file()
    assert len(result["checkpoint_sha256"]) == 64
    metadata = json.loads(Path(result["sidecar_path"]).read_text(encoding="utf-8"))
    assert metadata["training_metadata"]["lineage"] == "clean_random_initialization_v1"
