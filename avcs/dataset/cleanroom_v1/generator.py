"""Deterministic first-party synthetic corpus generator for System 1."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION
import torch

from avcs.dataset.cleanroom_v1 import ANNOTATION_SCHEMA, DATASET_VERSION
from avcs.dataset.cleanroom_v1.glyphs import (
    GLYPH_VERSION,
    SUPPORTED_CHARACTERS,
    draw_bitmap_text,
)

CANVAS_SIZE = (256, 256)
SUPPORTED_SPLITS = ("train", "validation", "test_heldout", "test_negative")
TARGET_LABELS = (
    "Save",
    "Search",
    "Settings",
    "Cancel",
    "File",
    "Edit",
    "Back",
    "Notes",
    "Menu",
    "OK",
    "Start",
    "Stop",
    "Play",
    "Pause",
    "View",
    "Tools",
    "Exit",
    "Bookmarks",
)
ELEMENT_KINDS = ("button", "input_field", "tab", "icon")
LAYOUT_FAMILIES = {
    "train": ("toolbar", "sidebar", "grid", "dialog"),
    "validation": ("compact",),
    "test_heldout": ("split_panel", "stacked"),
    "test_negative": ("empty_desktop",),
}
CATEGORY_BY_FAMILY = {
    "toolbar": "desktop_canvas",
    "sidebar": "settings_panel",
    "grid": "form_input",
    "dialog": "modal_dialog",
    "compact": "desktop_canvas",
    "split_panel": "settings_panel",
    "stacked": "form_input",
    "empty_desktop": "desktop_canvas",
}

PALETTES = (
    {
        "desktop": (28, 38, 52), "window": (241, 244, 248), "bar": (52, 72, 96),
        "ink": (21, 30, 41), "muted": (93, 105, 120), "accent": (55, 115, 180),
        "field": (255, 255, 255), "border": (122, 139, 158), "taskbar": (33, 44, 60),
    },
    {
        "desktop": (47, 54, 63), "window": (235, 232, 222), "bar": (91, 85, 72),
        "ink": (36, 37, 35), "muted": (100, 99, 90), "accent": (145, 104, 56),
        "field": (250, 248, 240), "border": (133, 127, 112), "taskbar": (38, 42, 47),
    },
    {
        "desktop": (29, 50, 47), "window": (231, 241, 237), "bar": (44, 83, 72),
        "ink": (22, 39, 34), "muted": (83, 112, 100), "accent": (39, 135, 104),
        "field": (250, 255, 252), "border": (111, 146, 132), "taskbar": (25, 43, 40),
    },
    {
        "desktop": (56, 39, 55), "window": (244, 235, 243), "bar": (93, 60, 88),
        "ink": (43, 30, 41), "muted": (116, 91, 111), "accent": (158, 81, 135),
        "field": (255, 249, 254), "border": (150, 120, 145), "taskbar": (42, 31, 43),
    },
)

_BASE_POSITIONS = {
    "toolbar": ((24, 69), (100, 108), (174, 69), (207, 142)),
    "sidebar": ((27, 65), (40, 105), (150, 164), (207, 116)),
    "grid": ((25, 61), (122, 61), (29, 123), (202, 126)),
    "dialog": ((43, 123), (124, 123), (83, 162), (202, 87)),
    "compact": ((21, 70), (88, 106), (184, 70), (207, 123)),
    "split_panel": ((24, 59), (125, 101), (37, 163), (204, 163)),
    "stacked": ((30, 60), (34, 103), (151, 160), (208, 66)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_revision() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            "git", "-C", str(repo_root), "status", "--porcelain", "--",
            "avcs/dataset/cleanroom_v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_commit": head.stdout.strip() if head.returncode == 0 else None,
        "generator_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _draw_icon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], ink: tuple[int, int, int], style: int) -> None:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if style % 3 == 0:
        draw.ellipse((cx - 6, cy - 6, cx + 2, cy + 2), outline=ink, width=2)
        draw.line((cx + 1, cy + 1, cx + 7, cy + 7), fill=ink, width=2)
    elif style % 3 == 1:
        draw.line((cx - 6, cy, cx + 6, cy), fill=ink, width=2)
        draw.line((cx, cy - 6, cx, cy + 6), fill=ink, width=2)
    else:
        for offset in (-4, 0, 4):
            draw.line((cx - 6, cy + offset, cx + 6, cy + offset), fill=ink, width=1)


def _draw_control(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    kind: str,
    label: str | None,
    palette: dict[str, tuple[int, int, int]],
    icon_style: int,
) -> None:
    x1, y1, x2, y2 = box
    if kind == "button":
        draw.rounded_rectangle(box, radius=3, fill=palette["accent"], outline=palette["border"])
        text_color = (255, 255, 255)
    elif kind == "input_field":
        draw.rectangle(box, fill=palette["field"], outline=palette["accent"], width=1)
        text_color = palette["muted"]
    elif kind == "tab":
        draw.rectangle(box, fill=palette["window"], outline=palette["border"], width=1)
        draw.line((x1 + 2, y2 - 2, x2 - 2, y2 - 2), fill=palette["accent"], width=2)
        text_color = palette["ink"]
    else:
        draw.rectangle(box, fill=palette["window"], outline=palette["border"], width=1)
        _draw_icon(draw, box, palette["accent"], icon_style)
        return

    if label:
        text_width = max(1, x2 - x1 - 8)
        text_bounds = draw_bitmap_text(
            draw, (x1 + 4, y1 + max(1, ((y2 - y1) - 7) // 2)), label, text_color, text_width
        )
        if text_bounds[2] > x2 - 3:
            raise ValueError(f"Rendered label does not fit its control: {label}")


def _positions_for(family: str, rng: random.Random) -> list[tuple[int, int]]:
    positions = _BASE_POSITIONS[family]
    return [
        (x + rng.randint(-3, 3), y + rng.randint(-2, 2))
        for x, y in positions
    ]


def _frame_for(family: str, seed: int, frame_id: str) -> tuple[Image.Image, dict[str, Any]]:
    rng = random.Random(seed)
    palette = PALETTES[rng.randrange(len(PALETTES))]
    image = Image.new("RGB", CANVAS_SIZE, palette["desktop"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((13, 18, 243, 231), fill=palette["window"], outline=palette["border"], width=2)
    draw.rectangle((14, 19, 242, 39), fill=palette["bar"])
    draw.ellipse((20, 26, 24, 30), fill=(214, 88, 82))
    draw.ellipse((28, 26, 32, 30), fill=(222, 181, 71))
    draw.ellipse((36, 26, 40, 30), fill=(95, 178, 126))
    draw_bitmap_text(draw, (48, 26), "WORKSPACE", (255, 255, 255), 70)
    draw.rectangle((14, 232, 242, 255), fill=palette["taskbar"])
    draw_bitmap_text(draw, (20, 240), "DESKTOP 01", (230, 235, 240), 90)
    unknown_text = f"Q{rng.randrange(10)}Z"
    unknown_box = draw_bitmap_text(
        draw, (174, 240), unknown_text, (190, 200, 210), 42
    )

    category = CATEGORY_BY_FAMILY[family]
    regions: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []

    if family == "empty_desktop":
        absent_label = TARGET_LABELS[rng.randrange(len(TARGET_LABELS))]
        queries.append({
            "query": f"click {absent_label} button",
            "target_presence": False,
            "target_region_id": None,
            "target_point_norm": None,
            "query_kind": "absent_supported_label",
        })
    else:
        positions = _positions_for(family, rng)
        labels = rng.sample(TARGET_LABELS, k=3)
        widths = (70, 104, 55, 26)
        heights = (23, 23, 23, 26)
        kinds = ("button", "input_field", "tab", "icon")
        boxes: list[tuple[int, int, int, int]] = []
        for index, (kind, label, position) in enumerate(zip(kinds, (*labels, None), positions)):
            x, y = position
            box = (x, y, x + widths[index], y + heights[index])
            boxes.append(box)
            _draw_control(draw, box, kind, label, palette, seed + index)
            x1, y1, x2, y2 = box
            region_id = f"{frame_id}_region_{index:02d}"
            region_label = label if label is not None else "<none>"
            regions.append({
                "region_id": region_id,
                "kind": kind,
                "bbox_xyxy_px": [x1, y1, x2, y2],
                "bbox_xyxy_norm": [
                    round(x1 / CANVAS_SIZE[0], 6),
                    round(y1 / CANVAS_SIZE[1], 6),
                    round(x2 / CANVAS_SIZE[0], 6),
                    round(y2 / CANVAS_SIZE[1], 6),
                ],
                "label": region_label,
                "word_class": region_label if label is not None else "<none>",
            })
            if label is None:
                query_text = "click icon"
            elif kind == "input_field":
                query_text = f"click {label} input field"
            elif kind == "tab":
                query_text = f"click {label} tab"
            else:
                query_text = f"click {label} button"
            queries.append({
                "query": query_text,
                "target_presence": True,
                "target_region_id": region_id,
                "target_point_norm": [
                    round((x1 + x2) / (2 * CANVAS_SIZE[0]), 6),
                    round((y1 + y2) / (2 * CANVAS_SIZE[1]), 6),
                ],
                "query_kind": "positive_element_query",
            })

        absent_candidates = [label for label in TARGET_LABELS if label not in labels]
        absent_label = rng.choice(absent_candidates)
        queries.append({
            "query": f"click {absent_label} button",
            "target_presence": False,
            "target_region_id": None,
            "target_point_norm": None,
            "query_kind": "absent_supported_label",
        })

    annotation = {
        "schema": ANNOTATION_SCHEMA,
        "frame_id": frame_id,
        "width": CANVAS_SIZE[0],
        "height": CANVAS_SIZE[1],
        "coordinate_convention": "normalized_xy_origin_top_left",
        "layout_family": family,
        "category": category,
        "is_negative_frame": family == "empty_desktop",
        "regions": regions,
        "queries": queries,
        "unknown_word": {
            "text": unknown_text,
            "bbox_xyxy_px": list(unknown_box),
            "word_class": "<none>",
        },
    }
    return image, annotation


def _validate_counts(counts: dict[str, int]) -> None:
    if set(counts) != set(SUPPORTED_SPLITS):
        raise ValueError(f"Counts must define exactly these splits: {SUPPORTED_SPLITS}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError("Every split count must be a positive integer")
    if counts["train"] < len(LAYOUT_FAMILIES["train"]):
        raise ValueError("Training count must cover every training layout family")
    if counts["test_heldout"] < len(LAYOUT_FAMILIES["test_heldout"]):
        raise ValueError("Held-out count must cover every held-out layout family")


def generate_dataset(
    output_dir: str | Path = "datasets/cleanroom_system1_v1",
    seed: int = 20261018,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Generate a corpus only in a new empty destination; never reads legacy data."""
    counts = counts or {
        "train": 64,
        "validation": 16,
        "test_heldout": 16,
        "test_negative": 16,
    }
    _validate_counts(counts)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    corpus_dir = Path(output_dir).resolve()
    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty corpus directory: {corpus_dir}")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("glyphs.py").resolve(),
        Path(__file__).with_name("dataset.py").resolve(),
        Path(__file__).with_name("__init__.py").resolve(),
    )
    source_records = [
        {
            "relative_path": path.relative_to(repo_root).as_posix(),
            "sha256": _sha256(path),
            "role": "first_party_data_pipeline_source",
            "source_method": "Project-authored code used for generation, annotation, or training-input construction",
            "ownership_basis": "Contribution and project ownership must be confirmed by the authorized owner.",
            "license_status": "UNRESOLVED_PENDING_OWNER_REVIEW",
            "included_in_training_data": False,
        }
        for path in source_paths
    ]
    revision = _repository_revision()
    split_seed_offsets = {
        "train": 0,
        "validation": 1_000_000,
        "test_heldout": 2_000_000,
        "test_negative": 3_000_000,
    }
    records: list[dict[str, Any]] = []
    used_families: dict[str, set[str]] = defaultdict(set)

    for split in SUPPORTED_SPLITS:
        families = LAYOUT_FAMILIES[split]
        for index in range(counts[split]):
            family = families[index % len(families)]
            sample_seed = seed + split_seed_offsets[split] + index
            frame_id = f"{split}_{index:05d}"
            image, annotation = _frame_for(family, sample_seed, frame_id)
            image_rel = f"frames/{split}/{frame_id}.png"
            annotation_rel = f"annotations/{split}/{frame_id}.json"
            image_path = corpus_dir / image_rel
            annotation_path = corpus_dir / annotation_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, format="PNG", optimize=False, compress_level=9)
            annotation_path.write_text(
                json.dumps(annotation, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            used_families[split].add(family)
            records.append({
                "frame_id": frame_id,
                "split": split,
                "layout_family": family,
                "sample_seed": sample_seed,
                "category": annotation["category"],
                "provenance": {
                    "source_method": "procedurally rendered from first-party bitmap glyphs and geometric primitives",
                    "ownership_basis": "First-party generation method is documented; legal ownership and permissions remain unverified pending the authorized owner attestation.",
                    "license_status": "UNRESOLVED_PENDING_OWNER_REVIEW",
                    "included_in_proposed_corpus": True,
                    "included_in_gradient_training": split == "train",
                    "included_in_validation_or_model_selection": split == "validation",
                    "included_in_final_test_only": split in ("test_heldout", "test_negative"),
                    "unresolved_rights": "Owner must attest that the exact corpus and resulting weights may be used and distributed.",
                },
                "image": {
                    "relative_path": image_rel,
                    "sha256": _sha256(image_path),
                    "width": CANVAS_SIZE[0],
                    "height": CANVAS_SIZE[1],
                    "source_type": "first_party_procedural_raster",
                    "source_method": "Pillow drawing primitives and project-authored bitmap glyphs; no system fonts or external images",
                    "license_status": "UNRESOLVED_PENDING_OWNER_REVIEW",
                },
                "annotation": {
                    "relative_path": annotation_rel,
                    "sha256": _sha256(annotation_path),
                    "schema": ANNOTATION_SCHEMA,
                    "source_type": "deterministic_renderer_ground_truth",
                    "source_method": "Exact geometry and text labels emitted by the clean-room renderer",
                    "license_status": "UNRESOLVED_PENDING_OWNER_REVIEW",
                },
                "rights_status": "PENDING_OWNER_ATTESTATION",
            })

    family_sets = {split: used_families[split] for split in SUPPORTED_SPLITS}
    for split_a, families_a in family_sets.items():
        for split_b, families_b in family_sets.items():
            if split_a < split_b and families_a & families_b:
                raise RuntimeError(f"Layout family leakage between {split_a} and {split_b}")

    config = {
        "seed": seed,
        "counts": counts,
        "canvas_size": list(CANVAS_SIZE),
        "palette_count": len(PALETTES),
        "position_jitter_px": {"x": [-3, 3], "y": [-2, 2]},
        "target_labels_sampled_without_replacement_per_frame": 3,
        "families_by_split": {split: sorted(LAYOUT_FAMILIES[split]) for split in SUPPORTED_SPLITS},
        "layout_assignment": "round-robin over disjoint template-family sets; split seed ranges do not overlap",
        "negative_queries_per_frame": 1,
        "positive_queries_per_positive_frame": 4,
    }
    manifest = {
        "schema": "avcs.system1.cleanroom.dataset.v1",
        "dataset_version": DATASET_VERSION,
        "generator_version": "cleanroom-renderer-v1",
        "generator_source_revision": revision,
        "generator_sources": source_records,
        "generation_config": config,
        "runtime": {
            "python": sys.version.split()[0],
            "pillow": PILLOW_VERSION,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "annotation_schema": ANNOTATION_SCHEMA,
        "vocabulary_version": "cleanroom-target-v1",
        "supported_target_labels": list(TARGET_LABELS),
        "supported_element_kinds": list(ELEMENT_KINDS),
        "supported_categories": sorted(set(CATEGORY_BY_FAMILY.values())),
        "limitations": [
            "Synthetic fictional UI only; not evidence of real-desktop accuracy.",
            "Bitmap text is uppercase 5x7 ASCII for the explicitly listed character set.",
            "Text vocabulary is limited to the listed target labels; unknown text is trained as <none>.",
            "Temporal motion behavior is not represented by this still-frame corpus.",
            "Visual region annotations cover only the renderer's explicitly drawn interactive controls.",
        ],
        "asset_provenance": {
            "screen_layouts": {
                "source_method": "first-party Python drawing code in avcs/dataset/cleanroom_v1/generator.py",
                "included_in_proposed_corpus": True,
                "ownership_basis": "First-party generation method is documented; owner confirmation remains required.",
                "license_status": "UNRESOLVED_PENDING_OWNER_REVIEW",
            },
            "glyphs": {
                "source_method": f"project-authored bitmap glyph rows in glyphs.py ({GLYPH_VERSION}); no operating-system font lookup",
                "included_in_proposed_corpus": True,
                "ownership_basis": "Authorship and contribution rights must be confirmed by the authorized owner.",
                "license_status": "UNRESOLVED_PENDING_OWNER_REVIEW",
            },
            "icons": {
                "source_method": "basic geometric primitives rendered by Pillow drawing operations",
                "included_in_proposed_corpus": True,
                "ownership_basis": "Generated shapes; legal rights and permitted weights use remain pending owner review.",
                "license_status": "UNRESOLVED_PENDING_OWNER_REVIEW",
            },
            "external_images_or_logos": {"included": False, "source_method": "none", "license_status": "NOT_APPLICABLE"},
            "captured_desktop_frames": {"included": False, "source_method": "none", "license_status": "NOT_APPLICABLE"},
            "third_party_fonts": {"included": False, "source_method": "none", "license_status": "NOT_APPLICABLE"},
            "legacy_masks_or_annotations": {"included": False, "source_method": "none", "license_status": "NOT_APPLICABLE"},
            "candidate_checkpoints": {"included": False, "source_method": "none", "license_status": "NOT_APPLICABLE"},
            "dependencies": {
                "generator": {"pillow": PILLOW_VERSION},
                "training_and_evaluation": {"numpy": np.__version__, "torch": torch.__version__},
                "license_status": "DEPENDENCY LICENSES REQUIRE OWNER REVIEW",
            },
        },
        "rights": {
            "source_type": "new_first_party_procedural_synthetic",
            "status": "PENDING_OWNER_ATTESTATION",
            "owner_approval_record": None,
            "unresolved_issue": "Project owner must review generator contributions and dependency licensing and sign the matching manifest digest.",
            "source_code_license_does_not_clear_corpus_or_model_weights": True,
        },
        "legacy_inputs_consumed": {
            "datasets/ui_grounding": False,
            "runs/demonstrations": False,
            "runs/weights": False,
            "candidate_checkpoints": False,
            "default_system1_trainer": False,
        },
        "splits": {
            split: {
                "count": counts[split],
                "layout_families": sorted(family_sets[split]),
                "seed_min": min(r["sample_seed"] for r in records if r["split"] == split),
                "seed_max": max(r["sample_seed"] for r in records if r["split"] == split),
                "purpose": {
                    "train": "gradient updates only after rights gate",
                    "validation": "training-selection/early-stopping only",
                    "test_heldout": "final evaluation only; excluded from training and selection",
                    "test_negative": "final absent-target evaluation only",
                }[split],
            }
            for split in SUPPORTED_SPLITS
        },
        "files": records,
    }
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_dataset(corpus_dir: str | Path, verify_generator_sources: bool = True) -> dict[str, Any]:
    """Validate hashes, annotations, split isolation, vocabulary and coordinates."""
    corpus_dir = Path(corpus_dir).resolve()
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Clean-room dataset manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "avcs.system1.cleanroom.dataset.v1":
        raise ValueError("Unsupported clean-room dataset manifest schema")
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError("Unsupported clean-room dataset version")
    if manifest.get("rights", {}).get("status") != "PENDING_OWNER_ATTESTATION":
        raise ValueError("Clean-room manifest rights status must remain pending owner attestation")
    legacy_inputs = manifest.get("legacy_inputs_consumed")
    if not isinstance(legacy_inputs, dict) or not legacy_inputs or any(legacy_inputs.values()):
        raise ValueError("Manifest does not affirm exclusion of every legacy input")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Dataset manifest has no file records")
    counts = Counter(record.get("split") for record in files)
    if any(counts[split] <= 0 for split in SUPPORTED_SPLITS):
        raise ValueError("Every required train/validation/held-out/negative split must be non-empty")
    if any(manifest.get("splits", {}).get(split, {}).get("count") != counts[split] for split in SUPPORTED_SPLITS):
        raise ValueError("Manifest split counts do not match the listed samples")

    supported_labels = set(manifest.get("supported_target_labels", []))
    supported_kinds = set(ELEMENT_KINDS)
    supported_categories = set(manifest.get("supported_categories", []))
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    families_by_split: dict[str, set[str]] = defaultdict(set)
    seeds: set[int] = set()
    seen_frame_ids: set[str] = set()
    image_hashes: list[str] = []
    annotation_hashes: list[str] = []

    for record in files:
        split = record.get("split")
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unknown split: {split!r}")
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or frame_id in seen_frame_ids:
            raise ValueError(f"Missing or duplicate frame_id: {frame_id!r}")
        seen_frame_ids.add(frame_id)
        family = record.get("layout_family")
        if not isinstance(family, str) or family not in LAYOUT_FAMILIES[split]:
            raise ValueError(f"Unknown or split-incompatible layout family: {family!r}")
        families_by_split[split].add(family)
        provenance = record.get("provenance", {})
        if (
            provenance.get("license_status") != "UNRESOLVED_PENDING_OWNER_REVIEW"
            or provenance.get("included_in_proposed_corpus") is not True
            or provenance.get("included_in_gradient_training") is not (split == "train")
            or provenance.get("included_in_validation_or_model_selection") is not (split == "validation")
            or provenance.get("included_in_final_test_only") is not (split in ("test_heldout", "test_negative"))
            or not provenance.get("ownership_basis")
            or not provenance.get("unresolved_rights")
        ):
            raise ValueError(f"Missing, inconsistent, or cleared provenance record in {frame_id}")
        sample_seed = record.get("sample_seed")
        if isinstance(sample_seed, bool) or not isinstance(sample_seed, int) or sample_seed in seeds:
            raise ValueError("Missing, invalid, or duplicate sample seed")
        seeds.add(sample_seed)

        for file_key, digest_list in (("image", image_hashes), ("annotation", annotation_hashes)):
            item = record.get(file_key, {})
            relative = item.get("relative_path")
            expected_hash = item.get("sha256")
            if item.get("license_status") != "UNRESOLVED_PENDING_OWNER_REVIEW" or not item.get("source_method"):
                raise ValueError(f"Missing pending rights or source-method metadata for {relative}")
            if not isinstance(relative, str) or not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"Invalid {file_key} file record for {frame_id}")
            path = (corpus_dir / relative).resolve()
            if corpus_dir not in path.parents:
                raise ValueError(f"Dataset path escapes corpus directory: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file missing: {relative}")
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {relative}")
            digest_list.append(actual_hash)
            hashes_by_split[split].add(actual_hash)

        image_path = corpus_dir / record["image"]["relative_path"]
        annotation_path = corpus_dir / record["annotation"]["relative_path"]
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.size != CANVAS_SIZE:
                raise ValueError(f"Invalid frame format or dimensions: {record['image']['relative_path']}")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation.get("schema") != ANNOTATION_SCHEMA or annotation.get("frame_id") != frame_id:
            raise ValueError(f"Annotation schema/frame mismatch for {frame_id}")
        if annotation.get("category") != record.get("category") or record.get("category") not in supported_categories:
            raise ValueError(f"Unknown or inconsistent category for {frame_id}")
        if annotation.get("width") != CANVAS_SIZE[0] or annotation.get("height") != CANVAS_SIZE[1]:
            raise ValueError(f"Annotation dimensions do not match the rendered frame for {frame_id}")
        regions = annotation.get("regions")
        queries = annotation.get("queries")
        if not isinstance(regions, list) or not isinstance(queries, list) or not queries:
            raise ValueError(f"Missing regions or queries for {frame_id}")
        region_ids: set[str] = set()
        region_labels: set[str] = set()
        for region in regions:
            region_id = region.get("region_id")
            if not isinstance(region_id, str) or region_id in region_ids:
                raise ValueError(f"Missing or duplicate region ID in {frame_id}")
            region_ids.add(region_id)
            kind = region.get("kind")
            if kind not in supported_kinds:
                raise ValueError(f"Unknown element kind {kind!r} in {frame_id}")
            label = region.get("label")
            word_class = region.get("word_class")
            if label != "<none>" and label not in supported_labels:
                raise ValueError(f"Unknown vocabulary label {label!r} in {frame_id}")
            if word_class != "<none>" and word_class not in supported_labels:
                raise ValueError(f"Unknown word class {word_class!r} in {frame_id}")
            region_labels.add(label)
            bbox = region.get("bbox_xyxy_norm")
            if (
                not isinstance(bbox, list) or len(bbox) != 4
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in bbox)
                or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)
            ):
                raise ValueError(f"Invalid normalized xyxy box in {frame_id}")
            px = region.get("bbox_xyxy_px")
            if (
                not isinstance(px, list) or len(px) != 4
                or any(not isinstance(v, int) or isinstance(v, bool) for v in px)
                or not (0 <= px[0] < px[2] <= CANVAS_SIZE[0] and 0 <= px[1] < px[3] <= CANVAS_SIZE[1])
            ):
                raise ValueError(f"Invalid pixel xyxy box in {frame_id}")
            normalized_from_pixels = [
                round(px[0] / CANVAS_SIZE[0], 6),
                round(px[1] / CANVAS_SIZE[1], 6),
                round(px[2] / CANVAS_SIZE[0], 6),
                round(px[3] / CANVAS_SIZE[1], 6),
            ]
            if bbox != normalized_from_pixels:
                raise ValueError(f"Normalized and pixel boxes disagree in {frame_id}")
        unknown = annotation.get("unknown_word")
        if not isinstance(unknown, dict) or unknown.get("word_class") != "<none>":
            raise ValueError(f"Missing <none>-labeled unknown-word sample in {frame_id}")
        unknown_text = unknown.get("text")
        unknown_box = unknown.get("bbox_xyxy_px")
        if (
            not isinstance(unknown_text, str)
            or not unknown_text
            or set(unknown_text.upper()) - set(SUPPORTED_CHARACTERS)
            or not isinstance(unknown_box, list)
            or len(unknown_box) != 4
            or any(not isinstance(v, int) or isinstance(v, bool) for v in unknown_box)
            or not (0 <= unknown_box[0] < unknown_box[2] <= CANVAS_SIZE[0]
                    and 0 <= unknown_box[1] < unknown_box[3] <= CANVAS_SIZE[1])
        ):
            raise ValueError(f"Invalid unknown-word annotation in {frame_id}")
        if (split == "test_negative") != bool(annotation.get("is_negative_frame")):
            raise ValueError(f"Negative-frame flag does not match its split in {frame_id}")
        for query in queries:
            if not isinstance(query.get("query"), str) or not query["query"].strip():
                raise ValueError(f"Empty query in {frame_id}")
            query_words = set(re.findall(r"[A-Za-z0-9]+", query["query"].casefold()))
            supported_query_words = {
                word.casefold() for word in supported_labels
            } | {"click", "button", "input", "field", "tab", "icon"}
            if query_words - supported_query_words:
                raise ValueError(f"Query has unsupported vocabulary in {frame_id}: {query['query']!r}")
            present = query.get("target_presence")
            target_region_id = query.get("target_region_id")
            point = query.get("target_point_norm")
            if present is True:
                if target_region_id not in region_ids or not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f"Positive query has invalid target reference in {frame_id}")
                target_region = next(region for region in regions if region["region_id"] == target_region_id)
                target_label = target_region["label"]
                if target_label != "<none>" and target_label.casefold() not in query_words:
                    raise ValueError(f"Positive query does not identify its target label in {frame_id}")
                if query.get("query_kind") != "positive_element_query":
                    raise ValueError(f"Positive query kind is inconsistent in {frame_id}")
                if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1 for v in point):
                    raise ValueError(f"Positive query target point out of range in {frame_id}")
                bbox = target_region["bbox_xyxy_norm"]
                if not (bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]):
                    raise ValueError(f"Positive target point lies outside its annotated box in {frame_id}")
            elif present is False:
                if target_region_id is not None or point is not None:
                    raise ValueError(f"Absent query carries a target in {frame_id}")
                if query.get("query_kind") != "absent_supported_label":
                    raise ValueError(f"Absent query kind is inconsistent in {frame_id}")
                queried_labels = {
                    label for label in supported_labels
                    if label.casefold() in query_words
                }
                if not queried_labels or queried_labels & region_labels:
                    raise ValueError(f"Absent query does not identify an absent supported label in {frame_id}")
            else:
                raise ValueError(f"Invalid target-presence label in {frame_id}")
        if split == "test_negative" and regions:
            raise ValueError("Negative-test frame must not contain interactive regions")
        if record.get("rights_status") != "PENDING_OWNER_ATTESTATION":
            raise ValueError("Generated corpus must remain pending owner attestation")

    for index, split_a in enumerate(SUPPORTED_SPLITS):
        for split_b in SUPPORTED_SPLITS[index + 1:]:
            if hashes_by_split[split_a] & hashes_by_split[split_b]:
                raise ValueError(f"Duplicate content hash across {split_a} and {split_b}")
            if families_by_split[split_a] & families_by_split[split_b]:
                raise ValueError(f"Layout-family leakage across {split_a} and {split_b}")
    if len(seeds) != len(files):
        raise ValueError("Sample seeds are not unique across splits")

    if verify_generator_sources:
        repo_root = Path(__file__).resolve().parents[3]
        source_records = manifest.get("generator_sources", [])
        required_sources = {
            "avcs/dataset/cleanroom_v1/generator.py",
            "avcs/dataset/cleanroom_v1/glyphs.py",
            "avcs/dataset/cleanroom_v1/dataset.py",
            "avcs/dataset/cleanroom_v1/__init__.py",
        }
        if {source.get("relative_path") for source in source_records} != required_sources:
            raise ValueError("Manifest does not contain the complete clean-room pipeline source inventory")
        for source in source_records:
            if (
                source.get("license_status") != "UNRESOLVED_PENDING_OWNER_REVIEW"
                or source.get("included_in_training_data") is not False
                or not source.get("ownership_basis")
                or not source.get("source_method")
            ):
                raise ValueError(f"Missing pending source provenance: {source.get('relative_path')}")
            path = (repo_root / source["relative_path"]).resolve()
            if not path.is_file() or _sha256(path) != source.get("sha256"):
                raise ValueError(f"Generator source hash mismatch: {source.get('relative_path')}")

    return {
        "manifest_sha256": _sha256(manifest_path),
        "frame_count": len(files),
        "split_counts": {split: counts[split] for split in SUPPORTED_SPLITS},
        "layout_families": {split: sorted(families_by_split[split]) for split in SUPPORTED_SPLITS},
        "image_hash_count": len(image_hashes),
        "annotation_hash_count": len(annotation_hashes),
        "rights_status": manifest["rights"]["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or validate a clean-room System 1 corpus.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/cleanroom_system1_v1"))
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--heldout", type=int, default=16)
    parser.add_argument("--negative", type=int, default=16)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_dataset(args.output_dir)
    else:
        result = generate_dataset(
            args.output_dir,
            seed=args.seed,
            counts={
                "train": args.train,
                "validation": args.validation,
                "test_heldout": args.heldout,
                "test_negative": args.negative,
            },
        )
        result = validate_dataset(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
