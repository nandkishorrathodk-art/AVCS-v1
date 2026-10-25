"""Generator Harmonizer for AVCS Streaming V2.

Part of StreamingVisualCortexV2 Milestone M2: Multi-Generator Dataset Pipeline.

Harmonizes `datasets/ui_grounding` (270 frames) and `datasets/cleanroom_system1_v1` (112 frames)
into a unified Draft-07 manifest with 5 generic UI classes:
    Class 0: Button
    Class 1: Input box
    Class 2: Icon/toggle
    Class 3: Dialog/popup
    Class 4: Text area

Key Responsibilities:
1. Translates coordinate systems to dual normalized bounding boxes:
   `bbox_xyxy_norm` [x1, y1, x2, y2] and `bbox_yxyx_norm` [y1, x1, y2, x2].
2. Synthesizes missing Class 3 (Dialog/popup) and Class 4 (Text area) annotations
   from underlying procedural generator geometry (modal frames, titlebars, text blocks).
3. Enforces leak-free split isolation following cleanroom `LAYOUT_FAMILIES` isolation pattern:
   zero template, session, or image hash overlap between train, val, and test splits.
4. Strips all closed-vocabulary word recognition / text token requirements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from avcs.dataset.unified.schema import (
    CLASS_BUTTON,
    CLASS_DIALOG_POPUP,
    CLASS_ICON_TOGGLE,
    CLASS_INPUT_BOX,
    CLASS_TEXT_AREA,
    GENERATOR_CLEANROOM,
    GENERATOR_UI_GROUNDING,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
    SPLIT_TRAIN,
    SPLIT_VAL,
    UI_CLASS_ID_TO_NAME,
    UnifiedFrame,
    UnifiedManifest,
    UnifiedObject,
    map_legacy_kind_to_class,
    normalize_split_name,
)


def _compute_centroid(box_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate center coordinate (cx, cy) from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box_xyxy
    return round((x1 + x2) / 2.0, 6), round((y1 + y2) / 2.0, 6)


def _xyxy_to_yxyx(box_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] to [y1, x1, y2, x2]."""
    x1, y1, x2, y2 = box_xyxy
    return (y1, x1, y2, x2)


# =========================================================================
# UI Grounding Harmonization & Annotation Synthesis
# =========================================================================
def synthesize_ui_grounding_objects(
    raw_frame: Dict[str, Any],
    frame_id: str,
) -> List[UnifiedObject]:
    """Extract and synthesize 5-class UI objects for a ui_grounding frame."""
    objects: List[UnifiedObject] = []
    category = str(raw_frame.get("category", "desktop_canvas"))
    is_neg = bool(raw_frame.get("is_negative", False))

    if is_neg:
        return objects

    raw_points = raw_frame.get("interactive_points", [])
    btn_boxes: List[Tuple[float, float, float, float]] = []

    # 1. Existing element kinds -> Classes 0, 1, 2
    for idx, pt in enumerate(raw_points):
        obj_id = str(pt.get("element_id") or f"{frame_id}_elem_{idx:02d}")
        kind = str(pt.get("element_kind", "button"))
        class_id, class_name = map_legacy_kind_to_class(kind)

        # ui_grounding stores bbox as [ymin, xmin, ymax, xmax] normalized
        raw_box = pt.get("bbox")
        if raw_box and len(raw_box) == 4:
            ymin, xmin, ymax, xmax = [float(v) for v in raw_box]
            x1 = round(max(0.0, min(1.0, xmin)), 6)
            y1 = round(max(0.0, min(1.0, ymin)), 6)
            x2 = round(max(0.0, min(1.0, xmax)), 6)
            y2 = round(max(0.0, min(1.0, ymax)), 6)
        else:
            cx = float(pt.get("x", 0.5))
            cy = float(pt.get("y", 0.5))
            x1 = round(max(0.0, cx - 0.05), 6)
            x2 = round(min(1.0, cx + 0.05), 6)
            y1 = round(max(0.0, cy - 0.03), 6)
            y2 = round(min(1.0, cy + 0.03), 6)

        box_xyxy = (x1, y1, x2, y2)
        box_yxyx = _xyxy_to_yxyx(box_xyxy)
        centroid = _compute_centroid(box_xyxy)
        action_type = str(pt.get("action_type", "clickable"))
        is_actionable = action_type != "static"

        obj = UnifiedObject(
            object_id=obj_id,
            class_id=class_id,
            class_name=class_name,
            bbox_xyxy_norm=box_xyxy,
            bbox_yxyx_norm=box_yxyx,
            centroid_xy_norm=centroid,
            is_actionable=is_actionable,
            validity_state="VALID",
            metadata={"original_kind": kind},
        )
        objects.append(obj)
        btn_boxes.append(box_xyxy)

    # 2. Synthesize Class 3 (Dialog/popup) for modal_dialog category
    if category == "modal_dialog" and btn_boxes:
        # Enclose modal controls with window padding matching DesktopComposer._compose_modal_dialog
        all_x1 = min(b[0] for b in btn_boxes)
        all_y1 = min(b[1] for b in btn_boxes)
        all_x2 = max(b[2] for b in btn_boxes)
        all_y2 = max(b[3] for b in btn_boxes)

        # Modal window extends above close btn and below action buttons
        dlg_x1 = round(max(0.04, all_x1 - 0.04), 6)
        dlg_y1 = round(max(0.08, all_y1 - 0.03), 6)
        dlg_x2 = round(min(0.96, all_x2 + 0.04), 6)
        dlg_y2 = round(min(0.92, all_y2 + 0.04), 6)

        dlg_box_xyxy = (dlg_x1, dlg_y1, dlg_x2, dlg_y2)
        dlg_obj = UnifiedObject(
            object_id=f"{frame_id}_modal_dialog_container",
            class_id=CLASS_DIALOG_POPUP,
            class_name="dialog_popup",
            bbox_xyxy_norm=dlg_box_xyxy,
            bbox_yxyx_norm=_xyxy_to_yxyx(dlg_box_xyxy),
            centroid_xy_norm=_compute_centroid(dlg_box_xyxy),
            is_actionable=False,  # Container window
            validity_state="VALID",
            metadata={"synthesized_from": "modal_dialog_geometry"},
        )
        objects.append(dlg_obj)

        # 3. Synthesize Class 4 (Text area) for modal prompt text
        # Prompt text sits between titlebar (y1 + 0.08) and action buttons (y2 - 0.12)
        txt_x1 = round(dlg_x1 + 0.04, 6)
        txt_y1 = round(dlg_y1 + 0.12, 6)
        txt_x2 = round(dlg_x2 - 0.04, 6)
        txt_y2 = round(all_y2 - 0.10, 6)
        if txt_y2 > txt_y1:
            txt_box_xyxy = (txt_x1, txt_y1, txt_x2, txt_y2)
            txt_obj = UnifiedObject(
                object_id=f"{frame_id}_modal_prompt_text",
                class_id=CLASS_TEXT_AREA,
                class_name="text_area",
                bbox_xyxy_norm=txt_box_xyxy,
                bbox_yxyx_norm=_xyxy_to_yxyx(txt_box_xyxy),
                centroid_xy_norm=_compute_centroid(txt_box_xyxy),
                is_actionable=False,
                validity_state="VALID",
                metadata={"synthesized_from": "modal_prompt_text"},
            )
            objects.append(txt_obj)

    # 4. Synthesize Class 4 (Text area) for editor_code, terminal_shell, etc.
    elif category == "editor_code":
        # Code lines region between tab strip (~0.28) and find input (~0.88)
        code_box = (0.05, 0.28, 0.95, 0.86)
        objects.append(
            UnifiedObject(
                object_id=f"{frame_id}_editor_code_block",
                class_id=CLASS_TEXT_AREA,
                class_name="text_area",
                bbox_xyxy_norm=code_box,
                bbox_yxyx_norm=_xyxy_to_yxyx(code_box),
                centroid_xy_norm=_compute_centroid(code_box),
                is_actionable=False,
                validity_state="VALID",
                metadata={"synthesized_from": "editor_code_block"},
            )
        )
    elif category == "terminal_shell":
        # Terminal output banner region between tab (~0.18) and prompt line (~0.45)
        term_box = (0.05, 0.18, 0.95, 0.44)
        objects.append(
            UnifiedObject(
                object_id=f"{frame_id}_terminal_output_banner",
                class_id=CLASS_TEXT_AREA,
                class_name="text_area",
                bbox_xyxy_norm=term_box,
                bbox_yxyx_norm=_xyxy_to_yxyx(term_box),
                centroid_xy_norm=_compute_centroid(term_box),
                is_actionable=False,
                validity_state="VALID",
                metadata={"synthesized_from": "terminal_output_banner"},
            )
        )
    elif category in ("browser_webpage", "desktop_canvas", "settings_panel", "form_input", "media_playback"):
        # Synthesize header / status text area
        header_box = (0.15, 0.08, 0.85, 0.18)
        objects.append(
            UnifiedObject(
                object_id=f"{frame_id}_header_text_area",
                class_id=CLASS_TEXT_AREA,
                class_name="text_area",
                bbox_xyxy_norm=header_box,
                bbox_yxyx_norm=_xyxy_to_yxyx(header_box),
                centroid_xy_norm=_compute_centroid(header_box),
                is_actionable=False,
                validity_state="VALID",
                metadata={"synthesized_from": "header_text_area"},
            )
        )

    return objects


def harmonize_ui_grounding(
    manifest_path: Union[str, Path] = "datasets/ui_grounding/manifest.json",
    dataset_root: Optional[Union[str, Path]] = None,
) -> UnifiedManifest:
    """Load and harmonize `datasets/ui_grounding` into 5-class UnifiedManifest."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"ui_grounding manifest not found at {m_path}")

    root = Path(dataset_root) if dataset_root is not None else m_path.parent
    raw_data = json.loads(m_path.read_text(encoding="utf-8"))

    frames: List[UnifiedFrame] = []
    for raw_fr in raw_data.get("frames", []):
        fid = str(raw_fr["frame_id"])
        rel_img = str(raw_fr["file_path"])
        full_img_path = (root / rel_img).as_posix()
        w = int(raw_fr.get("width", 256))
        h = int(raw_fr.get("height", 256))
        split = normalize_split_name(str(raw_fr.get("split", SPLIT_TRAIN)))
        category = str(raw_fr.get("category", "desktop_canvas"))
        session_id = str(raw_fr.get("session_id", ""))
        layout_template_id = f"{category}_{session_id}" if session_id else category
        is_neg = bool(raw_fr.get("is_negative", False)) or split == SPLIT_TEST_NEGATIVE

        if is_neg:
            neg_type = "blank_solid" if "solid" in fid else "blank_desktop"
            objects: List[UnifiedObject] = []
        else:
            neg_type = None
            objects = synthesize_ui_grounding_objects(raw_fr, fid)

        frame = UnifiedFrame(
            frame_id=fid,
            generator_source=GENERATOR_UI_GROUNDING,
            image_path=full_img_path,
            width=w,
            height=h,
            split=split,
            layout_template_id=layout_template_id,
            is_negative_frame=is_neg,
            negative_type=neg_type,
            objects=objects,
            metadata={"category": category, "session_id": session_id},
        )
        frames.append(frame)

    manifest = UnifiedManifest(
        schema_version="2.0.0",
        title="AVCS_Unified_5Class_UI_Manifest",
        metadata={"source": "ui_grounding", "harmonized": True},
        frames=frames,
    )
    manifest.validate()
    return manifest


# =========================================================================
# Cleanroom Harmonization & Annotation Synthesis
# =========================================================================
def synthesize_cleanroom_objects(
    raw_ann: Dict[str, Any],
    frame_id: str,
) -> List[UnifiedObject]:
    """Extract and synthesize 5-class UI objects for a cleanroom frame."""
    objects: List[UnifiedObject] = []
    is_neg = bool(raw_ann.get("is_negative_frame", False))
    family = str(raw_ann.get("layout_family", "toolbar"))

    if is_neg:
        return objects

    # 1. Existing regions -> Classes 0, 1, 2
    for idx, reg in enumerate(raw_ann.get("regions", [])):
        reg_id = str(reg.get("region_id") or f"{frame_id}_reg_{idx:02d}")
        kind = str(reg.get("kind", "button"))
        class_id, class_name = map_legacy_kind_to_class(kind)

        x1, y1, x2, y2 = [float(v) for v in reg["bbox_xyxy_norm"]]
        box_xyxy = (
            round(max(0.0, min(1.0, x1)), 6),
            round(max(0.0, min(1.0, y1)), 6),
            round(max(0.0, min(1.0, x2)), 6),
            round(max(0.0, min(1.0, y2)), 6),
        )
        box_yxyx = _xyxy_to_yxyx(box_xyxy)
        centroid = _compute_centroid(box_xyxy)

        obj = UnifiedObject(
            object_id=reg_id,
            class_id=class_id,
            class_name=class_name,
            bbox_xyxy_norm=box_xyxy,
            bbox_yxyx_norm=box_yxyx,
            centroid_xy_norm=centroid,
            is_actionable=True,
            validity_state="VALID",
            metadata={"original_kind": kind},
        )
        objects.append(obj)

    # 2. Synthesize Class 3 (Dialog/popup) for dialog layout family
    if family == "dialog":
        # In cleanroom generator, the window container for dialog family is (13, 18, 243, 231)
        dlg_box = (
            round(13.0 / 256.0, 6),
            round(18.0 / 256.0, 6),
            round(243.0 / 256.0, 6),
            round(231.0 / 256.0, 6),
        )
        dlg_obj = UnifiedObject(
            object_id=f"{frame_id}_cleanroom_dialog_container",
            class_id=CLASS_DIALOG_POPUP,
            class_name="dialog_popup",
            bbox_xyxy_norm=dlg_box,
            bbox_yxyx_norm=_xyxy_to_yxyx(dlg_box),
            centroid_xy_norm=_compute_centroid(dlg_box),
            is_actionable=False,
            validity_state="VALID",
            metadata={"synthesized_from": "cleanroom_dialog_family"},
        )
        objects.append(dlg_obj)

    # 3. Synthesize Class 4 (Text area) for titlebar text and taskbar text
    # Cleanroom titlebar "WORKSPACE" text at (48, 26, 118, 38)
    title_box = (
        round(48.0 / 256.0, 6),
        round(26.0 / 256.0, 6),
        round(118.0 / 256.0, 6),
        round(38.0 / 256.0, 6),
    )
    objects.append(
        UnifiedObject(
            object_id=f"{frame_id}_titlebar_text",
            class_id=CLASS_TEXT_AREA,
            class_name="text_area",
            bbox_xyxy_norm=title_box,
            bbox_yxyx_norm=_xyxy_to_yxyx(title_box),
            centroid_xy_norm=_compute_centroid(title_box),
            is_actionable=False,
            validity_state="VALID",
            metadata={"synthesized_from": "cleanroom_workspace_title"},
        )
    )

    # Unknown word text block if present
    unknown_info = raw_ann.get("unknown_word")
    if unknown_info and "bbox_xyxy_px" in unknown_info:
        px = unknown_info["bbox_xyxy_px"]
        unk_box = (
            round(float(px[0]) / 256.0, 6),
            round(float(px[1]) / 256.0, 6),
            round(float(px[2]) / 256.0, 6),
            round(float(px[3]) / 256.0, 6),
        )
        objects.append(
            UnifiedObject(
                object_id=f"{frame_id}_unknown_word_text",
                class_id=CLASS_TEXT_AREA,
                class_name="text_area",
                bbox_xyxy_norm=unk_box,
                bbox_yxyx_norm=_xyxy_to_yxyx(unk_box),
                centroid_xy_norm=_compute_centroid(unk_box),
                is_actionable=False,
                validity_state="VALID",
                metadata={"synthesized_from": "cleanroom_unknown_word"},
            )
        )

    return objects


def harmonize_cleanroom(
    corpus_dir: Union[str, Path] = "datasets/cleanroom_system1_v1",
) -> UnifiedManifest:
    """Load and harmonize `datasets/cleanroom_system1_v1` into 5-class UnifiedManifest."""
    c_dir = Path(corpus_dir)
    m_path = c_dir / "manifest.json"
    if not m_path.exists():
        raise FileNotFoundError(f"Cleanroom manifest not found at {m_path}")

    raw_manifest = json.loads(m_path.read_text(encoding="utf-8"))
    frames: List[UnifiedFrame] = []

    for file_entry in raw_manifest.get("files", []):
        img_rel = file_entry["image"]["relative_path"]
        ann_rel = file_entry["annotation"]["relative_path"]
        full_img_path = (c_dir / img_rel).as_posix()
        full_ann_path = c_dir / ann_rel

        if not full_ann_path.exists():
            continue

        raw_ann = json.loads(full_ann_path.read_text(encoding="utf-8"))
        fid = str(raw_ann.get("frame_id", file_entry["image"]["sha256"][:12]))
        split = normalize_split_name(str(file_entry.get("split", SPLIT_TRAIN)))
        family = str(raw_ann.get("layout_family", "toolbar"))
        is_neg = bool(raw_ann.get("is_negative_frame", False)) or split == SPLIT_TEST_NEGATIVE

        if is_neg:
            neg_type = "empty_desktop"
            objects: List[UnifiedObject] = []
        else:
            neg_type = None
            objects = synthesize_cleanroom_objects(raw_ann, fid)

        frame = UnifiedFrame(
            frame_id=fid,
            generator_source=GENERATOR_CLEANROOM,
            image_path=full_img_path,
            width=int(raw_ann.get("width", 256)),
            height=int(raw_ann.get("height", 256)),
            split=split,
            layout_template_id=family,
            is_negative_frame=is_neg,
            negative_type=neg_type,
            objects=objects,
            metadata={"layout_family": family, "category": raw_ann.get("category", "")},
        )
        frames.append(frame)

    manifest = UnifiedManifest(
        schema_version="2.0.0",
        title="AVCS_Unified_5Class_UI_Manifest",
        metadata={"source": "cleanroom_system1_v1", "harmonized": True},
        frames=frames,
    )
    manifest.validate()
    return manifest


# =========================================================================
# Multi-Generator Combined Harmonization & Leak-Free Split Verification
# =========================================================================
def harmonize_all_generators(
    ui_manifest_path: Union[str, Path] = "datasets/ui_grounding/manifest.json",
    cleanroom_dir: Union[str, Path] = "datasets/cleanroom_system1_v1",
) -> UnifiedManifest:
    """Harmonize all generators into a single unified 5-class manifest."""
    ui_manifest = harmonize_ui_grounding(ui_manifest_path)
    cleanroom_manifest = harmonize_cleanroom(cleanroom_dir)

    combined_frames = list(ui_manifest.frames) + list(cleanroom_manifest.frames)
    combined = UnifiedManifest(
        schema_version="2.0.0",
        title="AVCS_Unified_5Class_UI_Manifest",
        metadata={
            "sources": [GENERATOR_UI_GROUNDING, GENERATOR_CLEANROOM],
            "ui_frame_count": len(ui_manifest.frames),
            "cleanroom_frame_count": len(cleanroom_manifest.frames),
            "total_frame_count": len(combined_frames),
        },
        frames=combined_frames,
    )
    combined.validate()
    return combined


def verify_leak_free_split_isolation(manifest: UnifiedManifest) -> Dict[str, Any]:
    """Verify that templates, sessions, and image hashes in test_held_out are disjoint from train.

    Returns:
        Dictionary with leak-free status and detailed set intersection audits.
    """
    train_frames = manifest.filter_split(SPLIT_TRAIN).frames
    val_frames = manifest.filter_split(SPLIT_VAL).frames
    test_frames = manifest.filter_split(SPLIT_TEST_HELD_OUT).frames

    # 1. Cleanroom layout families
    cr_train_templates = {
        f.layout_template_id
        for f in train_frames
        if f.generator_source == GENERATOR_CLEANROOM and f.layout_template_id
    }
    cr_val_templates = {
        f.layout_template_id
        for f in val_frames
        if f.generator_source == GENERATOR_CLEANROOM and f.layout_template_id
    }
    cr_test_templates = {
        f.layout_template_id
        for f in test_frames
        if f.generator_source == GENERATOR_CLEANROOM and f.layout_template_id
    }

    cr_train_test_overlap = cr_train_templates.intersection(cr_test_templates)
    cr_train_val_overlap = cr_train_templates.intersection(cr_val_templates)

    # 2. UI Grounding sessions
    ui_train_sessions = {
        f.metadata.get("session_id", "")
        for f in train_frames
        if f.generator_source == GENERATOR_UI_GROUNDING and f.metadata.get("session_id")
    }
    ui_val_sessions = {
        f.metadata.get("session_id", "")
        for f in val_frames
        if f.generator_source == GENERATOR_UI_GROUNDING and f.metadata.get("session_id")
    }
    ui_test_sessions = {
        f.metadata.get("session_id", "")
        for f in test_frames
        if f.generator_source == GENERATOR_UI_GROUNDING and f.metadata.get("session_id")
    }

    ui_train_test_overlap = ui_train_sessions.intersection(ui_test_sessions)
    ui_train_val_overlap = ui_train_sessions.intersection(ui_val_sessions)

    # 3. Image file hashes
    def get_hashes(frames: List[UnifiedFrame]) -> Set[str]:
        hashes = set()
        for f in frames:
            p = Path(f.image_path)
            if p.exists():
                hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
            else:
                hashes.add(f.frame_id)
        return hashes

    train_hashes = get_hashes(train_frames)
    val_hashes = get_hashes(val_frames)
    test_hashes = get_hashes(test_frames)

    hash_train_test_overlap = train_hashes.intersection(test_hashes)
    hash_train_val_overlap = train_hashes.intersection(val_hashes)

    is_leak_free = (
        len(cr_train_test_overlap) == 0
        and len(cr_train_val_overlap) == 0
        and len(ui_train_test_overlap) == 0
        and len(ui_train_val_overlap) == 0
        and len(hash_train_test_overlap) == 0
        and len(hash_train_val_overlap) == 0
    )

    return {
        "is_leak_free": is_leak_free,
        "cleanroom_train_templates": sorted(cr_train_templates),
        "cleanroom_val_templates": sorted(cr_val_templates),
        "cleanroom_test_templates": sorted(cr_test_templates),
        "cleanroom_train_test_overlap": sorted(cr_train_test_overlap),
        "ui_train_sessions": sorted(ui_train_sessions),
        "ui_test_sessions": sorted(ui_test_sessions),
        "ui_train_test_overlap": sorted(ui_train_test_overlap),
        "hash_train_test_overlap_count": len(hash_train_test_overlap),
        "hash_train_val_overlap_count": len(hash_train_val_overlap),
    }
