"""Draft-07 Compliant Schema and Data Structures for Unified 5-Class UI Objectness.

Part of StreamingVisualCortexV2 Milestone M2: Multi-Generator Dataset Pipeline.

Provides typed Python dataclasses, dual bounding box representations,
coordinate validation, and pure-Python Draft-07 JSON schema validation
for 5 generic UI morphology classes without text token dependencies:
    Class 0: Button
    Class 1: Input box
    Class 2: Icon/toggle
    Class 3: Dialog/popup
    Class 4: Text area
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

# =========================================================================
# Generic 5 UI Classes
# =========================================================================
CLASS_BUTTON: int = 0
CLASS_INPUT_BOX: int = 1
CLASS_ICON_TOGGLE: int = 2
CLASS_DIALOG_POPUP: int = 3
CLASS_TEXT_AREA: int = 4

NUM_CLASSES: int = 5

UI_CLASS_NAMES: Tuple[str, ...] = (
    "button",
    "input_box",
    "icon_toggle",
    "dialog_popup",
    "text_area",
)

UI_CLASS_ID_TO_NAME: Dict[int, str] = {
    CLASS_BUTTON: "button",
    CLASS_INPUT_BOX: "input_box",
    CLASS_ICON_TOGGLE: "icon_toggle",
    CLASS_DIALOG_POPUP: "dialog_popup",
    CLASS_TEXT_AREA: "text_area",
}

UI_CLASS_NAME_TO_ID: Dict[str, int] = {
    "button": CLASS_BUTTON,
    "input_box": CLASS_INPUT_BOX,
    "icon_toggle": CLASS_ICON_TOGGLE,
    "dialog_popup": CLASS_DIALOG_POPUP,
    "text_area": CLASS_TEXT_AREA,
}

# =========================================================================
# Canonical Generator & Split Identifiers
# =========================================================================
GENERATOR_UI_GROUNDING: str = "ui_grounding"
GENERATOR_CLEANROOM: str = "cleanroom_system1_v1"
SUPPORTED_GENERATORS: Tuple[str, ...] = (GENERATOR_UI_GROUNDING, GENERATOR_CLEANROOM)

SPLIT_TRAIN: str = "train"
SPLIT_VAL: str = "val"
SPLIT_TEST_HELD_OUT: str = "test_held_out"
SPLIT_TEST_NEGATIVE: str = "test_negative"
CANONICAL_SPLITS: Tuple[str, ...] = (
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
)

VALIDITY_STATES: Tuple[str, ...] = ("VALID", "UNCERTAIN", "INVALID", "REJECTED")
COORDINATE_CONVENTION: str = "normalized_xyxy_origin_top_left"


def normalize_split_name(split: str) -> str:
    """Normalize split variations to canonical naming."""
    s = split.strip().lower()
    if s in ("val", "validation"):
        return SPLIT_VAL
    if s in ("test_held_out", "test_heldout", "held_out"):
        return SPLIT_TEST_HELD_OUT
    if s in ("test_negative", "negative"):
        return SPLIT_TEST_NEGATIVE
    if s == "train":
        return SPLIT_TRAIN
    return s


def map_legacy_kind_to_class(kind: str) -> Tuple[int, str]:
    """Map legacy generator kinds (button, input_field, tab, icon, etc.) to 5 generic UI classes."""
    k = kind.strip().lower()
    if k in ("button", "btn", "tab"):
        return CLASS_BUTTON, "button"
    if k in ("input_field", "input", "textfield", "textbox", "search_box"):
        return CLASS_INPUT_BOX, "input_box"
    if k in ("icon", "toggle", "checkbox", "radio", "switch"):
        return CLASS_ICON_TOGGLE, "icon_toggle"
    if k in ("dialog", "popup", "modal", "modal_dialog", "window", "dialog_popup"):
        return CLASS_DIALOG_POPUP, "dialog_popup"
    if k in ("text", "text_area", "textarea", "label", "header", "code", "terminal_output"):
        return CLASS_TEXT_AREA, "text_area"
    # Fallback to button for unknown clickable, else text_area
    return CLASS_BUTTON, "button"


# =========================================================================
# Draft-07 JSON Schema Specification
# =========================================================================
DRAFT_07_UNIFIED_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AVCS_Unified_5Class_UI_Manifest",
    "version": "2.0.0",
    "type": "object",
    "required": ["$schema", "title", "version", "coordinate_convention", "frames"],
    "properties": {
        "$schema": {"type": "string"},
        "title": {"type": "string"},
        "version": {"type": "string"},
        "coordinate_convention": {"type": "string"},
        "metadata": {"type": "object"},
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "frame_id",
                    "generator_source",
                    "image_path",
                    "width",
                    "height",
                    "split",
                    "is_negative_frame",
                    "objects",
                ],
                "properties": {
                    "frame_id": {"type": "string"},
                    "generator_source": {
                        "type": "string",
                        "enum": ["ui_grounding", "cleanroom_system1_v1", "synthetic_temporal"],
                    },
                    "image_path": {"type": "string"},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1},
                    "split": {
                        "type": "string",
                        "enum": ["train", "val", "test_held_out", "test_negative"],
                    },
                    "layout_template_id": {"type": ["string", "null"]},
                    "is_negative_frame": {"type": "boolean"},
                    "negative_type": {"type": ["string", "null"]},
                    "metadata": {"type": "object"},
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "object_id",
                                "class_id",
                                "class_name",
                                "bbox_xyxy_norm",
                                "bbox_yxyx_norm",
                                "centroid_xy_norm",
                                "is_actionable",
                                "validity_state",
                            ],
                            "properties": {
                                "object_id": {"type": "string"},
                                "class_id": {"type": "integer", "minimum": 0, "maximum": 4},
                                "class_name": {
                                    "type": "string",
                                    "enum": [
                                        "button",
                                        "input_box",
                                        "icon_toggle",
                                        "dialog_popup",
                                        "text_area",
                                    ],
                                },
                                "bbox_xyxy_norm": {
                                    "type": "array",
                                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                "bbox_yxyx_norm": {
                                    "type": "array",
                                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                "centroid_xy_norm": {
                                    "type": "array",
                                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "is_actionable": {"type": "boolean"},
                                "validity_state": {
                                    "type": "string",
                                    "enum": ["VALID", "UNCERTAIN", "INVALID", "REJECTED"],
                                },
                                "metadata": {"type": "object"},
                            },
                        },
                    },
                },
            },
        },
    },
}


# =========================================================================
# Dataclasses
# =========================================================================
@dataclass
class UnifiedObject:
    """A generic UI object annotation with dual bounding boxes and actionability."""

    object_id: str
    class_id: int
    class_name: str
    bbox_xyxy_norm: Tuple[float, float, float, float]
    bbox_yxyx_norm: Tuple[float, float, float, float]
    centroid_xy_norm: Tuple[float, float]
    is_actionable: bool = True
    validity_state: str = "VALID"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.bbox_xyxy_norm, (list, Sequence)):
            self.bbox_xyxy_norm = (
                float(self.bbox_xyxy_norm[0]),
                float(self.bbox_xyxy_norm[1]),
                float(self.bbox_xyxy_norm[2]),
                float(self.bbox_xyxy_norm[3]),
            )
        if isinstance(self.bbox_yxyx_norm, (list, Sequence)):
            self.bbox_yxyx_norm = (
                float(self.bbox_yxyx_norm[0]),
                float(self.bbox_yxyx_norm[1]),
                float(self.bbox_yxyx_norm[2]),
                float(self.bbox_yxyx_norm[3]),
            )
        if isinstance(self.centroid_xy_norm, (list, Sequence)):
            self.centroid_xy_norm = (
                float(self.centroid_xy_norm[0]),
                float(self.centroid_xy_norm[1]),
            )
        if self.class_id not in UI_CLASS_ID_TO_NAME:
            raise ValueError(f"Invalid class_id {self.class_id}; must be in range 0..4")
        expected_name = UI_CLASS_ID_TO_NAME[self.class_id]
        if self.class_name != expected_name:
            self.class_name = expected_name
        if self.validity_state not in VALIDITY_STATES:
            raise ValueError(
                f"Invalid validity_state {self.validity_state}; must be one of {VALIDITY_STATES}"
            )

    def validate_coordinates(self, tolerance: float = 1e-4) -> None:
        """Validate bounding box and centroid consistency."""
        x1, y1, x2, y2 = self.bbox_xyxy_norm
        y1_c, x1_c, y2_c, x2_c = self.bbox_yxyx_norm

        for v, name in [(x1, "x1"), (y1, "y1"), (x2, "x2"), (y2, "y2")]:
            if not (-tolerance <= v <= 1.0 + tolerance):
                raise ValueError(
                    f"Coordinate {name}={v} out of bounds [0.0, 1.0] in object {self.object_id}"
                )

        if x1 > x2 + tolerance:
            raise ValueError(
                f"Inverted x coords: x1 ({x1}) > x2 ({x2}) in object {self.object_id}"
            )
        if y1 > y2 + tolerance:
            raise ValueError(
                f"Inverted y coords: y1 ({y1}) > y2 ({y2}) in object {self.object_id}"
            )

        # Dual coordinate consistency
        if abs(x1 - x1_c) > tolerance or abs(x2 - x2_c) > tolerance:
            raise ValueError(
                f"Dual box X mismatch: xyxy={self.bbox_xyxy_norm} vs yxyx={self.bbox_yxyx_norm}"
            )
        if abs(y1 - y1_c) > tolerance or abs(y2 - y2_c) > tolerance:
            raise ValueError(
                f"Dual box Y mismatch: xyxy={self.bbox_xyxy_norm} vs yxyx={self.bbox_yxyx_norm}"
            )

        cx, cy = self.centroid_xy_norm
        if not (-tolerance <= cx <= 1.0 + tolerance and -tolerance <= cy <= 1.0 + tolerance):
            raise ValueError(
                f"Centroid ({cx}, {cy}) out of bounds [0, 1] in object {self.object_id}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert object annotation to Draft-07 compatible dictionary."""
        return {
            "object_id": self.object_id,
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "bbox_xyxy_norm": [round(float(v), 6) for v in self.bbox_xyxy_norm],
            "bbox_yxyx_norm": [round(float(v), 6) for v in self.bbox_yxyx_norm],
            "centroid_xy_norm": [round(float(v), 6) for v in self.centroid_xy_norm],
            "is_actionable": bool(self.is_actionable),
            "validity_state": self.validity_state,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedObject:
        """Create UnifiedObject from dictionary."""
        cid = int(data["class_id"])
        cname = str(data.get("class_name", UI_CLASS_ID_TO_NAME.get(cid, "button")))
        return cls(
            object_id=str(data["object_id"]),
            class_id=cid,
            class_name=cname,
            bbox_xyxy_norm=tuple(data["bbox_xyxy_norm"]),  # type: ignore
            bbox_yxyx_norm=tuple(data["bbox_yxyx_norm"]),  # type: ignore
            centroid_xy_norm=tuple(data["centroid_xy_norm"]),  # type: ignore
            is_actionable=bool(data.get("is_actionable", True)),
            validity_state=str(data.get("validity_state", "VALID")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class UnifiedFrame:
    """Frame-level annotation with image path, dimensions, split, and objects."""

    frame_id: str
    generator_source: str
    image_path: str
    width: int = 256
    height: int = 256
    split: str = SPLIT_TRAIN
    layout_template_id: Optional[str] = None
    is_negative_frame: bool = False
    negative_type: Optional[str] = None
    objects: List[UnifiedObject] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.split = normalize_split_name(self.split)

    def validate(self) -> None:
        """Validate frame-level invariants and all contained objects."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid frame dimensions ({self.width}x{self.height}) in {self.frame_id}")

        if self.is_negative_frame:
            if len(self.objects) > 0:
                raise ValueError(
                    f"Negative frame {self.frame_id} must not contain objects (found {len(self.objects)})"
                )
            if self.negative_type is None:
                raise ValueError(f"Negative frame {self.frame_id} must declare negative_type")

        for obj in self.objects:
            obj.validate_coordinates()

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame annotation to dictionary."""
        return {
            "frame_id": self.frame_id,
            "generator_source": self.generator_source,
            "image_path": self.image_path,
            "width": int(self.width),
            "height": int(self.height),
            "split": self.split,
            "layout_template_id": self.layout_template_id,
            "is_negative_frame": bool(self.is_negative_frame),
            "negative_type": self.negative_type,
            "objects": [obj.to_dict() for obj in self.objects],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedFrame:
        """Create UnifiedFrame from dictionary."""
        objects = [UnifiedObject.from_dict(o) for o in data.get("objects", [])]
        return cls(
            frame_id=str(data["frame_id"]),
            generator_source=str(data["generator_source"]),
            image_path=str(data["image_path"]),
            width=int(data.get("width", 256)),
            height=int(data.get("height", 256)),
            split=str(data.get("split", SPLIT_TRAIN)),
            layout_template_id=data.get("layout_template_id"),
            is_negative_frame=bool(data.get("is_negative_frame", False)),
            negative_type=data.get("negative_type"),
            objects=objects,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class UnifiedManifest:
    """Unified Manifest aggregating frames across generators."""

    schema_version: str = "2.0.0"
    title: str = "AVCS_Unified_5Class_UI_Manifest"
    coordinate_convention: str = COORDINATE_CONVENTION
    metadata: Dict[str, Any] = field(default_factory=dict)
    frames: List[UnifiedFrame] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> UnifiedFrame:
        return self.frames[index]

    def validate(self) -> None:
        """Validate entire manifest."""
        seen = set()
        for f in self.frames:
            if f.frame_id in seen:
                raise ValueError(f"Duplicate frame_id {f.frame_id} in manifest")
            seen.add(f.frame_id)
            f.validate()

    def filter_split(self, split: str) -> UnifiedManifest:
        """Filter manifest by split."""
        target = normalize_split_name(split)
        return UnifiedManifest(
            schema_version=self.schema_version,
            title=self.title,
            coordinate_convention=self.coordinate_convention,
            metadata=dict(self.metadata),
            frames=[f for f in self.frames if f.split == target],
        )

    def filter_generator(self, generator_source: str) -> UnifiedManifest:
        """Filter manifest by generator source."""
        target = generator_source.strip().lower()
        return UnifiedManifest(
            schema_version=self.schema_version,
            title=self.title,
            coordinate_convention=self.coordinate_convention,
            metadata=dict(self.metadata),
            frames=[f for f in self.frames if f.generator_source.lower() == target],
        )

    def get_class_counts(self) -> Dict[str, int]:
        """Return counts of each UI class across positive objects."""
        counts = {name: 0 for name in UI_CLASS_NAMES}
        for f in self.frames:
            for obj in f.objects:
                name = UI_CLASS_ID_TO_NAME.get(obj.class_id, "unknown")
                counts[name] = counts.get(name, 0) + 1
        return counts

    def get_split_counts(self) -> Dict[str, int]:
        """Return counts per split."""
        counts: Dict[str, int] = {}
        for f in self.frames:
            counts[f.split] = counts.get(f.split, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to Draft-07 compliant dictionary."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.title,
            "version": self.schema_version,
            "coordinate_convention": self.coordinate_convention,
            "metadata": dict(self.metadata),
            "frames": [f.to_dict() for f in self.frames],
        }

    def to_json(self, target: Union[str, Path, TextIO], indent: int = 2) -> None:
        """Serialize manifest to JSON file or stream."""
        data = self.to_dict()
        if hasattr(target, "write"):
            json.dump(data, target, indent=indent)  # type: ignore
        else:
            p = Path(target)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedManifest:
        """Create UnifiedManifest from dictionary."""
        frames = [UnifiedFrame.from_dict(f) for f in data.get("frames", [])]
        return cls(
            schema_version=str(data.get("version", "2.0.0")),
            title=str(data.get("title", "AVCS_Unified_5Class_UI_Manifest")),
            coordinate_convention=str(
                data.get("coordinate_convention", COORDINATE_CONVENTION)
            ),
            metadata=dict(data.get("metadata", {})),
            frames=frames,
        )

    @classmethod
    def from_json(cls, source: Union[str, Path, TextIO]) -> UnifiedManifest:
        """Load UnifiedManifest from JSON file or stream."""
        if hasattr(source, "read"):
            data = json.load(source)  # type: ignore
        else:
            p = Path(source)
            data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# =========================================================================
# Pure-Python Draft-07 Validator
# =========================================================================
def validate_unified_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate data dictionary against Draft-07 unified schema without external deps.

    Returns:
        (is_valid, list_of_error_strings)
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["Root document must be a JSON object"]

    for req in ("$schema", "title", "version", "coordinate_convention", "frames"):
        if req not in data:
            errors.append(f"Missing required manifest field: {req!r}")

    frames = data.get("frames")
    if not isinstance(frames, list):
        errors.append("Field 'frames' must be a list")
        return False, errors

    seen_ids = set()
    for idx, f in enumerate(frames):
        if not isinstance(f, dict):
            errors.append(f"Frame at index {idx} must be an object")
            continue

        fid = f.get("frame_id")
        if not fid or not isinstance(fid, str):
            errors.append(f"Frame at index {idx} missing valid 'frame_id'")
        elif fid in seen_ids:
            errors.append(f"Duplicate frame_id {fid!r}")
        else:
            seen_ids.add(fid)

        for req_f in (
            "generator_source",
            "image_path",
            "width",
            "height",
            "split",
            "is_negative_frame",
            "objects",
        ):
            if req_f not in f:
                errors.append(f"Frame {fid or idx} missing required field {req_f!r}")

        is_neg = bool(f.get("is_negative_frame", False))
        objects = f.get("objects", [])
        if not isinstance(objects, list):
            errors.append(f"Frame {fid or idx} 'objects' must be a list")
            continue

        if is_neg:
            if len(objects) > 0:
                errors.append(f"Negative frame {fid} must have 0 objects (found {len(objects)})")
            if not f.get("negative_type"):
                errors.append(f"Negative frame {fid} missing 'negative_type'")

        for obj_idx, obj in enumerate(objects):
            if not isinstance(obj, dict):
                errors.append(f"Frame {fid} object {obj_idx} must be an object")
                continue

            for req_obj in (
                "object_id",
                "class_id",
                "class_name",
                "bbox_xyxy_norm",
                "bbox_yxyx_norm",
                "centroid_xy_norm",
                "is_actionable",
                "validity_state",
            ):
                if req_obj not in obj:
                    errors.append(f"Frame {fid} object {obj_idx} missing required field {req_obj!r}")

            cid = obj.get("class_id")
            cname = obj.get("class_name")
            if not isinstance(cid, int) or not (0 <= cid <= 4):
                errors.append(f"Frame {fid} object {obj_idx} invalid class_id: {cid}")
            elif cname != UI_CLASS_ID_TO_NAME.get(cid):
                errors.append(
                    f"Frame {fid} object {obj_idx} class_name {cname!r} does not match class_id {cid}"
                )

            # Coordinates
            xyxy = obj.get("bbox_xyxy_norm")
            yxyx = obj.get("bbox_yxyx_norm")
            centroid = obj.get("centroid_xy_norm")

            if not (isinstance(xyxy, list) and len(xyxy) == 4):
                errors.append(f"Frame {fid} object {obj_idx} bbox_xyxy_norm must have 4 numbers")
            elif not (isinstance(yxyx, list) and len(yxyx) == 4):
                errors.append(f"Frame {fid} object {obj_idx} bbox_yxyx_norm must have 4 numbers")
            else:
                x1, y1, x2, y2 = xyxy
                y1_c, x1_c, y2_c, x2_c = yxyx
                tol = 1e-4
                if not all(isinstance(v, (int, float)) and -tol <= v <= 1.0 + tol for v in (x1, y1, x2, y2)):
                    errors.append(f"Frame {fid} object {obj_idx} bbox_xyxy_norm out of bounds [0, 1]")
                if x1 > x2 + tol or y1 > y2 + tol:
                    errors.append(f"Frame {fid} object {obj_idx} inverted bbox_xyxy_norm: {xyxy}")
                if abs(x1 - x1_c) > tol or abs(x2 - x2_c) > tol or abs(y1 - y1_c) > tol or abs(y2 - y2_c) > tol:
                    errors.append(f"Frame {fid} object {obj_idx} dual box mismatch: {xyxy} vs {yxyx}")

            if not (isinstance(centroid, list) and len(centroid) == 2):
                errors.append(f"Frame {fid} object {obj_idx} centroid_xy_norm must have 2 numbers")
            else:
                cx, cy = centroid
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    errors.append(f"Frame {fid} object {obj_idx} centroid out of bounds: {centroid}")

    return len(errors) == 0, errors
