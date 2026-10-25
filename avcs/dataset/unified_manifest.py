"""Unified Annotation Schema and Loaders for AVCS UI Objectness and Grounding.

Standardizes annotations across multiple UI screen generators (including
`datasets/ui_grounding` and `datasets/cleanroom_system1_v1`) into a single
unambiguous Draft-07 JSON manifest specification.

Key Design Elements:
- Dual bounding boxes (`bbox_xyxy_norm` and `bbox_yxyx_norm`) to eliminate coordinate
  inversion traps across computer vision and NLP conventions.
- Standardized centroids, actionability flags (`is_actionable`), objectness labels,
  validity states, and explicit negative frame annotations.
- Full Draft-07 schema compliance with zero external dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

# Canonical Generator Identifiers
GENERATOR_UI_GROUNDING = "ui_grounding"
GENERATOR_CLEANROOM = "cleanroom_system1_v1"

# Canonical Split Identifiers
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST_HELD_OUT = "test_held_out"
SPLIT_TEST_NEGATIVE = "test_negative"
CANONICAL_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST_HELD_OUT, SPLIT_TEST_NEGATIVE)

# Coordinate and Actionability Enums
COORDINATE_CONVENTION = "normalized_xyxy_origin_top_left"
VALIDITY_STATES = ("VALID", "INVALID", "UNCERTAIN", "REJECTED")
OBJECTNESS_LABELS = ("clickable", "typeable", "scrollable", "static")

# Draft-07 JSON Schema Specification
DRAFT_07_UNIFIED_MANIFEST_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AVCS_Unified_UI_Objectness_Manifest",
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
                    "generator_source": {"type": "string"},
                    "image_path": {"type": "string"},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1},
                    "split": {
                        "type": "string",
                        "enum": [
                            "train",
                            "val",
                            "validation",
                            "test_held_out",
                            "test_heldout",
                            "test_negative",
                        ],
                    },
                    "layout_template_id": {"type": ["string", "null"]},
                    "theme_name": {"type": ["string", "null"]},
                    "is_dark_mode": {"type": "boolean"},
                    "is_negative_frame": {"type": "boolean"},
                    "negative_type": {"type": ["string", "null"]},
                    "negative_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "metadata": {"type": "object"},
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "object_id",
                                "bbox_xyxy_norm",
                                "bbox_yxyx_norm",
                                "centroid_xy_norm",
                                "is_actionable",
                                "objectness_label",
                                "validity_state",
                                "element_kind",
                            ],
                            "properties": {
                                "object_id": {"type": "string"},
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
                                "objectness_label": {"type": "string"},
                                "validity_state": {
                                    "type": "string",
                                    "enum": ["VALID", "INVALID", "UNCERTAIN", "REJECTED"],
                                },
                                "element_kind": {"type": "string"},
                                "text_label": {"type": ["string", "null"]},
                                "target_query": {"type": ["string", "null"]},
                                "query_variations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


def _normalize_split_name(split: str) -> str:
    """Map generator-specific split names to canonical names."""
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


@dataclass
class UnifiedObjectAnnotation:
    """Annotation for a single UI element on a frame."""

    object_id: str
    bbox_xyxy_norm: Tuple[float, float, float, float]
    bbox_yxyx_norm: Tuple[float, float, float, float]
    centroid_xy_norm: Tuple[float, float]
    is_actionable: bool = True
    objectness_label: str = "clickable"
    validity_state: str = "VALID"
    element_kind: str = "button"
    text_label: Optional[str] = None
    target_query: Optional[str] = None
    query_variations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Convert list inputs to tuples if necessary
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

    def validate_coordinates(self, tolerance: float = 1e-4) -> None:
        """Validate bounding box and centroid coordinate consistency."""
        x1, y1, x2, y2 = self.bbox_xyxy_norm
        y1_check, x1_check, y2_check, x2_check = self.bbox_yxyx_norm

        for v, name in [(x1, "x1"), (y1, "y1"), (x2, "x2"), (y2, "y2")]:
            if not (-tolerance <= v <= 1.0 + tolerance):
                raise ValueError(f"Coordinate {name}={v} out of bounds [0.0, 1.0] for {self.object_id}")

        if x1 > x2 + tolerance:
            raise ValueError(f"Inverted x-coordinates: x1 ({x1}) > x2 ({x2}) for {self.object_id}")
        if y1 > y2 + tolerance:
            raise ValueError(f"Inverted y-coordinates: y1 ({y1}) > y2 ({y2}) for {self.object_id}")

        # Assert dual box consistency
        if abs(x1 - x1_check) > tolerance or abs(x2 - x2_check) > tolerance:
            raise ValueError(
                f"Dual box X mismatch: bbox_xyxy={self.bbox_xyxy_norm} vs bbox_yxyx={self.bbox_yxyx_norm}"
            )
        if abs(y1 - y1_check) > tolerance or abs(y2 - y2_check) > tolerance:
            raise ValueError(
                f"Dual box Y mismatch: bbox_xyxy={self.bbox_xyxy_norm} vs bbox_yxyx={self.bbox_yxyx_norm}"
            )

        cx, cy = self.centroid_xy_norm
        if not (-tolerance <= cx <= 1.0 + tolerance and -tolerance <= cy <= 1.0 + tolerance):
            raise ValueError(f"Centroid ({cx}, {cy}) out of bounds for {self.object_id}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert object annotation to dictionary."""
        return {
            "object_id": self.object_id,
            "bbox_xyxy_norm": [round(v, 6) for v in self.bbox_xyxy_norm],
            "bbox_yxyx_norm": [round(v, 6) for v in self.bbox_yxyx_norm],
            "centroid_xy_norm": [round(v, 6) for v in self.centroid_xy_norm],
            "is_actionable": self.is_actionable,
            "objectness_label": self.objectness_label,
            "validity_state": self.validity_state,
            "element_kind": self.element_kind,
            "text_label": self.text_label,
            "target_query": self.target_query,
            "query_variations": list(self.query_variations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedObjectAnnotation:
        """Instantiate an object annotation from dictionary."""
        return cls(
            object_id=str(data["object_id"]),
            bbox_xyxy_norm=tuple(data["bbox_xyxy_norm"]),  # type: ignore
            bbox_yxyx_norm=tuple(data["bbox_yxyx_norm"]),  # type: ignore
            centroid_xy_norm=tuple(data["centroid_xy_norm"]),  # type: ignore
            is_actionable=bool(data.get("is_actionable", True)),
            objectness_label=str(data.get("objectness_label", "clickable")),
            validity_state=str(data.get("validity_state", "VALID")),
            element_kind=str(data.get("element_kind", "button")),
            text_label=data.get("text_label"),
            target_query=data.get("target_query"),
            query_variations=list(data.get("query_variations", [])),
        )


@dataclass
class UnifiedAnnotationFrame:
    """Annotation for a single desktop frame."""

    frame_id: str
    generator_source: str
    image_path: str
    width: int = 256
    height: int = 256
    split: str = SPLIT_TRAIN
    layout_template_id: Optional[str] = None
    theme_name: Optional[str] = None
    is_dark_mode: bool = False
    is_negative_frame: bool = False
    negative_type: Optional[str] = None
    objects: List[UnifiedObjectAnnotation] = field(default_factory=list)
    negative_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.split = _normalize_split_name(self.split)

    def validate(self) -> None:
        """Validate frame-level invariants and all contained objects."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid frame dimensions ({self.width}x{self.height}) in {self.frame_id}")

        if self.is_negative_frame:
            if len(self.objects) > 0:
                raise ValueError(
                    f"Negative frame {self.frame_id} must not contain any object annotations (found {len(self.objects)})"
                )
            if self.negative_type is None:
                raise ValueError(f"Negative frame {self.frame_id} must declare a negative_type")

        for obj in self.objects:
            obj.validate_coordinates()

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame annotation to dictionary."""
        return {
            "frame_id": self.frame_id,
            "generator_source": self.generator_source,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "split": self.split,
            "layout_template_id": self.layout_template_id,
            "theme_name": self.theme_name,
            "is_dark_mode": self.is_dark_mode,
            "is_negative_frame": self.is_negative_frame,
            "negative_type": self.negative_type,
            "objects": [obj.to_dict() for obj in self.objects],
            "negative_queries": list(self.negative_queries),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedAnnotationFrame:
        """Instantiate frame annotation from dictionary."""
        objects = [UnifiedObjectAnnotation.from_dict(o) for o in data.get("objects", [])]
        return cls(
            frame_id=str(data["frame_id"]),
            generator_source=str(data["generator_source"]),
            image_path=str(data["image_path"]),
            width=int(data.get("width", 256)),
            height=int(data.get("height", 256)),
            split=str(data.get("split", SPLIT_TRAIN)),
            layout_template_id=data.get("layout_template_id"),
            theme_name=data.get("theme_name"),
            is_dark_mode=bool(data.get("is_dark_mode", False)),
            is_negative_frame=bool(data.get("is_negative_frame", False)),
            negative_type=data.get("negative_type"),
            objects=objects,
            negative_queries=list(data.get("negative_queries", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class UnifiedAnnotationManifest:
    """Manifest container aggregating frames across synthetic generators."""

    schema_version: str = "2.0.0"
    title: str = "AVCS_Unified_UI_Objectness_Manifest"
    coordinate_convention: str = COORDINATE_CONVENTION
    metadata: Dict[str, Any] = field(default_factory=dict)
    frames: List[UnifiedAnnotationFrame] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> UnifiedAnnotationFrame:
        return self.frames[index]

    def validate(self) -> None:
        """Validate entire manifest and all frames."""
        seen_ids = set()
        for frame in self.frames:
            if frame.frame_id in seen_ids:
                raise ValueError(f"Duplicate frame_id detected in manifest: {frame.frame_id}")
            seen_ids.add(frame.frame_id)
            frame.validate()

    def filter_split(self, split: str) -> UnifiedAnnotationManifest:
        """Return a new manifest filtered to the requested split name."""
        norm_split = _normalize_split_name(split)
        filtered = [f for f in self.frames if f.split == norm_split]
        return UnifiedAnnotationManifest(
            schema_version=self.schema_version,
            title=self.title,
            coordinate_convention=self.coordinate_convention,
            metadata=dict(self.metadata),
            frames=filtered,
        )

    def filter_generator(self, generator_source: str) -> UnifiedAnnotationManifest:
        """Return a new manifest filtered to a specific generator source."""
        target = generator_source.strip().lower()
        filtered = [f for f in self.frames if f.generator_source.lower() == target]
        return UnifiedAnnotationManifest(
            schema_version=self.schema_version,
            title=self.title,
            coordinate_convention=self.coordinate_convention,
            metadata=dict(self.metadata),
            frames=filtered,
        )

    def get_frame(self, frame_id: str) -> Optional[UnifiedAnnotationFrame]:
        """Find frame by ID."""
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        return None

    def get_split_counts(self) -> Dict[str, int]:
        """Return sample counts per split."""
        counts: Dict[str, int] = {}
        for f in self.frames:
            counts[f.split] = counts.get(f.split, 0) + 1
        return counts

    def get_generator_counts(self) -> Dict[str, int]:
        """Return sample counts per generator source."""
        counts: Dict[str, int] = {}
        for f in self.frames:
            counts[f.generator_source] = counts.get(f.generator_source, 0) + 1
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
        """Serialize manifest to JSON file or text stream."""
        data = self.to_dict()
        if hasattr(target, "write"):
            json.dump(data, target, indent=indent)  # type: ignore
        else:
            p = Path(target)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedAnnotationManifest:
        """Construct manifest from dictionary."""
        frames = [UnifiedAnnotationFrame.from_dict(f) for f in data.get("frames", [])]
        return cls(
            schema_version=str(data.get("version", "2.0.0")),
            title=str(data.get("title", "AVCS_Unified_UI_Objectness_Manifest")),
            coordinate_convention=str(
                data.get("coordinate_convention", COORDINATE_CONVENTION)
            ),
            metadata=dict(data.get("metadata", {})),
            frames=frames,
        )

    @classmethod
    def from_json(cls, source: Union[str, Path, TextIO]) -> UnifiedAnnotationManifest:
        """Load manifest from JSON file or text stream."""
        if hasattr(source, "read"):
            data = json.load(source)  # type: ignore
        else:
            p = Path(source)
            data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def validate_unified_manifest_dict(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate dictionary structure against Draft-07 unified manifest schema.

    Returns (is_valid, errors). Does not require external jsonschema library.
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["Root manifest must be a JSON object"]

    for req in ("$schema", "title", "version", "coordinate_convention", "frames"):
        if req not in data:
            errors.append(f"Missing required manifest field: {req!r}")

    frames = data.get("frames")
    if not isinstance(frames, list):
        errors.append("Field 'frames' must be a list")
        return False, errors

    frame_ids = set()
    for idx, f in enumerate(frames):
        if not isinstance(f, dict):
            errors.append(f"Frame at index {idx} must be a dictionary")
            continue

        fid = f.get("frame_id")
        if not fid or not isinstance(fid, str):
            errors.append(f"Frame at index {idx} missing valid 'frame_id'")
        elif fid in frame_ids:
            errors.append(f"Duplicate frame_id: {fid!r}")
        else:
            frame_ids.add(fid)

        for req in (
            "generator_source",
            "image_path",
            "width",
            "height",
            "split",
            "is_negative_frame",
            "objects",
        ):
            if req not in f:
                errors.append(f"Frame {fid or idx} missing required field {req!r}")

        is_neg = bool(f.get("is_negative_frame", False))
        objects = f.get("objects", [])
        if not isinstance(objects, list):
            errors.append(f"Frame {fid or idx} 'objects' must be a list")
            continue

        if is_neg:
            if len(objects) > 0:
                errors.append(
                    f"Negative frame {fid} contains {len(objects)} objects; must be empty"
                )
            if not f.get("negative_type"):
                errors.append(f"Negative frame {fid} missing 'negative_type'")

        for obj_idx, obj in enumerate(objects):
            if not isinstance(obj, dict):
                errors.append(f"Frame {fid} object {obj_idx} must be a dictionary")
                continue
            for req_obj in (
                "object_id",
                "bbox_xyxy_norm",
                "bbox_yxyx_norm",
                "centroid_xy_norm",
                "is_actionable",
                "objectness_label",
                "validity_state",
                "element_kind",
            ):
                if req_obj not in obj:
                    errors.append(
                        f"Frame {fid} object {obj_idx} missing required field {req_obj!r}"
                    )

            # Coordinate validation
            xyxy = obj.get("bbox_xyxy_norm")
            yxyx = obj.get("bbox_yxyx_norm")
            centroid = obj.get("centroid_xy_norm")

            if not (isinstance(xyxy, list) and len(xyxy) == 4):
                errors.append(f"Frame {fid} object {obj_idx} bbox_xyxy_norm must have 4 items")
            elif not (isinstance(yxyx, list) and len(yxyx) == 4):
                errors.append(f"Frame {fid} object {obj_idx} bbox_yxyx_norm must have 4 items")
            else:
                x1, y1, x2, y2 = xyxy
                y1_c, x1_c, y2_c, x2_c = yxyx
                if not (0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0):
                    errors.append(f"Frame {fid} object {obj_idx} bbox_xyxy_norm out of range [0, 1]")
                if x1 > x2 or y1 > y2:
                    errors.append(f"Frame {fid} object {obj_idx} inverted bbox_xyxy_norm: {xyxy}")
                if abs(x1 - x1_c) > 1e-4 or abs(y1 - y1_c) > 1e-4 or abs(x2 - x2_c) > 1e-4 or abs(y2 - y2_c) > 1e-4:
                    errors.append(f"Frame {fid} object {obj_idx} dual box mismatch: {xyxy} vs {yxyx}")

            if not (isinstance(centroid, list) and len(centroid) == 2):
                errors.append(f"Frame {fid} object {obj_idx} centroid_xy_norm must have 2 items")
            else:
                cx, cy = centroid
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    errors.append(f"Frame {fid} object {obj_idx} centroid out of range [0, 1]: {centroid}")

    return len(errors) == 0, errors


def load_ui_grounding_manifest(
    manifest_path: Union[str, Path] = "datasets/ui_grounding/manifest.json",
    dataset_root: Optional[Union[str, Path]] = None,
) -> UnifiedAnnotationManifest:
    """Load `datasets/ui_grounding/manifest.json` into `UnifiedAnnotationManifest`.

    Translates `bbox: [ymin, xmin, ymax, xmax]` into dual `bbox_xyxy_norm` and `bbox_yxyx_norm`,
    and standardizes actionability, objectness labels, and negative frames.
    """
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"ui_grounding manifest not found at: {m_path}")

    root = Path(dataset_root) if dataset_root is not None else m_path.parent
    raw_data = json.loads(m_path.read_text(encoding="utf-8"))

    frames: List[UnifiedAnnotationFrame] = []
    for raw_fr in raw_data.get("frames", []):
        frame_id = raw_fr["frame_id"]
        rel_img_path = raw_fr["file_path"]
        # Normalize relative path to be relative to repo or dataset root
        full_img_path = (root / rel_img_path).as_posix()
        width = int(raw_fr.get("width", 256))
        height = int(raw_fr.get("height", 256))
        split = _normalize_split_name(raw_fr.get("split", SPLIT_TRAIN))
        category = raw_fr.get("category", "desktop_canvas")
        session_id = raw_fr.get("session_id", "")
        layout_template_id = f"{category}_{session_id}" if session_id else category
        is_neg = bool(raw_fr.get("is_negative", False)) or split == SPLIT_TEST_NEGATIVE

        objects: List[UnifiedObjectAnnotation] = []
        if not is_neg:
            for pt_idx, pt in enumerate(raw_fr.get("interactive_points", [])):
                obj_id = pt.get("element_id") or f"{frame_id}_obj_{pt_idx:02d}"
                action_type = pt.get("action_type", "clickable")
                element_kind = pt.get("element_kind", "button")
                is_actionable = action_type != "static"

                # In ui_grounding, bbox is [ymin, xmin, ymax, xmax]
                raw_bbox = pt["bbox"]
                ymin, xmin, ymax, xmax = (
                    float(raw_bbox[0]),
                    float(raw_bbox[1]),
                    float(raw_bbox[2]),
                    float(raw_bbox[3]),
                )
                bbox_xyxy = (xmin, ymin, xmax, ymax)
                bbox_yxyx = (ymin, xmin, ymax, xmax)

                # Centroid
                cx = float(pt.get("x", (xmin + xmax) / 2.0))
                cy = float(pt.get("y", (ymin + ymax) / 2.0))
                centroid_xy = (cx, cy)

                target_query = pt.get("target_query")
                query_vars = pt.get("query_variations", [])

                obj = UnifiedObjectAnnotation(
                    object_id=obj_id,
                    bbox_xyxy_norm=bbox_xyxy,
                    bbox_yxyx_norm=bbox_yxyx,
                    centroid_xy_norm=centroid_xy,
                    is_actionable=is_actionable,
                    objectness_label=action_type,
                    validity_state="VALID",
                    element_kind=element_kind,
                    text_label=pt.get("text_label"),
                    target_query=target_query,
                    query_variations=list(query_vars),
                )
                obj.validate_coordinates()
                objects.append(obj)

        neg_queries = list(raw_fr.get("negative_queries", []))
        frame = UnifiedAnnotationFrame(
            frame_id=frame_id,
            generator_source=GENERATOR_UI_GROUNDING,
            image_path=full_img_path,
            width=width,
            height=height,
            split=split,
            layout_template_id=layout_template_id,
            theme_name=category,
            is_dark_mode=category not in ("light_clean", "form_input"),
            is_negative_frame=is_neg,
            negative_type="wallpaper" if is_neg else None,
            objects=objects,
            negative_queries=neg_queries,
            metadata={
                "source": raw_fr.get("source", "procedural_synthetic"),
                "session_id": session_id,
                "affordance_mask_path": raw_fr.get("affordance_mask_path"),
            },
        )
        frame.validate()
        frames.append(frame)

    manifest = UnifiedAnnotationManifest(
        schema_version="2.0.0",
        title="AVCS_Unified_UI_Objectness_Manifest",
        coordinate_convention=COORDINATE_CONVENTION,
        metadata={
            "source_dataset": "ui_grounding",
            "provenance": raw_data.get("dataset_provenance", "avcs_procedural_synthetic_desktop_v1"),
            "original_total_frames": raw_data.get("total_frames", len(frames)),
        },
        frames=frames,
    )
    return manifest


def load_cleanroom_manifest(
    corpus_dir: Union[str, Path] = "datasets/cleanroom_system1_v1",
    validate_hashes: bool = False,
) -> UnifiedAnnotationManifest:
    """Load `datasets/cleanroom_system1_v1` into `UnifiedAnnotationManifest`.

    Reads `manifest.json` and parses individual annotation files in `annotations/{split}/`.
    Standardizes bounding boxes, centroids, layout families, and empty desktop negative frames.
    """
    c_dir = Path(corpus_dir).resolve()
    manifest_p = c_dir / "manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Cleanroom manifest not found at: {manifest_p}")

    raw_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    frames: List[UnifiedAnnotationFrame] = []
    for file_entry in raw_manifest.get("files", []):
        frame_id = file_entry["frame_id"]
        raw_split = file_entry["split"]
        norm_split = _normalize_split_name(raw_split)
        layout_family = file_entry.get("layout_family", "unknown")

        img_rel = file_entry["image"]["relative_path"]
        full_img_path = (c_dir / img_rel).as_posix()
        width = int(file_entry["image"].get("width", 256))
        height = int(file_entry["image"].get("height", 256))

        annot_rel = file_entry["annotation"]["relative_path"]
        annot_file = c_dir / annot_rel
        if not annot_file.exists():
            raise FileNotFoundError(f"Cleanroom annotation file missing: {annot_file}")

        annot_data = json.loads(annot_file.read_text(encoding="utf-8"))
        is_neg = bool(annot_data.get("is_negative_frame", False)) or norm_split == SPLIT_TEST_NEGATIVE

        # Build query lookup by target_region_id
        queries = annot_data.get("queries", [])
        region_to_query: Dict[str, Dict[str, Any]] = {}
        absent_queries: List[str] = []
        for q in queries:
            if q.get("target_presence") and q.get("target_region_id"):
                region_to_query[q["target_region_id"]] = q
            elif not q.get("target_presence"):
                absent_queries.append(q["query"])

        objects: List[UnifiedObjectAnnotation] = []
        if not is_neg:
            for region in annot_data.get("regions", []):
                reg_id = region["region_id"]
                kind = region.get("kind", "button")
                label = region.get("label", "")

                # In cleanroom, bbox_xyxy_norm is [xmin, ymin, xmax, ymax]
                x1, y1, x2, y2 = [float(v) for v in region["bbox_xyxy_norm"]]
                bbox_xyxy = (x1, y1, x2, y2)
                bbox_yxyx = (y1, x1, y2, x2)

                # Centroid from query target point or center of box
                matched_q = region_to_query.get(reg_id)
                if matched_q and matched_q.get("target_point_norm"):
                    pt = matched_q["target_point_norm"]
                    cx, cy = float(pt[0]), float(pt[1])
                else:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                centroid_xy = (cx, cy)

                # Kind to objectness label
                if kind == "input_field":
                    obj_label = "typeable"
                else:
                    obj_label = "clickable"

                target_query = matched_q["query"] if matched_q else None
                query_vars = [target_query] if target_query else []

                obj = UnifiedObjectAnnotation(
                    object_id=reg_id,
                    bbox_xyxy_norm=bbox_xyxy,
                    bbox_yxyx_norm=bbox_yxyx,
                    centroid_xy_norm=centroid_xy,
                    is_actionable=True,
                    objectness_label=obj_label,
                    validity_state="VALID",
                    element_kind=kind,
                    text_label=label if label != "<none>" else None,
                    target_query=target_query,
                    query_variations=query_vars,
                )
                obj.validate_coordinates()
                objects.append(obj)

        frame = UnifiedAnnotationFrame(
            frame_id=frame_id,
            generator_source=GENERATOR_CLEANROOM,
            image_path=full_img_path,
            width=width,
            height=height,
            split=norm_split,
            layout_template_id=layout_family,
            theme_name=f"cleanroom_{file_entry.get('sample_seed', 0) % 4}",
            is_dark_mode=False,  # Cleanroom features light window interiors
            is_negative_frame=is_neg,
            negative_type="empty_desktop" if is_neg else None,
            objects=objects,
            negative_queries=absent_queries,
            metadata={
                "raw_split": raw_split,
                "layout_family": layout_family,
                "sample_seed": file_entry.get("sample_seed"),
                "provenance": file_entry.get("provenance", {}).get("source_method"),
            },
        )
        frame.validate()
        frames.append(frame)

    manifest = UnifiedAnnotationManifest(
        schema_version="2.0.0",
        title="AVCS_Unified_UI_Objectness_Manifest",
        coordinate_convention=COORDINATE_CONVENTION,
        metadata={
            "source_dataset": "cleanroom_system1_v1",
            "generator_version": raw_manifest.get("generator_version"),
            "dataset_version": raw_manifest.get("dataset_version"),
            "splits": raw_manifest.get("splits", {}),
        },
        frames=frames,
    )
    return manifest


def build_combined_manifest(
    ui_manifest_path: Union[str, Path] = "datasets/ui_grounding/manifest.json",
    cleanroom_dir: Union[str, Path] = "datasets/cleanroom_system1_v1",
    dataset_root_ui: Optional[Union[str, Path]] = None,
) -> UnifiedAnnotationManifest:
    """Build a unified combined manifest aggregating UI Grounding and Cleanroom frames."""
    ui_manifest = load_ui_grounding_manifest(ui_manifest_path, dataset_root=dataset_root_ui)
    clean_manifest = load_cleanroom_manifest(cleanroom_dir)

    combined_frames = list(ui_manifest.frames) + list(clean_manifest.frames)
    combined = UnifiedAnnotationManifest(
        schema_version="2.0.0",
        title="AVCS_Unified_UI_Objectness_Manifest",
        coordinate_convention=COORDINATE_CONVENTION,
        metadata={
            "sources": [GENERATOR_UI_GROUNDING, GENERATOR_CLEANROOM],
            "ui_frames_count": len(ui_manifest),
            "cleanroom_frames_count": len(clean_manifest),
            "total_frames": len(combined_frames),
        },
        frames=combined_frames,
    )
    combined.validate()
    return combined


def load_unified_manifest(source: Union[str, Path]) -> UnifiedAnnotationManifest:
    """Auto-detect source format and load as UnifiedAnnotationManifest.

    Supports:
    - Unified JSON manifest file
    - Canonical `datasets/ui_grounding/manifest.json`
    - Directory for `cleanroom_system1_v1`
    """
    p = Path(source).resolve()
    if p.is_dir():
        # Directory - check for cleanroom
        if (p / "manifest.json").exists() and (p / "frames").exists():
            return load_cleanroom_manifest(p)
        raise ValueError(f"Unrecognized dataset directory: {p}")

    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    title = data.get("title", "")
    if title == "AVCS_Unified_UI_Objectness_Manifest":
        return UnifiedAnnotationManifest.from_dict(data)
    elif title == "AVCS_UI_Grounding_Dataset_Manifest":
        return load_ui_grounding_manifest(p)
    else:
        # Check if cleanroom manifest
        if data.get("schema") == "avcs.system1.cleanroom.dataset.v1":
            return load_cleanroom_manifest(p.parent)
        raise ValueError(f"Unrecognized manifest format in {p}")
