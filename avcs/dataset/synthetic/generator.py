"""Canonical synthetic desktop generator and dataset exporter."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from avcs.dataset.synthetic.composer import DesktopComposer, UI_CATEGORIES
from avcs.dataset.synthetic.primitives import SyntheticUIElement
from avcs.dataset.synthetic.themes import AVAILABLE_THEMES


@dataclass
class SyntheticDesktopFrame:
    """Complete synthetic desktop screen frame with ground-truth metadata."""
    frame_id: str
    file_path: str
    width: int
    height: int
    category: str
    split: str
    session_id: str
    is_negative: bool
    affordance_mask_path: str
    source: str
    image: np.ndarray
    affordance_mask: np.ndarray
    elements: List[SyntheticUIElement] = field(default_factory=list)
    negative_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame entry to JSON manifest dictionary."""
        return {
            "frame_id": self.frame_id,
            "file_path": self.file_path.replace("\\", "/"),
            "width": self.width,
            "height": self.height,
            "category": self.category,
            "split": self.split,
            "session_id": self.session_id,
            "is_negative": self.is_negative,
            "affordance_mask_path": self.affordance_mask_path.replace("\\", "/"),
            "source": self.source,
            "interactive_points": [el.to_dict() for el in self.elements],
            "negative_queries": self.negative_queries,
        }


class SyntheticDesktopGenerator:
    """Main generator producing reproducible synthetic desktop frames and benchmark datasets."""

    def __init__(
        self,
        screen_size: Tuple[int, int] = (256, 256),
        grid_size: Tuple[int, int] = (16, 16),
    ) -> None:
        self.screen_size = screen_size
        self.grid_size = grid_size
        self.composer = DesktopComposer(screen_size=screen_size, grid_size=grid_size)

    def generate_frame(
        self,
        category: Optional[str] = None,
        theme_name: Optional[str] = None,
        is_negative: bool = False,
        negative_type: Optional[str] = None,
        seed: Optional[int] = None,
        split: str = "train",
        session_id: Optional[str] = None,
        frame_idx: int = 1,
    ) -> SyntheticDesktopFrame:
        """Generates a single synthetic desktop frame with verified ground truth."""
        rng = random.Random(seed)
        chosen_cat = category or rng.choice(UI_CATEGORIES)
        chosen_theme = theme_name or rng.choice(AVAILABLE_THEMES)
        chosen_session = session_id or f"session_{split}_{chosen_cat}"

        img_rgb, mask, elements, neg_queries = self.composer.compose(
            category=chosen_cat,
            theme_name=chosen_theme,
            is_negative=is_negative,
            negative_type=negative_type,
            seed=seed,
        )

        prefix = "test_neg" if is_negative else split
        frame_id = f"screen_{prefix}_{frame_idx:04d}"
        file_path = f"screens/{prefix}_{frame_idx:04d}.png"
        mask_path = f"masks/{prefix}_{frame_idx:04d}.npy"
        source = "procedural_negative" if is_negative else "procedural_synthetic"

        return SyntheticDesktopFrame(
            frame_id=frame_id,
            file_path=file_path,
            width=self.screen_size[0],
            height=self.screen_size[1],
            category=chosen_cat,
            split=split,
            session_id=chosen_session,
            is_negative=is_negative,
            affordance_mask_path=mask_path,
            source=source,
            image=img_rgb,
            affordance_mask=mask,
            elements=elements,
            negative_queries=neg_queries,
        )

    def generate_canonical_dataset(
        self,
        output_dir: str | Path = "datasets/ui_grounding",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates the authoritative 270-frame dataset with strict session isolation.
        
        Splits:
            train: 160 frames (sessions 1..8)
            val: 40 frames (sessions 9..12)
            test_held_out: 40 frames (sessions 13..16)
            test_negative: 30 frames (sessions 17..19)
        """
        out_path = Path(output_dir)
        screens_dir = out_path / "screens"
        masks_dir = out_path / "masks"
        splits_dir = out_path / "splits"

        screens_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        frames_manifest: List[Dict[str, Any]] = []
        split_frame_ids: Dict[str, List[str]] = {
            "train": [],
            "val": [],
            "test_held_out": [],
            "test_negative": [],
        }

        # Session partitions (ensuring ZERO leakage between train, val, and test)
        train_sessions = [f"session_train_{i:02d}" for i in range(1, 9)]
        val_sessions = [f"session_val_{i:02d}" for i in range(1, 5)]
        test_sessions = [f"session_test_{i:02d}" for i in range(1, 5)]
        neg_sessions = [f"session_neg_{i:02d}" for i in range(1, 4)]

        # 1. Train split: 160 frames (20 per category across 8 categories)
        train_idx = 1
        for cat_i, cat in enumerate(UI_CATEGORIES):
            for step in range(20):
                frame_seed = 1000 + cat_i * 100 + step
                sess = train_sessions[(cat_i * 2 + step // 10) % len(train_sessions)]
                theme = AVAILABLE_THEMES[step % len(AVAILABLE_THEMES)]
                frame = self.generate_frame(
                    category=cat,
                    theme_name=theme,
                    is_negative=False,
                    seed=frame_seed,
                    split="train",
                    session_id=sess,
                    frame_idx=train_idx,
                )
                self._save_frame_assets(out_path, frame)
                frames_manifest.append(frame.to_dict())
                split_frame_ids["train"].append(frame.frame_id)
                train_idx += 1

        # 2. Validation split: 40 frames (5 per category across 8 categories)
        val_idx = 1
        for cat_i, cat in enumerate(UI_CATEGORIES):
            for step in range(5):
                frame_seed = 5000 + cat_i * 50 + step
                sess = val_sessions[cat_i % len(val_sessions)]
                theme = AVAILABLE_THEMES[(step + 1) % len(AVAILABLE_THEMES)]
                frame = self.generate_frame(
                    category=cat,
                    theme_name=theme,
                    is_negative=False,
                    seed=frame_seed,
                    split="val",
                    session_id=sess,
                    frame_idx=val_idx,
                )
                self._save_frame_assets(out_path, frame)
                frames_manifest.append(frame.to_dict())
                split_frame_ids["val"].append(frame.frame_id)
                val_idx += 1

        # 3. Test Held-Out split: 40 frames (5 per category across 8 categories)
        test_idx = 1
        for cat_i, cat in enumerate(UI_CATEGORIES):
            for step in range(5):
                frame_seed = 8000 + cat_i * 50 + step
                sess = test_sessions[cat_i % len(test_sessions)]
                theme = AVAILABLE_THEMES[(step + 2) % len(AVAILABLE_THEMES)]
                frame = self.generate_frame(
                    category=cat,
                    theme_name=theme,
                    is_negative=False,
                    seed=frame_seed,
                    split="test_held_out",
                    session_id=sess,
                    frame_idx=test_idx,
                )
                self._save_frame_assets(out_path, frame)
                frames_manifest.append(frame.to_dict())
                split_frame_ids["test_held_out"].append(frame.frame_id)
                test_idx += 1

        # 4. Test Negative split: 30 frames (10 solid blank, 10 smooth gradients, 10 realistic wallpapers)
        neg_types = ["solid"] * 10 + ["gradient"] * 10 + ["wallpaper"] * 10
        neg_idx = 1
        for step, ntype in enumerate(neg_types):
            frame_seed = 9000 + step
            sess = neg_sessions[step // 10]
            theme = AVAILABLE_THEMES[step % len(AVAILABLE_THEMES)]
            frame = self.generate_frame(
                category="desktop_canvas",
                theme_name=theme,
                is_negative=True,
                negative_type=ntype,
                seed=frame_seed,
                split="test_negative",
                session_id=sess,
                frame_idx=neg_idx,
            )
            self._save_frame_assets(out_path, frame)
            frames_manifest.append(frame.to_dict())
            split_frame_ids["test_negative"].append(frame.frame_id)
            neg_idx += 1

        # Save split JSON index files
        for split_name, ids in split_frame_ids.items():
            with open(splits_dir / f"{split_name}.json", "w", encoding="utf-8") as f:
                json.dump(ids, f, indent=2)

        # Build Master Manifest conforming strictly to Draft-07 schema
        manifest: Dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "AVCS_UI_Grounding_Dataset_Manifest",
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_provenance": "avcs_procedural_synthetic_desktop_v1",
            "total_frames": len(frames_manifest),
            "splits": {
                "train": len(split_frame_ids["train"]),
                "val": len(split_frame_ids["val"]),
                "test_held_out": len(split_frame_ids["test_held_out"]),
                "test_negative": len(split_frame_ids["test_negative"]),
            },
            "classes": {
                "macro_affordances": ["static", "clickable", "typeable"],
                "foveal_element_kinds": [
                    "button",
                    "input_field",
                    "icon",
                    "tab",
                    "link",
                    "text_label",
                    "container",
                ],
                "ui_categories": UI_CATEGORIES,
            },
            "frames": frames_manifest,
        }

        manifest_file = out_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Write documentation README
        self._write_readme(out_path, manifest)

        return manifest

    def _save_frame_assets(self, out_path: Path, frame: SyntheticDesktopFrame) -> None:
        """Saves PNG image and NPY affordance mask to disk."""
        img_dest = out_path / frame.file_path
        mask_dest = out_path / frame.affordance_mask_path

        img_dest.parent.mkdir(parents=True, exist_ok=True)
        mask_dest.parent.mkdir(parents=True, exist_ok=True)

        pil_img = Image.fromarray(frame.image, mode="RGB")
        pil_img.save(img_dest, format="PNG", optimize=True)
        np.save(mask_dest, frame.affordance_mask)

    def _write_readme(self, out_path: Path, manifest: Dict[str, Any]) -> None:
        """Writes documentation for dataset provenance and benchmark reproducibility."""
        readme_path = out_path / "README.md"
        content = f"""# AVCS UI Grounding & Affordance Dataset

- **Version:** 1.0.0
- **Created At:** {manifest['created_at']}
- **Provenance:** {manifest['dataset_provenance']}
- **Total Frames:** {manifest['total_frames']}
- **Resolution:** {self.screen_size[0]}x{self.screen_size[1]} RGB PNG
- **Affordance Grid:** {self.grid_size[0]}x{self.grid_size[1]} int64 NumPy masks

## Partition Summary
- `train`: {manifest['splits']['train']} frames (8 sessions, balanced across 8 categories)
- `val`: {manifest['splits']['val']} frames (4 sessions, distinct seeds)
- `test_held_out`: {manifest['splits']['test_held_out']} frames (4 sessions, strictly disjoint from train/val)
- `test_negative`: {manifest['splits']['test_negative']} frames (10 solid blank, 10 smooth gradients, 10 procedural wallpapers, 0 interactive points)

## Verification
To verify dataset manifest schema and split isolation:
```bash
python3 -m pytest tests/test_ui_grounding_dataset.py -v
```
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
