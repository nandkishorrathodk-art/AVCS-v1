"""Anchor-Verified Skill Cache & Trajectory Vault for the Ace Autopilot.

Enforces P0 Safety Rule:
NEVER blindly replay cached trajectories on semantic goal hash alone.
UI State Anchors (active window title, foreground process, target visual bounding box or aspect)
MUST be verified against the live desktop state before approving a cache hit.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import threading
import time
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from avcs.protocol.actions import Action

logger = logging.getLogger("avcs.ace.skills_vault")


def visual_hash_similarity(hash1: Optional[str], hash2: Optional[str]) -> float:
    """Computes similarity in [0.0, 1.0] between two hexadecimal visual hashes."""
    if not hash1 or not hash2:
        return 0.0
    h1 = str(hash1).strip().lower()
    h2 = str(hash2).strip().lower()
    if h1.startswith("0x"):
        h1 = h1[2:]
    if h2.startswith("0x"):
        h2 = h2[2:]
    if not h1 or not h2:
        return 0.0
    if h1 == h2:
        return 1.0
    try:
        val1 = int(h1, 16)
        val2 = int(h2, 16)
        xor = val1 ^ val2
        bit_dist = bin(xor).count("1")
        total_bits = 64
        return max(0.0, min(1.0, 1.0 - (bit_dist / float(total_bits))))
    except Exception:
        return 1.0 if h1 == h2 else 0.0


def compute_visual_hash(
    image_or_obs: Any,
    coordinate: Optional[Tuple[int, int]] = None,
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
    radius: int = 50,
) -> Optional[str]:
    """Computes a structural regional visual hash (dHash) from an observation or image."""
    if image_or_obs is None:
        return None

    raw_rgb: Optional[np.ndarray] = None
    try:
        from PIL import Image
        if isinstance(image_or_obs, Image.Image):
            raw_rgb = np.asarray(image_or_obs.convert("RGB"))
    except Exception:
        raw_rgb = None

    if raw_rgb is None and hasattr(image_or_obs, "image_bytes") and image_or_obs.image_bytes:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_or_obs.image_bytes)).convert("RGB")
            raw_rgb = np.asarray(img)
        except Exception:
            raw_rgb = None
    elif raw_rgb is None and isinstance(image_or_obs, bytes) and image_or_obs:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_or_obs)).convert("RGB")
            raw_rgb = np.asarray(img)
        except Exception:
            raw_rgb = None
    elif raw_rgb is None and isinstance(image_or_obs, np.ndarray):
        raw_rgb = image_or_obs
    elif raw_rgb is None and hasattr(image_or_obs, "convert") and hasattr(image_or_obs, "size"):
        try:
            raw_rgb = np.asarray(image_or_obs.convert("RGB"))
        except Exception:
            raw_rgb = None

    if raw_rgb is None:
        b = getattr(image_or_obs, "image_bytes", None) or (image_or_obs if isinstance(image_or_obs, bytes) else None)
        if b:
            return hashlib.sha256(b[:512]).hexdigest()[:16]
        return None

    h, w = raw_rgb.shape[:2]
    if h == 0 or w == 0:
        return None

    # Determine region of interest (ROI)
    if bounding_box and len(bounding_box) == 4:
        x1, y1, x2, y2 = bounding_box
        nx1, nx2 = float(min(x1, x2)), float(max(x1, x2))
        ny1, ny2 = float(min(y1, y2)), float(max(y1, y2))
        if nx2 <= 1.0 and ny2 <= 1.0 and (nx2 > 0 or ny2 > 0):
            nx1 *= w
            nx2 *= w
            ny1 *= h
            ny2 *= h
        bx1 = max(0, min(w - 1, int(nx1)))
        by1 = max(0, min(h - 1, int(ny1)))
        bx2 = max(bx1 + 1, min(w, int(nx2)))
        by2 = max(by1 + 1, min(h, int(ny2)))
        if bx2 <= bx1 or by2 <= by1:
            crop = raw_rgb
        else:
            crop = raw_rgb[by1:by2, bx1:bx2]
    elif coordinate and len(coordinate) == 2:
        cx, cy = int(coordinate[0]), int(coordinate[1])
        bx1 = max(0, cx - radius)
        by1 = max(0, cy - radius)
        bx2 = min(w, cx + radius)
        by2 = min(h, cy + radius)
        if bx2 <= bx1 or by2 <= by1:
            crop = raw_rgb
        else:
            crop = raw_rgb[by1:by2, bx1:bx2]
    else:
        crop = raw_rgb

    # Convert crop to grayscale
    if crop.ndim == 3:
        gray = (crop[:, :, 0].astype(np.uint16) * 2 +
                crop[:, :, 1].astype(np.uint16) * 5 +
                crop[:, :, 2].astype(np.uint16)) >> 3
    else:
        gray = crop

    # Downsample to 8 rows, 9 columns for difference hash (full 64-bit hex coverage)
    try:
        from PIL import Image
        img_gray = Image.fromarray(gray.astype(np.uint8))
        resized = np.asarray(img_gray.resize((9, 8), Image.Resampling.BILINEAR))
    except Exception:
        ch, cw = gray.shape[:2]
        if ch < 8 or cw < 9:
            sy = np.clip(np.linspace(0, max(0, ch - 1), 8, dtype=int), 0, max(0, ch - 1))
            sx = np.clip(np.linspace(0, max(0, cw - 1), 9, dtype=int), 0, max(0, cw - 1))
        else:
            sy = np.linspace(0, ch - 1, 8, dtype=int)
            sx = np.linspace(0, cw - 1, 9, dtype=int)
        resized = gray[sy[:, None], sx]

    diff = resized[:, 1:] > resized[:, :-1]
    flat_diff = diff.flatten()
    val = 0
    for bit in flat_diff[:64]:
        val = (val << 1) | int(bit)
    return f"{val:016x}"


@dataclasses.dataclass
class UIStateAnchor:
    """UI state anchor representing the expected desktop context before trajectory execution."""
    window_title: Optional[str] = None
    process_name: Optional[str] = None
    class_name: Optional[str] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    aspect: Optional[str] = None
    visual_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.window_title is not None:
            d["window_title"] = self.window_title
        if self.process_name is not None:
            d["process_name"] = self.process_name
        if self.class_name is not None:
            d["class_name"] = self.class_name
        if self.bounding_box is not None:
            d["bounding_box"] = list(self.bounding_box)
        if self.aspect is not None:
            d["aspect"] = self.aspect
        if self.visual_hash is not None:
            d["visual_hash"] = self.visual_hash
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UIStateAnchor:
        bbox = None
        if "bounding_box" in data and data["bounding_box"]:
            bbox = tuple(data["bounding_box"])
        return cls(
            window_title=data.get("window_title"),
            process_name=data.get("process_name"),
            class_name=data.get("class_name"),
            bounding_box=bbox,
            aspect=data.get("aspect"),
            visual_hash=data.get("visual_hash"),
        )

    @classmethod
    def create_from_desktop(
        cls,
        computer: Optional[Any] = None,
        verifier: Optional[Any] = None,
        observation: Optional[Any] = None,
        target_coord: Optional[Tuple[int, int]] = None,
        bounding_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> UIStateAnchor:
        """Factory method constructing a UIStateAnchor capturing live desktop state."""
        title = None
        proc = None
        if verifier is not None:
            title = verifier.get_active_window_title() or None
            proc = verifier.get_foreground_process_name() or None
        v_hash = None
        if observation is not None:
            v_hash = compute_visual_hash(observation, coordinate=target_coord, bounding_box=bounding_box)
        return cls(
            window_title=title,
            process_name=proc,
            bounding_box=bounding_box,
            visual_hash=v_hash,
        )

    @staticmethod
    def _canonicalize_bbox(
        bbox: Optional[Union[Tuple[Any, ...], List[Any]]],
        canvas_w: float = 1280.0,
        canvas_h: float = 800.0,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Canonicalizes bounding box: (min_x, min_y, max_x, max_y) in canonical pixel coordinates."""
        if not bbox or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            # If coordinates are normalized in [0.0, 1.0], convert to canonical pixels
            if max_x <= 1.0 and max_y <= 1.0 and (max_x > 0.0 or max_y > 0.0 or min_x > 0.0 or min_y > 0.0):
                min_x *= canvas_w
                max_x *= canvas_w
                min_y *= canvas_h
                max_y *= canvas_h
            return (min_x, min_y, max_x, max_y)
        except Exception:
            return None

    def is_empty(self) -> bool:
        """Returns True if this anchor specifies zero constraints."""
        return (
            not (self.window_title and self.window_title.strip())
            and not (self.process_name and self.process_name.strip())
            and not (self.class_name and self.class_name.strip())
            and self.bounding_box is None
            and not (self.aspect and self.aspect.strip())
            and not (self.visual_hash and self.visual_hash.strip())
        )

    def similarity(self, current: Optional[UIStateAnchor], tolerance_px: int = 15) -> float:
        """Calculates matching similarity score in [0.0, 1.0] between this anchor and live state."""
        if current is None or self.is_empty():
            return 0.0

        scores: List[float] = []

        if self.window_title is not None:
            if not current.window_title:
                scores.append(0.0)
            else:
                wt_self = self.window_title.lower().strip()
                wt_curr = current.window_title.lower().strip()
                if wt_self in wt_curr or (len(wt_curr) >= 3 and wt_curr in wt_self):
                    scores.append(1.0)
                else:
                    scores.append(0.0)

        if self.process_name is not None:
            if not current.process_name:
                scores.append(0.0)
            else:
                p_self = os.path.basename(self.process_name).lower().replace(".exe", "").strip()
                p_curr = os.path.basename(current.process_name).lower().replace(".exe", "").strip()
                scores.append(1.0 if p_self == p_curr else 0.0)

        if self.class_name is not None:
            if not current.class_name:
                scores.append(0.0)
            else:
                scores.append(1.0 if self.class_name.lower().strip() == current.class_name.lower().strip() else 0.0)

        if self.bounding_box is not None:
            if not current.bounding_box or len(current.bounding_box) < 4:
                scores.append(0.0)
            else:
                b_self = self._canonicalize_bbox(self.bounding_box)
                b_curr = self._canonicalize_bbox(current.bounding_box)
                if b_self is None or b_curr is None:
                    scores.append(0.0)
                else:
                    x1, y1, x2, y2 = b_self
                    cx1, cy1, cx2, cy2 = b_curr
                    err = max(abs(x1 - cx1), abs(y1 - cy1), abs(x2 - cx2), abs(y2 - cy2))
                    scores.append(1.0 if err <= tolerance_px else max(0.0, 1.0 - (err - tolerance_px) / 100.0))

        if self.aspect is not None:
            if not current.aspect:
                scores.append(0.0)
            else:
                scores.append(1.0 if self.aspect.lower().strip() == current.aspect.lower().strip() else 0.0)

        if self.visual_hash is not None:
            if not current.visual_hash:
                scores.append(0.0)
            else:
                sim = visual_hash_similarity(self.visual_hash, current.visual_hash)
                scores.append(sim)

        if not scores:
            return 0.0
        return sum(scores) / float(len(scores))

    def matches(self, current: Optional[UIStateAnchor], tolerance_px: int = 15, min_score: float = 0.90) -> bool:
        """Verifies if current live UI state satisfies all constraints defined in this anchor.

        Enforces P0: an empty anchor with no constraints cannot approve a cache hit.
        Applies >= 90% structural match rule for visual hashes and overall state anchors.
        """
        if current is None or self.is_empty():
            return False

        # 1. Active Window Title check (case-insensitive substring or bidirectional match)
        if self.window_title is not None:
            if not current.window_title:
                return False
            wt_self = self.window_title.lower().strip()
            wt_curr = current.window_title.lower().strip()
            if wt_self not in wt_curr and (len(wt_curr) < 3 or wt_curr not in wt_self):
                return False

        # 2. Foreground Process Name check (handles paths and .exe suffix)
        if self.process_name is not None:
            if not current.process_name:
                return False
            p_self = os.path.basename(self.process_name).lower().replace(".exe", "").strip()
            p_curr = os.path.basename(current.process_name).lower().replace(".exe", "").strip()
            if p_self != p_curr:
                return False

        # 3. Class Name check
        if self.class_name is not None:
            if not current.class_name:
                return False
            if self.class_name.lower().strip() != current.class_name.lower().strip():
                return False

        # 4. Target Visual Bounding Box check (within tolerance)
        if self.bounding_box is not None:
            if not current.bounding_box or len(current.bounding_box) < 4:
                return False
            b_self = self._canonicalize_bbox(self.bounding_box)
            b_curr = self._canonicalize_bbox(current.bounding_box)
            if b_self is None or b_curr is None:
                return False
            x1, y1, x2, y2 = b_self
            cx1, cy1, cx2, cy2 = b_curr
            if (
                abs(x1 - cx1) > tolerance_px
                or abs(y1 - cy1) > tolerance_px
                or abs(x2 - cx2) > tolerance_px
                or abs(y2 - cy2) > tolerance_px
            ):
                return False

        # 5. Aspect Ratio check
        if self.aspect is not None:
            if not current.aspect:
                return False
            if self.aspect.lower().strip() != current.aspect.lower().strip():
                return False

        # 6. Visual Hash check (>= 90% match threshold or exact match)
        if self.visual_hash and self.visual_hash.strip():
            if not current.visual_hash or not current.visual_hash.strip():
                return False
            sim = visual_hash_similarity(self.visual_hash, current.visual_hash)
            if sim < min_score and self.visual_hash != current.visual_hash:
                return False

        return True


@dataclasses.dataclass
class SkillCacheEntry:
    """Stored skill trajectory and its required UI state anchors."""
    goal: str
    goal_hash: str
    goal_slug: str
    anchors: List[UIStateAnchor]
    actions: List[Dict[str, Any]]
    created_at: float
    entry_id: str = ""
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "goal": self.goal,
            "goal_hash": self.goal_hash,
            "goal_slug": self.goal_slug,
            "anchors": [a.to_dict() for a in self.anchors],
            "actions": self.actions,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillCacheEntry:
        anchors = [UIStateAnchor.from_dict(a) for a in data.get("anchors", [])]
        actions = data.get("actions", [])
        if not actions:
            sft_demo = data.get("sft_demonstration")
            if isinstance(sft_demo, dict) and sft_demo.get("trajectory"):
                actions = sft_demo["trajectory"]
            elif "steps" in data and isinstance(data["steps"], list):
                actions = [s["action"] for s in data["steps"] if isinstance(s, dict) and s.get("action")]
        goal = data.get("goal", "")
        g_hash = data.get("goal_hash") or (SkillCache.hash_goal(goal) if goal else "")
        g_slug = data.get("goal_slug") or (SkillCache.slugify(goal) if goal else "")
        entry_id = str(data.get("entry_id") or data.get("session_id", ""))
        return cls(
            entry_id=entry_id,
            goal=goal,
            goal_hash=g_hash,
            goal_slug=g_slug,
            anchors=anchors,
            actions=actions,
            created_at=float(data.get("created_at", time.time())),
            metadata=data.get("metadata", {}),
        )


@dataclasses.dataclass
class SkillLookupResult:
    """Result of an anchor-verified skill lookup."""
    hit: bool
    actions: Optional[List[Action]] = None
    reason: str = ""
    entry: Optional[SkillCacheEntry] = None


class SkillCache:
    """Anchor-Verified Skill Cache enforcing the P0 rule against blind replay.

    Supports both JSON file vault storage and SQLite database backend.
    """

    def __init__(self, vault_dir: Optional[Union[str, Path]] = None, backend: str = "json") -> None:
        if vault_dir is None:
            vault_dir = Path(__file__).resolve().parent
        self.vault_dir = Path(vault_dir)
        self.backend = backend.lower().strip()
        self.skills_dir = self.vault_dir / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.golden_dir = self.vault_dir / "golden"
        self.db_path = self.vault_dir / "skills.db"
        self._entries: Dict[str, List[SkillCacheEntry]] = {}
        self._conn: Optional[Any] = None
        self._lock = threading.RLock()
        if self.backend == "sqlite":
            self._init_sqlite()
        self._load_all()

    def _get_connection(self):
        """Returns the persistent/cached SQLite connection."""
        if self._conn is None:
            import sqlite3
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def _init_sqlite(self) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    entry_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    goal_hash TEXT NOT NULL,
                    goal_slug TEXT NOT NULL,
                    entry_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_goal_hash ON skills(goal_hash)")

    @staticmethod
    def slugify(text: str) -> str:
        s = re.sub(r'[^a-zA-Z0-9]+', '_', text.strip().lower()).strip('_')
        return s[:64] or "unnamed_goal"

    @staticmethod
    def hash_goal(goal: str) -> str:
        norm = re.sub(r'\s+', ' ', goal.strip().lower())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

    _ANCHOR_STOPWORDS: set = {
        "the", "a", "an", "this", "that", "it", "app", "application", "window",
        "icon", "menu", "button", "item",
        "results", "completed", "active", "for", "into", "with", "from",
        "some", "text", "file", "new", "player", "browser", "program", "software",
        "close", "open",
    }

    @staticmethod
    def _extract_anchor_from_text(text: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """Dynamically extract (window_title, process_name) from a text description.

        Returns:
            (window_title, process_name) tuple if a target app was found, else None.
            Start Menu maps to ("Start", "explorer.exe").
            Desktop maps to ("Windows Desktop", "explorer.exe").
            Any other app maps to (target.title(), f"{alphanum(target.lower())}.exe").
        """
        if not text:
            return None
        raw = str(text).strip()
        tl = raw.lower()
        sw = SkillCache._ANCHOR_STOPWORDS

        def _format_target(t: str) -> Tuple[str, str]:
            t_clean = t.strip()
            t_low = t_clean.lower()
            if t_low in ("start", "startmenu", "start_menu"):
                return "Start", "explorer.exe"
            if t_low in ("desktop", "windows desktop", "windows_desktop"):
                return "Windows Desktop", "explorer.exe"
            clean_proc = re.sub(r"[^a-zA-Z0-9]", "", t_low)
            return t_clean.title(), f"{clean_proc}.exe"

        # Generic shell: Start Menu
        if "start menu" in tl or "start_menu" in tl:
            return "Start", "explorer.exe"
        # Generic shell: Desktop
        if "desktop" in tl:
            return "Windows Desktop", "explorer.exe"

        # Pattern 1 – English verb prefix: launch|open|search|start|close|into|type|focus|run <App>
        m = re.search(
            r"(?:launch|open|search|start|close|into|type|focus|run|find)"
            r"\s+(?:(?:for|results\s+for|the|a|an|app|application|window|player|browser)\s+)*['\"]?([a-zA-Z0-9_\-\.]{2,30})['\"]?",
            raw,
            re.IGNORECASE,
        )
        if m and m.group(1).lower() not in sw:
            return _format_target(m.group(1))

        # Pattern 3 – <App> app|application|window|player|browser
        m = re.search(
            r"([a-zA-Z0-9_\-\.]{2,30})\s+(?:app|application|window|player|browser)\b",
            raw,
            re.IGNORECASE,
        )
        if m and m.group(1).lower() not in sw:
            return _format_target(m.group(1))

        # Pattern 4 – Bare single-token app name
        m = re.fullmatch(r"['\"]?([a-zA-Z0-9_\-\.]{2,30})['\"]?", raw)
        if m and m.group(1).lower() not in sw:
            return _format_target(m.group(1))

        return None

    def _load_all(self) -> None:
        with self._lock:
            self._entries.clear()
            if self.backend == "sqlite":
                if self.db_path.exists():
                    try:
                        conn = self._get_connection()
                        cursor = conn.cursor()
                        try:
                            cursor.execute("SELECT entry_json FROM skills ORDER BY created_at ASC")
                            for (row_json,) in cursor.fetchall():
                                entry = SkillCacheEntry.from_dict(json.loads(row_json))
                                self._entries.setdefault(entry.goal_hash, []).append(entry)
                        finally:
                            cursor.close()
                    except Exception as e:
                        logger.warning(f"Failed to load skills from SQLite {self.db_path}: {e}")
            else:
                if self.skills_dir.exists():
                    for file_path in self.skills_dir.glob("*.json"):
                        try:
                            data = json.loads(file_path.read_text(encoding="utf-8"))
                            entry = SkillCacheEntry.from_dict(data)
                            self._entries.setdefault(entry.goal_hash, []).append(entry)
                        except Exception as e:
                            logger.warning(f"Failed to load cached skill {file_path}: {e}")

        # Automatically load and sync verified golden trajectories from skills_vault/golden/*.json
        golden_dir = getattr(self, "golden_dir", self.vault_dir / "golden")
        if golden_dir.exists():
            for file_path in sorted(golden_dir.glob("*.json")):
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    if not (data.get("verified_success") is True or data.get("status") == "success"):
                        continue

                    goal = data.get("goal")
                    if not goal or not isinstance(goal, str) or not goal.strip():
                        continue

                    actions: List[Dict[str, Any]] = []
                    sft_demo = data.get("sft_demonstration")
                    if isinstance(sft_demo, dict) and sft_demo.get("trajectory"):
                        actions = sft_demo["trajectory"]
                    elif "actions" in data and isinstance(data["actions"], list):
                        actions = data["actions"]
                    elif "steps" in data and isinstance(data["steps"], list):
                        for s in data["steps"]:
                            if isinstance(s, dict) and s.get("action"):
                                act = s["action"]
                                if isinstance(act, dict):
                                    actions.append(act)
                                elif hasattr(act, "to_dict"):
                                    actions.append(act.to_dict())

                    if not actions:
                        continue

                    anchors: List[UIStateAnchor] = []
                    if "anchors" in data and isinstance(data["anchors"], list) and data["anchors"]:
                        for a in data["anchors"]:
                            if isinstance(a, dict):
                                anchors.append(UIStateAnchor.from_dict(a))
                            elif isinstance(a, UIStateAnchor):
                                anchors.append(a)

                    if not anchors or all(a.is_empty() for a in anchors):
                        window_title = None
                        process_name = None
                        # Dynamically extract anchor from subgoal text (expected_ui_state / description)
                        for sg in data.get("metadata", {}).get("subgoals", []):
                            if isinstance(sg, dict):
                                for text in (sg.get("expected_ui_state", ""), sg.get("description", "")):
                                    result = SkillCache._extract_anchor_from_text(text)
                                    if result is not None:
                                        window_title, process_name = result
                                        break
                            if window_title:
                                break

                        if not window_title:
                            # Fall back to dynamic extraction from the goal string
                            result = SkillCache._extract_anchor_from_text(goal)
                            if result is not None:
                                window_title, process_name = result
                            else:
                                window_title = "Active Desktop"

                        v_hash = None
                        img_path = None
                        if data.get("screenshot_file"):
                            p = golden_dir / data["screenshot_file"]
                            if p.exists():
                                img_path = p
                        if img_path is None:
                            for ext in (".jpeg", ".jpg", ".png"):
                                p = file_path.with_suffix(ext)
                                if p.exists():
                                    img_path = p
                                    break
                        if img_path is not None:
                            try:
                                v_hash = compute_visual_hash(img_path.read_bytes())
                            except Exception:
                                v_hash = None

                        if v_hash:
                            anchors.append(UIStateAnchor(window_title=window_title, process_name=process_name, visual_hash=v_hash))
                        else:
                            anchors.append(UIStateAnchor(window_title=window_title, process_name=process_name))

                    g_hash = self.hash_goal(goal)
                    g_slug = self.slugify(goal)
                    entry_id = str(data.get("session_id") or data.get("entry_id") or file_path.stem[:8])
                    created_at = float(data.get("created_at", time.time()))

                    entry = SkillCacheEntry(
                        entry_id=entry_id,
                        goal=goal,
                        goal_hash=g_hash,
                        goal_slug=g_slug,
                        anchors=anchors,
                        actions=actions,
                        created_at=created_at,
                        metadata=data.get("metadata", {}),
                    )

                    existing = self._entries.setdefault(g_hash, [])
                    if not any(e.entry_id == entry_id for e in existing):
                        existing.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to load golden trajectory {file_path}: {e}")

    def save_skill(
        self,
        goal: str,
        actions: List[Union[Action, Dict[str, Any]]],
        anchors: Union[UIStateAnchor, List[UIStateAnchor]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stores a skill trajectory with required UI State Anchors.

        Enforces P0:
        1. Non-empty goal required.
        2. At least one action required.
        3. At least one non-empty UIStateAnchor constraint required.
        """
        if not goal or not goal.strip():
            raise ValueError("Goal cannot be empty when storing skill.")

        act_dicts = []
        for a in actions:
            if isinstance(a, Action):
                act_dicts.append(a.to_dict())
            elif isinstance(a, dict):
                act_dicts.append(a)
            else:
                raise TypeError(f"Unsupported action type: {type(a)}")

        if not act_dicts:
            raise ValueError("P0 Violation: At least one action is required to cache a skill trajectory.")

        anchor_list = [anchors] if isinstance(anchors, UIStateAnchor) else list(anchors)
        if not anchor_list or all(a.is_empty() for a in anchor_list):
            raise ValueError("P0 Violation: At least one non-empty UIStateAnchor constraint is required to cache a skill.")

        g_hash = self.hash_goal(goal)
        g_slug = self.slugify(goal)
        created_time = time.time()
        
        # Unique entry id to ensure multiple skills per goal are never overwritten
        entry_seed = f"{g_hash}_{len(self._entries.get(g_hash, []))}_{created_time}_{json.dumps(act_dicts, sort_keys=True)}"
        entry_id = hashlib.sha256(entry_seed.encode("utf-8")).hexdigest()[:8]

        entry = SkillCacheEntry(
            entry_id=entry_id,
            goal=goal,
            goal_hash=g_hash,
            goal_slug=g_slug,
            anchors=anchor_list,
            actions=act_dicts,
            created_at=created_time,
            metadata=metadata or {},
        )

        with self._lock:
            if self.backend == "sqlite":
                self._init_sqlite()
                conn = self._get_connection()
                with conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO skills (entry_id, goal, goal_hash, goal_slug, entry_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (entry_id, goal, g_hash, g_slug, json.dumps(entry.to_dict()), created_time),
                    )
                self._entries.setdefault(g_hash, []).append(entry)
                logger.info(f"Cached skill saved in SQLite for '{goal}' ({len(act_dicts)} actions, {len(anchor_list)} anchors) -> id={entry_id}")
                return f"sqlite://{self.db_path}#skills/{entry_id}"

            filename = f"{g_hash}_{entry_id}_{g_slug}.json"
            target_path = self.skills_dir / filename
            target_path.write_text(json.dumps(entry.to_dict(), indent=2), encoding="utf-8")

            self._entries.setdefault(g_hash, []).append(entry)
            logger.info(f"Cached skill saved for '{goal}' ({len(act_dicts)} actions, {len(anchor_list)} anchors) -> {filename}")
            return str(target_path)

    def lookup(
        self,
        goal: str,
        current_anchor: Optional[UIStateAnchor] = None,
        tolerance_px: int = 15,
    ) -> SkillLookupResult:
        """Looks up a skill with mandatory UI State Anchor verification.

        P0 Rule:
        If current_anchor is None or does not match the cached anchors, returns a cache miss.
        Only returns a cache hit when both the goal match AND anchors are verified.
        """
        g_hash = self.hash_goal(goal)
        entries = self._entries.get(g_hash, [])
        if not entries:
            return SkillLookupResult(
                hit=False,
                reason=f"Cache miss: Goal '{goal}' not found in skill cache.",
            )

        # Enforce P0 Rule: NEVER blindly replay on semantic goal hash alone
        if current_anchor is None or current_anchor.is_empty():
            return SkillLookupResult(
                hit=False,
                reason="Cache miss: No valid UI State Anchor provided. P0 rule strictly forbids blind replay on semantic goal hash alone.",
                entry=entries[0],
            )

        # Check if current UI state matches any registered anchor for this goal
        for entry in entries:
            for anchor in entry.anchors:
                if anchor.matches(current_anchor, tolerance_px=tolerance_px):
                    action_objs = [Action.from_dict(d) for d in entry.actions]
                    return SkillLookupResult(
                        hit=True,
                        actions=action_objs,
                        reason=f"Cache hit: UI State Anchor verified for '{goal}'. Replaying {len(action_objs)} cached actions with 0ms model latency.",
                        entry=entry,
                    )

        return SkillLookupResult(
            hit=False,
            reason=f"Cache miss: UI State Anchor mismatch for goal '{goal}'. Live desktop state differs from cached anchor. Fallback to dynamic execution.",
            entry=entries[0],
        )

    def is_anchor_drift(
        self,
        goal: str,
        current_anchor: Optional[UIStateAnchor] = None,
        tolerance_px: int = 15,
    ) -> bool:
        """Returns True if the goal exists in cache but the UI State Anchor diverges (anchor drift)."""
        g_hash = self.hash_goal(goal)
        entries = self._entries.get(g_hash, [])
        if not entries:
            return False
        if current_anchor is None or current_anchor.is_empty():
            return True
        for entry in entries:
            for anchor in entry.anchors:
                if anchor.matches(current_anchor, tolerance_px=tolerance_px):
                    return False
        return True

    def get_trajectory(
        self,
        goal: str,
        current_anchor: Optional[UIStateAnchor] = None,
        tolerance_px: int = 15,
    ) -> Optional[List[Action]]:
        """Convenience method returning cached actions list on anchor-verified hit, else None."""
        res = self.lookup(goal, current_anchor=current_anchor, tolerance_px=tolerance_px)
        return res.actions if res.hit else None

    def list_skills(self) -> List[Dict[str, Any]]:
        """Lists all registered cached skills in the vault."""
        skills = []
        for entries in self._entries.values():
            for e in entries:
                skills.append({
                    "entry_id": e.entry_id,
                    "goal": e.goal,
                    "goal_hash": e.goal_hash,
                    "actions_count": len(e.actions),
                    "anchors_count": len(e.anchors),
                    "created_at": e.created_at,
                })
        return skills

    def close(self) -> None:
        """Closes any persistent or cached SQLite connection, releasing OS file locks."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> SkillCache:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def clear(self) -> None:
        """Clears memory and disk cache."""
        with self._lock:
            self._entries.clear()
            if self.skills_dir.exists():
                for f in self.skills_dir.glob("*.json"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            if self._conn is not None:
                try:
                    with self._conn:
                        self._conn.execute("DELETE FROM skills")
                except Exception:
                    pass
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = self.db_path.with_name(self.db_path.name + suffix)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


# Canonical Aliases
AnchorVerifiedSkillCache = SkillCache
SkillRecord = SkillCacheEntry

