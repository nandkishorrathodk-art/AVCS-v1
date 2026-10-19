"""PyTorch Dataset & DataLoaders for AVCS Training."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from forge.vision.cortex.core.learned import ACTION_NAMES, SCENE_ATTRIBUTES
from forge.vision.cortex.data.synthetic import ProceduralScreenGenerator, SyntheticScreenData

MAX_OBJECT_SLOTS = 16


def _connected_components(grid: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """4-connected component labelling without scipy.

    `grid` is a small (H, W) int array of class indices. Returns bounding boxes
    as (class_idx, x0, y0, x1, y1) for every non-zero component, largest first.
    """
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    boxes: List[Tuple[int, int, int, int, int]] = []
    for y0 in range(H):
        for x0 in range(W):
            if grid[y0, x0] == 0 or seen[y0, x0]:
                continue
            cls = int(grid[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            min_y = max_y = y0
            min_x = max_x = x0
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] == cls:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if size >= 2:
                boxes.append((cls, min_x, min_y, max_x, max_y))
    boxes.sort(key=lambda b: (b[3] - b[1] + 1) * (b[4] - b[2] + 1), reverse=True)
    return boxes


def _scene_attributes(data: SyntheticScreenData) -> np.ndarray:
    """Multi-label attribute target derived from the ground-truth scene."""
    mask = data.affordance_mask
    h, w = mask.shape
    clickable = float((mask == 1).sum()) / (h * w)
    typeable = float((mask == 2).sum()) / (h * w)
    scrollable = float((mask == 3).sum()) / (h * w)
    bright = float(data.frame.mean()) / 255.0

    values = {
        "has_text": 1.0 if data.category in ("editor_code", "terminal_shell", "browser_webpage") else 0.0,
        "has_toolbar": 1.0 if data.category != "desktop_canvas" else 0.0,
        "has_modal": 1.0 if data.category == "modal_dialog" else 0.0,
        "has_input_field": 1.0 if typeable > 0.01 else 0.0,
        "has_list": 1.0 if scrollable > 0.01 else 0.0,
        "has_image": 1.0 if data.category == "media_playback" else 0.0,
        "is_dark_theme": 1.0 if bright < 0.35 else 0.0,
        "is_dense": 1.0 if (clickable + typeable) > 0.15 else 0.0,
        "has_cursor": 1.0,
        "has_animation": 0.0,
    }
    return np.array([values[a] for a in SCENE_ATTRIBUTES], dtype=np.float32)


class ScreenDataset(Dataset):
    """Generates synthetic screen samples for supervised and self-supervised training."""

    def __init__(
        self,
        num_samples: int = 1000,
        width: int = 640,
        height: int = 360,
        latent_dim: int = 256,
        affordance_grid_size: tuple[int, int] = (32, 32),
    ) -> None:
        self.num_samples = num_samples
        self.width = width
        self.height = height
        self.latent_dim = latent_dim
        self.affordance_grid_size = affordance_grid_size
        self.generator = ProceduralScreenGenerator(width=width, height=height)

    def __len__(self) -> int:
        return self.num_samples

    def _object_targets(self, data: SyntheticScreenData) -> dict[str, torch.Tensor]:
        """Builds slot-detection targets from the ground-truth affordance mask."""
        mask_t = torch.from_numpy(data.affordance_mask).unsqueeze(0).unsqueeze(0).float()
        grid = F.interpolate(mask_t, size=self.affordance_grid_size, mode="nearest").squeeze().long().numpy()

        boxes = _connected_components(grid)
        H, W = self.affordance_grid_size
        out_boxes = torch.zeros(MAX_OBJECT_SLOTS, 4, dtype=torch.float32)
        out_classes = torch.zeros(MAX_OBJECT_SLOTS, dtype=torch.long)
        out_present = torch.zeros(MAX_OBJECT_SLOTS, dtype=torch.float32)

        for i, (cls, x0, y0, x1, y1) in enumerate(boxes[:MAX_OBJECT_SLOTS]):
            cx = (x0 + x1 + 1) / 2.0 / W
            cy = (y0 + y1 + 1) / 2.0 / H
            bw = (x1 - x0 + 1) / W
            bh = (y1 - y0 + 1) / H
            out_boxes[i] = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
            out_classes[i] = cls
            out_present[i] = 1.0

        return {
            "object_boxes": out_boxes,
            "object_classes": out_classes,
            "object_present": out_present,
            "object_count": torch.tensor(float(len(boxes)), dtype=torch.float32),
        }

    def _action_target(self, data: SyntheticScreenData) -> dict[str, torch.Tensor]:
        """Learned action target: which action, and where to apply it."""
        mask = data.affordance_mask
        has_typeable = bool((mask == 2).any())
        has_clickable = bool((mask == 1).any())
        has_scrollable = bool((mask == 3).any())

        if has_clickable:
            action = ACTION_NAMES.index("click")
        elif has_typeable:
            action = ACTION_NAMES.index("type")
        elif has_scrollable:
            action = ACTION_NAMES.index("scroll")
        else:
            action = ACTION_NAMES.index("noop")

        if data.interactive_points:
            pt = data.interactive_points[0]
            coord = torch.tensor([pt[0], pt[1]], dtype=torch.float32)
        else:
            coord = torch.tensor([0.5, 0.5], dtype=torch.float32)

        return {
            "action_idx": torch.tensor(action, dtype=torch.long),
            "action_coord": coord,
        }

    def _stability_target(self, frame_t: np.ndarray, frame_next: np.ndarray) -> torch.Tensor:
        """Learned stability target from the real inter-frame change magnitude."""
        diff = np.abs(frame_t.astype(np.float32) - frame_next.astype(np.float32)).mean() / 255.0
        if diff < 0.005:
            idx = 0  # stable
        elif diff < 0.02:
            idx = 1  # animating
        elif diff < 0.08:
            idx = 2  # transitioning
        else:
            idx = 3  # surprise_event
        return torch.tensor(idx, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Generate current frame
        data = self.generator.generate(seed=idx)

        # Temporal successor of the *same* scene (cursor nudge / modal emergence).
        # Re-seeding the generator would produce an unrelated scene and the world
        # model would only ever learn identity.
        data_next = self.generator.generate_successor(data)

        # Convert frame to Float Tensor (3, H, W) normalized to [0, 1]
        frame_t = torch.from_numpy(data.frame).permute(2, 0, 1).float() / 255.0
        frame_next = torch.from_numpy(data_next.frame).permute(2, 0, 1).float() / 255.0

        # Downsample affordance mask to affordance_grid_size using nearest neighbor
        mask_t = torch.from_numpy(data.affordance_mask).unsqueeze(0).unsqueeze(0).float()
        mask_down = F.interpolate(mask_t, size=self.affordance_grid_size, mode="nearest").squeeze().long()

        # Target grounding coordinate from first interactive point (if available)
        if data.interactive_points:
            pt = data.interactive_points[0]
            target_coord = torch.tensor([pt[0], pt[1]], dtype=torch.float32)
        else:
            target_coord = torch.tensor([0.5, 0.5], dtype=torch.float32)

        sample = {
            "frame_t": frame_t,
            "frame_next": frame_next,
            "affordance_mask": mask_down,
            "category_idx": torch.tensor(data.category_idx, dtype=torch.long),
            "target_coord": target_coord,
            "query_text": data.interactive_label(),
            "scene_attributes": torch.from_numpy(_scene_attributes(data)),
            "stability_target": self._stability_target(data.frame, data_next.frame),
        }
        sample.update(self._object_targets(data))
        sample.update(self._action_target(data))
        return sample


def create_dataloader(
    num_samples: int = 500,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ScreenDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

