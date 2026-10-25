"""Arm D2 Multi-Generator PyTorch Dataset and DataLoader for AVCS Streaming V2.

Part of StreamingVisualCortexV2 Milestone M2: Multi-Generator Dataset Pipeline.

Key Features:
1. Arm D2 Balanced Sampling:
   - Equalized sampling weight between `ui_grounding` and `cleanroom_system1_v1` per batch.
   - WeightedRandomSampler with inverse-frequency weights ensures ~50% UI Grounding and ~50%
     Cleanroom draw probability in expectation over an epoch of size (N_ui + N_cleanroom).
2. Native Augmentation Suite:
   - 50% photometric color inversion (I' = 1.0 - I).
   - Brightness jitter (+-20%).
   - Contrast jitter (0.8 - 1.2x).
   - Gamma jitter (0.8 - 1.2).
   - Photometric invariance preserves geometric bounding boxes and centroids exactly.
3. Hard-Negative Injection:
   - Injects 15-20% blank screens and static wallpapers into training batches with has_target = 0.0.
4. Unified 5-Class Collation:
   - Pads target boxes, classes, and centroids into fixed batch tensors with boolean box masks.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from avcs.dataset.unified.generator_harmonizer import (
    GENERATOR_CLEANROOM,
    GENERATOR_UI_GROUNDING,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
    SPLIT_TRAIN,
    SPLIT_VAL,
    UnifiedFrame,
    UnifiedManifest,
    harmonize_all_generators,
    normalize_split_name,
)

GENERATOR_ID_UI: int = 0
GENERATOR_ID_CLEANROOM: int = 1
GENERATOR_NAME_TO_ID: Dict[str, int] = {
    GENERATOR_UI_GROUNDING: GENERATOR_ID_UI,
    GENERATOR_CLEANROOM: GENERATOR_ID_CLEANROOM,
}
GENERATOR_ID_TO_NAME: Dict[int, str] = {
    GENERATOR_ID_UI: GENERATOR_UI_GROUNDING,
    GENERATOR_ID_CLEANROOM: GENERATOR_CLEANROOM,
}


def calculate_arm_d2_weights(ui_count: int, cleanroom_count: int) -> Tuple[float, float]:
    """Calculate inverse-frequency sampling weights for Arm D2 balanced sampling.

    Args:
        ui_count: number of frames from ui_grounding
        cleanroom_count: number of frames from cleanroom_system1_v1

    Returns:
        (w_ui, w_cleanroom) such that P(UI) = 0.5 and P(Cleanroom) = 0.5.
    """
    total = ui_count + cleanroom_count
    if total == 0:
        return 0.0, 0.0
    w_ui = total / (2.0 * ui_count) if ui_count > 0 else 0.0
    w_cleanroom = total / (2.0 * cleanroom_count) if cleanroom_count > 0 else 0.0
    return w_ui, w_cleanroom


def apply_photometric_augmentations(
    image: torch.Tensor,
    inversion_prob: float = 0.5,
    brightness_range: Tuple[float, float] = (-0.2, 0.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Apply native photometric augmentations to an image tensor [3, H, W] in [0.0, 1.0].

    Preserves spatial coordinates of bounding boxes and centroids.
    """
    r = rng or random.Random()
    img = image.clone()

    # 1. 50% Photometric color inversion
    if r.random() < inversion_prob:
        img = 1.0 - img

    # 2. Brightness jitter (+-20%)
    b_delta = r.uniform(brightness_range[0], brightness_range[1])
    img = torch.clamp(img + b_delta, 0.0, 1.0)

    # 3. Contrast jitter (0.8 - 1.2x)
    c_alpha = r.uniform(contrast_range[0], contrast_range[1])
    mean = img.mean(dim=(-2, -1), keepdim=True)
    img = torch.clamp(mean + c_alpha * (img - mean), 0.0, 1.0)

    # 4. Gamma jitter (0.8 - 1.2)
    gamma = r.uniform(gamma_range[0], gamma_range[1])
    img = torch.clamp(torch.pow(img + 1e-8, gamma), 0.0, 1.0)

    return img


def generate_hard_negative_screen(
    size: Tuple[int, int] = (128, 128),
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Generate high-gradient hard negative screen across 6 distinct procedural regimes with zero UI targets.

    Regimes:
      1. Wallpaper simulator (sky gradients, celestial circular disks, star points, mountain ridges)
      2. Multi-scale Gabor / Sinusoids (oriented sinusoidal waves and high-gradient stripe patterns)
      3. Fractal / Spectral noise (multi-octave 1/f^beta noise textures)
      4. Geometric grids / checkerboards (high-frequency grid lines and empty UI-like box boundaries)
      5. Gaussian / salt-and-pepper noise (random pixel noise across all channels)
      6. Flat solids & linear gradients (smooth screens)
    """
    r = rng or random.Random()
    w, h = size
    regime = r.choice([
        "wallpaper_simulator",
        "gabor_sinusoids",
        "fractal_noise",
        "geometric_grids",
        "random_noise",
        "flat_gradients",
    ])

    if regime == "wallpaper_simulator":
        # Mode 1: Wallpaper simulator (matching composer.py:jittered_draw_background distribution)
        palettes = [
            ((18, 24, 48), (78, 42, 85), (196, 92, 54), (45, 25, 55), (20, 15, 30)),
            ((15, 25, 50), (45, 65, 110), (220, 140, 70), (40, 45, 70), (25, 25, 45)),
            ((8, 12, 28), (20, 45, 55), (40, 140, 110), (18, 28, 42), (10, 16, 26)),
            ((25, 10, 40), (80, 20, 70), (180, 50, 110), (45, 15, 45), (22, 10, 28)),
            ((15, 30, 28), (35, 65, 50), (120, 150, 110), (30, 50, 40), (18, 30, 24)),
        ]
        sky_top, sky_mid, sky_low, r1_c, r2_c = r.choice(palettes)
        j_amt = r.randint(-8, 8)
        sky_top = tuple(max(0, min(255, c + j_amt)) for c in sky_top)
        sky_mid = tuple(max(0, min(255, c + j_amt)) for c in sky_mid)
        sky_low = tuple(max(0, min(255, c + j_amt)) for c in sky_low)

        img = Image.new("RGB", size)
        draw = ImageDraw.Draw(img)

        split_y = int(h * r.uniform(0.60, 0.74))
        for y in range(split_y):
            t = y / max(1, split_y)
            if t < 0.5:
                u = t * 2.0
                c = tuple(int(sky_top[i] * (1 - u) + sky_mid[i] * u) for i in range(3))
            else:
                u = (t - 0.5) * 2.0
                c = tuple(int(sky_mid[i] * (1 - u) + sky_low[i] * u) for i in range(3))
            draw.line([(0, y), (w, y)], fill=c)

        # Celestial body (sun / moon of radius 8-18px)
        if r.random() < 0.90:
            sun_x = r.randint(int(w * 0.18), int(w * 0.82))
            sun_y = r.randint(int(h * 0.12), int(h * 0.42))
            sun_r = r.randint(8, 18)
            sun_color = r.choice([(255, 230, 180), (255, 200, 120), (220, 240, 255), (255, 160, 80)])
            draw.ellipse([sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r], fill=sun_color)

        # Star field points
        num_stars = r.randint(6, 26)
        for _ in range(num_stars):
            sx = r.randint(2, w - 3)
            sy = r.randint(2, int(split_y * 0.7))
            s_bright = r.randint(180, 255)
            draw.point((sx, sy), fill=(s_bright, s_bright, s_bright))

        # Distant mountain ridge
        phase1 = r.uniform(0.0, 6.28)
        freq1 = r.uniform(0.03, 0.06)
        amp1 = r.uniform(0.05, 0.10) * h
        base_y1 = h * r.uniform(0.55, 0.65)
        ridge_pts = [(0, int(base_y1))]
        for x in range(0, w + 16, 16):
            ry = int(base_y1 + math.sin(x * freq1 + phase1) * amp1 + math.cos(x * 0.02) * (h * 0.03))
            ridge_pts.append((x, ry))
        ridge_pts.extend([(w, h), (0, h)])
        draw.polygon(ridge_pts, fill=r1_c)

        # Foreground mountain ridge
        phase2 = r.uniform(0.0, 6.28)
        freq2 = r.uniform(0.05, 0.09)
        amp2 = r.uniform(0.04, 0.08) * h
        base_y2 = h * r.uniform(0.68, 0.78)
        fg_pts = [(0, int(base_y2))]
        for x in range(0, w + 16, 16):
            ry = int(base_y2 + math.sin(x * freq2 + phase2) * amp2)
            fg_pts.append((x, ry))
        fg_pts.extend([(w, h), (0, h)])
        draw.polygon(fg_pts, fill=r2_c)

        arr = np.array(img, dtype=np.float32) / 255.0

    elif regime == "gabor_sinusoids":
        # Mode 2: Multi-scale Gabor / oriented sinusoids & high-gradient stripe patterns
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        wave_sum = np.zeros((h, w), dtype=np.float32)
        k_count = r.randint(2, 5)
        for _ in range(k_count):
            theta = r.uniform(0.0, math.pi)
            freq = r.uniform(0.02, 0.25)
            phase = r.uniform(0.0, 2.0 * math.pi)
            wave_sum += np.sin(2.0 * math.pi * (freq * np.cos(theta) * x + freq * np.sin(theta) * y) + phase)
        wave_norm = (wave_sum - wave_sum.min()) / (wave_sum.max() - wave_sum.min() + 1e-6)
        tint = np.array([r.uniform(0.2, 0.9), r.uniform(0.2, 0.9), r.uniform(0.2, 0.9)], dtype=np.float32)
        arr = np.clip(wave_norm[:, :, None] * tint[None, None, :], 0.0, 1.0)

    elif regime == "fractal_noise":
        # Mode 3: 2D spectral 1/f^beta fractal noise in frequency domain
        beta = r.uniform(1.0, 2.2)
        u = np.fft.fftfreq(h)[:, None]
        v = np.fft.fftfreq(w)[None, :]
        radius = np.sqrt(u**2 + v**2) + 1e-6
        phase_mat = np.exp(1j * r.uniform(0, 2 * math.pi) * np.random.randn(h, w))
        spectrum = (radius ** (-beta)) * phase_mat
        noise = np.real(np.fft.ifft2(spectrum)).astype(np.float32)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        c = np.array([r.uniform(0.2, 0.8), r.uniform(0.2, 0.8), r.uniform(0.2, 0.8)], dtype=np.float32)
        arr = np.clip(noise[:, :, None] * c[None, None, :], 0.0, 1.0)

    elif regime == "geometric_grids":
        # Mode 4: Geometric grids, checkerboards, hatched lines, and empty UI box outlines
        sub = r.choice(["checker", "hatched", "empty_boxes"])
        if sub == "checker":
            b = r.choice([4, 8, 16])
            y, x = np.mgrid[0:h, 0:w]
            cb = ((x // b) + (y // b)) % 2
            c1 = np.array([r.uniform(0.1, 0.3), r.uniform(0.1, 0.3), r.uniform(0.1, 0.3)], dtype=np.float32)
            c2 = np.array([r.uniform(0.5, 0.8), r.uniform(0.5, 0.8), r.uniform(0.5, 0.8)], dtype=np.float32)
            arr = cb[:, :, None] * c1 + (1 - cb)[:, :, None] * c2
        elif sub == "hatched":
            arr = np.zeros((h, w, 3), dtype=np.float32) + r.uniform(0.1, 0.3)
            d = r.choice([6, 8, 12])
            for i in range(0, w + h, d):
                for xi in range(w):
                    yi = i - xi
                    if 0 <= yi < h:
                        arr[yi, xi] = r.uniform(0.6, 0.9)
        else:
            arr = np.zeros((h, w, 3), dtype=np.float32) + r.uniform(0.1, 0.25)
            for _ in range(r.randint(2, 5)):
                bx = r.randint(4, max(5, w - 40))
                by = r.randint(4, max(5, h - 30))
                bw = r.randint(15, 35)
                bh = r.randint(10, 25)
                arr[by:min(h, by+bh), bx:min(w, bx+bw)] = r.uniform(0.3, 0.5)
                arr[by+2:min(h, by+bh-2), bx+2:min(w, bx+bw-2)] = r.uniform(0.1, 0.25)

    elif regime == "random_noise":
        # Mode 5: Gaussian and salt-and-pepper noise
        sub = r.choice(["gaussian", "salt_pepper"])
        if sub == "gaussian":
            mean_n = r.uniform(0.3, 0.7)
            std_n = r.uniform(0.08, 0.22)
            arr = np.clip(mean_n + std_n * np.random.randn(h, w, 3).astype(np.float32), 0.0, 1.0)
        else:
            arr = np.full((h, w, 3), r.uniform(0.3, 0.6), dtype=np.float32)
            mask_sp = np.random.rand(h, w) < r.uniform(0.02, 0.06)
            arr[mask_sp] = np.random.choice([0.0, 1.0], size=(int(np.sum(mask_sp)), 3))

    else:
        # Mode 6: Flat solids & smooth linear gradients
        if r.random() < 0.5:
            color = np.array([r.randint(15, 240), r.randint(15, 240), r.randint(15, 240)], dtype=np.float32) / 255.0
            arr = np.tile(color[None, None, :], (h, w, 1))
        else:
            c1 = np.array([r.randint(10, 80), r.randint(10, 80), r.randint(30, 120)], dtype=np.float32) / 255.0
            c2 = np.array([r.randint(40, 180), r.randint(40, 180), r.randint(80, 240)], dtype=np.float32) / 255.0
            t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
            arr = np.tile((1.0 - t) * c1[None, None, :] + t * c2[None, None, :], (1, w, 1))

    return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)  # [3, H, W]


class ArmD2MultiGeneratorDataset(Dataset):
    """PyTorch Dataset implementing Arm D2 balanced multi-generator loading."""

    def __init__(
        self,
        manifest: Optional[UnifiedManifest] = None,
        split: str = SPLIT_TRAIN,
        target_image_size: Tuple[int, int] = (128, 128),
        apply_augmentation: bool = True,
        hard_negative_prob: float = 0.175,  # 15-20% default 0.175 for train
        preload_images: bool = True,
        rng_seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self.split = normalize_split_name(split)
        self.target_image_size = target_image_size
        self.apply_augmentation = apply_augmentation and (self.split == SPLIT_TRAIN)
        self.hard_negative_prob = hard_negative_prob if (self.split == SPLIT_TRAIN) else 0.0
        self.preload_images = preload_images
        self.rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        # Load or filter manifest
        if manifest is None:
            full_manifest = harmonize_all_generators()
            self.manifest = full_manifest.filter_split(self.split)
        else:
            self.manifest = manifest.filter_split(self.split)

        self.frames: List[UnifiedFrame] = list(self.manifest.frames)
        self._image_cache: Dict[str, torch.Tensor] = {}

        # Precompute per-sample records and Arm D2 sample weights
        self._samples: List[Dict[str, Any]] = []
        self._sample_weights: List[float] = []

        ui_frames = [f for f in self.frames if f.generator_source == GENERATOR_UI_GROUNDING]
        cr_frames = [f for f in self.frames if f.generator_source == GENERATOR_CLEANROOM]
        w_ui, w_cr = calculate_arm_d2_weights(len(ui_frames), len(cr_frames))

        for frame in self.frames:
            gen_id = GENERATOR_NAME_TO_ID.get(frame.generator_source, GENERATOR_ID_UI)
            weight = w_ui if gen_id == GENERATOR_ID_UI else w_cr

            is_neg = frame.is_negative_frame or len(frame.objects) == 0

            if is_neg:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                classes = torch.zeros((0,), dtype=torch.int64)
                centroids = torch.zeros((0, 2), dtype=torch.float32)
                has_target = 0.0
            else:
                b_list = [list(o.bbox_xyxy_norm) for o in frame.objects]
                c_list = [o.class_id for o in frame.objects]
                cen_list = [list(o.centroid_xy_norm) for o in frame.objects]
                boxes = torch.tensor(b_list, dtype=torch.float32)
                classes = torch.tensor(c_list, dtype=torch.int64)
                centroids = torch.tensor(cen_list, dtype=torch.float32)
                has_target = 1.0

            sample = {
                "frame_id": frame.frame_id,
                "generator_id": gen_id,
                "generator_source": frame.generator_source,
                "image_path": frame.image_path,
                "target_boxes": boxes,
                "target_classes": classes,
                "target_centroids": centroids,
                "has_target": float(has_target),
                "is_negative": is_neg,
            }
            self._samples.append(sample)
            self._sample_weights.append(weight)

            if self.preload_images:
                self._load_image_tensor(frame.frame_id, frame.image_path)

    def _load_image_tensor(self, frame_id: str, image_path: str) -> torch.Tensor:
        """Load and resize image from disk into memory cache."""
        if frame_id in self._image_cache:
            return self._image_cache[frame_id]

        p = Path(image_path)
        if p.exists():
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                if img_rgb.size != self.target_image_size:
                    img_rgb = img_rgb.resize(self.target_image_size, Image.Resampling.BILINEAR)
                arr = np.array(img_rgb, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        else:
            # Fallback synthetic frame if image not on disk
            tensor = generate_hard_negative_screen(self.target_image_size, self.rng)

        if self.preload_images:
            self._image_cache[frame_id] = tensor
        return tensor

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        img = self._load_image_tensor(sample["frame_id"], sample["image_path"]).clone()

        # Hard-negative injection for training split (15-20% blank screens)
        if self.hard_negative_prob > 0.0 and self.rng.random() < self.hard_negative_prob:
            img = generate_hard_negative_screen(self.target_image_size, self.rng)
            return {
                "frame_id": f"{sample['frame_id']}_injected_neg",
                "image": img,
                "target_boxes": torch.zeros((0, 4), dtype=torch.float32),
                "target_classes": torch.zeros((0,), dtype=torch.int64),
                "target_centroids": torch.zeros((0, 2), dtype=torch.float32),
                "has_target": 0.0,
                "generator_id": sample["generator_id"],
                "is_negative": True,
            }

        # Apply photometric augmentations
        if self.apply_augmentation:
            img = apply_photometric_augmentations(img, rng=self.rng)

        return {
            "frame_id": sample["frame_id"],
            "image": img,
            "target_boxes": sample["target_boxes"].clone(),
            "target_classes": sample["target_classes"].clone(),
            "target_centroids": sample["target_centroids"].clone(),
            "has_target": sample["has_target"],
            "generator_id": sample["generator_id"],
            "is_negative": sample["is_negative"],
        }

    def get_arm_d2_sampler(self) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler configured for Arm D2 balanced draws."""
        weights = torch.tensor(self._sample_weights, dtype=torch.double)
        num_samples = len(self._samples)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
        )


def collate_multi_generator_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate sample dictionaries into unified padded batch tensors."""
    images = torch.stack([s["image"] for s in batch], dim=0)  # [B, 3, H, W]
    has_target = torch.tensor([s["has_target"] for s in batch], dtype=torch.float32)  # [B]
    generator_id = torch.tensor([s["generator_id"] for s in batch], dtype=torch.int64)  # [B]
    is_neg = torch.tensor([s["is_negative"] for s in batch], dtype=torch.bool)  # [B]

    # Find max objects in batch for padding
    max_objs = max(s["target_boxes"].shape[0] for s in batch)
    max_objs = max(1, max_objs)

    b_size = len(batch)
    padded_boxes = torch.zeros((b_size, max_objs, 4), dtype=torch.float32)
    padded_classes = torch.full((b_size, max_objs), -1, dtype=torch.int64)
    padded_centroids = torch.zeros((b_size, max_objs, 2), dtype=torch.float32)
    box_mask = torch.zeros((b_size, max_objs), dtype=torch.bool)

    for i, s in enumerate(batch):
        n = s["target_boxes"].shape[0]
        if n > 0:
            padded_boxes[i, :n] = s["target_boxes"]
            padded_classes[i, :n] = s["target_classes"]
            padded_centroids[i, :n] = s["target_centroids"]
            box_mask[i, :n] = True

    return {
        "images": images,
        "target_boxes": padded_boxes,
        "target_classes": padded_classes,
        "target_centroids": padded_centroids,
        "box_mask": box_mask,
        "has_target": has_target,
        "generator_id": generator_id,
        "is_negative": is_neg,
    }


def create_arm_d2_dataloader(
    manifest: Optional[UnifiedManifest] = None,
    split: str = SPLIT_TRAIN,
    batch_size: int = 16,
    target_image_size: Tuple[int, int] = (128, 128),
    apply_augmentation: bool = True,
    hard_negative_prob: float = 0.175,
    num_workers: int = 0,
    rng_seed: Optional[int] = 42,
) -> DataLoader:
    """Factory creating an Arm D2 balanced DataLoader."""
    dataset = ArmD2MultiGeneratorDataset(
        manifest=manifest,
        split=split,
        target_image_size=target_image_size,
        apply_augmentation=apply_augmentation,
        hard_negative_prob=hard_negative_prob,
        rng_seed=rng_seed,
    )

    if dataset.split == SPLIT_TRAIN and len(dataset) > 0:
        sampler: Optional[Sampler] = dataset.get_arm_d2_sampler()
        shuffle = False
    else:
        sampler = None
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_multi_generator_batch,
        num_workers=num_workers,
    )
