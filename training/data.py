"""Dataset indexing and loading for training the fire segmentation model."""

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
SEG_DIR = ROOT / "dataset" / "segmentation"
NEG_DIRS = [
    ROOT / "dataset" / "detection" / "train" / "not_fire",
    ROOT / "dataset" / "detection" / "valid" / "not_fire",
]
BOWFIRE_DIR = ROOT / "external" / "BoWFireDataset" / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIDE = 800  # large photos are downscaled once at load time


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None  # None = image without fire (empty mask)
    group: str  # "videos", "bowfire" or "negatives"


def video_samples(video: int) -> list[Sample]:
    """Frame/mask pairs of one video. Frame NNNN.jpg pairs with *_GT_NNN.jpg.

    Matching is by frame number only, because the Video06 masks are named Video02_GT_*.
    """
    raw = SEG_DIR / "raw" / f"Video{video:02d}"
    masks = {
        int(p.stem.rsplit("_", 1)[1]): p
        for p in (SEG_DIR / "labelled" / f"Video{video:02d}_GT").glob("*.jpg")
    }
    samples = []
    for frame in sorted(raw.glob("*.jpg")):
        mask = masks.get(int(frame.stem))
        if mask is None:
            raise FileNotFoundError(f"No mask for {frame}")
        samples.append(Sample(frame, mask, "videos"))
    return samples


def negative_samples() -> list[Sample]:
    """Fire-free photos from the detection dataset (empty masks)."""
    paths = [p for d in NEG_DIRS for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return [Sample(p, None, "negatives") for p in paths]


def bowfire_samples() -> list[Sample]:
    """BoWFire photos (fire and fire-like non-fire) with ground-truth masks."""
    if not (BOWFIRE_DIR / "img").is_dir():
        raise FileNotFoundError(f"{BOWFIRE_DIR} not found; run `python scripts/download_bowfire.py` first")
    return [
        Sample(img, BOWFIRE_DIR / "gt" / f"{img.stem}_gt.png", "bowfire")
        for img in sorted((BOWFIRE_DIR / "img").glob("*.png"))
    ]


def _holdout(samples: list[Sample], fraction: float, rng: random.Random, key=None):
    """Split off ``fraction`` of samples, stratified by ``key``."""
    train, val = [], []
    strata: dict[str, list[Sample]] = {}
    for sample in samples:
        strata.setdefault(key(sample) if key else "", []).append(sample)
    for items in strata.values():
        items = items[:]
        rng.shuffle(items)
        n_val = round(len(items) * fraction)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def split(val_videos: list[int], holdout: bool = True, photo_repeat: int = 5, seed: int = 0):
    """Return (train samples, {group: val samples}).

    Videos are split at video level so frames of one video never appear in both sets.
    BoWFire and the fire-free photos are split 70/30 and 80/20. With ``holdout=False``
    everything goes into training (for the final model). BoWFire photos are repeated
    ``photo_repeat`` times in training so the few diverse photos are not drowned out by
    thousands of similar video frames.
    """
    rng = random.Random(seed)
    train: list[Sample] = []
    val: dict[str, list[Sample]] = {"videos": [], "bowfire": [], "negatives": []}
    for video in range(1, 13):
        (val["videos"] if holdout and video in val_videos else train).extend(video_samples(video))

    bowfire = bowfire_samples()
    negatives = negative_samples()
    if holdout:
        bowfire, val["bowfire"] = _holdout(bowfire, 0.3, rng, key=lambda s: s.image.stem.startswith("fire"))
        negatives, val["negatives"] = _holdout(negatives, 0.2, rng)
    train.extend(bowfire * photo_repeat)
    train.extend(negatives)
    return train, {group: samples for group, samples in val.items() if samples}


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    scale = MAX_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(sample: Sample, shape: tuple[int, int]) -> np.ndarray:
    if sample.mask is None:
        return np.zeros(shape, dtype=np.uint8)
    mask = cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)  # video GT masks are JPEGs with compression noise


class FireDataset(Dataset):
    """Returns (image float32 CHW in [0, 1], mask float32 1HW in {0, 1})."""

    def __init__(self, samples: list[Sample], transform=None):
        self.samples = samples
        self.transform = transform
        # Cache decoded images: the whole dataset fits comfortably in memory.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, idx: int):
        if idx not in self._cache:
            sample = self.samples[idx]
            image = read_rgb(sample.image)
            self._cache[idx] = (image, read_mask(sample, image.shape[:2]))
        return self._cache[idx]

    def __getitem__(self, idx: int):
        image, mask = self._load(idx)
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mask = torch.from_numpy(mask[None].astype(np.float32))
        return image, mask
