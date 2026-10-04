"""Train the fire segmentation model and export it to ONNX.

    python -m training.train                     # train with held-out videos/photos for validation
    python -m training.train --no-holdout --export models/fire_segmentation.onnx   # final model

Requires the packages in requirements-train.txt (PyTorch with CUDA recommended) and the
BoWFire dataset (python scripts/download_bowfire.py).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from training.data import FireDataset, split
from training.model import FireSegmenter, ProbabilityHead

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VAL_VIDEOS = [3, 8, 11]


def train_transform(size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(size, size), scale=(0.35, 1.0), ratio=(0.6, 1.7)),
            A.HorizontalFlip(),
            A.Affine(rotate=(-10, 10), p=0.3),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.7),
            # Small hue shifts only: colour is the main cue for fire.
            A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=25, val_shift_limit=15, p=0.5),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
            A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
            A.ImageCompression(quality_range=(40, 95), p=0.3),
            A.ToGray(p=0.05),
        ]
    )


def eval_transform(size: int) -> A.Compose:
    return A.Compose([A.Resize(size, size)])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cv2.setRNGSeed(seed)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5, alarm_coverage: float = 0.005) -> dict:
    """Pixel metrics over all images and false-alarm rate over fire-free images."""
    model.eval()
    tp = fp = fn = tn = 0
    alarms = negatives = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            preds = torch.sigmoid(model(images)) > threshold
        truth = masks > 0.5
        tp += (preds & truth).sum().item()
        fp += (preds & ~truth).sum().item()
        fn += (~preds & truth).sum().item()
        tn += (~preds & ~truth).sum().item()
        empty = truth.flatten(1).sum(1) == 0
        coverage = preds.flatten(1).float().mean(1)
        negatives += empty.sum().item()
        alarms += (empty & (coverage > alarm_coverage)).sum().item()
    fire_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    return {
        "fire_iou": fire_iou,
        "mean_iou": (fire_iou + bg_iou) / 2,
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "false_alarm_rate": alarms / max(negatives, 1),
    }


def export_onnx(model: FireSegmenter, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ProbabilityHead(model).cpu().eval()
    torch.onnx.export(
        head,
        torch.rand(1, 3, size, size),
        str(path),
        input_names=["image"],
        output_names=["probability"],
        dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--encoder", default="efficientnet-b0")
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--val-videos",
        default=",".join(map(str, DEFAULT_VAL_VIDEOS)),
        help="comma-separated video ids held out for validation",
    )
    parser.add_argument("--no-holdout", action="store_true", help="train on all data (final model)")
    parser.add_argument("--photo-repeat", type=int, default=5, help="oversampling factor for BoWFire photos")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "latest")
    parser.add_argument("--export", type=Path, help="export the final model to this ONNX path")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_videos = [int(v) for v in args.val_videos.split(",")]
    train_samples, val_groups = split(
        val_videos, holdout=not args.no_holdout, photo_repeat=args.photo_repeat, seed=args.seed
    )
    sizes = {group: len(samples) for group, samples in val_groups.items()}
    print(f"device={device} train={len(train_samples)} val={sizes} val_videos={val_videos}")

    loader_kwargs = {"num_workers": args.workers, "persistent_workers": args.workers > 0, "pin_memory": True}
    train_loader = DataLoader(
        FireDataset(train_samples, train_transform(args.size)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loaders = {
        group: DataLoader(
            FireDataset(samples, eval_transform(args.size)), batch_size=args.batch_size, **loader_kwargs
        )
        for group, samples in val_groups.items()
    }

    model = FireSegmenter(args.encoder).to(device)
    dice = smp.losses.DiceLoss("binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps, warmup = args.epochs * len(train_loader), len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))),
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = bce(logits, masks) + dice(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        record = {"epoch": epoch, "loss": total / len(train_loader), "seconds": round(time.time() - start, 1)}
        if val_loaders:
            # Reported only: per-epoch scores on the small validation sets are noisy, so the
            # final weights (end of the cosine schedule) are used rather than a "best" epoch.
            for group, loader in val_loaders.items():
                metrics = evaluate(model, loader, device)
                record.update({f"{group}/{k}": v for k, v in metrics.items()})
        history.append(record)
        print(
            json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()}), flush=True
        )

    torch.save(model.state_dict(), args.out / "last.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    if args.export:
        export_onnx(model, args.export, args.size)


if __name__ == "__main__":
    main()
