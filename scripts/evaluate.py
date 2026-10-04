"""Evaluate an exported ONNX model through the same code path the app uses.

    python scripts/evaluate.py --model runs/<run>/model.onnx

Reports, for a model trained with the default hold-out split (training/data.py):
- pixel metrics (fire IoU, mean IoU, precision, recall) and false-alarm rate on each held-out
  group: video frames, BoWFire photos and fire-free photos;
- image-level results on dataset/testing (never used for training or model selection):
  share of fire photos where fire is found and of fire-free photos with a false alarm.

Note: the final model is trained on all data, so only the dataset/testing numbers are
unbiased for it; quote the held-out numbers from the hold-out run.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fire_segmentation import config  # noqa: E402
from fire_segmentation.inference import predict_probabilities  # noqa: E402
from fire_segmentation.model import load_model  # noqa: E402
from training.data import IMAGE_EXTENSIONS, Sample, read_mask, split  # noqa: E402

ALARM_COVERAGE = 0.5  # percent of pixels


def pixel_metrics(model, samples: list[Sample], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    alarms = negatives = 0
    for sample in samples:
        image = cv2.imread(str(sample.image))
        pred = predict_probabilities(model, image) > threshold
        truth = read_mask(sample, image.shape[:2]).astype(bool)
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())
        if not truth.any():
            negatives += 1
            alarms += int(pred.mean() * 100 > ALARM_COVERAGE)
    metrics = {"false_alarm_rate": alarms / negatives if negatives else None}
    if tp + fn:  # segmentation metrics only make sense when the group contains fire
        fire_iou = tp / (tp + fp + fn)
        metrics.update(
            fire_iou=fire_iou,
            mean_iou=(fire_iou + tn / max(tn + fp + fn, 1)) / 2,
            precision=tp / max(tp + fp, 1),
            recall=tp / (tp + fn),
        )
    return metrics


def unique_images(folder: Path) -> list[Path]:
    seen, paths = set(), []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        digest = hashlib.md5(cv2.resize(image, (64, 64)).tobytes()).hexdigest()
        if digest not in seen:  # the folder contains the same photo in two formats
            seen.add(digest)
            paths.append(path)
    return paths


def photo_metrics(model, threshold: float) -> dict:
    result = {}
    for name in ("fire", "non-fire"):
        coverages = {}
        for path in unique_images(config.SAMPLES_DIR / name):
            probs = predict_probabilities(model, cv2.imread(str(path)))
            coverages[path.name] = float((probs > threshold).mean() * 100)
        flagged = sum(c > ALARM_COVERAGE for c in coverages.values())
        result[name] = {"images": len(coverages), "flagged": flagged, "coverage": coverages}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--val-videos", default="3,8,11", help="videos held out in the hold-out run")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    parser.add_argument("--json", type=Path, help="also write the full report to this file")
    args = parser.parse_args()

    model = load_model(args.model)
    report = {"threshold": args.threshold, "held_out": {}}
    _, val_groups = split([int(v) for v in args.val_videos.split(",")])
    for group, samples in val_groups.items():
        metrics = pixel_metrics(model, samples, args.threshold)
        report["held_out"][group] = metrics
        print(
            f"Held-out {group} ({len(samples)} images):",
            {k: round(v, 4) for k, v in metrics.items() if v is not None},
        )
    report["photos"] = photo_metrics(model, args.threshold)
    for name, res in report["photos"].items():
        print(
            f"Photos {name}: {res['flagged']}/{res['images']} flagged as fire (> {ALARM_COVERAGE}% of pixels)"
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
