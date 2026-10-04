"""Pre/post-processing and prediction. Depends only on numpy and OpenCV, so it can be
tested without the model runtime; ``model`` is any callable mapping an (N, 3, H, W)
float32 batch to (N, 1, H, W) fire probabilities. If it has an ``input_size`` attribute,
images are resized to that size, otherwise to ``IMG_SIZE``."""

import cv2
import numpy as np

from .config import DEFAULT_THRESHOLD, IMG_SIZE


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode encoded image bytes (jpg/png/webp...) into a BGR array, or None."""
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess(image_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """BGR (or grayscale) image -> model input of shape (1, 3, size, size).

    Mirrors training: RGB, resize to size x size, scale to [0, 1].
    ImageNet normalisation is part of the exported model.
    """
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]


def predict_probabilities(model, image_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel fire probability, resized back to the input image size."""
    size = getattr(model, "input_size", IMG_SIZE)
    probs = np.asarray(model(preprocess(image_bgr, size)))[0, 0]
    height, width = image_bgr.shape[:2]
    return cv2.resize(probs.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def threshold_mask(probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Binary uint8 mask (0/1) of pixels whose probability exceeds ``threshold``."""
    return (probs > threshold).astype(np.uint8)


def predict_mask(model, image_bgr: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    return threshold_mask(predict_probabilities(model, image_bgr), threshold)


def make_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 220, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend ``color`` over the masked pixels of an RGB image."""
    overlay = image_rgb.copy()
    fire = mask.astype(bool)
    tint = np.array(color, dtype=np.float32)
    overlay[fire] = ((1 - alpha) * image_rgb[fire] + alpha * tint).astype(np.uint8)
    return overlay


def fire_coverage(mask: np.ndarray) -> float:
    """Percentage of pixels labelled as fire."""
    return float(mask.mean() * 100) if mask.size else 0.0


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image as PNG")
    return buf.tobytes()
