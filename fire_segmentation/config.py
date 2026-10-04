"""Central configuration. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Fallback input size; the ONNX model's own input shape takes precedence.
IMG_SIZE = 416
DEFAULT_THRESHOLD = 0.5

MODEL_PATH = Path(os.getenv("FIRESEG_MODEL_PATH", ROOT_DIR / "models" / "fire_segmentation.onnx"))

SAMPLES_DIR = ROOT_DIR / "dataset" / "testing"
