"""Loading the exported ONNX model."""

from pathlib import Path

import numpy as np

from . import config


class OnnxSegmenter:
    """Callable wrapper: (N, 3, H, W) float32 in [0, 1] -> (N, 1, H, W) fire probabilities."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        # The model is exported with a fixed spatial size: (batch, 3, size, size).
        self.input_size = int(model_input.shape[-1])

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch.astype(np.float32)})[0]


def load_model(path: Path | None = None) -> OnnxSegmenter:
    return OnnxSegmenter(Path(path or config.MODEL_PATH))
