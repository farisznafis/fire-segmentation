import cv2
import numpy as np
import pytest

from fire_segmentation.config import IMG_SIZE
from fire_segmentation.inference import (
    decode_image,
    encode_png,
    fire_coverage,
    make_overlay,
    predict_mask,
    predict_probabilities,
    preprocess,
    threshold_mask,
)


class BrightnessModel:
    """Stand-in for the U-Net: 'fire' probability equals pixel brightness."""

    def __init__(self):
        self.last_input = None

    def __call__(self, x):
        self.last_input = x
        return x.mean(axis=1, keepdims=True)


@pytest.fixture
def image_bgr():
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    img[:, 100:] = 255  # right half bright
    return img


def test_preprocess_shape_and_range(image_bgr):
    x = preprocess(image_bgr)
    assert x.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert x.dtype == np.float32
    assert x.min() == 0.0 and x.max() == 1.0


def test_preprocess_converts_bgr_to_rgb():
    blue = np.zeros((10, 10, 3), dtype=np.uint8)
    blue[..., 0] = 255  # BGR blue
    x = preprocess(blue)
    assert x[0, 2].min() == 1.0 and x[0, 0].max() == 0.0


def test_preprocess_accepts_grayscale(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    np.testing.assert_array_equal(preprocess(gray), preprocess(image_bgr))


def test_probabilities_match_input_size(image_bgr):
    model = BrightnessModel()
    probs = predict_probabilities(model, image_bgr)
    assert probs.shape == image_bgr.shape[:2]
    assert model.last_input.shape == (1, 3, IMG_SIZE, IMG_SIZE)


def test_predict_mask_marks_bright_half(image_bgr):
    mask = predict_mask(BrightnessModel(), image_bgr, threshold=0.5)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 1}
    assert mask[:, :90].sum() == 0
    assert mask[:, 110:].all()


def test_threshold_is_strict():
    probs = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)
    np.testing.assert_array_equal(threshold_mask(probs, 0.5), [[0, 0, 1]])


def test_overlay_only_changes_masked_pixels():
    image = np.full((4, 4, 3), 100, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    out = make_overlay(image, mask)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert not np.array_equal(out[0, 0], image[0, 0])
    np.testing.assert_array_equal(out[1:], image[1:])


def test_fire_coverage():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:2] = 1
    assert fire_coverage(mask) == pytest.approx(20.0)
    assert fire_coverage(np.zeros((0, 0))) == 0.0


def test_png_round_trip(image_bgr):
    decoded = decode_image(encode_png(image_bgr))
    np.testing.assert_array_equal(decoded, image_bgr)


def test_decode_invalid_bytes():
    assert decode_image(b"not an image") is None


def test_uses_model_input_size(image_bgr):
    model = BrightnessModel()
    model.input_size = 64
    probs = predict_probabilities(model, image_bgr)
    assert model.last_input.shape == (1, 3, 64, 64)
    assert probs.shape == image_bgr.shape[:2]
