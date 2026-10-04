"""Streamlit front-end for the fire segmentation model."""

import os

import streamlit as st

from fire_segmentation import config
from fire_segmentation.inference import (
    decode_image,
    encode_png,
    fire_coverage,
    make_overlay,
    predict_probabilities,
    threshold_mask,
    to_rgb,
)
from fire_segmentation.model import load_model

SAMPLE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(page_title="Fire Segmentation", page_icon="🔥", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


@st.cache_data
def list_samples() -> dict[str, str]:
    samples = {}
    for path in sorted(config.SAMPLES_DIR.glob("*/*")):
        if path.suffix.lower() in SAMPLE_EXTENSIONS:
            samples[f"{path.parent.name}/{path.name}"] = str(path)
    return samples


with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Mask threshold",
        min_value=0.05,
        max_value=0.95,
        value=config.DEFAULT_THRESHOLD,
        step=0.05,
        help="Pixels with predicted fire probability above this value are marked as fire.",
    )
    st.divider()
    st.subheader("About")
    st.markdown(
        "U-Net with an EfficientNet-B0 encoder, trained on fire videos and photos. "
        "On held-out photos it reaches a fire IoU of **0.81**.  \n"
        "[Source code](https://github.com/farisznafis/fire-segmentation)"
    )

st.title("🔥 Fire Segmentation")
st.write("Upload an image or pick a sample to highlight the pixels that contain fire.")

source = st.radio("Image source", ["Upload", "Sample"], horizontal=True, label_visibility="collapsed")
data, name = None, None
if source == "Upload":
    uploaded = st.file_uploader("Choose an image", type=sorted(ext.lstrip(".") for ext in SAMPLE_EXTENSIONS))
    if uploaded is not None:
        data, name = uploaded.getvalue(), uploaded.name
else:
    samples = list_samples()
    if samples:
        choice = st.selectbox("Sample image", list(samples))
        with open(samples[choice], "rb") as f:
            data, name = f.read(), choice
    else:
        st.info("No sample images found.")

if data is None:
    st.stop()

image_bgr = decode_image(data)
if image_bgr is None:
    st.error("Could not read this file as an image.")
    st.stop()

try:
    model = get_model()
except Exception as exc:  # missing or corrupt model file
    st.error(f"Failed to load the model: {exc}")
    st.stop()

with st.spinner("Segmenting..."):
    probs = predict_probabilities(model, image_bgr)
mask = threshold_mask(probs, threshold)
image_rgb = to_rgb(image_bgr)
coverage = fire_coverage(mask)

col1, col2, col3 = st.columns(3)
col1.image(image_rgb, caption="Original", width="stretch")
col2.image(mask * 255, caption="Predicted mask", width="stretch")
col3.image(make_overlay(image_rgb, mask), caption="Overlay", width="stretch")

m1, m2 = st.columns(2)
m1.metric("Fire coverage", f"{coverage:.2f}%")
m2.metric("Max fire probability", f"{probs.max():.2f}")

if mask.any():
    st.warning(f"Fire detected: {coverage:.2f}% of the image.")
else:
    st.success("No fire detected at the current threshold.")
st.caption(
    "Very small or distant flames can be missed, and vivid sunset clouds are occasionally "
    "highlighted. Raise the threshold to reduce false alarms, lower it to catch more fire."
)

stem = os.path.splitext(os.path.basename(name))[0]
st.download_button(
    "Download mask (PNG)",
    data=encode_png(mask * 255),
    file_name=f"{stem}_mask.png",
    mime="image/png",
)
