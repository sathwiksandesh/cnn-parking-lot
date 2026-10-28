"""
streamlit_app.py
~~~~~~~~~~~~~~~~
CNN Parking Lot Occupancy Detector – Streamlit front-end.

Workflow:
  1. User uploads a parking-lot image.
  2. App sends the image to the FastAPI /auto-detect endpoint.
  3. The API runs YOLOv8 + CNN to locate and classify every parking space.
  4. Results are displayed: annotated image, summary metrics, per-space table.
"""

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

def render_image(image_input, caption: str = ""):
    """
    Renders an image directly via inline base64 data URI to prevent Streamlit /
    Starlette ASGI media server crashes that produce broken image icons.
    """
    if isinstance(image_input, str):
        b64_str = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        b64_str = base64.b64encode(image_input).decode("ascii")
    elif hasattr(image_input, "getvalue"):
        b64_str = base64.b64encode(image_input.getvalue()).decode("ascii")
    elif isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="JPEG", quality=95)
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    else:
        b64_str = ""

    caption_html = (
        f'<p style="text-align: center; color: #a0aec0; font-size: 0.88rem; margin-top: 6px; margin-bottom: 0;">{caption}</p>'
        if caption else ""
    )
    st.markdown(
        f'<div style="margin: 12px 0; text-align: center;">'
        f'<img src="data:image/jpeg;base64,{b64_str}" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);" />'
        f'{caption_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN Parking Occupancy Detector",
    page_icon="🅿️",
    layout="wide",
)

# ── Sidebar – backend URL ─────────────────────────────────────────────────────
try:
    _secret_url = st.secrets.get("API_URL", "")
except Exception:
    _secret_url = ""

with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input(
        "FastAPI URL",
        value=_secret_url or os.getenv("API_URL", "http://127.0.0.1:8000"),
        help="Base URL of the running FastAPI backend.",
    ).strip().rstrip("/")

    st.markdown("---")
    st.subheader("🎯 Detection Options")
    threshold = st.slider(
        "Occupancy Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help="Probability threshold to classify a parking space as Occupied.",
    )

    slot_mode = st.radio(
        "Parking Space Count",
        options=["Auto-detect", "Specify exact count"],
        help="Let AI automatically determine spaces or specify the count.",
    )

    if slot_mode == "Specify exact count":
        target_slots = st.number_input(
            "Exact Number of Parking Bays",
            min_value=1,
            max_value=40,
            value=10,
            step=1,
            help="Specify the exact number of parking bays across the lot.",
        )
    else:
        target_slots = 0

    st.markdown("---")
    st.caption(
        "**How it works**\n\n"
        "1. AI identifies the active parking row and space boundaries.\n"
        "2. Custom CNN classifies each parking bay as Free or Occupied.\n"
        "3. Results are annotated with non-overlapping badges."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🅿️ CNN Parking Lot Occupancy Detector")
st.write(
    "Upload a parking-lot image. The system will **automatically** detect "
    "parking spaces, classify each one as **free** or **occupied**, and "
    "display the annotated result — no manual drawing required."
)

# ── Image upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a parking-lot image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Overhead or elevated camera views work best.",
)

if not uploaded:
    st.info("👆 Upload a parking-lot image to get started.")
    st.stop()

# Show original image
try:
    original = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("Could not read this image.")
    st.stop()

img_w, img_h = original.size
render_image(uploaded, caption=f"Uploaded image — {img_w} × {img_h} px")

# ── Detect button ─────────────────────────────────────────────────────────────
if not st.button(
    "🔍 Detect & Classify Parking Spaces",
    type="primary",
    use_container_width=True,
):
    st.stop()

# ── Call API ──────────────────────────────────────────────────────────────────
with st.spinner("Running parking-space detection and CNN classification…"):
    uploaded.seek(0)
    try:
        resp = requests.post(
            f"{api_url}/auto-detect",
            files={"file": (uploaded.name, uploaded.read(), "image/jpeg")},
            params={
                "threshold": threshold,
                "target_slots": target_slots,
            },
            timeout=180,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.ConnectionError:
        st.error(
            f"Could not connect to the API at **{api_url}**. "
            "Make sure the FastAPI server is running:\n\n"
            "```\nuvicorn api.main:app --reload\n```"
        )
        st.stop()
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"API returned an error ({exc.response.status_code}): {detail or exc}")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
total    = result.get("total_spaces", 0)
free_n   = result.get("free_spaces", 0)
occ_n    = result.get("occupied_spaces", 0)
spaces   = result.get("spaces", [])

if total == 0:
    st.warning(
        "No parking spaces were detected in this image. "
        "Try a different image with clearly visible vehicles or painted lane markings."
    )
    st.stop()

# Annotated image
ann_bytes = base64.b64decode(result["annotated_image"])
ann_img   = Image.open(io.BytesIO(ann_bytes))

st.subheader("📸 Detection Result")
render_image(
    result["annotated_image"],
    caption="🟢 Green = Free   🔴 Red = Occupied",
)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Spaces", total)
col2.metric("🟢 Free", free_n)
col3.metric("🔴 Occupied", occ_n)
rate = (occ_n / total * 100) if total > 0 else 0
col4.metric("Occupancy Rate", f"{rate:.1f}%")

# Download
out_buf = io.BytesIO(ann_bytes)
st.download_button(
    "⬇️ Download annotated image",
    data=out_buf.getvalue(),
    file_name="parking_occupancy_result.jpg",
    mime="image/jpeg",
)

# Per-space table 
if spaces:
    st.subheader("📋 Per-space details")
    df = pd.DataFrame([
        {
            "ID"                     : s["id"],
            "Status"                 : s["status"].capitalize(),
            "Occupied probability (%)" : round(s["occupied_probability"] * 100, 1),
            "Detected by"            : s["detection_source"],
        }
        for s in spaces
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Detection uses YOLOv8-nano (vehicle localisation) + a custom CNN "
    "(occupancy classification). Empty-space inference relies on row-gap "
    "analysis of detected vehicle positions and may miss spaces at row ends "
    "or in very sparse lots."
)
