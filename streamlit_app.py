
import os
import base64
import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Smart Parking | CNN Detector",
    page_icon="🅿️",
    layout="wide",
)

st.title("CNN Parking Lot Occupancy Detector")
st.write(
    "Upload an image of a parking lot to detect configured parking slots "
    "and classify each slot as free or occupied using a trained CNN."
)

# Load backend URL from Streamlit secrets or environment variables
try:
    configured_api_url = st.secrets.get("API_URL", "")
except Exception:
    configured_api_url = ""

configured_api_url = configured_api_url or os.getenv("API_URL", "")

# Sidebar settings
with st.sidebar:
    st.header("Backend settings")

    api_url = st.text_input(
        "FastAPI URL",
        value=configured_api_url,
        placeholder="https://your-service.onrender.com",
        help=(
            "Enter your deployed Render URL, or "
            "http://127.0.0.1:8000 for local testing."
        ),
    ).strip().rstrip("/")

    st.divider()

    st.subheader("Model information")
    st.write("Architecture: CNN")
    st.write("Input: 96 × 96 RGB")
    st.write("Parameters: 57,865")
    st.write("Occupied threshold: 0.30")

# Upload parking lot image
uploaded_file = st.file_uploader(
    "Choose a parking-lot image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("Uploaded image")
            st.image(
                image,
                use_container_width=True,
            )

        with right:
            st.subheader("Parking-slot detection")

            if st.button(
                "Detect all parking slots",
                type="primary",
                use_container_width=True,
            ):
                if not api_url:
                    st.error(
                        "Set the FastAPI URL in the sidebar first."
                    )
                else:
                    try:
                        with st.spinner(
                            "Detecting occupancy for each configured slot..."
                        ):
                            uploaded_file.seek(0)

                            response = requests.post(
                                f"{api_url}/detect",
                                files={
                                    "file": (
                                        uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        uploaded_file.type or "image/jpeg",
                                    )
                                },
                                timeout=180,
                            )

                        if response.ok:
                            result = response.json()

                            # Decode annotated image returned by the backend
                            annotated_bytes = base64.b64decode(
                                result["annotated_image"],
                                validate=True,
                            )

                            st.subheader("Parking-slot detection")

                            st.image(
                                annotated_bytes,
                                caption="Green = Free | Red = Occupied",
                                use_container_width=True,
                            )

                            # Display parking occupancy metrics
                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Total slots",
                                result["total_slots_detected"],
                            )

                            col2.metric(
                                "Free slots",
                                result["free_slots"],
                            )

                            col3.metric(
                                "Occupied slots",
                                result["occupied_slots"],
                            )

                            # Display individual slot predictions
                            with st.expander(
                                "Individual slot predictions"
                            ):
                                st.dataframe(
                                    result["slots"],
                                    use_container_width=True,
                                )

                        else:
                            try:
                                detail = response.json().get(
                                    "detail",
                                    response.text,
                                )
                            except ValueError:
                                detail = response.text

                            st.error(
                                f"Backend error ({response.status_code}): "
                                f"{detail}"
                            )

                    except requests.exceptions.Timeout:
                        st.error(
                            "The backend took too long to respond. "
                            "If this is a new Render deployment, wait "
                            "a little and try again."
                        )

                    except requests.exceptions.RequestException as exc:
                        st.error(
                            "Could not reach the backend. Check the URL "
                            "and confirm the FastAPI service is running. "
                            f"Details: {exc}"
                        )

                    except (ValueError, KeyError, TypeError) as exc:
                        st.error(
                            "The backend returned an invalid response. "
                            "Check that /detect returns a valid JSON "
                            "response with all required fields. "
                            f"Details: {exc}"
                        )

    except Exception:
        st.error(
            "This image could not be opened. Please try another image."
        )

st.divider()

st.caption(
    "Academic demonstration using a trained CNN and the CNRPark-EXT "
    "parking dataset. Upload a parking-lot image to detect configured "
    "parking slots. This version does not process a live CCTV stream."
)
