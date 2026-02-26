"""
Streamlit Frontend for Document Fraud Detection.

A simple web UI for uploading documents and viewing fraud analysis results.
Connects to the FastAPI backend.
"""

import streamlit as st
import requests
import json
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("Document Fraud Detection System")
st.markdown(
    "Upload a document (PDF, JPG, PNG) to analyze it for potential fraud or tampering."
)

# Sidebar with system info
with st.sidebar:
    st.header("System Info")
    try:
        info = requests.get(f"{API_URL}/model/info", timeout=5).json()
        st.json(info)
    except Exception:
        st.warning("API not reachable. Start the FastAPI server first.")
        st.code("python app.py", language="bash")

st.divider()

# File upload
uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "jpg", "jpeg", "png", "tiff", "bmp", "webp"],
    help="Supported formats: PDF, JPG, PNG, TIFF, BMP, WebP (max 50 MB)",
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Document")
        if uploaded_file.type and uploaded_file.type.startswith("image"):
            st.image(uploaded_file, use_container_width=True)
        else:
            st.info(f"File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

    with col2:
        st.subheader("Analysis Result")

        if st.button("Analyze Document", type="primary", use_container_width=True):
            with st.spinner("Analyzing document for fraud indicators..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(
                        f"{API_URL}/analyze", files=files, timeout=120
                    )

                    if response.status_code == 200:
                        result = response.json()

                        # Fraud probability gauge
                        fraud_prob = result["fraud_probability"]
                        is_fraud = result["is_fraud"]
                        confidence = result["confidence"]

                        if is_fraud:
                            st.error(f"FRAUD DETECTED - Probability: {fraud_prob:.2%}")
                        else:
                            st.success(f"DOCUMENT APPEARS GENUINE - Probability: {fraud_prob:.2%}")

                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.metric(
                            "Processing Time",
                            f"{result.get('processing_time_seconds', 0):.2f}s",
                        )

                        # Reasons
                        st.subheader("Detected Indicators")
                        for reason in result.get("reasons", []):
                            if is_fraud:
                                st.warning(reason)
                            else:
                                st.info(reason)

                        # Module scores
                        st.subheader("Module Scores")
                        details = result.get("details", {})
                        module_scores = details.get("module_scores", {})
                        for module, score in module_scores.items():
                            label = module.replace("_", " ").title()
                            st.progress(min(score, 1.0), text=f"{label}: {score:.4f}")

                        # Raw JSON
                        with st.expander("View Raw JSON Response"):
                            st.json(result)

                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")

                except requests.ConnectionError:
                    st.error(
                        "Cannot connect to API. Make sure the FastAPI server is running:\n"
                        "`python app.py`"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

st.divider()
st.markdown(
    """
    **How it works:**
    1. **ELA Analysis** - Detects compression inconsistencies from image tampering
    2. **CNN Model** - Deep learning classification of tampered vs genuine
    3. **Metadata Check** - Analyzes file metadata for editing tool traces
    4. **OCR Analysis** - Extracts text and checks for content anomalies
    5. **Copy-Move Detection** - Finds duplicated regions within the document
    """
)
