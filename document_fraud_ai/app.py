"""
Document Fraud Detection API - FastAPI Application.

Endpoints:
    POST /analyze       - Upload a document for fraud analysis
    POST /analyze/batch - Upload multiple documents for batch analysis
    GET  /health        - Health check
    GET  /model/info    - Model and pipeline information
"""

import os
import sys
import uuid
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(
    os.path.join(LOG_DIR, "fraud_api_{time}.log"),
    rotation="10 MB",
    retention="7 days",
    level="INFO",
)

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from fraud_model.pipeline import FraudDetectionPipeline

# Global pipeline instance (cached)
pipeline: Optional[FraudDetectionPipeline] = None

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the fraud detection pipeline on startup."""
    global pipeline
    model_path = os.environ.get("FRAUD_MODEL_PATH", None)
    device = os.environ.get("FRAUD_DEVICE", None)
    pipeline = FraudDetectionPipeline(model_path=model_path, device=device)
    logger.info("Fraud detection pipeline initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Document Fraud Detection API",
    description=(
        "AI-powered document fraud detection system. "
        "Analyzes PDFs and images for tampering using ELA, CNN, "
        "metadata analysis, OCR, and copy-move detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_file(file: UploadFile) -> str:
    """Validate uploaded file and return its extension."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


async def _save_upload(file: UploadFile, ext: str) -> str:
    """Save uploaded file to disk, rejecting files over MAX_FILE_SIZE before full load."""
    file_id = uuid.uuid4().hex[:12]
    filename = f"{file_id}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    # Read in 64 KB chunks — stops early if file exceeds size limit
    CHUNK = 64 * 1024
    total = 0
    chunks = []
    while True:
        chunk = await file.read(CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Max 50 MB.")
        chunks.append(chunk)

    with open(filepath, "wb") as f:
        for chunk in chunks:
            f.write(chunk)

    return filepath


@app.post("/analyze", tags=["Analysis"])
async def analyze_document(file: UploadFile = File(..., description="PDF, JPG, or PNG document")):
    """
    Analyze a single document for fraud.

    Upload a PDF or image file. The system will run multiple analysis
    modules and return a fraud probability with detailed reasons.

    **Response:**
    - `fraud_probability`: Float [0, 1] - likelihood of fraud
    - `is_fraud`: Boolean - true if probability > 0.5
    - `confidence`: Percentage - how confident the system is
    - `reasons`: List of detected anomalies
    """
    ext = _validate_file(file)
    filepath = await _save_upload(file, ext)

    if pipeline is None:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized")

    try:
        start_time = time.time()
        result = pipeline.analyze(filepath)
        result["processing_time_seconds"] = round(time.time() - start_time, 3)
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Analysis failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(files: List[UploadFile] = File(..., description="Multiple documents")):
    """
    Analyze multiple documents in a batch.

    Upload up to 10 files at once. Returns individual results for each file.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results = []
    for file in files:
        try:
            ext = _validate_file(file)
            filepath = await _save_upload(file, ext)

            start_time = time.time()
            result = pipeline.analyze(filepath)
            result["processing_time_seconds"] = round(time.time() - start_time, 3)
            result["filename"] = file.filename
            results.append(result)

            if os.path.exists(filepath):
                os.remove(filepath)
        except HTTPException as e:
            results.append({"filename": file.filename, "error": e.detail})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return JSONResponse(content={"results": results, "total": len(results)})


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline.model_loaded if pipeline else False,
        "device": pipeline.device if pipeline else "unknown",
    }


@app.get("/model/info", tags=["System"])
async def model_info():
    """Get information about the loaded model and pipeline configuration."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized")
    return {
        "model_type": "EfficientNet-B3" if pipeline.model_loaded else "Heuristic (no pretrained model)",
        "device": pipeline.device,
        "modules": [
            "Error Level Analysis (ELA) — adaptive spatial clustering",
            "EfficientNet-B3 Classification",
            "Metadata Consistency Check",
            "OCR + Text Anomaly Detection",
            "Copy-Move Forgery Detection (AKAZE + DBSCAN)",
            "Blur/Sharpness Inconsistency",
        ],
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
