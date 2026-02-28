# Document Fraud Detection System

An AI-powered system that analyzes PDF documents and images to detect tampering, forgery, and manipulation. It combines 6 independent detection modules — deep learning, signal processing, computer vision, and rule-based analysis — into a single weighted fraud probability score.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Performance & GPU Optimization](#performance--gpu-optimization)
- [Project Structure](#project-structure)
- [File-by-File Reference](#file-by-file-reference)
- [Detection Algorithms](#detection-algorithms)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Web UI](#web-ui)
- [Training a Custom Model](#training-a-custom-model)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## Overview

```
User uploads document (PDF / JPG / PNG / TIFF / BMP / WebP)
                        |
              +---------v----------+
              |   FastAPI Server   |
              |     (app.py)       |
              +---------+----------+
                        |
              +---------v----------+
              | Detection Pipeline |
              |   (pipeline.py)    |
              +--+--+--+--+--+--+-+
                 |  |  |  |  |  |
    +------------+  |  |  |  |  +----------------+
    |               |  |  |  |                   |
    v               v  v  v  v                   v
  ELA           CNN  Meta OCR  Copy-Move      Blur
 (25%)         (35%) (15%)(10%)  (10%)        (5%)
    |               |  |  |  |                   |
    +---------------+--+--+--+-------------------+
                        |
              +---------v----------+
              |   Weighted Score   |
              |  Fusion + Reasons  |
              +---------+----------+
                        |
              +---------v----------+
              |    JSON Response   |
              +--------------------+
```

The system works **without a trained model** (heuristic mode) and is significantly more accurate **with** a fine-tuned EfficientNet-B3 model.

---

## Features

- Detects JPEG compression tampering via Error Level Analysis (ELA) — multi-quality ensemble (85/90/95)
- Deep learning classification using EfficientNet-B3 (transfer learning)
- Copy-move forgery detection using AKAZE + DBSCAN clustering
- PDF and image metadata analysis (EXIF, dates, editing tools)
- OCR text extraction and anomaly detection
- Local sharpness inconsistency detection (Laplacian + MAD)
- JPEG ghost detection for cross-quality forgery
- Batch analysis of up to 10 documents
- REST API with Swagger documentation
- Interactive web UI (Streamlit)
- Docker support with health checks
- GPU acceleration (CUDA auto-detected) with FP16 mixed precision
- Confidence scoring based on module agreement

---

## Performance & GPU Optimization

| Optimization | Where | GPU VRAM | Speed | Accuracy |
|---|---|---|---|---|
| FP16 mixed precision inference (`torch.autocast`) | `pipeline.py` | **−50%** | +30% | neutral |
| `torch.compile(mode='reduce-overhead')` | `pipeline.py` | neutral | **+20-30%** | neutral |
| Multi-quality ELA ensemble (85/90/95) | `pipeline.py` | neutral (CPU) | −10% CPU | **+robustness** |
| FP16 mixed precision training (`GradScaler`) | `train_model.py` | **−50%** | +30% | neutral |
| Label smoothing (smoothing=0.1) | `train_model.py` | neutral | neutral | **+1-2% AUC** |
| Warmup + cosine LR (Phase 2) | `train_model.py` | neutral | neutral | **+convergence** |
| Gradient accumulation (`--accum_steps`) | `train_model.py` | **−50%** | neutral | +larger effective batch |
| `cudnn.benchmark = True` | `train_model.py` | neutral | +5-10% | neutral |

**Training example on a low-VRAM GPU (4-6 GB):**
```bash
python train_model.py \
  --data_dir ./dataset \
  --batch_size 4 \
  --accum_steps 8 \
  --epochs 50 \
  --freeze_epochs 5
# Effective batch size = 32, VRAM usage ≈ 3-4 GB with FP16
```

---

## Project Structure

```
document_fraud_ai/
├── app.py                        # FastAPI REST API server
├── streamlit_app.py              # Streamlit web UI frontend
├── train_model.py                # EfficientNet-B3 training script
├── prepare_custom_dataset.py     # Dataset augmentation tool
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container config
├── README.md                     # This file
├── fraud_model/
│   ├── __init__.py
│   ├── cnn_model.py              # CNN model architectures
│   └── pipeline.py               # Main orchestration pipeline
├── utils/
│   ├── __init__.py
│   ├── ela_analysis.py           # ELA, copy-move, blur, JPEG ghost
│   ├── metadata_analyzer.py      # PDF/image metadata analysis
│   └── ocr_extractor.py          # Text extraction and anomaly detection
├── uploads/                      # Temporary upload directory (auto-created)
└── logs/                         # Application logs (auto-created)
```

---

## File-by-File Reference

### `app.py` — FastAPI Backend Server

The main entry point. Starts the REST API server on port 8000.

**What it does:**
- Accepts file uploads via `POST /analyze` and `POST /analyze/batch`
- Validates file type (extension check) and file size (max 50 MB)
- Saves files temporarily, runs the detection pipeline, then deletes them immediately
- Returns a structured JSON response with fraud probability, confidence, reasons, and module scores
- Auto-initializes the `FraudDetectionPipeline` on startup via FastAPI lifespan

**Key settings:**
- Max file size: 50 MB (enforced at chunk-read level)
- Allowed formats: `.pdf`, `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`, `.webp`
- CORS: open to all origins (configure for production)
- Logs stored in `logs/` with 10 MB rotation and 7-day retention

**Environment variables read:**
- `FRAUD_MODEL_PATH` — path to trained `.pth` model file (optional)
- `FRAUD_DEVICE` — `cuda` or `cpu` (auto-detected if not set)

---

### `streamlit_app.py` — Web UI Frontend

A browser-based drag-and-drop interface built with Streamlit.

**What it does:**
- Sidebar shows live system info fetched from `/model/info` endpoint
- File uploader accepts all supported formats (max 50 MB)
- Shows inline image preview for image files
- On "Analyze Document" click: sends file to FastAPI backend, displays results
- Shows fraud verdict (red = fraud, green = genuine), confidence, and processing time
- Displays per-module score as progress bars
- Lists all detected anomaly reasons with warning/info styling
- Raw JSON response available in an expandable section

**Environment variable:**
- `API_URL` — backend URL (default: `http://localhost:8000`)

---

### `fraud_model/pipeline.py` — Detection Pipeline Orchestrator

The core module that runs all 6 detection modules and combines their scores into one fraud probability.

**Performance optimizations:**
- **Mixed precision inference**: CNN forward pass runs in FP16 via `torch.autocast` (~50% VRAM reduction on CUDA)
- **torch.compile**: Model is JIT-compiled with `mode='reduce-overhead'` on PyTorch 2.x for ~20-30% faster inference (graceful no-op on older versions)
- **Multi-quality ELA ensemble**: ELA scores computed at quality levels 85, 90, and 95 then averaged — catches tampering visible only at certain compression levels. CNN input uses quality=90 (unchanged).

**Module weights:**

| Module | Weight | Reason |
|--------|--------|--------|
| CNN (EfficientNet-B3) | 35% | Most accurate when model is trained |
| ELA Statistical | 25% | Ensemble ELA signal (3 quality levels averaged) |
| Metadata Analysis | 15% | Reliable but occasional false positives |
| Text/OCR Anomaly | 10% | Narrow but specific signal |
| Copy-Move Detection | 10% | Very specific to copy-paste forgery |
| Blur/Sharpness | 5% | Supplementary spatial signal |

**Score combination logic:**
- If a module fails or is unavailable, its weight is redistributed proportionally to remaining active modules
- Final score is always in range `[0.0, 1.0]`
- `is_fraud = True` when `fraud_probability > 0.5`

**Confidence scoring** uses three factors:
- **Agreement** (40% weight): low standard deviation among module scores = high agreement
- **Decisiveness** (40% weight): distance of final score from 0.5 (the uncertain midpoint)
- **Coverage** (20% weight): fraction of total weight from active modules

---

### `fraud_model/cnn_model.py` — CNN Model Architectures

Defines three model architectures and a `load_model()` function.

**FraudEfficientNetB3** (default, recommended):
```
Input:    ELA image (batch, 3, 300, 300)
Backbone: EfficientNet-B3 pretrained on ImageNet → 1536 features
Head:
  Dropout(0.4) → Linear(1536 → 512) → SiLU
  Dropout(0.3) → Linear(512  → 256) → SiLU
  Dropout(0.2) → Linear(256  →   1) → Sigmoid
Output:   fraud probability [0.0, 1.0]
```

**FraudEfficientNet** (B0, lighter and faster):
```
Input:    ELA image (batch, 3, 224, 224)
Backbone: EfficientNet-B0 pretrained on ImageNet → 1280 features
Head:     Dropout(0.3) → Linear(1280 → 1) → Sigmoid
```

**FraudCNN** (custom, no pretrained weights needed):
```
Input: ELA image (batch, 3, 128, 128)
Block 1: Conv2d(32)  → BN → ReLU → Conv2d(32)  → BN → ReLU → MaxPool → Dropout2d(0.25)
Block 2: Conv2d(64)  → BN → ReLU → Conv2d(64)  → BN → ReLU → MaxPool → Dropout2d(0.25)
Block 3: Conv2d(128) → BN → ReLU → Conv2d(128) → BN → ReLU → MaxPool → Dropout2d(0.25)
Head:    Flatten → Linear(32768→256) → ReLU (non-inplace) → Dropout(0.5) → Linear(256→1) → Sigmoid
```
Note: classifier ReLU is non-inplace for `torch.compile` compatibility.

**ImageNet normalization constants** (required for EfficientNet):
```
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

---

### `utils/ela_analysis.py` — Image Analysis Algorithms

Contains five independent functions:

**`compute_ela(image, quality=90, scale=15)`**
Re-saves image at JPEG quality 90, computes pixel difference vs original, amplifies by 15x.
Returns an ELA image where tampered regions appear brighter.

**`compute_ela_statistics(ela_image)`**
Computes ELA fraud score using:
- 99th-percentile threshold (avoids false positives from Gaussian noise)
- `scipy.ndimage.label` connected component analysis
- Only counts regions larger than 0.5% of image area (filters tiny noise spots)
- Returns `ela_score [0,1]` and `has_anomaly` flag

**`detect_copy_move(image, min_match_count=10)`**
Detects copy-pasted regions using AKAZE + FLANN + DBSCAN.
Returns `copy_move_score`, `cluster_count`, `matched_regions`.

**`compute_blur_sharpness(image)`**
Splits image into 8x8 grid, computes Laplacian variance per cell, flags outliers using MAD.
Returns `blur_score [0,1]` and `inconsistent_cell_count`.

**`compute_jpeg_ghost(image, qualities=[60..95])`**
Re-saves at 8 quality levels, builds per-pixel best-match quality map, computes spatial entropy.
Returns `ghost_score [0,1]`.

---

### `utils/metadata_analyzer.py` — Metadata Analysis

**For image files (EXIF analysis):**

| Signal | Risk Added |
|--------|-----------|
| Edited with Photoshop, GIMP, or Paint.NET | +0.30 |
| DateTimeOriginal differs from DateTime | +0.15 |
| Digitized date differs from original date | +0.10 |
| Partial EXIF (selective metadata stripping) | +0.15 |

**For PDF files:**

| Signal | Risk Added |
|--------|-----------|
| High-risk tool: pdftk, pdfedit, pdf-xchange, nitro, sejda | +0.35 |
| Medium-risk library: iText, ReportLab, fpdf, tcpdf | +0.15 |
| Modified >1 year after creation date | +0.25 |
| Modified >30 days after creation date | +0.10 |
| File size >5 MB per page | +0.10 |
| JavaScript embedded in PDF | +0.30 |

Uses **regex word-boundary matching** to avoid false substring matches (e.g., `itext` not matching `context`).

---

### `utils/ocr_extractor.py` — Text Extraction and Analysis

**`extract_text_from_pdf(pdf_path)`**
Uses PyMuPDF `page.get_text()` — fast, no external dependencies needed.
Detects scanned PDFs (avg chars per page < 50). Returns per-page text breakdown.

**`extract_text_from_image(image_path)`**
Uses Tesseract OCR via `pytesseract`. Returns extracted text and per-word confidence scores.
Gracefully handles missing Tesseract (OCR module is skipped, not crashed).

**`analyze_text_anomalies(text)`**

| Pattern Detected | Risk Added |
|-----------------|-----------|
| `Lorem ipsum`, `test document`, `sample text` | +0.30 |
| `John Doe`, `Jane Doe`, `123 Main Street`, `xxx...` | +0.30 |
| Encoding artifacts (mojibake) >5% of characters | +0.20 |
| Inconsistent currency formatting in same document | +0.15 |

---

### `train_model.py` — Model Training Script

Trains EfficientNet-B3 on a dataset of genuine and tampered documents using two-phase training with mixed precision and advanced optimizations.

**Training phases:**

| Phase | Duration | What is trained | Learning rate |
|-------|----------|-----------------|---------------|
| Phase 1 | `--freeze_epochs` (e.g. 5-15) | Classifier head only | `lr × 10`, cosine annealing |
| Phase 2 | Remaining epochs | Full model (backbone + head) | `lr` with 3-epoch linear warmup + cosine decay |

**Training configuration:**
- Loss: Binary cross-entropy with label smoothing (smoothing=0.1) + automatic class weighting (`pos_weight = n_genuine / n_tampered`)
- Optimizer: AdamW, weight decay `1e-4`
- LR schedule: Phase 1 — cosine annealing. Phase 2 — 3-epoch linear warmup then cosine decay (prevents backbone destabilization on unfreeze)
- Mixed precision: FP16 training via `torch.autocast` + `GradScaler` (~50% VRAM reduction, ~30% faster)
- Gradient accumulation: `--accum_steps` mini-batches per optimizer step (effective batch = `batch_size × accum_steps`)
- Gradient clipping: max norm 1.0 (applied after FP16 unscaling)
- `cudnn.benchmark = True` enabled on CUDA for faster convolution kernel selection
- Model saved: best by validation AUC-ROC (not accuracy)
- Early stopping: stops when AUC does not improve for `--patience` epochs

**Training augmentations (applied before ELA):**
- Random horizontal flip (50%)
- Random vertical flip (20%)
- Random rotation ±15° with gray fill
- Brightness and contrast jitter ×[0.8, 1.2]
- Random JPEG re-save at quality 70-95 (teaches ELA robustness to compression)
- Gaussian noise post-ELA (std: 0.01-0.05)

**Supported dataset layouts:**
```
# Structured (recommended, created by prepare_custom_dataset.py):
dataset/
├── train/genuine/    ├── train/tampered/
├── val/genuine/      ├── val/tampered/
└── test/genuine/     └── test/tampered/

# Flat (auto-split: 75% train / 12.5% val / 12.5% test):
dataset/
├── genuine/
└── tampered/
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Path to dataset |
| `--epochs` | 50 | Total training epochs |
| `--batch_size` | 16 | Batch size per step |
| `--lr` | 1e-4 | Base learning rate |
| `--freeze_epochs` | 5 | Backbone freeze duration (Phase 1) |
| `--patience` | 10 | Early stopping patience |
| `--output_dir` | `./fraud_model` | Where to save model weights |
| `--accum_steps` | 2 | Gradient accumulation steps (effective batch = `batch_size × accum_steps`) |

---

### `prepare_custom_dataset.py` — Dataset Augmentation Tool

Takes as few as 5 genuine + 5 tampered documents and augments them into a viable training dataset.

**Augmentations applied to each source document:**
- Random rotations (±30°)
- Random scaling (80-120%)
- Brightness and contrast jitter
- JPEG re-compression at various quality levels
- Gaussian noise injection
- Random perspective warping

Outputs a structured `train/val/test` split ready for `train_model.py`.

---

### `requirements.txt` — Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.115.6 | REST API framework |
| `uvicorn[standard]` | 0.34.0 | ASGI server for FastAPI |
| `python-multipart` | 0.0.20 | File upload parsing |
| `torch` | >=2.1.0 | Deep learning (PyTorch) |
| `torchvision` | >=0.16.0 | EfficientNet model definitions |
| `Pillow` | >=10.0.0 | Image loading and ELA computation |
| `scikit-learn` | >=1.3.0 | DBSCAN clustering, AUC-ROC metric |
| `numpy` | >=1.24.0 | Array math |
| `scipy` | >=1.11.0 | Connected component labeling |
| `pytesseract` | >=0.3.10 | Tesseract OCR Python wrapper |
| `PyMuPDF` | >=1.24.0 | PDF rendering and text extraction |
| `pikepdf` | >=8.0.0 | PDF manipulation |
| `opencv-python-headless` | >=4.8.0 | AKAZE, FLANN, Laplacian |
| `streamlit` | >=1.38.0 | Web UI framework |
| `requests` | >=2.31.0 | HTTP client for Streamlit→API calls |
| `loguru` | >=0.7.0 | Structured logging with rotation |
| `python-dotenv` | >=1.0.0 | Environment variable loading |
| `httpx` | >=0.27.0 | Docker health check HTTP client |

---

### `Dockerfile` — Container Configuration

- Base image: `python:3.11-slim`
- System packages installed: `tesseract-ocr`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, `libgl1-mesa-glx`
- Python packages installed with `--no-cache-dir` to minimize image size
- Exposes port 8000
- Health check: polls `GET /health` every 30 seconds (10s timeout, 3 retries)
- Start command: `uvicorn app:app --host 0.0.0.0 --port 8000`

---

## Detection Algorithms

### 1. Error Level Analysis (ELA) — Multi-Quality Ensemble

Detects JPEG compression inconsistencies caused by pasting content from a different source.

```
Original image
      |
      v
Re-save at JPEG quality 85, 90, and 95  (3 quality levels)
      |
      v
For each quality:
  pixel_diff = |original - resaved|   (per channel, per pixel)
  Amplify differences x15
  99th-percentile threshold → suspicious pixel mask
  scipy connected component labeling → filter regions < 0.5% of image area
  cluster_ratio + normalized_mean + channel_variance → ela_score [0, 1]
      |
      v
ensemble_ela_score = mean(ela_score_85, ela_score_90, ela_score_95)
      |
      v
CNN input uses quality=90 ELA image (separate from ensemble scoring)
```

Genuine images have **uniform** compression artifacts everywhere. Tampered regions stand out as brighter patches. Running at 3 quality levels catches forgeries that are only detectable at specific compression settings.

---

### 2. EfficientNet-B3 CNN

Deep learning classification trained on ELA-processed images.

```
Original image → ELA processing → 300x300 ELA image
      |
      v
ImageNet normalization
      |
      v
EfficientNet-B3 backbone (pretrained on ImageNet → 1536 features)
      |
      v
Custom fraud head:
  Dropout(0.4) → Linear(1536→512) → SiLU
  Dropout(0.3) → Linear(512→256)  → SiLU
  Dropout(0.2) → Linear(256→1)    → Sigmoid
      |
      v
fraud_probability [0.0, 1.0]
```

When no model is loaded, this module returns `None` and its weight (35%) is redistributed — it does **not** contribute a fixed 0.5 score.

---

### 3. AKAZE + FLANN + DBSCAN (Copy-Move Detection)

Detects regions that have been copied and pasted within the same document.

```
Grayscale image
      |
      v
AKAZE: detect keypoints + compute binary descriptors (rotation/scale invariant)
      |
      v
FLANN kd-tree: match descriptors to themselves (kNN, k=2)
      |
      v
Lowe's ratio test: keep only m.distance < 0.75 x n.distance
      |
      v
Filter: reject self-matches, reject pairs closer than 5% of image diagonal
      |
      v
Displacement vector (dx/width, dy/height) per match pair
      |
      v
DBSCAN (eps=0.08, min_samples=3) on displacement vectors
      |
      v
Clustered displacements  →  copy-move forgery detected
Scattered displacements  →  genuine repeated texture (wallpaper, pattern)
```

---

### 4. Laplacian Variance + MAD (Blur/Sharpness Inconsistency)

Detects regions with anomalous sharpness — an indicator of spliced or resized content.

```
Grayscale image → 8x8 grid (64 cells)
      |
      v
Per cell: cv2.Laplacian → variance (edge density / sharpness measure)
      |
      v
median  = median of all 64 variances
MAD     = median(|variance - median|)
      |
      v
Inconsistent cells: |variance - median| > 3 x MAD
      |
      v
blur_score = min(inconsistent_count / 8.0, 1.0)
```

---

### 5. JPEG Ghost Detection

Finds regions originally compressed at a different quality than the surrounding image.

```
Re-save image at 8 quality levels: [60, 65, 70, 75, 80, 85, 90, 95]
      |
      v
Per quality: MSE map = mean((original - resaved)^2, axis=channels)
      |
      v
Per pixel: find quality with minimum MSE → "best-match quality index"
      |
      v
Spatial entropy of quality index map
      |
      v
ghost_score = normalized_entropy x 0.6 + non_dominant_ratio x 0.4
```

---

### 6. Metadata Rule-Based Detection

Applies heuristic rules to PDF and image metadata. Each matching rule adds a fixed amount to the `risk_score`, capped at 1.0.

**PDF rules:**
- pdftk, pdfedit, pdf-xchange, nitro, sejda detected → +0.35
- iText, ReportLab, fpdf, tcpdf detected → +0.15
- Modified >1 year after creation → +0.25
- Modified >30 days after creation → +0.10
- >5 MB per page → +0.10
- JavaScript detected → +0.30

**Image EXIF rules:**
- Photoshop, GIMP, or Paint.NET in Software tag → +0.30
- Original date ≠ modified date → +0.15
- Digitized date ≠ original date → +0.10
- Partial camera EXIF (selective stripping) → +0.15

---

### 7. OCR Text Anomaly Detection

Regex and statistical checks on extracted document text.

**Dummy/placeholder text patterns:**
- `Lorem ipsum`, `test document`, `sample text`, `John Doe`, `Jane Doe`, `123 Main Street`, `xxx...` → +0.30 each

**Encoding artifacts (mojibake):**
- Characters with `ord > 65533` or in range `128-159` exceeding 5% of all characters → +0.20

**Number formatting inconsistency:**
- Mix of `$1,000` and `$1000` styles in same document → +0.15

---

### 8. Weighted Score Fusion

```
fraud_probability = sum(score_i x weight_i for active modules)
                  / sum(weight_i for active modules)
```

Inactive modules (failed or unavailable) have their weights redistributed proportionally so the denominator always equals 1.0.

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Must be added to PATH |
| Git | any | For cloning the repository |
| Tesseract OCR | 5.x | Optional — only needed for image OCR |
| CUDA + cuDNN | 11.8+ | Optional — for GPU acceleration |

### Step 1 — Install Tesseract OCR (optional)

**Windows:** Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Note the install path.

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Step 2 — Clone the Repository

```bash
git clone <your-repo-url>
cd document_fraud_ai
```

### Step 3 — Create Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux / macOS
source venv/bin/activate
```

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This downloads approximately 1.3 GB of packages (PyTorch, OpenCV, SciPy, etc.). Allow 5-15 minutes.

### Step 5 — Configure Tesseract Path (Windows only, if not auto-detected)

Add this near the top of `utils/ocr_extractor.py`:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

On Linux and macOS, Tesseract is auto-detected from PATH.

---

## Running the System

### Start the API Server

```bash
python app.py
```

Expected output:
```
INFO  | Fraud detection pipeline initialized
INFO  | Uvicorn running on http://0.0.0.0:8000
```

- API base URL: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Start the Web UI (in a second terminal)

```bash
# Activate venv first, then:
streamlit run streamlit_app.py
```

Opens automatically at `http://localhost:8501`

### Run with a Trained Model

```bash
# Windows
set FRAUD_MODEL_PATH=./fraud_model/fraud_efficientnet_b3_best.pth
python app.py

# Linux / macOS
export FRAUD_MODEL_PATH=./fraud_model/fraud_efficientnet_b3_best.pth
python app.py
```

---

## API Reference

### POST `/analyze` — Single Document Analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "fraud_probability": 0.7234,
  "is_fraud": true,
  "confidence": 82.5,
  "reasons": [
    "ELA: Suspicious compression artifacts detected (cluster_ratio: 0.0821, ela_score: 0.7234)",
    "CNN model: High tampering probability (75.00%)",
    "PDF modified with high-risk tool: pdftk",
    "Copy-move forgery detected (14 matching regions, 3 displacement clusters)"
  ],
  "details": {
    "module_scores": {
      "ela_statistical": 0.7234,
      "cnn_prediction": 0.75,
      "metadata": 0.60,
      "text_anomaly": 0.0,
      "copy_move": 0.45,
      "blur_sharpness": 0.125
    },
    "model_used": "EfficientNet-B3",
    "device": "cuda",
    "text_extracted": true,
    "text_length": 1250
  },
  "processing_time_seconds": 2.341,
  "filename": "document.pdf"
}
```

**Response field reference:**

| Field | Type | Description |
|-------|------|-------------|
| `fraud_probability` | float [0,1] | Weighted combined score from all modules |
| `is_fraud` | bool | True when probability > 0.5 |
| `confidence` | float [0,100] | Based on agreement + decisiveness + coverage |
| `reasons` | list[str] | Human-readable anomaly descriptions |
| `details.module_scores` | dict | Per-module score [0,1] or null if unavailable |
| `details.model_used` | str | "EfficientNet-B3" or "Heuristic (no pretrained model)" |
| `details.device` | str | "cuda" or "cpu" |
| `processing_time_seconds` | float | Total analysis time in seconds |
| `filename` | str | Original uploaded filename |

---

### POST `/analyze/batch` — Batch Analysis (up to 10 files)

```bash
curl -X POST http://localhost:8000/analyze/batch \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.jpg" \
  -F "files=@doc3.png"
```

**Response:**
```json
{
  "results": [
    { "filename": "doc1.pdf", "fraud_probability": 0.72, "is_fraud": true, "..." : "..." },
    { "filename": "doc2.jpg", "fraud_probability": 0.12, "is_fraud": false, "...": "..." },
    { "filename": "doc3.png", "fraud_probability": 0.45, "is_fraud": false, "...": "..." }
  ],
  "total": 3
}
```

---

### GET `/health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### GET `/model/info` — Model and Pipeline Information

```bash
curl http://localhost:8000/model/info
```

```json
{
  "model_type": "EfficientNet-B3",
  "device": "cuda",
  "modules": [
    "Error Level Analysis (ELA) — adaptive spatial clustering",
    "EfficientNet-B3 Classification",
    "Metadata Consistency Check",
    "OCR + Text Anomaly Detection",
    "Copy-Move Forgery Detection (AKAZE + DBSCAN)",
    "Blur/Sharpness Inconsistency"
  ],
  "supported_formats": [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"],
  "max_file_size_mb": 50
}
```

---

## Web UI

The Streamlit interface at `http://localhost:8501` provides:

- **File uploader** — drag and drop or click to browse (all supported formats)
- **Image preview** — inline preview for image files
- **Fraud verdict** — red banner (fraud detected) or green banner (genuine)
- **Confidence metric** — percentage display
- **Processing time** — seconds taken
- **Detected indicators** — list of anomaly descriptions
- **Module scores** — progress bar for each of the 6 detection modules (0.0 to 1.0)
- **Raw JSON** — expandable section with the full API response
- **System info sidebar** — live model type, device, and supported formats from `/model/info`

---

## Training a Custom Model

### Option A — Small Custom Dataset (5+ documents per class)

```bash
# Step 1: Organize your documents
mkdir -p my_docs/genuine my_docs/fake
# Copy genuine PDFs/images to my_docs/genuine/
# Copy tampered PDFs/images to my_docs/fake/

# Step 2: Augment to create a viable training dataset
python prepare_custom_dataset.py \
  --genuine_dir ./my_docs/genuine \
  --fake_dir ./my_docs/fake \
  --output_dir ./dataset \
  --augment_factor 30

# Step 3: Train
python train_model.py \
  --data_dir ./dataset \
  --epochs 30 \
  --batch_size 8 \
  --freeze_epochs 15 \
  --patience 8 \
  --accum_steps 4

# Step 4: Use the trained model
set FRAUD_MODEL_PATH=./fraud_model/fraud_efficientnet_b3_best.pth
python app.py
```

### Option B — Large Public Dataset (CASIA, COVERAGE, Columbia)

```bash
# Organize in structured layout first, then:
python train_model.py \
  --data_dir ./dataset \
  --epochs 50 \
  --batch_size 16 \
  --freeze_epochs 5 \
  --patience 10 \
  --lr 1e-4 \
  --accum_steps 2
```

### Recommended Public Datasets

| Dataset | Size | Focus |
|---------|------|-------|
| [CASIA v2.0](https://github.com/namtpham/casia2groundtruth) | ~10,000 pairs | General image tampering |
| [Columbia Uncompressed](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) | 180 pairs | Splicing detection |
| [CoMoFoD](https://www.vcl.fer.hr/comofod/) | 260+ images | Copy-move forgery |
| [COVERAGE](https://github.com/wenbihan/coverage) | 100 pairs | Copy-move with similar objects |

### Training Output

Best model saved to:
```
fraud_model/fraud_efficientnet_b3_best.pth
```

Sample training log:
```
Epoch 12/50 | Train Loss: 0.2341, Acc: 0.9123 | Val Loss: 0.2891, Acc: 0.8750, AUC: 0.9234
Best model saved: ./fraud_model/fraud_efficientnet_b3_best.pth (val_auc: 0.9234)
```

---

## Docker Deployment

### Basic Deployment

```bash
# Build the image
docker build -t document-fraud-ai .

# Run on CPU
docker run -p 8000:8000 document-fraud-ai

# Run on GPU (requires nvidia-container-toolkit)
docker run --gpus all -p 8000:8000 document-fraud-ai
```

### With a Trained Model

```bash
docker run -p 8000:8000 \
  -v ./fraud_model:/app/fraud_model \
  -e FRAUD_MODEL_PATH=/app/fraud_model/fraud_efficientnet_b3_best.pth \
  document-fraud-ai
```

### Docker Compose

```yaml
version: "3.8"
services:
  fraud-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FRAUD_MODEL_PATH=/app/fraud_model/fraud_efficientnet_b3_best.pth
      - FRAUD_DEVICE=cpu
    volumes:
      - ./fraud_model:/app/fraud_model
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_MODEL_PATH` | None | Path to trained `.pth` file. Omit to run in heuristic mode. |
| `FRAUD_DEVICE` | auto | `cuda` or `cpu`. Auto-detected from CUDA availability if not set. |
| `API_URL` | `http://localhost:8000` | Backend URL — used by Streamlit UI only. |

### Heuristic Mode (No Trained Model)

When `FRAUD_MODEL_PATH` is not set, the system still runs using:
- ELA statistical analysis
- Metadata consistency checks
- OCR text anomaly detection
- AKAZE copy-move detection
- Laplacian blur inconsistency

The CNN module returns `None` and its 35% weight is redistributed to the remaining active modules.

---

## Security

- Uploaded files are deleted immediately after analysis — no persistent storage of user documents
- File size is enforced at the chunked read level (not just from HTTP headers)
- File type is validated by extension before processing
- No code or scripts within uploaded documents are ever executed
- CORS is fully open by default — restrict `allow_origins` in production
- No authentication by default — add API key or OAuth2 middleware for production
- No rate limiting by default — add rate limiting middleware for production
- Logs are retained 7 days and rotated at 10 MB

**Production deployment checklist:**
- [ ] Restrict `allow_origins` in CORS middleware to known frontend domains
- [ ] Add authentication (API key header, OAuth2, or JWT)
- [ ] Add rate limiting (e.g., `slowapi` library)
- [ ] Serve behind HTTPS reverse proxy (nginx, Caddy, or Traefik)
- [ ] Set strict file permissions on `uploads/` and `logs/` directories
- [ ] Consider sandboxing the PyMuPDF PDF renderer

---

## Troubleshooting

**`ModuleNotFoundError` on startup**
Virtual environment is not activated. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/macOS) before `python app.py`.

**`tesseract is not installed or not in PATH`**
Install Tesseract OCR (see Installation section) or manually set the path in `utils/ocr_extractor.py`. Without Tesseract, the OCR module is skipped gracefully — the system still works.

**`torch` installation fails or takes too long**
Install PyTorch separately with the correct wheel URL:
```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Port 8000 already in use**
```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```
Then set `API_URL=http://localhost:8001` for the Streamlit UI.

**Streamlit shows "API not reachable"**
Make sure `python app.py` is running in a separate terminal before starting Streamlit.

**Analysis is slow**
Without a GPU, each document takes ~2-3 seconds (CPU). With CUDA GPU, ~0.5-1 second. Check `GET /health` to confirm whether `device` is `cuda` or `cpu`.

**High false positive rate without a trained model**
In heuristic mode, some legitimate documents may be flagged (e.g., PDFs generated with iText for valid business purposes). Training a custom model on your document type significantly reduces false positives.

---

## Future Improvements

- Multi-page PDF analysis (currently analyzes first page only)
- Signature verification module
- Font consistency analysis across the document
- GAN-based deepfake document detection
- Tampered region localization (pixel-level segmentation map)
- Confidence calibration with Platt scaling
- Support for handwritten document analysis
- Integration with external fraud databases
