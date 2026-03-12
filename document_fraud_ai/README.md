# Document Fraud Detection System

An AI-powered system that analyzes PDF documents and images to detect tampering, forgery, and manipulation. It combines 7 independent detection modules — deep learning, signal processing, computer vision, and rule-based analysis — into a single weighted fraud probability score.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Detection Modules](#detection-modules)
- [V2 Optimizations](#v2-optimizations)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Running the System](#running-the-system)
  - [Quick Start (Heuristic Mode)](#quick-start-heuristic-mode)
  - [FastAPI Server](#fastapi-server)
  - [Streamlit Web UI](#streamlit-web-ui)
  - [Running on Google Colab](#running-on-google-colab)
- [Training the Model](#training-the-model)
  - [Dataset Preparation](#dataset-preparation)
  - [Training Commands](#training-commands)
  - [Training Flags Explained](#training-flags-explained)
  - [Training Output](#training-output)
  - [Using the Trained Model](#using-the-trained-model)
- [API Reference](#api-reference)
- [Model Architecture](#model-architecture)
- [Performance & GPU Optimization](#performance--gpu-optimization)
- [File-by-File Reference](#file-by-file-reference)
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
                 |  |  |  |  |  |  |
   +-------------+  |  |  |  |  |  +---------------+
   |                |  |  |  |  |                   |
   v                v  v  v  v  v                   v
  ELA            CNN  MT Meta OCR Copy-Move      Blur
 (20%)          (25%)(25%)(10%)(8%)  (8%)        (4%)
   |                |  |  |  |  |                   |
   +----------------+--+--+--+-+-------------------+
                        |
              +---------v----------+
              |   Weighted Score   |
              |  Fusion + Reasons  |
              +---------+----------+
                        |
              +---------v----------+
              |    JSON Response   |
              +--------------------+

MT = ManTraNet (optional, pixel-level forgery detection)
```

The system works **without a trained model** (heuristic mode) and is significantly more accurate **with** a fine-tuned EfficientNet-B3 V2 model.

---

## Architecture

### How It Works (Step by Step)

1. **Input**: User uploads a document (PDF or image) via REST API or Streamlit UI
2. **Preprocessing**: PDFs are rendered to images at 200 DPI. Images are loaded as RGB.
3. **Parallel Analysis**: 7 independent modules each produce a risk score [0.0, 1.0]:
   - **ELA Statistical** — re-saves at 3 JPEG quality levels, detects anomalous compression regions
   - **CNN (EfficientNet-B3 V2)** — deep learning classification on ELA images with TTA
   - **ManTraNet** — pixel-level forgery detection (optional, 385 manipulation types)
   - **Metadata Analysis** — checks EXIF dates, editing tools, PDF producer strings
   - **OCR Text Anomaly** — extracts text, detects placeholders and encoding artifacts
   - **Copy-Move Detection** — AKAZE feature matching + DBSCAN displacement clustering
   - **Blur/Sharpness** — detects local sharpness inconsistencies via Laplacian variance
4. **Score Fusion**: Weighted combination with automatic redistribution for missing modules
5. **Output**: JSON with `fraud_probability`, `is_fraud`, `confidence`, per-module scores, and reasons

### Score Fusion Logic

- Each module outputs a score in [0.0, 1.0] or `None` (if unavailable)
- `None` modules are excluded; their weight is redistributed proportionally to active modules
- Final score is always normalized to [0.0, 1.0]
- `is_fraud = True` when `fraud_probability > 0.5`

### Confidence Scoring

Confidence is computed from three factors:

| Factor | Weight | What It Measures |
|--------|--------|------------------|
| **Agreement** | 40% | Low std deviation among module scores = high agreement |
| **Decisiveness** | 40% | Distance of final score from 0.5 (the uncertain midpoint) |
| **Coverage** | 20% | Fraction of total weight from active modules |

---

## Detection Modules

| Module | Weight | Algorithm | What It Detects |
|--------|--------|-----------|-----------------|
| **ELA Statistical** | 20% | Multi-quality JPEG re-compression (85/90/95) + adaptive 99th-percentile thresholding + spatial clustering | Compression artifacts from image editing |
| **CNN (EfficientNet-B3 V2)** | 25% | Deep learning on ELA images with TTA (4 views) + multi-quality (3 levels) | Learned tampering patterns |
| **ManTraNet** | 25% | Pre-trained on 385 manipulation types, tiled full-resolution inference | Pixel-level splicing, copy-move, removal, enhancement |
| **Metadata Analysis** | 10% | EXIF/PDF metadata parsing, date consistency, tool detection | Editing tool signatures, date mismatches, suspicious producers |
| **OCR Text Anomaly** | 8% | Tesseract OCR + regex pattern matching | Placeholder text, encoding artifacts, number formatting inconsistencies |
| **Copy-Move Detection** | 8% | AKAZE keypoints + FLANN matching + DBSCAN on displacement vectors | Cloned/duplicated regions within the same document |
| **Blur/Sharpness** | 4% | Laplacian variance on grid cells + MAD outlier detection | Locally blurred/sharpened regions (selective editing) |

---

## V2 Optimizations

### Model (cnn_model.py)

| Change | Why | Impact |
|--------|-----|--------|
| Raw logit output (no Sigmoid) | BCEWithLogitsLoss uses log-sum-exp trick, avoids underflow | Stable training with FP16 |
| SE (Squeeze-Excitation) attention | Learns per-channel importance weights | +0.5-1.5% accuracy |
| BatchNorm in classifier head | Stabilizes training with aggressive augmentation | Faster convergence |
| GELU activation | Smoother gradients than SiLU near zero | Better with label smoothing |
| Wider head (768-384-1 vs 512-256-1) | Captures more complex ELA patterns | +capacity |
| Kaiming initialization | Proper variance scaling for ReLU-family activations | Stable training start |

### Training (train_model.py)

| Feature | Flag | AUC Gain | Description |
|---------|------|----------|-------------|
| **Focal Loss** | `--focal_loss` | +1-2% | Down-weights easy examples, focuses on hard fraud cases |
| **Mixup** | `--mixup` | +1-2% | Blends training pairs, reduces overfitting by 30-50% |
| **EMA** | `--ema` | +0.5-1% | Exponential Moving Average of weights, smoother convergence |
| **TTA Validation** | `--tta` | +0.3-0.8% | Averages 4 augmented views for more reliable AUC |
| **OneCycleLR** | (default) | +convergence | Super-convergence scheduler, often outperforms cosine |
| **Perspective warp** | (default) | +robustness | Simulates photographed documents |
| **RandomErasing** | (default) | +robustness | Forces model to use global patterns, not local shortcuts |
| **Stronger color jitter** | (default) | +robustness | Wider brightness/contrast/saturation range |

### Inference (pipeline.py)

| Feature | Default | Description |
|---------|---------|-------------|
| **TTA** | On | Averages original + 3 flipped views (4x inference, more robust) |
| **Multi-quality CNN** | On | Runs CNN on ELA at 85/90/95 quality, averages predictions |
| **Auto V1/V2 detection** | Auto | Detects model version from state_dict keys |
| **Temperature scaling** | 1.0 | Calibrate probabilities on held-out set (adjustable) |

---

## Project Structure

```
document_fraud_ai/
+-- app.py                        # FastAPI REST API server
+-- streamlit_app.py              # Streamlit web UI frontend
+-- train_model.py                # EfficientNet-B3 V2 training (Focal Loss, Mixup, EMA, TTA)
+-- prepare_custom_dataset.py     # Dataset augmentation tool
+-- requirements.txt              # Python dependencies
+-- Dockerfile                    # Docker container config
+-- README.md                     # This file
+-- fraud_model/
|   +-- __init__.py
|   +-- cnn_model.py              # CNN architectures (V1 + V2 with SE attention)
|   +-- pipeline.py               # Main orchestration pipeline (TTA + multi-quality)
+-- utils/
|   +-- __init__.py
|   +-- ela_analysis.py           # ELA, copy-move, blur, JPEG ghost
|   +-- metadata_analyzer.py      # PDF/image metadata analysis
|   +-- ocr_extractor.py          # Text extraction and anomaly detection
|   +-- mantranet.py              # ManTraNet pixel-level forgery detection
+-- uploads/                      # Temporary upload directory (auto-created)
+-- logs/                         # Application logs (auto-created)
```

---

## System Requirements

### Minimum (CPU-only, heuristic mode)

| Resource | Requirement |
|----------|------------|
| **CPU** | Any modern 4-core |
| **RAM** | 8 GB |
| **Disk** | ~3 GB (Python + deps + PyTorch CPU) |
| **GPU** | Not needed |
| **Python** | 3.9+ |

### Recommended (GPU, full accuracy)

| Resource | Requirement |
|----------|------------|
| **CPU** | 4+ cores |
| **RAM** | 16 GB |
| **Disk** | ~8 GB (PyTorch CUDA + model weights) |
| **GPU** | NVIDIA with 4+ GB VRAM (CUDA) |
| **Python** | 3.10 or 3.11 |

### Disk Breakdown

| Component | Size |
|-----------|------|
| PyTorch (CPU) | ~800 MB |
| PyTorch (CUDA) | ~2.5 GB |
| EfficientNet-B3 weights | ~50 MB |
| ManTraNet weights | ~100 MB |
| Other pip packages | ~500 MB |

---

## Installation

### 1. Clone / Download

```bash
git clone <your-repo-url>
cd document_fraud_ai
```

### 2. Create Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (optional, for OCR module)

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Running the System

### Quick Start (Heuristic Mode)

No trained model needed. Runs ELA, metadata, OCR, copy-move, and blur modules:

```python
from fraud_model.pipeline import FraudDetectionPipeline

pipeline = FraudDetectionPipeline(device="cpu")
result = pipeline.analyze("my_document.pdf")

print(f"Fraud Probability: {result['fraud_probability']:.1%}")
print(f"Is Fraud: {result['is_fraud']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Reasons: {result['reasons']}")
```

### FastAPI Server

```bash
python app.py
# Server starts at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

With a trained model:
```bash
FRAUD_MODEL_PATH=./fraud_model/fraud_efficientnet_b3_v2_best.pth python app.py
```

Test with curl:
```bash
# Single document
curl -X POST http://localhost:8000/analyze -F "file=@document.pdf"

# Batch (up to 10)
curl -X POST http://localhost:8000/analyze/batch \
  -F "files=@doc1.pdf" -F "files=@doc2.jpg"

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

### Streamlit Web UI

```bash
streamlit run streamlit_app.py
# Opens browser at http://localhost:8501
```

Make sure the FastAPI server is running first (Streamlit connects to it).

### Running on Google Colab

Use the included `Document_Fraud_Detection_Colab.ipynb` notebook, or:

```python
# Cell 1: Install dependencies
!pip install -q fastapi uvicorn[standard] python-multipart PyMuPDF loguru \
    pytesseract gdown pyngrok nest_asyncio
!apt-get install -y -q tesseract-ocr

# Cell 2: Initialize pipeline
from fraud_model.pipeline import FraudDetectionPipeline
pipeline = FraudDetectionPipeline(device="cuda")

# Cell 3: Analyze a document
result = pipeline.analyze("your_document.pdf")
print(result)

# Cell 4: Start API server with public URL
import nest_asyncio, threading, uvicorn
nest_asyncio.apply()
from app import app
import app as app_module
app_module.pipeline = pipeline
threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000),
    daemon=True
).start()

from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"API: {public_url}/docs")
```

---

## Training the Model

### Dataset Preparation

#### Option A: Large Dataset (50+ images per class)

Organize your images into two folders:

```
dataset/
+-- genuine/     <- real, unmodified documents
+-- tampered/    <- forged, edited, or manipulated documents
```

Or use the structured layout (pre-split):

```
dataset/
+-- train/
|   +-- genuine/
|   +-- tampered/
+-- val/
|   +-- genuine/
|   +-- tampered/
+-- test/
|   +-- genuine/
|   +-- tampered/
```

#### Option B: Tiny Dataset (5-10 images per class)

Use the augmentation tool to expand your dataset:

```bash
python prepare_custom_dataset.py \
    --genuine_dir ./my_docs/genuine \
    --fake_dir ./my_docs/fake \
    --output_dir ./dataset \
    --augment_factor 30
```

This applies 13 augmentation types (rotation, perspective warp, color jitter, JPEG re-compression, blur, noise, etc.) to expand each image into 30 variants.

**Output**: 5 genuine + 5 fake originals become ~150 + 150 training images.

### Training Commands

**Standard training (recommended starting point):**

```bash
python train_model.py \
    --data_dir ./dataset \
    --epochs 50 \
    --batch_size 16 \
    --freeze_epochs 5 \
    --patience 10
```

**Maximum accuracy (all V2 optimizations):**

```bash
python train_model.py \
    --data_dir ./dataset \
    --epochs 60 \
    --batch_size 8 \
    --accum_steps 4 \
    --freeze_epochs 5 \
    --patience 12 \
    --focal_loss \
    --mixup \
    --ema \
    --tta
```

**Low VRAM GPU (4-6 GB):**

```bash
python train_model.py \
    --data_dir ./dataset \
    --batch_size 4 \
    --accum_steps 8 \
    --epochs 50 \
    --freeze_epochs 5 \
    --focal_loss --mixup --ema
# Effective batch size = 32, VRAM usage ~ 3-4 GB with FP16
```

**Tiny dataset (5+5 documents, after augmentation):**

```bash
python train_model.py \
    --data_dir ./dataset \
    --epochs 30 \
    --batch_size 8 \
    --freeze_epochs 15 \
    --patience 8 \
    --focal_loss --mixup --ema
```

**Google Colab:**

```bash
!python train_model.py \
    --data_dir /content/dataset \
    --batch_size 8 \
    --accum_steps 4 \
    --epochs 50 \
    --freeze_epochs 5 \
    --focal_loss --mixup --ema --tta
```

### Training Flags Explained

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | *required* | Path to dataset directory |
| `--epochs` | 50 | Total training epochs |
| `--batch_size` | 16 | Batch size per step |
| `--lr` | 1e-4 | Base learning rate |
| `--freeze_epochs` | 5 | Epochs to freeze backbone (train head only) |
| `--patience` | 10 | Early stopping patience (epochs without AUC improvement) |
| `--output_dir` | ./fraud_model | Where to save model and training history |
| `--accum_steps` | 2 | Gradient accumulation steps (effective batch = batch_size x accum_steps) |
| `--focal_loss` | off | Use Focal Loss (better for imbalanced data, +1-2% AUC) |
| `--mixup` | off | Enable Mixup augmentation (blends training pairs, +1-2% AUC) |
| `--ema` | off | Enable EMA weight averaging (smoother convergence, +0.5-1% AUC) |
| `--tta` | off | Enable TTA at validation (4x slower but more reliable AUC, +0.3-0.8%) |

### Training Process (What Happens Internally)

```
Phase 1 (epochs 1 to freeze_epochs):
  - Backbone FROZEN (only classifier head trains)
  - Higher LR (10x base) for fast head adaptation
  - CosineAnnealingLR scheduler
  - Purpose: adapt the classifier to ELA features without destroying pretrained backbone

Phase 2 (epochs freeze_epochs+1 to end):
  - Backbone UNFROZEN (full fine-tuning)
  - OneCycleLR scheduler (10% warmup + cosine decay)
  - EMA reinitialized with all parameters
  - Purpose: fine-tune entire network for fraud-specific features
```

Each epoch:
1. Training loop with optional Mixup, FP16 mixed precision, gradient accumulation
2. EMA weight update after each optimizer step
3. Validation with optional TTA (4 views averaged)
4. Metrics: loss, accuracy, AUC-ROC, F1, precision, recall
5. Best model saved by AUC-ROC (EMA weights if enabled)
6. Early stopping if no AUC improvement for `patience` epochs

### Training Output

After training completes, you will find in `--output_dir`:

| File | Description |
|------|-------------|
| `fraud_efficientnet_b3_v2_best.pth` | Best model weights (saved at peak validation AUC) |
| `training_history.json` | All metrics per epoch (loss, acc, AUC, F1, LR, time) |
| `training_curves.png` | Visual plot: loss, accuracy, AUC+F1, learning rate |

### Using the Trained Model

```python
from fraud_model.pipeline import FraudDetectionPipeline

# Load with trained model — CNN module becomes active (25% weight)
pipeline = FraudDetectionPipeline(
    model_path="./fraud_model/fraud_efficientnet_b3_v2_best.pth",
    device="cuda",          # or "cpu"
    enable_tta=True,        # TTA at inference (default: True)
    enable_multi_quality_cnn=True,  # Multi-quality ELA for CNN (default: True)
)

result = pipeline.analyze("suspicious_document.pdf")

print(f"Fraud: {result['fraud_probability']:.1%}")
print(f"Verdict: {'FRAUD' if result['is_fraud'] else 'GENUINE'}")
print(f"Confidence: {result['confidence']:.1f}%")
for reason in result['reasons']:
    print(f"  - {reason}")
```

Environment variables for the FastAPI server:
```bash
export FRAUD_MODEL_PATH=./fraud_model/fraud_efficientnet_b3_v2_best.pth
export FRAUD_DEVICE=cuda   # or cpu
python app.py
```

---

## API Reference

### `POST /analyze`

Upload a single document for fraud analysis.

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "fraud_probability": 0.7234,
  "is_fraud": true,
  "confidence": 78.5,
  "reasons": [
    "ELA: Suspicious compression artifacts detected (cluster_ratio: 0.0312, ela_score: 0.4521)",
    "CNN model: High tampering probability (82.34%)"
  ],
  "details": {
    "module_scores": {
      "ela_statistical": 0.4521,
      "cnn_prediction": 0.8234,
      "mantranet": null,
      "metadata": 0.15,
      "text_anomaly": 0.0,
      "copy_move": 0.0,
      "blur_sharpness": 0.12
    },
    "model_used": "efficientnet_b3_v2 + TTA + MultiQ-CNN",
    "device": "cuda",
    "metadata_summary": { ... },
    "text_extracted": true,
    "text_length": 1523
  },
  "processing_time_seconds": 2.341,
  "filename": "document.pdf"
}
```

### `POST /analyze/batch`

Upload up to 10 documents. Returns array of individual results.

### `GET /health`

```json
{"status": "healthy", "model_loaded": true, "device": "cuda"}
```

### `GET /model/info`

```json
{
  "model_type": "EfficientNet-B3 V2",
  "device": "cuda",
  "modules": ["ELA", "CNN", "Metadata", "OCR", "Copy-Move", "Blur"],
  "supported_formats": [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"],
  "max_file_size_mb": 50
}
```

---

## Model Architecture

### EfficientNet-B3 V2 (Recommended)

```
Input:       ELA image (batch, 3, 300, 300)
                 |
Backbone:    EfficientNet-B3 pretrained on ImageNet
                 |
             Global Average Pooling -> 1536 features
                 |
Head:        Dropout(0.5)
             Linear(1536 -> 768) + BatchNorm + GELU
             SE Attention(768, reduction=16)
             Dropout(0.3)
             Linear(768 -> 384) + BatchNorm + GELU
             Dropout(0.15)
             Linear(384 -> 1) -> raw logit
                 |
Output:      Raw logit (apply sigmoid at inference)
```

**SE (Squeeze-Excitation) Attention**:
```
Input features (768) -> AdaptiveAvgPool -> Linear(768->48) -> GELU -> Linear(48->768) -> Sigmoid -> Scale input
```

Learns per-channel importance: which of the 768 features are most relevant for fraud detection.

### EfficientNet-B3 V1 (Backward Compatible)

```
Input:    (batch, 3, 300, 300)
Backbone: EfficientNet-B3 -> 1536 features
Head:     Dropout(0.4) -> Linear(1536->512) -> SiLU
          Dropout(0.3) -> Linear(512->256) -> SiLU
          Dropout(0.2) -> Linear(256->1) -> Sigmoid
Output:   Probability [0.0, 1.0]
```

### Lightweight CNN (No pretrained weights)

```
Input: (batch, 3, 128, 128)
Block 1: Conv2d(32) -> BN -> ReLU -> Conv2d(32) -> BN -> ReLU -> MaxPool -> Dropout2d(0.25)
Block 2: Conv2d(64) -> BN -> ReLU -> Conv2d(64) -> BN -> ReLU -> MaxPool -> Dropout2d(0.25)
Block 3: Conv2d(128) -> BN -> ReLU -> Conv2d(128) -> BN -> ReLU -> MaxPool -> Dropout2d(0.25)
Head:    Flatten -> Linear(32768->256) -> ReLU -> Dropout(0.5) -> Linear(256->1) -> Sigmoid
```

---

## Performance & GPU Optimization

### Inference

| Optimization | Where | VRAM | Speed | Accuracy |
|---|---|---|---|---|
| FP16 mixed precision (`torch.autocast`) | pipeline.py | **-50%** | +30% | neutral |
| `torch.compile(mode='reduce-overhead')` | pipeline.py | neutral | **+20-30%** | neutral |
| Multi-quality ELA ensemble (85/90/95) | pipeline.py | neutral | -10% CPU | **+robustness** |
| TTA (4 augmented views) | pipeline.py | neutral | -4x CNN | **+0.3-0.8% AUC** |
| Multi-quality CNN (3 ELA levels) | pipeline.py | neutral | -3x CNN | **+robustness** |

### Training

| Optimization | Where | VRAM | Speed | Accuracy |
|---|---|---|---|---|
| FP16 mixed precision (`GradScaler`) | train_model.py | **-50%** | +30% | neutral |
| Focal Loss | train_model.py | neutral | neutral | **+1-2% AUC** |
| Mixup augmentation | train_model.py | neutral | neutral | **+1-2% AUC** |
| EMA (decay=0.999) | train_model.py | +small | neutral | **+0.5-1% AUC** |
| Label smoothing (0.1) | train_model.py | neutral | neutral | **+0.5-1% AUC** |
| OneCycleLR (Phase 2) | train_model.py | neutral | neutral | **+convergence** |
| Gradient accumulation (`--accum_steps`) | train_model.py | **-50%** | neutral | +larger effective batch |
| `cudnn.benchmark = True` | train_model.py | neutral | +5-10% | neutral |
| Perspective warp augmentation | train_model.py | neutral | neutral | **+robustness** |
| RandomErasing | train_model.py | neutral | neutral | **+robustness** |

---


## File-by-File Reference

### `app.py` — FastAPI Backend Server

The main entry point. Starts the REST API server on port 8000.

- Accepts file uploads via `POST /analyze` and `POST /analyze/batch`
- Validates file type and size (max 50 MB)
- Saves files temporarily, runs the detection pipeline, then deletes immediately
- Auto-initializes `FraudDetectionPipeline` on startup via FastAPI lifespan
- CORS open to all origins (configure for production)
- Logs stored in `logs/` with 10 MB rotation and 7-day retention

**Environment variables:**
- `FRAUD_MODEL_PATH` — path to trained `.pth` model file (optional)
- `FRAUD_DEVICE` — `cuda` or `cpu` (auto-detected if not set)

### `streamlit_app.py` — Web UI Frontend

Browser-based drag-and-drop interface built with Streamlit.

- Sidebar shows live system info from `/model/info` endpoint
- File uploader with inline image preview
- Displays fraud verdict (color-coded), confidence, processing time
- Per-module score progress bars
- Expandable raw JSON response

**Environment variable:**
- `API_URL` — backend URL (default: `http://localhost:8000`)

### `fraud_model/cnn_model.py` — CNN Model Architectures

Defines model architectures and `load_model()` function.

- **FraudEfficientNetB3V2** (default, recommended) — logit output, SE attention, BatchNorm, GELU
- **FraudEfficientNetB3** (V1, backward compatible) — sigmoid output, SiLU
- **FraudEfficientNet** (B0, lighter) — sigmoid output
- **FraudCNN** (custom, no pretrained) — 3-block conv net
- **SEBlock** — Squeeze-and-Excitation channel attention module

### `fraud_model/pipeline.py` — Detection Pipeline Orchestrator

Core module that runs all detection modules and combines scores.

- **V2 features**: TTA inference, multi-quality CNN, auto V1/V2 detection, temperature scaling
- Score combination with weight redistribution for missing modules
- Confidence scoring based on agreement, decisiveness, and coverage

### `train_model.py` — Training Script

Full training pipeline for EfficientNet-B3 V2.

- **Losses**: FocalBCEWithLogitsLoss, SmoothedBCEWithLogitsLoss
- **Augmentation**: perspective warp, RandomErasing, strong color jitter, JPEG re-save
- **Optimization**: Mixup, EMA, OneCycleLR, gradient accumulation, FP16
- **Validation**: TTA, AUC-ROC, F1, precision, recall
- **Output**: best model .pth, training_history.json, training_curves.png

### `prepare_custom_dataset.py` — Dataset Augmentation

Expands a small document set into a full training dataset.

- 13 augmentation types (rotation, perspective, color, blur, JPEG, noise, etc.)
- Stratified train/val/test split
- PDF rendering to images at 200 DPI
- Configurable augmentation factor (default: 30x)

### `utils/ela_analysis.py` — ELA & Image Analysis

- `compute_ela()` — JPEG re-compression difference
- `compute_ela_statistics()` — adaptive thresholding + spatial clustering
- `detect_copy_move()` — AKAZE + FLANN + DBSCAN
- `compute_blur_sharpness()` — Laplacian grid analysis

### `utils/metadata_analyzer.py` — Metadata Analysis

- PDF metadata via PyMuPDF (producer, creator, dates, JavaScript detection)
- Image EXIF via Pillow (dates, camera info, editing software)
- Tiered scoring for date mismatches and tool detection

### `utils/ocr_extractor.py` — OCR Text Extraction

- PDF text via PyMuPDF (fast, no OCR needed)
- Image text via Tesseract OCR
- Anomaly detection: placeholders, encoding artifacts, number formatting

### `utils/mantranet.py` — ManTraNet Integration

- Wraps ManTraNet-pytorch (RonyAbecidan's reimplementation)
- Tiled full-resolution inference (512x512 tiles with 64px overlap)
- Spatial cluster scoring to distinguish real tampering from noise
- FP16 + torch.compile optimizations

**Setup:**
```python
from utils.mantranet import setup_mantranet
setup_mantranet(weights_gdrive_id="YOUR_FILE_ID")
# or
setup_mantranet(weights_path="/path/to/ManTraNet.pt")
```

---

## Docker Deployment

### Build

```bash
docker build -t fraud-detector .
```

### Run

```bash
# CPU only
docker run -p 8000:8000 fraud-detector

# With GPU
docker run --gpus all -p 8000:8000 fraud-detector

# With trained model
docker run -p 8000:8000 \
  -v /path/to/model:/app/model \
  -e FRAUD_MODEL_PATH=/app/model/fraud_efficientnet_b3_v2_best.pth \
  fraud-detector
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_MODEL_PATH` | None | Path to trained model `.pth` file |
| `FRAUD_DEVICE` | auto | `cuda` or `cpu` (auto-detects GPU) |
| `API_URL` | `http://localhost:8000` | Backend URL (for Streamlit) |

### Pipeline Tuning

```python
pipeline = FraudDetectionPipeline(
    model_path="model.pth",     # Path to trained model
    device="cuda",               # Device
    enable_tta=True,             # TTA at inference (4 views)
    enable_multi_quality_cnn=True,  # CNN on 3 ELA quality levels
)
pipeline.temperature = 1.0       # Adjust for probability calibration
```

### Module Weights

Edit `WEIGHTS` dict in `fraud_model/pipeline.py` to adjust module importance:

```python
WEIGHTS = {
    "ela_statistical": 0.20,
    "cnn_prediction":  0.25,
    "mantranet":       0.25,
    "metadata":        0.10,
    "text_anomaly":    0.08,
    "copy_move":       0.08,
    "blur_sharpness":  0.04,
}
```

---

## Security

- Uploaded files are deleted immediately after analysis
- File size limited to 50 MB (enforced at chunk-read level, not post-upload)
- File type validated by extension before processing
- No user data is stored or logged (only analysis metadata in logs)
- CORS is open by default — **restrict `allow_origins` for production**
- No JavaScript execution in uploaded PDFs (detected and flagged as anomaly)

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'fitz'` | `pip install PyMuPDF` |
| `TesseractNotFoundError` | Install Tesseract OCR binary (see Installation) |
| `CUDA out of memory` | Reduce `--batch_size`, increase `--accum_steps` |
| `No training samples found` | Check dataset structure (needs `genuine/` and `tampered/` subdirs) |
| CNN score is always `None` | No model loaded — train one or set `FRAUD_MODEL_PATH` |
| ManTraNet score is `None` | Run `setup_mantranet()` to download weights |
| Low AUC after training | Try `--focal_loss --mixup --ema`, increase `--augment_factor` |
| Training too slow | Enable GPU, use `--batch_size 16`, reduce `--epochs` |
| `torch.compile` warning | Safe to ignore — falls back to eager mode on PyTorch < 2.0 |

### Performance Tips

- **Best accuracy**: use all flags: `--focal_loss --mixup --ema --tta`
- **Fastest training**: skip `--tta` (4x slower validation), use large `--batch_size`
- **Low VRAM**: `--batch_size 4 --accum_steps 8` (effective batch = 32, ~3-4 GB)
- **Tiny dataset**: `--augment_factor 40-50` in `prepare_custom_dataset.py`, then `--freeze_epochs 15`
- **Inference speed**: set `enable_tta=False` and `enable_multi_quality_cnn=False` in pipeline for 12x faster CNN

---

## License

MIT License. See LICENSE file for details.
