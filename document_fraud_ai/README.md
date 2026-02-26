# Document Fraud Detection System

AI-powered document fraud detection system that analyzes PDFs and images for tampering, forgery, and manipulation.

## Architecture

```
Document Upload
       |
       v
+------+-------+
| FastAPI Server |
+------+-------+
       |
       v
+------+-----------+
| Analysis Pipeline |
+--+---+---+---+---+
   |   |   |   |   |
   v   v   v   v   v
 ELA  CNN Meta OCR Copy-Move
   |   |   |   |   |
   +---+---+---+---+
       |
       v
  Weighted Score
       |
       v
  JSON Response
```

### Detection Modules

| Module | Technique | What It Detects |
|---|---|---|
| **ELA** | Error Level Analysis | Compression inconsistencies from splicing/editing |
| **CNN** | Convolutional Neural Network | Learned visual patterns of tampering |
| **Metadata** | EXIF/PDF metadata parsing | Editing tool traces, date mismatches |
| **OCR** | Tesseract / PyMuPDF | Placeholder text, encoding artifacts |
| **Copy-Move** | ORB feature matching | Duplicated regions within a document |

Each module produces a risk score [0, 1]. Final fraud probability is a weighted combination.

## Setup

### Prerequisites

- Python 3.10+
- Tesseract OCR (optional, for image OCR)

### Install Tesseract (Optional)

**Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the API Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### Run the Streamlit UI (Optional)

In a second terminal:
```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## API Usage

### Analyze a Single Document

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@document.pdf"
```

### Response Format

```json
{
  "fraud_probability": 0.7234,
  "is_fraud": true,
  "confidence": 82.5,
  "reasons": [
    "ELA: Suspicious compression artifacts detected (ratio: 0.0821)",
    "Edited with image editing software: Adobe Photoshop",
    "PDF modified after creation"
  ],
  "details": {
    "module_scores": {
      "ela_statistical": 0.821,
      "cnn_prediction": 0.75,
      "metadata": 0.6,
      "text_anomaly": 0.0,
      "copy_move": 0.0
    }
  }
}
```

### Batch Analysis

```bash
curl -X POST http://localhost:8000/analyze/batch \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.jpg"
```

## Training Your Own Model

1. Organize your dataset:
```
dataset/
├── genuine/       (authentic documents)
└── tampered/      (forged documents)
```

2. Run training:
```bash
python train_model.py --data_dir ./dataset --epochs 30 --batch_size 32
```

3. Use the trained model:
```bash
# Set environment variable before starting server
set FRAUD_MODEL_PATH=./fraud_model/fraud_cnn_best.pth
python app.py
```

### Recommended Datasets

- [CASIA v2.0](https://github.com/namtpham/casia2groundtruth) - Image tampering dataset
- [Columbia Uncompressed](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) - Splicing detection
- [CoMoFoD](https://www.vcl.fer.hr/comofod/) - Copy-move forgery detection

## Docker Deployment

```bash
# Build
docker build -t document-fraud-ai .

# Run
docker run -p 8000:8000 document-fraud-ai

# With GPU
docker run --gpus all -p 8000:8000 document-fraud-ai

# With trained model
docker run -p 8000:8000 -e FRAUD_MODEL_PATH=/app/fraud_model/fraud_cnn_best.pth document-fraud-ai
```

## Model Explanation

### Error Level Analysis (ELA)

When a JPEG image is resaved, uniform regions compress similarly. Tampered regions that were pasted from a different source show different compression artifacts. ELA amplifies these differences making them detectable.

### CNN Architecture

```
Input (3 x 128 x 128)
  -> Conv2D(32) -> BN -> ReLU -> Conv2D(32) -> BN -> ReLU -> MaxPool -> Dropout
  -> Conv2D(64) -> BN -> ReLU -> Conv2D(64) -> BN -> ReLU -> MaxPool -> Dropout
  -> Conv2D(128) -> BN -> ReLU -> Conv2D(128) -> BN -> ReLU -> MaxPool -> Dropout
  -> Flatten -> FC(256) -> ReLU -> Dropout -> FC(1) -> Sigmoid
Output: Fraud probability [0, 1]
```

### Heuristic Mode

Without a pretrained model, the system still provides value through:
- ELA statistical analysis (suspicious region ratio)
- Metadata inconsistency detection
- OCR text anomaly detection
- Copy-move forgery detection via ORB features

## Security Considerations

- Uploaded files are deleted immediately after analysis
- File size limited to 50 MB
- File type validation on extension and content
- No user data is stored persistently
- API does not execute any content within uploaded documents
- For production: add authentication, rate limiting, and HTTPS

## Future Improvements

- Fine-tune EfficientNet backbone on large document dataset
- Add signature verification module
- Add font consistency analysis
- GAN-based deepfake document detection
- Multi-page PDF analysis (currently analyzes first page)
- Integration with external fraud databases
- Add confidence calibration with Platt scaling
- Support for handwritten document analysis

## Project Structure

```
document_fraud_ai/
├── app.py                    # FastAPI backend
├── streamlit_app.py          # Streamlit frontend
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container deployment
├── fraud_model/
│   ├── __init__.py
│   ├── cnn_model.py          # CNN architecture (FraudCNN + EfficientNet)
│   └── pipeline.py           # Orchestration pipeline
├── utils/
│   ├── __init__.py
│   ├── ela_analysis.py       # ELA + copy-move detection
│   ├── metadata_analyzer.py  # PDF/image metadata analysis
│   └── ocr_extractor.py      # OCR text extraction
├── uploads/                  # Temporary upload directory
└── logs/                     # Application logs
```
