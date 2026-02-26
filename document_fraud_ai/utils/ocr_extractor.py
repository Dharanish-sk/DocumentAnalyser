"""
OCR text extraction module.

Extracts text from documents (PDF and images) for content analysis.
Uses PyMuPDF for PDFs (fast, no external dependencies) and
Tesseract OCR for scanned images.
"""

import os
import re
from typing import Optional
from PIL import Image


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary with extracted text and page-level breakdown.
    """
    import fitz

    result = {
        "total_text": "",
        "pages": [],
        "page_count": 0,
        "has_text": False,
        "is_scanned": False,
    }

    try:
        doc = fitz.open(pdf_path)
        result["page_count"] = len(doc)

        full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            result["pages"].append({
                "page": page_num + 1,
                "text": text[:500],  # Limit per page for API response
                "char_count": len(text),
            })
            full_text.append(text)

        result["total_text"] = "\n".join(full_text)
        result["has_text"] = len(result["total_text"].strip()) > 0

        # If very little text found, likely a scanned document
        if result["page_count"] > 0:
            avg_chars = len(result["total_text"]) / result["page_count"]
            result["is_scanned"] = avg_chars < 50

        doc.close()
    except Exception as e:
        result["error"] = str(e)

    return result


def extract_text_from_image(image_path: str, lang: str = "eng") -> dict:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image_path: Path to the image file.
        lang: Tesseract language code.

    Returns:
        Dictionary with extracted text and confidence info.
    """
    result = {
        "text": "",
        "has_text": False,
        "ocr_available": False,
    }

    try:
        import pytesseract
        result["ocr_available"] = True

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        result["text"] = text.strip()
        result["has_text"] = len(result["text"]) > 0

        # Get confidence data
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data["conf"] if int(c) > 0]
        if confidences:
            result["avg_confidence"] = round(sum(confidences) / len(confidences), 2)
            result["min_confidence"] = min(confidences)
        else:
            result["avg_confidence"] = 0
            result["min_confidence"] = 0

    except ImportError:
        result["error"] = "pytesseract not installed"
        # ocr_available stays False (set above)
    except Exception as e:
        result["error"] = f"OCR failed: {str(e)}"
        result["ocr_available"] = False  # Was True — logic was inverted

    return result


def analyze_text_anomalies(text: str) -> dict:
    """
    Analyze extracted text for anomalies that may indicate fraud.

    Checks for:
    - Mixed fonts/encoding artifacts
    - Suspicious patterns (placeholder text, test data)
    - Inconsistent formatting
    """
    anomalies = []
    risk_score = 0.0

    if not text or len(text.strip()) < 10:
        return {"anomalies": [], "risk_score": 0.0}

    # Check for placeholder/dummy text
    dummy_patterns = [
        r"\bLorem\s+ipsum\b",
        r"\btest\s+document\b",
        r"\bsample\s+text\b",
        r"\bxxx+\b",
        r"\b(John|Jane)\s+Doe\b",
        r"\b123\s*Main\s*(St|Street)\b",
    ]
    for pattern in dummy_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            anomalies.append(f"Placeholder/dummy text detected: {pattern}")
            risk_score += 0.3

    # Check for encoding artifacts (mojibake)
    artifact_chars = sum(1 for c in text if ord(c) > 65533 or (128 <= ord(c) <= 159))
    if artifact_chars > len(text) * 0.05:
        anomalies.append("Encoding artifacts detected - possible text manipulation")
        risk_score += 0.2

    # Check for inconsistent number formatting
    numbers = re.findall(r"\$[\d,]+\.?\d*", text)
    if numbers:
        has_comma = any("," in n for n in numbers)
        has_no_comma = any("," not in n and len(re.sub(r"[^\d]", "", n)) > 3 for n in numbers)
        if has_comma and has_no_comma:
            anomalies.append("Inconsistent number formatting detected")
            risk_score += 0.15

    return {
        "anomalies": anomalies,
        "risk_score": min(risk_score, 1.0),
    }


def extract_text(file_path: str) -> dict:
    """
    Route to the appropriate text extractor based on file type.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"):
        return extract_text_from_image(file_path)
    else:
        return {"text": "", "has_text": False, "error": f"Unsupported: {ext}"}
