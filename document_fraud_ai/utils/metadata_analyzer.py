"""
Document metadata analysis module.

Extracts and analyzes metadata from PDFs and images to detect
inconsistencies that may indicate tampering:
- Creation/modification date mismatches
- Software tool anomalies
- Producer/creator inconsistencies
- Suspicious metadata patterns
"""

import os
import re
from datetime import datetime
from typing import Optional
from PIL import Image
from PIL.ExifTags import TAGS


def _parse_pdf_date(date_str: str) -> Optional[datetime]:
    """Parse PDF date format D:YYYYMMDDHHmmSS to datetime."""
    if not date_str:
        return None
    s = date_str.strip()
    if s.startswith("D:"):
        s = s[2:]
    # Keep only digits (strips timezone like +05'30')
    s = re.sub(r"[^0-9]", "", s)
    # Try formats from most to least specific; each has a fixed digit count
    for fmt, length in [("%Y%m%d%H%M%S", 14), ("%Y%m%d%H%M", 12), ("%Y%m%d", 8)]:
        if len(s) >= length:
            try:
                return datetime.strptime(s[:length], fmt)
            except ValueError:
                continue
    return None


def analyze_image_metadata(image_path: str) -> dict:
    """
    Extract and analyze EXIF metadata from an image file.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary with metadata fields and anomaly flags.
    """
    result = {
        "metadata_found": False,
        "anomalies": [],
        "raw_metadata": {},
        "risk_score": 0.0,
    }

    try:
        img = Image.open(image_path)
        exif_data = img.getexif()  # Public API (Pillow >= 6.0); returns empty Exif, never None
    except Exception:
        result["anomalies"].append("Unable to read image metadata")
        result["risk_score"] = 0.3
        return result

    if not exif_data:
        # Missing EXIF is normal for scanned docs, screenshots, and most PDFs.
        # Do NOT flag as suspicious — risk_score stays 0.0.
        return result

    result["metadata_found"] = True
    decoded = {}
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, str(tag_id))
        try:
            decoded[tag_name] = str(value)
        except Exception:
            decoded[tag_name] = "<unreadable>"

    result["raw_metadata"] = decoded

    # Check for software editing tools
    # Removed: canva, fotor, pixlr (legitimate design tools)
    software = decoded.get("Software", "")
    suspicious_tools = ["photoshop", "gimp", "paint.net"]
    for tool in suspicious_tools:
        if tool.lower() in software.lower():
            result["anomalies"].append(f"Edited with image editing software: {software}")
            result["risk_score"] += 0.3
            break

    # Check date consistency
    date_original = decoded.get("DateTimeOriginal", "")
    date_digitized = decoded.get("DateTimeDigitized", "")
    date_modified = decoded.get("DateTime", "")

    if date_original and date_modified:
        if date_original != date_modified:
            result["anomalies"].append(
                f"Date mismatch: original={date_original}, modified={date_modified}"
            )
            result["risk_score"] += 0.15  # Reduced from 0.25

    if date_original and date_digitized:
        if date_original != date_digitized:
            result["anomalies"].append(
                f"Digitized date differs from original: {date_digitized} vs {date_original}"
            )
            result["risk_score"] += 0.10

    # Missing camera fields: only flag as suspicious if SOME camera metadata exists
    # (indicates selective stripping). If NO camera EXIF at all, that's normal.
    camera_fields = ["Make", "Model"]
    has_any_camera_exif = any(f in decoded for f in camera_fields)
    expected_fields = ["Make", "Model", "DateTimeOriginal"]
    missing = [f for f in expected_fields if f not in decoded]

    if has_any_camera_exif and len(missing) >= 2:
        # Inconsistency: partial camera EXIF suggests selective metadata stripping
        result["anomalies"].append(f"Inconsistent EXIF: partial camera metadata, missing {', '.join(missing)}")
        result["risk_score"] += 0.15

    result["risk_score"] = min(result["risk_score"], 1.0)
    return result


def analyze_pdf_metadata(pdf_path: str) -> dict:
    """
    Extract and analyze metadata from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary with metadata fields and anomaly flags.
    """
    import fitz  # PyMuPDF

    result = {
        "metadata_found": False,
        "anomalies": [],
        "raw_metadata": {},
        "risk_score": 0.0,
        "page_count": 0,
    }

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        result["anomalies"].append(f"Cannot open PDF: {str(e)}")
        result["risk_score"] = 0.5
        return result

    result["page_count"] = len(doc)
    metadata = doc.metadata
    if metadata:
        result["metadata_found"] = True
        result["raw_metadata"] = {k: v for k, v in metadata.items() if v}

    producer = metadata.get("producer", "").lower() if metadata else ""
    creator = metadata.get("creator", "").lower() if metadata else ""
    combined = producer + " " + creator

    # High-severity tools: dedicated PDF manipulation utilities
    high_severity_tools = ["pdftk", "pdfedit", "pdf-xchange", "nitro", "sejda"]
    # Medium-severity tools: programmatic PDF generation libraries
    # itextsharp must be checked before itext (it's a longer, more specific name)
    medium_severity_tools = ["itextsharp", "itext", "reportlab", "fpdf", "tcpdf"]
    # NOT flagged: libreoffice, openoffice — used by millions of legitimate institutions

    def _tool_found(tool: str, text: str) -> bool:
        """Match tool name using word boundaries to avoid false substring matches."""
        return bool(re.search(r"(?<!\w)" + re.escape(tool) + r"(?!\w)", text, re.IGNORECASE))

    tool_label = metadata.get("producer", "") or metadata.get("creator", "")
    for tool in high_severity_tools:
        if _tool_found(tool, combined):
            result["anomalies"].append(f"PDF modified with high-risk tool: {tool_label}")
            result["risk_score"] += 0.35
            break
    else:
        for tool in medium_severity_tools:
            if _tool_found(tool, combined):
                result["anomalies"].append(f"PDF generated with programmatic library: {tool_label}")
                result["risk_score"] += 0.15
                break

    # Date mismatch with tiered scoring
    creation_date_str = metadata.get("creationDate", "") if metadata else ""
    mod_date_str = metadata.get("modDate", "") if metadata else ""

    if creation_date_str and mod_date_str and creation_date_str != mod_date_str:
        creation_dt = _parse_pdf_date(creation_date_str)
        mod_dt = _parse_pdf_date(mod_date_str)

        if creation_dt and mod_dt:
            delta_days = abs((mod_dt - creation_dt).days)
            if delta_days > 365:
                result["anomalies"].append(
                    f"PDF modified >1 year after creation: created={creation_date_str}, modified={mod_date_str}"
                )
                result["risk_score"] += 0.25
            elif delta_days > 30:
                result["anomalies"].append(
                    f"PDF modified >30 days after creation: created={creation_date_str}, modified={mod_date_str}"
                )
                result["risk_score"] += 0.10
            # delta_days <= 30: routine saves, not flagged

    # Large file size per page
    try:
        file_size = os.path.getsize(pdf_path)
        if result["page_count"] > 0:
            avg_size_per_page = file_size / result["page_count"]
            if avg_size_per_page > 5_000_000:
                result["anomalies"].append(
                    f"Unusually large file size per page: {avg_size_per_page / 1_000_000:.1f}MB"
                )
                result["risk_score"] += 0.1
    except Exception:
        pass

    # JavaScript detection (malicious PDF indicator)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if "/JavaScript" in text or "/JS" in text:
            result["anomalies"].append(f"JavaScript detected on page {page_num + 1}")
            result["risk_score"] += 0.3
            break

    doc.close()
    result["risk_score"] = min(result["risk_score"], 1.0)
    return result


def analyze_file_metadata(file_path: str) -> dict:
    """
    Route to the appropriate metadata analyzer based on file type.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return analyze_pdf_metadata(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"):
        return analyze_image_metadata(file_path)
    else:
        return {
            "metadata_found": False,
            "anomalies": [f"Unsupported file type: {ext}"],
            "raw_metadata": {},
            "risk_score": 0.1,
        }
