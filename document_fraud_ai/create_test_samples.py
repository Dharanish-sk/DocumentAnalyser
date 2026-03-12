"""
Create tampered test images to verify ManTraNet and the full pipeline are working.

Generates 5 types of tampering that commonly appear in marksheet fraud:
  1. Grade digit replaced   (most common — e.g., 75 → 95)
  2. Region copy-pasted     (copy a passing grade onto a failing one)
  3. Text erased + retyped  (white-out a name/number and retype)
  4. Local brightness boost  (wash out ink to hide original text)
  5. Splice from another image (paste a grade box from a different scan)

For each tampered image, the pipeline should return a HIGHER fraud_probability
than the genuine original.

Usage:
    python create_test_samples.py --input ./my_docs/genuine/marksheet.jpg \
                                  --output ./test_samples

Then analyze with the pipeline:
    python create_test_samples.py --input ./my_docs/genuine/marksheet.jpg \
                                  --output ./test_samples --analyze
"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from loguru import logger


# ── Tampering functions ────────────────────────────────────────────────────────

def tamper_digit_replace(image: Image.Image) -> Image.Image:
    """
    Simulate replacing a grade digit (e.g., 75 → 95).

    Picks a random small region in the upper-centre of the image
    (where marks are typically printed), whites it out, and
    draws a new number over it.

    Detection signal: The re-drawn region is a fresh JPEG artifact island
    surrounded by the original compression pattern → ManTraNet flags it.
    """
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Pick a region typical for a grade cell
    x0 = int(w * 0.55)
    y0 = int(h * 0.30)
    x1 = x0 + int(w * 0.07)
    y1 = y0 + int(h * 0.03)

    # White-out the original value
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

    # Draw replacement text (simulate "95" over where "75" was)
    font_size = max(int((y1 - y0) * 0.8), 10)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text((x0 + 2, y0 + 1), "95", fill=(0, 0, 0), font=font)

    return _resave_jpeg(img, quality=85)   # Re-save to embed JPEG artifacts


def tamper_copy_paste(image: Image.Image) -> Image.Image:
    """
    Copy a region from one part of the image and paste it over another.

    Simulates copying a 'Pass' grade stamp onto a different row.

    Detection signal: Copy-move creates two identical noise patches
    at different spatial positions → ManTraNet's SRM filters detect the mismatch.
    """
    img = image.copy().convert("RGB")
    w, h = img.size

    # Source region: top-right area (a good grade)
    sx0 = int(w * 0.60); sy0 = int(h * 0.25)
    sx1 = int(w * 0.75); sy1 = int(h * 0.30)

    # Destination: lower area (a failing row)
    dx0 = int(w * 0.60); dy0 = int(h * 0.55)

    src_crop = img.crop((sx0, sy0, sx1, sy1))
    img.paste(src_crop, (dx0, dy0))

    return _resave_jpeg(img, quality=85)


def tamper_erase_retype(image: Image.Image) -> Image.Image:
    """
    White-out a name/roll-number region and type new text.

    Simulates changing a student's name or roll number on the marksheet.

    Detection signal: The white-painted block has zero texture (flat region).
    The new text layer has its own JPEG artifact pattern that doesn't match
    the original scan's noise → ELA and ManTraNet both flag it.
    """
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Target: top area where name/roll number usually appears
    x0 = int(w * 0.20); y0 = int(h * 0.12)
    x1 = int(w * 0.60); y1 = y0 + int(h * 0.035)

    # Erase with white box
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

    # Retype replacement name
    font_size = max(int((y1 - y0) * 0.75), 10)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text((x0 + 4, y0 + 2), "RAJESH KUMAR", fill=(10, 10, 10), font=font)

    return _resave_jpeg(img, quality=88)


def tamper_local_brightness(image: Image.Image) -> Image.Image:
    """
    Boost brightness in a small region to wash out printed text.

    Simulates using photo-editing to overexpose a specific mark field
    to make the original value unreadable.

    Detection signal: Local histogram of the altered region is shifted
    compared to surrounding areas → ELA and blur/sharpness modules detect it.
    """
    img = image.copy().convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Region to brighten (a grade column)
    r0 = int(h * 0.40); r1 = int(h * 0.50)
    c0 = int(w * 0.65); c1 = int(w * 0.78)

    # Boost brightness by 80 (out of 255)
    arr[r0:r1, c0:c1] = np.clip(arr[r0:r1, c0:c1] + 80, 0, 255)

    img_out = Image.fromarray(arr.astype(np.uint8))
    return _resave_jpeg(img_out, quality=82)


def tamper_splice_block(image: Image.Image) -> Image.Image:
    """
    Paste a random block from a second JPEG save of the image.

    Simulates inserting a grade stamp or seal from a different document.
    Re-saving at a different quality creates a 'foreign' noise layer that
    ManTraNet's SRM filters detect as inconsistent with the background.

    Detection signal: The spliced block has a different compression history
    (different JPEG quantization table) → strong ManTraNet anomaly.
    """
    img = image.copy().convert("RGB")
    w, h = img.size

    # Create the 'foreign' source: re-save the image at a very different quality
    foreign = _resave_jpeg(image, quality=60)   # heavily compressed source

    # Crop a block from the foreign source
    fx0 = int(w * 0.50); fy0 = int(h * 0.45)
    fx1 = int(w * 0.68); fy1 = int(h * 0.55)
    foreign_block = foreign.crop((fx0, fy0, fx1, fy1))

    # Paste into the original at a different location
    px = int(w * 0.20); py = int(h * 0.45)
    img.paste(foreign_block, (px, py))

    return _resave_jpeg(img, quality=90)


# ── Utility ────────────────────────────────────────────────────────────────────

def _resave_jpeg(image: Image.Image, quality: int = 85) -> Image.Image:
    """
    Re-save image through a JPEG buffer.

    This is critical — PIL edits are lossless (no JPEG artifacts).
    Re-saving introduces the JPEG compression artifacts that ELA and
    ManTraNet use as their primary detection signal. Without this step
    the tampered regions would be indistinguishable from the original.
    """
    import io
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Main ───────────────────────────────────────────────────────────────────────

TAMPER_TYPES = {
    "digit_replace":    tamper_digit_replace,
    "copy_paste":       tamper_copy_paste,
    "erase_retype":     tamper_erase_retype,
    "local_brightness": tamper_local_brightness,
    "splice_block":     tamper_splice_block,
}


def create_samples(input_path: str, output_dir: str) -> list:
    """Generate all tampered variants and save them."""
    os.makedirs(output_dir, exist_ok=True)

    genuine_img = Image.open(input_path).convert("RGB")
    saved_paths = []

    # Save the genuine original
    genuine_out = os.path.join(output_dir, "genuine_original.jpg")
    _resave_jpeg(genuine_img, quality=95).save(genuine_out, format="JPEG", quality=95)
    logger.info(f"Saved genuine:  {genuine_out}")
    saved_paths.append(("genuine", genuine_out))

    # Save each tampered variant
    for name, fn in TAMPER_TYPES.items():
        try:
            tampered = fn(genuine_img)
            out_path = os.path.join(output_dir, f"tampered_{name}.jpg")
            tampered.save(out_path, format="JPEG", quality=90)
            logger.info(f"Saved tampered: {out_path}  [{name}]")
            saved_paths.append((name, out_path))
        except Exception as e:
            logger.error(f"Failed to create {name}: {e}")

    return saved_paths


def analyze_samples(saved_paths: list):
    """Run the full pipeline on all samples and print a comparison table."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from fraud_model.pipeline import FraudDetectionPipeline

    logger.info("Loading pipeline ...")
    pipeline = FraudDetectionPipeline()   # no trained CNN needed for this test

    print("\n" + "=" * 72)
    print(f"{'Sample':<28} {'Fraud Prob':>10} {'ManTraNet':>10} {'ELA':>8} {'Verdict'}")
    print("=" * 72)

    for label, path in saved_paths:
        result  = pipeline.analyze(path)
        scores  = result["details"]["module_scores"]
        prob    = result["fraud_probability"]
        is_fraud = result["is_fraud"]
        mt_score = scores.get("mantranet")
        ela_score = scores.get("ela_statistical")

        mt_str  = f"{mt_score:.4f}"  if mt_score  is not None else "N/A"
        ela_str = f"{ela_score:.4f}" if ela_score is not None else "N/A"
        verdict = "FRAUD ✗" if is_fraud else "genuine ✓"

        print(
            f"{label:<28} {prob:>10.4f} {mt_str:>10} {ela_str:>8}  {verdict}"
        )

    print("=" * 72)
    print("\nExpected: genuine_original → low score,  all tampered_* → higher scores")
    print("If ManTraNet shows N/A, run setup_mantranet() first (see utils/mantranet.py)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tampered marksheet test images and optionally analyze them."
    )
    parser.add_argument(
        "--input",  required=True,
        help="Path to a genuine marksheet image (JPG/PNG/PDF)"
    )
    parser.add_argument(
        "--output", default="./test_samples",
        help="Directory to save tampered images (default: ./test_samples)"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run the full fraud detection pipeline on all generated samples"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        raise SystemExit(1)

    paths = create_samples(args.input, args.output)

    if args.analyze:
        analyze_samples(paths)
    else:
        print(f"\nSamples saved to: {args.output}/")
        print("To analyze them run:")
        print(f"  python create_test_samples.py --input {args.input} --output {args.output} --analyze")
