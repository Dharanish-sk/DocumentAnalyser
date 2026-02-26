"""
Error Level Analysis (ELA) module for detecting image tampering.

ELA works by re-saving an image at a known quality level and comparing
the difference with the original. Tampered regions show different
error levels than untouched regions because they were compressed
at different rates.
"""

import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
from scipy import ndimage
from sklearn.cluster import DBSCAN


def compute_ela(image: Image.Image, quality: int = 90, scale: int = 15) -> Image.Image:
    """
    Compute Error Level Analysis on an image.

    Args:
        image: PIL Image to analyze.
        quality: JPEG re-compression quality (lower = more aggressive).
        scale: Multiplier to amplify differences for visibility.

    Returns:
        ELA difference image as PIL Image.
    """
    original = image.convert("RGB")

    # Re-save at the specified JPEG quality
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).copy()  # .copy() forces eager load before buffer goes out of scope

    # Compute pixel-level difference
    diff = ImageChops.difference(original, resaved)

    # Amplify differences
    extrema = diff.getextrema()
    max_diff = max(ex[1] for ex in extrema)
    if max_diff == 0:
        max_diff = 1

    amplification = 255.0 / max_diff * scale
    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(amplification)

    return ela_image


def ela_to_array(ela_image: Image.Image, target_size: tuple = (300, 300)) -> np.ndarray:
    """
    Convert an ELA image to a normalized numpy array for model input.

    Args:
        ela_image: ELA PIL Image.
        target_size: Resize dimensions (width, height). Default 300x300 for EfficientNet-B3.

    Returns:
        Normalized numpy array of shape (3, H, W).
    """
    resized = ela_image.resize(target_size, Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    # HWC -> CHW for PyTorch
    arr = arr.transpose(2, 0, 1)
    return arr


def _compute_ela_fraud_score(cluster_ratio: float, normalized_mean: float, ch_var: float) -> float:
    """Compute continuous ELA fraud score from spatial clustering metrics."""
    score = (
        (cluster_ratio / 0.05) * 0.50
        + (normalized_mean / 0.5) * 0.30
        + (ch_var / 20.0) * 0.20
    )
    return float(min(score, 1.0))


def compute_ela_statistics(ela_image: Image.Image) -> dict:
    """
    Compute statistical features from the ELA image to detect anomalies.

    Uses adaptive 99th-percentile threshold + spatial clustering to avoid
    the ~5% false positive rate of mean+2*std thresholding.

    Returns:
        Dictionary with ELA metrics, cluster_ratio, ela_score, and has_anomaly flag.
    """
    arr_rgb = np.array(ela_image.convert("RGB"), dtype=np.float32)
    arr = np.array(ela_image.convert("L"), dtype=np.float32)

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    max_val = float(np.max(arr))

    # Per-channel stats for variance signal
    ch_r = arr_rgb[:, :, 0]
    ch_g = arr_rgb[:, :, 1]
    ch_b = arr_rgb[:, :, 2]
    std_r = float(np.std(ch_r))
    std_g = float(np.std(ch_g))
    std_b = float(np.std(ch_b))
    channel_std_variance = float(np.var([std_r, std_g, std_b]))

    # Adaptive threshold: 99th percentile avoids always-flagging Gaussian noise
    threshold = float(np.percentile(arr, 99))
    suspicious_mask = (arr > threshold).astype(np.int32)

    # Spatial clustering: only count connected regions > 0.5% of image area
    labeled, num_features = ndimage.label(suspicious_mask)
    total_pixels = arr.size
    min_component_size = int(total_pixels * 0.005)

    significant_pixels = 0
    for region_id in range(1, num_features + 1):
        region_size = int(np.sum(labeled == region_id))
        if region_size >= min_component_size:
            significant_pixels += region_size

    cluster_ratio = float(significant_pixels / total_pixels) if total_pixels > 0 else 0.0
    normalized_mean = mean_val / 255.0

    ela_score = _compute_ela_fraud_score(cluster_ratio, normalized_mean, channel_std_variance)
    has_anomaly = cluster_ratio > 0.02 or normalized_mean > 0.4

    return {
        "ela_mean": round(mean_val, 4),
        "ela_std": round(std_val, 4),
        "ela_max": round(max_val, 4),
        "suspicious_region_ratio": round(cluster_ratio, 6),
        "channel_std_variance": round(channel_std_variance, 6),
        "ela_score": round(ela_score, 6),
        "has_anomaly": has_anomaly,
    }


def detect_copy_move(image: Image.Image, min_match_count: int = 10) -> dict:
    """
    Detect copy-move forgery using AKAZE feature matching + DBSCAN clustering.

    AKAZE is rotation/scale invariant (unlike ORB). DBSCAN on displacement
    vectors distinguishes genuine repeated patterns (scattered displacements)
    from copy-move (clustered displacements).

    Returns:
        Dictionary with copy_move_detected, copy_move_score, cluster_count,
        and matched_regions.
    """
    img_rgb = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = img_cv.shape
    img_diagonal = float(np.sqrt(h * h + w * w))

    # AKAZE: rotation/scale invariant, floating-point descriptors
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(img_cv, None)

    if descriptors is None or len(keypoints) < min_match_count:
        return {
            "copy_move_detected": False,
            "copy_move_score": 0.0,
            "cluster_count": 0,
            "matched_regions": 0,
        }

    # FLANN for floating-point descriptors
    flann_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(flann_params, search_params)

    try:
        matches = flann.knnMatch(descriptors.astype(np.float32),
                                 descriptors.astype(np.float32), k=2)
    except Exception:
        return {
            "copy_move_detected": False,
            "copy_move_score": 0.0,
            "cluster_count": 0,
            "matched_regions": 0,
        }

    # Minimum spatial separation: 5% of image diagonal
    min_dist = img_diagonal * 0.05

    suspicious_matches = []
    displacement_vectors = []

    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.queryIdx == m.trainIdx:
                continue
            if m.distance < 0.75 * n.distance:
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                dist = float(np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2))
                if dist > min_dist:
                    suspicious_matches.append(m)
                    # Normalize displacement by image size for DBSCAN
                    dx = (pt2[0] - pt1[0]) / w
                    dy = (pt2[1] - pt1[1]) / h
                    displacement_vectors.append([dx, dy])

    if len(suspicious_matches) < min_match_count:
        return {
            "copy_move_detected": False,
            "copy_move_score": 0.0,
            "cluster_count": 0,
            "matched_regions": len(suspicious_matches),
        }

    # DBSCAN: clustered displacement vectors → copy-move; scattered → genuine repetition
    X = np.array(displacement_vectors)
    db = DBSCAN(eps=0.08, min_samples=3).fit(X)
    labels = db.labels_
    cluster_count = int(len(set(labels)) - (1 if -1 in labels else 0))

    if cluster_count > 0:
        clustered = int(np.sum(labels >= 0))
        copy_move_score = float(min(clustered / max(len(suspicious_matches), 1), 1.0))
    else:
        copy_move_score = 0.0

    detected = cluster_count > 0 and copy_move_score > 0.3

    return {
        "copy_move_detected": detected,
        "copy_move_score": round(copy_move_score, 4),
        "cluster_count": cluster_count,
        "matched_regions": len(suspicious_matches),
    }


def compute_blur_sharpness(image: Image.Image) -> dict:
    """
    Detect local sharpness inconsistencies that may indicate spliced regions.

    Divides image into 8x8 grid; computes Laplacian variance per cell.
    Flags cells > 3 MADs from the median as inconsistent.

    Returns:
        Dictionary with blur_score [0,1] and inconsistent_cell_count.
    """
    img_gray = np.array(image.convert("L"), dtype=np.float32)
    h, w = img_gray.shape
    grid_size = 8
    cell_h = max(h // grid_size, 1)
    cell_w = max(w // grid_size, 1)

    lap_vars = []
    for row in range(grid_size):
        for col in range(grid_size):
            r0, r1 = row * cell_h, min((row + 1) * cell_h, h)
            c0, c1 = col * cell_w, min((col + 1) * cell_w, w)
            cell = img_gray[r0:r1, c0:c1]
            if cell.size == 0:
                continue
            lap = cv2.Laplacian(cell, cv2.CV_32F)
            lap_vars.append(float(np.var(lap)))

    if not lap_vars:
        return {"blur_score": 0.0, "inconsistent_cell_count": 0}

    lap_arr = np.array(lap_vars)
    median = float(np.median(lap_arr))
    mad = float(np.median(np.abs(lap_arr - median)))

    if mad == 0:
        return {"blur_score": 0.0, "inconsistent_cell_count": 0}

    inconsistent = int(np.sum(np.abs(lap_arr - median) > 3 * mad))
    blur_score = float(min(inconsistent / 8.0, 1.0))

    return {
        "blur_score": round(blur_score, 4),
        "inconsistent_cell_count": inconsistent,
    }


def compute_jpeg_ghost(image: Image.Image, qualities: list = None) -> dict:
    """
    JPEG ghost detection: finds regions compressed at a different quality than the rest.

    Re-saves at multiple quality levels; per-pixel minimum-quality map spatial
    entropy reveals inconsistently compressed regions (spliced content).

    Returns:
        Dictionary with ghost_score [0,1].
    """
    if qualities is None:
        qualities = list(range(60, 96, 5))  # [60, 65, 70, 75, 80, 85, 90, 95]

    # Resize very large images to cap memory use (~8 MSE maps × H×W×4 bytes)
    max_side = 1024
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    original = np.array(image.convert("RGB"), dtype=np.float32)
    mse_maps = []

    for q in qualities:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        resaved = np.array(Image.open(buf).copy().convert("RGB"), dtype=np.float32)
        mse_map = np.mean((original - resaved) ** 2, axis=2)
        mse_maps.append(mse_map)

    mse_stack = np.stack(mse_maps, axis=0)  # (Q, H, W)
    min_quality_map = np.argmin(mse_stack, axis=0)  # per-pixel best quality index

    # Spatial entropy of the minimum-quality map
    counts = np.bincount(min_quality_map.ravel(), minlength=len(qualities))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(len(qualities)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Non-dominant ratio: fraction of pixels not at the most common quality level
    dominant_count = int(counts.max())
    total_pixels = int(min_quality_map.size)
    non_dominant_ratio = float(1.0 - dominant_count / total_pixels)

    ghost_score = float(normalized_entropy * 0.6 + non_dominant_ratio * 0.4)

    return {
        "ghost_score": round(min(ghost_score, 1.0), 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "non_dominant_ratio": round(non_dominant_ratio, 4),
    }
