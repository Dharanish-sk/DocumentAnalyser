"""
ManTraNet: Manipulation Tracing Network for pixel-level image forgery detection.

Wraps the ManTraNet-pytorch reimplementation by RonyAbecidan.
ManTraNet is pre-trained on 385 manipulation types (splicing, copy-move, removal,
enhancement) and produces a per-pixel anomaly map without any fine-tuning.

Efficiency improvements over the baseline wrapper:
  - torch.inference_mode()  — faster than no_grad, no grad-tracking overhead
  - torch.autocast (FP16)   — halves VRAM on CUDA, ~30% faster
  - torch.compile            — JIT kernel fusion, ~20-30% faster
  - Tiled inference          — full-resolution detail, constant memory cost
  - Spatial cluster scoring  — distinguishes real tampered regions from noise

Reference:
    Y. Wu, W. Abd-Almageed, P. Natarajan
    "ManTraNet: Manipulation Tracing Network for Detection and Localization
    of Image Forgeries with Anomalous Features", CVPR 2019.

PyTorch port:
    https://github.com/RonyAbecidan/ManTraNet-pytorch

──────────────────────────────────────────────────────────────────────────────
SETUP (run once on Google Colab before using the pipeline):

    from utils.mantranet import setup_mantranet
    setup_mantranet(weights_gdrive_id="YOUR_GDRIVE_FILE_ID")

To get the Google Drive file ID:
    1. Go to https://github.com/RonyAbecidan/ManTraNet-pytorch
    2. Follow the instructions there to download ManTraNet.pt
    3. Upload it to your Colab session OR provide its Google Drive ID
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import subprocess

import numpy as np
import torch
from PIL import Image
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)

MANTRANET_SRC_DIR = os.path.join(_PROJECT_DIR, "mantranet_src")
MANTRANET_WEIGHTS = os.path.join(MANTRANET_SRC_DIR, "ManTraNet.pt")

_REPO_URL = "https://github.com/RonyAbecidan/ManTraNet-pytorch.git"

# ── Tiling config ──────────────────────────────────────────────────────────────
# Tile size to use for full-resolution inference on large documents.
# 512×512 fits comfortably in CPU RAM and a Colab T4 GPU.
_TILE_SIZE = 512
# Overlap on each side — prevents hard seams at tile edges.
_OVERLAP   = 64


# ── Setup ──────────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if ManTraNet source and weights are both present."""
    return (
        os.path.isdir(MANTRANET_SRC_DIR)
        and os.path.exists(os.path.join(MANTRANET_SRC_DIR, "ManTraNet.py"))
        and os.path.exists(MANTRANET_WEIGHTS)
    )


def setup_mantranet(weights_gdrive_id: str = None, weights_path: str = None):
    """
    One-time setup: clone ManTraNet-pytorch and download pretrained weights.

    Call this once on Google Colab before starting the pipeline.

    Args:
        weights_gdrive_id:
            Google Drive file ID for ManTraNet.pt.
            Find it at https://github.com/RonyAbecidan/ManTraNet-pytorch
            Example (Colab cell):
                from utils.mantranet import setup_mantranet
                setup_mantranet(weights_gdrive_id="1A9xBcD2efG...")

        weights_path:
            Local path to a ManTraNet.pt file you already have.
            If provided, the file is copied to the expected location.
    """
    # ── 1. Clone source repo ───────────────────────────────────────────────────
    model_py = os.path.join(MANTRANET_SRC_DIR, "ManTraNet.py")
    if not os.path.exists(model_py):
        logger.info(f"Cloning ManTraNet-pytorch → {MANTRANET_SRC_DIR} ...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", _REPO_URL, MANTRANET_SRC_DIR],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed:\n{result.stderr}\n"
                "Make sure git is installed and you have internet access."
            )
        logger.info("ManTraNet-pytorch cloned successfully")
    else:
        logger.info("ManTraNet source already present")

    # ── 2. Get weights ─────────────────────────────────────────────────────────
    if not os.path.exists(MANTRANET_WEIGHTS):
        if weights_path and os.path.exists(weights_path):
            import shutil
            shutil.copy(weights_path, MANTRANET_WEIGHTS)
            logger.info(f"Weights copied from {weights_path}")

        elif weights_gdrive_id:
            _download_from_gdrive(weights_gdrive_id, MANTRANET_WEIGHTS)

        else:
            logger.warning(
                "ManTraNet weights not found.\n"
                "Option A — provide a Google Drive file ID:\n"
                "    setup_mantranet(weights_gdrive_id='YOUR_FILE_ID')\n"
                "Option B — provide a local weights file:\n"
                "    setup_mantranet(weights_path='/path/to/ManTraNet.pt')\n"
                "Get the weights from: https://github.com/RonyAbecidan/ManTraNet-pytorch"
            )
            return
    else:
        size_mb = os.path.getsize(MANTRANET_WEIGHTS) / 1_048_576
        logger.info(f"Weights already present ({size_mb:.1f} MB)")

    # ── 3. Register path once ──────────────────────────────────────────────────
    _add_to_sys_path()
    logger.info("ManTraNet setup complete. Call load_mantranet() to load the model.")


def _download_from_gdrive(file_id: str, dest: str):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"],
            check=True,
        )
        import gdown

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading ManTraNet weights (Google Drive: {file_id}) ...")
    gdown.download(url, dest, quiet=False)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1_048_576
        logger.info(f"Weights downloaded ({size_mb:.1f} MB) → {dest}")
    else:
        raise RuntimeError(
            f"Download appeared to complete but {dest} was not created.\n"
            "Check that the file_id is correct and the file is publicly shared."
        )


def _add_to_sys_path():
    """Add ManTraNet source directory to sys.path (idempotent)."""
    if MANTRANET_SRC_DIR not in sys.path:
        sys.path.insert(0, MANTRANET_SRC_DIR)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_mantranet(device: str = "cpu"):
    """
    Load the ManTraNet model with pretrained weights.

    Applies torch.compile (PyTorch 2.x) after loading for JIT kernel fusion.

    Args:
        device: 'cuda' or 'cpu'.

    Returns:
        ManTraNet model in eval mode, or None if not set up.
    """
    _add_to_sys_path()

    try:
        from ManTraNet import ManTraNet  # type: ignore  (from cloned repo)
    except ImportError:
        logger.warning(
            "ManTraNet class not importable. Run setup_mantranet() first.\n"
            "ManTraNet module will be skipped in the pipeline."
        )
        return None

    model = ManTraNet()

    if os.path.exists(MANTRANET_WEIGHTS):
        try:
            try:
                state = torch.load(
                    MANTRANET_WEIGHTS, map_location=device, weights_only=True
                )
            except TypeError:
                state = torch.load(MANTRANET_WEIGHTS, map_location=device)

            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"ManTraNet: {len(missing)} missing keys (minor mismatch)")
            if unexpected:
                logger.warning(f"ManTraNet: {len(unexpected)} unexpected keys (minor mismatch)")
            logger.info(f"ManTraNet weights loaded from {MANTRANET_WEIGHTS}")

        except Exception as e:
            logger.warning(
                f"ManTraNet weight loading failed: {e}\n"
                "Model will run but predictions will be unreliable (random weights)."
            )
    else:
        logger.warning(
            f"No weights at {MANTRANET_WEIGHTS}. Run setup_mantranet() to download them."
        )

    model.to(device)
    model.eval()

    # JIT compile for kernel fusion (PyTorch 2.x; graceful no-op on older)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("ManTraNet compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            logger.warning(f"torch.compile skipped for ManTraNet: {e}")

    return model


# ── Inference helpers ──────────────────────────────────────────────────────────

def _make_blend_weights(h: int, w: int) -> torch.Tensor:
    """
    Smooth center-weighted tile blend map.

    Weights peak at the tile center (1.0) and taper toward edges (min 0.1).
    When tiles are averaged using these weights, seams between tiles disappear
    because edge pixels from each tile contribute less than center pixels.
    """
    y = torch.linspace(0.0, 1.0, h)
    x = torch.linspace(0.0, 1.0, w)
    # Tent function: 0 at boundaries, 1 at midpoint
    wy = 1.0 - (2.0 * y - 1.0).abs()   # shape (h,)
    wx = 1.0 - (2.0 * x - 1.0).abs()   # shape (w,)
    weight = wy.unsqueeze(1) * wx.unsqueeze(0)   # shape (h, w)
    return weight.clamp(min=0.1)


def _infer_tile(
    model,
    tile: torch.Tensor,
    device: str,
    use_fp16: bool,
) -> torch.Tensor:
    """
    Run ManTraNet on a single tile tensor (1, 3, H, W).

    Uses torch.inference_mode() instead of no_grad():
      - Disables gradient computation AND version tracking
      - ~10-15% faster, lower peak memory than no_grad()

    Uses torch.autocast on CUDA:
      - Runs matmuls/convolutions in FP16
      - ~30-40% less VRAM, ~30% faster throughput

    Returns anomaly map as (H, W) float32 CPU tensor.
    """
    with torch.inference_mode():
        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            output = model(tile)

    # Normalize output to (H, W) regardless of model's output shape convention
    if output.dim() == 4:
        amap = output.squeeze(0).squeeze(0)   # (1,1,H,W) → (H,W)
    elif output.dim() == 3:
        amap = output.squeeze(0)              # (1,H,W)   → (H,W)
    else:
        amap = output                          # already (H,W)

    return amap.float().cpu()


def _tiled_inference(
    model,
    tensor: torch.Tensor,
    device: str,
    use_fp16: bool,
    tile_size: int = _TILE_SIZE,
    overlap: int = _OVERLAP,
) -> np.ndarray:
    """
    Process a large image at FULL RESOLUTION by running ManTraNet on
    overlapping tiles and blending the resulting anomaly maps.

    Why tiling instead of resizing:
      - Resizing to 1024px loses fine detail (digit edits, character changes)
        that are the most common form of marksheet fraud.
      - Tiling preserves every pixel while keeping memory constant
        (one tile at a time, not the full image).
      - Center-weighted blending eliminates seam artefacts at tile boundaries.

    Args:
        tensor:    (1, 3, H, W) float32 image tensor in [0, 1], on CPU.
        tile_size: Tile height/width in pixels (512 by default).
        overlap:   Overlap on each side of each tile (64px by default).

    Returns:
        anomaly_np: np.ndarray shape (H, W), float32, values in [0, 1].
    """
    _, _, H, W = tensor.shape
    step = tile_size - 2 * overlap

    # Weighted accumulator maps (kept on CPU to avoid CUDA OOM)
    acc    = torch.zeros(H, W, dtype=torch.float32)
    weight = torch.zeros(H, W, dtype=torch.float32)

    def _starts(dim_size: int) -> list:
        """Generate tile start positions covering the full dimension."""
        positions = list(range(0, max(dim_size - tile_size, 0) + 1, step))
        last = max(dim_size - tile_size, 0)
        if not positions or positions[-1] < last:
            positions.append(last)
        return sorted(set(positions))

    for y0 in _starts(H):
        y1 = min(y0 + tile_size, H)
        for x0 in _starts(W):
            x1 = min(x0 + tile_size, W)

            tile = tensor[:, :, y0:y1, x0:x1].to(device)
            tile_map = _infer_tile(model, tile, device, use_fp16)   # (th, tw)

            th, tw    = tile_map.shape
            w_tile    = _make_blend_weights(th, tw)

            acc[y0:y1, x0:x1]    += tile_map * w_tile
            weight[y0:y1, x0:x1] += w_tile

    anomaly = (acc / (weight + 1e-8)).clamp(0.0, 1.0)
    return anomaly.numpy()


def _spatial_cluster_score(anomaly_np: np.ndarray) -> tuple:
    """
    Derive a fraud score from the anomaly map using spatial cluster analysis.

    Why this beats a simple top-N% mean:
      Scattered noise pixels produce many small, low-intensity anomalies.
      Real tampering (e.g., a changed digit or spliced grade) produces a
      CONCENTRATED region of HIGH anomaly values.

      Simple top-5% mean treats both cases the same.
      Cluster scoring penalizes scattered noise and rewards concentrated
      anomalies — exactly the pattern for marksheet fraud.

    Algorithm:
      1. Threshold at 99th percentile → binary mask of highest-anomaly pixels
      2. Label connected components → find spatial clusters
      3. Discard tiny clusters (<0.1% of image) — they are sensor noise
      4. Score each surviving cluster: anomaly_strength × spatial_weight
         (spatial_weight saturates at 5% of image — small or large edits
          both count equally as long as they are genuinely anomalous)
      5. Return max cluster score and fraction of image in significant clusters

    Returns:
        (score: float [0, 1], suspicious_ratio: float [0, 1])
    """
    from scipy import ndimage

    total_pixels = anomaly_np.size

    # Step 1: threshold at 99th percentile
    threshold    = float(np.percentile(anomaly_np, 99))
    binary_mask  = (anomaly_np >= threshold).astype(np.int32)

    # Step 2: connected components
    labeled, num_components = ndimage.label(binary_mask)

    # Step 3: filter tiny clusters (noise)
    min_region_px = max(int(total_pixels * 0.001), 5)   # ≥ 0.1% or 5px

    significant_pixels = 0
    cluster_scores     = []

    for region_id in range(1, num_components + 1):
        region_mask = labeled == region_id
        region_size = int(np.sum(region_mask))
        if region_size < min_region_px:
            continue

        region_mean  = float(np.mean(anomaly_np[region_mask]))
        area_ratio   = region_size / total_pixels

        # spatial_weight: saturates at 5% of image (small edit ≈ large edit)
        spatial_weight = min(area_ratio / 0.05, 1.0)

        # Combined: anomaly strength is primary, spatial extent is secondary
        cluster_scores.append(region_mean * (0.6 + 0.4 * spatial_weight))
        significant_pixels += region_size

    suspicious_ratio = significant_pixels / total_pixels

    if cluster_scores:
        score = float(np.clip(max(cluster_scores), 0.0, 1.0))
    else:
        # Fallback: top-1% mean (more conservative than top-5%)
        p99   = float(np.percentile(anomaly_np, 99))
        top   = anomaly_np[anomaly_np >= p99]
        score = float(np.mean(top)) if len(top) > 0 else 0.0
        score = float(np.clip(score, 0.0, 1.0))

    return score, float(np.clip(suspicious_ratio, 0.0, 1.0))


# ── Public inference API ───────────────────────────────────────────────────────

def run_mantranet(model, image: Image.Image, device: str = "cpu") -> dict:
    """
    Run ManTraNet on a PIL image and return a forgery score.

    Inference strategy:
      - Small images (≤ 512px on longest side): single-pass, fastest path.
      - Large images: overlapping tiled inference at FULL RESOLUTION.
        No resizing — fine detail (individual digits, characters) is preserved.

    Args:
        model:  Loaded ManTraNet model (from load_mantranet()).
        image:  PIL image (any mode, any size).
        device: 'cuda' or 'cpu'.

    Returns:
        dict:
            mantranet_score (float [0, 1]):   overall tampering probability.
            suspicious_region_ratio (float):  fraction of image in significant clusters.
    """
    _NONE_RESULT = {"mantranet_score": None, "suspicious_region_ratio": 0.0}

    if model is None:
        return _NONE_RESULT

    try:
        img      = image.convert("RGB")
        use_fp16 = (device == "cuda")

        # Build (1, 3, H, W) float32 tensor on CPU — tiles are moved to device
        arr    = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        _, _, H, W = tensor.shape

        if H <= _TILE_SIZE and W <= _TILE_SIZE:
            # ── Fast path: image fits in one tile ─────────────────────────────
            tile_map   = _infer_tile(model, tensor.to(device), device, use_fp16)
            anomaly_np = np.clip(tile_map.numpy(), 0.0, 1.0)
        else:
            # ── Tiled path: full-resolution inference ─────────────────────────
            logger.debug(
                f"ManTraNet tiled inference: {W}×{H}px image, "
                f"tile={_TILE_SIZE}px, overlap={_OVERLAP}px"
            )
            anomaly_np = _tiled_inference(model, tensor, device, use_fp16)

        # Spatial cluster scoring — separates real tampering from noise
        mantranet_score, suspicious_ratio = _spatial_cluster_score(anomaly_np)

        return {
            "mantranet_score":        round(mantranet_score,  4),
            "suspicious_region_ratio": round(suspicious_ratio, 4),
        }

    except Exception as e:
        logger.error(f"ManTraNet inference error: {e}")
        return _NONE_RESULT
