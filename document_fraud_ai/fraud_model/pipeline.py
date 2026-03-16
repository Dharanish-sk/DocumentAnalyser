"""
Document Fraud Detection Pipeline — Optimized V2.

Key improvements over V1:
  - TTA (Test-Time Augmentation) for CNN inference: averages predictions
    over original + flipped views for more robust scores
  - Multi-quality ELA for CNN: runs CNN on 3 ELA quality levels (85/90/95)
    and averages predictions — catches tampering visible at specific qualities
  - Supports V2 model (raw logits + sigmoid) and V1 model (sigmoid output)
  - Better score calibration with temperature scaling

Orchestrates all analysis modules into a single prediction pipeline:
1. ELA Analysis (image tampering detection)
2. CNN Model Inference (deep learning classification)
3. Metadata Analysis (consistency checks)
4. OCR + Text Analysis (content anomalies)
5. Copy-Move Detection (feature matching)
6. Blur/Sharpness Inconsistency
7. ManTraNet (pixel-level forgery detection — optional)

Each module produces a risk score [0, 1] or None (if unavailable).
The final fraud probability is a weighted combination using only active modules.
"""

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger

from fraud_model.cnn_model import load_model, IMAGENET_MEAN, IMAGENET_STD
from utils.ela_analysis import (
    compute_ela, ela_to_array,
    compute_ela_statistics, detect_copy_move,
    compute_blur_sharpness,
)
from utils.metadata_analyzer import analyze_file_metadata
from utils.ocr_extractor import extract_text, analyze_text_anomalies
from utils.mantranet import load_mantranet, run_mantranet, is_available as mantranet_available


# Weights for combining module scores into final fraud probability
WEIGHTS = {
    "ela_statistical": 0.20,   # ELA ensemble (3 quality levels)
    "cnn_prediction":  0.25,   # EfficientNet-B3 V2 — best single-model accuracy
    "mantranet":       0.25,   # ManTraNet — pixel-level forgery detection
    "metadata":        0.10,   # Metadata consistency checks
    "text_anomaly":    0.08,   # OCR + text anomaly detection
    "copy_move":       0.08,   # AKAZE-based copy-move detection
    "blur_sharpness":  0.00,   # Local sharpness inconsistency
}


class FraudDetectionPipeline:
    """End-to-end document fraud detection pipeline — V2 Optimized."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 enable_tta: bool = True, enable_multi_quality_cnn: bool = True):
        """
        Initialize the pipeline.

        Args:
            model_path: Path to a trained CNN model. If None, uses ELA heuristics only.
            device: 'cuda' or 'cpu'. Auto-detects if None.
            enable_tta: Enable Test-Time Augmentation for CNN (more robust, 4x slower).
            enable_multi_quality_cnn: Run CNN on 3 ELA quality levels and average.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.model_loaded = False
        self.model_type = "none"
        self.enable_tta = enable_tta
        self.enable_multi_quality_cnn = enable_multi_quality_cnn

        # Temperature for score calibration (can be tuned on a held-out set)
        self.temperature = 1.0

        # ManTraNet
        self.mantranet_model = None
        if mantranet_available():
            try:
                self.mantranet_model = load_mantranet(device=self.device)
                logger.info("ManTraNet loaded successfully")
            except Exception as e:
                logger.warning(f"ManTraNet load failed (skipping): {e}")
        else:
            logger.info(
                "ManTraNet not set up — run utils.mantranet.setup_mantranet() to enable it. "
                "Pipeline will run without ManTraNet."
            )

        if model_path and os.path.exists(model_path):
            try:
                # Auto-detect model version based on state_dict keys
                try:
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                except TypeError:
                    state_dict = torch.load(model_path, map_location=self.device)

                # V2 models have 'head.' keys; V1 models have 'backbone.classifier.' keys
                is_v2 = any(k.startswith("head.") for k in state_dict.keys())
                self.model_type = "efficientnet_b3_v2" if is_v2 else "efficientnet_b3"

                self.model = load_model(model_path, model_type=self.model_type, device=self.device)
                self.model_loaded = True
                logger.info(f"{self.model_type} model loaded from {model_path} on {self.device}")

                # JIT-compile for faster inference
                if hasattr(torch, "compile"):
                    try:
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        logger.info("Model compiled with torch.compile(mode='reduce-overhead')")
                    except Exception as e:
                        logger.warning(f"torch.compile failed (continuing without): {e}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using heuristics only.")
        else:
            logger.info("No pretrained model provided. Using ELA heuristics + metadata analysis.")

    def _get_image(self, file_path: str) -> Optional[Image.Image]:
        """Get a PIL Image from file. For PDFs, renders the first page."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(file_path)
                if len(doc) == 0:
                    return None
                page = doc[0]
                mat = fitz.Matrix(200 / 72, 200 / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                return img
            except Exception as e:
                logger.error(f"PDF rendering failed: {e}")
                return None
        else:
            try:
                return Image.open(file_path).convert("RGB")
            except Exception as e:
                logger.error(f"Image loading failed: {e}")
                return None

    def _preprocess_for_cnn(self, ela_image: Image.Image) -> torch.Tensor:
        """Convert ELA image to normalized tensor for CNN input."""
        arr = ela_to_array(ela_image, target_size=(300, 300))  # (3, 300, 300), [0,1]

        # Apply ImageNet normalization (required for EfficientNet)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std

        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, 3, 300, 300)

    def _run_cnn_single(self, tensor: torch.Tensor) -> float:
        """Run CNN on a single preprocessed tensor. Returns probability."""
        with torch.no_grad():
            if self.enable_tta:
                # TTA: average over original + flipped views
                views = [
                    tensor,
                    torch.flip(tensor, dims=[3]),     # horizontal flip
                    torch.flip(tensor, dims=[2]),     # vertical flip
                    torch.flip(tensor, dims=[2, 3]),  # both flips
                ]
                all_outputs = []
                for view in views:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
                        output = self.model(view.to(self.device))
                    all_outputs.append(output)
                output = torch.stack(all_outputs).mean(dim=0)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
                    output = self.model(tensor.to(self.device))

        logit = output.squeeze().cpu().float().item()

        # V2 model outputs raw logits — apply sigmoid
        # V1 model already applies sigmoid — detect and skip
        if self.model_type == "efficientnet_b3_v2":
            # Temperature-scaled sigmoid for calibrated probabilities
            prob = float(torch.sigmoid(torch.tensor(logit / self.temperature)))
        else:
            prob = logit  # V1 already has sigmoid

        return prob

    def _run_cnn(self, ela_image: Image.Image, original_image: Image.Image = None) -> Optional[float]:
        """
        Run the CNN model on an ELA image.

        V2 improvements:
          - Multi-quality ELA: runs CNN on ELA at qualities 85, 90, 95 and
            averages predictions. Catches tampering visible at specific
            compression levels that single-quality ELA might miss.
          - TTA: each quality level uses 4 augmented views.

        Returns fraud probability [0, 1], or None if no model is loaded.
        """
        if not self.model_loaded:
            return None

        try:
            if self.enable_multi_quality_cnn and original_image is not None:
                # Multi-quality ELA for CNN (V2 improvement)
                probs = []
                for q in [85, 90, 95]:
                    ela_q = compute_ela(original_image, quality=q)
                    tensor_q = self._preprocess_for_cnn(ela_q)
                    prob_q = self._run_cnn_single(tensor_q)
                    probs.append(prob_q)
                return float(np.mean(probs))
            else:
                # Single quality (fallback)
                tensor = self._preprocess_for_cnn(ela_image)
                return self._run_cnn_single(tensor)

        except Exception as e:
            logger.error(f"CNN inference failed: {e}")
            return None

    def _combine_scores(self, scores: dict) -> float:
        """Combine module scores using weight redistribution."""
        active = {k: v for k, v in scores.items() if v is not None}
        if not active:
            return 0.0

        total_weight = sum(WEIGHTS[k] for k in active if k in WEIGHTS)
        if total_weight == 0:
            return 0.0

        fraud_prob = sum(v * WEIGHTS[k] for k, v in active.items() if k in WEIGHTS)
        return float(min(max(fraud_prob / total_weight, 0.0), 1.0))

    def _compute_confidence(self, scores: dict, fraud_prob: float) -> float:
        """Compute confidence score based on agreement, decisiveness, coverage."""
        active = {k: v for k, v in scores.items() if v is not None}
        if not active:
            return 0.0

        active_values = list(active.values())
        active_weight = sum(WEIGHTS[k] for k in active if k in WEIGHTS)
        total_weight = sum(WEIGHTS.values())

        score_std = float(np.std(active_values))
        agreement_score = max(0.0, 1.0 - 2.0 * score_std)
        decisiveness = abs(fraud_prob - 0.5) * 2.0
        coverage_score = active_weight / total_weight if total_weight > 0 else 0.0

        confidence = (agreement_score * 0.4 + decisiveness * 0.4 + coverage_score * 0.2) * 100
        return round(float(max(0.0, min(confidence, 100.0))), 2)

    def analyze(self, file_path: str) -> dict:
        """
        Run the full fraud detection pipeline on a document.

        Args:
            file_path: Path to the document (PDF, JPG, PNG, etc.)

        Returns:
            Complete analysis result with fraud probability and reasons.
        """
        logger.info(f"Analyzing: {file_path}")

        if not os.path.exists(file_path):
            return {
                "error": "File not found",
                "fraud_probability": 0.0,
                "is_fraud": False,
                "confidence": 0.0,
                "reasons": [],
            }

        reasons = []
        scores = {}

        # ---- Module 1: ELA Statistical Analysis (multi-quality ensemble) ----
        image = self._get_image(file_path)
        if image:
            ela_scores_ensemble = []
            for q in [85, 90, 95]:
                ela_q = compute_ela(image, quality=q)
                stats_q = compute_ela_statistics(ela_q)
                ela_scores_ensemble.append(stats_q["ela_score"])
            ensemble_ela_score = float(np.mean(ela_scores_ensemble))

            ela_image = compute_ela(image, quality=90)
            ela_stats = compute_ela_statistics(ela_image)

            scores["ela_statistical"] = ensemble_ela_score

            if ela_stats["has_anomaly"]:
                reasons.append(
                    f"ELA: Suspicious compression artifacts detected "
                    f"(cluster_ratio: {ela_stats['suspicious_region_ratio']:.4f}, "
                    f"ela_score: {ensemble_ela_score:.4f})"
                )

            # ---- Module 2: CNN Prediction (V2: TTA + multi-quality) ----
            cnn_score = self._run_cnn(ela_image, original_image=image)
            scores["cnn_prediction"] = cnn_score
            if cnn_score is not None and cnn_score > 0.6:
                reasons.append(f"CNN model: High tampering probability ({cnn_score:.2%})")

            # ---- Module 5: Copy-Move Detection ----
            copy_move = detect_copy_move(image)
            scores["copy_move"] = copy_move["copy_move_score"]
            if copy_move["copy_move_detected"]:
                reasons.append(
                    f"Copy-move forgery detected ({copy_move['matched_regions']} matching regions, "
                    f"{copy_move['cluster_count']} displacement clusters)"
                )

            # ---- Module 6: Blur/Sharpness Inconsistency ----
            blur_result = compute_blur_sharpness(image)
            scores["blur_sharpness"] = blur_result["blur_score"]
            if blur_result["blur_score"] > 0.4:
                reasons.append(
                    f"Local sharpness inconsistency detected "
                    f"({blur_result['inconsistent_cell_count']} inconsistent cells)"
                )

            # ---- Module 7: ManTraNet ----
            if self.mantranet_model is not None:
                mt_result = run_mantranet(self.mantranet_model, image, device=self.device)
                scores["mantranet"] = mt_result["mantranet_score"]
                if mt_result["mantranet_score"] is not None and mt_result["mantranet_score"] > 0.5:
                    reasons.append(
                        f"ManTraNet: Pixel-level tampering detected "
                        f"(score: {mt_result['mantranet_score']:.4f}, "
                        f"suspicious_region: {mt_result['suspicious_region_ratio']:.2%})"
                    )
            else:
                scores["mantranet"] = None

        else:
            scores["ela_statistical"] = None
            scores["cnn_prediction"] = None
            scores["copy_move"] = None
            scores["blur_sharpness"] = None
            scores["mantranet"] = None
            reasons.append("Could not render document for image analysis")

        # ---- Module 3: Metadata Analysis ----
        meta_result = analyze_file_metadata(file_path)
        scores["metadata"] = meta_result["risk_score"]
        reasons.extend(meta_result["anomalies"])

        # ---- Module 4: Text/OCR Analysis ----
        text_result = extract_text(file_path)
        extracted_text = text_result.get("total_text") or text_result.get("text", "")
        text_anomaly_result = analyze_text_anomalies(extracted_text)
        scores["text_anomaly"] = text_anomaly_result["risk_score"]
        reasons.extend(text_anomaly_result["anomalies"])

        # ---- Combine Scores ----
        fraud_probability = self._combine_scores(scores)
        is_fraud = fraud_probability > 0.5
        confidence = self._compute_confidence(scores, fraud_probability)

        if not reasons:
            reasons.append("No anomalies detected - document appears genuine")

        result = {
            "fraud_probability": round(fraud_probability, 4),
            "is_fraud": is_fraud,
            "confidence": confidence,
            "reasons": reasons,
            "details": {
                "module_scores": {
                    k: round(v, 4) if v is not None else None
                    for k, v in scores.items()
                },
                "model_used": (
                    (self.model_type if self.model_loaded else "Heuristic")
                    + (" + ManTraNet" if self.mantranet_model is not None else "")
                    + (" + TTA" if self.enable_tta and self.model_loaded else "")
                    + (" + MultiQ-CNN" if self.enable_multi_quality_cnn and self.model_loaded else "")
                ),
                "device": self.device,
                "metadata_summary": {
                    k: v for k, v in meta_result.items() if k != "raw_metadata"
                },
                "text_extracted": bool(extracted_text),
                "text_length": len(extracted_text),
            },
        }

        logger.info(
            f"Result: fraud_prob={result['fraud_probability']}, "
            f"is_fraud={result['is_fraud']}, confidence={result['confidence']}%"
        )

        return result
