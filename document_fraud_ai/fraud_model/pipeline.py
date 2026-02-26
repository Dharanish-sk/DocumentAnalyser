"""
Document Fraud Detection Pipeline.

Orchestrates all analysis modules into a single prediction pipeline:
1. ELA Analysis (image tampering detection)
2. CNN Model Inference (deep learning classification)
3. Metadata Analysis (consistency checks)
4. OCR + Text Analysis (content anomalies)
5. Copy-Move Detection (feature matching)
6. Blur/Sharpness Inconsistency

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


# Weights for combining module scores into final fraud probability
WEIGHTS = {
    "ela_statistical": 0.25,   # ↑ improved with spatial clustering
    "cnn_prediction":  0.35,   # ↑ EfficientNet-B3 is most accurate module
    "metadata":        0.15,   # ↓ reduced — still some false positives
    "text_anomaly":    0.10,   # ↓ narrow signal
    "copy_move":       0.10,   # AKAZE-based, continuous score
    "blur_sharpness":  0.05,   # NEW — local sharpness inconsistency
}


class FraudDetectionPipeline:
    """
    End-to-end document fraud detection pipeline.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            model_path: Path to a trained CNN model. If None, uses ELA heuristics only.
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.model_loaded = False

        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path, model_type="efficientnet_b3", device=self.device)
                self.model_loaded = True
                logger.info(f"EfficientNet-B3 model loaded from {model_path} on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using heuristics only.")
        else:
            logger.info("No pretrained model provided. Using ELA heuristics + metadata analysis.")

    def _get_image(self, file_path: str) -> Optional[Image.Image]:
        """
        Get a PIL Image from file. For PDFs, renders the first page.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(file_path)
                if len(doc) == 0:
                    return None
                page = doc[0]
                mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI matches train_model.py
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

    def _run_cnn(self, ela_image: Image.Image) -> Optional[float]:
        """
        Run the CNN model on an ELA image.

        Returns fraud probability [0, 1], or None if no model is loaded.
        Returning None (not 0.5) prevents a fixed score bias when no model is available.
        """
        if not self.model_loaded:
            return None  # Do NOT return 0.5 — that adds a fixed +0.15 offset to all scores

        try:
            arr = ela_to_array(ela_image, target_size=(300, 300))  # (3, 300, 300), [0,1]

            # Apply ImageNet normalization (required for EfficientNet)
            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
            std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
            arr = (arr - mean) / std

            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.model(tensor)

            return float(prediction.squeeze().cpu().item())
        except Exception as e:
            logger.error(f"CNN inference failed: {e}")
            return None

    def _combine_scores(self, scores: dict) -> float:
        """
        Combine module scores using weight redistribution.

        None values are excluded; remaining weights are renormalized so the
        total always sums to 1.0 regardless of which modules are active.
        """
        active = {k: v for k, v in scores.items() if v is not None}
        if not active:
            return 0.0

        total_weight = sum(WEIGHTS[k] for k in active if k in WEIGHTS)
        if total_weight == 0:
            return 0.0

        fraud_prob = sum(v * WEIGHTS[k] for k, v in active.items() if k in WEIGHTS)
        return float(min(max(fraud_prob / total_weight, 0.0), 1.0))

    def _compute_confidence(self, scores: dict, fraud_prob: float) -> float:
        """
        Compute confidence score with corrected logic.

        High confidence requires:
        - Modules AGREE with each other (low std)
        - Probability is DECISIVE (far from 0.5)
        - Good COVERAGE (many active modules)

        The old implementation used 1 - std, which gave HIGH confidence when
        all modules returned ~0.5 (the worst-case unknown scenario).
        """
        active = {k: v for k, v in scores.items() if v is not None}
        if not active:
            return 0.0

        active_values = list(active.values())
        active_weight = sum(WEIGHTS[k] for k in active if k in WEIGHTS)
        total_weight = sum(WEIGHTS.values())

        # Agreement: low variance among module scores = high agreement
        score_std = float(np.std(active_values))
        agreement_score = max(0.0, 1.0 - 2.0 * score_std)

        # Decisiveness: distance from 0.5 (uncertain midpoint)
        decisiveness = abs(fraud_prob - 0.5) * 2.0

        # Coverage: fraction of total weight represented by active modules
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

        # ---- Module 1: ELA Statistical Analysis ----
        image = self._get_image(file_path)
        if image:
            ela_image = compute_ela(image)
            ela_stats = compute_ela_statistics(ela_image)

            # Use pre-computed ela_score (not arbitrary ratio * 10 scaling)
            scores["ela_statistical"] = ela_stats["ela_score"]

            if ela_stats["has_anomaly"]:
                reasons.append(
                    f"ELA: Suspicious compression artifacts detected "
                    f"(cluster_ratio: {ela_stats['suspicious_region_ratio']:.4f}, "
                    f"ela_score: {ela_stats['ela_score']:.4f})"
                )

            # ---- Module 2: CNN Prediction ----
            cnn_score = self._run_cnn(ela_image)
            scores["cnn_prediction"] = cnn_score  # May be None if no model loaded
            if cnn_score is not None and cnn_score > 0.6:
                reasons.append(f"CNN model: High tampering probability ({cnn_score:.2%})")

            # ---- Module 5: Copy-Move Detection ----
            copy_move = detect_copy_move(image)
            scores["copy_move"] = copy_move["copy_move_score"]  # Continuous [0,1]
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
        else:
            # Image unavailable — leave CNN/ELA/copy-move/blur as None (not 0.5)
            scores["ela_statistical"] = None
            scores["cnn_prediction"] = None
            scores["copy_move"] = None
            scores["blur_sharpness"] = None
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

        # ---- Combine Scores (weight redistribution for missing modules) ----
        fraud_probability = self._combine_scores(scores)

        # Determine fraud threshold
        is_fraud = fraud_probability > 0.5

        # Confidence with corrected logic
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
                "model_used": "EfficientNet-B3" if self.model_loaded else "Heuristic (no pretrained model)",
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
