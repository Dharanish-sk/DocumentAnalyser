"""
Training script for Document Fraud Detection — Optimized V2.

Key improvements over V1:
  - BCEWithLogitsLoss (numerically stable, works with raw logits from V2 model)
  - Focal Loss option (better for class-imbalanced fraud detection)
  - Mixup augmentation (proven +1-2% AUC in image classification)
  - EMA (Exponential Moving Average) of weights — smoother, better generalization
  - Stronger augmentations: RandomErasing, perspective warp, elastic distortion
  - OneCycleLR scheduler (super-convergence, often outperforms cosine)
  - Test-Time Augmentation (TTA) at validation for more reliable AUC
  - num_workers=2 for faster data loading
  - Training history saved as JSON + plotted as PNG
  - Configurable via CLI flags: --focal_loss, --mixup, --ema, --tta

Supports two dataset layouts automatically:

  Structured layout (recommended, created by prepare_custom_dataset.py):
    data_dir/
    +-- train/genuine/   +-- train/tampered/
    +-- val/genuine/     +-- val/tampered/
    +-- test/genuine/    +-- test/tampered/

  Flat layout (raw documents, auto-split at runtime):
    data_dir/
    +-- genuine/     (all genuine documents)
    +-- tampered/    (all fake/tampered documents)

Usage:
    # Standard training (recommended defaults)
    python train_model.py --data_dir ./dataset --epochs 50 --batch_size 16

    # Maximum accuracy (all optimizations)
    python train_model.py --data_dir ./dataset --epochs 60 --batch_size 8 \
        --accum_steps 4 --focal_loss --mixup --ema --tta \
        --freeze_epochs 5 --patience 12

    # Tiny dataset (5+5 docs, after prepare_custom_dataset.py)
    python train_model.py --data_dir ./dataset --epochs 30 --batch_size 8 \
        --freeze_epochs 15 --patience 8 --focal_loss --mixup --ema
"""

import argparse
import io
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from fraud_model.cnn_model import FraudEfficientNetB3V2, IMAGENET_MEAN, IMAGENET_STD
from utils.ela_analysis import compute_ela, ela_to_array


# ============================================================================
# Augmentation
# ============================================================================

def _random_jpeg_resave(image: Image.Image, quality_range=(70, 95)) -> Image.Image:
    """Re-save image at random JPEG quality to simulate compression artifacts."""
    q = random.randint(*quality_range)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def _apply_training_augmentation(image: Image.Image) -> Image.Image:
    """
    Apply training-time augmentations to the original image BEFORE ELA.

    V2 additions over V1:
      - Perspective warp (simulates photographed documents)
      - More aggressive color jitter range
      - Random sharpness adjustment
      - Gaussian blur with wider range
    """
    import torchvision.transforms.functional as TF

    # RandomHorizontalFlip
    if random.random() < 0.5:
        image = TF.hflip(image)

    # RandomVerticalFlip
    if random.random() < 0.2:
        image = TF.vflip(image)

    # RandomRotation +/-20 deg (wider than V1's +/-15)
    angle = random.uniform(-20, 20)
    image = TF.rotate(image, angle, fill=128)

    # Perspective warp (NEW in V2 — simulates photographed docs)
    if random.random() < 0.3:
        distortion = random.uniform(0.05, 0.2)
        w, h = image.size
        half_h, half_w = h // 2, w // 2
        topleft = [int(random.uniform(0, distortion * half_w)), int(random.uniform(0, distortion * half_h))]
        topright = [int(w - random.uniform(0, distortion * half_w)), int(random.uniform(0, distortion * half_h))]
        botright = [int(w - random.uniform(0, distortion * half_w)), int(h - random.uniform(0, distortion * half_h))]
        botleft = [int(random.uniform(0, distortion * half_w)), int(h - random.uniform(0, distortion * half_h))]
        startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
        endpoints = [topleft, topright, botright, botleft]
        image = TF.perspective(image, startpoints, endpoints, fill=128)

    # ColorJitter: wider range than V1
    brightness_factor = random.uniform(0.7, 1.3)
    contrast_factor = random.uniform(0.7, 1.3)
    saturation_factor = random.uniform(0.8, 1.2)
    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)
    image = TF.adjust_saturation(image, saturation_factor)

    # Random sharpness (NEW in V2)
    if random.random() < 0.3:
        sharpness = random.uniform(0.5, 2.0)
        image = TF.adjust_sharpness(image, sharpness)

    # Gaussian blur (wider range)
    if random.random() < 0.25:
        from PIL import ImageFilter
        radius = random.uniform(0.5, 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    # Random JPEG re-save (teaches ELA robustness to compression)
    image = _random_jpeg_resave(image, quality_range=(65, 95))

    return image


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf")


def _collect_files(directory: str) -> list:
    """Return sorted list of file paths from a directory."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    )


class ELADataset(Dataset):
    """
    Dataset that applies ELA preprocessing with augmentation for training.

    V2 additions:
      - RandomErasing on the ELA tensor (simulates occluded regions)
      - Elastic-like noise distortion on ELA image
      - Wider Gaussian noise range
    """

    def __init__(self, data_dir: str, split: str = "train", target_size: tuple = (300, 300),
                 flat_split_seed: int = 42):
        self.split = split
        self.target_size = target_size
        self.is_train = (split == "train")
        self.samples = []

        split_dir = os.path.join(data_dir, split)

        if os.path.isdir(split_dir):
            n_genuine, n_tampered = self._load_from_dir(split_dir)
            logger.info(f"[{split}] structured layout: {n_genuine} genuine, {n_tampered} tampered")
        else:
            genuine_dir  = os.path.join(data_dir, "genuine")
            tampered_dir = os.path.join(data_dir, "tampered")

            if not os.path.isdir(genuine_dir) or not os.path.isdir(tampered_dir):
                logger.error(
                    f"[{split}] Cannot find '{split_dir}' or flat 'genuine'/'tampered' dirs in {data_dir}"
                )
                self.pos_weight = 1.0
                return

            all_genuine  = _collect_files(genuine_dir)
            all_tampered = _collect_files(tampered_dir)
            total = len(all_genuine) + len(all_tampered)

            if total < 10:
                logger.warning(
                    f"Only {total} raw documents found. "
                    f"Run prepare_custom_dataset.py first to augment to a viable training size."
                )

            g_subset  = self._flat_split(all_genuine,  split, flat_split_seed)
            t_subset  = self._flat_split(all_tampered, split, flat_split_seed + 1)

            self.samples = [(p, 0) for p in g_subset] + [(p, 1) for p in t_subset]
            n_genuine, n_tampered = len(g_subset), len(t_subset)
            logger.info(f"[{split}] flat layout (auto-split): {n_genuine} genuine, {n_tampered} tampered")

        n_genuine  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_tampered = sum(1 for _, lbl in self.samples if lbl == 1)
        self.pos_weight = (n_genuine / n_tampered) if n_tampered > 0 else 1.0
        logger.info(f"[{split}] pos_weight={self.pos_weight:.2f}")

    def _load_from_dir(self, directory: str):
        n_genuine = n_tampered = 0
        genuine_dir = os.path.join(directory, "genuine")
        tampered_dir = os.path.join(directory, "tampered")
        if os.path.isdir(genuine_dir):
            for fname in sorted(os.listdir(genuine_dir)):
                if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                    self.samples.append((os.path.join(genuine_dir, fname), 0))
                    n_genuine += 1
        if os.path.isdir(tampered_dir):
            for fname in sorted(os.listdir(tampered_dir)):
                if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                    self.samples.append((os.path.join(tampered_dir, fname), 1))
                    n_tampered += 1
        return n_genuine, n_tampered

    @staticmethod
    def _flat_split(paths: list, split: str, seed: int) -> list:
        lst = list(paths)
        rng = random.Random(seed)
        rng.shuffle(lst)
        n = len(lst)
        n_train = max(1, int(n * 0.75))
        n_val   = max(0, min(int(n * 0.125), n - n_train))
        if split == "train":
            return lst[:n_train]
        elif split == "val":
            return lst[n_train : n_train + n_val]
        else:
            return lst[n_train + n_val :]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        if path.lower().endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(path)
                page = doc[0]
                mat = fitz.Matrix(200 / 72, 200 / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                return img
            except Exception as e:
                raise RuntimeError(f"PDF render failed: {e}")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = self._load_image(path)

            if self.is_train:
                image = _apply_training_augmentation(image)

            ela_quality = random.randint(75, 95) if self.is_train else 90
            ela_image = compute_ela(image, quality=ela_quality)
            arr = ela_to_array(ela_image, self.target_size)  # (3, H, W), [0,1]

            # Training-only tensor augmentations
            if self.is_train:
                # Gaussian noise (wider range in V2)
                noise_std = random.uniform(0.01, 0.08)
                arr = arr + np.random.normal(0, noise_std, arr.shape).astype(np.float32)
                arr = np.clip(arr, 0.0, 1.0)

                # RandomErasing on ELA tensor (NEW in V2)
                # Simulates occluded/missing regions — forces model to use global patterns
                if random.random() < 0.3:
                    _, h, w = arr.shape
                    erase_h = random.randint(int(h * 0.05), int(h * 0.25))
                    erase_w = random.randint(int(w * 0.05), int(w * 0.25))
                    top = random.randint(0, h - erase_h)
                    left = random.randint(0, w - erase_w)
                    arr[:, top:top+erase_h, left:left+erase_w] = np.random.uniform(0, 1, (3, erase_h, erase_w)).astype(np.float32)

            # ImageNet normalization
            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
            arr = (arr - mean) / std

            return torch.tensor(arr, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            arr = np.zeros((3, *self.target_size), dtype=np.float32)
            return torch.tensor(arr), torch.tensor(float(label), dtype=torch.float32)


# ============================================================================
# Loss Functions
# ============================================================================

LABEL_SMOOTHING = 0.1


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss for binary classification with logits.

    Focal Loss down-weights easy examples and focuses training on hard cases.
    Critical for fraud detection where genuine docs are "easy" and the model
    needs to learn subtle tampering signals.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    With gamma=2.0: a correctly classified sample with p=0.9 gets 100x LESS
    gradient than a misclassified sample with p=0.1.  This forces the model
    to focus on the hardest, most informative examples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 reduces to standard BCE. Recommended: 1.5-2.5.
        alpha: Balance factor for positive class. Set > 0.5 if positives are rare.
        label_smoothing: Smooth labels before loss computation.
        pos_weight: Weight for positive class (for class imbalance).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5,
                 label_smoothing: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Class-weighted BCE with logits (numerically stable)
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")

        # Focal modulation
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class SmoothedBCEWithLogitsLoss(nn.Module):
    """Standard BCE with logits + label smoothing + class weights.
    
    Uses BCEWithLogitsLoss internally which applies the log-sum-exp trick
    for numerical stability (avoids the log(sigmoid(x)) underflow that
    happens with BCE + Sigmoid for extreme logit values).
    """

    def __init__(self, label_smoothing: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


# ============================================================================
# Mixup Augmentation
# ============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Mixup: creates convex combinations of training pairs.

    Mixup trains the model on interpolated examples:
        x_mixed = lambda * x_i + (1 - lambda) * x_j
        y_mixed = lambda * y_i + (1 - lambda) * y_j

    This regularizes the model by:
      - Smoothing decision boundaries (less overconfident)
      - Providing infinite virtual training examples
      - Acting as a strong regularizer (reduces overfitting by 30-50%)

    Proven to improve AUC by 1-2% on image classification benchmarks.

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.

    Args:
        alpha: Beta distribution parameter. Higher = more mixing.
               alpha=0.4 works well for most tasks. alpha=1.0 = uniform mixing.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 (keep dominant sample)

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup-augmented batch."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights as a running average:
        shadow = decay * shadow + (1 - decay) * current_weights

    Why this helps:
      - Individual training steps are noisy (especially with augmentation + mixup)
      - EMA smooths out these oscillations, giving more stable weights
      - Typically improves generalization by 0.5-1% AUC
      - The shadow model is used for validation and final inference

    Args:
        model: PyTorch model to track.
        decay: Smoothing factor. 0.999 = slow update (more stable).
               0.99 = faster adaptation. Default 0.999 is recommended.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """Replace model weights with shadow (for validation/inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model weights (after validation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()



# ============================================================================
# Test-Time Augmentation (TTA)
# ============================================================================

def tta_predict(model, images: torch.Tensor, device: str) -> torch.Tensor:
    """Test-Time Augmentation: average predictions over augmented versions.

    Runs the model on 4 views of each image:
      1. Original
      2. Horizontal flip
      3. Vertical flip
      4. Both flips

    Then averages the logits.  This reduces variance in predictions and
    typically improves AUC by 0.3-0.8% at the cost of 4x inference time
    (acceptable for validation, not used during training).

    Args:
        model: Model in eval mode.
        images: (batch, 3, H, W) tensor.
        device: 'cuda' or 'cpu'.

    Returns:
        Averaged logits (batch, 1).
    """
    views = [
        images,                           # original
        torch.flip(images, dims=[3]),     # horizontal flip
        torch.flip(images, dims=[2]),     # vertical flip
        torch.flip(images, dims=[2, 3]),  # both flips
    ]
    all_logits = []
    for view in views:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(view.to(device))
        all_logits.append(logits)

    return torch.stack(all_logits).mean(dim=0)


# ============================================================================
# Training History
# ============================================================================

class TrainingHistory:
    """Track and save training metrics per epoch."""

    def __init__(self):
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_auc": [],
            "val_f1": [], "val_precision": [], "val_recall": [],
            "lr": [], "epoch_time": [],
        }

    def append(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.history:
                self.history[k].append(v)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {path}")

    def plot(self, path: str):
        """Save training curves as PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Loss
            axes[0, 0].plot(self.history["train_loss"], label="Train Loss", color="blue")
            axes[0, 0].plot(self.history["val_loss"], label="Val Loss", color="red")
            axes[0, 0].set_title("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Accuracy
            axes[0, 1].plot(self.history["train_acc"], label="Train Acc", color="blue")
            axes[0, 1].plot(self.history["val_acc"], label="Val Acc", color="red")
            axes[0, 1].set_title("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # AUC + F1
            axes[1, 0].plot(self.history["val_auc"], label="Val AUC", color="green", linewidth=2)
            axes[1, 0].plot(self.history["val_f1"], label="Val F1", color="orange")
            axes[1, 0].set_title("AUC-ROC & F1")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Learning Rate
            axes[1, 1].plot(self.history["lr"], label="LR", color="purple")
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].set_yscale("log")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info(f"Training curves saved: {path}")
        except ImportError:
            logger.warning("matplotlib not available — skipping plot")


# ============================================================================
# Main Training Function
# ============================================================================

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")
    logger.info(f"Config: {vars(args)}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Load datasets
    train_set = ELADataset(args.data_dir, split="train")
    val_set = ELADataset(args.data_dir, split="val")

    if len(train_set) == 0:
        logger.error("No training samples found. Check data_dir structure.")
        return

    # num_workers=2 for faster data loading (V2 improvement)
    nw = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=(device == "cuda")
    )

    # Model — V2 with logits output
    model = FraudEfficientNetB3V2(pretrained=True).to(device)

    # Loss function
    if args.focal_loss:
        criterion = FocalBCEWithLogitsLoss(
            gamma=2.0, alpha=0.5,
            label_smoothing=LABEL_SMOOTHING,
            pos_weight=train_set.pos_weight
        )
        logger.info(f"Using Focal Loss (gamma=2.0, alpha=0.5, pos_weight={train_set.pos_weight:.2f})")
    else:
        criterion = SmoothedBCEWithLogitsLoss(
            label_smoothing=LABEL_SMOOTHING,
            pos_weight=train_set.pos_weight
        )
        logger.info(f"Using Smoothed BCE Loss (pos_weight={train_set.pos_weight:.2f})")

    # EMA
    ema = None
    if args.ema:
        ema = EMA(model, decay=0.999)
        logger.info("EMA enabled (decay=0.999)")

    # Phase 1: freeze backbone, train classifier only with higher LR
    model.freeze_backbone()
    classifier_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(classifier_params, lr=args.lr * 10, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.freeze_epochs, 1))

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best_val_auc = 0.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "fraud_efficientnet_b3_v2_best.pth")
    history = TrainingHistory()

    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Phase 2: unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs:
            logger.info(f"Epoch {epoch+1}: Unfreezing backbone for full fine-tuning")
            model.unfreeze_backbone()
            if ema:
                ema = EMA(model, decay=0.999)  # Reinit EMA with all params

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

            # OneCycleLR for phase 2 (V2 improvement — super-convergence)
            phase2_epochs = args.epochs - args.freeze_epochs
            steps_per_epoch = len(train_loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr * 3,
                epochs=phase2_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,          # 10% warmup
                anneal_strategy="cos",
                div_factor=10,          # initial_lr = max_lr / 10
                final_div_factor=100,   # final_lr = initial_lr / 100
            )
            logger.info(f"OneCycleLR: max_lr={args.lr*3:.6f}, {phase2_epochs} epochs, {steps_per_epoch} steps/epoch")

        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Mixup (V2 improvement)
            use_mixup = args.mixup and random.random() < 0.5  # 50% chance
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(images).squeeze(1)

                if use_mixup:
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam) / args.accum_steps
                else:
                    loss = criterion(logits, labels) / args.accum_steps

            scaler.scale(loss).backward()

            is_last_step = (step + 1 == len(train_loader))
            if (step + 1) % args.accum_steps == 0 or is_last_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # EMA update after each optimizer step
                if ema:
                    ema.update()

            # OneCycleLR steps per batch (not per epoch)
            if epoch >= args.freeze_epochs:
                scheduler.step()

            train_loss += loss.item() * args.accum_steps * images.size(0)
            with torch.no_grad():
                predicted = (torch.sigmoid(logits.detach()) > 0.5).float()
                if use_mixup:
                    train_correct += (lam * (predicted == labels_a).float() + (1 - lam) * (predicted == labels_b).float()).sum().item()
                else:
                    train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

        # Phase 1 scheduler steps per epoch
        if epoch < args.freeze_epochs:
            scheduler.step()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # --- Validation ---
        if ema:
            ema.apply_shadow()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # TTA at validation (V2 improvement)
                if args.tta:
                    logits = tta_predict(model, images, device).squeeze(1)
                else:
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                        logits = model(images).squeeze(1)

                loss = F.binary_cross_entropy_with_logits(logits, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        if ema:
            ema.restore()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        # Metrics
        try:
            val_auc = float(roc_auc_score(all_labels, all_preds)) if len(set(all_labels)) > 1 else 0.5
        except Exception:
            val_auc = 0.5

        binary_preds = [1.0 if p > 0.5 else 0.0 for p in all_preds]
        try:
            val_f1 = float(f1_score(all_labels, binary_preds, zero_division=0))
            val_prec = float(precision_score(all_labels, binary_preds, zero_division=0))
            val_rec = float(recall_score(all_labels, binary_preds, zero_division=0))
        except Exception:
            val_f1 = val_prec = val_rec = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Record history
        history.append(
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc, val_auc=val_auc,
            val_f1=val_f1, val_precision=val_prec, val_recall=val_rec,
            lr=current_lr, epoch_time=epoch_time,
        )

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
            f"AUC: {val_auc:.4f}, F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best by AUC-ROC
        if val_auc > best_val_auc:
            best_val_auc = val_auc

            # Save EMA weights if available, otherwise current weights
            if ema:
                ema.apply_shadow()
                torch.save(model.state_dict(), save_path)
                ema.restore()
            else:
                torch.save(model.state_dict(), save_path)

            logger.info(f"Best model saved: {save_path} (val_auc: {val_auc:.4f}, f1: {val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no AUC improvement for {args.patience} epochs)")
                break

    total_time = time.time() - total_start
    logger.info(f"Training complete in {total_time/60:.1f} min. Best val AUC: {best_val_auc:.4f}")

    # Save training history
    history.save(os.path.join(args.output_dir, "training_history.json"))
    history.plot(os.path.join(args.output_dir, "training_curves.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Document Fraud Detection (EfficientNet-B3 V2 — Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Epochs to freeze backbone")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--output_dir", type=str, default="./fraud_model", help="Model output directory")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")

    # V2 optimization flags
    parser.add_argument("--focal_loss", action="store_true", help="Use Focal Loss instead of BCE (better for imbalanced data)")
    parser.add_argument("--mixup", action="store_true", help="Enable Mixup augmentation (+1-2%% AUC)")
    parser.add_argument("--ema", action="store_true", help="Enable EMA weight averaging (smoother convergence)")
    parser.add_argument("--tta", action="store_true", help="Enable TTA at validation (4x slower but more reliable AUC)")

    args = parser.parse_args()
    train(args)
