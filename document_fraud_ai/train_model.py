"""
Training script for the Document Fraud Detection model (EfficientNet-B3).

Supports two dataset layouts automatically:

  Structured layout (recommended, created by prepare_custom_dataset.py):
    data_dir/
    ├── train/genuine/   ├── train/tampered/
    ├── val/genuine/     ├── val/tampered/
    └── test/genuine/    └── test/tampered/

  Flat layout (raw documents, auto-split at runtime):
    data_dir/
    ├── genuine/     (all genuine documents)
    └── tampered/    (all fake/tampered documents)

Usage — large dataset (CASIA/COVERAGE):
    python train_model.py --data_dir ./dataset --epochs 50 --batch_size 16 \
        --freeze_epochs 5 --patience 10

Usage — tiny custom dataset (5+5 documents, use after prepare_custom_dataset.py):
    python train_model.py --data_dir ./dataset --epochs 30 --batch_size 8 \
        --freeze_epochs 15 --patience 8
"""

import argparse
import io
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from loguru import logger
from sklearn.metrics import roc_auc_score

from fraud_model.cnn_model import FraudEfficientNetB3, IMAGENET_MEAN, IMAGENET_STD
from utils.ela_analysis import compute_ela, ela_to_array


def _random_jpeg_resave(image: Image.Image, quality_range=(70, 95)) -> Image.Image:
    """Re-save image at random JPEG quality to simulate compression artifacts."""
    q = random.randint(*quality_range)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def _apply_training_augmentation(image: Image.Image) -> Image.Image:
    """Apply training-time augmentations to the original image before ELA."""
    import torchvision.transforms.functional as TF

    # RandomHorizontalFlip
    if random.random() < 0.5:
        image = TF.hflip(image)

    # RandomVerticalFlip
    if random.random() < 0.2:
        image = TF.vflip(image)

    # RandomRotation ±15° (fill with gray)
    angle = random.uniform(-15, 15)
    image = TF.rotate(image, angle, fill=128)

    # ColorJitter: brightness/contrast ×[0.8, 1.2]
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)

    # Random JPEG re-save (teaches ELA robustness to compression)
    image = _random_jpeg_resave(image, quality_range=(70, 95))

    return image


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf")


def _collect_files(directory: str) -> list:
    """Return sorted list of (path, label) pairs from a directory."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    )


class ELADataset(Dataset):
    """
    Dataset that applies ELA preprocessing with augmentation for training.

    Auto-detects layout:
    - Structured: data_dir/split/genuine/ + data_dir/split/tampered/
    - Flat:       data_dir/genuine/ + data_dir/tampered/  (auto-split by seed)
    """

    def __init__(self, data_dir: str, split: str = "train", target_size: tuple = (300, 300),
                 flat_split_seed: int = 42):
        self.split = split
        self.target_size = target_size
        self.is_train = (split == "train")
        self.samples = []

        split_dir = os.path.join(data_dir, split)

        if os.path.isdir(split_dir):
            # Structured layout: data_dir/train/genuine/, data_dir/val/genuine/, etc.
            n_genuine, n_tampered = self._load_from_dir(split_dir)
            logger.info(f"[{split}] structured layout: {n_genuine} genuine, {n_tampered} tampered")
        else:
            # Flat layout: data_dir/genuine/ + data_dir/tampered/
            # Deterministically split by seed so train/val/test see different subsets
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
        """Load from directory/genuine/ and directory/tampered/."""
        n_genuine = n_tampered = 0
        for fname in sorted(os.listdir(os.path.join(directory, "genuine")) if os.path.isdir(os.path.join(directory, "genuine")) else []):
            if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                self.samples.append((os.path.join(directory, "genuine", fname), 0))
                n_genuine += 1
        for fname in sorted(os.listdir(os.path.join(directory, "tampered")) if os.path.isdir(os.path.join(directory, "tampered")) else []):
            if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                self.samples.append((os.path.join(directory, "tampered", fname), 1))
                n_tampered += 1
        return n_genuine, n_tampered

    @staticmethod
    def _flat_split(paths: list, split: str, seed: int) -> list:
        """Deterministically split a flat file list into train/val/test."""
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
        else:  # test
            return lst[n_train + n_val :]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        """Load image or render first PDF page."""
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

            # Training augmentation on original image before ELA
            if self.is_train:
                image = _apply_training_augmentation(image)

            # Random ELA quality during training, fixed q=90 for val/test
            ela_quality = random.randint(75, 95) if self.is_train else 90
            ela_image = compute_ela(image, quality=ela_quality)

            arr = ela_to_array(ela_image, self.target_size)  # (3, H, W), [0,1]

            # Apply Gaussian noise during training
            if self.is_train:
                noise_std = random.uniform(0.01, 0.05)
                arr = arr + np.random.normal(0, noise_std, arr.shape).astype(np.float32)
                arr = np.clip(arr, 0.0, 1.0)

            # ImageNet normalization
            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
            arr = (arr - mean) / std

            return torch.tensor(arr, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            arr = np.zeros((3, *self.target_size), dtype=np.float32)
            return torch.tensor(arr), torch.tensor(float(label), dtype=torch.float32)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")

    # Load datasets
    train_set = ELADataset(args.data_dir, split="train")
    val_set = ELADataset(args.data_dir, split="val")

    if len(train_set) == 0:
        logger.error("No training samples found. Check data_dir structure.")
        return

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda")
    )

    # Model
    model = FraudEfficientNetB3(pretrained=True).to(device)
    pos_weight_tensor = torch.tensor([train_set.pos_weight], dtype=torch.float32).to(device)

    # Phase 1: freeze backbone, train classifier only with higher LR
    model.freeze_backbone()
    classifier_params = [p for p in model.backbone.classifier.parameters()]
    optimizer = torch.optim.AdamW(classifier_params, lr=args.lr * 10, weight_decay=1e-4)

    # Cosine annealing over total epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "fraud_efficientnet_b3_best.pth")

    for epoch in range(args.epochs):
        # Phase 2: unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs:
            logger.info(f"Epoch {epoch+1}: Unfreezing backbone for full fine-tuning")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs
            )

        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)

            # Class-weighted BCE loss
            bce_per_sample = nn.functional.binary_cross_entropy(
                outputs, labels, reduction="none"
            )
            weight_tensor = torch.where(labels == 1, pos_weight_tensor, torch.ones_like(labels))
            loss = (bce_per_sample * weight_tensor).mean()

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze(1)

                bce_per_sample = nn.functional.binary_cross_entropy(
                    outputs, labels, reduction="none"
                )
                weight_tensor = torch.where(labels == 1, pos_weight_tensor, torch.ones_like(labels))
                loss = (bce_per_sample * weight_tensor).mean()

                val_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(outputs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        # AUC-ROC (more robust than accuracy for imbalanced sets)
        try:
            val_auc = float(roc_auc_score(all_labels, all_preds)) if len(set(all_labels)) > 1 else 0.5
        except Exception:
            val_auc = 0.5

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}"
        )

        # Save best by AUC-ROC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved: {save_path} (val_auc: {val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no AUC improvement for {args.patience} epochs)")
                break

    logger.info(f"Training complete. Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Document Fraud Detection (EfficientNet-B3)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (B3 is larger than B0)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Epochs to freeze backbone")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--output_dir", type=str, default="./fraud_model", help="Model output directory")
    args = parser.parse_args()
    train(args)
