"""
Prepare a custom document dataset for training.

Designed to work with tiny datasets — as few as 5 genuine + 5 fake documents.

Workflow:
  1. Load your genuine and fake documents (PDF, JPG, PNG, BMP, TIFF supported)
  2. Convert PDFs to images (renders first page at 200 DPI for high quality)
  3. Apply heavy augmentation to expand each original into many variants
  4. Create a stratified train/val/test split
  5. Output the directory structure expected by train_model.py

Usage (5 + 5 documents):
    python prepare_custom_dataset.py \
        --genuine_dir ./my_docs/genuine \
        --fake_dir    ./my_docs/fake    \
        --output_dir  ./dataset         \
        --augment_factor 30

    Then train:
    python train_model.py \
        --data_dir   ./dataset  \
        --epochs     30         \
        --batch_size 8          \
        --freeze_epochs 15      \
        --patience   8

Output structure:
    dataset/
    ├── train/
    │   ├── genuine/    (original + augmented variants)
    │   └── tampered/
    ├── val/
    │   ├── genuine/    (originals only — no augmentation)
    │   └── tampered/
    └── test/
        ├── genuine/
        └── tampered/
"""

import argparse
import io
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"}


def render_pdf_first_page(pdf_path: str, dpi: int = 200) -> Image.Image:
    """Render the first page of a PDF to a PIL Image."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF required for PDF support: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        raise ValueError(f"Empty PDF: {pdf_path}")
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def load_document(path: str) -> Image.Image:
    """Load any supported document format as a PIL RGB Image."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return render_pdf_first_page(path)
    return Image.open(path).convert("RGB")


def augment_image(image: Image.Image) -> Image.Image:
    """
    Apply a random combination of augmentations to a document image.

    Augmentations are chosen to simulate real-world document variation:
    - Rotation (scanned at slight angles)
    - Scale + random crop (zoom in on regions)
    - Brightness / contrast (different scanner/camera settings)
    - Gaussian blur (focus variation, low-res scans)
    - JPEG re-compression at random quality (critical for ELA robustness)
    - Gaussian pixel noise (sensor noise)
    - Perspective warp (photographed documents)
    """
    try:
        import torchvision.transforms.functional as TF
    except ImportError:
        raise ImportError("torchvision required: pip install torchvision")

    img = image.copy()
    w, h = img.size

    # Horizontal flip (ELA artifacts are position-agnostic)
    if random.random() < 0.5:
        img = TF.hflip(img)

    # Rotation ±25° with gray fill
    angle = random.uniform(-25, 25)
    img = TF.rotate(img, angle, fill=128)

    # Random scale crop (zoom 75–100% of original)
    scale = random.uniform(0.75, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    left = random.randint(0, max(w - cw, 0))
    top = random.randint(0, max(h - ch, 0))
    img = img.crop((left, top, left + cw, top + ch))
    img = img.resize((w, h), Image.LANCZOS)

    # Brightness + contrast jitter
    img = TF.adjust_brightness(img, random.uniform(0.65, 1.35))
    img = TF.adjust_contrast(img,   random.uniform(0.65, 1.35))
    img = TF.adjust_saturation(img, random.uniform(0.75, 1.25))

    # Gaussian blur (simulate scan quality variation)
    if random.random() < 0.4:
        radius = random.uniform(0.3, 2.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # JPEG re-compression at random quality — most important for ELA
    jpeg_quality = random.randint(55, 97)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    img = Image.open(buf).copy()

    # Gaussian pixel noise
    if random.random() < 0.6:
        arr = np.array(img, dtype=np.float32)
        noise_std = random.uniform(1.0, 12.0)
        arr += np.random.normal(0, noise_std, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def collect_files(directory: str) -> list:
    """Return sorted list of supported document file paths in a directory."""
    paths = []
    for fname in os.listdir(directory):
        if Path(fname).suffix.lower() in VALID_EXTS:
            paths.append(os.path.join(directory, fname))
    return sorted(paths)


def stratified_split(items: list, train_ratio: float, val_ratio: float, seed: int):
    """Split a list into train/val/test with a fixed seed."""
    lst = list(items)
    rng = random.Random(seed)
    rng.shuffle(lst)
    n = len(lst)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(0, min(int(n * val_ratio), n - n_train))
    train = lst[:n_train]
    val   = lst[n_train : n_train + n_val]
    test  = lst[n_train + n_val :]
    return train, val, test


def save_images_to(items, out_dir: str, augment: bool, augment_factor: int):
    """Save original images (and optionally augmented variants) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for img, stem in items:
        # Always save the original
        orig_path = os.path.join(out_dir, f"{stem}_orig.jpg")
        img.save(orig_path, format="JPEG", quality=95)
        saved += 1

        if augment and augment_factor > 1:
            for i in range(augment_factor - 1):
                try:
                    aug = augment_image(img)
                    aug_path = os.path.join(out_dir, f"{stem}_aug{i:04d}.jpg")
                    aug.save(aug_path, format="JPEG", quality=90)
                    saved += 1
                except Exception as e:
                    print(f"    Warning: augmentation failed for {stem} #{i}: {e}")
    return saved


def prepare_dataset(
    genuine_dir: str,
    fake_dir: str,
    output_dir: str,
    augment_factor: int = 30,
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    seed: int = 42,
):
    genuine_paths = collect_files(genuine_dir)
    fake_paths    = collect_files(fake_dir)

    print(f"Found {len(genuine_paths)} genuine, {len(fake_paths)} fake documents")

    if len(genuine_paths) == 0 or len(fake_paths) == 0:
        print("ERROR: Need at least 1 genuine and 1 fake document.")
        return

    # Load all documents into memory (gives cleaner error messages early)
    def load_all(paths, label_name):
        loaded = []
        for p in paths:
            try:
                img = load_document(p)
                loaded.append((img, Path(p).stem))
                print(f"  Loaded {label_name}: {Path(p).name}  ({img.size[0]}×{img.size[1]})")
            except Exception as e:
                print(f"  SKIP {p}: {e}")
        return loaded

    print("\nLoading genuine documents...")
    genuine_imgs = load_all(genuine_paths, "genuine")
    print("\nLoading fake documents...")
    fake_imgs = load_all(fake_paths, "fake")

    if not genuine_imgs or not fake_imgs:
        print("ERROR: No usable images loaded.")
        return

    # Stratified split on originals
    g_train, g_val, g_test = stratified_split(genuine_imgs, train_ratio, val_ratio, seed)
    f_train, f_val, f_test = stratified_split(fake_imgs,    train_ratio, val_ratio, seed)

    print(f"\nOriginal split (before augmentation):")
    print(f"  train : {len(g_train)} genuine, {len(f_train)} fake")
    print(f"  val   : {len(g_val)} genuine, {len(f_val)} fake")
    print(f"  test  : {len(g_test)} genuine, {len(f_test)} fake")

    print(f"\nGenerating dataset with augment_factor={augment_factor} ...")

    splits = {
        "train": (g_train, f_train, True,  augment_factor),
        "val":   (g_val,   f_val,   False, 1),
        "test":  (g_test,  f_test,  False, 1),
    }

    for split_name, (genuine, fake, do_aug, factor) in splits.items():
        g_out = os.path.join(output_dir, split_name, "genuine")
        f_out = os.path.join(output_dir, split_name, "tampered")
        g_count = save_images_to(genuine, g_out, do_aug, factor)
        f_count = save_images_to(fake,    f_out, do_aug, factor)
        print(f"  {split_name:5s}: {g_count:4d} genuine, {f_count:4d} tampered  →  {g_out}")

    total_train_g = len(g_train) * augment_factor
    total_train_f = len(f_train) * augment_factor

    print(f"\nDataset ready at: {output_dir}")
    print(f"\nRecommended training command:")
    print(
        f"  python train_model.py \\\n"
        f"    --data_dir   {output_dir} \\\n"
        f"    --epochs     30 \\\n"
        f"    --batch_size 8 \\\n"
        f"    --freeze_epochs 15 \\\n"
        f"    --patience   8"
    )
    print(f"\nTip: with {len(genuine_imgs)}+{len(fake_imgs)} original docs and augment_factor={augment_factor},")
    print(f"  training set has ~{total_train_g} genuine and ~{total_train_f} tampered images.")
    if total_train_g < 50 or total_train_f < 50:
        print(f"  Consider increasing --augment_factor (e.g. 40–50) for better generalization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a custom document dataset for fraud detection training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--genuine_dir", required=True,
        help="Directory containing genuine/authentic document files",
    )
    parser.add_argument(
        "--fake_dir", required=True,
        help="Directory containing fake/tampered document files",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where to write the prepared dataset (train/val/test subdirs)",
    )
    parser.add_argument(
        "--augment_factor", type=int, default=30,
        help="How many images to generate per original (default: 30, so 5 docs → 150 images)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.75,
        help="Fraction of originals used for training (default: 0.75)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.125,
        help="Fraction of originals used for validation (default: 0.125)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    args = parser.parse_args()

    prepare_dataset(
        genuine_dir=args.genuine_dir,
        fake_dir=args.fake_dir,
        output_dir=args.output_dir,
        augment_factor=args.augment_factor,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
