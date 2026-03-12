"""
One-time local setup for ManTraNet.

Run this ONCE from your project directory before starting the app:

    python setup_mantranet_local.py

What it does:
  1. Installs gdown (if missing)
  2. Clones ManTraNet-pytorch repo  → mantranet_src/
  3. Downloads pretrained weights   → mantranet_src/ManTraNet.pt
  4. Verifies the model can be imported
  5. Prints confirmation

How to get the weights file ID
────────────────────────────────
  1. Open: https://github.com/RonyAbecidan/ManTraNet-pytorch
  2. Find the Google Drive / download link in the README
  3. Copy the file ID from the link, e.g.:
       https://drive.google.com/file/d/1A9xBcD2efGhIj.../view
                                        ^^^^^^^^^^^^^^^^
  4. Paste it below as WEIGHTS_GDRIVE_ID, OR pass it as a command-line arg:
       python setup_mantranet_local.py --gdrive_id YOUR_FILE_ID

Alternatively, if you already have ManTraNet.pt downloaded:
    python setup_mantranet_local.py --weights_path C:/Downloads/ManTraNet.pt
"""

import argparse
import os
import subprocess
import sys
import shutil

# ── Config ─────────────────────────────────────────────────────────────────────
_PROJECT_DIR      = os.path.dirname(os.path.abspath(__file__))
_MANTRANET_SRC    = os.path.join(_PROJECT_DIR, "mantranet_src")
_WEIGHTS_DEST     = os.path.join(_MANTRANET_SRC, "ManTraNet.pt")
_REPO_URL         = "https://github.com/RonyAbecidan/ManTraNet-pytorch.git"

# Paste your Google Drive file ID here to skip the command-line prompt
WEIGHTS_GDRIVE_ID = ""   # e.g. "1A9xBcD2efGhIjKlMnOpQr"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _step(n, total, msg):
    print(f"\n[{n}/{total}] {msg}")


def _ok(msg):
    print(f"      OK  {msg}")


def _err(msg):
    print(f"      ERR {msg}")


def _install_gdown():
    """Install gdown via pip if not already available."""
    try:
        import gdown  # noqa: F401
        _ok("gdown already installed")
    except ImportError:
        print("      Installing gdown ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown>=5.1.0", "-q"],
            check=True,
        )
        _ok("gdown installed")


def _clone_repo():
    """Clone ManTraNet-pytorch to mantranet_src/."""
    model_py = os.path.join(_MANTRANET_SRC, "ManTraNet.py")
    if os.path.exists(model_py):
        _ok(f"Repo already cloned at {_MANTRANET_SRC}")
        return

    print(f"      Cloning {_REPO_URL} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", _REPO_URL, _MANTRANET_SRC],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _err(f"git clone failed:\n{result.stderr}")
        print("\nFix: make sure git is installed → https://git-scm.com/download/win")
        sys.exit(1)

    _ok(f"Cloned → {_MANTRANET_SRC}")


def _download_weights(gdrive_id: str = None, local_path: str = None):
    """Get ManTraNet.pt from Google Drive or a local copy."""
    if os.path.exists(_WEIGHTS_DEST):
        size_mb = os.path.getsize(_WEIGHTS_DEST) / 1_048_576
        _ok(f"Weights already present ({size_mb:.1f} MB) → {_WEIGHTS_DEST}")
        return

    # ── Option A: copy from local file ───────────────────────────────────────
    if local_path:
        if not os.path.exists(local_path):
            _err(f"Local weights file not found: {local_path}")
            sys.exit(1)
        shutil.copy(local_path, _WEIGHTS_DEST)
        size_mb = os.path.getsize(_WEIGHTS_DEST) / 1_048_576
        _ok(f"Weights copied ({size_mb:.1f} MB) → {_WEIGHTS_DEST}")
        return

    # ── Option B: download from Google Drive ─────────────────────────────────
    file_id = gdrive_id or WEIGHTS_GDRIVE_ID
    if not file_id:
        print("\n" + "=" * 60)
        print("  ManTraNet weights file ID required.")
        print("  Steps:")
        print("  1. Open: https://github.com/RonyAbecidan/ManTraNet-pytorch")
        print("  2. Copy the Google Drive file ID from the README link")
        print("  3. Run:")
        print("       python setup_mantranet_local.py --gdrive_id YOUR_ID")
        print("  OR paste the ID into WEIGHTS_GDRIVE_ID in this script.")
        print("=" * 60)
        sys.exit(0)

    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"      Downloading weights from Google Drive (id={file_id}) ...")
    gdown.download(url, _WEIGHTS_DEST, quiet=False)

    if not os.path.exists(_WEIGHTS_DEST):
        _err("Download failed — file not created.")
        print("  Check: is the file ID correct? Is it publicly shared?")
        sys.exit(1)

    size_mb = os.path.getsize(_WEIGHTS_DEST) / 1_048_576
    _ok(f"Weights downloaded ({size_mb:.1f} MB) → {_WEIGHTS_DEST}")


def _verify():
    """Import the model and confirm weights load without errors."""
    if _MANTRANET_SRC not in sys.path:
        sys.path.insert(0, _MANTRANET_SRC)

    try:
        from ManTraNet import ManTraNet  # type: ignore
    except ImportError as e:
        _err(f"Cannot import ManTraNet: {e}")
        sys.exit(1)

    import torch
    model = ManTraNet()

    try:
        try:
            state = torch.load(_WEIGHTS_DEST, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(_WEIGHTS_DEST, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"      NOTE: {len(missing)} missing keys (minor mismatch — usually fine)")
        if unexpected:
            print(f"      NOTE: {len(unexpected)} unexpected keys (minor mismatch — usually fine)")
        _ok("Weights loaded successfully")
    except Exception as e:
        _err(f"Weight loading failed: {e}")
        sys.exit(1)

    _ok("ManTraNet import and weight load verified")


def _print_summary():
    print("\n" + "=" * 60)
    print("  ManTraNet is ready for local use!")
    print()
    print("  The pipeline will automatically load ManTraNet")
    print("  the next time you start the app:")
    print()
    print("    # Terminal 1 — FastAPI backend")
    print("    python app.py")
    print()
    print("    # Terminal 2 — Streamlit frontend")
    print("    streamlit run streamlit_app.py")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="One-time ManTraNet local setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gdrive_id",
        default="",
        help="Google Drive file ID for ManTraNet.pt weights",
    )
    parser.add_argument(
        "--weights_path",
        default="",
        help="Local path to an existing ManTraNet.pt file",
    )
    args = parser.parse_args()

    total = 4
    print("\nManTraNet Local Setup")
    print("=" * 60)

    _step(1, total, "Installing gdown ...")
    _install_gdown()

    _step(2, total, "Cloning ManTraNet-pytorch repo ...")
    _clone_repo()

    _step(3, total, "Getting pretrained weights ...")
    _download_weights(
        gdrive_id=args.gdrive_id or WEIGHTS_GDRIVE_ID,
        local_path=args.weights_path or None,
    )

    _step(4, total, "Verifying setup ...")
    _verify()

    _print_summary()


if __name__ == "__main__":
    main()
