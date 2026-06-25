#!/usr/bin/env python3
"""
download_model.py — One-time model download script (build time only)

Run this ONCE before executing rank.py to download and bundle the
all-MiniLM-L6-v2 sentence-transformer model weights locally.

After running this script, set SEMANTIC_BACKEND=transformer and the
ranking pipeline will use the transformer embeddings instead of TF-IDF.

Usage:
    python download_model.py

Requirements:
    pip install sentence-transformers torch  (CPU-only: add --extra-index-url)

The model weights (~90 MB) are saved to lib/models/all-MiniLM-L6-v2/.
They are excluded from git via .gitignore to avoid bloating the repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH = Path(__file__).parent / "lib" / "models" / MODEL_NAME


def main() -> None:
    print(f"Downloading {MODEL_NAME} to {SAVE_PATH} ...")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print(
            "INFO: sentence-transformers is not installed.\n"
            "Skipping model download since TF-IDF is the default scoring mechanism."
        )
        sys.exit(0)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(SAVE_PATH))
    print(f"Model saved to: {SAVE_PATH}")
    print()
    print("To use transformer embeddings instead of TF-IDF, run rank.py with:")
    print("    SEMANTIC_BACKEND=transformer python rank.py --candidates ... --out ...")


if __name__ == "__main__":
    main()
