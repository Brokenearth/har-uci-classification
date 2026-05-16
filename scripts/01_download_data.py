#!/usr/bin/env python3
"""Descarga y extrae el UCI HAR Dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from har_classifier.download import get_dataset_root

if __name__ == "__main__":
    root = get_dataset_root()
    print(f"Dataset listo: {root}")
