#!/usr/bin/env python3
"""Preprocesa datos: split por sujeto, normalización, har_processed.npz."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from har_classifier.preprocessing import prepare_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    prepare_and_save(val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
