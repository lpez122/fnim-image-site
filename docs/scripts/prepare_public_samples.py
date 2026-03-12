#!/usr/bin/env python3
"""
Copy a lightweight 1-3 image public subset for every category into the published site assets.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMAGE_ROOT = Path("/Users/lukepezanko/Downloads/beh10/images/allimages")
DEFAULT_OUTPUT_ROOT = Path("/Users/lukepezanko/Documents/New project/docs/assets/sample-images")


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-per-category", type=int, default=3)
    args = parser.parse_args()

    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    category_count = 0

    for category_dir in sorted([path for path in image_root.iterdir() if path.is_dir()]):
        files = sorted(
            [path for path in category_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS],
            key=natural_key,
        )
        if not files:
            continue

        dst_dir = output_root / category_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file_path in files[: max(1, args.max_per_category)]:
            shutil.copy2(file_path, dst_dir / file_path.name)
            copied += 1
        category_count += 1

    print(f"[SAVED] {category_count} categories with {copied} public sample images into {output_root}")


if __name__ == "__main__":
    main()
