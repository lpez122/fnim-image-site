#!/usr/bin/env python3
"""
beh10_within_category_triplets_vgg.py

Within-category triplet selection using VGG16 embeddings.

Folder structure expected:
  <image_root>/
    bed/
      bed_1.png
      bed_2.png
      ...
    accordion/
      accordion_1.png
      ...

Goal:
  Collect up to N within-category triplets total, with the constraint that
  EACH CATEGORY CAN BE USED AT MOST ONCE per run (no reuse).

Cross-run exclusion (NEW):
  When you run HIGH after LOW, pass:
    --exclude_used_from_dir "<low_out_dir>/collected_triplets"
  The script will parse used categories from collected filenames like:
    <cat>__1.png
  and skip those categories.

Or pass:
  --exclude_used_from_file "/path/to/used_categories.txt"
to skip categories listed one per line.

Example (LOW band first):
python /Users/lukepezanko/Downloads/beh10/scripts/beh10_within_category_triplets_vgg.py \
  --image_root "/Users/lukepezanko/Downloads/beh10/images/allimages" \
  --out_dir "/Users/lukepezanko/Downloads/beh10/CNN/within/low_band_v2" \
  --layer block5_conv2 \
  --vgg_rule all_pairs --vgg_min -1.0 --vgg_max 0.20 \
  --max_imgs_per_cat 50 \
  --sample --max_samples_per_cat 900000 \
  --collect_n 18 \
  --exclude_used_from_file "/Users/lukepezanko/Downloads/beh10/images/betweencat/used_categories.txt"

Example (HIGH band next, exclude cats used in LOW):
python /Users/lukepezanko/Downloads/beh10/scripts/beh10_within_category_triplets_vgg.py \
  --image_root "/Users/lukepezanko/Downloads/beh10/images/allimages" \
  --out_dir "/Users/lukepezanko/Downloads/beh10/CNN/within/high_band_v2" \
  --layer block5_conv2 \
  --vgg_rule all_pairs --vgg_min 0.60 --vgg_max 0.80 \
  --max_imgs_per_cat 50 \
  --sample --max_samples_per_cat 200000 \
  --collect_n 18 \
  --exclude_used_from_dir "/Users/lukepezanko/Downloads/beh10/CNN/within/low_band_v2/collected_triplets"

Banding:
  Use --vgg_min/--vgg_max with --vgg_rule applied to the three pairwise cosines in a triplet.
  For strict “every pair must be in the band”, use: --vgg_rule all_pairs

Embedding procedure:
  VGG16(include_top=False) -> chosen conv layer -> flatten -> L2 normalize -> cosine via dot
"""

import os
import re
import shutil
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# -----------------------------
# Helpers
# -----------------------------
def safe_token(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unk"


def list_images_recursive(root: str) -> Dict[str, List[str]]:
    """
    Returns mapping: category -> list of image paths
    Category is the immediate subfolder name under root.
    """
    rootp = Path(os.path.expanduser(root))
    if not rootp.exists():
        raise FileNotFoundError(f"image_root not found: {rootp}")

    pools: Dict[str, List[str]] = {}
    for cat_dir in sorted([p for p in rootp.iterdir() if p.is_dir()]):
        cat = cat_dir.name.lower().strip()
        imgs = []
        for p in cat_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                imgs.append(str(p))
        imgs = sorted(imgs)
        if imgs:
            pools[cat] = imgs
    return pools


def build_vgg_model(layer_name: str) -> Model:
    base = VGG16(weights="imagenet", include_top=False)
    try:
        out = base.get_layer(layer_name).output
    except ValueError as e:
        available = [l.name for l in base.layers]
        raise ValueError(
            f"Layer '{layer_name}' not found in VGG16. Example layers: {available[:15]}"
        ) from e
    return Model(inputs=base.input, outputs=out)


def load_batch(paths: List[str]) -> np.ndarray:
    batch = []
    for p in paths:
        img = tf.keras.preprocessing.image.load_img(p, target_size=(224, 224))
        arr = tf.keras.preprocessing.image.img_to_array(img)
        batch.append(arr)
    x = np.stack(batch, axis=0)
    return preprocess_input(x)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def compute_embeddings(model: Model, paths: List[str], batch_size: int = 16) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        x = load_batch(batch_paths)
        feat = model.predict(x, verbose=0)                # (B,H,W,C)
        b = feat.shape[0]
        feat = feat.reshape((b, -1)).astype(np.float32)   # flatten
        feat = l2_normalize(feat)
        for j, p in enumerate(batch_paths):
            out[p] = feat[j]
    return out


def cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.clip(np.dot(u, v), -1.0, 1.0))


def in_band(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


def triplet_pass(vals: List[float], rule: str, lo: float, hi: float) -> bool:
    """
    vals: [v12, v13, v23]
    """
    if any([not np.isfinite(v) for v in vals]):
        return False

    flags = [in_band(v, lo, hi) for v in vals]

    if rule == "all_pairs":
        return all(flags)
    if rule == "two_of_three":
        return sum(flags) >= 2
    if rule == "mean":
        return in_band(float(np.mean(vals)), lo, hi)
    if rule == "max":
        return in_band(float(np.max(vals)), lo, hi)
    if rule == "min":
        return in_band(float(np.min(vals)), lo, hi)

    raise ValueError(f"Unknown rule: {rule}")


def unique_dest_path(dest_dir: str, base_name: str) -> str:
    dest_dir = os.path.expanduser(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    base = Path(base_name)
    stem = base.stem
    ext = base.suffix
    cand = os.path.join(dest_dir, base_name)
    if not os.path.exists(cand):
        return cand
    for k in range(1, 10000):
        cand2 = os.path.join(dest_dir, f"{stem}_r{k:03d}{ext}")
        if not os.path.exists(cand2):
            return cand2
    raise RuntimeError(f"Could not find unique filename for {base_name} in {dest_dir}")


def save_triplet(
    collect_dir: str,
    cat: str,
    p1: str, p2: str, p3: str,
    copy_mode: str = "copy",
) -> Tuple[str, str, str]:
    """
    Saves the triplet in collect_dir as:
      <cat>_1.png, <cat>_2.png, <cat>_3.png
    """
    cat_s = safe_token(cat)

    ext1 = Path(p1).suffix.lower() or ".png"
    ext2 = Path(p2).suffix.lower() or ".png"
    ext3 = Path(p3).suffix.lower() or ".png"

    name1 = f"{cat_s}_1{ext1}"
    name2 = f"{cat_s}_2{ext2}"
    name3 = f"{cat_s}_3{ext3}"

    d1 = unique_dest_path(collect_dir, name1)
    d2 = unique_dest_path(collect_dir, name2)
    d3 = unique_dest_path(collect_dir, name3)

    op = shutil.copy2 if copy_mode == "copy" else shutil.move
    op(p1, d1)
    op(p2, d2)
    op(p3, d3)

    return d1, d2, d3


def sample_triplets(rng: np.random.Generator, img_paths: List[str], n_samples: int):
    if len(img_paths) < 3:
        return []
    return [tuple(rng.choice(img_paths, size=3, replace=False)) for _ in range(n_samples)]


def exhaustive_triplets(img_paths: List[str]):
    return itertools.combinations(img_paths, 3)


def parse_used_categories_from_dir(dir_path: str) -> Set[str]:
    """
    Supports collected filenames like:
      <cat>_1.png
      <cat>_2.png
      <cat>_3.png
    and legacy forms:
      <cat>_<uid>_1.png
      <cat>_<uid>_2.png
      <cat>_<uid>_3.png

    Returns set({cat, ...})
    """
    if not dir_path:
        return set()

    d = Path(os.path.expanduser(dir_path))
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"--exclude_used_from_dir not found or not a directory: {d}")

    used = set()
    for p in d.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue

        stem = p.stem  # e.g., "bed__123456789_1"
        if "__" not in stem:
            continue

        cat = stem.split("_", 1)[0].strip().lower()
        if cat:
            used.add(cat)

    return used


def parse_used_categories_from_file(file_path: str) -> Set[str]:
    """
    Expects a text file with one category per line.
    Empty lines and lines starting with '#' are ignored.
    """
    if not file_path:
        return set()

    p = Path(os.path.expanduser(file_path))
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"--exclude_used_from_file not found or not a file: {p}")

    used = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        used.add(raw.lower())

    return used


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", type=str, required=True,
                    help="Root folder containing category subfolders.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for manifests and collected images.")

    ap.add_argument("--layer", type=str, default="block5_conv2")
    ap.add_argument("--batch_size", type=int, default=16)

    ap.add_argument("--vgg_rule", type=str, default="all_pairs",
                    choices=["all_pairs", "two_of_three", "mean", "max", "min"])
    ap.add_argument("--vgg_min", type=float, default=-1.0)
    ap.add_argument("--vgg_max", type=float, default=0.20)

    ap.add_argument("--max_imgs_per_cat", type=int, default=0,
                    help="Cap images per category (0 = no cap).")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--collect_n", type=int, default=17,
                    help="Total triplets to collect (each from a unique category).")
    ap.add_argument("--no_reuse_categories", action="store_true", default=True,
                    help="If enabled (default), each category can contribute at most 1 triplet in this run.")
    ap.add_argument("--collect_copy_mode", type=str, default="copy",
                    choices=["copy", "move"])
    ap.add_argument("--collect_subdir", type=str, default="collected_triplets")

    ap.add_argument(
        "--exclude_used_from_dir",
        type=str,
        default="",
        help="If set, parse previously-used categories from filenames in this folder (e.g., prior run collected_triplets) "
             "and exclude them from selection in this run."
    )
    ap.add_argument(
        "--exclude_used_from_file",
        type=str,
        default="",
        help="If set, parse previously-used categories from a text file (one category per line) "
             "and exclude them from selection in this run."
    )

    ap.add_argument("--sample", action="store_true",
                    help="Sample random triplets per category instead of exhaustive combinations.")
    ap.add_argument("--max_samples_per_cat", type=int, default=200000,
                    help="If --sample, number of sampled triplets to evaluate per category.")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    image_root = os.path.expanduser(args.image_root)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    collect_dir = os.path.join(out_dir, args.collect_subdir)
    os.makedirs(collect_dir, exist_ok=True)

    pools = list_images_recursive(image_root)
    cats = sorted(pools.keys())
    if not cats:
        raise ValueError(f"No category subfolders with images found under: {image_root}")

    if args.max_imgs_per_cat and args.max_imgs_per_cat > 0:
        for c in cats:
            pools[c] = pools[c][:args.max_imgs_per_cat]

    # Flatten all paths for embedding
    all_paths = []
    for c in cats:
        all_paths.extend(pools[c])
    all_paths = sorted(list(dict.fromkeys(all_paths)))

    print(f"[INFO] Categories found: {len(cats)}")
    print(f"[INFO] Total images found: {len(all_paths)}")

    vgg = build_vgg_model(args.layer)
    emb = compute_embeddings(vgg, all_paths, batch_size=args.batch_size)

    manifest_rows = []
    summary_rows = []

    collected_total = 0
    used_categories: Set[str] = set()

    # Seed used categories from a prior run (LOW first -> HIGH after)
    if args.exclude_used_from_dir:
        prior_used = parse_used_categories_from_dir(args.exclude_used_from_dir)
        used_categories |= prior_used
        print(f"[INFO] Preloaded {len(prior_used)} used categories from: {args.exclude_used_from_dir}")
    if args.exclude_used_from_file:
        prior_used_file = parse_used_categories_from_file(args.exclude_used_from_file)
        used_categories |= prior_used_file
        print(f"[INFO] Preloaded {len(prior_used_file)} used categories from: {args.exclude_used_from_file}")

    for c in cats:
        if collected_total >= args.collect_n:
            break
        if args.no_reuse_categories and c in used_categories:
            continue

        imgs = pools[c]
        if len(imgs) < 3:
            summary_rows.append({
                "cat": c, "n_imgs": len(imgs),
                "evaluated": 0, "collected": 0,
                "note": "fewer_than_3_images"
            })
            continue

        evaluated = 0
        collected_here = 0

        triplet_iter = sample_triplets(rng, imgs, args.max_samples_per_cat) if args.sample else exhaustive_triplets(imgs)

        for (p1, p2, p3) in triplet_iter:
            if collected_total >= args.collect_n:
                break
            if collected_here >= 1:  # hard stop: only 1 triplet per category
                break

            e1, e2, e3 = emb[p1], emb[p2], emb[p3]
            v12 = cos(e1, e2)
            v13 = cos(e1, e3)
            v23 = cos(e2, e3)
            evaluated += 1

            if not triplet_pass([v12, v13, v23], args.vgg_rule, args.vgg_min, args.vgg_max):
                continue

            d1, d2, d3 = save_triplet(
                collect_dir=collect_dir,
                cat=c,
                p1=p1, p2=p2, p3=p3,
                copy_mode=args.collect_copy_mode
            )

            collected_here = 1
            collected_total += 1
            used_categories.add(c)

            v_triplet_min = float(np.min([v12, v13, v23]))
            v_triplet_max = float(np.max([v12, v13, v23]))

            manifest_rows.append({
                "cat": c,
                "src1": Path(p1).name, "src2": Path(p2).name, "src3": Path(p3).name,
                "dst1": Path(d1).name, "dst2": Path(d2).name, "dst3": Path(d3).name,
                "vgg_12": v12, "vgg_13": v13, "vgg_23": v23,
                "vgg_mean": float(np.mean([v12, v13, v23])),
                "vgg_triplet_min": v_triplet_min,
                "vgg_triplet_max": v_triplet_max,
                "vgg_rule": args.vgg_rule,
                "vgg_band_min": args.vgg_min,
                "vgg_band_max": args.vgg_max,
                "layer": args.layer,
                "mode": "sample" if args.sample else "exhaustive",
            })

            print(f"[COLLECTED {collected_total}/{args.collect_n}] {c} (triplet_min={v_triplet_min:.4f}, triplet_max={v_triplet_max:.4f})")

        summary_rows.append({
            "cat": c,
            "n_imgs": len(imgs),
            "evaluated": evaluated,
            "collected": collected_here,
            "note": "" if collected_here else "no_passing_triplet_found"
        })

    manifest_csv = os.path.join(out_dir, "triplets_manifest.csv")
    summary_csv = os.path.join(out_dir, "category_summary.csv")
    used_cats_txt = os.path.join(out_dir, "used_categories.txt")

    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    Path(used_cats_txt).write_text("\n".join(sorted(used_categories)), encoding="utf-8")

    print(f"[SAVED] {manifest_csv}")
    print(f"[SAVED] {summary_csv}")
    print(f"[SAVED] {used_cats_txt}")
    print(f"[COLLECT_DIR] {collect_dir}")

    if collected_total == 0:
        print("[WARN] No triplets collected. Consider relaxing vgg thresholds, increasing --max_samples_per_cat, or using a different rule.")
    elif collected_total < args.collect_n:
        print(f"[WARN] Collected only {collected_total}/{args.collect_n} unique-category triplets. "
              f"Consider increasing --max_samples_per_cat or relaxing band thresholds.")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
