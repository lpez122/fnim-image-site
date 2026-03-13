#!/usr/bin/env python3
"""
Precompute category-level CNN similarity data for the published site.

The site publishes 1-3 copied thumbnails per category, while the similarity
analysis uses all images found in each category folder.

Outputs:
- docs/data/cnn-similarity-data.json
- docs/data/exports/*.xlsx and *.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gensim.downloader as gensim_api
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from timm.data import create_transform, resolve_data_config
from torchvision.models import AlexNet_Weights, VGG16_Weights, alexnet, vgg16

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

SITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = Path("/Users/lukepezanko/Downloads/beh10/images/allimages")
DEFAULT_OUTPUT_PATH = SITE_ROOT / "data" / "cnn-similarity-data.json"
DEFAULT_SAMPLE_ROOT = SITE_ROOT / "assets" / "sample-images"
DEFAULT_EXPORT_ROOT = SITE_ROOT / "data" / "exports"
DEFAULT_GLOVE_MODEL = "glove-wiki-gigaword-50"

ACCENT_PALETTE = [
    "#c75b12",
    "#e98300",
    "#b26e3a",
    "#6f8f48",
    "#3b8ea5",
    "#8c6bb1",
    "#c98c2b",
    "#7b6d63",
    "#2a9d8f",
    "#b85c38",
    "#d16d7b",
    "#5c7cfa",
]

CATEGORY_LABEL_OVERRIDES = {
    "saxaphone": "Saxophone",
    "teddybear": "Teddy Bear",
}

SEMANTIC_TOKEN_OVERRIDES = {
    "birdfeeder": ["bird", "feeder"],
    "bonzai": ["bonsai"],
    "cellphone": ["cell", "phone"],
    "cheesegrater": ["cheese", "grater"],
    "christmasstreeornament": ["christmas", "tree", "ornament"],
    "coatrack": ["coat", "rack"],
    "coffeemaker": ["coffee", "maker"],
    "coffeemug": ["coffee", "mug"],
    "computertower": ["computer", "tower"],
    "doorknob": ["door", "knob"],
    "dryingrack": ["drying", "rack"],
    "fryingpan": ["frying", "pan"],
    "hairbrush": ["hair", "brush"],
    "hairclip": ["hair", "clip"],
    "headphones": ["headphones"],
    "hourglass": ["hourglass"],
    "lawnmower": ["lawn", "mower"],
    "lightbulb": ["light", "bulb"],
    "necktie": ["necktie"],
    "orifan": ["fan"],
    "paintbrush": ["paint", "brush"],
    "pictureframe": ["picture", "frame"],
    "recordplayer": ["record", "player"],
    "remotecontrol": ["remote", "control"],
    "rollerskates": ["roller", "skates"],
    "saltpeppershaker": ["salt", "pepper", "shaker"],
    "saxaphone": ["saxophone"],
    "skiis": ["ski"],
    "snowglobe": ["snow", "globe"],
    "stapleremover": ["stapler", "remover"],
    "storagebin": ["storage", "bin"],
    "teddybear": ["teddy", "bear"],
    "tennisracket": ["tennis", "racket"],
    "toothbrush": ["tooth", "brush"],
    "waterbottle": ["water", "bottle"],
    "watergun": ["water", "gun"],
    "wateringcan": ["water", "can"],
    "windchime": ["wind", "chime"],
}

MODEL_CONFIG = {
    "vgg16": {
        "id": "vgg16",
        "label": "VGG16",
        "year": 2014,
        "family": "torchvision",
        "weights": VGG16_Weights.IMAGENET1K_V1,
        "builder": vgg16,
        "defaults": {
            "selectedLayers": ["block2_conv2", "block4_conv3", "block5_conv3"],
            "activeLayer": "block5_conv3",
        },
        "layers": [
            {
                "id": "block1_conv2",
                "label": "block1_conv2",
                "stage": "early",
                "note": "local edges and repeated texture",
                "descriptor": "Early VGG16 features group categories through local texture and contrast patterns.",
                "module_index": 3,
            },
            {
                "id": "block2_conv2",
                "label": "block2_conv2",
                "stage": "early",
                "note": "contours and simple shape fragments",
                "descriptor": "Contour-level cues begin to separate rigid, curved, and textured object families.",
                "module_index": 8,
            },
            {
                "id": "block3_conv3",
                "label": "block3_conv3",
                "stage": "mid",
                "note": "parts and recurring motifs",
                "descriptor": "Mid-level VGG16 layers emphasize reusable parts and larger local motifs.",
                "module_index": 15,
            },
            {
                "id": "block4_conv3",
                "label": "block4_conv3",
                "stage": "late",
                "note": "semantic part groupings",
                "descriptor": "Later VGG16 layers reflect broader semantic neighborhoods between categories.",
                "module_index": 22,
            },
            {
                "id": "block5_conv2",
                "label": "block5_conv2",
                "stage": "late",
                "note": "stable category structure",
                "descriptor": "Category identity becomes more stable and cross-category gaps are easier to interpret.",
                "module_index": 27,
            },
            {
                "id": "block5_conv3",
                "label": "block5_conv3",
                "stage": "deep",
                "note": "high-level category abstraction",
                "descriptor": "The deepest selected VGG16 layer emphasizes semantic grouping over surface appearance.",
                "module_index": 29,
            },
        ],
    },
    "alexnet": {
        "id": "alexnet",
        "label": "AlexNet",
        "year": 2012,
        "family": "torchvision",
        "weights": AlexNet_Weights.IMAGENET1K_V1,
        "builder": alexnet,
        "defaults": {
            "selectedLayers": ["conv2", "conv4", "conv5"],
            "activeLayer": "conv5",
        },
        "layers": [
            {
                "id": "conv1",
                "label": "conv1",
                "stage": "early",
                "note": "coarse oriented edges",
                "descriptor": "The first AlexNet layer emphasizes broad structure and strong low-level contrast.",
                "module_index": 0,
            },
            {
                "id": "conv2",
                "label": "conv2",
                "stage": "early",
                "note": "sharper contours and motifs",
                "descriptor": "Second-layer AlexNet features reinforce contour and repeated texture groupings.",
                "module_index": 3,
            },
            {
                "id": "conv3",
                "label": "conv3",
                "stage": "mid",
                "note": "local part combinations",
                "descriptor": "Mid-level AlexNet features capture larger part combinations and recurring motifs.",
                "module_index": 6,
            },
            {
                "id": "conv4",
                "label": "conv4",
                "stage": "late",
                "note": "category-relevant motifs",
                "descriptor": "Later AlexNet layers reflect stronger category-level motifs and part arrangements.",
                "module_index": 8,
            },
            {
                "id": "conv5",
                "label": "conv5",
                "stage": "deep",
                "note": "broad category identity",
                "descriptor": "The deepest AlexNet convolutional map emphasizes coarse semantic identity.",
                "module_index": 10,
            },
        ],
    },
    "convnextv2": {
        "id": "convnextv2",
        "label": "ConvNeXt V2",
        "year": 2023,
        "family": "timm",
        "weights_name": "convnextv2_tiny.fcmae_ft_in1k",
        "defaults": {
            "selectedLayers": ["stage2", "stage3", "stage4"],
            "activeLayer": "stage4",
        },
        "layers": [
            {
                "id": "stage1",
                "label": "stage1",
                "stage": "early",
                "note": "local texture organization",
                "descriptor": "Early ConvNeXt V2 features retain local texture structure with modern normalization and downsampling.",
                "feature_index": 0,
                "pool_strategy": "flatten",
            },
            {
                "id": "stage2",
                "label": "stage2",
                "stage": "mid",
                "note": "shape fragments and repeated parts",
                "descriptor": "Mid-level ConvNeXt V2 representations strengthen recurring parts and meso-scale structure.",
                "feature_index": 1,
                "pool_strategy": "flatten",
            },
            {
                "id": "stage3",
                "label": "stage3",
                "stage": "late",
                "note": "larger semantic compositions",
                "descriptor": "Later ConvNeXt V2 stages organize categories through broader semantic part composition.",
                "feature_index": 2,
                "pool_strategy": "flatten",
            },
            {
                "id": "stage4",
                "label": "stage4",
                "stage": "deep",
                "note": "high-level semantic grouping",
                "descriptor": "The deepest selected ConvNeXt V2 stage emphasizes high-level category neighborhoods.",
                "feature_index": 3,
                "pool_strategy": "flatten",
            },
        ],
    },
    "glove": {
        "id": "glove",
        "label": "Semantic (GloVe)",
        "year": 2014,
        "family": "semantic",
        "weights_name": DEFAULT_GLOVE_MODEL,
        "defaults": {
            "selectedLayers": ["semantic_embedding"],
            "activeLayer": "semantic_embedding",
        },
        "layers": [
            {
                "id": "semantic_embedding",
                "label": "semantic_embedding",
                "stage": "semantic",
                "note": "averaged category word embeddings",
                "descriptor": (
                    "Semantic similarity is computed from GloVe word embeddings over the category "
                    "names rather than image features."
                ),
            }
        ],
    },
}


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: List[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return key


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS],
        key=natural_key,
    )


def relative_to_site(path: Path) -> str:
    return path.relative_to(SITE_ROOT).as_posix()


def humanize_category_id(category_id: str) -> str:
    if category_id in CATEGORY_LABEL_OVERRIDES:
        return CATEGORY_LABEL_OVERRIDES[category_id]
    return category_id.replace("_", " ").replace("-", " ").title()


def split_compound_token(
    token: str,
    vocab: set[str],
    cache: Dict[str, Optional[List[str]]],
) -> Optional[List[str]]:
    token = token.lower()
    if token in cache:
        return cache[token]

    override = SEMANTIC_TOKEN_OVERRIDES.get(token)
    if override is not None:
        if not all(part in vocab for part in override):
            missing = [part for part in override if part not in vocab]
            raise KeyError(f"Missing GloVe parts for {token}: {', '.join(missing)}")
        cache[token] = list(override)
        return cache[token]

    if token in vocab:
        cache[token] = [token]
        return cache[token]

    singular = token[:-1] if token.endswith("s") else ""
    if singular and singular in vocab:
        cache[token] = [singular]
        return cache[token]

    best: Optional[List[str]] = None
    for index in range(1, len(token)):
        left = token[:index]
        right = token[index:]
        if left not in vocab:
            continue
        right_tokens = split_compound_token(right, vocab, cache)
        if not right_tokens:
            continue
        candidate = [left, *right_tokens]
        if best is None or len(candidate) < len(best):
            best = candidate

    cache[token] = best
    return best


def normalize_semantic_token(token: str, vocab: set[str], cache: Dict[str, Optional[List[str]]]) -> List[str]:
    result = split_compound_token(token, vocab, cache)
    if result:
        return result
    raise KeyError(f"No GloVe mapping found for token: {token}")


def accent_for_index(index: int) -> str:
    return ACCENT_PALETTE[index % len(ACCENT_PALETTE)]


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available on this machine.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchvisionFeatureExtractor(torch.nn.Module):
    def __init__(self, features: torch.nn.Module, layer_config: Sequence[Dict[str, object]]):
        super().__init__()
        self.features = features.eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.layer_indices = {int(layer["module_index"]): str(layer["id"]) for layer in layer_config}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for module_index, module in enumerate(self.features):
            x = module(x)
            layer_id = self.layer_indices.get(module_index)
            if layer_id is not None:
                outputs[layer_id] = x
        return outputs


class TimmFeatureExtractor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, layer_config: Sequence[Dict[str, object]]):
        super().__init__()
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.feature_indices = {int(layer["feature_index"]): str(layer["id"]) for layer in layer_config}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.model(x)
        return {
            layer_id: features[index]
            for index, layer_id in self.feature_indices.items()
            if index < len(features)
        }


def pool_activations(activations: torch.Tensor, strategy: str = "avg") -> torch.Tensor:
    if strategy == "flatten":
        return activations.flatten(start_dim=1)
    if activations.ndim > 2:
        dims = tuple(range(2, activations.ndim))
        return activations.mean(dim=dims)
    return activations.flatten(start_dim=1)


def load_batch(batch_paths: Sequence[Path], transform, device: torch.device) -> torch.Tensor:
    images = []
    for path in batch_paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images, dim=0).to(device)


def build_category_records(
    image_root: Path,
    sample_root: Path,
) -> Tuple[List[Dict[str, object]], List[Tuple[str, Path]], Dict[str, object]]:
    category_records: List[Dict[str, object]] = []
    image_items: List[Tuple[str, Path]] = []

    category_dirs = sorted([path for path in image_root.iterdir() if path.is_dir()], key=natural_key)
    if not category_dirs:
        raise FileNotFoundError(f"No category folders were found in {image_root}")

    for index, category_dir in enumerate(category_dirs):
        source_images = list_images(category_dir)
        if not source_images:
            continue

        sample_dir = sample_root / category_dir.name
        sample_images = list_images(sample_dir) if sample_dir.exists() else []
        if not sample_images:
            raise FileNotFoundError(
                f"No copied public sample images were found for {category_dir.name}: {sample_dir}"
            )

        source_count = len(source_images)
        public_count = len(sample_images)
        category_id = category_dir.name

        category_records.append(
            {
                "id": category_id,
                "label": humanize_category_id(category_id),
                "accent": accent_for_index(index),
                "note": f"{source_count} image{'s' if source_count != 1 else ''} in the full dataset",
                "sourceCount": source_count,
                "publicCount": public_count,
                "thumbnail": relative_to_site(sample_images[0]),
                "samples": [relative_to_site(path) for path in sample_images],
            }
        )
        image_items.extend((category_id, path) for path in source_images)

    counts = [int(category["sourceCount"]) for category in category_records]
    public_counts = [int(category["publicCount"]) for category in category_records]
    dataset_summary = {
        "categoryCount": len(category_records),
        "totalImages": len(image_items),
        "publicImageCount": int(sum(public_counts)),
        "minImagesPerCategory": min(counts) if counts else 0,
        "maxImagesPerCategory": max(counts) if counts else 0,
        "singleImageCategoryCount": sum(1 for count in counts if count == 1),
        "singleImageCategoryIds": [category["id"] for category in category_records if category["sourceCount"] == 1],
        "description": (
            "The website shows 1-3 copied public thumbnails per category while CNN similarities are computed "
            "from all images in every category folder."
        ),
    }

    return category_records, image_items, dataset_summary


def load_model_runtime(model_spec: Dict[str, object], device: torch.device):
    family = str(model_spec["family"])
    if family == "torchvision":
        model = model_spec["builder"](weights=model_spec["weights"])
        extractor = TorchvisionFeatureExtractor(model.features, model_spec["layers"]).to(device).eval()
        transform = model_spec["weights"].transforms()
        weights_label = str(model_spec["weights"])
        return extractor, transform, weights_label

    if family == "timm":
        out_indices = tuple(int(layer["feature_index"]) for layer in model_spec["layers"])
        model = timm.create_model(
            str(model_spec["weights_name"]),
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )
        extractor = TimmFeatureExtractor(model, model_spec["layers"]).to(device).eval()
        data_config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**data_config)
        weights_label = str(model_spec["weights_name"])
        return extractor, transform, weights_label

    raise ValueError(f"Unsupported model family: {family}")


def compute_similarity_matrices(
    categories: Sequence[Dict[str, object]],
    image_items: Sequence[Tuple[str, Path]],
    model_spec: Dict[str, object],
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]], Dict[str, List[int]], Dict[str, Dict[str, object]], str]:
    category_ids = [str(category["id"]) for category in categories]
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    category_counts = np.array([int(category["sourceCount"]) for category in categories], dtype=np.float32)
    layer_config = model_spec["layers"]

    extractor, transform, weights_label = load_model_runtime(model_spec, device)
    sums: Dict[str, List[Optional[torch.Tensor]]] = {
        str(layer["id"]): [None for _ in category_ids] for layer in layer_config
    }

    total_images = len(image_items)
    print(f"[INFO] {model_spec['label']}: processing {total_images} images on device={device.type}")

    try:
        with torch.inference_mode():
            for start in range(0, total_images, batch_size):
                batch_items = image_items[start : start + batch_size]
                batch_paths = [path for _, path in batch_items]
                batch_categories = [category_id for category_id, _ in batch_items]
                batch_tensor = load_batch(batch_paths, transform, device)
                outputs = extractor(batch_tensor)

                for layer in layer_config:
                    layer_id = str(layer["id"])
                    pooled = pool_activations(
                        outputs[layer_id],
                        strategy=str(layer.get("pool_strategy", "avg")),
                    )
                    embeddings = F.normalize(pooled, p=2, dim=1).cpu()

                    for row_index, category_id in enumerate(batch_categories):
                        bucket = category_index[category_id]
                        current = sums[layer_id][bucket]
                        if current is None:
                            sums[layer_id][bucket] = embeddings[row_index].clone()
                        else:
                            current.add_(embeddings[row_index])

                end = min(start + batch_size, total_images)
                print(f"[INFO] {model_spec['label']}: embedded {end}/{total_images} images")
    finally:
        del extractor
        if device.type == "mps":
            torch.mps.empty_cache()

    matrices: Dict[str, List[List[float]]] = {}
    maps: Dict[str, List[List[float]]] = {}
    matrix_orders: Dict[str, List[int]] = {}
    summaries: Dict[str, Dict[str, object]] = {}

    for layer in layer_config:
        layer_id = str(layer["id"])
        category_means = []
        for category_id, count in zip(category_ids, category_counts):
            category_sum = sums[layer_id][category_index[category_id]]
            if category_sum is None:
                raise RuntimeError(
                    f"Missing embedding sum for model={model_spec['id']}, layer={layer_id}, category={category_id}"
                )
            category_means.append(category_sum / count)

        stacked = torch.stack(category_means, dim=0)
        matrix = torch.clamp(stacked @ stacked.T, -1.0, 1.0).numpy()
        np.fill_diagonal(matrix, 1.0)
        matrices[layer_id] = round_matrix(matrix)
        maps[layer_id] = compute_semantic_map(matrix)
        matrix_orders[layer_id] = compute_matrix_order(matrix)
        summaries[layer_id] = summarize_matrix(matrix, category_ids)
        print(f"[INFO] {model_spec['label']}: computed matrix for {layer_id}")

    return matrices, maps, matrix_orders, summaries, weights_label


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def compute_semantic_similarity_payload(
    categories: Sequence[Dict[str, object]],
    model_spec: Dict[str, object],
    glove_model_name: str,
    keyed_vectors,
    vocab: set[str],
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]], Dict[str, List[int]], Dict[str, Dict[str, object]], str]:
    category_ids = [str(category["id"]) for category in categories]
    category_vectors: List[np.ndarray] = []
    category_notes: Dict[str, str] = {}
    semantic_cache: Dict[str, Optional[List[str]]] = {}
    token_cache: Dict[str, np.ndarray] = {}
    layer_id = str(model_spec["layers"][0]["id"])

    print(f"[INFO] {model_spec['label']}: processing {len(category_ids)} category labels")

    for category in categories:
        category_id = str(category["id"])
        semantic_tokens = normalize_semantic_token(category_id, vocab, semantic_cache)
        cache_key = "|".join(semantic_tokens)
        if cache_key not in token_cache:
            stacked = np.stack([keyed_vectors[token] for token in semantic_tokens], axis=0).astype(np.float32)
            token_cache[cache_key] = l2_normalize(stacked.mean(axis=0))

        category["semanticTokens"] = semantic_tokens
        category_vectors.append(token_cache[cache_key])
        category_notes[category_id] = " + ".join(semantic_tokens)

    matrix = np.stack(category_vectors, axis=0)
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    print(f"[INFO] {model_spec['label']}: computed matrix for {layer_id}")

    matrices = {layer_id: round_matrix(similarity)}
    maps = {layer_id: compute_semantic_map(similarity)}
    matrix_orders = {layer_id: compute_matrix_order(similarity)}
    summaries = {layer_id: summarize_matrix(similarity, category_ids)}
    return matrices, maps, matrix_orders, summaries, glove_model_name


def round_matrix(matrix: np.ndarray) -> List[List[float]]:
    return [[round(float(value), 6) for value in row] for row in matrix]


def compute_semantic_map(matrix: np.ndarray) -> List[List[float]]:
    category_count = matrix.shape[0]
    if category_count == 0:
        return []
    if category_count == 1:
        return [[0.0, 0.0]]

    distances = np.clip(1.0 - matrix, 0.0, None)
    squared = distances ** 2
    identity = np.eye(category_count)
    centering = identity - np.full((category_count, category_count), 1.0 / category_count)
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = np.clip(eigenvalues[order][:2], 0.0, None)
    vectors = eigenvectors[:, order][:, :2]
    scales = np.sqrt(positive)
    coordinates = vectors * scales

    if coordinates.shape[1] < 2:
        padding = np.zeros((coordinates.shape[0], 2 - coordinates.shape[1]), dtype=np.float64)
        coordinates = np.concatenate([coordinates, padding], axis=1)

    return [[round(float(x), 6), round(float(y), 6)] for x, y in coordinates]


def compute_matrix_order(matrix: np.ndarray) -> List[int]:
    category_count = matrix.shape[0]
    if category_count < 3:
        return list(range(category_count))

    distances = np.clip(1.0 - matrix, 0.0, None)
    np.fill_diagonal(distances, 0.0)
    if np.allclose(distances, 0.0):
        return list(range(category_count))

    condensed = squareform(distances, checks=False)
    linkage_matrix = linkage(condensed, method="average", optimal_ordering=True)
    return [int(value) for value in leaves_list(linkage_matrix)]


def summarize_matrix(matrix: np.ndarray, category_ids: Sequence[str]) -> Dict[str, object]:
    upper = np.triu_indices_from(matrix, k=1)
    values = matrix[upper]
    if values.size == 0:
        return {
            "meanOffDiagonal": 1.0,
            "stdOffDiagonal": 0.0,
            "maxPair": [],
            "maxValue": 1.0,
            "minPair": [],
            "minValue": 1.0,
        }

    max_index = int(np.argmax(values))
    min_index = int(np.argmin(values))
    max_pair = [category_ids[int(upper[0][max_index])], category_ids[int(upper[1][max_index])]]
    min_pair = [category_ids[int(upper[0][min_index])], category_ids[int(upper[1][min_index])]]

    return {
        "meanOffDiagonal": round(float(values.mean()), 6),
        "stdOffDiagonal": round(float(values.std()), 6),
        "maxPair": max_pair,
        "maxValue": round(float(values[max_index]), 6),
        "minPair": min_pair,
        "minValue": round(float(values[min_index]), 6),
    }


def build_payload(
    image_root: Path,
    sample_root: Path,
    batch_size: int,
    device: torch.device,
    model_ids: Sequence[str],
    glove_model_name: str,
    glove_vectors=None,
    glove_vocab: Optional[set[str]] = None,
) -> Dict[str, object]:
    categories, image_items, dataset_summary = build_category_records(image_root, sample_root)
    models_payload: Dict[str, object] = {}

    for model_id in model_ids:
        model_spec = MODEL_CONFIG[model_id]
        if model_spec["family"] == "semantic":
            if glove_vectors is None or glove_vocab is None:
                raise RuntimeError("Semantic model requested without loaded GloVe vectors.")
            matrices, maps, matrix_orders, summaries, weights_label = compute_semantic_similarity_payload(
                categories=categories,
                model_spec=model_spec,
                glove_model_name=glove_model_name,
                keyed_vectors=glove_vectors,
                vocab=glove_vocab,
            )
            model_device = "cpu"
        else:
            matrices, maps, matrix_orders, summaries, weights_label = compute_similarity_matrices(
                categories=categories,
                image_items=image_items,
                model_spec=model_spec,
                batch_size=batch_size,
                device=device,
            )
            model_device = device.type

        models_payload[model_id] = {
            "id": model_id,
            "label": model_spec["label"],
            "year": model_spec["year"],
            "family": model_spec["family"],
            "weights": weights_label,
            "device": model_device,
            "defaults": model_spec["defaults"],
            "aggregation": build_aggregation_copy(model_spec),
            "layers": [
                {
                    "id": layer["id"],
                    "label": layer["label"],
                    "stage": layer["stage"],
                    "note": layer["note"],
                    "descriptor": layer["descriptor"],
                    "poolStrategy": layer.get("pool_strategy", "avg"),
                }
                for layer in model_spec["layers"]
            ],
            "matrices": matrices,
            "maps": maps,
            "matrixOrders": matrix_orders,
            "summaries": summaries,
        }

    focus_category_id = categories[0]["id"] if categories else ""
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "imageRoot": str(image_root),
        "sampleRoot": str(sample_root),
        "dataset": dataset_summary,
        "defaults": {
            "modelId": model_ids[0] if model_ids else "",
            "focusCategoryId": focus_category_id,
        },
        "categories": categories,
        "modelsOrder": list(model_ids),
        "models": models_payload,
    }


def build_aggregation_copy(model_spec: Dict[str, object]) -> str:
    if model_spec["family"] == "semantic":
        return (
            "Each category label is mapped to one or more GloVe tokens, averaged into a normalized "
            "word embedding, and category-pair similarity is computed as cosine similarity between "
            "those semantic embeddings."
        )

    if model_spec["id"] == "convnextv2":
        return (
            "All images in each category folder are passed through ConvNeXt V2, the selected stage is flattened "
            "across channels and spatial positions, L2-normalized per image, and category-pair similarity is "
            "computed as the mean cosine across all cross-category image pairs."
        )

    return (
        "All images in each category folder are passed through the pretrained model, pooled at the selected "
        "layer, L2-normalized per image, and category-pair similarity is computed as the mean cosine across "
        "all cross-category image pairs."
    )


def export_tabular_outputs(payload: Dict[str, object], export_root: Path) -> None:
    export_root.mkdir(parents=True, exist_ok=True)
    labels = [str(category["label"]) for category in payload["categories"]]
    summary_rows = []

    for model_id in payload["modelsOrder"]:
        model = payload["models"][model_id]
        workbook_path = export_root / f"{model_id}_similarity_matrices.xlsx"
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for layer in model["layers"]:
                layer_id = layer["id"]
                frame = pd.DataFrame(model["matrices"][layer_id], index=labels, columns=labels)
                frame.to_excel(writer, sheet_name=layer_id[:31])

                csv_path = export_root / f"{model_id}_{layer_id}.csv"
                frame.to_csv(csv_path, index=True)

                summary = model["summaries"][layer_id]
                summary_rows.append(
                    {
                        "model_id": model_id,
                        "model_label": model["label"],
                        "model_year": model["year"],
                        "layer_id": layer_id,
                        "layer_label": layer["label"],
                        "stage": layer["stage"],
                        "note": layer["note"],
                        "mean_off_diagonal": summary["meanOffDiagonal"],
                        "std_off_diagonal": summary["stdOffDiagonal"],
                        "max_pair": " | ".join(summary["maxPair"]),
                        "max_value": summary["maxValue"],
                        "min_pair": " | ".join(summary["minPair"]),
                        "min_value": summary["minValue"],
                        "workbook_file": str(workbook_path),
                        "csv_file": str(csv_path),
                    }
                )

        print(f"[SAVED] {workbook_path}")

    summary_path = export_root / "cnn_similarity_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[SAVED] {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--glove-model", type=str, default=DEFAULT_GLOVE_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONFIG.keys()),
        default=["vgg16", "alexnet", "convnextv2", "glove"],
    )
    args = parser.parse_args()

    image_root = args.image_root.expanduser().resolve()
    sample_root = args.sample_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    export_root = args.export_root.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    glove_vectors = None
    glove_vocab = None
    if "glove" in args.models:
        print(f"[INFO] GloVe: loading {args.glove_model}")
        glove_vectors = gensim_api.load(args.glove_model)
        glove_vocab = set(glove_vectors.key_to_index.keys())

    payload = build_payload(
        image_root=image_root,
        sample_root=sample_root,
        batch_size=args.batch_size,
        device=device,
        model_ids=args.models,
        glove_model_name=args.glove_model,
        glove_vectors=glove_vectors,
        glove_vocab=glove_vocab,
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVED] {output}")
    export_tabular_outputs(payload, export_root=export_root)


if __name__ == "__main__":
    main()
