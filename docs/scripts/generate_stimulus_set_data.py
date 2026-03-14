#!/usr/bin/env python3
"""
Build resized preview assets and precomputed visual/semantic similarity data for the
visual-semantic stimulus set.

The stimulus set is composed of four flat folders:
- within-category / high
- within-category / low
- between-category / high
- between-category / low

Each stimulus group contains exactly three images. Within-category groups map to one
category token. Between-category groups map to three category tokens encoded in the
filename stem, for example:
  almond_rugbyball_soapbar_1.png

Outputs:
- docs/assets/stimulus-set/<condition>/<group>/*.jpg
- docs/data/stimulus-set-data.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gensim.downloader as gensim_api
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from torchvision import transforms
from timm.data import create_transform, resolve_data_config
from torchvision.models import AlexNet_Weights, VGG16_Weights, alexnet, vgg16

try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16 as KerasVGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input as keras_vgg_preprocess_input
    from tensorflow.keras.models import Model as KerasModel
except Exception:
    tf = None
    KerasVGG16 = None
    keras_vgg_preprocess_input = None
    KerasModel = None

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USED_CATEGORIES = Path("/Users/lukepezanko/Downloads/beh10/images/old/used_categories_final.txt")
DEFAULT_OUTPUT_PATH = SITE_ROOT / "data" / "stimulus-set-data.json"
DEFAULT_SAMPLE_ROOT = SITE_ROOT / "assets" / "stimulus-set"
DEFAULT_GLOVE_MODEL = "glove-wiki-gigaword-50"
RELEASE_DOWNLOAD_ROOT = "https://github.com/lpez122/fnim-image-site/releases/latest/download"

CONDITION_CONFIG = {
    "within_high": {
        "root": Path("/Users/lukepezanko/Downloads/beh10/images/withincat/high"),
        "relation": "within",
        "band": "high",
        "label": "Within-category / high",
        "description": "Within-category triplets selected from the high-similarity band.",
        "accent": "#e98300",
        "bundle_id": "within",
    },
    "within_low": {
        "root": Path("/Users/lukepezanko/Downloads/beh10/images/withincat/low"),
        "relation": "within",
        "band": "low",
        "label": "Within-category / low",
        "description": "Within-category triplets selected from the low-similarity band.",
        "accent": "#b26e3a",
        "bundle_id": "within",
    },
    "between_high": {
        "root": Path("/Users/lukepezanko/Downloads/beh10/images/betweencat/high"),
        "relation": "between",
        "band": "high",
        "label": "Between-category / high",
        "description": "Between-category triplets selected from the high-similarity band.",
        "accent": "#3b8ea5",
        "bundle_id": "between",
    },
    "between_low": {
        "root": Path("/Users/lukepezanko/Downloads/beh10/images/betweencat/low"),
        "relation": "between",
        "band": "low",
        "label": "Between-category / low",
        "description": "Between-category triplets selected from the low-similarity band.",
        "accent": "#5c7cfa",
        "bundle_id": "between",
    },
}

BUNDLE_CONFIG = {
    "within": {
        "label": "Within-category stimulus bundle",
        "path": Path("/Users/lukepezanko/Downloads/beh10/images/withincat.zip"),
        "asset_name": "withincat.zip",
    },
    "between": {
        "label": "Between-category stimulus bundle",
        "path": Path("/Users/lukepezanko/Downloads/beh10/images/betweencat.zip"),
        "asset_name": "betweencat.zip",
    },
}

VGG16_CONFIG = {
    "id": "vgg16",
    "label": "VGG16",
    "year": 2014,
    "family": "tensorflow",
    "weights_name": "imagenet",
    "defaults": {
        "selectedLayers": ["block2_conv2", "block4_conv3", "block5_conv2"],
        "activeLayer": "block5_conv2",
    },
    "layers": [
        {
            "id": "block1_conv2",
            "label": "block1_conv2",
            "stage": "early",
            "note": "local edges and repeated texture",
            "descriptor": "Early VGG16 features group stimulus triplets through local texture and contrast patterns.",
            "module_index": 3,
            "pool_strategy": "flatten",
        },
        {
            "id": "block2_conv2",
            "label": "block2_conv2",
            "stage": "early",
            "note": "contours and simple shape fragments",
            "descriptor": "Contour-level cues begin to separate the stimulus groups into broader visual families.",
            "module_index": 8,
            "pool_strategy": "flatten",
        },
        {
            "id": "block3_conv3",
            "label": "block3_conv3",
            "stage": "mid",
            "note": "parts and recurring motifs",
            "descriptor": "Mid-level VGG16 layers emphasize reusable parts and larger local motifs in the triplets.",
            "module_index": 15,
            "pool_strategy": "flatten",
        },
        {
            "id": "block4_conv3",
            "label": "block4_conv3",
            "stage": "late",
            "note": "semantic part groupings",
            "descriptor": "Later VGG16 layers reflect broader visual neighborhoods between the stimulus groups.",
            "module_index": 22,
            "pool_strategy": "flatten",
        },
        {
            "id": "block5_conv2",
            "label": "block5_conv2",
            "stage": "late",
            "note": "stable category structure",
            "descriptor": "Category identity becomes more stable and cross-group separation is easier to read.",
            "module_index": 27,
            "pool_strategy": "flatten",
        },
        {
            "id": "block5_conv3",
            "label": "block5_conv3",
            "stage": "deep",
            "note": "high-level visual abstraction",
            "descriptor": "The deepest selected VGG16 layer emphasizes higher-level category structure over surface detail.",
            "module_index": 29,
            "pool_strategy": "flatten",
        },
    ],
}

ALEXNET_CONFIG = {
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
            "descriptor": "The first AlexNet layer emphasizes broad structure and strong low-level contrast in the stimulus images.",
            "module_index": 0,
        },
        {
            "id": "conv2",
            "label": "conv2",
            "stage": "early",
            "note": "sharper contours and motifs",
            "descriptor": "Second-layer AlexNet features reinforce contour and repeated texture groupings across the stimulus groups.",
            "module_index": 3,
        },
        {
            "id": "conv3",
            "label": "conv3",
            "stage": "mid",
            "note": "local part combinations",
            "descriptor": "Mid-level AlexNet features capture larger part combinations and recurring motifs across the triplets.",
            "module_index": 6,
        },
        {
            "id": "conv4",
            "label": "conv4",
            "stage": "late",
            "note": "category-relevant motifs",
            "descriptor": "Later AlexNet layers reflect stronger stimulus-level motifs and part arrangements.",
            "module_index": 8,
        },
        {
            "id": "conv5",
            "label": "conv5",
            "stage": "deep",
            "note": "broad category identity",
            "descriptor": "The deepest AlexNet convolutional map emphasizes broad category structure across the selected groups.",
            "module_index": 10,
        },
    ],
}

CONVNEXTV2_CONFIG = {
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
            "descriptor": "Mid-level ConvNeXt V2 representations strengthen recurring parts and meso-scale structure in the stimulus images.",
            "feature_index": 1,
            "pool_strategy": "flatten",
        },
        {
            "id": "stage3",
            "label": "stage3",
            "stage": "late",
            "note": "larger part compositions",
            "descriptor": "Later ConvNeXt V2 stages organize the stimulus groups through broader part composition.",
            "feature_index": 2,
            "pool_strategy": "flatten",
        },
        {
            "id": "stage4",
            "label": "stage4",
            "stage": "deep",
            "note": "high-level group structure",
            "descriptor": "The deepest selected ConvNeXt V2 stage emphasizes high-level neighborhoods across the selected groups.",
            "feature_index": 3,
            "pool_strategy": "flatten",
        },
    ],
}

VISUAL_MODEL_CONFIGS = {
    "vgg16": VGG16_CONFIG,
    "alexnet": ALEXNET_CONFIG,
    "convnextv2": CONVNEXTV2_CONFIG,
}

GLOVE_CONFIG = {
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
            "note": "averaged word embeddings",
            "descriptor": (
                "Semantic similarity is computed from GloVe word embeddings, averaging the category words "
                "that define each stimulus triplet."
            ),
        }
    ],
}

SEMANTIC_TOKEN_OVERRIDES = {
    "bonzai": ["bonsai"],
    "coffeemug": ["coffee", "mug"],
    "coatrack": ["coat", "rack"],
    "computertower": ["computer", "tower"],
    "doorknob": ["door", "knob"],
    "fieldhockeystick": ["field", "hockey", "stick"],
    "glassescase": ["glasses", "case"],
    "hairclip": ["hair", "clip"],
    "lightbulb": ["light", "bulb"],
    "necktie": ["necktie"],
    "notepad": ["note", "pad"],
    "orifan": ["fan"],
    "ovenmit": ["oven", "mitt"],
    "paintbrush": ["paint", "brush"],
    "partyhat": ["party", "hat"],
    "pictureframe": ["picture", "frame"],
    "pingpongball": ["ping", "pong", "ball"],
    "recordplayer": ["record", "player"],
    "remotecontrol": ["remote", "control"],
    "rollerskates": ["roller", "skates"],
    "rugbyball": ["rugby", "ball"],
    "saltpeppershaker": ["salt", "pepper", "shaker"],
    "towelrack": ["towel", "rack"],
    "vinylrecord": ["vinyl", "record"],
    "waterbottle": ["water", "bottle"],
    "whiteboard": ["white", "board"],
}

LABEL_OVERRIDES = {
    "allenwrench": "Allen Wrench",
    "beanbag": "Bean Bag",
    "billiardball": "Billiard Ball",
    "bonzai": "Bonsai",
    "broomstick": "Broom Stick",
    "coffeemug": "Coffee Mug",
    "coffeemaker": "Coffee Maker",
    "coatrack": "Coat Rack",
    "computertower": "Computer Tower",
    "cuestick": "Cue Stick",
    "doorknob": "Doorknob",
    "ducttape": "Duct Tape",
    "fieldhockeystick": "Field Hockey Stick",
    "glassescase": "Glasses Case",
    "golfball": "Golf Ball",
    "hairclip": "Hair Clip",
    "hairbrush": "Hair Brush",
    "lightbulb": "Light Bulb",
    "notepad": "Notepad",
    "ovenmit": "Oven Mitt",
    "paintbrush": "Paint Brush",
    "partyhat": "Party Hat",
    "pictureframe": "Picture Frame",
    "pingpongball": "Ping Pong Ball",
    "pinecone": "Pine Cone",
    "recordplayer": "Record Player",
    "remotecontrol": "Remote Control",
    "rollerskates": "Roller Skates",
    "rugbyball": "Rugby Ball",
    "saltpeppershaker": "Salt Pepper Shaker",
    "soapbar": "Soap Bar",
    "sugarcube": "Sugar Cube",
    "towelrack": "Towel Rack",
    "toiletseat": "Toilet Seat",
    "toothbrush": "Tooth Brush",
    "frenchfry": "French Fry",
    "vinylrecord": "Vinyl Record",
    "waterbottle": "Water Bottle",
    "whiteboard": "Whiteboard",
}


@dataclass
class StimulusGroup:
    id: str
    key: str
    label: str
    relation: str
    band: str
    condition_id: str
    condition_label: str
    accent: str
    image_paths: List[Path]
    sample_paths: List[Path]
    category_tokens: List[str]
    semantic_tokens: List[List[str]]


@dataclass
class StimulusImage:
    id: str
    label: str
    note: str
    group_id: str
    group_label: str
    condition_id: str
    condition_label: str
    relation: str
    band: str
    image_index: int
    category_token: str
    semantic_tokens: List[str]
    source_path: Path
    sample_path: Path


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    result: List[object] = []
    for part in parts:
        result.append(int(part) if part.isdigit() else part)
    return result


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS],
        key=natural_key,
    )


def relative_to_site(path: Path) -> str:
    return path.relative_to(SITE_ROOT).as_posix()


def humanize_token(token: str) -> str:
    if token in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[token]
    return token.replace("_", " ").replace("-", " ").title()


def pretty_group_label(tokens: Sequence[str]) -> str:
    return " / ".join(humanize_token(token) for token in tokens)


def safe_group_id(condition_id: str, group_key: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", group_key.lower()).strip("_")
    return f"{condition_id}--{cleaned}"


def safe_image_id(group_id: str, image_index: int) -> str:
    return f"{group_id}--img{image_index + 1}"


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


def load_batch(batch_paths: Sequence[Path], transform, device: torch.device) -> torch.Tensor:
    images = []
    for path in batch_paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images, dim=0).to(device)


def load_keras_vgg_batch(batch_paths: Sequence[Path]) -> np.ndarray:
    if tf is None or keras_vgg_preprocess_input is None:
        raise RuntimeError("TensorFlow/Keras is not available for the VGG16 stimulus pipeline.")

    batch = []
    for path in batch_paths:
        image = tf.keras.preprocessing.image.load_img(str(path), target_size=(224, 224))
        batch.append(tf.keras.preprocessing.image.img_to_array(image))
    x = np.stack(batch, axis=0)
    return keras_vgg_preprocess_input(x)


def pool_activations(activations: torch.Tensor, strategy: str = "avg") -> torch.Tensor:
    if strategy == "flatten":
        return activations.flatten(start_dim=1)
    if activations.ndim > 2:
        dims = tuple(range(2, activations.ndim))
        return activations.mean(dim=dims)
    return activations.flatten(start_dim=1)


def l2_normalize_rows(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def round_matrix(matrix: np.ndarray) -> List[List[float]]:
    return [[round(float(value), 6) for value in row] for row in matrix]


def compute_map(matrix: np.ndarray) -> List[List[float]]:
    item_count = matrix.shape[0]
    if item_count == 0:
        return []
    if item_count == 1:
        return [[0.0, 0.0]]

    distances = np.clip(1.0 - matrix, 0.0, None)
    squared = distances ** 2
    identity = np.eye(item_count)
    centering = identity - np.full((item_count, item_count), 1.0 / item_count)
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = np.clip(eigenvalues[order][:2], 0.0, None)
    vectors = eigenvectors[:, order][:, :2]
    coordinates = vectors * np.sqrt(positive)

    if coordinates.shape[1] < 2:
        padding = np.zeros((coordinates.shape[0], 2 - coordinates.shape[1]), dtype=np.float64)
        coordinates = np.concatenate([coordinates, padding], axis=1)

    return [[round(float(x), 6), round(float(y), 6)] for x, y in coordinates]


def compute_matrix_order(matrix: np.ndarray) -> List[int]:
    item_count = matrix.shape[0]
    if item_count < 3:
        return list(range(item_count))

    distances = np.clip(1.0 - matrix, 0.0, None)
    np.fill_diagonal(distances, 0.0)
    if np.allclose(distances, 0.0):
        return list(range(item_count))

    condensed = squareform(distances, checks=False)
    linkage_matrix = linkage(condensed, method="average", optimal_ordering=True)
    return [int(value) for value in leaves_list(linkage_matrix)]


def summarize_matrix(matrix: np.ndarray, item_ids: Sequence[str]) -> Dict[str, object]:
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
    max_pair = [item_ids[int(upper[0][max_index])], item_ids[int(upper[1][max_index])]]
    min_pair = [item_ids[int(upper[0][min_index])], item_ids[int(upper[1][min_index])]]

    return {
        "meanOffDiagonal": round(float(values.mean()), 6),
        "stdOffDiagonal": round(float(values.std()), 6),
        "maxPair": max_pair,
        "maxValue": round(float(values[max_index]), 6),
        "minPair": min_pair,
        "minValue": round(float(values[min_index]), 6),
    }


def format_size(byte_count: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(byte_count)
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024.0
        unit += 1
    return f"{size:.0f} {units[unit]}" if unit else f"{int(size)} {units[unit]}"


def make_square_preview(source_path: Path, target_path: Path, thumb_size: int) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        thumb = ImageOps.fit(rgb, (thumb_size, thumb_size), method=Image.Resampling.LANCZOS)
        thumb.save(target_path, format="JPEG", quality=88, optimize=True)


def find_group_key(stem: str) -> str:
    match = re.match(r"(.+)_([0-9]+)$", stem)
    return match.group(1) if match else stem


def split_compound_token(token: str, vocab: set[str], cache: Dict[str, Optional[List[str]]]) -> Optional[List[str]]:
    if token in cache:
        return cache[token]

    if token in SEMANTIC_TOKEN_OVERRIDES:
        cache[token] = SEMANTIC_TOKEN_OVERRIDES[token]
        return cache[token]

    if token in vocab:
        cache[token] = [token]
        return cache[token]

    parts = [part for part in re.split(r"[_-]+", token) if part]
    if len(parts) > 1:
        expanded: List[str] = []
        for part in parts:
            result = split_compound_token(part, vocab, cache)
            if not result:
                cache[token] = None
                return None
            expanded.extend(result)
        cache[token] = expanded
        return expanded

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


def build_stimulus_groups(
    sample_root: Path,
    used_categories_file: Path,
    thumb_size: int,
    vocab: set[str],
) -> Tuple[List[StimulusGroup], Dict[str, object]]:
    if sample_root.exists():
        shutil.rmtree(sample_root)
    sample_root.mkdir(parents=True, exist_ok=True)

    semantic_cache: Dict[str, Optional[List[str]]] = {}
    used_categories = [
        line.strip().lower()
        for line in used_categories_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    used_category_set = set(used_categories)
    represented_tokens: set[str] = set()
    groups: List[StimulusGroup] = []
    condition_summaries: List[Dict[str, object]] = []

    for condition_id, config in CONDITION_CONFIG.items():
        root = config["root"]
        if not root.exists():
            raise FileNotFoundError(f"Stimulus folder not found: {root}")

        group_map: Dict[str, List[Path]] = {}
        for image_path in list_images(root):
            group_key = find_group_key(image_path.stem)
            group_map.setdefault(group_key, []).append(image_path)

        ordered_keys = sorted(group_map.keys())
        condition_image_count = 0

        for group_key in ordered_keys:
            image_paths = sorted(group_map[group_key], key=natural_key)
            if not image_paths:
                continue

            if config["relation"] == "within":
                category_tokens = [group_key]
            else:
                category_tokens = group_key.split("_")

            semantic_tokens = [
                normalize_semantic_token(token, vocab, semantic_cache) for token in category_tokens
            ]
            represented_tokens.update(category_tokens)

            group_id = safe_group_id(condition_id, group_key)
            sample_paths: List[Path] = []
            output_dir = sample_root / condition_id / group_key
            for image_path in image_paths:
                target_path = output_dir / f"{image_path.stem}.jpg"
                make_square_preview(image_path, target_path, thumb_size)
                sample_paths.append(target_path)

            groups.append(
                StimulusGroup(
                    id=group_id,
                    key=group_key,
                    label=pretty_group_label(category_tokens),
                    relation=str(config["relation"]),
                    band=str(config["band"]),
                    condition_id=condition_id,
                    condition_label=str(config["label"]),
                    accent=str(config["accent"]),
                    image_paths=image_paths,
                    sample_paths=sample_paths,
                    category_tokens=category_tokens,
                    semantic_tokens=semantic_tokens,
                )
            )
            condition_image_count += len(image_paths)

        condition_summaries.append(
            {
                "id": condition_id,
                "label": config["label"],
                "relation": config["relation"],
                "band": config["band"],
                "description": config["description"],
                "groupCount": len(ordered_keys),
                "imageCount": condition_image_count,
                "accent": config["accent"],
            }
        )

    missing_categories = sorted(used_category_set - represented_tokens)
    dataset_summary = {
        "groupCount": len(groups),
        "totalImages": int(sum(len(group.image_paths) for group in groups)),
        "publicImageCount": int(sum(len(group.sample_paths) for group in groups)),
        "conditionCount": len(CONDITION_CONFIG),
        "usedCategoryCount": len(used_category_set),
        "representedUsedCategoryCount": len(represented_tokens & used_category_set),
        "missingUsedCategoryCount": len(missing_categories),
        "missingUsedCategoryIds": missing_categories,
        "description": (
            "This tab analyzes the visual-semantic stimulus set itself. Visual similarities are computed from the "
            "stimulus images, while semantic similarities are computed from GloVe word embeddings over the category "
            "terms that define each triplet."
        ),
    }

    condition_order = {condition_id: index for index, condition_id in enumerate(CONDITION_CONFIG.keys())}
    groups.sort(key=lambda group: (condition_order[group.condition_id], group.label))
    return groups, {"conditions": condition_summaries, "dataset": dataset_summary}


def infer_token_index(group: StimulusGroup, image_path: Path, fallback_index: int) -> int:
    if group.relation != "between":
        return 0

    match = re.search(r"_([0-9]+)$", image_path.stem)
    if match:
        value = max(1, int(match.group(1))) - 1
        if value < len(group.category_tokens):
            return value

    return min(fallback_index, len(group.category_tokens) - 1)


def build_stimulus_images(groups: Sequence[StimulusGroup]) -> List[StimulusImage]:
    images: List[StimulusImage] = []
    for group in groups:
        for fallback_index, (source_path, sample_path) in enumerate(zip(group.image_paths, group.sample_paths)):
            token_index = infer_token_index(group, source_path, fallback_index)
            category_token = group.category_tokens[token_index]
            semantic_tokens = group.semantic_tokens[token_index]
            label = (
                f"{humanize_token(category_token)} {fallback_index + 1}"
                if group.relation == "within"
                else humanize_token(category_token)
            )
            note = (
                f"{group.condition_label} / {group.label}"
                if group.relation == "between"
                else f"{group.condition_label} / {humanize_token(category_token)}"
            )

            images.append(
                StimulusImage(
                    id=safe_image_id(group.id, fallback_index),
                    label=label,
                    note=note,
                    group_id=group.id,
                    group_label=group.label,
                    condition_id=group.condition_id,
                    condition_label=group.condition_label,
                    relation=group.relation,
                    band=group.band,
                    image_index=fallback_index,
                    category_token=category_token,
                    semantic_tokens=semantic_tokens,
                    source_path=source_path,
                    sample_path=sample_path,
                )
            )

    return images


def load_visual_runtime(model_spec: Dict[str, object], device: torch.device):
    family = str(model_spec["family"])
    if family == "tensorflow":
        if KerasVGG16 is None or KerasModel is None:
            raise RuntimeError(
                "TensorFlow/Keras is required for the stimulus-set VGG16 pipeline but is not installed."
            )

        base = KerasVGG16(weights=str(model_spec["weights_name"]), include_top=False)
        outputs = [base.get_layer(str(layer["id"])).output for layer in model_spec["layers"]]
        extractor = KerasModel(inputs=base.input, outputs=outputs)
        return extractor, None, str(model_spec["weights_name"])

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

    raise ValueError(f"Unsupported visual model family: {family}")


def compute_visual_payload(
    model_spec: Dict[str, object],
    groups: Sequence[StimulusGroup],
    stimulus_images: Sequence[StimulusImage],
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    if model_spec["family"] == "tensorflow":
        return compute_tensorflow_visual_payload(
            model_spec=model_spec,
            groups=groups,
            stimulus_images=stimulus_images,
            batch_size=batch_size,
        )

    extractor, transform, weights_label = load_visual_runtime(model_spec, device)
    group_ids = [group.id for group in groups]
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    counts = np.array([len(group.image_paths) for group in groups], dtype=np.float32)
    image_ids = [image.id for image in stimulus_images]
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    image_items = [(image.group_id, image.id, image.source_path) for image in stimulus_images]

    sums: Dict[str, List[Optional[torch.Tensor]]] = {
        str(layer["id"]): [None for _ in group_ids] for layer in model_spec["layers"]
    }
    image_embeddings: Dict[str, List[Optional[torch.Tensor]]] = {
        str(layer["id"]): [None for _ in image_ids] for layer in model_spec["layers"]
    }

    total_images = len(image_items)
    print(f"[INFO] {model_spec['label']}: processing {total_images} stimulus images on device={device.type}")

    try:
        with torch.inference_mode():
            for start in range(0, total_images, batch_size):
                batch_items = image_items[start : start + batch_size]
                batch_paths = [path for _, _, path in batch_items]
                batch_groups = [group_id for group_id, _, _ in batch_items]
                batch_image_ids = [image_id for _, image_id, _ in batch_items]
                batch_tensor = load_batch(batch_paths, transform, device)
                outputs = extractor(batch_tensor)

                for layer in model_spec["layers"]:
                    layer_id = str(layer["id"])
                    pooled = pool_activations(outputs[layer_id], str(layer.get("pool_strategy", "avg")))
                    embeddings = F.normalize(pooled, p=2, dim=1).cpu()

                    for row_index, group_id in enumerate(batch_groups):
                        bucket = group_index[group_id]
                        current = sums[layer_id][bucket]
                        if current is None:
                            sums[layer_id][bucket] = embeddings[row_index].clone()
                        else:
                            current.add_(embeddings[row_index])
                        image_embeddings[layer_id][image_index[batch_image_ids[row_index]]] = embeddings[row_index].clone()

                end = min(start + batch_size, total_images)
                print(f"[INFO] {model_spec['label']}: embedded {end}/{total_images} images")
    finally:
        del extractor
        if device.type == "mps":
            torch.mps.empty_cache()

    matrices: Dict[str, List[List[float]]] = {}
    image_matrices: Dict[str, List[List[float]]] = {}
    maps: Dict[str, List[List[float]]] = {}
    matrix_orders: Dict[str, List[int]] = {}
    summaries: Dict[str, Dict[str, object]] = {}

    for layer in model_spec["layers"]:
        layer_id = str(layer["id"])
        means: List[torch.Tensor] = []
        for group, count in zip(groups, counts):
            group_sum = sums[layer_id][group_index[group.id]]
            if group_sum is None:
                raise RuntimeError(
                    f"Missing {model_spec['label']} embedding sum for group={group.id}, layer={layer_id}"
                )
            means.append(group_sum / float(count))

        stacked = torch.stack(means, dim=0)
        matrix = torch.clamp(stacked @ stacked.T, -1.0, 1.0).numpy()
        np.fill_diagonal(matrix, 1.0)
        image_stacked = torch.stack([embedding for embedding in image_embeddings[layer_id] if embedding is not None], dim=0)
        image_matrix = torch.clamp(image_stacked @ image_stacked.T, -1.0, 1.0).numpy()
        np.fill_diagonal(image_matrix, 1.0)
        matrices[layer_id] = round_matrix(matrix)
        image_matrices[layer_id] = round_matrix(image_matrix)
        maps[layer_id] = compute_map(matrix)
        matrix_orders[layer_id] = compute_matrix_order(matrix)
        summaries[layer_id] = summarize_matrix(matrix, group_ids)
        print(f"[INFO] {model_spec['label']}: computed matrix for {layer_id}")

    if model_spec["id"] == "convnextv2":
        aggregation = {"visualStrategy": "mean normalized flattened stage embedding"}
    else:
        aggregation = {"visualStrategy": "mean normalized image embedding"}

    return {
        "id": model_spec["id"],
        "label": model_spec["label"],
        "year": model_spec["year"],
        "family": model_spec["family"],
        "weights": weights_label,
        "device": device.type,
        "defaults": model_spec["defaults"],
        "aggregation": aggregation,
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
        "imageMatrices": image_matrices,
        "maps": maps,
        "matrixOrders": matrix_orders,
        "summaries": summaries,
    }


def compute_tensorflow_visual_payload(
    model_spec: Dict[str, object],
    groups: Sequence[StimulusGroup],
    stimulus_images: Sequence[StimulusImage],
    batch_size: int,
) -> Dict[str, object]:
    extractor, _, weights_label = load_visual_runtime(model_spec, torch.device("cpu"))
    group_ids = [group.id for group in groups]
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    counts = np.array([len(group.image_paths) for group in groups], dtype=np.float32)
    image_ids = [image.id for image in stimulus_images]
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    image_items = [(image.group_id, image.id, image.source_path) for image in stimulus_images]

    sums: Dict[str, List[Optional[np.ndarray]]] = {
        str(layer["id"]): [None for _ in group_ids] for layer in model_spec["layers"]
    }
    image_embeddings: Dict[str, List[Optional[np.ndarray]]] = {
        str(layer["id"]): [None for _ in image_ids] for layer in model_spec["layers"]
    }

    total_images = len(image_items)
    print(f"[INFO] {model_spec['label']}: processing {total_images} stimulus images with TensorFlow/Keras")

    for start in range(0, total_images, batch_size):
        batch_items = image_items[start : start + batch_size]
        batch_paths = [path for _, _, path in batch_items]
        batch_groups = [group_id for group_id, _, _ in batch_items]
        batch_image_ids = [image_id for _, image_id, _ in batch_items]
        batch_tensor = load_keras_vgg_batch(batch_paths)
        outputs = extractor.predict(batch_tensor, verbose=0)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for layer, output in zip(model_spec["layers"], outputs):
            layer_id = str(layer["id"])
            flattened = output.reshape((output.shape[0], -1)).astype(np.float32)
            embeddings = l2_normalize_rows(flattened)

            for row_index, group_id in enumerate(batch_groups):
                bucket = group_index[group_id]
                current = sums[layer_id][bucket]
                if current is None:
                    sums[layer_id][bucket] = embeddings[row_index].copy()
                else:
                    current += embeddings[row_index]
                image_embeddings[layer_id][image_index[batch_image_ids[row_index]]] = embeddings[row_index].copy()

        end = min(start + batch_size, total_images)
        print(f"[INFO] {model_spec['label']}: embedded {end}/{total_images} images")

    matrices: Dict[str, List[List[float]]] = {}
    image_matrices: Dict[str, List[List[float]]] = {}
    maps: Dict[str, List[List[float]]] = {}
    matrix_orders: Dict[str, List[int]] = {}
    summaries: Dict[str, Dict[str, object]] = {}

    for layer in model_spec["layers"]:
        layer_id = str(layer["id"])
        means: List[np.ndarray] = []
        for group, count in zip(groups, counts):
            group_sum = sums[layer_id][group_index[group.id]]
            if group_sum is None:
                raise RuntimeError(
                    f"Missing {model_spec['label']} embedding sum for group={group.id}, layer={layer_id}"
                )
            means.append(group_sum / float(count))

        stacked = np.stack(means, axis=0).astype(np.float32)
        matrix = np.clip(stacked @ stacked.T, -1.0, 1.0)
        np.fill_diagonal(matrix, 1.0)
        image_stacked = np.stack(
            [embedding for embedding in image_embeddings[layer_id] if embedding is not None], axis=0
        )
        image_matrix = np.clip(image_stacked @ image_stacked.T, -1.0, 1.0)
        np.fill_diagonal(image_matrix, 1.0)
        matrices[layer_id] = round_matrix(matrix)
        image_matrices[layer_id] = round_matrix(image_matrix)
        maps[layer_id] = compute_map(matrix)
        matrix_orders[layer_id] = compute_matrix_order(matrix)
        summaries[layer_id] = summarize_matrix(matrix, group_ids)
        print(f"[INFO] {model_spec['label']}: computed matrix for {layer_id}")

    return {
        "id": model_spec["id"],
        "label": model_spec["label"],
        "year": model_spec["year"],
        "family": model_spec["family"],
        "weights": weights_label,
        "device": "tensorflow",
        "defaults": model_spec["defaults"],
        "aggregation": {"visualStrategy": "flattened TensorFlow/Keras VGG16 feature map"},
        "layers": [
            {
                "id": layer["id"],
                "label": layer["label"],
                "stage": layer["stage"],
                "note": layer["note"],
                "descriptor": layer["descriptor"],
                "poolStrategy": layer.get("pool_strategy", "flatten"),
            }
            for layer in model_spec["layers"]
        ],
        "matrices": matrices,
        "imageMatrices": image_matrices,
        "maps": maps,
        "matrixOrders": matrix_orders,
        "summaries": summaries,
    }


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def build_glove_payload(
    groups: Sequence[StimulusGroup],
    stimulus_images: Sequence[StimulusImage],
    glove_model_name: str,
    keyed_vectors,
) -> Dict[str, object]:
    token_cache: Dict[str, np.ndarray] = {}
    group_vectors: List[np.ndarray] = []
    image_vectors: List[np.ndarray] = []
    semantic_notes: Dict[str, str] = {}

    for group in groups:
        category_vectors: List[np.ndarray] = []
        readable_parts: List[str] = []
        for raw_token, semantic_parts in zip(group.category_tokens, group.semantic_tokens):
            cache_key = "|".join(semantic_parts)
            if cache_key not in token_cache:
                stacked = np.stack([keyed_vectors[part] for part in semantic_parts], axis=0).astype(np.float32)
                token_cache[cache_key] = l2_normalize(stacked.mean(axis=0))
            category_vectors.append(token_cache[cache_key])
            readable_parts.append(f"{humanize_token(raw_token)} -> {' + '.join(semantic_parts)}")

        group_vector = l2_normalize(np.stack(category_vectors, axis=0).mean(axis=0))
        group_vectors.append(group_vector)
        semantic_notes[group.id] = "; ".join(readable_parts)

    for image in stimulus_images:
        cache_key = "|".join(image.semantic_tokens)
        if cache_key not in token_cache:
            stacked = np.stack([keyed_vectors[part] for part in image.semantic_tokens], axis=0).astype(np.float32)
            token_cache[cache_key] = l2_normalize(stacked.mean(axis=0))
        image_vectors.append(token_cache[cache_key])

    matrix = np.stack(group_vectors, axis=0)
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    image_matrix = np.stack(image_vectors, axis=0)
    image_similarity = np.clip(image_matrix @ image_matrix.T, -1.0, 1.0)
    np.fill_diagonal(image_similarity, 1.0)
    group_ids = [group.id for group in groups]

    return {
        "id": GLOVE_CONFIG["id"],
        "label": GLOVE_CONFIG["label"],
        "year": GLOVE_CONFIG["year"],
        "family": GLOVE_CONFIG["family"],
        "weights": glove_model_name,
        "device": "cpu",
        "defaults": GLOVE_CONFIG["defaults"],
        "aggregation": {"semanticStrategy": "mean normalized category-word embedding"},
        "layers": [
            {
                "id": "semantic_embedding",
                "label": "semantic_embedding",
                "stage": "semantic",
                "note": "averaged category word embeddings",
                "descriptor": GLOVE_CONFIG["layers"][0]["descriptor"],
                "poolStrategy": "token_mean",
            }
        ],
        "matrices": {"semantic_embedding": round_matrix(similarity)},
        "imageMatrices": {"semantic_embedding": round_matrix(image_similarity)},
        "maps": {"semantic_embedding": compute_map(similarity)},
        "matrixOrders": {"semantic_embedding": compute_matrix_order(similarity)},
        "summaries": {"semantic_embedding": summarize_matrix(similarity, group_ids)},
        "semanticNotes": semantic_notes,
    }


def build_download_payload() -> List[Dict[str, object]]:
    downloads: List[Dict[str, object]] = []
    for bundle_id, config in BUNDLE_CONFIG.items():
        bundle_path = config["path"]
        size_bytes = bundle_path.stat().st_size if bundle_path.exists() else 0
        asset_name = str(config["asset_name"])
        downloads.append(
            {
                "id": bundle_id,
                "label": config["label"],
                "sizeBytes": size_bytes,
                "sizeLabel": format_size(size_bytes) if size_bytes else "Unavailable",
                "localPath": str(bundle_path),
                "downloadUrl": f"{RELEASE_DOWNLOAD_ROOT}/{asset_name}",
                "releasePageUrl": "https://github.com/lpez122/fnim-image-site/releases/latest",
                "assetName": asset_name,
                "note": (
                    "Hosted through GitHub Releases so the large ZIP bundle stays outside the Pages build. "
                    f"Upload this asset to the latest release as {asset_name}."
                ),
            }
        )
    return downloads


def build_payload(
    groups: Sequence[StimulusGroup],
    stimulus_images: Sequence[StimulusImage],
    metadata: Dict[str, object],
    sample_root: Path,
    used_categories_file: Path,
    glove_model_name: str,
    glove_vectors,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    categories_payload = []
    for group in groups:
        semantic_preview = ", ".join(
            " + ".join(parts) for parts in group.semantic_tokens
        )
        categories_payload.append(
            {
                "id": group.id,
                "groupKey": group.key,
                "label": group.label,
                "accent": group.accent,
                "note": f"{group.condition_label} / semantic tokens: {semantic_preview}",
                "sourceCount": len(group.image_paths),
                "publicCount": len(group.sample_paths),
                "thumbnail": relative_to_site(group.sample_paths[0]),
                "samples": [relative_to_site(path) for path in group.sample_paths],
                "conditionId": group.condition_id,
                "conditionLabel": group.condition_label,
                "relation": group.relation,
                "band": group.band,
                "categoryTokens": group.category_tokens,
                "semanticTokens": group.semantic_tokens,
            }
        )

    models_payload = {
        model_id: compute_visual_payload(
            model_spec=model_spec,
            groups=groups,
            stimulus_images=stimulus_images,
            batch_size=batch_size,
            device=device,
        )
        for model_id, model_spec in VISUAL_MODEL_CONFIGS.items()
    }
    models_payload.update({
        "glove": build_glove_payload(
            groups,
            stimulus_images=stimulus_images,
            glove_model_name=glove_model_name,
            keyed_vectors=glove_vectors,
        ),
    })

    focus_group_id = groups[0].id if groups else ""
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "usedCategoriesFile": str(used_categories_file),
        "sampleRoot": str(sample_root),
        "downloads": build_download_payload(),
        "conditions": metadata["conditions"],
        "dataset": metadata["dataset"],
        "defaults": {
            "modelId": "vgg16",
            "focusCategoryId": focus_group_id,
        },
        "images": [
            {
                "id": image.id,
                "label": image.label,
                "note": image.note,
                "groupId": image.group_id,
                "groupLabel": image.group_label,
                "conditionId": image.condition_id,
                "conditionLabel": image.condition_label,
                "relation": image.relation,
                "band": image.band,
                "imageIndex": image.image_index,
                "categoryToken": image.category_token,
                "semanticTokens": image.semantic_tokens,
                "thumbnail": relative_to_site(image.sample_path),
            }
            for image in stimulus_images
        ],
        "categories": categories_payload,
        "modelsOrder": ["vgg16", "alexnet", "convnextv2", "glove"],
        "models": models_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--used-categories-file", type=Path, default=DEFAULT_USED_CATEGORIES)
    parser.add_argument("--glove-model", type=str, default=DEFAULT_GLOVE_MODEL)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    output_path = args.output_path.expanduser().resolve()
    sample_root = args.sample_root.expanduser().resolve()
    used_categories_file = args.used_categories_file.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] GloVe: loading {args.glove_model}")
    glove_vectors = gensim_api.load(args.glove_model)
    glove_vocab = set(glove_vectors.key_to_index.keys())
    groups, metadata = build_stimulus_groups(
        sample_root=sample_root,
        used_categories_file=used_categories_file,
        thumb_size=args.thumb_size,
        vocab=glove_vocab,
    )
    stimulus_images = build_stimulus_images(groups)
    device = choose_device(args.device)
    payload = build_payload(
        groups=groups,
        stimulus_images=stimulus_images,
        metadata=metadata,
        sample_root=sample_root,
        used_categories_file=used_categories_file,
        glove_model_name=args.glove_model,
        glove_vectors=glove_vectors,
        batch_size=args.batch_size,
        device=device,
    )

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
