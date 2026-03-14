from __future__ import annotations

import random
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".tif", ".tiff"}
SKETCH_KEYWORDS = {
    "artist",
    "composite",
    "drawn",
    "forensic",
    "sk",
    "sketch",
    "sketches",
}
PHOTO_KEYWORDS = {
    "face",
    "faces",
    "frontal",
    "gallery",
    "image",
    "images",
    "img",
    "mugshot",
    "photo",
    "photos",
}
MANIFEST_FILENAMES = ("pairs_manifest.csv", "pairs.csv", "manifest.csv")


@dataclass(frozen=True)
class SketchPhotoPair:
    identity: str
    sketch_path: Path
    photo_path: Path


def build_transforms(image_size: int, train: bool) -> T.Compose:
    ops: list[object] = [T.Resize((image_size, image_size))]
    if train:
        ops.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return T.Compose(ops)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _split_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


def _score_keywords(path: Path, keywords: set[str]) -> int:
    path_tokens = _split_tokens(path.as_posix())
    return sum(token in keywords for token in path_tokens)


def infer_modality(path: Path) -> str:
    sketch_score = _score_keywords(path, SKETCH_KEYWORDS)
    photo_score = _score_keywords(path, PHOTO_KEYWORDS)
    if sketch_score > photo_score:
        return "sketch"
    if photo_score > sketch_score:
        return "photo"
    return "unknown"


def _clean_identity_tokens(parts: Iterable[str]) -> list[str]:
    blocked_tokens = SKETCH_KEYWORDS | PHOTO_KEYWORDS
    tokens: list[str] = []
    for part in parts:
        for token in _split_tokens(part):
            if token not in blocked_tokens:
                tokens.append(token)
    return tokens


def build_identity_key(path: Path, data_root: Path) -> str:
    rel_path = path.relative_to(data_root)
    parent_tokens = _clean_identity_tokens(rel_path.parts[:-1])
    stem_tokens = _clean_identity_tokens([path.stem])
    combined = parent_tokens + stem_tokens
    if combined:
        return "-".join(combined)
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def _looks_grayscale(path: Path) -> bool:
    image = load_image(path).resize((32, 32))
    array = np.asarray(image, dtype=np.float32)
    channel_delta = np.mean(np.abs(array[..., 0] - array[..., 1]))
    channel_delta += np.mean(np.abs(array[..., 1] - array[..., 2]))
    return channel_delta < 5.0


def _select_path(paths: list[Path], fallback_label: str) -> Path:
    if len(paths) == 1:
        return paths[0]
    ordered = sorted(paths)
    warnings.warn(
        f"Multiple {fallback_label} candidates were found for one identity. "
        f"Using {ordered[0].name} and ignoring the rest.",
        stacklevel=2,
    )
    return ordered[0]


def _load_manifest(data_root: Path) -> list[SketchPhotoPair] | None:
    manifest_path = next(
        (data_root / filename for filename in MANIFEST_FILENAMES if (data_root / filename).exists()),
        None,
    )
    if manifest_path is None:
        return None

    manifest = pd.read_csv(manifest_path)
    required_columns = {"identity", "sketch_path", "photo_path"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest at {manifest_path} is missing columns: {sorted(missing_columns)}"
        )

    pairs: list[SketchPhotoPair] = []
    for row in manifest.itertuples(index=False):
        sketch_path = (data_root / row.sketch_path).resolve()
        photo_path = (data_root / row.photo_path).resolve()
        if not sketch_path.exists():
            raise FileNotFoundError(f"Sketch file from manifest was not found: {sketch_path}")
        if not photo_path.exists():
            raise FileNotFoundError(f"Photo file from manifest was not found: {photo_path}")
        pairs.append(
            SketchPhotoPair(
                identity=str(row.identity),
                sketch_path=sketch_path,
                photo_path=photo_path,
            )
        )
    return pairs


def discover_pairs(data_root: str | Path) -> list[SketchPhotoPair]:
    data_root = Path(data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    manifest_pairs = _load_manifest(data_root)
    if manifest_pairs:
        return sorted(manifest_pairs, key=lambda pair: pair.identity)

    files = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No image files were found under {data_root}. Supported extensions: "
            f"{sorted(IMAGE_EXTENSIONS)}"
        )

    grouped: dict[str, list[Path]] = defaultdict(list)
    for file_path in files:
        identity = build_identity_key(file_path, data_root)
        grouped[identity].append(file_path)

    pairs: list[SketchPhotoPair] = []
    skipped_identities: list[str] = []
    for identity, paths in grouped.items():
        modalities = {path: infer_modality(path) for path in paths}
        sketches = [path for path, modality in modalities.items() if modality == "sketch"]
        photos = [path for path, modality in modalities.items() if modality == "photo"]
        unknowns = [path for path, modality in modalities.items() if modality == "unknown"]

        grayscale_unknowns = [path for path in unknowns if _looks_grayscale(path)]
        nongray_unknowns = [path for path in unknowns if path not in grayscale_unknowns]

        if not sketches:
            sketches = grayscale_unknowns
        if not photos:
            photos = nongray_unknowns

        if not sketches or not photos:
            skipped_identities.append(identity)
            continue

        pairs.append(
            SketchPhotoPair(
                identity=identity,
                sketch_path=_select_path(sketches, "sketch"),
                photo_path=_select_path(photos, "photo"),
            )
        )

    if not pairs:
        raise RuntimeError(
            "Automatic pair discovery failed. Add a CSV manifest named "
            "'pairs_manifest.csv' inside the dataset root."
        )
    if skipped_identities:
        warnings.warn(
            f"Skipped {len(skipped_identities)} identities because both modalities "
            f"could not be inferred automatically.",
            stacklevel=2,
        )
    return sorted(pairs, key=lambda pair: pair.identity)


def split_pairs(
    pairs: list[SketchPhotoPair],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[SketchPhotoPair], list[SketchPhotoPair], list[SketchPhotoPair]]:
    if len(pairs) < 3:
        raise ValueError("At least three paired identities are required for train/val/test splits.")

    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)

    total = len(pairs)
    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))
    test_count = total - train_count - val_count

    if test_count <= 0:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        else:
            val_count = max(1, val_count - 1)

    if val_count <= 0:
        val_count = 1
        train_count = max(1, train_count - 1)

    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count : train_count + val_count]
    test_pairs = pairs[train_count + val_count :]

    if not val_pairs or not test_pairs:
        raise RuntimeError("Failed to create non-empty validation and test splits.")

    return train_pairs, val_pairs, test_pairs


class PairedFaceDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        pairs: list[SketchPhotoPair],
        image_size: int = 224,
        train: bool = False,
    ) -> None:
        self.pairs = pairs
        self.transform = build_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        sketch = self.transform(load_image(pair.sketch_path))
        photo = self.transform(load_image(pair.photo_path))
        return {
            "identity": pair.identity,
            "sketch": sketch,
            "photo": photo,
            "sketch_path": str(pair.sketch_path),
            "photo_path": str(pair.photo_path),
        }


class FaceImageDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        image_paths: list[Path],
        identities: list[str],
        modality: str,
        image_size: int = 224,
    ) -> None:
        self.image_paths = image_paths
        self.identities = identities
        self.modality = modality
        self.transform = build_transforms(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.image_paths[index]
        return {
            "identity": self.identities[index],
            "image": self.transform(load_image(image_path)),
            "path": str(image_path),
            "modality": self.modality,
        }

