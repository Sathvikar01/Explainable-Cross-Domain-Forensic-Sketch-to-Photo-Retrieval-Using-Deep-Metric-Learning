from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .data import PairedFaceDataset, discover_pairs, split_pairs
from .metrics import encode_pairs, retrieval_metrics
from .model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained sketch-photo retrieval model.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    pairs = discover_pairs(args.data_root)

    if args.split == "all":
        selected_pairs = pairs
    else:
        train_args = checkpoint.get("train_args", {})
        train_pairs, val_pairs, test_pairs = split_pairs(
            pairs,
            train_ratio=float(train_args.get("train_ratio", 0.7)),
            val_ratio=float(train_args.get("val_ratio", 0.15)),
            seed=int(train_args.get("seed", 42)),
        )
        selected_pairs = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs,
        }[args.split]

    image_size = int(checkpoint.get("train_args", {}).get("image_size", 224))
    dataset = PairedFaceDataset(selected_pairs, image_size=image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    encoded = encode_pairs(model=model, loader=loader, device=args.device)
    metrics = retrieval_metrics(
        sketch_embeddings=encoded["sketch_embeddings"],
        photo_embeddings=encoded["photo_embeddings"],
        sketch_ids=encoded["identities"],
        photo_ids=encoded["identities"],
    )

    metrics_path = args.output_dir / f"{args.split}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics.to_dict(), file, indent=2)

    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()

