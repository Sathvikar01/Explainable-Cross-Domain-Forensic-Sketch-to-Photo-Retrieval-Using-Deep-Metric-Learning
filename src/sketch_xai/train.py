from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PairedFaceDataset, discover_pairs, split_pairs
from .losses import batch_accuracy, cross_domain_contrastive_loss
from .metrics import encode_pairs, retrieval_metrics
from .model import ModelConfig, build_model, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sketch-photo retrieval model.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: str,
    temperature: float,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        sketches = batch["sketch"].to(device)
        photos = batch["photo"].to(device)

        optimizer.zero_grad(set_to_none=True)
        sketch_embeddings, photo_embeddings = model(sketches, photos)
        loss, logits = cross_domain_contrastive_loss(
            sketch_embeddings=sketch_embeddings,
            photo_embeddings=photo_embeddings,
            temperature=temperature,
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += batch_accuracy(logits)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    total_batches = max(1, len(loader))
    return {
        "loss": running_loss / total_batches,
        "batch_accuracy": running_accuracy / total_batches,
    }


def evaluate_split(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    encoded = encode_pairs(model=model, loader=loader, device=device)
    metrics = retrieval_metrics(
        sketch_embeddings=encoded["sketch_embeddings"],
        photo_embeddings=encoded["photo_embeddings"],
        sketch_ids=encoded["identities"],
        photo_ids=encoded["identities"],
    )
    return metrics.to_dict()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.data_root)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if len(train_pairs) < 2:
        raise RuntimeError("Training split is too small. Need at least two identities.")

    train_dataset = PairedFaceDataset(train_pairs, image_size=args.image_size, train=True)
    val_dataset = PairedFaceDataset(val_pairs, image_size=args.image_size, train=False)
    test_dataset = PairedFaceDataset(test_pairs, image_size=args.image_size, train=False)
    train_batch_size = min(args.batch_size, len(train_dataset))
    eval_batch_size = min(args.batch_size, len(val_dataset), len(test_dataset))
    if train_batch_size < 2:
        raise RuntimeError("Need at least two training examples per batch for contrastive learning.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        ModelConfig(
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            pretrained=args.pretrained,
        )
    ).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history: list[dict[str, object]] = []
    best_rank1 = -1.0
    best_checkpoint_path = args.output_dir / "best.pt"
    last_checkpoint_path = args.output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            temperature=args.temperature,
        )
        val_metrics = evaluate_split(model=model, loader=val_loader, device=args.device)
        test_metrics = evaluate_split(model=model, loader=test_loader, device=args.device)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(epoch_record)

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            epoch=epoch,
            train_args=vars(args),
            metrics=val_metrics,
        )
        if val_metrics["rank@1"] > best_rank1:
            best_rank1 = val_metrics["rank@1"]
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                epoch=epoch,
                train_args=vars(args),
                metrics=val_metrics,
            )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_batch_acc={train_metrics['batch_accuracy']:.4f} "
            f"val_rank1={val_metrics['rank@1']:.4f} "
            f"test_rank1={test_metrics['rank@1']:.4f}"
        )

    split_manifest = {
        "train": [pair.identity for pair in train_pairs],
        "val": [pair.identity for pair in val_pairs],
        "test": [pair.identity for pair in test_pairs],
    }
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (args.output_dir / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(split_manifest, file, indent=2)


if __name__ == "__main__":
    main()
