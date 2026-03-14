from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import FaceImageDataset, PairedFaceDataset, discover_pairs
from .gradcam import SimilarityGradCAM, save_explanation_figure
from .model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sketch-to-photo retrieval with Grad-CAM.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-id", type=str, default=None)
    parser.add_argument("--query-path", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/infer"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _select_query_pair(pairs, query_id: str | None):
    if query_id is None:
        return pairs[0]
    for pair in pairs:
        if pair.identity == query_id:
            return pair
    available = ", ".join(pair.identity for pair in pairs[:10])
    raise ValueError(f"Query id {query_id!r} was not found. Example ids: {available}")


def _build_query_dataset(query_pair, query_path: Path | None, image_size: int):
    if query_path is not None:
        query_path = query_path.resolve()
        if not query_path.exists():
            raise FileNotFoundError(f"Query sketch was not found: {query_path}")
        return FaceImageDataset(
            image_paths=[query_path],
            identities=["external-query"],
            modality="sketch",
            image_size=image_size,
        )
    return PairedFaceDataset([query_pair], image_size=image_size, train=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    pairs = discover_pairs(args.data_root)
    query_pair = _select_query_pair(pairs, args.query_id)

    image_size = int(checkpoint.get("train_args", {}).get("image_size", 224))

    query_dataset = _build_query_dataset(query_pair, args.query_path, image_size=image_size)
    query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    query_batch = next(iter(query_loader))
    query_sketch = query_batch.get("sketch", query_batch.get("image")).to(args.device)

    gallery_paths = [pair.photo_path for pair in pairs]
    gallery_ids = [pair.identity for pair in pairs]
    gallery_dataset = FaceImageDataset(
        image_paths=gallery_paths,
        identities=gallery_ids,
        modality="photo",
        image_size=image_size,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model.eval()
    with torch.no_grad():
        query_embedding = model.embed_image(query_sketch, "sketch")
        gallery_embeddings: list[torch.Tensor] = []
        gallery_records: list[dict[str, str]] = []
        for batch in gallery_loader:
            images = batch["image"].to(args.device)
            embeddings = model.embed_image(images, "photo")
            gallery_embeddings.append(embeddings.cpu())
            gallery_records.extend(
                {"identity": identity, "path": path}
                for identity, path in zip(batch["identity"], batch["path"], strict=True)
            )

    gallery_matrix = torch.cat(gallery_embeddings, dim=0)
    similarities = torch.matmul(query_embedding.cpu(), gallery_matrix.T).squeeze(0)
    top_indices = torch.topk(similarities, k=min(args.topk, len(gallery_records))).indices.tolist()

    rankings_path = args.output_dir / "rankings.csv"
    with rankings_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["rank", "identity", "path", "similarity"])
        writer.writeheader()
        for rank, index in enumerate(top_indices, start=1):
            record = gallery_records[index]
            writer.writerow(
                {
                    "rank": rank,
                    "identity": record["identity"],
                    "path": record["path"],
                    "similarity": f"{float(similarities[index]):.6f}",
                }
            )

    top_record = gallery_records[top_indices[0]]
    top_photo_dataset = FaceImageDataset(
        image_paths=[Path(top_record["path"])],
        identities=[top_record["identity"]],
        modality="photo",
        image_size=image_size,
    )
    top_photo_loader = DataLoader(top_photo_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    top_photo_batch = next(iter(top_photo_loader))
    top_photo = top_photo_batch["image"].to(args.device)

    gradcam = SimilarityGradCAM(model)
    with torch.no_grad():
        top_photo_embedding = model.embed_image(top_photo, "photo")
        query_embedding_for_cam = model.embed_image(query_sketch, "sketch")
    sketch_cam = gradcam.generate(query_sketch, "sketch", top_photo_embedding)
    photo_cam = gradcam.generate(top_photo, "photo", query_embedding_for_cam)
    gradcam.remove()

    explanation_path = args.output_dir / "explanation_top1.png"
    save_explanation_figure(
        sketch_tensor=query_sketch.cpu(),
        photo_tensor=top_photo.cpu(),
        sketch_cam=sketch_cam.cpu(),
        photo_cam=photo_cam.cpu(),
        score=float(similarities[top_indices[0]]),
        output_path=explanation_path,
    )

    summary = {
        "query_identity": query_pair.identity if args.query_path is None else "external-query",
        "top_identity": top_record["identity"],
        "top_similarity": float(similarities[top_indices[0]]),
        "rankings_csv": str(rankings_path),
        "explanation_image": str(explanation_path),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
