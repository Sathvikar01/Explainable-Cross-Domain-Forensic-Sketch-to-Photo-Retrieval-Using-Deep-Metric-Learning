from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import CrossDomainSiameseNet


@dataclass
class RetrievalMetrics:
    rank_at_1: float
    rank_at_5: float
    rank_at_10: float
    mrr: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rank@1": self.rank_at_1,
            "rank@5": self.rank_at_5,
            "rank@10": self.rank_at_10,
            "mrr": self.mrr,
        }


def encode_pairs(
    model: CrossDomainSiameseNet,
    loader: DataLoader,
    device: str | torch.device,
) -> dict[str, object]:
    sketch_embeddings: list[torch.Tensor] = []
    photo_embeddings: list[torch.Tensor] = []
    identities: list[str] = []
    sketch_paths: list[str] = []
    photo_paths: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            sketches = batch["sketch"].to(device)
            photos = batch["photo"].to(device)
            sketch_emb, photo_emb = model(sketches, photos)
            sketch_embeddings.append(sketch_emb.cpu())
            photo_embeddings.append(photo_emb.cpu())
            identities.extend(batch["identity"])
            sketch_paths.extend(batch["sketch_path"])
            photo_paths.extend(batch["photo_path"])

    return {
        "sketch_embeddings": torch.cat(sketch_embeddings, dim=0),
        "photo_embeddings": torch.cat(photo_embeddings, dim=0),
        "identities": identities,
        "sketch_paths": sketch_paths,
        "photo_paths": photo_paths,
    }


def retrieval_metrics(
    sketch_embeddings: torch.Tensor,
    photo_embeddings: torch.Tensor,
    sketch_ids: Iterable[str],
    photo_ids: Iterable[str],
) -> RetrievalMetrics:
    sketch_ids = list(sketch_ids)
    photo_ids = list(photo_ids)
    similarities = torch.matmul(sketch_embeddings, photo_embeddings.T).numpy()
    rankings = np.argsort(-similarities, axis=1)

    rank_at_1 = 0.0
    rank_at_5 = 0.0
    rank_at_10 = 0.0
    reciprocal_ranks: list[float] = []

    for row_index, ranked_indices in enumerate(rankings):
        target_identity = sketch_ids[row_index]
        ranked_ids = [photo_ids[index] for index in ranked_indices]
        first_match = ranked_ids.index(target_identity)
        reciprocal_ranks.append(1.0 / float(first_match + 1))
        rank_at_1 += float(first_match < 1)
        rank_at_5 += float(first_match < 5)
        rank_at_10 += float(first_match < 10)

    total = float(len(sketch_ids))
    return RetrievalMetrics(
        rank_at_1=rank_at_1 / total,
        rank_at_5=rank_at_5 / total,
        rank_at_10=rank_at_10 / total,
        mrr=sum(reciprocal_ranks) / total,
    )

