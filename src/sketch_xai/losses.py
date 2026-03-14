from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_domain_contrastive_loss(
    sketch_embeddings: torch.Tensor,
    photo_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sketch_embeddings.shape != photo_embeddings.shape:
        raise ValueError("Sketch and photo embeddings must have the same shape.")
    if sketch_embeddings.size(0) < 2:
        raise ValueError("Contrastive training requires a batch size of at least 2.")

    logits = torch.matmul(sketch_embeddings, photo_embeddings.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    sketch_loss = F.cross_entropy(logits, labels)
    photo_loss = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (sketch_loss + photo_loss)
    return loss, logits


def batch_accuracy(logits: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    labels = torch.arange(logits.size(0), device=logits.device)
    return (predictions == labels).float().mean().item()

