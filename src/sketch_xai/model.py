from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import resnet18

try:
    from torchvision.models import ResNet18_Weights
except ImportError:  # pragma: no cover
    ResNet18_Weights = None


def _build_resnet18(pretrained: bool) -> nn.Module:
    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    return resnet18(pretrained=pretrained)


@dataclass
class ModelConfig:
    embedding_dim: int = 256
    dropout: float = 0.2
    pretrained: bool = False


class CrossDomainSiameseNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        backbone = _build_resnet18(pretrained=pretrained)
        self.sketch_adapter = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.photo_adapter = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.pretrained = pretrained

    def _apply_adapter(self, images: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "sketch":
            return self.sketch_adapter(images)
        if modality == "photo":
            return self.photo_adapter(images)
        raise ValueError(f"Unsupported modality: {modality}")

    def extract_feature_map(self, images: torch.Tensor, modality: str) -> torch.Tensor:
        features = self._apply_adapter(images, modality)
        return self.backbone(features)

    def embed_image(self, images: torch.Tensor, modality: str) -> torch.Tensor:
        feature_map = self.extract_feature_map(images, modality)
        pooled = self.pool(feature_map)
        embeddings = self.projector(pooled)
        return torch.nn.functional.normalize(embeddings, dim=1)

    def forward(
        self,
        sketches: torch.Tensor,
        photos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embed_image(sketches, "sketch"), self.embed_image(photos, "photo")

    def similarity(self, sketches: torch.Tensor, photos: torch.Tensor) -> torch.Tensor:
        sketch_embeddings, photo_embeddings = self(sketches, photos)
        return torch.matmul(sketch_embeddings, photo_embeddings.T)

    def get_gradcam_layer(self) -> nn.Module:
        layer4 = self.backbone[-1]
        return layer4[-1].conv2

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
            pretrained=self.pretrained,
        )


def build_model(config: ModelConfig) -> CrossDomainSiameseNet:
    return CrossDomainSiameseNet(
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        pretrained=config.pretrained,
    )


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def save_checkpoint(
    path: str | Path,
    model: CrossDomainSiameseNet,
    epoch: int,
    train_args: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "model_config": model.model_config().__dict__,
        "train_args": _to_serializable(train_args),
        "metrics": metrics,
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CrossDomainSiameseNet, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    model = build_model(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload
