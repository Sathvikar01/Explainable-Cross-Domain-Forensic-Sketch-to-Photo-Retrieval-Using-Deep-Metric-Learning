from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .model import CrossDomainSiameseNet


class SimilarityGradCAM:
    def __init__(self, model: CrossDomainSiameseNet) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer = self.model.get_gradcam_layer()
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_inputs, grad_outputs) -> None:
        self.gradients = grad_outputs[0].detach()

    def remove(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def _compute_cam(self, output_size: tuple[int, int]) -> torch.Tensor:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return cam.detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        modality: str,
        reference_embedding: torch.Tensor,
    ) -> torch.Tensor:
        image_tensor = image_tensor.detach().clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        embedding = self.model.embed_image(image_tensor, modality)
        score = torch.sum(embedding * reference_embedding.detach())
        score.backward()
        return self._compute_cam(output_size=(image_tensor.shape[-2], image_tensor.shape[-1]))


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def blend_heatmap(image: np.ndarray, cam: torch.Tensor, alpha: float = 0.35) -> np.ndarray:
    heatmap = plt.get_cmap("jet")(cam.detach().cpu().squeeze().numpy())[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heatmap, 0.0, 1.0)


def save_explanation_figure(
    sketch_tensor: torch.Tensor,
    photo_tensor: torch.Tensor,
    sketch_cam: torch.Tensor,
    photo_cam: torch.Tensor,
    score: float,
    output_path: str | Path,
    title: str = "Top-1 Match Explanation",
) -> None:
    sketch_image = denormalize_image(sketch_tensor)
    photo_image = denormalize_image(photo_tensor)
    sketch_overlay = blend_heatmap(sketch_image, sketch_cam)
    photo_overlay = blend_heatmap(photo_image, photo_cam)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    figure.suptitle(f"{title} | similarity={score:.4f}")
    axes[0, 0].imshow(sketch_image)
    axes[0, 0].set_title("Query sketch")
    axes[0, 1].imshow(photo_image)
    axes[0, 1].set_title("Matched photo")
    axes[1, 0].imshow(sketch_overlay)
    axes[1, 0].set_title("Sketch Grad-CAM")
    axes[1, 1].imshow(photo_overlay)
    axes[1, 1].set_title("Photo Grad-CAM")

    for axis in axes.ravel():
        axis.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

