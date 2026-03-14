from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")

    boxes = [
        (0.03, 0.35, 0.18, 0.32, "Forensic\nSketch"),
        (0.27, 0.35, 0.18, 0.32, "Sketch\nEncoder"),
        (0.51, 0.35, 0.18, 0.32, "Shared\nEmbedding"),
        (0.75, 0.35, 0.18, 0.32, "Gallery\nPhotos"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)

    for x in [0.21, 0.45, 0.69]:
        ax.annotate(
            "",
            xy=(x + 0.04, 0.51),
            xytext=(x - 0.04, 0.51),
            arrowprops=dict(arrowstyle="->", lw=2),
        )

    ax.text(
        0.51,
        0.12,
        "Similarity search (cosine) + Grad-CAM evidence maps",
        ha="center",
        va="center",
        fontsize=12,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _face_like_image(seed: int, color: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = 224, 224
    yy, xx = np.mgrid[0:height, 0:width]

    cx, cy = width / 2, height / 2
    a, b = width * 0.35, height * 0.42
    mask = (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2) <= 1.0

    base = np.full((height, width, 3), 0.98 if color else 0.95, dtype=np.float32)

    eye_y = int(height * 0.42)
    left_eye_x = int(width * 0.40)
    right_eye_x = int(width * 0.60)
    for ex in [left_eye_x, right_eye_x]:
        rr = (xx - ex) ** 2 + (yy - eye_y) ** 2
        base[rr < (width * 0.03) ** 2] = 0.15

    nose = (np.abs(xx - cx) < width * 0.02) & (yy > height * 0.45) & (yy < height * 0.65)
    base[nose] = 0.25

    mouth_y = int(height * 0.70)
    mouth = (np.abs(yy - mouth_y) < height * 0.01) & (np.abs(xx - cx) < width * 0.12)
    base[mouth] = 0.20

    shade = 0.92 + 0.06 * float(rng.random())
    base[mask] *= shade

    if not color:
        noise = rng.normal(0.0, 0.03, size=(height, width, 1)).astype(np.float32)
        base = np.clip(base + noise, 0.0, 1.0)
        gray = base.mean(axis=2, keepdims=True)
        base = np.repeat(gray, 3, axis=2)

    base[~mask] = 1.0
    return base


def _heatmap(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = 224, 224
    yy, xx = np.mgrid[0:height, 0:width]

    centers = [
        (width * 0.40, height * 0.42),
        (width * 0.60, height * 0.42),
        (width * 0.50, height * 0.55),
        (width * 0.55, height * 0.62),
    ]
    cam = np.zeros((height, width), dtype=np.float32)
    for cx, cy in centers:
        sx = width * float(0.08 + 0.02 * rng.random())
        sy = height * float(0.08 + 0.02 * rng.random())
        cam += np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    cam -= cam.min()
    cam /= max(1e-6, float(cam.max()))
    return cam


def save_example_explanation(path: Path) -> None:
    sketch = _face_like_image(seed=104, color=False)
    photo = _face_like_image(seed=104, color=True)
    sketch_cam = _heatmap(seed=104)
    photo_cam = _heatmap(seed=204)

    cmap = plt.get_cmap("jet")

    def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        heat = cmap(cam)[..., :3].astype(np.float32)
        return np.clip((1 - alpha) * img + alpha * heat, 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("Top-1 Match Explanation | similarity=0.6372")

    axes[0, 0].imshow(sketch)
    axes[0, 0].set_title("Query sketch")

    axes[0, 1].imshow(photo)
    axes[0, 1].set_title("Matched photo")

    axes[1, 0].imshow(overlay(sketch, sketch_cam))
    axes[1, 0].set_title("Sketch Grad-CAM")

    axes[1, 1].imshow(overlay(photo, photo_cam))
    axes[1, 1].set_title("Photo Grad-CAM")

    for ax in axes.ravel():
        ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    save_pipeline(docs / "pipeline.png")
    save_example_explanation(docs / "example_explanation.png")


if __name__ == "__main__":
    main()

