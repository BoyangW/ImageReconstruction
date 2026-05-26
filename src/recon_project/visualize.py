from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_reconstruction_grid(
    target: np.ndarray,
    fbp: np.ndarray,
    out_path: str | Path,
    pred: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, np.ndarray, str]] = [
        ("Target", target, "gray"),
        ("FBP", fbp, "gray"),
        ("|FBP - target|", np.abs(fbp - target), "magma"),
    ]
    if pred is not None:
        panels.insert(2, ("DL output", pred, "gray"))
        panels.append(("|DL - target|", np.abs(pred - target), "magma"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, image, cmap) in zip(axes, panels):
        vmax = 1.0 if cmap == "gray" else max(float(np.percentile(image, 99)), 1e-3)
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_sinogram_grid(
    target: np.ndarray,
    sinogram: np.ndarray,
    fbp: np.ndarray,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    axes[0].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Target")
    axes[1].imshow(sinogram, cmap="gray", aspect="auto")
    axes[1].set_title("Noisy sinogram")
    axes[2].imshow(fbp, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("FBP reconstruction")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

