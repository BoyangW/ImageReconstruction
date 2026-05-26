from __future__ import annotations

import numpy as np
from skimage.filters import sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(target, pred, data_range=1.0))


def edge_sharpness(pred: np.ndarray, target: np.ndarray) -> float:
    target_edges = sobel(target.astype(np.float32))
    threshold = np.percentile(target_edges, 85)
    mask = target_edges >= threshold
    if not np.any(mask):
        return 0.0
    pred_edges = sobel(pred.astype(np.float32))
    return float(np.mean(pred_edges[mask]))


def high_frequency_error(pred: np.ndarray, target: np.ndarray) -> float:
    residual = pred.astype(np.float32) - target.astype(np.float32)
    residual_edges = sobel(residual)
    return float(np.std(residual_edges))


def image_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "rmse": rmse(pred, target),
        "edge_sharpness": edge_sharpness(pred, target),
        "high_frequency_error": high_frequency_error(pred, target),
    }


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}

