from __future__ import annotations

import numpy as np
from skimage.transform import iradon, radon


NOISE_LEVELS = {
    "none": 0.0,
    "mild": 0.015,
    "moderate": 0.035,
    "severe": 0.065,
}


def projection_angles(views: int) -> np.ndarray:
    return np.linspace(0.0, 180.0, int(views), endpoint=False)


def forward_project(image: np.ndarray, views: int) -> tuple[np.ndarray, np.ndarray]:
    theta = projection_angles(views)
    sinogram = radon(image, theta=theta, circle=True).astype(np.float32)
    return sinogram, theta


def add_sinogram_noise(
    sinogram: np.ndarray,
    noise: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise not in NOISE_LEVELS:
        raise ValueError(f"Unknown noise level {noise!r}; choose from {sorted(NOISE_LEVELS)}")
    sigma = NOISE_LEVELS[noise]
    if sigma == 0.0:
        return sinogram.astype(np.float32)
    scale = max(float(np.percentile(np.abs(sinogram), 99)), 1.0)
    noisy = sinogram + rng.normal(0.0, sigma * scale, size=sinogram.shape)
    return noisy.astype(np.float32)


def fbp_reconstruct(sinogram: np.ndarray, theta: np.ndarray, size: int) -> np.ndarray:
    recon = iradon(
        sinogram,
        theta=theta,
        filter_name="ramp",
        circle=True,
        output_size=size,
    )
    return np.clip(recon, 0.0, 1.0).astype(np.float32)


def simulate_fbp(
    image: np.ndarray,
    views: int,
    noise: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sinogram, theta = forward_project(image, views)
    noisy = add_sinogram_noise(sinogram, noise, rng)
    recon = fbp_reconstruct(noisy, theta, image.shape[-1])
    return recon, noisy

