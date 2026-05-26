from __future__ import annotations

import numpy as np
from skimage.draw import ellipse
from skimage.transform import rotate


def shepp_logan_base(size: int) -> np.ndarray:
    """Analytic modified Shepp-Logan phantom on a [-1, 1] grid."""
    yy, xx = np.mgrid[:size, :size]
    x = (xx - size / 2) / (size / 2)
    y = (yy - size / 2) / (size / 2)

    ellipses = [
        (1.00, 0.69, 0.92, 0.00, 0.00, 0.0),
        (-0.80, 0.6624, 0.8740, 0.00, -0.0184, 0.0),
        (-0.20, 0.1100, 0.3100, 0.22, 0.0000, -18.0),
        (-0.20, 0.1600, 0.4100, -0.22, 0.0000, 18.0),
        (0.10, 0.2100, 0.2500, 0.00, 0.3500, 0.0),
        (0.10, 0.0460, 0.0460, 0.00, 0.1000, 0.0),
        (0.10, 0.0460, 0.0460, 0.00, -0.1000, 0.0),
        (0.10, 0.0460, 0.0230, -0.08, -0.6050, 0.0),
        (0.10, 0.0230, 0.0230, 0.00, -0.6060, 0.0),
        (0.10, 0.0230, 0.0460, 0.06, -0.6050, 0.0),
    ]

    image = np.zeros((size, size), dtype=np.float32)
    for intensity, axis_x, axis_y, center_x, center_y, angle_deg in ellipses:
        angle = np.deg2rad(angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x0 = x - center_x
        y0 = y - center_y
        xp = x0 * cos_a + y0 * sin_a
        yp = -x0 * sin_a + y0 * cos_a
        mask = (xp / axis_x) ** 2 + (yp / axis_y) ** 2 <= 1.0
        image[mask] += intensity

    image -= float(image.min())
    image /= max(float(image.max()), 1e-6)
    return image.astype(np.float32)


def make_random_phantom(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a simple CT-like phantom with anatomy-like ellipses and small lesions."""
    image = shepp_logan_base(size)

    image = rotate(
        image,
        angle=float(rng.uniform(-18, 18)),
        resize=False,
        preserve_range=True,
        mode="constant",
    ).astype(np.float32)

    image *= float(rng.uniform(0.75, 1.1))
    yy, xx = np.mgrid[:size, :size]
    circle = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= (size * 0.49) ** 2
    image *= circle.astype(np.float32)

    for _ in range(int(rng.integers(3, 8))):
        cy = int(rng.integers(size * 0.22, size * 0.78))
        cx = int(rng.integers(size * 0.22, size * 0.78))
        ry = int(rng.integers(max(3, size // 24), max(4, size // 9)))
        rx = int(rng.integers(max(3, size // 24), max(4, size // 9)))
        angle = float(rng.uniform(0, np.pi))
        rr, cc = ellipse(cy, cx, ry, rx, rotation=angle, shape=image.shape)
        if not circle[rr, cc].all():
            continue
        image[rr, cc] += float(rng.uniform(-0.18, 0.22))

    for _ in range(int(rng.integers(1, 4))):
        cy = int(rng.integers(size * 0.25, size * 0.75))
        cx = int(rng.integers(size * 0.25, size * 0.75))
        radius = int(rng.integers(max(2, size // 48), max(3, size // 28)))
        rr, cc = ellipse(cy, cx, radius, radius, shape=image.shape)
        if circle[rr, cc].all():
            image[rr, cc] += float(rng.uniform(0.12, 0.28))

    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def make_phantom_batch(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    images = [make_random_phantom(size, rng) for _ in range(n)]
    return np.stack(images).astype(np.float32)
