from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .ct_ops import simulate_fbp
from .ct_phantom import make_random_phantom


def build_split(
    n: int,
    size: int,
    views: int,
    noise: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clean_images = []
    fbp_images = []
    sinograms = []

    for _ in tqdm(range(n), desc=f"generate n={n} views={views} noise={noise}"):
        clean = make_random_phantom(size, rng)
        fbp, sinogram = simulate_fbp(clean, views=views, noise=noise, rng=rng)
        clean_images.append(clean)
        fbp_images.append(fbp)
        sinograms.append(sinogram)

    return (
        np.stack(clean_images).astype(np.float32),
        np.stack(fbp_images).astype(np.float32),
        np.stack(sinograms).astype(np.float32),
    )


def save_split(
    out_dir: Path,
    name: str,
    n: int,
    size: int,
    views: int,
    noise: str,
    seed: int,
) -> None:
    clean, fbp, sinogram = build_split(n, size, views, noise, seed)
    np.savez_compressed(out_dir / f"{name}.npz", clean=clean, fbp=fbp, sinogram=sinogram)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic sparse-view CT reconstruction data.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--noise", choices=["none", "mild", "moderate", "severe"], default="mild")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = {
        "size": args.size,
        "views": args.views,
        "noise": args.noise,
        "seed": args.seed,
        "splits": {"train": args.n_train, "val": args.n_val, "test": args.n_test},
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_split(args.out, "train", args.n_train, args.size, args.views, args.noise, args.seed)
    save_split(args.out, "val", args.n_val, args.size, args.views, args.noise, args.seed + 1)
    save_split(args.out, "test", args.n_test, args.size, args.views, args.noise, args.seed + 2)


if __name__ == "__main__":
    main()

