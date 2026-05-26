from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import image_metrics, summarize_metrics
from .visualize import save_reconstruction_grid, save_sinogram_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FBP baseline on generated CT data.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-figures", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    data = np.load(args.data / f"{args.split}.npz")
    clean = data["clean"]
    fbp = data["fbp"]
    sinogram = data["sinogram"]

    rows = []
    for idx in range(clean.shape[0]):
        row = {"index": idx, **image_metrics(fbp[idx], clean[idx])}
        rows.append(row)
        if idx < args.max_figures:
            save_reconstruction_grid(
                clean[idx],
                fbp[idx],
                args.out / f"fbp_sample_{idx:03d}.png",
                title=f"FBP baseline sample {idx}",
            )
            save_sinogram_grid(
                clean[idx],
                sinogram[idx],
                fbp[idx],
                args.out / f"sinogram_sample_{idx:03d}.png",
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "fbp_metrics.csv", index=False)

    summary = summarize_metrics([{k: v for k, v in row.items() if k != "index"} for row in rows])
    (args.out / "fbp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

