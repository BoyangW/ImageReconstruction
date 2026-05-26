from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import image_metrics, summarize_metrics
from .model import build_model
from .train import CTDataset
from .visualize import save_reconstruction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained reconstruction model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-figures", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = CTDataset(args.data / f"{args.split}.npz")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    fbp_rows = []
    dl_rows = []
    runtimes = []
    index = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            start = time.perf_counter()
            pred = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            runtimes.append(elapsed / x.shape[0])

            pred_np = pred.cpu().numpy()[:, 0]
            x_np = x.cpu().numpy()[:, 0]
            y_np = y.numpy()[:, 0]

            for fbp_img, pred_img, target_img in zip(x_np, pred_np, y_np):
                fbp_rows.append({"index": index, **image_metrics(fbp_img, target_img)})
                dl_rows.append({"index": index, **image_metrics(pred_img, target_img)})
                if index < args.max_figures:
                    save_reconstruction_grid(
                        target_img,
                        fbp_img,
                        args.out / f"dl_sample_{index:03d}.png",
                        pred=pred_img,
                        title=f"Sample {index}",
                    )
                index += 1

    pd.DataFrame(fbp_rows).to_csv(args.out / "fbp_metrics.csv", index=False)
    pd.DataFrame(dl_rows).to_csv(args.out / "dl_metrics.csv", index=False)

    summary = {
        "fbp": summarize_metrics([{k: v for k, v in row.items() if k != "index"} for row in fbp_rows]),
        "dl": summarize_metrics([{k: v for k, v in row.items() if k != "index"} for row in dl_rows]),
        "runtime_sec_per_image": float(np.mean(runtimes)),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
