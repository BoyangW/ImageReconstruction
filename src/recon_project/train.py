from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .metrics import image_metrics, summarize_metrics
from .model import build_model
from .visualize import save_reconstruction_grid


class CTDataset(Dataset):
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.fbp = data["fbp"].astype(np.float32)
        self.clean = data["clean"].astype(np.float32)

    def __len__(self) -> int:
        return int(self.clean.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.fbp[index][None, ...])
        y = torch.from_numpy(self.clean[index][None, ...])
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small DL model for sparse-view CT reconstruction.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["unet_tiny", "unet_small", "residual_cnn"], default="unet_small")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    criterion = nn.L1Loss()
    losses = []
    rows = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        losses.append(float(criterion(pred, y).item()))
        pred_np = pred.detach().cpu().numpy()[:, 0]
        y_np = y.detach().cpu().numpy()[:, 0]
        for p, t in zip(pred_np, y_np):
            rows.append(image_metrics(p, t))
    return float(np.mean(losses)), summarize_metrics(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    data_config = args.data / "config.json"
    if data_config.exists():
        shutil.copyfile(data_config, args.run_dir / "data_config.json")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_set = CTDataset(args.data / "train.npz")
    val_set = CTDataset(args.data / "val.npz")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    rows = []
    best_ssim = -1.0
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = l1(pred, y) + 0.25 * mse(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_metrics = validate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "elapsed_sec": time.perf_counter() - start,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.run_dir / "metrics.csv", index=False)
        print(json.dumps(row, indent=2))

        if val_metrics.get("ssim", -1.0) > best_ssim:
            best_ssim = val_metrics["ssim"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                args.run_dir / "best.pt",
            )

    model.eval()
    x, y = next(iter(val_loader))
    with torch.no_grad():
        pred = model(x.to(device)).cpu().numpy()
    save_reconstruction_grid(
        y.numpy()[0, 0],
        x.numpy()[0, 0],
        args.run_dir / "validation_preview.png",
        pred=pred[0, 0],
        title=f"{args.model} validation preview",
    )


if __name__ == "__main__":
    main()
