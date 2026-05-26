from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_view_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    mild = df[df["noise"] == "mild"].sort_values("views")
    if mild.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(mild["views"], mild["fbp_psnr"], marker="o", label="FBP")
    axes[0].plot(mild["views"], mild["dl_psnr"], marker="o", label="DL")
    axes[0].set_title("PSNR vs views")
    axes[0].set_xlabel("Projection views")
    axes[0].set_ylabel("PSNR")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(mild["views"], mild["fbp_ssim"], marker="o", label="FBP")
    axes[1].plot(mild["views"], mild["dl_ssim"], marker="o", label="DL")
    axes[1].set_title("SSIM vs views")
    axes[1].set_xlabel("Projection views")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "view_ablation.png", dpi=180)
    plt.close(fig)


def plot_noise_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["none", "mild", "moderate", "severe"]
    noise = df[df["views"] == 60].copy()
    noise["noise_order"] = noise["noise"].map({name: idx for idx, name in enumerate(order)})
    noise = noise.sort_values("noise_order")
    if noise.empty:
        return

    x = range(len(noise))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar([i - width / 2 for i in x], noise["fbp_psnr"], width=width, label="FBP")
    axes[0].bar([i + width / 2 for i in x], noise["dl_psnr"], width=width, label="DL")
    axes[0].set_title("PSNR vs noise")
    axes[0].set_xticks(list(x), noise["noise"])
    axes[0].set_ylabel("PSNR")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x], noise["fbp_ssim"], width=width, label="FBP")
    axes[1].bar([i + width / 2 for i in x], noise["dl_ssim"], width=width, label="DL")
    axes[1].set_title("SSIM vs noise")
    axes[1].set_xticks(list(x), noise["noise"])
    axes[1].set_ylabel("SSIM")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "noise_ablation.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reconstruction ablation summaries.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)
    plot_view_ablation(df, args.out_dir)
    plot_noise_ablation(df, args.out_dir)


if __name__ == "__main__":
    main()

