from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(path.parent.parent for path in root.glob("**/eval/summary.json"))


def build_row(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "eval" / "summary.json")
    run_config = load_json(run_dir / "config.json")
    data_config = load_json(run_dir / "data_config.json")
    if not summary:
        raise FileNotFoundError(f"Missing summary for {run_dir}")

    row: dict[str, Any] = {
        "run": str(run_dir),
        "model": run_config.get("model"),
        "views": data_config.get("views"),
        "noise": data_config.get("noise"),
        "size": data_config.get("size"),
        "train_n": (data_config.get("splits") or {}).get("train"),
        "epochs": run_config.get("epochs"),
        "runtime_sec_per_image": summary.get("runtime_sec_per_image"),
    }
    for metric in ["psnr", "ssim", "rmse", "edge_sharpness", "high_frequency_error"]:
        fbp = float(summary["fbp"][metric])
        dl = float(summary["dl"][metric])
        row[f"fbp_{metric}"] = fbp
        row[f"dl_{metric}"] = dl
        row[f"delta_{metric}"] = dl - fbp
    return row


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    display_cols = [
        "run",
        "views",
        "noise",
        "model",
        "fbp_psnr",
        "dl_psnr",
        "delta_psnr",
        "fbp_ssim",
        "dl_ssim",
        "delta_ssim",
        "runtime_sec_per_image",
    ]
    table = df[display_cols].copy()
    for col in table.select_dtypes(include="number").columns:
        table[col] = table[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    header = "| " + " | ".join(display_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in display_cols) + " |"
        for _, row in table.iterrows()
    ]
    lines = [
        "# Ablation Summary",
        "",
        "This table summarizes FBP vs learned reconstruction across controlled sparse-view CT experiments.",
        "",
        header,
        separator,
        *rows,
        "",
        "Metric interpretation:",
        "",
        "- Higher PSNR and SSIM generally indicate closer similarity to the synthetic target.",
        "- Lower RMSE and high-frequency error indicate lower reconstruction residuals.",
        "- Edge sharpness is a proxy; improvement is encouraging, but visual review is still required.",
        "- Runtime is measured for batch inference on the remote GPU and is not a production benchmark.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reconstruction experiment summaries.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = find_run_dirs(args.root)
    if not run_dirs:
        raise FileNotFoundError(f"No runs with eval/summary.json found under {args.root}")

    rows = [build_row(run_dir) for run_dir in run_dirs]
    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["noise", "views", "model", "run"] if col in df.columns]
    df = df.sort_values(sort_cols, na_position="last")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    write_markdown(df, args.out_md)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
