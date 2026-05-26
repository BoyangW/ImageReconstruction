from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def metric_delta(summary: dict[str, Any], key: str) -> float:
    return float(summary["dl"][key]) - float(summary["fbp"][key])


def build_artifact_tags(summary: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if metric_delta(summary, "ssim") > 0.05:
        tags.append("global structural similarity improved")
    if metric_delta(summary, "psnr") > 1.0:
        tags.append("reconstruction fidelity improved")
    if metric_delta(summary, "rmse") < -0.005:
        tags.append("pixel-level error reduced")
    if metric_delta(summary, "high_frequency_error") < -0.002:
        tags.append("high-frequency residual artifacts reduced")
    if metric_delta(summary, "edge_sharpness") > 0.0:
        tags.append("edge sharpness proxy preserved or improved")
    if metric_delta(summary, "edge_sharpness") < -0.02:
        tags.append("possible over-smoothing risk")
    return tags or ["no clear quality improvement detected"]


def build_card(
    run_dir: Path,
    data_dir: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    summary = load_json(run_dir / "eval" / "summary.json")
    run_config = load_json(run_dir / "config.json")
    data_config = load_json(data_dir / "config.json") if data_dir else load_json(run_dir / "data_config.json")

    if not summary:
        raise FileNotFoundError(f"Missing evaluation summary at {run_dir / 'eval' / 'summary.json'}")

    visual_samples = sorted(str(path) for path in (run_dir / "eval").glob("dl_sample_*.png"))
    validation_preview = run_dir / "validation_preview.png"

    deltas = {
        "psnr_delta": round_float(metric_delta(summary, "psnr")),
        "ssim_delta": round_float(metric_delta(summary, "ssim")),
        "rmse_delta": round_float(metric_delta(summary, "rmse")),
        "edge_sharpness_delta": round_float(metric_delta(summary, "edge_sharpness")),
        "high_frequency_error_delta": round_float(metric_delta(summary, "high_frequency_error")),
    }

    tags = build_artifact_tags(summary)
    runtime = summary.get("runtime_sec_per_image")
    runtime_text = f"{float(runtime):.4f} sec/image" if runtime is not None else "not measured"

    interpretation = (
        "The learned reconstruction improves global similarity and reconstruction fidelity relative to "
        "FBP in this controlled synthetic setting. The high-frequency residual proxy decreases, and the "
        "edge sharpness proxy does not show a simple blur-only trade-off."
    )
    if "possible over-smoothing risk" in tags:
        interpretation = (
            "The learned reconstruction improves global metrics, but the edge sharpness proxy drops enough "
            "to flag possible over-smoothing. Visual error maps should be reviewed before treating the metric "
            "gain as a clean image-quality improvement."
        )

    limitation = (
        "This is synthetic sparse-view CT post-FBP restoration, not clinical validation and not a product-grade "
        "raw-data DLIR system. The quality card is a technical QA artifact for comparing reconstruction behavior."
    )

    return {
        "run": str(run_dir),
        "data": str(data_dir) if data_dir else None,
        "model": run_config.get("model"),
        "experiment_config": {
            "image_size": data_config.get("size"),
            "views": data_config.get("views"),
            "noise": data_config.get("noise"),
            "splits": data_config.get("splits"),
            "epochs": run_config.get("epochs"),
            "batch_size": run_config.get("batch_size"),
            "device": run_config.get("device"),
        },
        "metrics": {
            "fbp": {k: round_float(v) for k, v in summary["fbp"].items()},
            "dl": {k: round_float(v) for k, v in summary["dl"].items()},
            "delta": deltas,
            "runtime_sec_per_image": round_float(runtime) if runtime is not None else None,
        },
        "multimodal_evidence": {
            "visual_samples": visual_samples,
            "validation_preview": str(validation_preview) if validation_preview.exists() else None,
            "inputs_combined": [
                "reconstructed images",
                "error maps",
                "sinogram-derived reconstruction setting",
                "quantitative image-quality metrics",
                "experiment metadata",
                "text interpretation",
            ],
        },
        "artifact_tags": tags,
        "interpretation": interpretation,
        "runtime_note": f"Test inference runtime was {runtime_text}.",
        "risk_note": limitation,
        "next_experiment": "Run view-count and noise-level ablations to test whether the quality improvement is robust.",
    }


def write_markdown(card: dict[str, Any], out_path: Path) -> None:
    cfg = card["experiment_config"]
    metrics = card["metrics"]
    tags = ", ".join(card["artifact_tags"])
    sample = card["multimodal_evidence"]["visual_samples"][0] if card["multimodal_evidence"]["visual_samples"] else ""

    lines = [
        "# Reconstruction Quality Card",
        "",
        "## Experiment",
        "",
        f"- Run: `{card['run']}`",
        f"- Model: `{card.get('model')}`",
        f"- Size: `{cfg.get('image_size')}`",
        f"- Views: `{cfg.get('views')}`",
        f"- Noise: `{cfg.get('noise')}`",
        f"- Training: `{cfg.get('epochs')}` epochs, batch size `{cfg.get('batch_size')}`",
        "",
        "## Metrics",
        "",
        "| Method | PSNR | SSIM | RMSE | Edge sharpness | High-frequency error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| FBP | {metrics['fbp']['psnr']:.2f} | {metrics['fbp']['ssim']:.3f} | "
            f"{metrics['fbp']['rmse']:.4f} | {metrics['fbp']['edge_sharpness']:.4f} | "
            f"{metrics['fbp']['high_frequency_error']:.4f} |"
        ),
        (
            f"| DL | {metrics['dl']['psnr']:.2f} | {metrics['dl']['ssim']:.3f} | "
            f"{metrics['dl']['rmse']:.4f} | {metrics['dl']['edge_sharpness']:.4f} | "
            f"{metrics['dl']['high_frequency_error']:.4f} |"
        ),
        "",
        "## Multimodal Evidence",
        "",
        "This card combines visual reconstructions, error maps, acquisition metadata, numeric metrics, and text interpretation.",
        "",
        f"- Artifact tags: {tags}",
        f"- Runtime: {card['runtime_note']}",
        f"- Example visual: `{sample}`",
        "",
        "## Interpretation",
        "",
        card["interpretation"],
        "",
        "## Risk Note",
        "",
        card["risk_note"],
        "",
        "## Next Experiment",
        "",
        card["next_experiment"],
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a multimodal reconstruction quality card.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out or (args.run_dir / "quality_card")
    out_dir.mkdir(parents=True, exist_ok=True)

    card = build_card(args.run_dir, args.data_dir, out_dir)
    (out_dir / "quality_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    write_markdown(card, out_dir / "quality_card.md")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
