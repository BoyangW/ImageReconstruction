# Sparse-View CT Reconstruction

This project is a compact medical image reconstruction prototype. It simulates sparse-view CT acquisition, reconstructs images with filtered back projection (FBP), trains small neural networks for artifact reduction, and evaluates reconstruction quality with quantitative metrics and visual comparisons.

The goal is to demonstrate an end-to-end workflow for an inverse imaging problem:

```text
synthetic CT-like slice -> Radon transform -> sparse/noisy sinogram -> FBP -> learned reconstruction -> evaluation
```

## Features

- Synthetic CT-like phantom generation with controllable image size, view count, and noise level.
- Projection-domain simulation using the Radon transform.
- Classical FBP baseline.
- PyTorch reconstruction models:
  - small residual U-Net
  - tiny residual U-Net
  - residual CNN baseline
- Metrics:
  - PSNR
  - SSIM
  - RMSE
  - edge-sharpness proxy
  - high-frequency residual proxy
  - inference runtime
- Experiment aggregation and ablation plotting.
- Reconstruction quality cards combining images, errors, metrics, settings, and interpretation notes.

## Project Structure

```text
src/recon_project/
  ct_phantom.py        synthetic CT-like phantom generation
  ct_ops.py            Radon transform, noise simulation, FBP reconstruction
  generate_data.py     dataset generation CLI
  model.py             U-Net and residual CNN models
  train.py             training CLI
  evaluate.py          learned-model evaluation CLI
  evaluate_fbp.py      FBP baseline evaluation CLI
  metrics.py           image quality metrics
  visualize.py         reconstruction and sinogram figures
  make_quality_card.py quality-card generation
  aggregate_runs.py    experiment summary aggregation
  plot_ablation.py     ablation plots
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches the CUDA version on the target machine.

## Quick Start

Generate a tiny synthetic dataset:

```bash
PYTHONPATH=src python3 -m recon_project.generate_data \
  --out data/smoke_ct \
  --n-train 4 \
  --n-val 2 \
  --n-test 2 \
  --size 64 \
  --views 45 \
  --noise mild
```

Evaluate the FBP baseline:

```bash
PYTHONPATH=src python3 -m recon_project.evaluate_fbp \
  --data data/smoke_ct \
  --split test \
  --out runs/smoke_fbp
```

Train a small model:

```bash
PYTHONPATH=src python3 -m recon_project.train \
  --data data/smoke_ct \
  --run-dir runs/smoke_unet \
  --model unet_tiny \
  --epochs 2 \
  --batch-size 2 \
  --device cpu
```

Evaluate the learned model:

```bash
PYTHONPATH=src python3 -m recon_project.evaluate \
  --data data/smoke_ct \
  --checkpoint runs/smoke_unet/best.pt \
  --split test \
  --out runs/smoke_unet/eval \
  --device cpu
```

Create a quality card:

```bash
PYTHONPATH=src python3 -m recon_project.make_quality_card \
  --run-dir runs/smoke_unet \
  --data-dir data/smoke_ct
```

## Example Experiment

A representative sparse-view setting used 60 projection views with mild sinogram noise. In that setup, the learned residual U-Net improved SSIM over the FBP baseline on the synthetic test set:

```text
FBP SSIM: 0.611
DL SSIM:  0.946
```

These results are from synthetic phantoms and should be interpreted as a controlled technical demonstration.
