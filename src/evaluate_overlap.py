"""Evaluation 3: Overlap percentage sweep with Hann-weighted blending."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import experiment_utils as eu
from evaluate_blending import list_dataset_images, load_rgb_chw


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation 3: Overlap sweep")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "DIV2K_valid_HR")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "experiments" / "exp3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" else "cpu" if args.device == "cpu" else eu.DEVICE)
    eu.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = list_dataset_images(args.dataset_dir, args.limit)
    dncnn_models = {k: eu.load_dncnn(k, device=device) for k in eu.DNCNN_CHECKPOINTS}

    overlaps = [
        ("0%", 0, 128),
        ("12.5%", 16, 112),
        ("25%", 32, 96),
        ("37.5%", 48, 80),
        ("50%", 64, 64),
    ]

    rows = []
    for img_idx, img_path in enumerate(images):
        clean = load_rgb_chw(img_path)
        _, h, w = clean.shape
        for kind, severities in eu.NOISE_LEVELS.items():
            model = dncnn_models[kind]
            for sev_idx, sev_name in enumerate(severities):
                pair_seed = args.seed + img_idx * 100 + sev_idx * 10
                noisy = eu.add_noise(clean, kind, sev_name, seed=pair_seed)
                base_psnr = None
                for ov_label, ov_px, stride_px in overlaps:
                    n_patches = len(eu.patch_coordinates(h, w, args.patch_size, ov_px))
                    blend_mode = "none" if ov_px == 0 else "hann"
                    restored, elapsed = eu.timed_inference(
                        eu.run_patch_inference,
                        noisy,
                        model,
                        patch_size=args.patch_size,
                        overlap=ov_px,
                        blend=blend_mode,
                        min_weight=0.05,
                        device=device,
                    )
                    psnr_val, ssim_val = eu.calculate_psnr_ssim(clean, restored)
                    if base_psnr is None:
                        base_psnr = psnr_val
                    pct_gain = ((psnr_val - base_psnr) / abs(base_psnr)) * 100.0 if base_psnr != 0 else 0.0
                    rows.append({
                        "image": img_path.name,
                        "noise_type": kind,
                        "severity": sev_name,
                        "overlap_pct": ov_label,
                        "overlap_px": ov_px,
                        "nominal_stride": stride_px,
                        "num_patches": n_patches,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "psnr_pct_gain_vs_0pct": pct_gain,
                        "inference_sec": elapsed,
                    })

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "exp3_overlap_sweep.csv"
    df.to_csv(csv_path, index=False)

    summary = df.groupby(["overlap_px", "overlap_pct"], as_index=False)[["psnr", "ssim", "inference_sec", "num_patches"]].mean().sort_values("overlap_px")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["overlap_pct"], summary["psnr"], marker="o")
    ax.set_title("PSNR vs Overlap")
    ax.set_xlabel("Overlap")
    ax.set_ylabel("Mean PSNR (dB)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "exp3_overlap_psnr.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["overlap_pct"], summary["ssim"], marker="o")
    ax.set_title("SSIM vs Overlap")
    ax.set_xlabel("Overlap")
    ax.set_ylabel("Mean SSIM")
    fig.tight_layout()
    fig.savefig(args.output_dir / "exp3_overlap_ssim.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["overlap_pct"], summary["inference_sec"], marker="o")
    ax.set_title("Inference Time vs Overlap")
    ax.set_xlabel("Overlap")
    ax.set_ylabel("Mean Time (s)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "exp3_overlap_time.png", dpi=150)
    plt.close(fig)

    print(f"Saved {csv_path} and plots ({len(df)} rows)")


if __name__ == "__main__":
    main()