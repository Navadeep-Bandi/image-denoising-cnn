"""Evaluation 2: Patch-boundary vs. interior PSNR and SSIM."""
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
    parser = argparse.ArgumentParser(description="Evaluation 2: Boundary vs interior metrics")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "DIV2K_valid_HR")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "experiments" / "exp2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--boundary-width", type=int, default=8)
    parser.add_argument("--blend", choices=["hann", "uniform", "none"], default="hann")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" else "cpu" if args.device == "cpu" else eu.DEVICE)
    eu.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = list_dataset_images(args.dataset_dir, args.limit)
    dncnn_models = {k: eu.load_dncnn(k, device=device) for k in eu.DNCNN_CHECKPOINTS}

    rows = []
    vis_saved = False

    for img_idx, img_path in enumerate(images):
        clean = load_rgb_chw(img_path)
        _, h, w = clean.shape
        eff_overlap = 0 if args.blend == "none" else args.overlap
        b_mask, i_mask = eu.make_boundary_interior_masks(
            h, w, patch_size=args.patch_size, overlap=eff_overlap, boundary_width=args.boundary_width
        )
        for kind, severities in eu.NOISE_LEVELS.items():
            model = dncnn_models[kind]
            for sev_idx, sev_name in enumerate(severities):
                pair_seed = args.seed + img_idx * 100 + sev_idx * 10
                noisy = eu.add_noise(clean, kind, sev_name, seed=pair_seed)
                restored, elapsed = eu.timed_inference(
                    eu.run_patch_inference,
                    noisy,
                    model,
                    patch_size=args.patch_size,
                    overlap=eff_overlap,
                    blend=args.blend,
                    min_weight=0.05,
                    device=device,
                )
                b_psnr, b_ssim = eu.calculate_masked_psnr_ssim(clean, restored, b_mask)
                i_psnr, i_ssim = eu.calculate_masked_psnr_ssim(clean, restored, i_mask)
                rows.append({
                    "image": img_path.name,
                    "noise_type": kind,
                    "severity": sev_name,
                    "blend": args.blend,
                    "overlap": eff_overlap,
                    "psnr_boundary": b_psnr,
                    "psnr_interior": i_psnr,
                    "delta_psnr_interior_minus_boundary": i_psnr - b_psnr,
                    "ssim_boundary": b_ssim,
                    "ssim_interior": i_ssim,
                    "delta_ssim_interior_minus_boundary": i_ssim - b_ssim,
                    "inference_sec": elapsed,
                })

                if not vis_saved and kind == "gaussian" and sev_name == "medium":
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    axes[0, 0].imshow(np.moveaxis(clean.numpy(), 0, -1))
                    axes[0, 0].set_title(f"Clean ({img_path.name})")
                    axes[0, 1].imshow(np.moveaxis(restored.numpy(), 0, -1))
                    axes[0, 1].set_title(f"Restored ({args.blend})")
                    axes[1, 0].imshow(b_mask.numpy(), cmap="Reds")
                    axes[1, 0].set_title(f"Boundary Mask ({args.boundary_width}px)")
                    axes[1, 1].imshow(i_mask.numpy(), cmap="Blues")
                    axes[1, 1].set_title("Interior Mask")
                    for ax in axes.flat:
                        ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(args.output_dir / "exp2_boundary_visualization.png", dpi=150)
                    plt.close(fig)
                    vis_saved = True

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "exp2_boundary_interior.csv"
    df.to_csv(csv_path, index=False)

    by_noise = df.groupby("noise_type", as_index=False)[["psnr_boundary", "psnr_interior", "ssim_boundary", "ssim_interior"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(by_noise))
    w = 0.35
    axes[0].bar(x - w / 2, by_noise["psnr_boundary"], width=w, label="Boundary")
    axes[0].bar(x + w / 2, by_noise["psnr_interior"], width=w, label="Interior")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(by_noise["noise_type"])
    axes[0].set_title("Boundary vs Interior PSNR (dB)")
    axes[0].legend()

    axes[1].bar(x - w / 2, by_noise["ssim_boundary"], width=w, label="Boundary")
    axes[1].bar(x + w / 2, by_noise["ssim_interior"], width=w, label="Interior")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(by_noise["noise_type"])
    axes[1].set_title("Boundary vs Interior SSIM")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(args.output_dir / "exp2_boundary_interior.png", dpi=150)
    plt.close(fig)
    print(f"Saved {csv_path} and plots ({len(df)} rows)")


if __name__ == "__main__":
    main()