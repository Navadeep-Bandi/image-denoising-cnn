"""Evaluation 1: With vs. without overlap-weighted blending."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import experiment_utils as eu


def list_dataset_images(dataset_dir: Path, limit: int | None = None) -> List[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    files = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not files:
        raise FileNotFoundError(f"No images found in {dataset_dir}")
    return files[:limit] if limit is not None else files


def load_rgb_chw(path: Path) -> torch.Tensor:
    from PIL import Image
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.moveaxis(arr, -1, 0)).contiguous()


def estimate_seam_discontinuity(restored_chw: torch.Tensor, patch_size: int = 128, overlap: int = 32, boundary_width: int = 4) -> float:
    _, h, w = restored_chw.shape
    b_mask, _ = eu.make_boundary_interior_masks(h, w, patch_size=patch_size, overlap=overlap, boundary_width=boundary_width)
    if not b_mask.any():
        return 0.0
    m = b_mask.numpy()
    dx = torch.abs(restored_chw[:, :, 1:] - restored_chw[:, :, :-1]).mean(dim=0).cpu().numpy()
    dy = torch.abs(restored_chw[:, 1:, :] - restored_chw[:, :-1, :]).mean(dim=0).cpu().numpy()
    vals = []
    if m[:, 1:].any():
        vals.append(dx[m[:, 1:]])
    if m[1:, :].any():
        vals.append(dy[m[1:, :]])
    return float(np.mean(np.concatenate(vals))) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation 1: Blending comparison")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "DIV2K_valid_HR")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "experiments" / "exp1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" else "cpu" if args.device == "cpu" else eu.DEVICE)
    eu.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = list_dataset_images(args.dataset_dir, args.limit)
    dncnn_models = {k: eu.load_dncnn(k, device=device) for k in eu.DNCNN_CHECKPOINTS}

    configs = [
        ("A_no_overlap", 0, "none"),
        ("B_uniform_25pct", args.overlap, "uniform"),
        ("C_hann_25pct", args.overlap, "hann"),
    ]

    warm_x = torch.rand((1, 3, args.patch_size, args.patch_size), device=device)
    with torch.inference_mode():
        for m in dncnn_models.values():
            _ = m(warm_x)
    eu.synchronize(device)

    rows = []
    for img_idx, img_path in enumerate(images):
        clean = load_rgb_chw(img_path)
        for kind, severities in eu.NOISE_LEVELS.items():
            model = dncnn_models[kind]
            for sev_idx, sev_name in enumerate(severities):
                pair_seed = args.seed + img_idx * 100 + sev_idx * 10
                noisy = eu.add_noise(clean, kind, sev_name, seed=pair_seed)
                for cfg_name, ov, blend_mode in configs:
                    eu.reset_peak_memory(device)
                    restored, elapsed = eu.timed_inference(
                        eu.run_patch_inference,
                        noisy,
                        model,
                        patch_size=args.patch_size,
                        overlap=ov,
                        blend=blend_mode,
                        min_weight=0.05,
                        device=device,
                    )
                    mem = eu.memory_usage_mb(device=device)
                    psnr_val, ssim_val = eu.calculate_psnr_ssim(clean, restored)
                    seam_score = estimate_seam_discontinuity(restored, patch_size=args.patch_size, overlap=ov)
                    rows.append({
                        "image": img_path.name,
                        "noise_type": kind,
                        "severity": sev_name,
                        "config": cfg_name,
                        "nominal_overlap": ov,
                        "blend": blend_mode,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "seam_discontinuity": seam_score,
                        "inference_sec": elapsed,
                        "cpu_rss_mb": mem["cpu_rss_mb"],
                        "gpu_peak_mb": mem["gpu_peak_allocated_mb"],
                    })

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "exp1_blending_comparison.csv"
    df.to_csv(csv_path, index=False)

    summary = df.groupby("config", as_index=False)[["psnr", "ssim", "seam_discontinuity", "inference_sec"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(summary["config"], summary["psnr"])
    axes[0].set_title("Mean PSNR (dB)")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(summary["config"], summary["ssim"])
    axes[1].set_title("Mean SSIM")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(summary["config"], summary["inference_sec"])
    axes[2].set_title("Mean Inference Time (s)")
    axes[2].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(args.output_dir / "exp1_blending_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {csv_path} and plot ({len(df)} rows)")


if __name__ == "__main__":
    main()