"""Evaluation 4: Image quality vs. inference time and memory usage."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import experiment_utils as eu
from common import LABELS, REMOVE_ORDER
from evaluate_blending import list_dataset_images, load_rgb_chw


def run_auto_pipeline(noisy_chw: torch.Tensor, detector: torch.nn.Module, denoisers: dict,
                      patch_size: int = 128, overlap: int = 32, threshold: float = 0.50,
                      device=eu.DEVICE) -> torch.Tensor:
    c, orig_h, orig_w = noisy_chw.shape
    pad_h, pad_w = max(0, patch_size - orig_h), max(0, patch_size - orig_w)
    padded = torch.nn.functional.pad(noisy_chw.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
    _, h, w = padded.shape
    coords = eu.patch_coordinates(h, w, patch_size=patch_size, overlap=0)
    logits_list = []
    detector.eval()
    with torch.inference_mode():
        for y, x in coords:
            p = padded[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
            logits_list.append(detector(p).squeeze(0).cpu())
    probs = torch.sigmoid(torch.stack(logits_list, dim=0).mean(dim=0))
    detected = {LABELS[i] for i, prob in enumerate(probs.tolist()) if prob >= threshold}

    current = noisy_chw.clone()
    for kind in REMOVE_ORDER:
        if kind in detected:
            current = eu.run_patch_inference(
                current, denoisers[kind], patch_size=patch_size, overlap=overlap,
                blend="hann", min_weight=0.05, device=device
            )
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation 4: Quality vs latency and memory")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "DIV2K_valid_HR")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "experiments" / "exp4")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = torch.device("cuda" if args.device == "cuda" else "cpu" if args.device == "cpu" else eu.DEVICE)
    eu.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()

    images = list_dataset_images(args.dataset_dir, args.limit)
    print(f"[Exp4] Device={device}, images={len(images)}, repeats={args.repeats}", flush=True)

    detector = eu.load_detector(device=device)
    dncnn_models = {k: eu.load_dncnn(k, device=device) for k in eu.DNCNN_CHECKPOINTS}
    basic_models = {
        (k, sev): eu.load_basic_cnn(fname, device=device)
        for k, fnames in eu.BASIC_CHECKPOINTS.items()
        for sev, fname in zip(["low", "medium", "high"], fnames)
    }

    configs = [
        "1_basic_cnn",
        "2_dncnn_no_overlap",
        "3_dncnn_uniform_25pct",
        "4_dncnn_hann_25pct",
        "5_auto_pipeline",
    ]

    warm_x = torch.rand((1, 3, 128, 128), device=device)
    with torch.inference_mode():
        _ = detector(warm_x)
        for m in dncnn_models.values():
            _ = m(warm_x)
        for m in basic_models.values():
            _ = m(warm_x)
    eu.synchronize(device)

    preloaded_clean = [load_rgb_chw(p) for p in images]
    rows = []

    for cfg in configs:
        cfg_t0 = time.perf_counter()
        print(f"[Exp4] Running {cfg}...", flush=True)
        for img_idx, (img_path, clean) in enumerate(zip(images, preloaded_clean)):
            for kind, severities in eu.NOISE_LEVELS.items():
                for sev_idx, sev_name in enumerate(severities):
                    pair_seed = args.seed + img_idx * 100 + sev_idx * 10
                    noisy = eu.add_noise(clean, kind, sev_name, seed=pair_seed)

                    latencies = []
                    restored = None
                    eu.reset_peak_memory(device)
                    for _ in range(max(1, args.repeats)):
                        if cfg == "1_basic_cnn":
                            res, dt = eu.timed_inference(
                                eu.run_patch_inference, noisy, basic_models[(kind, sev_name)],
                                patch_size=128, overlap=32, blend="hann", min_weight=0.05, device=device
                            )
                        elif cfg == "2_dncnn_no_overlap":
                            res, dt = eu.timed_inference(
                                eu.run_patch_inference, noisy, dncnn_models[kind],
                                patch_size=128, overlap=0, blend="none", min_weight=0.05, device=device
                            )
                        elif cfg == "3_dncnn_uniform_25pct":
                            res, dt = eu.timed_inference(
                                eu.run_patch_inference, noisy, dncnn_models[kind],
                                patch_size=128, overlap=32, blend="uniform", min_weight=0.05, device=device
                            )
                        elif cfg == "4_dncnn_hann_25pct":
                            res, dt = eu.timed_inference(
                                eu.run_patch_inference, noisy, dncnn_models[kind],
                                patch_size=128, overlap=32, blend="hann", min_weight=0.05, device=device
                            )
                        else:
                            res, dt = eu.timed_inference(
                                run_auto_pipeline, noisy, detector, dncnn_models,
                                patch_size=128, overlap=32, threshold=args.threshold, device=device
                            )
                        latencies.append(dt)
                        restored = res

                    mem = eu.memory_usage_mb(proc, device=device)
                    psnr_val, ssim_val = eu.calculate_psnr_ssim(clean, restored)
                    mean_t = float(np.mean(latencies))
                    rows.append({
                        "config": cfg,
                        "image": img_path.name,
                        "noise_type": kind,
                        "severity": sev_name,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "mean_sec": mean_t,
                        "median_sec": float(np.median(latencies)),
                        "p95_sec": float(np.percentile(latencies, 95)),
                        "throughput_imgs_per_sec": (1.0 / mean_t) if mean_t > 0 else 0.0,
                        "cpu_rss_mb": mem["cpu_rss_mb"],
                        "gpu_peak_mb": mem["gpu_peak_allocated_mb"],
                    })
        print(f"[Exp4] Finished {cfg} in {time.perf_counter() - cfg_t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "exp4_efficiency.csv"
    df.to_csv(csv_path, index=False)

    summary = df.groupby("config", as_index=False)[
        ["psnr", "ssim", "mean_sec", "median_sec", "p95_sec", "throughput_imgs_per_sec", "cpu_rss_mb", "gpu_peak_mb"]
    ].mean()
    summary_path = args.output_dir / "exp4_performance_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(summary["mean_sec"], summary["psnr"])
    for _, r in summary.iterrows():
        ax.annotate(r["config"], (r["mean_sec"], r["psnr"]), fontsize=8)
    ax.set_xlabel("Mean Inference Time (s)")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("Quality vs Inference Time")
    fig.tight_layout()
    fig.savefig(args.output_dir / "exp4_quality_vs_time.png", dpi=150)
    plt.close(fig)

    mem_col = "gpu_peak_mb" if summary["gpu_peak_mb"].notna().any() else "cpu_rss_mb"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(summary[mem_col], summary["psnr"])
    for _, r in summary.iterrows():
        ax.annotate(r["config"], (r[mem_col], r["psnr"]), fontsize=8)
    ax.set_xlabel(f"Mean {mem_col} (MiB)")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("Quality vs Memory")
    fig.tight_layout()
    fig.savefig(args.output_dir / "exp4_quality_vs_memory.png", dpi=150)
    plt.close(fig)

    print(f"[Exp4] Done in {time.perf_counter() - t_start:.1f}s -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()