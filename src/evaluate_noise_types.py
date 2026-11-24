"""Evaluation 5: Noise detection and restoration across noise types and severities."""
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
from evaluate_efficiency import run_auto_pipeline


def predict_noise_probs(noisy_chw: torch.Tensor, detector: torch.nn.Module,
                        patch_size: int = 128, device=eu.DEVICE) -> np.ndarray:
    c, orig_h, orig_w = noisy_chw.shape
    pad_h, pad_w = max(0, patch_size - orig_h), max(0, patch_size - orig_w)
    padded = torch.nn.functional.pad(noisy_chw.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
    _, h, w = padded.shape
    coords = eu.patch_coordinates(h, w, patch_size=patch_size, overlap=0)
    logits = []
    detector.eval()
    with torch.inference_mode():
        for y, x in coords:
            p = padded[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
            logits.append(detector(p).squeeze(0).cpu())
    return torch.sigmoid(torch.stack(logits, dim=0).mean(dim=0)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation 5: Noise types and severities")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "DIV2K_valid_HR")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "experiments" / "exp5")
    parser.add_argument("--limit", type=int, default=None)
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
    print(f"[Exp5] Device={device}, images={len(images)}", flush=True)

    detector = eu.load_detector(device=device)
    dncnn_models = {k: eu.load_dncnn(k, device=device) for k in eu.DNCNN_CHECKPOINTS}

    det_rows = []
    res_rows = []

    for img_idx, img_path in enumerate(images):
        img_t0 = time.perf_counter()
        clean = load_rgb_chw(img_path)
        for kind, severities in eu.NOISE_LEVELS.items():
            true_vec = np.array([1 if lbl == kind else 0 for lbl in LABELS], dtype=int)
            for sev_idx, sev_name in enumerate(severities):
                pair_seed = args.seed + img_idx * 100 + sev_idx * 10
                noisy = eu.add_noise(clean, kind, sev_name, seed=pair_seed)

                probs = predict_noise_probs(noisy, detector, patch_size=128, device=device)
                pred_vec = (probs >= args.threshold).astype(int)
                pred_labels = [LABELS[i] for i, v in enumerate(pred_vec) if v == 1]

                eu.reset_peak_memory(device)
                restored, elapsed = eu.timed_inference(
                    run_auto_pipeline, noisy, detector, dncnn_models,
                    patch_size=128, overlap=32, threshold=args.threshold, device=device
                )
                mem = eu.memory_usage_mb(proc, device=device)

                noisy_psnr, noisy_ssim = eu.calculate_psnr_ssim(clean, noisy)
                rest_psnr, rest_ssim = eu.calculate_psnr_ssim(clean, restored)

                det_rows.append({
                    "image": img_path.name,
                    "true_noise": kind,
                    "severity": sev_name,
                    "prob_gaussian": float(probs[0]),
                    "prob_sp": float(probs[1]),
                    "prob_speckle": float(probs[2]),
                    "pred_gaussian": int(pred_vec[0]),
                    "pred_sp": int(pred_vec[1]),
                    "pred_speckle": int(pred_vec[2]),
                    "true_gaussian": int(true_vec[0]),
                    "true_sp": int(true_vec[1]),
                    "true_speckle": int(true_vec[2]),
                    "exact_match": int(np.array_equal(pred_vec, true_vec)),
                    "selected_models_order": "->".join([k for k in REMOVE_ORDER if k in pred_labels]),
                })

                res_rows.append({
                    "image": img_path.name,
                    "noise_type": kind,
                    "severity": sev_name,
                    "noisy_psnr": noisy_psnr,
                    "restored_psnr": rest_psnr,
                    "psnr_improvement": rest_psnr - noisy_psnr,
                    "noisy_ssim": noisy_ssim,
                    "restored_ssim": rest_ssim,
                    "ssim_improvement": rest_ssim - noisy_ssim,
                    "inference_sec": elapsed,
                    "cpu_rss_mb": mem["cpu_rss_mb"],
                    "gpu_peak_mb": mem["gpu_peak_allocated_mb"],
                })
        print(f"[Exp5] Image {img_idx+1}/{len(images)} ({img_path.name}) done in {time.perf_counter() - img_t0:.1f}s", flush=True)

    det_df = pd.DataFrame(det_rows)
    res_df = pd.DataFrame(res_rows)
    det_df.to_csv(args.output_dir / "exp5_noise_detection.csv", index=False)
    res_df.to_csv(args.output_dir / "exp5_restoration_results.csv", index=False)

    metrics_summary = []
    for idx, lbl in enumerate(LABELS):
        y_t = det_df[f"true_{lbl}"].to_numpy()
        y_p = det_df[f"pred_{lbl}"].to_numpy()
        tp = int(((y_t == 1) & (y_p == 1)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())
        tn = int(((y_t == 0) & (y_p == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / max(1, len(y_t))
        metrics_summary.append({"label": lbl, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc})
    pd.DataFrame(metrics_summary).to_csv(args.output_dir / "exp5_detection_metrics_by_label.csv", index=False)

    top_pred = []
    for _, r in det_df.iterrows():
        p_arr = [r["prob_gaussian"], r["prob_sp"], r["prob_speckle"]]
        max_idx = int(np.argmax(p_arr))
        top_pred.append(LABELS[max_idx] if p_arr[max_idx] >= args.threshold else "none")
    det_df["top_pred"] = top_pred
    cm_classes = ["gaussian", "sp", "speckle", "none"]
    cm = pd.crosstab(
        pd.Categorical(det_df["true_noise"], categories=["gaussian", "sp", "speckle"]),
        pd.Categorical(det_df["top_pred"], categories=cm_classes),
        dropna=False,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm_classes)))
    ax.set_xticklabels(cm_classes)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["gaussian", "sp", "speckle"])
    ax.set_xlabel("Predicted (Top >= Threshold)")
    ax.set_ylabel("True Noise")
    ax.set_title("Noise Detector Matrix")
    for i in range(3):
        for j in range(len(cm_classes)):
            ax.text(j, i, str(cm.values[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output_dir / "exp5_confusion_matrix.png", dpi=150)
    plt.close(fig)

    q_sum = res_df.groupby("noise_type", as_index=False)[["noisy_psnr", "restored_psnr", "noisy_ssim", "restored_ssim"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(q_sum))
    w = 0.35
    axes[0].bar(x - w / 2, q_sum["noisy_psnr"], width=w, label="Noisy")
    axes[0].bar(x + w / 2, q_sum["restored_psnr"], width=w, label="Restored")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(q_sum["noise_type"])
    axes[0].set_title("Mean PSNR Before vs After")
    axes[0].legend()

    axes[1].bar(x - w / 2, q_sum["noisy_ssim"], width=w, label="Noisy")
    axes[1].bar(x + w / 2, q_sum["restored_ssim"], width=w, label="Restored")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(q_sum["noise_type"])
    axes[1].set_title("Mean SSIM Before vs After")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(args.output_dir / "exp5_quality_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(q_sum["noise_type"], q_sum["restored_psnr"] - q_sum["noisy_psnr"])
    ax.set_title("Mean PSNR Gain by Noise Type")
    ax.set_ylabel("PSNR Gain (dB)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "exp5_noise_comparison.png", dpi=150)
    plt.close(fig)

    print(f"[Exp5] Done in {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()