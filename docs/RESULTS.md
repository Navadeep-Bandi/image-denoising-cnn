# Results

All numbers come from the per-case CSVs in [`results/experiments/`](../results/experiments) and can be regenerated with
`paper/analysis/make_tables.py`. Experiments 1-5 use the first 20 DIV2K validation images (0801-0820) at full resolution,
3 noise families x 3 severities = **180 image-noise-severity cases**. The supplementary comparison (section 6) uses all 100
validation images resized to 128x128.

> **Timing note.** Each experiment was timed in its own single pass (one repeat, hardware not recorded), so absolute times
> differ slightly between tables (no-overlap: 0.52 s, 0.47 s, 0.46 s). Compare times within a table only. An older run
> (`summary_limit20.txt`, 19 Nov) gave different timings and is intentionally **not** included; the CSVs dated 20 Nov agree
> with each other.

## 1. Ablation: one component at a time (Exp. 4)

| Configuration | PSNR (dB) | SSIM | Time (s) | Img/s | Peak GPU (MB) |
|---|---:|---:|---:|---:|---:|
| A  Basic CNN, Hann 25% | 24.6125 | 0.7466 | 0.247 | 4.09 | 37.2 |
| B  DnCNN, no overlap | 36.7670 | 0.8945 | 0.463 | 2.18 | 39.4 |
| C  DnCNN, uniform 25% | 36.9545 | 0.8972 | 0.785 | 1.28 | 39.4 |
| D  DnCNN, Hann 25% | 37.0285 | 0.8987 | 0.792 | 1.27 | 39.4 |
| E  Auto pipeline (D + detector) | 37.0285 | 0.8987 | 1.129 | 0.91 | 39.5 |

Row A uses the same Hann 25% reconstruction as row D, so A vs D isolates the model (+12.4 dB). Noise type is known in A-D;
E adds the detector and gives identical PSNR/SSIM for +0.34 s (95% CI 0.31-0.36; +43%).

## 2. Paired comparisons over 180 cases (Exp. 1 and 3)

| Comparison | mean dPSNR (dB) | 95% bootstrap CI | cases improved |
|---|---:|---|---:|
| Uniform 25% vs no overlap | +0.1874 | [+0.1691, +0.2078] | 100% |
| Hann 25% vs uniform 25% | +0.0741 | [+0.0666, +0.0818] | 99% |
| **Hann 25% vs no overlap** | **+0.2615** | [+0.2347, +0.2882] | 100% |
| 12.5% vs 0% overlap | +0.2421 | [+0.2177, +0.2683] | 100% |
| 25% vs 12.5% overlap | +0.0194 | [+0.0169, +0.0221] | 97% |
| 37.5% vs 25% overlap | +0.0002 | [-0.0002, +0.0005] | 52% |
| 50% vs 25% overlap | +0.0003 | [-0.0000, +0.0007] | 58% |

Paired design: same image, noise realisation and seed in both arms; percentile bootstrap (5000 resamples); Wilcoxon p-values
are in the paper. The intervals describe variation over these 20 images x 9 noise settings only.

## 3. Overlap sweep (Exp. 3, Hann blending)

| Overlap | Pixels | PSNR (dB) | SSIM | Time (s) | Patches / image |
|---|---:|---:|---:|---:|---:|
| 0% | 0 | 36.7670 | 0.89449 | 0.474 | 179.2 |
| 12.5% | 16 | 37.0091 | 0.89819 | 0.618 | 237.5 |
| 25% | 32 | 37.0285 | 0.89871 | 0.791 | 304.5 |
| 37.5% | 48 | 37.0287 | 0.89875 | 1.120 | 435.0 |
| 50% | 64 | 37.0289 | 0.89875 | 1.686 | 660.3 |

Most of the gain arrives by 12.5% overlap; beyond 25% the PSNR change is indistinguishable from zero while time rises 2.1x.

## 4. By noise family and severity (full pipeline, Exp. 5)

| Noise | Level | Noisy PSNR | Restored PSNR | Gain (dB) | SSIM | Basic CNN gain (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | Low | 28.29 | 34.67 | +6.38 | 0.911 | -2.44 |
| Gaussian | Medium | 20.57 | 30.83 | +10.26 | 0.829 | +4.02 |
| Gaussian | High | 15.07 | 26.81 | +11.73 | 0.660 | +7.96 |
| Salt-and-pepper | Low | 24.88 | 51.54 | +26.66 | 0.998 | +0.42 |
| Salt-and-pepper | Medium | 20.11 | 46.60 | +26.49 | 0.994 | +4.88 |
| Salt-and-pepper | High | 17.90 | 43.83 | +25.93 | 0.989 | +6.00 |
| Speckle | Low | 26.41 | 35.74 | +9.33 | 0.944 | -1.06 |
| Speckle | Medium | 20.70 | 32.52 | +11.82 | 0.901 | +3.92 |
| Speckle | High | 17.91 | 30.72 | +12.81 | 0.863 | +5.96 |

Gaussian is the hardest family; the severity-specific Basic CNN is *worse than the noisy input* at low noise
(43 of 180 cases overall).

## 5. Boundary vs interior (Exp. 2, 32 px overlap, 8 px band)

| Noise | dPSNR interior - boundary | 95% CI | positive in |
|---|---:|---|---:|
| Gaussian | +0.0701 | [0.047, 0.095] | 72% |
| Salt-and-pepper | +0.0504 | [0.012, 0.090] | 63% |
| Speckle | +0.1356 | [0.102, 0.170] | 80% |
| All | +0.0854 | [0.066, 0.105] | 72% |

A small, real PSNR gap; SSIM differences are tiny and mixed in sign (mean -0.0007).

## 6. Basic CNN vs DnCNN on all 100 validation images (128x128, no tiling)

| Noise | Level | Noisy PSNR | Basic CNN PSNR | DnCNN PSNR | Basic SSIM | DnCNN SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | Low | 28.27 | 24.39 | 32.39 | 0.846 | 0.934 |
| Gaussian | Medium | 20.58 | 23.65 | 28.27 | 0.778 | 0.863 |
| Gaussian | High | 15.10 | 22.33 | 24.81 | 0.686 | 0.727 |
| Salt-and-pepper | Low | 24.90 | 24.26 | 47.92 | 0.819 | 0.998 |
| Salt-and-pepper | Medium | 20.12 | 23.67 | 42.97 | 0.755 | 0.996 |
| Salt-and-pepper | High | 17.88 | 22.89 | 40.39 | 0.724 | 0.992 |
| Speckle | Low | 27.14 | 24.40 | 33.18 | 0.825 | 0.957 |
| Speckle | Medium | 21.38 | 23.72 | 29.81 | 0.782 | 0.917 |
| Speckle | High | 18.56 | 23.25 | 28.07 | 0.751 | 0.882 |

Averaged over the nine conditions: Basic CNN 23.62 dB / 0.774 SSIM, DnCNN 34.20 dB / 0.918 SSIM. The two systems were
trained differently, so this is a comparison of trained systems, not of architectures alone.

## 7. Noise detector (Exp. 5)

Precision, recall, F1 and accuracy are 1.00 for all three labels (180/180 exact matches; 60 positives and 120 negatives per
label). Lowest positive score 0.994 and highest negative score 0.167 (both speckle), so the 0.50 threshold is not close to
being crossed. This is a single-noise synthetic test; it says nothing about mixed or real noise.

## 8. Failure cases

* The five lowest restored PSNRs are all Gaussian with sigma >= 25/255 (worst: image 0807, sigma 50/255, 22.55 dB).
* Image 0807 has a mean gain of 8.8 dB against 15.7 dB overall (next lowest image: 14.3 dB). Cause not yet inspected.
* Lowest SSIM values (0.55-0.59): Gaussian sigma = 50/255 on images 0816, 0813, 0805.
* Basic CNN lowers PSNR at low noise (mean -2.44 dB Gaussian-low, -1.06 dB speckle-low).
* To view crops of the worst cases run `python paper/analysis/make_failure_figure.py` (needs torch and the dataset).

## 9. Limitations

20 images at full resolution; synthetic single-family noise only; no mixed-noise, real-noise or cross-dataset test; single timed run per case on unrecorded hardware; training scripts are not in this repository, and the Basic CNN (2 epochs) and DnCNN (40 epochs) were trained with very different budgets; no matched comparison with SwinIR / Restormer / NAFNet / PromptIR / MambaIR.
