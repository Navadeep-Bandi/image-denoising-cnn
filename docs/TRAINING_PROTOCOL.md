# Training protocol (from `src/common.py`)

The checkpoints in `models/` were produced by training code that is **not part of this repository**; only the shared data and
noise utilities (`src/common.py`) are. They define:

| Item | Setting |
|---|---|
| Clean images | folder `data/clean` (random 256x256 crops) - *contents not recorded* |
| Split | 70 / 15 / 15 train / validation / test **by source image**, shuffled with seed 42 |
| Augmentation | random horizontal and vertical flips |
| Quantisation | noisy images clipped and rounded to 8 bit |
| Training noise strengths | Gaussian sigma in [5, 60]/255, speckle variance in [0.005, 0.10], salt-and-pepper rate in [0.005, 0.07] (uniform) |
| Test levels | Gaussian 10/25/50 /255, speckle 0.01/0.04/0.08, SP 0.01/0.03/0.05 (inside the training ranges) |
| Noise composition | added in the order speckle -> Gaussian -> salt-and-pepper; removed in reverse (SP, Gaussian, speckle) |
| Denoiser targets | image *before* that noise was added; earlier noises present with probability 0.5 |
| Detector labels | 8 subsets (clean, 3 single, 3 pairs, all three), probabilities .15/.15/.15/.15/.08/.08/.08/.16 |
| Detector inputs | also partially cleaned by the real denoisers to mimic cascades |
| Checkpoint validation PSNR | Gaussian 30.34 dB, SP 46.23 dB, speckle 33.01 dB (stored in the `.pt` files) |

**Still to document by the authors:** optimizer, learning-rate schedule, epochs, loss, the origin of `data/clean`, and
confirmation that no DIV2K validation image 0801-0820 is in it. Add the training script(s) under `src/` when available.
