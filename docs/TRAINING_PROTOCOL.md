# Training protocol

The checkpoints in `models/` were produced by training code that is **not part of this repository**; only the shared data
and noise utilities (`src/common.py`) are. The optimizer settings below were reported by the authors from their own
training runs, not extracted from code in this repository.

## Optimizer and schedule (reported by the authors)

| Model | Optimizer | Learning rate | LR schedule | Epochs | Loss |
|---|---|---|---|---|---|
| Basic CNN (9 checkpoints, per family and severity) | Adam | 0.001 | not specified | 2 | MSE |
| DnCNN (3 checkpoints, one per noise family) | Adam | 0.001 (initial) | cosine annealing | 40 | MSE |
| Noise detector | not documented | not documented | not documented | not documented | not documented |

**Important for interpreting results:** the Basic CNN was trained for 2 epochs and the DnCNN for 40, so the quality gap
between them (about 12 dB PSNR) reflects both the architecture and the training budget. It is a comparison of the models as
trained, not of architectures at equal training. The Basic CNN may be undertrained, which could contribute to it making
low-noise images worse.

## Data and noise (from `src/common.py`)

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

## Still to document by the authors

* Training settings of the noise detector (optimizer, learning rate, schedule, epochs, loss).
* The origin of the images in `data/clean`, and confirmation that none of DIV2K validation images 0801-0820 is among them
  (otherwise the evaluation would overlap with training data).
* The training scripts themselves; add them under `src/` when available.
