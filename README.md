# CNN-Based Image Denoising

A comparative image-denoising project evaluating a **Basic CNN** and **DnCNN** on synthetic Gaussian, salt-and-pepper, and speckle noise.

## Team
- Bandi Navadeep
- Rajan Nayak
- Rohit

## Project status
The Streamlit application has been run successfully by the team, and all **9 noise-type × severity combinations** have been tested. The evaluation covers 2 architectures × 3 noise types × 3 severity levels, using 100 DIV2K validation images per combination (1,800 model predictions).

## Features
- Upload an image and synthesize a selected noise type and severity.
- Run the same noisy input through Basic CNN and DnCNN.
- Compare original, noisy, and denoised images.
- Report MSE, PSNR, and SSIM.
- Download denoised outputs.

## Architectures
**Basic CNN:** four 3×3 convolution layers (3→32→64→32→3), ReLU after the first three layers, sigmoid output. It predicts the clean image directly.

**DnCNN:** depth 17 with 64 feature channels. It predicts the noise residual and subtracts it from the noisy input (residual learning).

## Evaluation results

Mean over the 9 noise/severity combinations (100 validation images per combination):

| Metric | Basic CNN | DnCNN |
|---|---:|---:|
| Average PSNR (dB) | 23.6174 | 34.2023 |
| Average SSIM | 0.7740 | 0.9184 |

DnCNN achieved higher PSNR and SSIM in all nine evaluated combinations. See [results/all_models_summary.csv](results/all_models_summary.csv) for the full breakdown.

## Run locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the project checkpoints in the `models/` directory and the DIV2K validation images in `dataset/DIV2K_valid_HR/` if running the evaluation script.
5. Run the Streamlit app:
   ```bash
   python -m streamlit run app.py
   ```

> **Important:** The current repository setup documentation is being prepared. The working `app.py`, source scripts, and checkpoint files must be added to this repository before the clone-and-run instructions are complete. The DIV2K dataset is not included; obtain it from the official dataset source and follow its terms.

## Evaluation

The all-model evaluation script is `src/evaluate_all_models.py`. Its reported CSV outputs are in `results/`. The experiment evaluates 100 validation images at 128×128 resolution for each of the 18 model/noise/severity combinations.

## Methodology notes and limitations
- This is a comparison of the two **trained systems**, not a controlled experiment isolating architecture alone: the Basic CNN was trained for 2 epochs, while DnCNN was trained for 40 epochs using a different training procedure.
- DnCNN training used a mixed-noise strategy, while evaluation applied one noise type at a time.
- The validation set uses synthetic noise and resized images; real-world camera noise may behave differently.
- Inference-time values in the original CSV are implausibly small and are not reported as reliable timing results.
- The unusually high DnCNN PSNR for salt-and-pepper noise should be interpreted alongside the sparse corruption pattern and global image-level MSE/PSNR calculation.

## Repository layout
```
app.py                 # Streamlit application (to be committed)
src/                   # model, noise, and evaluation scripts
models/                # trained model checkpoints
results/               # evaluation CSVs and comparison figures
dataset/               # DIV2K validation images (not committed)
requirements.txt
```

## Authors
Bandi Navadeep, Rajan Nayak, and Rohit.
