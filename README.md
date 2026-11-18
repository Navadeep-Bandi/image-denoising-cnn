# Intelligent Image Denoising Using CNNs

A research and implementation project for image denoising with convolutional neural networks. The project explores three synthetic noise types—Gaussian, salt-and-pepper, and speckle—and compares a Basic CNN with DnCNN. The latest local project bundle also includes a multi-label noise detector and a Streamlit interface for automatic detection/restoration and manual noise-generation experiments.

> **Repository synchronization status:** This GitHub repository currently contains the project documentation and dependency metadata, but it does not yet contain the latest application source, `src/` modules, evaluation outputs, or trained checkpoints. The complete local project archive is being reviewed before those files are synchronized. Do not treat this repository as runnable until that synchronization is complete.

## Project objectives

- Generate and study Gaussian, salt-and-pepper, and speckle noise.
- Compare direct image prediction (Basic CNN) with residual learning (DnCNN).
- Detect the presence of supported noise types using a multi-label CNN detector.
- Restore images through a Streamlit application.
- Evaluate denoising quality using mean squared error (MSE), peak signal-to-noise ratio (PSNR), and structural similarity (SSIM).

## Implemented model families

### Basic CNN

The project’s Basic CNN uses four convolutional layers with channel progression 3 → 32 → 64 → 32 → 3. ReLU activations are used in the intermediate layers, with a sigmoid output. The network predicts the clean image directly. Nine checkpoints are used for the combinations of three noise types and three severity levels.

### DnCNN

The DnCNN model uses a 17-layer residual-learning architecture with 64 feature channels. It estimates the noise residual and returns the restored image by subtracting that residual from the noisy input. Three checkpoints are used, one for each supported noise family.

### Multi-label noise detector

The latest local application includes a detector trained to identify Gaussian, salt-and-pepper, and speckle noise. Its predictions are used to select the corresponding DnCNN restoration stages. Detection confidence is not equivalent to a calibrated probability unless calibration has been separately evaluated.

## Evaluation

The supplied local results include an evaluation across 100 DIV2K validation images, three noise families, three severity settings, and two denoising approaches (1,800 model predictions). The current summary reports mean PSNR of 23.6174 dB for Basic CNN and 34.2023 dB for DnCNN, and mean SSIM of 0.7740 and 0.9184 respectively.

These values are preliminary project results, not independently reproduced benchmarks. The comparison is between the trained systems rather than a controlled architecture-only comparison: the Basic CNN and DnCNN used different training durations and training procedures. The evaluation uses synthetic noise and resized 128 × 128 inputs, so the findings should not be generalized directly to real camera noise or arbitrary-resolution images.

Inference-time values are intentionally omitted from headline results until the timing methodology and units are revalidated.

## Local setup

The commands below describe the intended setup once the application source and checkpoints have been synchronized:

```bash
git clone https://github.com/Navadeep-Bandi/image-denoising-cnn.git
cd image-denoising-cnn

python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```

The complete application requires the source modules under `src/` and the trained model files under `models/`. The DIV2K dataset is not included in the project archive or repository; download it separately from its official source if you need to reproduce dataset-based evaluation, and follow the dataset's terms.

## Expected project structure

```text
image-denoising-cnn/
├── app.py
├── requirements.txt
├── src/
│   ├── common.py
│   ├── model.py
│   ├── dncnn_model.py
│   ├── noise.py
│   └── evaluation and verification scripts
├── models/                 # trained .pth / .pt checkpoints
├── results/                # CSV metrics and comparison figures
└── dataset/                # not committed
```

## Reproducibility and interpretation notes

- Basic CNN and DnCNN are trained systems with different training configurations; observed differences cannot be attributed solely to architecture.
- The evaluation uses synthetic noise, not a representative collection of real-world camera-noise conditions.
- Inputs in the reported 100-image evaluation were resized to 128 × 128.
- Salt-and-pepper corruption is sparse at the tested severities; interpret its PSNR alongside the corruption density and visual comparisons.
- Model confidence scores should not be described as calibrated probabilities without a calibration study.
- Report hardware, software versions, random seeds, image preprocessing, and timing protocol when publishing runtime comparisons.

## Team

- Bandi Navadeep
- Rajan Nayak
- Rohit

## License

No license has been specified yet. Until a license is added, all rights remain with the repository owner; do not assume the code is open for reuse.
