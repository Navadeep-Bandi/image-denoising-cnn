
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "DIV2K_valid_HR"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "denoising_gaussian_medium.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

# Evaluation settings
IMAGE_SIZE = (128, 128)
SIGMA = 25 / 255.0
BASE_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

to_tensor = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

def calculate_metrics(reference, prediction):
    psnr = peak_signal_noise_ratio(
        reference, prediction, data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        prediction,
        data_range=1.0,
        channel_axis=2
    )

    return psnr, ssim


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATASET_DIR.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(
            f"No PNG images found in: {DATASET_DIR}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print(f"Device: {device}")
    print(f"Validation images found: {len(image_paths)}")

    # Load model
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from model import DenoisingCNN
    from noise import add_gaussian_noise

    model = DenoisingCNN().to(device)

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Model loaded successfully.")
    print("\nStarting evaluation...\n")

    results = []

    noisy_psnr_values = []
    denoised_psnr_values = []
    noisy_ssim_values = []
    denoised_ssim_values = []
    inference_times = []

    for index, image_path in enumerate(image_paths):

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        clean_tensor = to_tensor(image).unsqueeze(0).to(device)

        # Generate reproducible Gaussian noise
        noisy_tensor = add_gaussian_noise(
            clean_tensor,
            sigma=SIGMA,
            seed=BASE_SEED + index
        )

        # Run inference
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            denoised_tensor = model(noisy_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - start_time

        # Convert tensors to NumPy images
        clean_image = (
            clean_tensor.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        noisy_image = (
            noisy_tensor.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        denoised_image = (
            denoised_tensor.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        # Calculate metrics
        noisy_psnr, noisy_ssim = calculate_metrics(
            clean_image, noisy_image
        )

        denoised_psnr, denoised_ssim = calculate_metrics(
            clean_image, denoised_image
        )

        noisy_psnr_values.append(noisy_psnr)
        denoised_psnr_values.append(denoised_psnr)
        noisy_ssim_values.append(noisy_ssim)
        denoised_ssim_values.append(denoised_ssim)
        inference_times.append(elapsed_time)

        results.append({
            "image": image_path.name,
            "noisy_psnr": noisy_psnr,
            "denoised_psnr": denoised_psnr,
            "noisy_ssim": noisy_ssim,
            "denoised_ssim": denoised_ssim,
            "inference_time_seconds": elapsed_time
        })

        print(
            f"[{index + 1:03d}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"Noisy PSNR: {noisy_psnr:.2f} | "
            f"Denoised PSNR: {denoised_psnr:.2f} | "
            f"Denoised SSIM: {denoised_ssim:.4f}"
        )

    # Calculate averages
    avg_noisy_psnr = np.mean(noisy_psnr_values)
    avg_denoised_psnr = np.mean(denoised_psnr_values)

    avg_noisy_ssim = np.mean(noisy_ssim_values)
    avg_denoised_ssim = np.mean(denoised_ssim_values)

    avg_inference_time = np.mean(inference_times)

    # Save individual results
    csv_path = RESULTS_DIR / "gaussian_medium_evaluation.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print("\n" + "=" * 55)
    print("GAUSSIAN DENOISING EVALUATION SUMMARY")
    print("=" * 55)

    print(f"Images evaluated:       {len(image_paths)}")

    print(f"\nAverage Noisy PSNR:     {avg_noisy_psnr:.4f} dB")
    print(f"Average Denoised PSNR:  {avg_denoised_psnr:.4f} dB")
    print(f"PSNR Improvement:       {avg_denoised_psnr - avg_noisy_psnr:.4f} dB")

    print(f"\nAverage Noisy SSIM:     {avg_noisy_ssim:.4f}")
    print(f"Average Denoised SSIM:  {avg_denoised_ssim:.4f}")
    print(f"SSIM Improvement:       {avg_denoised_ssim - avg_noisy_ssim:.4f}")

    print(f"\nAverage inference time: {avg_inference_time * 1000:.2f} ms/image")

    print(f"\nIndividual results saved to:\n{csv_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()

