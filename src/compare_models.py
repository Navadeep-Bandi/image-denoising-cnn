
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
RESULTS_DIR = PROJECT_ROOT / "results"

CNN_CHECKPOINT = PROJECT_ROOT / "models" / "denoising_gaussian_medium.pth"
DNCNN_CHECKPOINT = PROJECT_ROOT / "models" / "denoise_gaussian.pt"

# Evaluation settings
IMAGE_SIZE = (128, 128)
SIGMA = 25 / 255.0
BASE_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
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


def tensor_to_image(tensor):
    return (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATASET_DIR.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError("No validation PNG images found.")

    if not CNN_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CNN_CHECKPOINT}")

    if not DNCNN_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {DNCNN_CHECKPOINT}")

    # Import model definitions
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from model import DenoisingCNN
    from dncnn_model import DnCNN
    from noise import add_gaussian_noise

    print("Device:", device)
    print("Validation images:", len(image_paths))

    # Load first model
    cnn_model = DenoisingCNN().to(device)

    cnn_state = torch.load(
        CNN_CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    cnn_model.load_state_dict(cnn_state, strict=True)
    cnn_model.eval()

    # Load second model
    dncnn_model = DnCNN().to(device)

    dncnn_checkpoint = torch.load(
        DNCNN_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    dncnn_model.load_state_dict(
        dncnn_checkpoint["model"],
        strict=True
    )
    dncnn_model.eval()

    print("Both models loaded successfully.")
    print("\nStarting comparison...\n")

    results = []

    metric_values = {
        "noisy_psnr": [],
        "cnn_psnr": [],
        "dncnn_psnr": [],
        "noisy_ssim": [],
        "cnn_ssim": [],
        "dncnn_ssim": [],
        "cnn_time": [],
        "dncnn_time": []
    }

    for index, image_path in enumerate(image_paths):

        # Load clean image
        image = Image.open(image_path).convert("RGB")
        clean_tensor = transform(image).unsqueeze(0).to(device)

        # Generate exactly one noisy input for both models
        noisy_tensor = add_gaussian_noise(
            clean_tensor,
            sigma=SIGMA,
            seed=BASE_SEED + index
        )

        # First model inference
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            cnn_output = cnn_model(noisy_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        cnn_time = time.perf_counter() - start
        cnn_output = cnn_output.clamp(0, 1)

        # Second model inference
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            dncnn_output = dncnn_model(noisy_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        dncnn_time = time.perf_counter() - start
        dncnn_output = dncnn_output.clamp(0, 1)

        # Convert to NumPy
        clean_image = tensor_to_image(clean_tensor)
        noisy_image = tensor_to_image(noisy_tensor)
        cnn_image = tensor_to_image(cnn_output)
        dncnn_image = tensor_to_image(dncnn_output)

        # Calculate metrics
        noisy_psnr, noisy_ssim = calculate_metrics(
            clean_image, noisy_image
        )

        cnn_psnr, cnn_ssim = calculate_metrics(
            clean_image, cnn_image
        )

        dncnn_psnr, dncnn_ssim = calculate_metrics(
            clean_image, dncnn_image
        )

        metric_values["noisy_psnr"].append(noisy_psnr)
        metric_values["cnn_psnr"].append(cnn_psnr)
        metric_values["dncnn_psnr"].append(dncnn_psnr)

        metric_values["noisy_ssim"].append(noisy_ssim)
        metric_values["cnn_ssim"].append(cnn_ssim)
        metric_values["dncnn_ssim"].append(dncnn_ssim)

        metric_values["cnn_time"].append(cnn_time)
        metric_values["dncnn_time"].append(dncnn_time)

        results.append({
            "image": image_path.name,
            "noisy_psnr": noisy_psnr,
            "cnn_psnr": cnn_psnr,
            "dncnn_psnr": dncnn_psnr,
            "noisy_ssim": noisy_ssim,
            "cnn_ssim": cnn_ssim,
            "dncnn_ssim": dncnn_ssim,
            "cnn_time_ms": cnn_time * 1000,
            "dncnn_time_ms": dncnn_time * 1000
        })

        print(
            f"[{index + 1:03d}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"CNN PSNR: {cnn_psnr:.2f} | "
            f"DnCNN PSNR: {dncnn_psnr:.2f} | "
            f"CNN SSIM: {cnn_ssim:.4f} | "
            f"DnCNN SSIM: {dncnn_ssim:.4f}"
        )

    # Calculate average metrics
    averages = {
        key: float(np.mean(values))
        for key, values in metric_values.items()
    }

    # Save individual image results
    csv_path = RESULTS_DIR / "gaussian_model_comparison.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    # Print final comparison
    print("\n" + "=" * 65)
    print("GAUSSIAN DENOISING MODEL COMPARISON")
    print("=" * 65)

    print(f"Images evaluated: {len(image_paths)}")
    print(f"Gaussian sigma: {SIGMA:.6f}")

    print("\nAVERAGE PSNR")
    print(f"Noisy input: {averages['noisy_psnr']:.4f} dB")
    print(f"Basic CNN:   {averages['cnn_psnr']:.4f} dB")
    print(f"DnCNN:       {averages['dncnn_psnr']:.4f} dB")

    print("\nAVERAGE SSIM")
    print(f"Noisy input: {averages['noisy_ssim']:.4f}")
    print(f"Basic CNN:   {averages['cnn_ssim']:.4f}")
    print(f"DnCNN:       {averages['dncnn_ssim']:.4f}")

    print("\nAVERAGE INFERENCE TIME")
    print(f"Basic CNN: {averages['cnn_time'] * 1000:.2f} ms/image")
    print(f"DnCNN:     {averages['dncnn_time'] * 1000:.2f} ms/image")

    print("\nPSNR IMPROVEMENT OVER NOISY INPUT")
    print(
        f"Basic CNN: "
        f"{averages['cnn_psnr'] - averages['noisy_psnr']:.4f} dB"
    )
    print(
        f"DnCNN:     "
        f"{averages['dncnn_psnr'] - averages['noisy_psnr']:.4f} dB"
    )

    print("\nIndividual results saved to:")
    print(csv_path)
    print("=" * 65)


if __name__ == "__main__":
    main()
