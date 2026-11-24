
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms

from model import DenoisingCNN
from dncnn_model import DnCNN


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "DIV2K_valid_HR"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_SIZE = (128, 128)
BASE_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


# ==========================================
# 2. NOISE SETTINGS
# ==========================================

NOISE_LEVELS = {
    "Gaussian": {
        "Low": 10 / 255.0,
        "Medium": 25 / 255.0,
        "High": 50 / 255.0
    },
    "Salt-and-pepper": {
        "Low": 0.01,
        "Medium": 0.03,
        "High": 0.05
    },
    "Speckle": {
        "Low": 0.01,
        "Medium": 0.04,
        "High": 0.08
    }
}


BASIC_CHECKPOINTS = {
    "Gaussian": {
        "Low": "denoising_gaussian_low.pth",
        "Medium": "denoising_gaussian_medium.pth",
        "High": "denoising_gaussian_high.pth"
    },
    "Salt-and-pepper": {
        "Low": "denoising_salt-and-pepper_low.pth",
        "Medium": "denoising_salt-and-pepper_medium.pth",
        "High": "denoising_salt-and-pepper_high.pth"
    },
    "Speckle": {
        "Low": "denoising_speckle_low.pth",
        "Medium": "denoising_speckle_medium.pth",
        "High": "denoising_speckle_high.pth"
    }
}


DNCNN_CHECKPOINTS = {
    "Gaussian": "denoise_gaussian.pt",
    "Salt-and-pepper": "denoise_sp.pt",
    "Speckle": "denoise_speckle.pt"
}


# ==========================================
# 3. REPRODUCIBLE NOISE GENERATION
# ==========================================
def generate_noisy_image(clean, noise_type, level, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    amount = NOISE_LEVELS[noise_type][level]
    noisy = clean.clone()
    c, h, w = noisy.shape

    if noise_type == "Gaussian":
        noise = torch.randn(
            noisy.shape, generator=generator, dtype=noisy.dtype
        ) * amount
        noisy = torch.clamp(noisy + noise, 0.0, 1.0)

    elif noise_type == "Salt-and-pepper":
        salt_mask = torch.rand((h, w), generator=generator) < (amount / 2)
        pepper_mask = torch.rand((h, w), generator=generator) < (amount / 2)

        for channel in range(c):
            noisy[channel][salt_mask] = 1.0
            noisy[channel][pepper_mask] = 0.0

    elif noise_type == "Speckle":
        noise = torch.randn(
            noisy.shape, generator=generator, dtype=noisy.dtype
        ) * (amount ** 0.5)
        noisy = torch.clamp(noisy + noisy * noise, 0.0, 1.0)

    return noisy


# ==========================================
# 4. METRIC CALCULATION
# ==========================================

def tensor_to_numpy(tensor):

    return (
        tensor.squeeze(0)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )


def calculate_metrics(reference, prediction):

    prediction = np.clip(prediction, 0.0, 1.0)

    psnr = peak_signal_noise_ratio(
        reference,
        prediction,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        prediction,
        data_range=1.0,
        channel_axis=2
    )

    return psnr, ssim


# ==========================================
# 5. LOAD ALL MODELS
# ==========================================

def load_all_models():

    basic_models = {}
    dncnn_models = {}

    print("\nLoading Basic CNN models...")

    for noise_type in BASIC_CHECKPOINTS:

        for level, filename in BASIC_CHECKPOINTS[noise_type].items():

            path = MODEL_DIR / filename

            model = DenoisingCNN().to(device)

            state_dict = torch.load(
                path,
                map_location=device,
                weights_only=True
            )

            model.load_state_dict(state_dict, strict=True)
            model.eval()

            basic_models[(noise_type, level)] = model

            print("[OK] Basic CNN:", noise_type, level)

    print("\nLoading DnCNN models...")

    for noise_type, filename in DNCNN_CHECKPOINTS.items():

        path = MODEL_DIR / filename

        model = DnCNN().to(device)

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model"],
            strict=True
        )

        model.eval()

        dncnn_models[noise_type] = model

        print("[OK] DnCNN:", noise_type)

    return basic_models, dncnn_models


# ==========================================
# 6. MODEL INFERENCE
# ==========================================

def run_inference(model, noisy_tensor):

    noisy_tensor = noisy_tensor.unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model(noisy_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return output, elapsed


# ==========================================
# 7. MAIN EVALUATION
# ==========================================

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATASET_DIR.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(
            f"No validation images found in {DATASET_DIR}"
        )

    print("=" * 60)
    print("IMAGE DENOISING - COMPLETE MODEL EVALUATION")
    print("=" * 60)

    print("Device:", device)
    print("Validation images:", len(image_paths))

    basic_models, dncnn_models = load_all_models()

    # Warm up models before measuring inference time.
    dummy = torch.rand(1, 3, 128, 128, device=device)

    with torch.inference_mode():
        for model in list(basic_models.values()) + list(dncnn_models.values()):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    detailed_results = []

    summary_results = []

    total_experiments = 18
    experiment_number = 0

    # ======================================
    # Evaluate every noise type and level
    # ======================================

    for noise_type in NOISE_LEVELS:

        for level in NOISE_LEVELS[noise_type]:

            experiment_number += 1

            print("\n" + "=" * 60)
            print(
                f"EXPERIMENT {experiment_number}/{total_experiments}: "
                f"{noise_type} - {level}"
            )
            print("=" * 60)

            basic_model = basic_models[(noise_type, level)]
            dncnn_model = dncnn_models[noise_type]

            models_to_test = {
                "Basic CNN": basic_model,
                "DnCNN": dncnn_model
            }

            noisy_psnr_values = []
            noisy_ssim_values = []

            model_metrics = {
                name: {
                    "psnr": [],
                    "ssim": [],
                    "time": []
                }
                for name in models_to_test
            }

            for index, image_path in enumerate(image_paths):

                image = Image.open(image_path).convert("RGB")

                clean_tensor = transform(image)

                # Identical noisy input for both models.
                noisy_tensor = generate_noisy_image(
                    clean_tensor,
                    noise_type,
                    level,
                    BASE_SEED + index
                )

                clean_np = tensor_to_numpy(
                    clean_tensor.unsqueeze(0)
                )

                noisy_np = tensor_to_numpy(
                    noisy_tensor.unsqueeze(0)
                )

                noisy_psnr, noisy_ssim = calculate_metrics(
                    clean_np, noisy_np
                )

                noisy_psnr_values.append(noisy_psnr)
                noisy_ssim_values.append(noisy_ssim)

                for model_name, model in models_to_test.items():

                    output, elapsed = run_inference(
                        model, noisy_tensor
                    )

                    output_np = tensor_to_numpy(output)

                    denoised_psnr, denoised_ssim = calculate_metrics(
                        clean_np, output_np
                    )

                    model_metrics[model_name]["psnr"].append(
                        denoised_psnr
                    )

                    model_metrics[model_name]["ssim"].append(
                        denoised_ssim
                    )

                    model_metrics[model_name]["time"].append(
                        elapsed
                    )

                    detailed_results.append({
                        "noise_type": noise_type,
                        "severity": level,
                        "model": model_name,
                        "image": image_path.name,
                        "noisy_psnr": noisy_psnr,
                        "denoised_psnr": denoised_psnr,
                        "noisy_ssim": noisy_ssim,
                        "denoised_ssim": denoised_ssim,
                        "inference_time_ms": elapsed * 1000
                    })

                print(
                    f"[{index + 1:03d}/{len(image_paths)}] "
                    f"{image_path.name} completed"
                )

            # ==================================
            # Save condition summary
            # ==================================

            avg_noisy_psnr = statistics.mean(noisy_psnr_values)
            avg_noisy_ssim = statistics.mean(noisy_ssim_values)

            for model_name, metrics in model_metrics.items():

                avg_psnr = statistics.mean(metrics["psnr"])
                avg_ssim = statistics.mean(metrics["ssim"])
                avg_time = statistics.mean(metrics["time"])

                summary_results.append({
                    "noise_type": noise_type,
                    "severity": level,
                    "model": model_name,
                    "images_evaluated": len(image_paths),
                    "noisy_psnr": avg_noisy_psnr,
                    "denoised_psnr": avg_psnr,
                    "psnr_improvement": avg_psnr - avg_noisy_psnr,
                    "noisy_ssim": avg_noisy_ssim,
                    "denoised_ssim": avg_ssim,
                    "ssim_improvement": avg_ssim - avg_noisy_ssim,
                    "average_inference_time_ms": avg_time*1000
                })

                print(f"\n{model_name}")
                print(f"Average PSNR: {avg_psnr:.4f} dB")
                print(f"Average SSIM: {avg_ssim:.4f}")
                print(f"Inference time: {avg_time:.3f} ms/image")

    # ======================================
    # SAVE CSV FILES
    # ======================================

    detailed_path = RESULTS_DIR / "all_models_detailed_results.csv"
    summary_path = RESULTS_DIR / "all_models_summary.csv"

    with open(detailed_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=detailed_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(detailed_results)

    with open(summary_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary_results)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("Total experiments:", total_experiments)
    print("Images per experiment:", len(image_paths))
    print("Total model predictions:", len(detailed_results))

    print("\nDetailed results:")
    print(detailed_path)

    print("\nSummary results:")
    print(summary_path)

    print("=" * 60)


if __name__ == "__main__":
    main()
