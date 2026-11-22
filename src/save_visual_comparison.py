
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "DIV2K_valid_HR"
RESULTS_DIR = PROJECT_ROOT / "results"

CNN_CHECKPOINT = PROJECT_ROOT / "models" / "denoising_gaussian_medium.pth"
DNCNN_CHECKPOINT = PROJECT_ROOT / "models" / "denoise_gaussian.pt"

# Evaluation settings (same as the 100-image comparison)
IMAGE_SIZE = (128, 128)
SIGMA = 25 / 255.0
BASE_SEED = 42

# Select a validation image
IMAGE_NAME = "0874.png"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import DenoisingCNN
from dncnn_model import DnCNN
from noise import add_gaussian_noise


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_path = DATASET_DIR / IMAGE_NAME

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    clean_image = Image.open(image_path).convert("RGB")
    clean_tensor = transform(clean_image).unsqueeze(0).to(device)

    # Generate one noisy input shared by both models
    image_index = int(IMAGE_NAME.split(".")[0]) - 801
    noisy_tensor = add_gaussian_noise(
        clean_tensor,
        sigma=SIGMA,
        seed=BASE_SEED + image_index
    )

    # Load Basic CNN
    cnn_model = DenoisingCNN().to(device)
    cnn_checkpoint = torch.load(
        CNN_CHECKPOINT, map_location=device, weights_only=True
    )
    cnn_model.load_state_dict(cnn_checkpoint, strict=True)
    cnn_model.eval()

    # Load DnCNN
    dncnn_model = DnCNN().to(device)
    dncnn_checkpoint = torch.load(
        DNCNN_CHECKPOINT, map_location=device, weights_only=False
    )
    dncnn_model.load_state_dict(dncnn_checkpoint["model"], strict=True)
    dncnn_model.eval()

    # Run both models on exactly the same noisy input
    with torch.no_grad():
        cnn_output = torch.clamp(cnn_model(noisy_tensor), 0, 1)
        dncnn_output = torch.clamp(dncnn_model(noisy_tensor), 0, 1)

    # Convert tensors to images for plotting
    def to_image(tensor):
        return tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

    images = [
        to_image(clean_tensor),
        to_image(noisy_tensor),
        to_image(cnn_output),
        to_image(dncnn_output)
    ]

    titles = [
        "Original Clean Image",
        "Noisy Image",
        "Basic CNN Output",
        "DnCNN Output"
    ]

    # Create four-panel comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"Gaussian Denoising Comparison - {IMAGE_NAME}",
        fontsize=14
    )
    plt.tight_layout()

    output_path = RESULTS_DIR / "four_panel_comparison_0874.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nComparison image saved successfully!")
    print(output_path)


if __name__ == "__main__":
    main()
