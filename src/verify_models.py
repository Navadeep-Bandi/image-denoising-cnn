
import os
import torch

from model import DenoisingCNN
from dncnn_model import DnCNN

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

basic_models = [
    "denoising_gaussian_low.pth",
    "denoising_gaussian_medium.pth",
    "denoising_gaussian_high.pth",
    "denoising_salt-and-pepper_low.pth",
    "denoising_salt-and-pepper_medium.pth",
    "denoising_salt-and-pepper_high.pth",
    "denoising_speckle_low.pth",
    "denoising_speckle_medium.pth",
    "denoising_speckle_high.pth",
]

dncnn_models = [
    "denoise_gaussian.pt",
    "denoise_sp.pt",
    "denoise_speckle.pt",
]

print("Device:", device)
print("\nChecking model checkpoints...\n")

success = 0
failed = 0

for filename in basic_models + dncnn_models:
    path = os.path.join(MODEL_DIR, filename)

    try:
        if filename in basic_models:
            model = DenoisingCNN().to(device)
            checkpoint = torch.load(
                path, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint, strict=True)

        else:
            model = DnCNN().to(device)
            checkpoint = torch.load(
                path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model"], strict=True)

        model.eval()

        with torch.no_grad():
            test_input = torch.rand(1, 3, 128, 128, device=device)
            output = model(test_input)

        assert output.shape == test_input.shape

        print("[OK]", filename)
        success += 1

    except Exception as error:
        print("[FAILED]", filename)
        print("Reason:", error)
        failed += 1

print("\n========== VERIFICATION SUMMARY ==========")
print("Successfully loaded:", success)
print("Failed:", failed)
print("Total:", len(basic_models) + len(dncnn_models))
