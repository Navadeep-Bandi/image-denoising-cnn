
import torch


def add_gaussian_noise(clean_image, sigma=25/255.0, seed=42):
    """Add reproducible Gaussian noise."""
    generator = torch.Generator(device=clean_image.device)
    generator.manual_seed(seed)

    noise = torch.randn(
        clean_image.shape,
        generator=generator,
        device=clean_image.device,
        dtype=clean_image.dtype
    ) * sigma

    return torch.clamp(clean_image + noise, 0.0, 1.0)



def add_salt_pepper_noise(clean_image, amount=0.03, seed=42):
    noisy = clean_image.clone()

    c, h, w = noisy.shape

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    salt_mask = torch.rand(
        (h, w), generator=generator
    ) < (amount / 2)

    pepper_mask = torch.rand(
        (h, w), generator=generator
    ) < (amount / 2)

    for i in range(c):
        noisy[i][salt_mask] = 1.0
        noisy[i][pepper_mask] = 0.0

    return noisy
def add_speckle_noise(clean_image, variance=0.04, seed=42):
    """Add reproducible multiplicative speckle noise."""
    generator = torch.Generator(device=clean_image.device)
    generator.manual_seed(seed)

    noise = torch.randn(
        clean_image.shape,
        generator=generator,
        device=clean_image.device,
        dtype=clean_image.dtype
    ) * (variance ** 0.5)

    noisy_image = clean_image + clean_image * noise

    return torch.clamp(noisy_image, 0.0, 1.0)
