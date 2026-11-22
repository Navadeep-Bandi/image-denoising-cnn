"""Shared, reproducible utilities for the image-denoising experiments.

This module deliberately does not import or modify app.py. Noise generation mirrors
app.add_project_noise: CPU-seeded Gaussian/speckle noise, shared-channel salt/pepper
mask, clipping and 8-bit rounding for Gaussian and speckle.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_LEVELS = {
    "gaussian": {"low": 10 / 255.0, "medium": 25 / 255.0, "high": 50 / 255.0},
    "sp": {"low": 0.01, "medium": 0.03, "high": 0.05},
    "speckle": {"low": 0.01, "medium": 0.04, "high": 0.08},
}
BASIC_CHECKPOINTS = {
    "gaussian": ["denoising_gaussian_low.pth", "denoising_gaussian_medium.pth", "denoising_gaussian_high.pth"],
    "sp": ["denoising_salt-and-pepper_low.pth", "denoising_salt-and-pepper_medium.pth", "denoising_salt-and-pepper_high.pth"],
    "speckle": ["denoising_speckle_low.pth", "denoising_speckle_medium.pth", "denoising_speckle_high.pth"],
}
DNCNN_CHECKPOINTS = {k: f"denoise_{k}.pt" for k in ("gaussian", "sp", "speckle")}
DETECTOR_CHECKPOINT = "noise_detector.pt"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: os.PathLike | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hashes(model_dir: Path | str = MODEL_DIR) -> Dict[str, str]:
    root = Path(model_dir)
    return {p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in {".pt", ".pth"}}


def _torch_load(path: Path, device: torch.device):
    # Project checkpoints are locally supplied trusted files and some contain metadata.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced
        return torch.load(path, map_location=device)


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_dncnn(noise_type: str, model_dir: Path | str = MODEL_DIR, device=DEVICE):
    from common import DnCNN
    key = noise_type.lower()
    path = Path(model_dir) / DNCNN_CHECKPOINTS[key]
    model = DnCNN(depth=17, ch=64).to(device)
    model.load_state_dict(_state_dict(_torch_load(path, device)), strict=True)
    model.eval()
    return model


def load_detector(model_dir: Path | str = MODEL_DIR, device=DEVICE):
    from common import NoiseDetector, LABELS
    path = Path(model_dir) / DETECTOR_CHECKPOINT
    checkpoint = _torch_load(path, device)
    if isinstance(checkpoint, dict) and "labels" in checkpoint and list(checkpoint["labels"]) != list(LABELS):
        raise ValueError(f"Detector labels mismatch: {checkpoint['labels']} != {LABELS}")
    model = NoiseDetector().to(device)
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    model.eval()
    return model


def load_basic_cnn(checkpoint_name: str, model_dir: Path | str = MODEL_DIR, device=DEVICE):
    from model import DenoisingCNN
    path = Path(model_dir) / checkpoint_name
    model = DenoisingCNN().to(device)
    model.load_state_dict(_state_dict(_torch_load(path, device)), strict=True)
    model.eval()
    return model


def load_all_project_checkpoints(model_dir: Path | str = MODEL_DIR, device=DEVICE) -> Dict[str, torch.nn.Module]:
    """Strictly instantiate and load every detector, DnCNN and Basic-CNN checkpoint."""
    loaded = {"detector": load_detector(model_dir, device)}
    for kind in DNCNN_CHECKPOINTS:
        loaded[f"dncnn_{kind}"] = load_dncnn(kind, model_dir, device)
    for kind, filenames in BASIC_CHECKPOINTS.items():
        for filename in filenames:
            loaded[f"basic_{kind}_{Path(filename).stem.rsplit('_', 1)[-1]}"] = load_basic_cnn(filename, model_dir, device)
    return loaded


def quantize_8bit(image: torch.Tensor) -> torch.Tensor:
    """Clip to [0,1] and round to the nearest representable 8-bit level."""
    return torch.round(image.clamp(0.0, 1.0) * 255.0) / 255.0


def add_noise(clean: torch.Tensor, noise_type: str, severity: str = "medium", seed: int = 42) -> torch.Tensor:
    """Generate noise identically to app.add_project_noise for CHW RGB tensors.

    Gaussian parameter is sigma; speckle parameter is variance; SP parameter is
    fraction of spatial pixels corrupted. Gaussian/speckle are quantized to 8-bit.
    """
    if clean.ndim != 3 or clean.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW tensor, got shape {tuple(clean.shape)}")
    kind = noise_type.lower().replace("salt-and-pepper", "sp").replace("salt_pepper", "sp")
    if kind not in NOISE_LEVELS or severity.lower() not in NOISE_LEVELS[kind]:
        raise ValueError(f"Unsupported noise/severity: {noise_type}/{severity}")
    amount = NOISE_LEVELS[kind][severity.lower()]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    source = clean.detach().to(device="cpu", dtype=torch.float32).clone()
    channels, height, width = source.shape
    if kind == "gaussian":
        noise = torch.randn(source.shape, generator=generator, dtype=source.dtype) * amount
        result = quantize_8bit(source + noise)
    elif kind == "speckle":
        noise = torch.randn(source.shape, generator=generator, dtype=source.dtype) * (amount ** 0.5)
        result = quantize_8bit(source + source * noise)
    else:
        random_values = torch.rand((height, width), generator=generator)
        pepper = random_values < amount / 2
        salt = (random_values >= amount / 2) & (random_values < amount)
        result = source.clone()
        result[:, pepper] = 0.0
        result[:, salt] = 1.0
    return result.to(device=clean.device, dtype=clean.dtype)


def get_patch_positions(length: int, patch_size: int = 128, overlap: int = 32) -> List[int]:
    """Match app.py exactly, including appending the final boundary-aligned patch."""
    if patch_size <= 0 or overlap < 0 or overlap >= patch_size:
        raise ValueError("Require patch_size > 0 and 0 <= overlap < patch_size")
    if length <= patch_size:
        return [0]
    stride = patch_size - overlap
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def patch_coordinates(height: int, width: int, patch_size: int = 128, overlap: int = 32) -> List[Tuple[int, int]]:
    ys = get_patch_positions(height, patch_size, overlap)
    xs = get_patch_positions(width, patch_size, overlap)
    return [(y, x) for y in ys for x in xs]


def blending_window(patch_size: int = 128, method: str = "hann", min_weight: float = 0.05, device=DEVICE) -> torch.Tensor:
    method = method.lower()
    if method == "hann":
        one_d = torch.hann_window(patch_size, periodic=False, device=device).clamp(min=min_weight)
        return torch.outer(one_d, one_d)[None, None]
    if method in {"uniform", "none", "no_overlap"}:
        return torch.ones((1, 1, patch_size, patch_size), device=device)
    raise ValueError(f"Unknown blending method: {method}")


def reconstruct_patches(image: torch.Tensor, patches: Sequence[torch.Tensor], coords: Sequence[Tuple[int, int]],
                        patch_size: int = 128, method: str = "hann", min_weight: float = 0.05) -> torch.Tensor:
    """Weighted reconstruction. image is CHW and patches are CHW in coordinate order."""
    if method.lower() == "none":
        expected_coords = patch_coordinates(height, width, patch_size, overlap=0)
        if list(coords) != expected_coords:
            raise ValueError("method='none' requires non-overlapping patch coordinates (overlap=0)")
    if image.ndim != 3:
        raise ValueError("image must be CHW")
    if len(patches) != len(coords):
        raise ValueError("patches and coords must have equal lengths")
    _, height, width = image.shape
    device, dtype = image.device, image.dtype
    weight = blending_window(patch_size, method, min_weight, device).to(dtype=dtype)
    accum = torch.zeros_like(image)
    denom = torch.zeros((1, height, width), device=device, dtype=dtype)
    for patch, (y, x) in zip(patches, coords):
        if tuple(patch.shape) != (image.shape[0], patch_size, patch_size):
            raise ValueError(f"Patch shape {tuple(patch.shape)} is not {(image.shape[0], patch_size, patch_size)}")
        accum[:, y:y+patch_size, x:x+patch_size] += patch * weight[0]
        denom[:, y:y+patch_size, x:x+patch_size] += weight[0]
    if torch.any(denom <= 0):
        raise RuntimeError("Patch coordinates left uncovered pixels")
    return accum / denom


def make_boundary_interior_masks(height: int, width: int, patch_size: int = 128, overlap: int = 32,
                                 boundary_width: int = 8, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Return boolean masks; boundary is within boundary_width px of any patch edge.

    Patch-edge lines are derived from the actual starts/ends, including the shifted
    final patch. Interior is the complement. The image's outer frame is not treated
    as a patch seam; only internal patch edges are evaluated.
    """
    if boundary_width < 1:
        raise ValueError("boundary_width must be >= 1")
    ys = get_patch_positions(height, patch_size, overlap)
    xs = get_patch_positions(width, patch_size, overlap)
    boundary = torch.zeros((height, width), dtype=torch.bool, device=device)
    y_edges = sorted({p for p in ys if p > 0} | {p + patch_size for p in ys if p + patch_size < height})
    x_edges = sorted({p for p in xs if p > 0} | {p + patch_size for p in xs if p + patch_size < width})
    for edge in y_edges:
        boundary[max(0, edge-boundary_width):min(height, edge+boundary_width), :] = True
    for edge in x_edges:
        boundary[:, max(0, edge-boundary_width):min(width, edge+boundary_width)] = True
    return boundary, ~boundary


def calculate_psnr_ssim(reference, prediction, data_range: float = 1.0, channel_axis: int = 2) -> Tuple[float, float]:
    """Calculate full-image PSNR and SSIM; accepts HWC NumPy or CHW torch inputs."""
    def as_hwc(value):
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        return np.clip(arr, 0.0, 1.0)
    ref, pred = as_hwc(reference), as_hwc(prediction)
    if ref.shape != pred.shape:
        raise ValueError(f"Reference/prediction shapes differ: {ref.shape} vs {pred.shape}")
    psnr = float(peak_signal_noise_ratio(ref, pred, data_range=data_range))
    h, w = ref.shape[:2]
    win = min(7, h if h % 2 else h-1, w if w % 2 else w-1)
    if win < 3:
        ssim = float("nan")
    else:
        ssim = float(structural_similarity(ref, pred, data_range=data_range,
                                           channel_axis=channel_axis if ref.ndim == 3 else None,
                                           win_size=win))
    return psnr, ssim


def calculate_masked_psnr_ssim(reference, prediction, mask: np.ndarray | torch.Tensor,
                                data_range: float = 1.0) -> Tuple[float, float]:
    """PSNR on selected pixels and mean local SSIM map over selected centers."""
    def hwc(v):
        if torch.is_tensor(v): v = v.detach().float().cpu().numpy()
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 3 and a.shape[0] in (1, 3): a = np.moveaxis(a, 0, -1)
        return np.clip(a, 0, 1)
    ref, pred = hwc(reference), hwc(prediction)
    m = mask.detach().cpu().numpy().astype(bool) if torch.is_tensor(mask) else np.asarray(mask, dtype=bool)
    if m.shape != ref.shape[:2] or ref.shape != pred.shape:
        raise ValueError("Mask must match image H,W and reference/prediction shapes")
    if not m.any(): return float("nan"), float("nan")
    diff = (ref - pred)[m]
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))
    _, ssim_map = structural_similarity(ref, pred, data_range=data_range, channel_axis=2,
                                        win_size=min(7, ref.shape[0] if ref.shape[0] % 2 else ref.shape[0]-1,
                                                     ref.shape[1] if ref.shape[1] % 2 else ref.shape[1]-1),
                                        full=True)
    # skimage's multichannel full map may retain a channel dimension.
    if ssim_map.ndim == 3: ssim_map = ssim_map.mean(axis=2)
    return psnr, float(np.mean(ssim_map[m]))


def synchronize(device=DEVICE) -> None:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def timed_inference(fn, *args, device=DEVICE, **kwargs):
    """Return (function_result, elapsed_seconds); caller handles warm-up/repeats."""
    synchronize(device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    synchronize(device)
    return result, time.perf_counter() - start


def memory_usage_mb(process: Optional[psutil.Process] = None, device=DEVICE) -> Dict[str, Optional[float]]:
    """Current process RSS and CUDA allocated/reserved/peak memory in MiB."""
    process = process or psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 ** 2)
    out = {"cpu_rss_mb": float(rss), "gpu_allocated_mb": None,
           "gpu_reserved_mb": None, "gpu_peak_allocated_mb": None}
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        out.update({
            "gpu_allocated_mb": torch.cuda.memory_allocated(dev) / (1024 ** 2),
            "gpu_reserved_mb": torch.cuda.memory_reserved(dev) / (1024 ** 2),
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(dev) / (1024 ** 2),
        })
    return out


def reset_peak_memory(device=DEVICE) -> None:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)


def run_patch_inference(image: torch.Tensor, model: torch.nn.Module, patch_size: int = 128,
                        overlap: int = 32, blend: str = "hann", min_weight: float = 0.05,
                        device=DEVICE) -> torch.Tensor:
    """Full-resolution CHW inference with replicate padding and selectable blending.

    blend='none' forces overlap=0; uniform/hann use the supplied overlap.
    """
    if image.ndim != 3: raise ValueError("image must be CHW")
    if blend == "none": overlap = 0
    c, original_h, original_w = image.shape
    pad_h, pad_w = max(0, patch_size-original_h), max(0, patch_size-original_w)
    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
    _, h, w = padded.shape
    coords = patch_coordinates(h, w, patch_size, overlap)
    patches = []
    model.eval()
    with torch.inference_mode():
        for y, x in coords:
            inp = padded[:, y:y+patch_size, x:x+patch_size].unsqueeze(0).to(device)
            patches.append(model(inp).clamp(0, 1).squeeze(0).to(image.device))
    output = reconstruct_patches(padded, patches, coords, patch_size,
                                 "uniform" if blend == "none" else blend, min_weight)
    return output[:, :original_h, :original_w].cpu()
