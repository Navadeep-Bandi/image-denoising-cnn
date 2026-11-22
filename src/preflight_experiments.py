"""Quick pre-flight validation for the five denoising experiments.

Run from project root: python src/preflight_experiments.py
Optional: python src/preflight_experiments.py --device cpu
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import experiment_utils as eu


def check(name, fn, failures):
    try:
        detail = fn()
        print(f"[PASS] {name}" + (f": {detail}" if detail is not None else ""))
    except Exception as exc:
        failures.append((name, repr(exc)))
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "cuda" else "cpu" if args.device == "cpu" else eu.DEVICE)
    failures = []
    print(f"Project: {ROOT}\nDevice: {device}\nModel directory: {eu.MODEL_DIR}\n")

    hashes = {}
    def hash_check():
        nonlocal hashes
        hashes = eu.checkpoint_hashes()
        expected = [eu.DETECTOR_CHECKPOINT, *eu.DNCNN_CHECKPOINTS.values(),
                    *[f for group in eu.BASIC_CHECKPOINTS.values() for f in group]]
        missing = [f for f in expected if f not in hashes]
        if missing: raise FileNotFoundError(f"Missing checkpoints: {missing}")
        return f"{len(hashes)} checkpoint hashes calculated; SHA-256 recorded"
    check("Checkpoint inventory and SHA-256", hash_check, failures)

    models = {}
    def load_check():
        nonlocal models
        models = eu.load_all_project_checkpoints(device=device)
        # Exercise one real forward pass and the shared tiled-inference path.
        test = torch.rand((1, 3, 128, 128), device=device)
        with torch.inference_mode():
            output = models["dncnn_gaussian"](test).clamp(0, 1)
        if output.shape != test.shape or not torch.isfinite(output).all():
            raise AssertionError("DnCNN forward smoke test failed")
        tiled = eu.run_patch_inference(torch.rand((3, 160, 176)), models["dncnn_gaussian"],
                                       patch_size=128, overlap=32, blend="hann", device=device)
        if tuple(tiled.shape) != (3, 160, 176) or not torch.isfinite(tiled).all():
            raise AssertionError("full-resolution patch inference smoke test failed")
        return f"strictly loaded {len(models)} models; DnCNN forward + tiled inference passed"
    check("Strict checkpoint loading (detector + DnCNN + Basic CNN)", load_check, failures)

    def noise_check():
        clean = torch.linspace(0, 1, 3*64*64).reshape(3, 64, 64)
        for kind in ("gaussian", "sp", "speckle"):
            a = eu.add_noise(clean, kind, "medium", seed=1234)
            b = eu.add_noise(clean, kind, "medium", seed=1234)
            if not torch.equal(a, b): raise AssertionError(f"{kind} noise is not deterministic")
            if a.min() < 0 or a.max() > 1: raise AssertionError(f"{kind} noise outside [0,1]")
            if kind != "sp":
                levels = a * 255
                if not torch.allclose(levels, levels.round(), atol=1e-5):
                    raise AssertionError(f"{kind} noise is not 8-bit quantized")
        # Explicitly verify the same 8-bit quantization operation used by app.py.
        x = torch.tensor([-0.1, 0.001, 0.5, 1.2])
        expected = torch.round(x.clamp(0, 1)*255)/255
        if not torch.equal(eu.quantize_8bit(x), expected): raise AssertionError("quantizer mismatch")
        return "deterministic Gaussian/SP/speckle; Gaussian and speckle are clipped and 8-bit rounded"
    check("Shared noise generation and 8-bit rounding", noise_check, failures)

    def coordinate_check():
        cases = [(128, [0]), (300, [0, 96, 172]), (384, [0, 96, 192, 256]), (80, [0])]
        for length, expected in cases:
            got = eu.get_patch_positions(length, 128, 32)
            if got != expected: raise AssertionError(f"length={length}: {got} != {expected}")
        coords = eu.patch_coordinates(300, 384, 128, 32)
        if len(coords) != 12: raise AssertionError(f"Expected 12 patches, got {len(coords)}")
        return f"boundary-shifted positions verified; 300x384 produces {len(coords)} patches"
    check("Actual patch coordinate generation", coordinate_check, failures)

    def blending_check():
        size, patch, overlap = 300, 128, 32
        base = torch.zeros((3, size, size))
        coords = eu.patch_coordinates(size, size, patch, overlap)
        patches = [torch.ones((3, patch, patch)) * (i+1) for i in range(len(coords))]
        for mode in ("hann", "uniform"):
            out = eu.reconstruct_patches(base, patches, coords, patch, mode, 0.05)
            if not torch.isfinite(out).all(): raise AssertionError(f"{mode} output has non-finite values")
        # No-overlap path: each patch location is tiled without overlap.
        no_coords = eu.patch_coordinates(size, size, patch, 0)
        no_patches = [torch.ones((3, patch, patch)) for _ in no_coords]
        out = eu.reconstruct_patches(base, no_patches, no_coords, patch, "uniform")
        if not torch.allclose(out, torch.ones_like(out)): raise AssertionError("No-overlap reconstruction failed")
        win = eu.blending_window(patch, "hann", 0.05, "cpu")
        if float(win.min()) < (0.05 ** 2) - 1e-7: raise AssertionError("2D Hann outer-product minimum below 0.05^2")
        return f"1D Hann window clamped at 0.05 (2D outer-product floor 0.0025), uniform overlap, and non-overlap reconstruction passed"
    check("Hann / uniform / no-overlap blending", blending_check, failures)

    def mask_check():
        boundary, interior = eu.make_boundary_interior_masks(300, 384, 128, 32, 8)
        if boundary.dtype != torch.bool or interior.dtype != torch.bool: raise AssertionError("Masks must be boolean")
        if not torch.equal(boundary | interior, torch.ones_like(boundary)): raise AssertionError("Masks do not cover image")
        if torch.any(boundary & interior): raise AssertionError("Masks overlap")
        if not boundary.any() or not interior.any(): raise AssertionError("Empty region mask")
        return f"8px seam mask: {int(boundary.sum())} boundary / {int(interior.sum())} interior pixels"
    check("Boundary/interior masks", mask_check, failures)

    def metrics_check():
        ref = np.zeros((64, 64, 3), dtype=np.float32)
        pred = ref.copy()
        psnr, ssim = eu.calculate_psnr_ssim(ref, pred)
        if not np.isinf(psnr) or not np.isclose(ssim, 1.0): raise AssertionError(f"identity metrics unexpected: {psnr}, {ssim}")
        mask, inside = eu.make_boundary_interior_masks(64, 64, 32, 8, 8)
        bp, bs = eu.calculate_masked_psnr_ssim(ref, pred, mask)
        if not np.isinf(bp) or not np.isclose(bs, 1.0): raise AssertionError("masked metrics identity check failed")
        return "PSNR/SSIM and masked-region metrics return expected identity values"
    check("PSNR / SSIM calculations", metrics_check, failures)

    def performance_check():
        x = torch.ones((1, 3, 8, 8), device=device)
        result, elapsed = eu.timed_inference(lambda z: z + 1, x, device=device)
        if not torch.all(result == 2) or elapsed < 0: raise AssertionError("timing helper failed")
        mem = eu.memory_usage_mb(device=device)
        if not isinstance(mem["cpu_rss_mb"], float): raise AssertionError("psutil RSS missing")
        eu.reset_peak_memory(device=device)
        return f"perf_counter + CUDA synchronization; memory={mem}"
    check("Timing and CPU/GPU memory helpers", performance_check, failures)

    if failures:
        print(f"\nPREFLIGHT FAILED: {len(failures)} check(s)")
        for name, error in failures: print(f" - {name}: {error}")
        return 1
    print("\nALL PREFLIGHT CHECKS PASSED")
    print("Checkpoint SHA-256 manifest:")
    for filename, digest in sorted(hashes.items()): print(f"  {filename}: {digest}")
    manifest_dir = ROOT / "results" / "experiments"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "preflight_checkpoint_sha256.json"
    manifest_path.write_text(__import__("json").dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(f"\nHash manifest saved: {manifest_path.relative_to(ROOT)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
