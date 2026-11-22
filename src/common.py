import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"

LABELS = ["gaussian", "sp", "speckle"]          # classifier output order
APPLY_ORDER = ["speckle", "gaussian", "sp"]     # order noise is ADDED
REMOVE_ORDER = APPLY_ORDER[::-1]                # order noise is REMOVED

# strengths used for TRAINING (random in this range)
RANGES = {"gaussian": (5 / 255, 60 / 255), "speckle": (0.005, 0.10), "sp": (0.005, 0.07)}
# strengths used for TESTING (your table): Low / Medium / High
LEVELS = {"gaussian": [10 / 255, 25 / 255, 50 / 255],
          "speckle": [0.01, 0.04, 0.08],
          "sp": [0.01, 0.03, 0.05]}
LEVEL_NAMES = ["Low", "Medium", "High"]

# which noise types are present (order = LABELS: gaussian, sp, speckle)
SUBSETS = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                       dtype=torch.float32, device=DEVICE)
SUBSET_P = torch.tensor([.15, .15, .15, .15, .08, .08, .08, .16], device=DEVICE)
SUBSET_NAMES = ["Clean", "Gaussian", "Salt&Pepper", "Speckle",
                "Gauss+S&P", "Gauss+Speckle", "S&P+Speckle", "All three"]


def set_seed(s, device=None):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    target_dev = torch.device(device) if device is not None else DEVICE
    if target_dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ------------------------------------------------------------- data
def list_images():
    files = sorted(f for f in CLEAN_DIR.iterdir()
                   if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(files) < 3:
        raise FileNotFoundError(f"Need at least 3 images in {CLEAN_DIR}")
    return files


def split_files(files=None):
    """Same split as the first classifier (seed 42): 70/15/15 by source image."""
    files = list(files or list_images())
    np.random.default_rng(SEED).shuffle(files)
    n = len(files)
    n_val = max(1, round(0.15 * n))
    n_test = max(1, round(0.15 * n))
    return files[n_val + n_test:], files[:n_val], files[n_val:n_val + n_test]


def load_pool(files, per_img):
    """Random 256x256 crops of each image, kept on the GPU as uint8."""
    rng = np.random.default_rng(SEED)
    crops = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None or min(img.shape[:2]) < 256:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        for _ in range(per_img):
            y, x = rng.integers(0, h - 255), rng.integers(0, w - 255)
            crops.append(img[y:y + 256, x:x + 256])
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(files)} images")
    arr = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).contiguous()
    return arr.to(DEVICE)


def sample_clean(pool, B, size, train=True):
    n = pool.shape[0]
    idx = np.random.randint(0, n, B)
    ys = np.random.randint(0, 256 - size + 1, B)
    xs = np.random.randint(0, 256 - size + 1, B)
    out = torch.stack([pool[i, :, y:y + size, x:x + size]
                       for i, y, x in zip(idx, ys, xs)]).float() / 255
    if train:
        if np.random.rand() < 0.5:
            out = out.flip(3)
        if np.random.rand() < 0.5:
            out = out.flip(2)
    return out


# ------------------------------------------------------------- noise (GPU)
def strength(kind, B, device, level=None):
    if level is not None:
        return torch.full((B,), LEVELS[kind][level], device=device)
    lo, hi = RANGES[kind]
    return lo + (hi - lo) * torch.rand(B, device=device)


def q(x):  # clip + 8-bit rounding, like a saved image
    return torch.round(x.clamp(0, 1) * 255) / 255


def apply_noise(kind, x, s):
    s = s.view(-1, 1, 1, 1)
    if kind == "gaussian":                       # additive, sigma = s
        return q(x + torch.randn_like(x) * s)
    if kind == "speckle":                        # multiplicative, variance = s
        return q(x + x * torch.randn_like(x) * s.sqrt())
    if kind == "sp":                             # s = fraction of pixels hit
        r = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = torch.where(r < s / 2, torch.zeros_like(x), x)
        x = torch.where((r >= s / 2) & (r < s), torch.ones_like(x), x)
        return q(x)
    raise ValueError(kind)


def make_pair(kind, clean, level=None, prior_p=0.5):
    """(noisy input, target) for the denoiser of `kind`.
    The target is the image BEFORE this noise was added; earlier noises in
    APPLY_ORDER are present in it with probability prior_p."""
    B, dev = clean.shape[0], clean.device
    target = clean
    for k in APPLY_ORDER:
        if k == kind:
            break
        m = (torch.rand(B, 1, 1, 1, device=dev) < prior_p).float()
        target = m * apply_noise(k, target, strength(k, B, dev)) + (1 - m) * target
    return apply_noise(kind, target, strength(kind, B, dev, level)), target


@torch.no_grad()
def partial_clean(x, present, denoisers, frac):
    """Run the real denoisers (removal order) on a random fraction of the batch."""
    B, dev = x.shape[0], x.device
    x, present = x.clone(), present.clone()
    passes = torch.zeros(B, dtype=torch.long, device=dev)
    sel = torch.rand(B, device=dev) < frac
    passes[sel] = torch.randint(1, 3, (int(sel.sum()),), device=dev)
    for j in (1, 2):
        nxt = torch.full((B,), -1, dtype=torch.long, device=dev)
        for k in REMOVE_ORDER:
            c = LABELS.index(k)
            nxt = torch.where((nxt < 0) & (present[:, c] > 0.5),
                              torch.full_like(nxt, c), nxt)
        for c, k in enumerate(LABELS):
            idx = ((passes >= j) & (nxt == c)).nonzero().flatten()
            if len(idx):
                with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                    out = denoisers[k](x[idx])
                x[idx] = out.float().clamp(0, 1)
                present[idx, c] = 0
    return x, present


def make_cls_batch(pool, B, size, denoisers=None, frac=0.0,
                   level=None, present=None, train=True):
    """Noisy images + multi-label targets (gaussian, sp, speckle)."""
    x = sample_clean(pool, B, size, train)
    if present is None:
        present = SUBSETS[torch.multinomial(SUBSET_P, B, replacement=True)]
    for k in APPLY_ORDER:
        m = present[:, LABELS.index(k)].view(-1, 1, 1, 1)
        x = m * apply_noise(k, x, strength(k, B, x.device, level)) + (1 - m) * x
    if denoisers is not None and frac > 0:
        x, present = partial_clean(x, present, denoisers, frac)
    return x, present


# ------------------------------------------------------------- models
class DnCNN(nn.Module):
    """Residual denoiser: predicts the noise and subtracts it."""
    def __init__(self, depth=17, ch=64):
        super().__init__()
        L = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(True)]
        for _ in range(depth - 2):
            L += [nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(ch), nn.ReLU(True)]
        L += [nn.Conv2d(ch, 3, 3, padding=1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return x - self.net(x)          # clamp to [0,1] after calling


def _cb(i, o):
    return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]


class NoiseDetector(nn.Module):
    """Multi-label noise detector. Input = image + high-pass (noise-only) view."""
    def __init__(self):
        super().__init__()
        L = (_cb(6, 48) + _cb(48, 48) + [nn.MaxPool2d(2)]
             + _cb(48, 96) + _cb(96, 96) + [nn.MaxPool2d(2)]
             + _cb(96, 160) + _cb(160, 160) + [nn.MaxPool2d(2)]
             + _cb(160, 256))
        self.features = nn.Sequential(*L)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 3))

    def forward(self, x):
        blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
        f = self.features(torch.cat([x, (x - blur) * 4], 1))
        return self.head(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))


def load_denoisers():
    out = {}
    for k in LABELS:
        p = MODEL_DIR / f"denoise_{k}.pt"
        if not p.exists():
            return None
        m = DnCNN().to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE)["model"])
        m.eval()
        out[k] = m
    return out


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1 / mse)).mean().item()
