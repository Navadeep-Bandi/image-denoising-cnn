"""Create fig/failure-cases.png (clean / noisy / restored / error map) for the lowest-PSNR cases.
Run from the repository root (needs torch + dataset/DIV2K_valid_HR):
    python analysis/make_failure_figure.py        (copy this file into ./ first)
Uses the project's own helpers (src/experiment_utils.py); seeds follow evaluate_blending.py.
NOTE: written against the project source but not executed by the author of this file.
"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]   # repository root
sys.path.insert(0, str(ROOT / 'src'))
import experiment_utils as eu

DATA = ROOT / 'dataset' / 'DIV2K_valid_HR'
CASES = [('0807.png', 'gaussian', 'high'), ('0804.png', 'gaussian', 'high'), ('0805.png', 'gaussian', 'high')]
SEV = ['low', 'medium', 'high']; CROP = 256
dev = eu.DEVICE
fig, ax = plt.subplots(len(CASES), 4, figsize=(11, 2.9 * len(CASES)))
for r, (name, kind, sev) in enumerate(CASES):
    img = np.asarray(Image.open(DATA / name).convert('RGB'), dtype=np.float32) / 255
    clean = torch.from_numpy(img).permute(2, 0, 1)
    idx = int(name[:4]) - 801                      # first 20 images, sorted
    noisy = eu.add_noise(clean, kind, sev, seed=42 + idx * 100 + SEV.index(sev) * 10)
    model = eu.load_dncnn(kind, device=dev)
    out = eu.run_patch_inference(noisy, model, 128, 32, 'hann', 0.05, dev)
    err = (out - clean).abs().mean(0).numpy()
    psnr = 10 * np.log10(1 / ((out - clean) ** 2).mean().item())
    h, w = err.shape                               # crop around the worst 256x256 block
    best = max(((y, x) for y in range(0, h - CROP, 128) for x in range(0, w - CROP, 128)),
               key=lambda p: err[p[0]:p[0] + CROP, p[1]:p[1] + CROP].mean())
    y, x = best; sl = (slice(None), slice(y, y + CROP), slice(x, x + CROP))
    for c, (t, im) in enumerate([('clean', clean[sl]), ('noisy', noisy[sl]), ('restored', out[sl])]):
        ax[r, c].imshow(im.permute(1, 2, 0).numpy()); ax[r, c].set_title(f'{name[:4]} {t}', fontsize=9)
    ax[r, 3].imshow(err[y:y + CROP, x:x + CROP], cmap='magma', vmin=0, vmax=0.2)
    ax[r, 3].set_title(f'|error| ({kind} {sev}, PSNR {psnr:.2f} dB)', fontsize=9)
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); out = ROOT / 'paper' / 'fig'; out.mkdir(exist_ok=True); plt.savefig(out / 'failure-cases.png', dpi=200)
print('saved fig/failure-cases.png')
