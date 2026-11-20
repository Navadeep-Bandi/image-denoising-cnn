# Repository synchronization audit

## Inspected GitHub state

- Repository: `Navadeep-Bandi/image-denoising-cnn` (private; authenticated connection reported push permission).
- Original branch: `main`; it was unprotected at inspection time.
- Original HEAD: `daa352d5477662c59ac1d935a911fcb699099a7c` (`docs: correct Windows virtual environment command`).
- Only one branch existed at inspection: `main`.
- The original tree contained `.gitignore`, `README.md`, `requirements.txt`, `docs/PROJECT_REPORT.md`, `docs/RESEARCH_PAPER.md`, and `results/all_models_summary.csv`. It did not contain `app.py`, `src/`, `models/`, `dataset/`, or `results/experiments/`.
- A separate branch, `sync/reproducibility-audit`, was created from the inspected `main` HEAD. No force push is used; the intended update is a fast-forward commit on that new branch.

## Local archive inspected

The uploaded `image_denoising_v1.zip` contains the application, Python source modules, model checkpoints, experiment CSVs and PNGs, a checkpoint-hash JSON, and `summary_limit20.txt`. It does not contain the DIV2K dataset, the requested 0801–0820 source images, the revised DOCX research paper, or the HTML presentation.

The per-image experiment CSVs contain these rows and image coverage:

- EXP1 blending: 540 rows; 20 distinct names, 0801.png–0820.png.
- EXP2 boundary/interior: 180 rows; same 20 names.
- EXP3 overlap sweep: 900 rows; same 20 names.
- EXP4 efficiency: 900 rows; same 20 names.
- EXP5 noise detection/restoration: 180 rows each; same 20 names.

The supplied summary text was stale relative to the per-image CSVs. `summary_limit20.txt` in this branch is regenerated from the supplied CSVs. The CSV contents support 20-image coverage, but the archive does not contain a run log that independently proves the exact `--limit 20 --seed 42 --device cuda` command, CUDA device identity, software versions, or checkpoint hashes used for these outputs. The runner's defaults alone are not proof of the command used.

## Code audit finding

In the local `src/experiment_utils.py`, `reconstruct_patches()` references `height` and `width` in the `method="none"` guard before assigning them from `image.shape`. This raises `UnboundLocalError`. The corrected ordering is:

```python
if image.ndim != 3:
    raise ValueError("image must be CHW")
if len(patches) != len(coords):
    raise ValueError("patches and coords must have equal lengths")
_, height, width = image.shape
if method.lower() == "none":
    expected_coords = patch_coordinates(height, width, patch_size, overlap=0)
    if list(coords) != expected_coords:
        raise ValueError("method='none' requires non-overlapping patch coordinates (overlap=0)")
```

This fix was applied to the inspected local extraction and tested: Python compilation passed, exact non-overlap reconstruction passed, and overlapping coordinates were rejected. The six requested source modules and the app are not yet present in the GitHub branch because this connector session did not expose a direct local-file-to-repository transfer path for the uploaded archive. Apply/commit the corrected local module and the other reviewed modules from the machine holding the extracted project.

## Current branch additions

This branch is intended to contain the regenerated summary, consolidated aggregate metrics, selected compact experiment summaries, and expanded scratch-file ignore rules. It intentionally does not overwrite or upload `.pt` / `.pth` checkpoints or any dataset folder.

## Still required for a complete reproducible release

1. Upload `app.py`, `run_all_experiments.py`, and the full `src/*.py` tree from the corrected local extraction.
2. Upload all per-image CSVs under `results/experiments/exp1` through `exp5`, plus the PNG plots and `preflight_checkpoint_sha256.json`.
3. Preserve a run manifest with exact command, seed, `torch.cuda.is_available()`, selected CUDA device/name, Python/PyTorch/CUDA versions, and checkpoint SHA-256 values.
4. Upload `noise_adaptive_image_restoration_research_paper_revised_sections_1-6.docx` and `noise_adaptive_restoration_presentation.html`; neither file was among the provided uploads.
5. Keep `dataset/` out of Git. Checkpoint files were deliberately left untouched and are not committed in this branch.
