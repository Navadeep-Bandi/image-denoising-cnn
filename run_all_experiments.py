"""Run all five evaluation experiments and write consolidated summary tables."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def run_step(script_rel: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / script_rel), *extra_args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evaluations 1-5")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "DIV2K_valid_HR")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    common_args = [
        "--dataset-dir", str(args.dataset_dir),
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if args.limit is not None:
        common_args.extend(["--limit", str(args.limit)])

    run_step("src/evaluate_blending.py", common_args)
    run_step("src/evaluate_boundaries.py", common_args)
    run_step("src/evaluate_overlap.py", common_args)
    run_step("src/evaluate_efficiency.py", common_args)
    run_step("src/evaluate_noise_types.py", common_args)

    summary_dir = ROOT / "results" / "experiments" / "final_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    exp_specs = [
        ("exp1_blending", ROOT / "results" / "experiments" / "exp1" / "exp1_blending_comparison.csv", ["config"]),
        ("exp2_boundaries", ROOT / "results" / "experiments" / "exp2" / "exp2_boundary_interior.csv", ["noise_type"]),
        ("exp3_overlap", ROOT / "results" / "experiments" / "exp3" / "exp3_overlap_sweep.csv", ["overlap_pct"]),
        ("exp4_efficiency", ROOT / "results" / "experiments" / "exp4" / "exp4_performance_summary.csv", ["config"]),
        ("exp5_restoration", ROOT / "results" / "experiments" / "exp5" / "exp5_restoration_results.csv", ["noise_type"]),
    ]

    combined = []
    for name, path, group_cols in exp_specs:
        if path.exists():
            df = pd.read_csv(path)
            num_cols = [c for c in df.select_dtypes(include="number").columns if c not in group_cols]
            grouped = df.groupby(group_cols, as_index=False)[num_cols].mean()
            grouped.insert(0, "experiment", name)
            grouped.insert(1, "group_key", grouped[group_cols[0]].astype(str))
            grouped = grouped.drop(columns=group_cols)
            combined.append(grouped)

    if combined:
        out_csv = summary_dir / "consolidated_metrics.csv"
        pd.concat(combined, ignore_index=True).to_csv(out_csv, index=False)
        print(f"Consolidated summary saved to {out_csv}")

if __name__ == "__main__":
    main()