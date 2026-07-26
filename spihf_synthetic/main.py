"""
main.py
=======
Single entry-point that orchestrates the complete SPIHF synthetic-data
pipeline from raw CSV to final reports and visualisations.

Usage (from the project root)::

    python -m spihf_synthetic.main

Pipeline stages:
  1. Load & preprocess the original SPIHF_Data.csv.
  2. Compute per-material statistics.
  3. Generate physics-informed synthetic dataset.
  4. Save synthetic CSV.
  5. Run statistical validation.
  6. Generate comparative visualisations.
  7. Compile publication-quality Markdown reports.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from spihf_synthetic.augmentation import (
    compute_feature_statistics,
    generate_synthetic_dataset,
    load_data,
    preprocess_data,
    save_synthetic_data,
)
from spihf_synthetic.config import NUM_SYNTHETIC_SAMPLES, RANDOM_SEED
from spihf_synthetic.reporting import run_reporting
from spihf_synthetic.validation import run_validation
from spihf_synthetic.visualization import run_visualization


def main(
    data_path: str = "SPIHF_Data.csv",
    output_dir: str = "outputs",
    n_samples: int = NUM_SYNTHETIC_SAMPLES,
) -> None:
    """Run the full SPIHF synthetic-data pipeline.

    Parameters
    ----------
    data_path : str
        Path to the raw ``SPIHF_Data.csv``.
    output_dir : str
        Root directory for all outputs.
    n_samples : int
        Number of synthetic samples to generate.

    Returns
    -------
    None
    """
    np.random.seed(RANDOM_SEED)
    start = time.time()

    # ── Ensure output directories exist ────────────────────────────
    synth_csv = os.path.join(output_dir, "synthetic_SPIHF.csv")
    reports_dir = os.path.join(output_dir, "reports")
    figures_dir = os.path.join(output_dir, "figures")
    for d in [output_dir, reports_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 72)
    print("  SPIHF Synthetic Data Augmentation Pipeline")
    print("=" * 72)
    print(f"  Data source : {data_path}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Target size : {n_samples}")
    print(f"  Random seed : {RANDOM_SEED}")
    print("=" * 72 + "\n")

    # ── Stage 1 : Load & preprocess ────────────────────────────────
    print("[Stage 1/7] Loading and preprocessing data ...")
    raw_df = load_data(data_path)
    clean_df = preprocess_data(raw_df)

    # ── Stage 2 : Feature statistics ───────────────────────────────
    print("\n[Stage 2/7] Computing per-material feature statistics ...")
    stats = compute_feature_statistics(clean_df)

    # ── Stage 3 : Generate synthetic data ──────────────────────────
    print("\n[Stage 3/7] Generating synthetic dataset ...")
    synth_df = generate_synthetic_dataset(
        clean_df, stats, target_size=n_samples
    )

    # ── Stage 4 : Save synthetic CSV ───────────────────────────────
    print("\n[Stage 4/7] Saving synthetic dataset ...")
    save_synthetic_data(synth_df, filepath=synth_csv)

    # ── Stage 5 : Statistical validation ───────────────────────────
    print("\n[Stage 5/7] Running statistical validation ...")
    summary = run_validation(
        real_path=data_path,
        synth_path=synth_csv,
        report_path=os.path.join(reports_dir, "validation_report.txt"),
        json_path=os.path.join(reports_dir, "validation_metrics.json"),
    )

    # ── Stage 6 : Visualisations ──────────────────────────────────
    print("\n[Stage 6/7] Generating comparative visualisations ...")
    run_visualization(
        real_path=data_path,
        synth_path=synth_csv,
        output_dir=figures_dir,
    )

    # ── Stage 7 : Reports ─────────────────────────────────────────
    print("\n[Stage 7/7] Compiling publication-quality reports ...")
    run_reporting(
        real_path=data_path,
        synth_path=synth_csv,
        metrics_path=os.path.join(reports_dir, "validation_metrics.json"),
        output_dir=reports_dir,
    )

    elapsed = time.time() - start
    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)
    print(f"  Total time         : {elapsed:.1f} s")
    print(f"  Synthetic samples  : {len(synth_df)}")
    print(f"  Overall grade      : {summary.get('grade', '?')} "
          f"({summary.get('composite_score', 0):.1f}%)")
    print(f"  KS pass rate       : "
          f"{summary.get('ks_pass_rate', 0) * 100:.1f}%")
    print(f"  Outputs saved to   : {os.path.abspath(output_dir)}/")
    print("=" * 72)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SPIHF Physics-Informed Synthetic Data Augmentation Pipeline",
    )
    parser.add_argument(
        "--data", default="SPIHF_Data.csv",
        help="Path to the raw SPIHF_Data.csv (default: SPIHF_Data.csv)",
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Root directory for all outputs (default: outputs)",
    )
    parser.add_argument(
        "--samples", type=int, default=NUM_SYNTHETIC_SAMPLES,
        help=f"Number of synthetic samples to generate (default: {NUM_SYNTHETIC_SAMPLES})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the random seed (default: use config.RANDOM_SEED)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        from spihf_synthetic import config as _cfg
        _cfg.RANDOM_SEED = args.seed

    main(data_path=args.data, output_dir=args.output, n_samples=args.samples)
