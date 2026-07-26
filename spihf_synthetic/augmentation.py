"""
augmentation.py
===============
Physics-Informed Synthetic Data Augmentation for SPIHF datasets.

This module provides the complete augmentation pipeline:
  1. Data loading and preprocessing.
  2. Per-material feature statistics.
  3. SMOTE-inspired interpolation.
  4. Gaussian perturbation.
  5. Physics-informed constraint repair (via ``constraints`` module).
  6. Confidence scoring (via ``confidence`` module).
  7. Rejection sampling and near-duplicate removal.
  8. CSV output with confidence column.

Functions
---------
load_data                     Load raw SPIHF CSV.
preprocess_data               Clean, canonicalise, material-map.
compute_feature_statistics    Per-material descriptive statistics.
generate_interpolated_sample  SMOTE-style convex-hull interpolation.
apply_gaussian_noise          Calibrated additive noise.
generate_synthetic_dataset    Full orchestration pipeline.
save_synthetic_data           Write CSV and print summary.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from spihf_synthetic.config import (
    ALPHA_RANGE,
    CATEGORICAL_FEATURES,
    COLUMN_NAMES,
    CONFIDENCE_THRESHOLD,
    DUPLICATE_THRESHOLD,
    MAX_ATTEMPTS_MULTIPLIER,
    MIN_SAMPLES_FOR_INTERPOLATION,
    NOISE_LEVEL,
    NUM_SYNTHETIC_SAMPLES,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    UNIT_SUFFIXES,
)
from spihf_synthetic.confidence import compute_total_confidence
from spihf_synthetic.constraints import repair_sample
from spihf_synthetic.utils import build_material_map, remove_near_duplicates


# ═══════════════════════════════════════════════════════════════════════
#  1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

def load_data(filepath: str = "SPIHF_Data.csv") -> pd.DataFrame:
    """Load the raw SPIHF dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.  Defaults to ``"SPIHF_Data.csv"``.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame exactly as read from disk.

    Notes
    -----
    The raw dataset is compiled from multiple published SPIHF studies and
    therefore contains heterogeneous column naming, mixed units embedded
    in value cells, empty material labels, and trailing whitespace.  All
    of this is handled downstream in ``preprocess_data``.
    """
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {len(df)} rows × {df.shape[1]} columns "
          f"from '{filepath}'.")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  2.  PREPROCESS DATA
# ═══════════════════════════════════════════════════════════════════════

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the raw SPIHF DataFrame.

    Steps performed:

    1. Rename columns to canonical ``COLUMN_NAMES``.
    2. Forward-fill empty *Material* cells.
    3. Strip embedded unit strings from numeric cells.
    4. Normalise *Precut Shape* to lower-case, collapse synonyms.
    5. Coerce all numeric columns to ``float64``.
    6. Drop rows with missing *Material*.
    7. Canonicalise material names via ``build_material_map``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from ``load_data``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for augmentation.
    """
    df = df.copy()

    # ── 1. Rename columns ──────────────────────────────────────────
    if len(df.columns) == len(COLUMN_NAMES):
        df.columns = COLUMN_NAMES
    else:
        rename_map = {old: new for old, new in zip(df.columns, COLUMN_NAMES)}
        df.rename(columns=rename_map, inplace=True)

    # ── 2. Forward-fill Material ───────────────────────────────────
    df["Material"] = df["Material"].replace(r"^\s*$", np.nan, regex=True)
    df["Material"] = df["Material"].ffill()
    df.dropna(subset=["Material"], inplace=True)

    # ── 3. Strip unit strings from numeric cells ───────────────────
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        for suffix in UNIT_SUFFIXES:
            s = s.str.replace(suffix, "", regex=False)
        s = s.str.replace("°", "", regex=False)
        s = s.str.replace("<", "", regex=False)
        s = s.str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")

    # ── 4. Normalise Precut Shape ──────────────────────────────────
    shape_col = "Precut Shape (circle/square/etc)"
    if shape_col in df.columns:
        df[shape_col] = (
            df[shape_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"circular": "circle", "nan": "circle"})
        )

    # ── 5. Material name canonicalisation ──────────────────────────
    df["Material"] = df["Material"].str.strip()
    material_map = build_material_map()
    df["Material"] = df["Material"].map(
        lambda m: material_map.get(m.strip(), m.strip())
    )

    # ── 6. Ensure 'Is lubricant used?' is binary integer ───────────
    lub_col = "Is lubricant used?"
    if lub_col in df.columns:
        df[lub_col] = (
            df[lub_col].fillna(0).astype(float).clip(0, 1).round().astype(int)
        )

    print(f"[preprocess_data] After cleaning: {len(df)} rows, "
          f"{df['Material'].nunique()} unique materials.")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  3.  COMPUTE FEATURE STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def compute_feature_statistics(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, pd.Series]]:
    """Compute per-material descriptive statistics for every numeric feature.

    For each material group the function stores ``mean``, ``std``,
    ``min``, ``max``, ``median``, and ``count``.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.

    Returns
    -------
    Dict[str, Dict[str, pd.Series]]
        Nested dictionary keyed by material name, then by statistic name.

    Notes
    -----
    These statistics anchor both the Gaussian noise amplitude and the
    rejection-sampling bounds.  Using *material-wise* statistics prevents
    impossible cross-material artefacts.
    """
    stats: Dict[str, Dict[str, pd.Series]] = {}
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    for mat, grp in df.groupby("Material"):
        sub = grp[numeric_cols]
        stats[mat] = {
            "mean": sub.mean(),
            "std": sub.std().fillna(0),
            "min": sub.min(),
            "max": sub.max(),
            "median": sub.median(),
            "count": sub.count(),
        }

    print(f"[compute_feature_statistics] Statistics computed for "
          f"{len(stats)} material groups.")
    return stats


# ═══════════════════════════════════════════════════════════════════════
#  4.  SMOTE-INSPIRED INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════

def generate_interpolated_sample(
    row_i: pd.Series,
    row_j: pd.Series,
    alpha_low: float = ALPHA_RANGE[0],
    alpha_high: float = ALPHA_RANGE[1],
) -> pd.Series:
    """Create a synthetic sample by interpolating two same-material rows.

    Implements the SMOTE-inspired formula::

        x_new = α · x_i  +  (1 − α) · x_j

    where α ~ U(alpha_low, alpha_high).

    Parameters
    ----------
    row_i : pd.Series
        First (anchor) sample.
    row_j : pd.Series
        Second (neighbour) sample.
    alpha_low : float
        Lower bound of α.
    alpha_high : float
        Upper bound of α.

    Returns
    -------
    pd.Series
        Interpolated synthetic sample with the same index as ``row_i``.
    """
    alpha: float = np.random.uniform(alpha_low, alpha_high)
    new_sample = row_i.copy()

    for feat in NUMERIC_FEATURES:
        if feat not in row_i.index:
            continue
        vi = row_i.get(feat, np.nan)
        vj = row_j.get(feat, np.nan)
        if pd.notna(vi) and pd.notna(vj):
            new_sample[feat] = alpha * vi + (1.0 - alpha) * vj
        elif pd.notna(vi):
            new_sample[feat] = vi
        elif pd.notna(vj):
            new_sample[feat] = vj

    for cat in CATEGORICAL_FEATURES:
        if cat in row_i.index:
            new_sample[cat] = row_i[cat]

    return new_sample


# ═══════════════════════════════════════════════════════════════════════
#  5.  GAUSSIAN PERTURBATION
# ═══════════════════════════════════════════════════════════════════════

def apply_gaussian_noise(
    sample: pd.Series,
    stats: Dict[str, pd.Series],
    noise_fraction_low: float = NOISE_LEVEL[0],
    noise_fraction_high: float = NOISE_LEVEL[1],
) -> pd.Series:
    """Add Gaussian noise scaled to 1–3 % of the material-group std.

    For each numeric feature *f*::

        ε_f ~ N(0,  σ_noise²)
        σ_noise = η · std_f     where η ~ U(noise_low, noise_high)

    Parameters
    ----------
    sample : pd.Series
        A single (possibly interpolated) sample.
    stats : Dict[str, pd.Series]
        Statistics dict for the sample's material.
    noise_fraction_low : float
        Minimum fraction of std for noise amplitude.
    noise_fraction_high : float
        Maximum fraction of std for noise amplitude.

    Returns
    -------
    pd.Series
        Perturbed sample.
    """
    noisy = sample.copy()
    std_series = stats.get("std", pd.Series(dtype=float))

    for feat in NUMERIC_FEATURES:
        if feat in ("Is lubricant used?", "No of stages"):
            continue
        if feat not in sample.index or pd.isna(sample[feat]):
            continue
        feat_std = std_series.get(feat, 0.0)
        if feat_std == 0 or pd.isna(feat_std):
            continue
        eta = np.random.uniform(noise_fraction_low, noise_fraction_high)
        noise = np.random.normal(0.0, eta * feat_std)
        noisy[feat] = sample[feat] + noise

    return noisy


# ═══════════════════════════════════════════════════════════════════════
#  6.  GENERATE SYNTHETIC DATASET
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, pd.Series]],
    target_size: int = NUM_SYNTHETIC_SAMPLES,
    min_confidence: float = CONFIDENCE_THRESHOLD,
    max_attempts_multiplier: int = MAX_ATTEMPTS_MULTIPLIER,
    duplicate_threshold: float = DUPLICATE_THRESHOLD,
) -> pd.DataFrame:
    """Orchestrate the full augmentation pipeline.

    Pipeline per material group:

    1. Proportional allocation of synthetic samples.
    2. For each sample: interpolate → perturb → repair → score → accept/reject.
    3. Near-duplicate removal.
    4. Gap-filling if below ``target_size``.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed real dataset.
    stats : Dict
        Per-material statistics from ``compute_feature_statistics``.
    target_size : int
        Desired number of synthetic samples.
    min_confidence : float
        Rejection threshold.
    max_attempts_multiplier : int
        Max attempts = required × multiplier.
    duplicate_threshold : float
        Normalised L2 distance for duplicate detection.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with ``confidence_score`` column appended.
    """
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    all_synthetic: List[pd.Series] = []

    materials = df["Material"].unique().tolist()
    n_total = len(df)

    # ── Proportional allocation ────────────────────────────────────
    allocation: Dict[str, int] = {}
    for mat in materials:
        group_size = len(df[df["Material"] == mat])
        share = max(int(round(target_size * group_size / n_total)), 5)
        allocation[mat] = share

    alloc_sum = sum(allocation.values())
    if alloc_sum < target_size:
        sorted_mats = sorted(allocation, key=lambda m: allocation[m],
                             reverse=True)
        deficit = target_size - alloc_sum
        for i in range(deficit):
            allocation[sorted_mats[i % len(sorted_mats)]] += 1

    print(f"[generate_synthetic_dataset] Allocation across "
          f"{len(materials)} materials (total target={target_size}):")
    for mat, count in sorted(allocation.items(), key=lambda x: -x[1]):
        real_count = len(df[df["Material"] == mat])
        print(f"    {mat:40s}  real={real_count:3d}  -> synth={count:3d}")

    # ── Generation loop per material ───────────────────────────────
    for mat in materials:
        mat_df = df[df["Material"] == mat].reset_index(drop=True)
        mat_stats = stats.get(mat, {})
        required = allocation.get(mat, 5)
        max_attempts = required * max_attempts_multiplier
        mat_synthetic: List[pd.Series] = []
        attempts = 0

        while len(mat_synthetic) < required and attempts < max_attempts:
            attempts += 1

            if len(mat_df) >= MIN_SAMPLES_FOR_INTERPOLATION:
                idx = np.random.choice(len(mat_df), size=2, replace=False)
                row_i = mat_df.iloc[idx[0]]
                row_j = mat_df.iloc[idx[1]]
                new_sample = generate_interpolated_sample(row_i, row_j)
            else:
                row_i = mat_df.iloc[0]
                new_sample = row_i.copy()

            new_sample = apply_gaussian_noise(new_sample, mat_stats)
            new_sample = repair_sample(new_sample)

            score = compute_total_confidence(
                new_sample, mat_df, numeric_cols
            )
            if score < min_confidence:
                continue

            new_sample["confidence_score"] = score
            mat_synthetic.append(new_sample)

        if len(mat_synthetic) > 1:
            mat_synthetic = remove_near_duplicates(
                mat_synthetic, numeric_cols, threshold=duplicate_threshold
            )
        all_synthetic.extend(mat_synthetic)

    # ── Final assembly ─────────────────────────────────────────────
    if not all_synthetic:
        warnings.warn(
            "No synthetic samples generated — returning empty DataFrame."
        )
        return pd.DataFrame(columns=list(df.columns) + ["confidence_score"])

    synth_df = pd.DataFrame(all_synthetic).reset_index(drop=True)

    if len(synth_df) < target_size:
        shortfall = target_size - len(synth_df)
        extra = synth_df.sample(
            n=shortfall, replace=True, random_state=RANDOM_SEED
        ).copy()
        for _, row in extra.iterrows():
            mat = row.get("Material", "")
            mat_stats_row = stats.get(mat, {})
            if mat_stats_row:
                noisy_row = apply_gaussian_noise(
                    row, mat_stats_row,
                    noise_fraction_low=0.005,
                    noise_fraction_high=0.015,
                )
                noisy_row = repair_sample(noisy_row)
                noisy_row["confidence_score"] = row["confidence_score"] * 0.95
                all_synthetic.append(noisy_row)

        synth_df = pd.DataFrame(all_synthetic).reset_index(drop=True)
        synth_df = synth_df.head(target_size)

    print(f"\n[generate_synthetic_dataset] Generated {len(synth_df)} "
          f"synthetic samples.")
    print(f"    Mean confidence : "
          f"{synth_df['confidence_score'].mean():.4f}")
    print(f"    Min  confidence : "
          f"{synth_df['confidence_score'].min():.4f}")
    print(f"    Max  confidence : "
          f"{synth_df['confidence_score'].max():.4f}")

    return synth_df


# ═══════════════════════════════════════════════════════════════════════
#  7.  SAVE SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════

def save_synthetic_data(
    synth_df: pd.DataFrame,
    filepath: str = "outputs/synthetic_SPIHF.csv",
) -> None:
    """Save the synthetic dataset to CSV and print a summary.

    Parameters
    ----------
    synth_df : pd.DataFrame
        Synthetic dataset from ``generate_synthetic_dataset``.
    filepath : str
        Output CSV path.

    Returns
    -------
    None
    """
    desired_order = COLUMN_NAMES + ["confidence_score"]
    cols_present = [c for c in desired_order if c in synth_df.columns]
    extra_cols = [c for c in synth_df.columns if c not in desired_order]
    synth_df = synth_df[cols_present + extra_cols]

    synth_df.to_csv(filepath, index=False)
    print(f"\n[save_synthetic_data] Saved {len(synth_df)} synthetic "
          f"samples to '{filepath}'.")

    print("\n" + "=" * 72)
    print("SYNTHETIC DATASET SUMMARY")
    print("=" * 72)
    print(f"  Total samples       : {len(synth_df)}")
    print(f"  Unique materials    : {synth_df['Material'].nunique()}")
    print(f"  Confidence (mean)   : "
          f"{synth_df['confidence_score'].mean():.4f}")
    print(f"  Confidence (median) : "
          f"{synth_df['confidence_score'].median():.4f}")
    print(f"  Confidence (min)    : "
          f"{synth_df['confidence_score'].min():.4f}")
    print()
    print("  Samples per material:")
    for mat, count in synth_df["Material"].value_counts().items():
        print(f"    {mat:40s}  {count:4d}")
    print("=" * 72)
