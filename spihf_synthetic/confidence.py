"""
confidence.py
=============
Multi-component confidence scoring for synthetic SPIHF samples.

Each synthetic sample receives a score in [0, 1] that quantifies how
plausible it is relative to its material group.  The score combines
three orthogonal signals:

1. **Mahalanobis-inspired range score** — fraction of features within the
   observed [min, max] of the material group (with 15 % margin).
2. **Distance score** — inverse normalised Euclidean distance to the
   nearest real sample (closer → higher).
3. **Correlation / distribution score** — alignment with the pairwise
   Pearson correlation structure of the group.

Functions
---------
compute_mahalanobis_score     Range-based plausibility (component 1).
compute_physics_score         Nearest-neighbour distance (component 2).
compute_distribution_score    Correlation-direction agreement (component 3).
compute_total_confidence      Weighted combination of all three.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# ═══════════════════════════════════════════════════════════════════════
#  COMPONENT 1: RANGE SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_mahalanobis_score(
    synthetic_sample: pd.Series,
    material_data: pd.DataFrame,
    numeric_cols: List[str],
    margin_fraction: float = 0.15,
) -> float:
    """Fraction of features within the observed [min, max] of the group.

    A 15 % margin around the observed range prevents penalising samples
    that sit just outside the convex hull due to noise injection.

    Parameters
    ----------
    synthetic_sample : pd.Series
        A single synthetic sample.
    material_data : pd.DataFrame
        All real samples of the same material.
    numeric_cols : List[str]
        Numeric columns to evaluate.
    margin_fraction : float
        Fractional margin around observed min/max.

    Returns
    -------
    float
        Range score in [0, 1].  Higher is better.
    """
    available = [
        c for c in numeric_cols
        if c in synthetic_sample.index and c in material_data.columns
    ]
    real_vals = material_data[available].dropna(axis=1, how="all") if available else pd.DataFrame()
    avail = [c for c in available if c in real_vals.columns]
    if not avail or len(real_vals) == 0:
        return 0.0

    in_range_count = 0
    total_checked = 0
    for col in avail:
        sv = synthetic_sample.get(col, np.nan)
        if pd.isna(sv):
            continue
        col_min = real_vals[col].min()
        col_max = real_vals[col].max()
        if pd.notna(col_min) and pd.notna(col_max):
            total_checked += 1
            margin = margin_fraction * (col_max - col_min + 1e-9)
            if col_min - margin <= sv <= col_max + margin:
                in_range_count += 1

    return in_range_count / max(total_checked, 1)


# ═══════════════════════════════════════════════════════════════════════
#  COMPONENT 2: DISTANCE SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_physics_score(
    synthetic_sample: pd.Series,
    material_data: pd.DataFrame,
    numeric_cols: List[str],
) -> float:
    """Inverse normalised Euclidean distance to the nearest real sample.

    The distance is computed in min-max normalised space, then mapped
    to a score via exponential decay: ``score = exp(−d_min)``.

    Parameters
    ----------
    synthetic_sample : pd.Series
        A single synthetic sample.
    material_data : pd.DataFrame
        All real samples of the same material.
    numeric_cols : List[str]
        Numeric columns to evaluate.

    Returns
    -------
    float
        Distance score in [0, 1].  Higher = closer to real data.
    """
    available = [
        c for c in numeric_cols
        if c in synthetic_sample.index and c in material_data.columns
    ]
    real_clean = material_data[available].dropna() if available else pd.DataFrame()
    if len(real_clean) == 0:
        return 0.0

    syn_vec: list[float] = []
    cols_used: list[str] = []
    for col in available:
        sv = synthetic_sample.get(col, np.nan)
        if pd.notna(sv) and col in real_clean.columns:
            syn_vec.append(sv)
            cols_used.append(col)

    if not cols_used:
        return 0.0

    real_sub = real_clean[cols_used].dropna()
    if len(real_sub) == 0:
        return 0.0

    # Normalise to [0, 1]
    r_min = real_sub.min()
    r_max = real_sub.max()
    denom = (r_max - r_min).replace(0, 1)
    real_norm = (real_sub - r_min) / denom
    syn_norm = np.array(
        [(sv - r_min[c]) / denom[c] for sv, c in zip(syn_vec, cols_used)]
    ).reshape(1, -1)

    dists = cdist(syn_norm, real_norm.values, metric="euclidean")
    min_dist = dists.min()
    return float(np.exp(-min_dist))


# ═══════════════════════════════════════════════════════════════════════
#  COMPONENT 3: DISTRIBUTION / CORRELATION SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_distribution_score(
    synthetic_sample: pd.Series,
    material_data: pd.DataFrame,
    numeric_cols: List[str],
) -> float:
    """Correlation-direction agreement with the material group.

    For each pair of non-trivially correlated features (|ρ| > 0.3),
    check whether the synthetic sample's deviations from the group
    mean are in the same direction as implied by the correlation.

    Parameters
    ----------
    synthetic_sample : pd.Series
        A single synthetic sample.
    material_data : pd.DataFrame
        All real samples of the same material.
    numeric_cols : List[str]
        Numeric columns to evaluate.

    Returns
    -------
    float
        Correlation score in [0, 1].  0.5 if insufficient data.
    """
    available = [
        c for c in numeric_cols
        if c in synthetic_sample.index and c in material_data.columns
    ]
    real_clean = material_data[available].dropna() if available else pd.DataFrame()

    syn_vec: list[float] = []
    cols_used: list[str] = []
    for col in available:
        sv = synthetic_sample.get(col, np.nan)
        if pd.notna(sv) and col in real_clean.columns:
            syn_vec.append(sv)
            cols_used.append(col)

    real_sub = real_clean[cols_used].dropna() if cols_used else pd.DataFrame()

    if len(cols_used) < 3 or len(real_sub) < 3:
        return 0.5  # not enough data to judge

    try:
        corr_mat = real_sub.corr()
        agreements = 0
        pairs_checked = 0
        for i in range(len(cols_used)):
            for j in range(i + 1, len(cols_used)):
                ci, cj = cols_used[i], cols_used[j]
                rho = corr_mat.loc[ci, cj]
                if pd.isna(rho):
                    continue
                mean_i = real_sub[ci].mean()
                mean_j = real_sub[cj].mean()
                dev_i = syn_vec[i] - mean_i
                dev_j = syn_vec[j] - mean_j
                if abs(rho) > 0.3:
                    pairs_checked += 1
                    if (np.sign(dev_i * dev_j) == np.sign(rho)
                            or abs(dev_i * dev_j) < 1e-9):
                        agreements += 1
        return agreements / max(pairs_checked, 1)
    except Exception:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════
#  TOTAL CONFIDENCE
# ═══════════════════════════════════════════════════════════════════════

def compute_total_confidence(
    synthetic_sample: pd.Series,
    material_data: pd.DataFrame,
    numeric_cols: List[str],
    weight_range: float = 0.40,
    weight_distance: float = 0.40,
    weight_correlation: float = 0.20,
) -> float:
    """Compute the weighted total confidence score for a synthetic sample.

    The score combines three components:

    1. Range / Mahalanobis-inspired score (default weight 0.40).
    2. Nearest-neighbour distance score (default weight 0.40).
    3. Correlation-direction agreement score (default weight 0.20).

    Parameters
    ----------
    synthetic_sample : pd.Series
        A single synthetic sample.
    material_data : pd.DataFrame
        All real samples of the same material.
    numeric_cols : List[str]
        Numeric columns to evaluate.
    weight_range : float
        Weight for the range score.
    weight_distance : float
        Weight for the distance score.
    weight_correlation : float
        Weight for the correlation score.

    Returns
    -------
    float
        Total confidence score in [0, 1].  Higher is better.

    Notes
    -----
    This score is *not* a statistical p-value.  It is a heuristic that
    penalises samples far from the data manifold while rewarding those
    that preserve inter-feature relationships.
    """
    range_score = compute_mahalanobis_score(
        synthetic_sample, material_data, numeric_cols
    )
    dist_score = compute_physics_score(
        synthetic_sample, material_data, numeric_cols
    )
    corr_score = compute_distribution_score(
        synthetic_sample, material_data, numeric_cols
    )

    total = (
        weight_range * range_score
        + weight_distance * dist_score
        + weight_correlation * corr_score
    )
    return float(np.clip(total, 0.0, 1.0))
