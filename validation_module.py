"""
validation_module.py
====================
Statistical validation of synthetic SPIHF (Single Point Incremental Hole
Flanging) data against the original experimental dataset.

This module answers the central question: *does the synthetic data
preserve the statistical fingerprint of the real manufacturing process?*

It validates:
  - Univariate distributions (mean, std, median, IQR, category frequencies)
  - Multivariate structure (correlation matrices, mutual information)
  - Feature importance rankings (Random Forest + mutual information)
  - Distributional distances (KS, Wasserstein, Jensen-Shannon, Mahalanobis, EMD)
  - Domain-specific correlation pairs from SPIHF process physics

Outputs:
  - validation_report.txt   (human-readable summary)
  - validation_metrics.json  (machine-readable metrics)

Author : Validation Module (auto-generated)
Seed   : np.random.seed(42)
"""

from __future__ import annotations

import json
import textwrap
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────── Global seed ────────────────────────────
np.random.seed(42)

# ──────────────────────────── Constants ──────────────────────────────

# Canonical column names (must match the synthetic CSV produced by
# augmentation_pipeline.py).
NUMERIC_FEATURES: List[str] = [
    "Thickness (mm)",
    "Precut dimensions (diameter/side length) mm",
    "Total Strain/Elongation (%)",
    "UTS (MPa)",
    "YS (MPa)",
    "Strength Coefficient (k in MPa)",
    "Strain hardening coefficient (n)",
    "Anisotropic (R Value)",
    "Is lubricant used?",
    "Feed rate (mm/min)",
    "Tool speed (rpm)",
    "Step depth (mm)",
    "No of stages",
    "HER",
    "Flange Height (mm)",
    "Roughness (um)",
    "Minimum thickness (after final stage, mm)",
    "Final angle after the final stage (degrees)",
]

CATEGORICAL_FEATURES: List[str] = [
    "Material",
    "Precut Shape (circle/square/etc)",
]

# Domain-specific correlation pairs from SPIHF process physics.
# Each tuple is (feature_a, feature_b, expected_direction_sign).
#   +1 = expected positive correlation
#   -1 = expected negative correlation
#    0 = sign is data-dependent (just check preservation)
PHYSICS_CORRELATION_PAIRS: List[Tuple[str, str, int]] = [
    ("HER", "Flange Height (mm)", +1),
    ("Step depth (mm)", "Roughness (um)", +1),
    ("Step depth (mm)", "Minimum thickness (after final stage, mm)", -1),
    ("No of stages", "HER", +1),
    ("Is lubricant used?", "Roughness (um)", -1),
    ("Total Strain/Elongation (%)", "HER", +1),
    ("Anisotropic (R Value)", "HER", 0),
]

# ── Column name mapping ──────────────────────────────────────────────
# The raw SPIHF_Data.csv uses slightly different column names from the
# canonical names in the synthetic CSV.  This mapping normalises them.
_RAW_TO_CANONICAL: Dict[str, str] = {
    "Precut dimensions (diameter/ side length) mm": "Precut dimensions (diameter/side length) mm",
    "Precut Shape (circle/ square/etc)": "Precut Shape (circle/square/etc)",
    "Total Strain/ Elongation (%)": "Total Strain/Elongation (%)",
    "Step depth(mm)": "Step depth (mm)",
    "Final angle after the final stage, degrees": "Final angle after the final stage (degrees)",
}


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════

def _harmonise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw-CSV columns to canonical names and coerce numerics.

    This allows the validation module to work with either the raw
    ``SPIHF_Data.csv`` (which has spaces/slashes in column names) or
    the cleaned synthetic CSV, without requiring the user to run the
    augmentation pipeline's preprocessor first.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (raw or synthetic).

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical column names.
    """
    df = df.copy()
    df.rename(columns=_RAW_TO_CANONICAL, inplace=True)

    # Handle the roughness column name encoding issue (mu symbol)
    for col in df.columns:
        if "Roughness" in col and col not in NUMERIC_FEATURES:
            df.rename(columns={col: "Roughness (um)"}, inplace=True)
            break

    # Strip embedded unit strings from numeric cells
    unit_suffixes = [
        " mm/min", " rpm clockwise", " rpm", " mm/cycle",
        " mm", " um", "°", "° ",
    ]
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        for suffix in unit_suffixes:
            s = s.str.replace(suffix, "", regex=False)
        s = s.str.replace("°", "", regex=False)
        s = s.str.replace("<", "", regex=False)
        s = s.str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")

    # Forward-fill material (raw CSV has blank rows for merged cells)
    if "Material" in df.columns:
        df["Material"] = df["Material"].replace(r"^\s*$", np.nan, regex=True)
        df["Material"] = df["Material"].ffill()
        df["Material"] = df["Material"].astype(str).str.strip()

    return df


def _safe_corr(
    df: pd.DataFrame, col_a: str, col_b: str
) -> Optional[float]:
    """Compute Pearson correlation, returning None if insufficient data."""
    if col_a not in df.columns or col_b not in df.columns:
        return None
    valid = df[[col_a, col_b]].dropna()
    if len(valid) < 3:
        return None
    r = valid[col_a].corr(valid[col_b])
    return float(r) if pd.notna(r) else None


def _jensen_shannon_divergence(
    p: np.ndarray, q: np.ndarray, n_bins: int = 30
) -> float:
    """Compute Jensen-Shannon divergence between two 1-D samples.

    Both arrays are histogrammed into ``n_bins`` bins spanning the
    combined range, converted to probability distributions, and then
    the symmetric JSD is computed as::

        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    where  M = 0.5 * (P + Q).

    Parameters
    ----------
    p, q : np.ndarray
        1-D arrays of sample values.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        JSD in [0, ln(2)] (nats).  Smaller = more similar.
    """
    combined = np.concatenate([p, q])
    lo, hi = np.nanmin(combined), np.nanmax(combined)
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    hist_p, _ = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bins, density=True)
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    hist_p = hist_p.astype(float) + eps
    hist_q = hist_q.astype(float) + eps
    # Normalise to probability
    hist_p /= hist_p.sum()
    hist_q /= hist_q.sum()
    m = 0.5 * (hist_p + hist_q)
    jsd = 0.5 * entropy(hist_p, m) + 0.5 * entropy(hist_q, m)
    return float(jsd)


def _mahalanobis_distance(
    real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]
) -> Optional[float]:
    """Compute Mahalanobis distance between centroids of two datasets.

    The covariance matrix is estimated from the *real* data.  The
    distance measures how far the synthetic centroid is from the real
    centroid in units of the real data's natural scatter.

    Parameters
    ----------
    real, synth : pd.DataFrame
        DataFrames with matching numeric columns.
    cols : List[str]
        Columns to include.

    Returns
    -------
    float or None
        Mahalanobis distance, or None if the covariance is singular.

    Engineering note
    ----------------
    A Mahalanobis distance close to 0 means the synthetic centroid sits
    right at the real centroid.  Values < 3 are generally acceptable
    for manufacturing datasets with moderate dimensionality.
    """
    available = [c for c in cols if c in real.columns and c in synth.columns]
    r = real[available].dropna()
    s = synth[available].dropna()
    if len(r) < len(available) + 1 or len(s) < 1:
        return None

    mean_r = r.mean().values
    mean_s = s.mean().values
    cov_r = r.cov().values

    try:
        cov_inv = np.linalg.inv(cov_r)
        dist = float(mahalanobis(mean_r, mean_s, cov_inv))
        return dist
    except np.linalg.LinAlgError:
        # Singular covariance — use pseudoinverse
        try:
            cov_pinv = np.linalg.pinv(cov_r)
            diff = mean_r - mean_s
            dist = float(np.sqrt(np.dot(np.dot(diff, cov_pinv), diff)))
            return dist
        except Exception:
            return None


# ════════════════════════════════════════════════════════════════════
#  1.  VALIDATE DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════
def validate_distributions(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Compare univariate descriptive statistics between real and synthetic data.

    For every numeric feature, this computes and compares:
      - Mean
      - Standard deviation
      - Median
      - IQR (interquartile range, Q3 - Q1)

    For categorical features it compares:
      - Category frequency distributions (as proportions)

    Parameters
    ----------
    real : pd.DataFrame
        Original (real) dataset with canonical column names.
    synth : pd.DataFrame
        Synthetic dataset with canonical column names.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Nested dict keyed by feature name.  Each entry contains
        ``real_*`` and ``synth_*`` statistics, plus ``pct_diff_*``
        relative differences (as percentages).

    Engineering note
    ----------------
    In manufacturing data, preserving the *spread* (std, IQR) is as
    important as preserving the *centre* (mean, median) because process
    variability directly maps to tolerance bands and reject rates.
    Large IQR inflation in the synthetic data would signal that the
    noise perturbation is too aggressive.
    """
    results: Dict[str, Dict[str, Any]] = {}

    # ── Numeric features ───────────────────────────────────────────
    for col in NUMERIC_FEATURES:
        if col not in real.columns or col not in synth.columns:
            continue
        rv = real[col].dropna()
        sv = synth[col].dropna()
        if len(rv) == 0 or len(sv) == 0:
            continue

        r_mean, s_mean = rv.mean(), sv.mean()
        r_std, s_std = rv.std(), sv.std()
        r_med, s_med = rv.median(), sv.median()
        r_iqr = rv.quantile(0.75) - rv.quantile(0.25)
        s_iqr = sv.quantile(0.75) - sv.quantile(0.25)

        def _pct(a: float, b: float) -> float:
            denom = abs(a) if abs(a) > 1e-12 else 1e-12
            return float(abs(a - b) / denom * 100.0)

        results[col] = {
            "real_mean": float(r_mean),
            "synth_mean": float(s_mean),
            "pct_diff_mean": _pct(r_mean, s_mean),
            "real_std": float(r_std),
            "synth_std": float(s_std),
            "pct_diff_std": _pct(r_std, s_std),
            "real_median": float(r_med),
            "synth_median": float(s_med),
            "pct_diff_median": _pct(r_med, s_med),
            "real_iqr": float(r_iqr),
            "synth_iqr": float(s_iqr),
            "pct_diff_iqr": _pct(r_iqr, s_iqr),
            "real_n": int(len(rv)),
            "synth_n": int(len(sv)),
        }

    # ── Categorical features ──────────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col not in real.columns or col not in synth.columns:
            continue
        r_freq = real[col].value_counts(normalize=True).to_dict()
        s_freq = synth[col].value_counts(normalize=True).to_dict()
        all_cats = sorted(set(list(r_freq.keys()) + list(s_freq.keys())))
        freq_comparison = {}
        for cat in all_cats:
            rp = r_freq.get(cat, 0.0)
            sp = s_freq.get(cat, 0.0)
            freq_comparison[cat] = {
                "real_proportion": round(float(rp), 4),
                "synth_proportion": round(float(sp), 4),
                "abs_diff": round(abs(float(rp) - float(sp)), 4),
            }
        results[f"{col} (frequencies)"] = freq_comparison

    return results


# ════════════════════════════════════════════════════════════════════
#  2.  KOLMOGOROV-SMIRNOV TESTS
# ════════════════════════════════════════════════════════════════════
def compute_ks_tests(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Run two-sample KS tests for every numeric feature.

    The KS test measures the maximum absolute difference between the
    empirical CDFs of the real and synthetic samples.  A large
    p-value (> alpha) means we *cannot reject* the null hypothesis
    that both samples are drawn from the same distribution.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    alpha : float
        Significance level.  Default 0.05.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Per-feature dict with ``ks_statistic``, ``p_value``, and
        ``same_distribution`` (bool, True if p >= alpha).

    Engineering note
    ----------------
    In manufacturing validation, KS tests on *process parameters*
    (feed rate, tool speed, step depth) should almost always pass
    because those are input settings, not measured outputs.  Failure
    on *response variables* (HER, flange height, roughness) is more
    informative and may indicate that the physics-correction layer
    shifted the output distribution.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for col in NUMERIC_FEATURES:
        if col not in real.columns or col not in synth.columns:
            continue
        rv = real[col].dropna().values
        sv = synth[col].dropna().values
        if len(rv) < 2 or len(sv) < 2:
            continue
        stat, pval = ks_2samp(rv, sv)
        results[col] = {
            "ks_statistic": round(float(stat), 6),
            "p_value": round(float(pval), 6),
            "same_distribution": bool(pval >= alpha),
        }
    return results


# ════════════════════════════════════════════════════════════════════
#  3.  WASSERSTEIN / EMD METRICS
# ════════════════════════════════════════════════════════════════════
def compute_wasserstein_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compute Wasserstein-1 distance (Earth Mover's Distance) per feature.

    The Wasserstein distance quantifies the minimum "work" needed to
    transform the real distribution into the synthetic one.  Unlike
    the KS statistic, it accounts for the *magnitude* of distributional
    differences, not just the maximum gap.

    Additionally, the Jensen-Shannon Divergence is computed for each
    feature as a bounded, symmetric measure of distributional similarity.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per-feature dict with ``wasserstein_distance`` (also called EMD),
        ``wasserstein_normalised`` (divided by feature range for cross-
        feature comparability), and ``jensen_shannon_divergence``.

    Engineering note
    ----------------
    Wasserstein distance has physical units matching the feature, so
    a Wasserstein distance of 5.2 on UTS (MPa) means the distributions
    differ by the equivalent of moving 5.2 MPa of "probability mass".
    The normalised version divides by the real data range, yielding a
    unit-free [0, 1] score where 0 = identical distributions.
    """
    results: Dict[str, Dict[str, float]] = {}
    for col in NUMERIC_FEATURES:
        if col not in real.columns or col not in synth.columns:
            continue
        rv = real[col].dropna().values.astype(float)
        sv = synth[col].dropna().values.astype(float)
        if len(rv) < 2 or len(sv) < 2:
            continue

        wd = wasserstein_distance(rv, sv)
        r_range = float(np.ptp(rv))
        wd_norm = wd / r_range if r_range > 1e-12 else 0.0
        jsd = _jensen_shannon_divergence(rv, sv)

        results[col] = {
            "wasserstein_distance": round(float(wd), 6),
            "wasserstein_normalised": round(float(wd_norm), 6),
            "jensen_shannon_divergence": round(float(jsd), 6),
        }
    return results


# ════════════════════════════════════════════════════════════════════
#  4.  CORRELATION COMPARISON
# ════════════════════════════════════════════════════════════════════
def compare_correlations(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare correlation matrices and domain-specific pairs.

    This function computes:

    1. **Full correlation matrix difference**: the Frobenius norm of
       ``corr_real - corr_synth``, measuring overall structural fidelity.

    2. **Physics correlation pairs**: for each of the 7 domain-specific
       pairs (HER <-> Flange Height, etc.), the real and synthetic
       Pearson correlations are compared.  The function also checks
       whether the *sign* of the correlation is preserved.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    Dict[str, Any]
        Contains:
        - ``frobenius_norm``: overall correlation matrix difference.
        - ``mean_abs_corr_diff``: average absolute element-wise difference.
        - ``physics_pairs``: per-pair breakdown with real/synth rho,
          sign preservation flag, and absolute difference.
        - ``mutual_information``: per-pair MI for both real and synthetic.

    Engineering note
    ----------------
    In SPIHF, the correlation between HER and Flange Height is
    physically grounded: a larger hole expansion ratio means more
    material is displaced, producing a taller flange.  Similarly,
    deeper step depths increase surface roughness due to greater
    tool-sheet contact pressure and sliding distance.  These
    correlations *must* be preserved for the synthetic data to be
    useful in downstream ML models.
    """
    results: Dict[str, Any] = {}

    # ── Full correlation matrix comparison ─────────────────────────
    common_num = [c for c in NUMERIC_FEATURES
                  if c in real.columns and c in synth.columns]
    r_corr = real[common_num].corr()
    s_corr = synth[common_num].corr()

    # Align indices
    shared = sorted(set(r_corr.columns) & set(s_corr.columns))
    if shared:
        r_sub = r_corr.loc[shared, shared].fillna(0)
        s_sub = s_corr.loc[shared, shared].fillna(0)
        diff_mat = (r_sub - s_sub).values
        frob = float(np.linalg.norm(diff_mat, "fro"))
        mean_abs = float(np.mean(np.abs(diff_mat)))
    else:
        frob = float("nan")
        mean_abs = float("nan")

    results["frobenius_norm"] = round(frob, 6)
    results["mean_abs_corr_diff"] = round(mean_abs, 6)

    # ── Physics-specific correlation pairs ─────────────────────────
    pairs_results: Dict[str, Dict[str, Any]] = {}
    for col_a, col_b, expected_sign in PHYSICS_CORRELATION_PAIRS:
        r_rho = _safe_corr(real, col_a, col_b)
        s_rho = _safe_corr(synth, col_a, col_b)

        pair_key = f"{col_a} <-> {col_b}"
        entry: Dict[str, Any] = {
            "real_correlation": round(r_rho, 4) if r_rho is not None else None,
            "synth_correlation": round(s_rho, 4) if s_rho is not None else None,
            "expected_sign": expected_sign,
        }

        if r_rho is not None and s_rho is not None:
            entry["abs_difference"] = round(abs(r_rho - s_rho), 4)
            entry["sign_preserved"] = bool(np.sign(r_rho) == np.sign(s_rho))
            # Check if the expected sign (from physics) is satisfied
            if expected_sign != 0:
                entry["physics_sign_real"] = bool(np.sign(r_rho) == expected_sign)
                entry["physics_sign_synth"] = bool(np.sign(s_rho) == expected_sign)
        else:
            entry["abs_difference"] = None
            entry["sign_preserved"] = None

        pairs_results[pair_key] = entry

    results["physics_pairs"] = pairs_results

    # ── Mutual information for physics pairs ───────────────────────
    mi_results: Dict[str, Dict[str, Optional[float]]] = {}
    for col_a, col_b, _ in PHYSICS_CORRELATION_PAIRS:
        pair_key = f"{col_a} <-> {col_b}"
        mi_entry: Dict[str, Optional[float]] = {}

        for label, data in [("real", real), ("synth", synth)]:
            if col_a in data.columns and col_b in data.columns:
                valid = data[[col_a, col_b]].dropna()
                if len(valid) >= 5:
                    try:
                        mi = mutual_info_regression(
                            valid[[col_a]].values,
                            valid[col_b].values,
                            random_state=42,
                            n_neighbors=min(3, len(valid) - 1),
                        )[0]
                        mi_entry[f"{label}_mi"] = round(float(mi), 4)
                    except Exception:
                        mi_entry[f"{label}_mi"] = None
                else:
                    mi_entry[f"{label}_mi"] = None
            else:
                mi_entry[f"{label}_mi"] = None

        mi_results[pair_key] = mi_entry

    results["mutual_information"] = mi_results

    return results


# ════════════════════════════════════════════════════════════════════
#  5.  FEATURE IMPORTANCE SIMILARITY
# ════════════════════════════════════════════════════════════════════
def compute_feature_importance_similarity(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target: str = "HER",
) -> Dict[str, Any]:
    """Compare feature importance rankings between real and synthetic data.

    Two methods are used:

    1. **Random Forest importance**: A ``RandomForestRegressor`` is
       trained on both datasets (predicting ``target``), and the
       Gini importances are compared via Spearman rank correlation.

    2. **Mutual information importance**: ``mutual_info_regression``
       scores each feature's non-linear dependence with the target.
       Rankings are again compared via Spearman correlation.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    target : str
        Target column name for importance ranking.  Default ``"HER"``
        as it is the primary response variable in SPIHF studies.

    Returns
    -------
    Dict[str, Any]
        Contains per-feature importances, Spearman rank correlation
        between real and synthetic rankings, and a verdict.

    Engineering note
    ----------------
    If the synthetic data preserves the same feature importance
    hierarchy as the real data, downstream ML models trained on
    synthetic data will learn similar decision boundaries.  A Spearman
    rho > 0.7 is generally considered acceptable for manufacturing
    surrogate models.
    """
    results: Dict[str, Any] = {}

    if target not in real.columns or target not in synth.columns:
        results["error"] = f"Target '{target}' not found in both datasets."
        return results

    predictors = [c for c in NUMERIC_FEATURES
                  if c != target
                  and c in real.columns
                  and c in synth.columns]
    if len(predictors) < 2:
        results["error"] = "Insufficient predictor columns."
        return results

    # ── Prepare clean subsets ──────────────────────────────────────
    r_clean = real[predictors + [target]].dropna()
    s_clean = synth[predictors + [target]].dropna()

    if len(r_clean) < 10 or len(s_clean) < 10:
        results["error"] = "Insufficient non-NaN rows for importance analysis."
        return results

    X_r, y_r = r_clean[predictors].values, r_clean[target].values
    X_s, y_s = s_clean[predictors].values, s_clean[target].values

    # ── Method 1: Random Forest ────────────────────────────────────
    rf_r = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_s = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_r.fit(X_r, y_r)
        rf_s.fit(X_s, y_s)

    imp_r_rf = rf_r.feature_importances_
    imp_s_rf = rf_s.feature_importances_

    # ── Method 2: Mutual Information ───────────────────────────────
    mi_r = mutual_info_regression(X_r, y_r, random_state=42,
                                  n_neighbors=min(5, len(X_r) - 1))
    mi_s = mutual_info_regression(X_s, y_s, random_state=42,
                                  n_neighbors=min(5, len(X_s) - 1))

    # ── Spearman rank correlation ──────────────────────────────────
    from scipy.stats import spearmanr
    rho_rf, pval_rf = spearmanr(imp_r_rf, imp_s_rf)
    rho_mi, pval_mi = spearmanr(mi_r, mi_s)

    # ── Assemble per-feature table ─────────────────────────────────
    feature_table: Dict[str, Dict[str, float]] = {}
    for i, feat in enumerate(predictors):
        feature_table[feat] = {
            "real_rf_importance": round(float(imp_r_rf[i]), 4),
            "synth_rf_importance": round(float(imp_s_rf[i]), 4),
            "real_mi_score": round(float(mi_r[i]), 4),
            "synth_mi_score": round(float(mi_s[i]), 4),
        }

    results["target_variable"] = target
    results["n_predictors"] = len(predictors)
    results["feature_importances"] = feature_table
    results["spearman_rf"] = {
        "rho": round(float(rho_rf), 4),
        "p_value": round(float(pval_rf), 6),
    }
    results["spearman_mi"] = {
        "rho": round(float(rho_mi), 4),
        "p_value": round(float(pval_mi), 6),
    }
    results["verdict_rf"] = (
        "PASS" if rho_rf >= 0.7 else
        "MARGINAL" if rho_rf >= 0.5 else
        "FAIL"
    )
    results["verdict_mi"] = (
        "PASS" if rho_mi >= 0.7 else
        "MARGINAL" if rho_mi >= 0.5 else
        "FAIL"
    )

    return results


# ════════════════════════════════════════════════════════════════════
#  6.  GENERATE VALIDATION SUMMARY
# ════════════════════════════════════════════════════════════════════
def generate_validation_summary(
    dist_results: Dict[str, Dict[str, Any]],
    ks_results: Dict[str, Dict[str, Any]],
    wd_results: Dict[str, Dict[str, float]],
    corr_results: Dict[str, Any],
    importance_results: Dict[str, Any],
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Dict[str, Any]:
    """Aggregate all validation results into a single summary dictionary.

    This function also computes the Mahalanobis distance between the
    real and synthetic dataset centroids and assigns an overall
    quality grade.

    Grading rubric
    --------------
    The overall grade is based on:
      - Fraction of KS tests passed (weight 25 %)
      - Mean normalised Wasserstein distance (weight 25 %)
      - Correlation Frobenius norm (weight 20 %)
      - Feature importance Spearman rho (weight 15 %)
      - Mean % difference in descriptive statistics (weight 15 %)

    Grades: A (>= 85), B (>= 70), C (>= 55), D (>= 40), F (< 40).

    Parameters
    ----------
    dist_results : Dict
        From ``validate_distributions``.
    ks_results : Dict
        From ``compute_ks_tests``.
    wd_results : Dict
        From ``compute_wasserstein_metrics``.
    corr_results : Dict
        From ``compare_correlations``.
    importance_results : Dict
        From ``compute_feature_importance_similarity``.
    real, synth : pd.DataFrame
        Original and synthetic DataFrames.

    Returns
    -------
    Dict[str, Any]
        Full summary with sub-scores and overall grade.
    """
    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "real_samples": len(real),
        "synth_samples": len(synth),
    }

    # ── KS pass rate ───────────────────────────────────────────────
    if ks_results:
        n_pass = sum(1 for v in ks_results.values() if v.get("same_distribution"))
        n_total = len(ks_results)
        ks_pass_rate = n_pass / n_total if n_total > 0 else 0.0
    else:
        ks_pass_rate = 0.0
    summary["ks_pass_rate"] = round(ks_pass_rate, 4)

    # ── Mean normalised Wasserstein ────────────────────────────────
    if wd_results:
        wd_norms = [v.get("wasserstein_normalised", 0.0)
                    for v in wd_results.values()]
        mean_wd = np.mean(wd_norms)
        wd_score = max(0.0, 1.0 - mean_wd)  # invert: lower distance = better
    else:
        mean_wd = 1.0
        wd_score = 0.0
    summary["mean_wasserstein_normalised"] = round(float(mean_wd), 4)

    # ── Mean JSD ───────────────────────────────────────────────────
    if wd_results:
        jsds = [v.get("jensen_shannon_divergence", 0.0)
                for v in wd_results.values()]
        summary["mean_jsd"] = round(float(np.mean(jsds)), 6)
    else:
        summary["mean_jsd"] = None

    # ── Mahalanobis distance ───────────────────────────────────────
    common_num = [c for c in NUMERIC_FEATURES
                  if c in real.columns and c in synth.columns]
    maha = _mahalanobis_distance(real, synth, common_num)
    summary["mahalanobis_distance"] = round(maha, 4) if maha is not None else None

    # ── Correlation Frobenius norm ─────────────────────────────────
    frob = corr_results.get("frobenius_norm", float("nan"))
    # Normalise: typical Frobenius norm for N features is sqrt(N)
    n_feat = len(common_num)
    frob_norm = frob / max(np.sqrt(n_feat), 1.0) if not np.isnan(frob) else 1.0
    corr_score = max(0.0, 1.0 - frob_norm)
    summary["correlation_frobenius_norm"] = round(frob, 4) if not np.isnan(frob) else None

    # ── Physics pair preservation ──────────────────────────────────
    pairs = corr_results.get("physics_pairs", {})
    if pairs:
        n_sign_preserved = sum(
            1 for v in pairs.values()
            if v.get("sign_preserved") is True
        )
        n_pairs_total = sum(
            1 for v in pairs.values()
            if v.get("sign_preserved") is not None
        )
        sign_pres_rate = n_sign_preserved / max(n_pairs_total, 1)
    else:
        sign_pres_rate = 0.0
    summary["physics_sign_preservation_rate"] = round(sign_pres_rate, 4)

    # ── Feature importance Spearman rho ────────────────────────────
    imp_rho = importance_results.get("spearman_rf", {}).get("rho", 0.0)
    if imp_rho is None or np.isnan(imp_rho):
        imp_rho = 0.0
    imp_score = max(0.0, imp_rho)

    # ── Mean % difference in descriptive stats ─────────────────────
    pct_diffs = []
    for feat, vals in dist_results.items():
        if isinstance(vals, dict) and "pct_diff_mean" in vals:
            pct_diffs.append(vals["pct_diff_mean"])
            pct_diffs.append(vals["pct_diff_std"])
    if pct_diffs:
        mean_pct = np.mean(pct_diffs)
        stats_score = max(0.0, 1.0 - mean_pct / 100.0)
    else:
        mean_pct = 100.0
        stats_score = 0.0
    summary["mean_pct_diff_stats"] = round(float(mean_pct), 2)

    # ── Overall composite score ────────────────────────────────────
    composite = (
        0.25 * ks_pass_rate +
        0.25 * wd_score +
        0.20 * corr_score +
        0.15 * imp_score +
        0.15 * stats_score
    )
    composite_pct = composite * 100.0

    if composite_pct >= 85:
        grade = "A"
    elif composite_pct >= 70:
        grade = "B"
    elif composite_pct >= 55:
        grade = "C"
    elif composite_pct >= 40:
        grade = "D"
    else:
        grade = "F"

    summary["composite_score"] = round(float(composite_pct), 2)
    summary["grade"] = grade
    summary["sub_scores"] = {
        "ks_pass_rate_25pct": round(ks_pass_rate * 100, 2),
        "wasserstein_score_25pct": round(wd_score * 100, 2),
        "correlation_score_20pct": round(corr_score * 100, 2),
        "importance_score_15pct": round(imp_score * 100, 2),
        "stats_fidelity_15pct": round(stats_score * 100, 2),
    }

    return summary


# ════════════════════════════════════════════════════════════════════
#  7.  SAVE VALIDATION REPORT
# ════════════════════════════════════════════════════════════════════
def save_validation_report(
    summary: Dict[str, Any],
    dist_results: Dict[str, Dict[str, Any]],
    ks_results: Dict[str, Dict[str, Any]],
    wd_results: Dict[str, Dict[str, float]],
    corr_results: Dict[str, Any],
    importance_results: Dict[str, Any],
    report_path: str = "validation_report.txt",
    json_path: str = "validation_metrics.json",
) -> None:
    """Write a human-readable report and a machine-readable JSON file.

    Parameters
    ----------
    summary : Dict
        From ``generate_validation_summary``.
    dist_results, ks_results, wd_results, corr_results, importance_results
        Results from the individual validation functions.
    report_path : str
        Path for the .txt report.
    json_path : str
        Path for the .json metrics.

    Returns
    -------
    None
    """
    # ════════════════════════════════════════════════════════════════
    #  HUMAN-READABLE REPORT
    # ════════════════════════════════════════════════════════════════
    lines: List[str] = []
    w = 72  # line width

    def hr(char: str = "=") -> str:
        return char * w

    def section(title: str) -> None:
        lines.append("")
        lines.append(hr("="))
        lines.append(f"  {title}")
        lines.append(hr("="))

    def subsection(title: str) -> None:
        lines.append("")
        lines.append(hr("-"))
        lines.append(f"  {title}")
        lines.append(hr("-"))

    lines.append(hr())
    lines.append("  SPIHF SYNTHETIC DATA VALIDATION REPORT")
    lines.append(hr())
    lines.append(f"  Generated : {summary.get('timestamp', 'N/A')}")
    lines.append(f"  Real samples     : {summary.get('real_samples', 'N/A')}")
    lines.append(f"  Synthetic samples: {summary.get('synth_samples', 'N/A')}")
    lines.append(f"  Overall Grade    : {summary.get('grade', 'N/A')}  "
                 f"(score = {summary.get('composite_score', 'N/A')}%)")
    lines.append(hr())

    # ── Sub-scores ─────────────────────────────────────────────────
    section("COMPOSITE SCORE BREAKDOWN")
    sub = summary.get("sub_scores", {})
    lines.append(f"  KS pass rate          (25%) : {sub.get('ks_pass_rate_25pct', 'N/A')}%")
    lines.append(f"  Wasserstein score     (25%) : {sub.get('wasserstein_score_25pct', 'N/A')}%")
    lines.append(f"  Correlation score     (20%) : {sub.get('correlation_score_20pct', 'N/A')}%")
    lines.append(f"  Importance score      (15%) : {sub.get('importance_score_15pct', 'N/A')}%")
    lines.append(f"  Stats fidelity        (15%) : {sub.get('stats_fidelity_15pct', 'N/A')}%")
    lines.append("")
    maha = summary.get("mahalanobis_distance")
    lines.append(f"  Mahalanobis distance        : {maha if maha is not None else 'N/A'}")
    lines.append(f"  Mean JSD                    : {summary.get('mean_jsd', 'N/A')}")
    lines.append(f"  Physics sign preservation   : "
                 f"{summary.get('physics_sign_preservation_rate', 'N/A')}")

    # ── Descriptive statistics ─────────────────────────────────────
    section("DESCRIPTIVE STATISTICS COMPARISON")
    for feat, vals in dist_results.items():
        if not isinstance(vals, dict):
            continue
        if "pct_diff_mean" in vals:
            subsection(feat)
            lines.append(f"  {'Statistic':<20s}  {'Real':>12s}  {'Synthetic':>12s}  {'%Diff':>8s}")
            lines.append(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*8}")
            for stat_name in ["mean", "std", "median", "iqr"]:
                rv = vals.get(f"real_{stat_name}", "N/A")
                sv = vals.get(f"synth_{stat_name}", "N/A")
                pd_val = vals.get(f"pct_diff_{stat_name}", "N/A")
                rv_s = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
                sv_s = f"{sv:.4f}" if isinstance(sv, float) else str(sv)
                pd_s = f"{pd_val:.2f}" if isinstance(pd_val, float) else str(pd_val)
                lines.append(f"  {stat_name:<20s}  {rv_s:>12s}  {sv_s:>12s}  {pd_s:>8s}")
            lines.append(f"  Real N = {vals.get('real_n', '?')},  "
                         f"Synth N = {vals.get('synth_n', '?')}")
        else:
            # Category frequencies
            subsection(feat)
            lines.append(f"  {'Category':<30s}  {'Real':>8s}  {'Synth':>8s}  {'Diff':>8s}")
            lines.append(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
            for cat, cv in vals.items():
                rp = cv.get("real_proportion", 0.0)
                sp = cv.get("synth_proportion", 0.0)
                ad = cv.get("abs_diff", 0.0)
                cat_short = cat[:30] if len(cat) > 30 else cat
                lines.append(f"  {cat_short:<30s}  {rp:>8.4f}  {sp:>8.4f}  {ad:>8.4f}")

    # ── KS tests ───────────────────────────────────────────────────
    section("KOLMOGOROV-SMIRNOV TESTS (alpha = 0.05)")
    lines.append(f"  {'Feature':<50s}  {'KS Stat':>8s}  {'p-value':>10s}  {'Result':>6s}")
    lines.append(f"  {'-'*50}  {'-'*8}  {'-'*10}  {'-'*6}")
    for feat, vals in ks_results.items():
        stat = vals.get("ks_statistic", 0.0)
        pval = vals.get("p_value", 0.0)
        same = "PASS" if vals.get("same_distribution") else "FAIL"
        f_short = feat[:50]
        lines.append(f"  {f_short:<50s}  {stat:>8.4f}  {pval:>10.6f}  {same:>6s}")

    n_pass = sum(1 for v in ks_results.values() if v.get("same_distribution"))
    lines.append(f"\n  Summary: {n_pass}/{len(ks_results)} features passed KS test.")

    # ── Wasserstein / JSD ──────────────────────────────────────────
    section("WASSERSTEIN DISTANCE & JENSEN-SHANNON DIVERGENCE")
    lines.append(f"  {'Feature':<45s}  {'WD':>10s}  {'WD_norm':>8s}  {'JSD':>10s}")
    lines.append(f"  {'-'*45}  {'-'*10}  {'-'*8}  {'-'*10}")
    for feat, vals in wd_results.items():
        wd_val = vals.get("wasserstein_distance", 0.0)
        wn_val = vals.get("wasserstein_normalised", 0.0)
        jsd_val = vals.get("jensen_shannon_divergence", 0.0)
        f_short = feat[:45]
        lines.append(f"  {f_short:<45s}  {wd_val:>10.4f}  {wn_val:>8.4f}  {jsd_val:>10.6f}")

    # ── Correlation pairs ──────────────────────────────────────────
    section("PHYSICS CORRELATION PAIR ANALYSIS")
    pairs = corr_results.get("physics_pairs", {})
    lines.append(f"  {'Pair':<50s}  {'Real r':>7s}  {'Synth r':>8s}  {'Diff':>6s}  {'Sign':>6s}")
    lines.append(f"  {'-'*50}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}")
    for pair_name, pv in pairs.items():
        rr = pv.get("real_correlation")
        sr = pv.get("synth_correlation")
        ad = pv.get("abs_difference")
        sp_flag = "OK" if pv.get("sign_preserved") else "FLIP"
        rr_s = f"{rr:.4f}" if rr is not None else "N/A"
        sr_s = f"{sr:.4f}" if sr is not None else "N/A"
        ad_s = f"{ad:.4f}" if ad is not None else "N/A"
        p_short = pair_name[:50]
        lines.append(f"  {p_short:<50s}  {rr_s:>7s}  {sr_s:>8s}  {ad_s:>6s}  {sp_flag:>6s}")

    # ── Mutual information for pairs ───────────────────────────────
    mi_data = corr_results.get("mutual_information", {})
    if mi_data:
        subsection("MUTUAL INFORMATION FOR PHYSICS PAIRS")
        lines.append(f"  {'Pair':<50s}  {'Real MI':>8s}  {'Synth MI':>9s}")
        lines.append(f"  {'-'*50}  {'-'*8}  {'-'*9}")
        for pair_name, mv in mi_data.items():
            r_mi = mv.get("real_mi")
            s_mi = mv.get("synth_mi")
            r_s = f"{r_mi:.4f}" if r_mi is not None else "N/A"
            s_s = f"{s_mi:.4f}" if s_mi is not None else "N/A"
            p_short = pair_name[:50]
            lines.append(f"  {p_short:<50s}  {r_s:>8s}  {s_s:>9s}")

    # ── Feature importance ─────────────────────────────────────────
    section("FEATURE IMPORTANCE RANKING SIMILARITY")
    if "error" in importance_results:
        lines.append(f"  Error: {importance_results['error']}")
    else:
        lines.append(f"  Target variable: {importance_results.get('target_variable', 'N/A')}")
        lines.append(f"  Number of predictors: {importance_results.get('n_predictors', 'N/A')}")
        lines.append("")

        sp_rf = importance_results.get("spearman_rf", {})
        sp_mi = importance_results.get("spearman_mi", {})
        lines.append(f"  Random Forest Spearman rho  : {sp_rf.get('rho', 'N/A')}  "
                     f"(p = {sp_rf.get('p_value', 'N/A')})  "
                     f"[{importance_results.get('verdict_rf', 'N/A')}]")
        lines.append(f"  Mutual Info Spearman rho    : {sp_mi.get('rho', 'N/A')}  "
                     f"(p = {sp_mi.get('p_value', 'N/A')})  "
                     f"[{importance_results.get('verdict_mi', 'N/A')}]")

        fi_table = importance_results.get("feature_importances", {})
        if fi_table:
            subsection("PER-FEATURE IMPORTANCE TABLE")
            lines.append(f"  {'Feature':<42s}  {'RF_real':>8s}  {'RF_syn':>7s}  "
                         f"{'MI_real':>8s}  {'MI_syn':>7s}")
            lines.append(f"  {'-'*42}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}")
            for feat, fv in fi_table.items():
                f_short = feat[:42]
                r_rf = fv.get("real_rf_importance", 0.0)
                s_rf = fv.get("synth_rf_importance", 0.0)
                r_mi = fv.get("real_mi_score", 0.0)
                s_mi = fv.get("synth_mi_score", 0.0)
                lines.append(f"  {f_short:<42s}  {r_rf:>8.4f}  {s_rf:>7.4f}  "
                             f"{r_mi:>8.4f}  {s_mi:>7.4f}")

    # ── Footer ─────────────────────────────────────────────────────
    lines.append("")
    lines.append(hr())
    lines.append("  END OF VALIDATION REPORT")
    lines.append(hr())

    report_text = "\n".join(lines)

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[save_validation_report] Written '{report_path}' ({len(lines)} lines).")

    # ════════════════════════════════════════════════════════════════
    #  JSON METRICS
    # ════════════════════════════════════════════════════════════════
    json_output: Dict[str, Any] = {
        "summary": summary,
        "distribution_statistics": dist_results,
        "ks_tests": ks_results,
        "wasserstein_and_jsd": wd_results,
        "correlation_analysis": corr_results,
        "feature_importance": importance_results,
    }

    # Sanitise for JSON serialisation
    def _sanitise(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitise(v) for v in obj]
        return obj

    json_clean = _sanitise(json_output)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_clean, f, indent=2, ensure_ascii=False)
    print(f"[save_validation_report] Written '{json_path}'.")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run the complete validation pipeline.

    Workflow
    --------
    1. Load and harmonise both datasets.
    2. Run all six validation functions.
    3. Generate summary with composite score and grade.
    4. Write validation_report.txt and validation_metrics.json.
    """
    np.random.seed(42)

    # ── Load data ──────────────────────────────────────────────────
    print("Loading datasets...")
    real_raw = pd.read_csv("SPIHF_Data.csv")
    synth_raw = pd.read_csv("synthetic_SPIHF.csv")

    real = _harmonise_columns(real_raw)
    synth = _harmonise_columns(synth_raw)

    print(f"  Real  : {real.shape[0]} rows x {real.shape[1]} cols")
    print(f"  Synth : {synth.shape[0]} rows x {synth.shape[1]} cols")
    print()

    # ── 1. Descriptive statistics ──────────────────────────────────
    print("[1/6] Validating distributions...")
    dist_results = validate_distributions(real, synth)

    # ── 2. KS tests ────────────────────────────────────────────────
    print("[2/6] Computing Kolmogorov-Smirnov tests...")
    ks_results = compute_ks_tests(real, synth)

    # ── 3. Wasserstein + JSD ───────────────────────────────────────
    print("[3/6] Computing Wasserstein & Jensen-Shannon metrics...")
    wd_results = compute_wasserstein_metrics(real, synth)

    # ── 4. Correlation comparison ──────────────────────────────────
    print("[4/6] Comparing correlations & mutual information...")
    corr_results = compare_correlations(real, synth)

    # ── 5. Feature importance ──────────────────────────────────────
    print("[5/6] Computing feature importance similarity...")
    importance_results = compute_feature_importance_similarity(real, synth, target="HER")

    # ── 6. Summary ─────────────────────────────────────────────────
    print("[6/6] Generating validation summary...")
    summary = generate_validation_summary(
        dist_results, ks_results, wd_results, corr_results,
        importance_results, real, synth,
    )

    # ── Save outputs ───────────────────────────────────────────────
    save_validation_report(
        summary, dist_results, ks_results, wd_results,
        corr_results, importance_results,
        report_path="validation_report.txt",
        json_path="validation_metrics.json",
    )

    # ── Console summary ────────────────────────────────────────────
    print()
    print(f"  OVERALL GRADE : {summary['grade']}  ({summary['composite_score']}%)")
    print(f"  KS pass rate  : {summary['ks_pass_rate'] * 100:.1f}%")
    print(f"  Mahalanobis   : {summary.get('mahalanobis_distance', 'N/A')}")
    print(f"  Mean JSD      : {summary.get('mean_jsd', 'N/A')}")
    print()
    print("[OK] Validation complete.")


if __name__ == "__main__":
    main()
