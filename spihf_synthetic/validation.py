"""
validation.py
=============
Statistical validation of synthetic SPIHF data against the original
experimental dataset.

Validates:
  - Univariate distributions (mean, std, median, IQR, category frequencies).
  - Distributional distances (KS, Wasserstein, Jensen-Shannon, Mahalanobis).
  - Multivariate structure (correlation matrices, mutual information).
  - Feature importance rankings (Random Forest + mutual information).
  - Overall quality grade (A–F composite scoring).

Functions
---------
validate_distributions                 Univariate descriptive comparison.
compute_ks_tests                       Two-sample Kolmogorov-Smirnov tests.
compute_wasserstein_metrics            Wasserstein-1 + Jensen-Shannon per feature.
compare_correlations                   Correlation matrix + physics pairs.
compute_feature_importance_similarity  RF and MI importance ranking comparison.
generate_validation_summary            Aggregate into composite score/grade.
save_validation_report                 Write .txt report + .json metrics.
run_validation                         Full validation pipeline (convenience).
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy, ks_2samp, spearmanr, wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

from spihf_synthetic.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES_VALIDATION,
    PHYSICS_CORRELATION_PAIRS,
    RANDOM_SEED,
)
from spihf_synthetic.utils import harmonise_columns, sanitise_for_json


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _safe_corr(
    df: pd.DataFrame, col_a: str, col_b: str,
) -> Optional[float]:
    """Compute Pearson correlation, returning ``None`` if insufficient data."""
    if col_a not in df.columns or col_b not in df.columns:
        return None
    valid = df[[col_a, col_b]].dropna()
    if len(valid) < 3:
        return None
    r = valid[col_a].corr(valid[col_b])
    return float(r) if pd.notna(r) else None


def _jensen_shannon_divergence(
    p: np.ndarray, q: np.ndarray, n_bins: int = 30,
) -> float:
    """Compute Jensen-Shannon divergence between two 1-D samples.

    Parameters
    ----------
    p, q : np.ndarray
        1-D arrays of sample values.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        JSD in [0, ln(2)] (nats).
    """
    combined = np.concatenate([p, q])
    lo, hi = np.nanmin(combined), np.nanmax(combined)
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    hist_p, _ = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bins, density=True)
    eps = 1e-12
    hist_p = hist_p.astype(float) + eps
    hist_q = hist_q.astype(float) + eps
    hist_p /= hist_p.sum()
    hist_q /= hist_q.sum()
    m = 0.5 * (hist_p + hist_q)
    jsd = 0.5 * entropy(hist_p, m) + 0.5 * entropy(hist_q, m)
    return float(jsd)


def _mahalanobis_distance(
    real: pd.DataFrame, synth: pd.DataFrame, cols: List[str],
) -> Optional[float]:
    """Mahalanobis distance between centroids of two datasets.

    Parameters
    ----------
    real, synth : pd.DataFrame
        DataFrames with matching numeric columns.
    cols : List[str]
        Columns to include.

    Returns
    -------
    float or None
        Distance, or ``None`` if covariance is singular.
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
        return float(mahalanobis(mean_r, mean_s, cov_inv))
    except np.linalg.LinAlgError:
        try:
            cov_pinv = np.linalg.pinv(cov_r)
            diff = mean_r - mean_s
            return float(np.sqrt(np.dot(np.dot(diff, cov_pinv), diff)))
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════
#  1.  VALIDATE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════

def validate_distributions(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Compare univariate descriptive statistics between real and synthetic data.

    For every numeric feature: mean, std, median, IQR.
    For categorical features: category frequency distributions.

    Parameters
    ----------
    real : pd.DataFrame
        Original (real) dataset with canonical column names.
    synth : pd.DataFrame
        Synthetic dataset with canonical column names.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Nested dict keyed by feature name.
    """
    results: Dict[str, Dict[str, Any]] = {}

    def _pct(a: float, b: float) -> float:
        denom = abs(a) if abs(a) > 1e-12 else 1e-12
        return float(abs(a - b) / denom * 100.0)

    for col in NUMERIC_FEATURES_VALIDATION:
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

        results[col] = {
            "real_mean": float(r_mean), "synth_mean": float(s_mean),
            "pct_diff_mean": _pct(r_mean, s_mean),
            "real_std": float(r_std), "synth_std": float(s_std),
            "pct_diff_std": _pct(r_std, s_std),
            "real_median": float(r_med), "synth_median": float(s_med),
            "pct_diff_median": _pct(r_med, s_med),
            "real_iqr": float(r_iqr), "synth_iqr": float(s_iqr),
            "pct_diff_iqr": _pct(r_iqr, s_iqr),
            "real_n": int(len(rv)), "synth_n": int(len(sv)),
        }

    for col in CATEGORICAL_FEATURES:
        if col not in real.columns or col not in synth.columns:
            continue
        r_freq = real[col].value_counts(normalize=True).to_dict()
        s_freq = synth[col].value_counts(normalize=True).to_dict()
        all_cats = sorted(set(list(r_freq.keys()) + list(s_freq.keys())))
        freq_comparison: Dict[str, Any] = {}
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


# ═══════════════════════════════════════════════════════════════════════
#  2.  KOLMOGOROV-SMIRNOV TESTS
# ═══════════════════════════════════════════════════════════════════════

def compute_ks_tests(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Run two-sample KS tests for every numeric feature.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    alpha : float
        Significance level.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Per-feature ``ks_statistic``, ``p_value``, ``same_distribution``.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for col in NUMERIC_FEATURES_VALIDATION:
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


# ═══════════════════════════════════════════════════════════════════════
#  3.  WASSERSTEIN / EMD METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_wasserstein_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compute Wasserstein-1 distance and JSD per feature.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per-feature ``wasserstein_distance``, ``wasserstein_normalised``,
        ``jensen_shannon_divergence``.
    """
    results: Dict[str, Dict[str, float]] = {}
    for col in NUMERIC_FEATURES_VALIDATION:
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


# ═══════════════════════════════════════════════════════════════════════
#  4.  CORRELATION COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def compare_correlations(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare correlation matrices and domain-specific physics pairs.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.

    Returns
    -------
    Dict[str, Any]
        ``frobenius_norm``, ``mean_abs_corr_diff``, ``physics_pairs``,
        ``mutual_information``.
    """
    results: Dict[str, Any] = {}

    common_num = [c for c in NUMERIC_FEATURES_VALIDATION
                  if c in real.columns and c in synth.columns]
    r_corr = real[common_num].corr()
    s_corr = synth[common_num].corr()

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

    # Physics-specific pairs
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
            if expected_sign != 0:
                entry["physics_sign_real"] = bool(
                    np.sign(r_rho) == expected_sign
                )
                entry["physics_sign_synth"] = bool(
                    np.sign(s_rho) == expected_sign
                )
        else:
            entry["abs_difference"] = None
            entry["sign_preserved"] = None
        pairs_results[pair_key] = entry
    results["physics_pairs"] = pairs_results

    # Mutual information for physics pairs
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
                            random_state=RANDOM_SEED,
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


# ═══════════════════════════════════════════════════════════════════════
#  5.  FEATURE IMPORTANCE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════

def compute_feature_importance_similarity(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target: str = "HER",
) -> Dict[str, Any]:
    """Compare feature importance rankings between datasets.

    Uses Random Forest Gini importance and mutual information,
    with Spearman rank correlation for ranking comparison.

    Parameters
    ----------
    real : pd.DataFrame
        Original dataset.
    synth : pd.DataFrame
        Synthetic dataset.
    target : str
        Target column for importance ranking.

    Returns
    -------
    Dict[str, Any]
        Per-feature importances, Spearman rho, and verdict.
    """
    results: Dict[str, Any] = {}

    if target not in real.columns or target not in synth.columns:
        results["error"] = f"Target '{target}' not found in both datasets."
        return results

    predictors = [
        c for c in NUMERIC_FEATURES_VALIDATION
        if c != target and c in real.columns and c in synth.columns
    ]
    if len(predictors) < 2:
        results["error"] = "Insufficient predictor columns."
        return results

    r_clean = real[predictors + [target]].dropna()
    s_clean = synth[predictors + [target]].dropna()

    if len(r_clean) < 10 or len(s_clean) < 10:
        results["error"] = "Insufficient non-NaN rows for importance analysis."
        return results

    X_r, y_r = r_clean[predictors].values, r_clean[target].values
    X_s, y_s = s_clean[predictors].values, s_clean[target].values

    rf_r = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf_s = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_r.fit(X_r, y_r)
        rf_s.fit(X_s, y_s)

    imp_r_rf = rf_r.feature_importances_
    imp_s_rf = rf_s.feature_importances_

    mi_r = mutual_info_regression(
        X_r, y_r, random_state=RANDOM_SEED,
        n_neighbors=min(5, len(X_r) - 1),
    )
    mi_s = mutual_info_regression(
        X_s, y_s, random_state=RANDOM_SEED,
        n_neighbors=min(5, len(X_s) - 1),
    )

    rho_rf, pval_rf = spearmanr(imp_r_rf, imp_s_rf)
    rho_mi, pval_mi = spearmanr(mi_r, mi_s)

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
        "MARGINAL" if rho_rf >= 0.5 else "FAIL"
    )
    results["verdict_mi"] = (
        "PASS" if rho_mi >= 0.7 else
        "MARGINAL" if rho_mi >= 0.5 else "FAIL"
    )
    return results


# ═══════════════════════════════════════════════════════════════════════
#  6.  GENERATE VALIDATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def generate_validation_summary(
    dist_results: Dict[str, Dict[str, Any]],
    ks_results: Dict[str, Dict[str, Any]],
    wd_results: Dict[str, Dict[str, float]],
    corr_results: Dict[str, Any],
    importance_results: Dict[str, Any],
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> Dict[str, Any]:
    """Aggregate all validation results into a composite score and grade.

    Grading rubric (weighted):
      - KS pass rate (25 %)
      - Wasserstein score (25 %)
      - Correlation score (20 %)
      - Feature importance Spearman rho (15 %)
      - Descriptive stats fidelity (15 %)

    Grades: A (≥85), B (≥70), C (≥55), D (≥40), F (<40).

    Parameters
    ----------
    dist_results, ks_results, wd_results, corr_results, importance_results
        Results from the individual validation functions.
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

    # KS pass rate
    if ks_results:
        n_pass = sum(
            1 for v in ks_results.values() if v.get("same_distribution")
        )
        ks_pass_rate = n_pass / len(ks_results)
    else:
        ks_pass_rate = 0.0
    summary["ks_pass_rate"] = round(ks_pass_rate, 4)

    # Wasserstein score
    if wd_results:
        wd_norms = [
            v.get("wasserstein_normalised", 0.0) for v in wd_results.values()
        ]
        mean_wd = np.mean(wd_norms)
        wd_score = max(0.0, 1.0 - mean_wd)
    else:
        mean_wd = 1.0
        wd_score = 0.0
    summary["mean_wasserstein_normalised"] = round(float(mean_wd), 4)

    # Mean JSD
    if wd_results:
        jsds = [
            v.get("jensen_shannon_divergence", 0.0) for v in wd_results.values()
        ]
        summary["mean_jsd"] = round(float(np.mean(jsds)), 6)
    else:
        summary["mean_jsd"] = None

    # Mahalanobis distance
    common_num = [
        c for c in NUMERIC_FEATURES_VALIDATION
        if c in real.columns and c in synth.columns
    ]
    maha = _mahalanobis_distance(real, synth, common_num)
    summary["mahalanobis_distance"] = (
        round(maha, 4) if maha is not None else None
    )

    # Correlation Frobenius norm
    frob = corr_results.get("frobenius_norm", float("nan"))
    n_feat = len(common_num)
    frob_norm = (
        frob / max(np.sqrt(n_feat), 1.0) if not np.isnan(frob) else 1.0
    )
    corr_score = max(0.0, 1.0 - frob_norm)
    summary["correlation_frobenius_norm"] = (
        round(frob, 4) if not np.isnan(frob) else None
    )

    # Physics pair preservation
    pairs = corr_results.get("physics_pairs", {})
    if pairs:
        n_sign_preserved = sum(
            1 for v in pairs.values() if v.get("sign_preserved") is True
        )
        n_pairs_total = sum(
            1 for v in pairs.values() if v.get("sign_preserved") is not None
        )
        sign_pres_rate = n_sign_preserved / max(n_pairs_total, 1)
    else:
        sign_pres_rate = 0.0
    summary["physics_sign_preservation_rate"] = round(sign_pres_rate, 4)

    # Feature importance
    imp_rho = importance_results.get("spearman_rf", {}).get("rho", 0.0)
    if imp_rho is None or np.isnan(imp_rho):
        imp_rho = 0.0
    imp_score = max(0.0, imp_rho)

    # Mean % difference in descriptive stats
    pct_diffs: list[float] = []
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

    # Composite
    composite = (
        0.25 * ks_pass_rate
        + 0.25 * wd_score
        + 0.20 * corr_score
        + 0.15 * imp_score
        + 0.15 * stats_score
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


# ═══════════════════════════════════════════════════════════════════════
#  7.  SAVE VALIDATION REPORT
# ═══════════════════════════════════════════════════════════════════════

def save_validation_report(
    summary: Dict[str, Any],
    dist_results: Dict[str, Dict[str, Any]],
    ks_results: Dict[str, Dict[str, Any]],
    wd_results: Dict[str, Dict[str, float]],
    corr_results: Dict[str, Any],
    importance_results: Dict[str, Any],
    report_path: str = "outputs/reports/validation_report.txt",
    json_path: str = "outputs/reports/validation_metrics.json",
) -> None:
    """Write a human-readable .txt report and a machine-readable .json file.

    Parameters
    ----------
    summary : Dict
        From ``generate_validation_summary``.
    dist_results, ks_results, wd_results, corr_results, importance_results
        Results from individual validation functions.
    report_path : str
        Path for the .txt report.
    json_path : str
        Path for the .json metrics.

    Returns
    -------
    None
    """
    w = 72

    def hr(char: str = "=") -> str:
        return char * w

    lines: List[str] = []

    def section(title: str) -> None:
        lines.extend(["", hr("="), f"  {title}", hr("=")])

    def subsection(title: str) -> None:
        lines.extend(["", hr("-"), f"  {title}", hr("-")])

    lines.extend([hr(), "  SPIHF SYNTHETIC DATA VALIDATION REPORT", hr()])
    lines.append(f"  Generated : {summary.get('timestamp', 'N/A')}")
    lines.append(f"  Real samples     : {summary.get('real_samples', 'N/A')}")
    lines.append(f"  Synthetic samples: {summary.get('synth_samples', 'N/A')}")
    lines.append(
        f"  Overall Grade    : {summary.get('grade', 'N/A')}  "
        f"(score = {summary.get('composite_score', 'N/A')}%)"
    )
    lines.append(hr())

    # Sub-scores
    section("COMPOSITE SCORE BREAKDOWN")
    sub = summary.get("sub_scores", {})
    lines.append(f"  KS pass rate          (25%) : "
                 f"{sub.get('ks_pass_rate_25pct', 'N/A')}%")
    lines.append(f"  Wasserstein score     (25%) : "
                 f"{sub.get('wasserstein_score_25pct', 'N/A')}%")
    lines.append(f"  Correlation score     (20%) : "
                 f"{sub.get('correlation_score_20pct', 'N/A')}%")
    lines.append(f"  Importance score      (15%) : "
                 f"{sub.get('importance_score_15pct', 'N/A')}%")
    lines.append(f"  Stats fidelity        (15%) : "
                 f"{sub.get('stats_fidelity_15pct', 'N/A')}%")

    # KS tests
    section("KOLMOGOROV-SMIRNOV TESTS (alpha = 0.05)")
    lines.append(f"  {'Feature':<50s}  {'KS Stat':>8s}  "
                 f"{'p-value':>10s}  {'Result':>6s}")
    lines.append(f"  {'-'*50}  {'-'*8}  {'-'*10}  {'-'*6}")
    for feat, vals in ks_results.items():
        stat = vals.get("ks_statistic", 0.0)
        pval = vals.get("p_value", 0.0)
        same = "PASS" if vals.get("same_distribution") else "FAIL"
        lines.append(f"  {feat[:50]:<50s}  {stat:>8.4f}  "
                     f"{pval:>10.6f}  {same:>6s}")
    n_pass = sum(
        1 for v in ks_results.values() if v.get("same_distribution")
    )
    lines.append(f"\n  Summary: {n_pass}/{len(ks_results)} features "
                 f"passed KS test.")

    # Footer
    lines.extend(["", hr(), "  END OF VALIDATION REPORT", hr()])

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[save_validation_report] Written '{report_path}' "
          f"({len(lines)} lines).")

    # JSON metrics
    json_output: Dict[str, Any] = {
        "summary": summary,
        "distribution_statistics": dist_results,
        "ks_tests": ks_results,
        "wasserstein_and_jsd": wd_results,
        "correlation_analysis": corr_results,
        "feature_importance": importance_results,
    }
    json_clean = sanitise_for_json(json_output)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_clean, f, indent=2, ensure_ascii=False)
    print(f"[save_validation_report] Written '{json_path}'.")


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_validation(
    real_path: str = "SPIHF_Data.csv",
    synth_path: str = "outputs/synthetic_SPIHF.csv",
    report_path: str = "outputs/reports/validation_report.txt",
    json_path: str = "outputs/reports/validation_metrics.json",
) -> Dict[str, Any]:
    """Run the complete validation pipeline end-to-end.

    Parameters
    ----------
    real_path : str
        Path to the original dataset CSV.
    synth_path : str
        Path to the synthetic dataset CSV.
    report_path : str
        Output path for the .txt report.
    json_path : str
        Output path for the .json metrics.

    Returns
    -------
    Dict[str, Any]
        The validation summary dictionary.
    """
    np.random.seed(RANDOM_SEED)

    print("Loading datasets...")
    real = harmonise_columns(pd.read_csv(real_path))
    synth = harmonise_columns(pd.read_csv(synth_path))
    print(f"  Real  : {real.shape[0]} rows x {real.shape[1]} cols")
    print(f"  Synth : {synth.shape[0]} rows x {synth.shape[1]} cols\n")

    print("[1/6] Validating distributions...")
    dist_results = validate_distributions(real, synth)

    print("[2/6] Computing Kolmogorov-Smirnov tests...")
    ks_results = compute_ks_tests(real, synth)

    print("[3/6] Computing Wasserstein & Jensen-Shannon metrics...")
    wd_results = compute_wasserstein_metrics(real, synth)

    print("[4/6] Comparing correlations & mutual information...")
    corr_results = compare_correlations(real, synth)

    print("[5/6] Computing feature importance similarity...")
    importance_results = compute_feature_importance_similarity(
        real, synth, target="HER"
    )

    print("[6/6] Generating validation summary...")
    summary = generate_validation_summary(
        dist_results, ks_results, wd_results, corr_results,
        importance_results, real, synth,
    )

    save_validation_report(
        summary, dist_results, ks_results, wd_results,
        corr_results, importance_results,
        report_path=report_path, json_path=json_path,
    )

    print(f"\n  OVERALL GRADE : {summary['grade']}  "
          f"({summary['composite_score']}%)")
    print(f"  KS pass rate  : {summary['ks_pass_rate'] * 100:.1f}%")
    print(f"  Mahalanobis   : {summary.get('mahalanobis_distance', 'N/A')}")
    print(f"  Mean JSD      : {summary.get('mean_jsd', 'N/A')}\n")
    print("[OK] Validation complete.")
    return summary
