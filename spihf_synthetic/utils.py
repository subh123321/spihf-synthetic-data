"""
utils.py
========
Shared utility functions for the ``spihf_synthetic`` package.

Functions
---------
harmonise_columns       Rename raw CSV columns to canonical names and coerce numerics.
build_material_map      Build alias → canonical material name mapping.
short_label             Short axis-friendly label for a column name.
fmt_float               Format a float with NaN handling.
fmt_pct                 Format a 0–100 percentage.
detect_outliers_iqr     Count IQR-based outliers in a numeric series.
sanitise_for_json       Recursively convert NumPy types for JSON serialisation.
remove_near_duplicates  Remove near-duplicate synthetic samples via normalised L2.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from spihf_synthetic.config import (
    MATERIAL_ALIAS_GROUPS,
    NUMERIC_FEATURES,
    NUMERIC_FEATURES_VALIDATION,
    RAW_TO_CANONICAL,
    SHORT_LABELS,
    UNIT_SUFFIXES,
)


# ═══════════════════════════════════════════════════════════════════════
#  COLUMN HARMONISATION
# ═══════════════════════════════════════════════════════════════════════

def harmonise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CSV columns to canonical names and coerce numerics.

    Handles the inconsistent column naming in the raw SPIHF_Data.csv
    (spaces, slashes, embedded units) so all modules share identical
    column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (raw or synthetic).

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical column names and numeric types.
    """
    df = df.copy()
    df.rename(columns=RAW_TO_CANONICAL, inplace=True)

    # Handle roughness column name (mu symbol encoding varies)
    target_roughness = "Roughness (um)"
    for col in list(df.columns):
        if "Roughness" in col and col not in NUMERIC_FEATURES_VALIDATION:
            df.rename(columns={col: target_roughness}, inplace=True)
            break

    # Strip embedded unit strings from numeric cells
    for col in NUMERIC_FEATURES_VALIDATION:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        for suffix in UNIT_SUFFIXES:
            s = s.str.replace(suffix, "", regex=False)
        s = s.str.replace("°", "", regex=False)
        s = s.str.replace("<", "", regex=False)
        s = s.str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")

    # Forward-fill Material (raw CSV has blank rows for merged cells)
    if "Material" in df.columns:
        df["Material"] = df["Material"].replace(r"^\s*$", np.nan, regex=True)
        df["Material"] = df["Material"].ffill()
        df["Material"] = df["Material"].astype(str).str.strip()

    return df


# ═══════════════════════════════════════════════════════════════════════
#  MATERIAL ALIAS MAP
# ═══════════════════════════════════════════════════════════════════════

def build_material_map() -> Dict[str, str]:
    """Build a mapping of raw material aliases to canonical names.

    Returns
    -------
    Dict[str, str]
        ``{alias.strip(): canonical_name}`` for every known alias.
    """
    material_map: Dict[str, str] = {}
    for canonical, aliases in MATERIAL_ALIAS_GROUPS:
        for alias in aliases:
            material_map[alias.strip()] = canonical
    return material_map


# ═══════════════════════════════════════════════════════════════════════
#  FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════

def short_label(col: str) -> str:
    """Return a short, axis-friendly label for a column name.

    Parameters
    ----------
    col : str
        Full canonical column name.

    Returns
    -------
    str
        Abbreviated label, or first 20 characters if unknown.
    """
    return SHORT_LABELS.get(col, col[:20])


def fmt_float(val: Optional[float], decimals: int = 4) -> str:
    """Format a float to *decimals* places, handling ``None``/``NaN``.

    Parameters
    ----------
    val : float or None
        Value to format.
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted string, or ``"N/A"`` for missing values.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def fmt_pct(val: float) -> str:
    """Format a 0–100 percentage to two decimal places.

    Parameters
    ----------
    val : float
        Percentage value.

    Returns
    -------
    str
        Formatted string like ``"42.31%"``.
    """
    return f"{val:.2f}%"


# ═══════════════════════════════════════════════════════════════════════
#  OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_outliers_iqr(series: pd.Series) -> int:
    """Count IQR-based outliers (Tukey's fence) in a numeric series.

    An observation is an outlier if it lies below ``Q1 - 1.5·IQR`` or
    above ``Q3 + 1.5·IQR``.

    Parameters
    ----------
    series : pd.Series
        Numeric series.

    Returns
    -------
    int
        Number of outlier observations.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return 0
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


# ═══════════════════════════════════════════════════════════════════════
#  JSON SANITISATION
# ═══════════════════════════════════════════════════════════════════════

def sanitise_for_json(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python for JSON serialisation.

    Parameters
    ----------
    obj : Any
        Arbitrary nested structure (dict, list, numpy scalar, etc.).

    Returns
    -------
    Any
        JSON-safe equivalent.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise_for_json(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════
#  NEAR-DUPLICATE REMOVAL
# ═══════════════════════════════════════════════════════════════════════

def remove_near_duplicates(
    samples: List[pd.Series],
    numeric_cols: List[str],
    threshold: float = 0.005,
) -> List[pd.Series]:
    """Remove near-duplicate synthetic samples using normalised L2 distance.

    Two samples are "near-duplicates" if their Euclidean distance in
    min-max normalised feature space is below ``threshold``.

    Parameters
    ----------
    samples : List[pd.Series]
        List of synthetic samples.
    numeric_cols : List[str]
        Features to use for distance computation.
    threshold : float
        L2 distance below which a sample is considered a duplicate.

    Returns
    -------
    List[pd.Series]
        De-duplicated list (order preserved, later duplicates removed).
    """
    if len(samples) <= 1:
        return samples

    cols_avail = [c for c in numeric_cols if all(c in s.index for s in samples)]
    if not cols_avail:
        return samples

    mat = np.array(
        [[s.get(c, np.nan) for c in cols_avail] for s in samples],
        dtype=float,
    )

    # Replace NaN with column median for distance calc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_medians = np.nanmedian(mat, axis=0)
    for j in range(mat.shape[1]):
        mask = np.isnan(mat[:, j])
        mat[mask, j] = col_medians[j]

    # Normalise to [0, 1]
    mins = mat.min(axis=0)
    maxs = mat.max(axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    mat_norm = (mat - mins) / denom

    keep = [True] * len(samples)
    for i in range(len(samples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(samples)):
            if not keep[j]:
                continue
            dist = np.linalg.norm(mat_norm[i] - mat_norm[j])
            if dist < threshold:
                keep[j] = False

    return [s for s, k in zip(samples, keep) if k]
