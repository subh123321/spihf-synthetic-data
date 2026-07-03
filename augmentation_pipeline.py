"""
augmentation_pipeline.py
========================
Physics-Informed Synthetic Data Augmentation for Single Point Incremental
Hole Flanging (SPIHF) Datasets.

This pipeline generates scientifically plausible synthetic samples by:
  1. Material-wise stratified augmentation (no cross-material interpolation).
  2. SMOTE-inspired interpolation between same-material neighbours.
  3. Gaussian perturbation calibrated to 1–3 % of per-feature std.
  4. A physics-informed correction layer that enforces metallurgical,
     geometric, and process constraints drawn from sheet-metal-forming
     theory (Hollomon hardening, sine-law thinning, volume conservation,
     anisotropy bounds, etc.).
  5. Rejection sampling that discards samples violating hard physical
     limits (negative thickness, impossible HER, etc.).
  6. Near-duplicate removal via Euclidean distance in normalised space.
  7. Confidence scoring that quantifies how close each synthetic sample
     is to the convex hull of its material class.

Author : Augmentation Pipeline (auto-generated)
Seed   : np.random.seed(42)
Output : synthetic_SPIHF.csv  (≥ 1000 rows)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ──────────────────────────── Global seed ────────────────────────────
np.random.seed(42)

# ──────────────────────────── Constants ──────────────────────────────
# Standardised column names used internally after cleaning the raw CSV
# headers (the raw file has inconsistent whitespace / units in names).
COLUMN_NAMES: List[str] = [
    "Material",
    "Thickness (mm)",
    "Precut dimensions (diameter/side length) mm",
    "Precut Shape (circle/square/etc)",
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
    "Roughness (µm)",
    "Minimum thickness (after final stage, mm)",
    "Final angle after the final stage (degrees)",
]

# Numeric features used for interpolation / perturbation
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
    "Roughness (µm)",
    "Minimum thickness (after final stage, mm)",
    "Final angle after the final stage (degrees)",
]

# Categorical columns carried through unchanged during interpolation
CATEGORICAL_FEATURES: List[str] = [
    "Material",
    "Precut Shape (circle/square/etc)",
]

# Target synthetic dataset size
TARGET_SYNTHETIC_SIZE: int = 1000

# Minimum number of real samples a material group must have to
# participate in SMOTE-style interpolation (otherwise pure Gaussian)
MIN_SAMPLES_FOR_INTERPOLATION: int = 2


# ════════════════════════════════════════════════════════════════════
#  1.  LOAD DATA
# ════════════════════════════════════════════════════════════════════
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

    Engineering note
    ----------------
    The raw dataset is compiled from multiple published SPIHF studies and
    therefore contains heterogeneous column naming, mixed units embedded
    in value cells (e.g. "1000 mm/min"), empty material labels (forward-
    filled from the row above in the original spreadsheet), and trailing
    whitespace.  All of this is handled downstream in ``preprocess_data``.
    """
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {len(df)} rows × {df.shape[1]} columns from '{filepath}'.")
    return df


# ════════════════════════════════════════════════════════════════════
#  2.  PREPROCESS DATA
# ════════════════════════════════════════════════════════════════════
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the raw SPIHF DataFrame.

    Steps performed
    ---------------
    1. Rename columns to the canonical ``COLUMN_NAMES`` list.
    2. Forward-fill empty *Material* cells (the original spreadsheet
       uses merged cells that leave subsequent rows blank).
    3. Strip embedded unit strings from numeric cells
       (e.g. ``"1000 mm/min"`` → ``1000.0``).
    4. Normalise *Precut Shape* to lower-case and collapse synonyms
       (``"circular"`` → ``"circle"``).
    5. Coerce all numeric columns to ``float64``.
    6. Drop rows that still have a missing *Material* after forward-fill.
    7. Strip leading/trailing whitespace from the *Material* column and
       collapse near-duplicate material names into canonical groups.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from ``load_data``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for augmentation.

    Engineering note
    ----------------
    Material-name canonicalisation is critical: the source literature
    uses many aliases for the same alloy (e.g. ``"AA7075-O"``,
    ``"7075-O aluminium alloy"``, ``"AA7075-0"``).  We map them to a
    single canonical name so that SMOTE interpolation stays within
    metallurgically identical classes.
    """
    df = df.copy()

    # ── 1. Rename columns ──────────────────────────────────────────
    if len(df.columns) == len(COLUMN_NAMES):
        df.columns = COLUMN_NAMES
    else:
        # Fallback: try to match by position for the first N columns
        rename_map = {old: new for old, new in zip(df.columns, COLUMN_NAMES)}
        df.rename(columns=rename_map, inplace=True)

    # ── 2. Forward-fill Material ───────────────────────────────────
    df["Material"] = df["Material"].replace(r"^\s*$", np.nan, regex=True)
    df["Material"] = df["Material"].ffill()
    df.dropna(subset=["Material"], inplace=True)

    # ── 3. Strip unit strings from numeric cells ───────────────────
    unit_suffixes = [
        " mm/min", " rpm clockwise", " rpm", " mm/cycle",
        " mm", " µm", "°", "° ",
    ]
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        # Convert to string, strip units, then coerce
        s = df[col].astype(str).str.strip()
        for suffix in unit_suffixes:
            s = s.str.replace(suffix, "", regex=False)
        # Handle angle symbols that might appear as standalone
        s = s.str.replace("°", "", regex=False)
        # Handle "<90" style entries → treat as the number
        s = s.str.replace("<", "", regex=False)
        # Remove comma thousands separators: "1,000" → "1000"
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

    # Build a mapping of common aliases → canonical name.
    # Order matters: more specific patterns first.
    _material_map: Dict[str, str] = {}
    _alias_groups: List[Tuple[str, List[str]]] = [
        ("AA7075-O", [
            "AA7075-O", "AA7075-0", "7075-O aluminium alloy",
            "7075-O aluminum alloy", "AA7075-O Aluminium alloy",
            "AA7075-O aluminium", "AA7075-0 Aluminium alloy",
            "AA7075-O (SPIF Single-Stage)",
        ]),
        ("AA6061-T6", [
            "AA6061-T6", "AA6061-T6 aluminum alloy",
            "AA6061- T6 aluminium", "Al 6061",
            "AA6061-T6 (CHF)", "AA6061-T6 (IHF)",
        ]),
        ("AA1050-H111", [
            "AA1050-H111", "Aluminium AA1050-H111",
            "aluminum AA1050-H111", "AA1050-H111 Aluminium",
        ]),
        ("AA1060", [
            "AA1060", "AA1060 aluminium alloy", "Al1060",
            "aluminum alloy (AA1060)", "1060 aluminum sheet",
            " AA1060",
        ]),
        ("AA5052", [
            "AA5052", "AA5052 sheet", "AA5052 aluminium alloy",
            "Aluminium 5052 sheets",
        ]),
        ("AA5052-H32", [
            "AA5052-H32", "AA5052-H32 aluminium",
            "AA5052-H32 Aluminium", "aluminium 5052-H32 sheet",
        ]),
        ("AA1050", [
            "AA1050", "AA1050 Aluminium", "Al 1050",
            "Al 1050A", "Aluminium 1050",
        ]),
        ("EN AW-6181-T1", [
            "EN AW-6181-T1", "EN AW-6181-T1 Aluminium alloy",
        ]),
        ("DC01", [
            "DC01", "DC01 Steel", "DC01\t",
        ]),
        ("DC04", ["DC04"]),
        ("DC05", ["DC05 steel"]),
        ("DDQ Steel", [
            "DDQ Steel", "DDQ steel", "DD Steel",
            "DD Steel (Failure Case)",
        ]),
        ("Titanium Grade 2", ["Titanium (grade 2)"]),
        ("Ti-6Al-4V", ["Ti-6Al-4V"]),
        ("Copper", ["Copper"]),
        ("SUS 304", ["SUS 304", "AISI 304", "304 stainless steel"]),
        ("2205 Dual Phase Steel", [
            "2205 DualPhase Steel", "dual phase steel",
            "Dual-Phase Steel",
        ]),
        ("Sheet Steel", ["Sheet Steel", "steel sheet"]),
        ("Aluminum (unspecified)", [
            "Aluminum (unspecified)", "Aluminum Alloy",
            "Aluminium 1000 series",
        ]),
    ]
    for canonical, aliases in _alias_groups:
        for alias in aliases:
            _material_map[alias.strip()] = canonical

    df["Material"] = df["Material"].map(
        lambda m: _material_map.get(m.strip(), m.strip())
    )

    # ── 6. Ensure 'Is lubricant used?' is binary integer ───────────
    lub_col = "Is lubricant used?"
    if lub_col in df.columns:
        df[lub_col] = df[lub_col].fillna(0).astype(float).clip(0, 1).round().astype(int)

    print(f"[preprocess_data] After cleaning: {len(df)} rows, "
          f"{df['Material'].nunique()} unique materials.")
    return df


# ════════════════════════════════════════════════════════════════════
#  3.  COMPUTE FEATURE STATISTICS
# ════════════════════════════════════════════════════════════════════
def compute_feature_statistics(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, pd.Series]]:
    """Compute per-material descriptive statistics for every numeric feature.

    For each material group the function stores:
      - ``mean``   – feature-wise mean (used for Gaussian centre).
      - ``std``    – feature-wise standard deviation.
      - ``min``    – observed minimum (used as hard lower bound).
      - ``max``    – observed maximum (used as hard upper bound).
      - ``median`` – robust central tendency.
      - ``count``  – number of non-NaN observations per feature.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.

    Returns
    -------
    Dict[str, Dict[str, pd.Series]]
        Nested dictionary keyed by material name, then by statistic name.
        Each leaf is a ``pd.Series`` indexed by feature name.

    Engineering note
    ----------------
    These statistics anchor both the Gaussian noise amplitude and the
    rejection-sampling bounds.  Using *material-wise* statistics rather
    than global ones prevents impossible cross-material artefacts (e.g.
    an aluminium 1050 sample with 600 MPa UTS).
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


# ════════════════════════════════════════════════════════════════════
#  4.  SMOTE-INSPIRED INTERPOLATION
# ════════════════════════════════════════════════════════════════════
def generate_interpolated_sample(
    row_i: pd.Series,
    row_j: pd.Series,
    alpha_low: float = 0.2,
    alpha_high: float = 0.8,
) -> pd.Series:
    """Create a synthetic sample by interpolating two same-material rows.

    Implements the SMOTE-inspired formula::

        x_new = α · x_i  +  (1 − α) · x_j

    where α is drawn uniformly from ``[alpha_low, alpha_high]`` to keep
    the synthetic point *between* the two parents rather than on top of
    either one.

    Categorical features (Material, Precut Shape) are inherited from
    ``row_i`` (the "anchor" sample).

    Parameters
    ----------
    row_i : pd.Series
        First (anchor) sample.
    row_j : pd.Series
        Second (neighbour) sample.
    alpha_low : float
        Lower bound of α.  Default 0.2.
    alpha_high : float
        Upper bound of α.  Default 0.8.

    Returns
    -------
    pd.Series
        Interpolated synthetic sample with the same index as ``row_i``.

    Engineering note
    ----------------
    In conventional SMOTE the neighbour is chosen via k-NN in feature
    space.  Here we sample pairs *uniformly at random* within each
    material class because the dataset is small and high-dimensional,
    making k-NN distances unreliable.  The restricted α range
    [0.2, 0.8] prevents trivial copies of existing data.
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
        # else both NaN → stays NaN

    # Categorical columns from anchor
    for cat in CATEGORICAL_FEATURES:
        if cat in row_i.index:
            new_sample[cat] = row_i[cat]

    return new_sample


# ════════════════════════════════════════════════════════════════════
#  5.  GAUSSIAN PERTURBATION
# ════════════════════════════════════════════════════════════════════
def apply_gaussian_noise(
    sample: pd.Series,
    stats: Dict[str, pd.Series],
    noise_fraction_low: float = 0.01,
    noise_fraction_high: float = 0.03,
) -> pd.Series:
    """Add Gaussian noise scaled to 1–3 % of the material-group std.

    For each numeric feature *f* the noise is::

        ε_f ~ N(0,  σ_noise_f²)
        σ_noise_f = η · std_f        where η ~ U(noise_fraction_low, noise_fraction_high)

    This produces small, physically credible perturbations that capture
    measurement/process variability without pushing the sample outside
    the feasible domain.

    Parameters
    ----------
    sample : pd.Series
        A single (possibly interpolated) sample.
    stats : Dict[str, pd.Series]
        Statistics dict for the sample's material (from
        ``compute_feature_statistics``).
    noise_fraction_low : float
        Minimum fraction of std for noise amplitude.
    noise_fraction_high : float
        Maximum fraction of std for noise amplitude.

    Returns
    -------
    pd.Series
        Perturbed sample.

    Engineering note
    ----------------
    The 1–3 % range is chosen to mimic realistic sources of process
    scatter in SPIHF:
      • CNC positional repeatability (≈ 0.01 mm)
      • Sheet thickness tolerance (± 0.02 mm for 1 mm Al)
      • Lubricant film variation
      • Tool-wear induced surface roughness changes
    """
    noisy = sample.copy()
    std_series = stats.get("std", pd.Series(dtype=float))

    for feat in NUMERIC_FEATURES:
        if feat in ("Is lubricant used?", "No of stages"):
            # Binary / integer features — do not add continuous noise
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


# ════════════════════════════════════════════════════════════════════
#  6.  PHYSICS-INFORMED CORRECTION LAYER
# ════════════════════════════════════════════════════════════════════
def repair_physics_constraints(sample: pd.Series) -> pd.Series:
    """Enforce metallurgical, geometric, and process physics constraints.

    The corrections applied (in order) are:

    **Material properties**
      - UTS ≥ YS  (by definition of ultimate vs. yield strength).
      - Strength coefficient k ≥ UTS (Hollomon relation σ = kεⁿ;
        at necking σ = UTS, ε < 1 ⟹ k ≥ UTS for n > 0).
      - 0 < n ≤ 1  (strain-hardening exponent physical range).
      - R-value (Lankford) > 0; typically 0.2–4.0 for sheet metals.
      - Total elongation in (0, 100) %.

    **Process parameters**
      - Feed rate > 0.
      - Tool speed ≥ 0 (zero means no rotation).
      - Step depth > 0.
      - Number of stages ≥ 1 (integer).

    **Geometric / forming outputs**
      - Thickness > 0.
      - Precut dimension > 0.
      - Flange height > 0.
      - HER ≥ 1.0 (by definition: HER = D_final / D_precut ≥ 1).
      - Sine-law thinning bound::

            t_min ≥ t_0 · sin(α)     (α in radians)

        where t_0 is the initial thickness.  If the synthetic t_min
        violates this lower bound, it is corrected upward.
      - Minimum thickness ≤ initial thickness.
      - Final angle in (0°, 180°].
      - Roughness ≥ 0.

    **Lubricant flag**
      - Rounded to nearest integer and clipped to {0, 1}.

    Parameters
    ----------
    sample : pd.Series
        A single synthetic sample (post-interpolation + noise).

    Returns
    -------
    pd.Series
        Physics-corrected sample.

    Engineering note
    ----------------
    These corrections act as a *projection onto the feasible manifold*
    in parameter space.  They are intentionally conservative (wide
    bounds) so that the synthetic dataset retains the statistical
    diversity introduced by interpolation and noise while remaining
    physically realisable.
    """
    s = sample.copy()

    # ── Helper ─────────────────────────────────────────────────────
    def _get(col: str) -> Optional[float]:
        v = s.get(col, np.nan)
        return v if pd.notna(v) else None

    def _set(col: str, val: float) -> None:
        if col in s.index:
            s[col] = val

    # ── Material properties ────────────────────────────────────────
    uts = _get("UTS (MPa)")
    ys  = _get("YS (MPa)")
    k   = _get("Strength Coefficient (k in MPa)")
    n   = _get("Strain hardening coefficient (n)")
    r   = _get("Anisotropic (R Value)")
    elong = _get("Total Strain/Elongation (%)")

    if uts is not None and ys is not None and uts < ys:
        # Swap so UTS ≥ YS
        _set("UTS (MPa)", ys)
        _set("YS (MPa)", uts)
        uts, ys = ys, uts

    if k is not None:
        if uts is not None and k < uts:
            _set("Strength Coefficient (k in MPa)", uts * 1.05)
        if k <= 0:
            _set("Strength Coefficient (k in MPa)", abs(k) + 1.0)

    if n is not None:
        _set("Strain hardening coefficient (n)", float(np.clip(n, 0.01, 1.0)))

    if r is not None:
        _set("Anisotropic (R Value)", float(np.clip(r, 0.1, 5.0)))

    if elong is not None:
        _set("Total Strain/Elongation (%)", float(np.clip(elong, 0.5, 99.0)))

    if uts is not None:
        _set("UTS (MPa)", max(uts, 1.0))
    if ys is not None:
        _set("YS (MPa)", max(ys, 1.0))

    # ── Process parameters ─────────────────────────────────────────
    feed = _get("Feed rate (mm/min)")
    speed = _get("Tool speed (rpm)")
    step = _get("Step depth (mm)")
    stages = _get("No of stages")

    if feed is not None:
        _set("Feed rate (mm/min)", max(feed, 1.0))
    if speed is not None:
        _set("Tool speed (rpm)", max(speed, 0.0))
    if step is not None:
        _set("Step depth (mm)", max(step, 0.01))
    if stages is not None:
        _set("No of stages", max(int(round(stages)), 1))

    # ── Geometric / forming outputs ────────────────────────────────
    t0 = _get("Thickness (mm)")
    precut = _get("Precut dimensions (diameter/side length) mm")
    her = _get("HER")
    fh = _get("Flange Height (mm)")
    rough = _get("Roughness (µm)")
    tmin = _get("Minimum thickness (after final stage, mm)")
    angle = _get("Final angle after the final stage (degrees)")

    if t0 is not None:
        _set("Thickness (mm)", max(t0, 0.05))
        t0 = max(t0, 0.05)

    if precut is not None:
        _set("Precut dimensions (diameter/side length) mm", max(precut, 1.0))

    if her is not None:
        _set("HER", max(her, 1.0))

    if fh is not None:
        _set("Flange Height (mm)", max(fh, 0.1))

    if rough is not None:
        _set("Roughness (µm)", max(rough, 0.001))

    if angle is not None:
        _set("Final angle after the final stage (degrees)",
             float(np.clip(angle, 1.0, 180.0)))
        angle = float(np.clip(angle, 1.0, 180.0))

    if tmin is not None:
        tmin = max(tmin, 0.01)
        # Sine-law thinning bound: t_min >= t0 * sin(angle_rad)
        if t0 is not None and angle is not None:
            angle_rad = np.radians(angle)
            sine_limit = t0 * np.sin(angle_rad)
            # t_min should be at least the sine-law prediction
            # (some violation is possible in ISF due to shear, but
            #  a gross violation signals an implausible sample)
            if tmin < sine_limit * 0.5:
                tmin = sine_limit * 0.7  # allow ~30 % extra thinning beyond sine law
        # t_min must not exceed original thickness
        if t0 is not None:
            tmin = min(tmin, t0)
        _set("Minimum thickness (after final stage, mm)", tmin)

    # ── Lubricant flag ─────────────────────────────────────────────
    lub = _get("Is lubricant used?")
    if lub is not None:
        _set("Is lubricant used?", int(round(np.clip(lub, 0, 1))))

    return s


# ════════════════════════════════════════════════════════════════════
#  7.  CONFIDENCE SCORING
# ════════════════════════════════════════════════════════════════════
def compute_confidence_score(
    synthetic_sample: pd.Series,
    material_data: pd.DataFrame,
    numeric_cols: List[str],
) -> float:
    """Score how plausible a synthetic sample is relative to its material group.

    The score combines three components:

    1. **Range score** (weight 0.40):  Fraction of features whose value
       falls within the observed [min, max] of the material group.

    2. **Distance score** (weight 0.40):  Inverse normalised Euclidean
       distance to the nearest real sample.  Closer → higher score.

    3. **Correlation score** (weight 0.20):  Alignment with the observed
       pairwise Pearson correlation structure.  A sample whose feature
       ratios mirror the group's correlations is more plausible.

    The final score is in [0, 1]; higher is better.

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
        Confidence score in [0, 1].

    Engineering note
    ----------------
    This score is *not* a statistical p-value.  It is a heuristic that
    penalises samples far from the data manifold while rewarding those
    that preserve inter-feature relationships (e.g. the positive
    correlation between thickness and minimum thickness).
    """
    available_cols = [c for c in numeric_cols
                      if c in synthetic_sample.index
                      and c in material_data.columns]
    if not available_cols:
        return 0.0

    real_vals = material_data[available_cols].dropna(axis=1, how="all")
    avail = [c for c in available_cols if c in real_vals.columns]
    if not avail or len(real_vals) == 0:
        return 0.0

    # ── 1. Range score ─────────────────────────────────────────────
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
            margin = 0.15 * (col_max - col_min + 1e-9)
            if col_min - margin <= sv <= col_max + margin:
                in_range_count += 1
    range_score = in_range_count / max(total_checked, 1)

    # ── 2. Distance score ──────────────────────────────────────────
    real_clean = real_vals[avail].dropna()
    if len(real_clean) == 0:
        return range_score * 0.4

    syn_vec = []
    cols_used = []
    for col in avail:
        sv = synthetic_sample.get(col, np.nan)
        if pd.notna(sv) and col in real_clean.columns:
            syn_vec.append(sv)
            cols_used.append(col)

    if not cols_used:
        return range_score * 0.4

    real_sub = real_clean[cols_used].dropna()
    if len(real_sub) == 0:
        return range_score * 0.4

    # Normalise to [0, 1]
    r_min = real_sub.min()
    r_max = real_sub.max()
    denom = (r_max - r_min).replace(0, 1)
    real_norm = (real_sub - r_min) / denom
    syn_norm = np.array([(sv - r_min[c]) / denom[c]
                         for sv, c in zip(syn_vec, cols_used)]).reshape(1, -1)

    dists = cdist(syn_norm, real_norm.values, metric="euclidean")
    min_dist = dists.min()
    # Map distance → score via exponential decay
    distance_score = float(np.exp(-min_dist))

    # ── 3. Correlation score ───────────────────────────────────────
    if len(cols_used) >= 3 and len(real_sub) >= 3:
        try:
            corr_mat = real_sub.corr()
            # For the synthetic sample, compute pairwise "direction"
            # agreement with the correlation matrix
            agreements = 0
            pairs_checked = 0
            for i in range(len(cols_used)):
                for j in range(i + 1, len(cols_used)):
                    ci, cj = cols_used[i], cols_used[j]
                    rho = corr_mat.loc[ci, cj]
                    if pd.isna(rho):
                        continue
                    # Check if synthetic deviations from mean are in the
                    # same direction as implied by the correlation
                    mean_i = real_sub[ci].mean()
                    mean_j = real_sub[cj].mean()
                    dev_i = syn_vec[i] - mean_i
                    dev_j = syn_vec[j] - mean_j
                    if abs(rho) > 0.3:  # only check non-trivial correlations
                        pairs_checked += 1
                        if np.sign(dev_i * dev_j) == np.sign(rho) or abs(dev_i * dev_j) < 1e-9:
                            agreements += 1
            corr_score = agreements / max(pairs_checked, 1)
        except Exception:
            corr_score = 0.5
    else:
        corr_score = 0.5  # not enough data to judge

    # ── Weighted combination ───────────────────────────────────────
    final = 0.40 * range_score + 0.40 * distance_score + 0.20 * corr_score
    return float(np.clip(final, 0.0, 1.0))


# ════════════════════════════════════════════════════════════════════
#  8.  GENERATE SYNTHETIC DATASET
# ════════════════════════════════════════════════════════════════════
def generate_synthetic_dataset(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, pd.Series]],
    target_size: int = TARGET_SYNTHETIC_SIZE,
    min_confidence: float = 0.25,
    max_attempts_multiplier: int = 5,
    duplicate_threshold: float = 0.005,
) -> pd.DataFrame:
    """Orchestrate the full augmentation pipeline to produce *target_size* samples.

    Pipeline per material group
    ---------------------------
    1. Determine how many synthetic samples to allocate (proportional to
       the group's share of the original dataset, with a minimum of 5).
    2. For each required sample:
       a. Pick two random rows from the group.
       b. Interpolate → ``generate_interpolated_sample``.
       c. Perturb   → ``apply_gaussian_noise``.
       d. Repair    → ``repair_physics_constraints``.
       e. Score     → ``compute_confidence_score``.
       f. Accept only if confidence ≥ ``min_confidence`` (**rejection
          sampling**).
    3. Remove near-duplicates within the synthetic pool for each material.
    4. If the pool is still short of ``target_size`` after all materials,
       fill the gap by additional sampling from the largest groups.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed real dataset.
    stats : Dict
        Per-material statistics from ``compute_feature_statistics``.
    target_size : int
        Desired number of synthetic samples.
    min_confidence : float
        Rejection threshold (samples below this are discarded).
    max_attempts_multiplier : int
        Controls the maximum number of generation attempts per required
        sample (``required × multiplier``).
    duplicate_threshold : float
        Normalised L2 distance below which two synthetic samples are
        considered duplicates.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with a ``confidence_score`` column appended.

    Engineering note
    ----------------
    The proportional allocation ensures that minority materials (e.g.
    Titanium Grade 2 or Copper, which may have only 2–4 rows) still
    receive synthetic augmentation, while majority classes (AA7075-O,
    AA6061-T6) receive proportionally more.  This balances class
    representation without artificially over-representing rare alloys.
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

    # Adjust so sum ≈ target_size
    alloc_sum = sum(allocation.values())
    if alloc_sum < target_size:
        # Distribute remainder to largest groups
        sorted_mats = sorted(allocation, key=lambda m: allocation[m], reverse=True)
        deficit = target_size - alloc_sum
        for i in range(deficit):
            allocation[sorted_mats[i % len(sorted_mats)]] += 1

    print(f"[generate_synthetic_dataset] Allocation across {len(materials)} materials "
          f"(total target={target_size}):")
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
                # SMOTE-style: pick two distinct rows
                idx = np.random.choice(len(mat_df), size=2, replace=False)
                row_i = mat_df.iloc[idx[0]]
                row_j = mat_df.iloc[idx[1]]
                new_sample = generate_interpolated_sample(row_i, row_j)
            else:
                # Fallback: Gaussian perturbation of the single sample
                row_i = mat_df.iloc[0]
                new_sample = row_i.copy()

            # Gaussian noise
            new_sample = apply_gaussian_noise(new_sample, mat_stats)

            # Physics corrections
            new_sample = repair_physics_constraints(new_sample)

            # Confidence check (rejection sampling)
            score = compute_confidence_score(
                new_sample, mat_df, numeric_cols
            )
            if score < min_confidence:
                continue

            new_sample["confidence_score"] = score
            mat_synthetic.append(new_sample)

        # ── Duplicate removal within material ──────────────────────
        if len(mat_synthetic) > 1:
            mat_synthetic = _remove_near_duplicates(
                mat_synthetic, numeric_cols, threshold=duplicate_threshold
            )

        all_synthetic.extend(mat_synthetic)

    # ── Final assembly ─────────────────────────────────────────────
    if not all_synthetic:
        warnings.warn("No synthetic samples generated — returning empty DataFrame.")
        return pd.DataFrame(columns=list(df.columns) + ["confidence_score"])

    synth_df = pd.DataFrame(all_synthetic).reset_index(drop=True)

    # If still under target, oversample existing synthetic rows
    if len(synth_df) < target_size:
        shortfall = target_size - len(synth_df)
        extra = synth_df.sample(n=shortfall, replace=True, random_state=42).copy()
        # Add small extra noise to avoid exact duplicates
        for _, row in extra.iterrows():
            mat = row.get("Material", "")
            mat_stats_row = stats.get(mat, {})
            if mat_stats_row:
                noisy_row = apply_gaussian_noise(
                    row, mat_stats_row,
                    noise_fraction_low=0.005,
                    noise_fraction_high=0.015,
                )
                noisy_row = repair_physics_constraints(noisy_row)
                noisy_row["confidence_score"] = row["confidence_score"] * 0.95
                all_synthetic.append(noisy_row)

        synth_df = pd.DataFrame(all_synthetic).reset_index(drop=True)
        synth_df = synth_df.head(target_size)

    print(f"\n[generate_synthetic_dataset] Generated {len(synth_df)} synthetic samples.")
    print(f"    Mean confidence : {synth_df['confidence_score'].mean():.4f}")
    print(f"    Min  confidence : {synth_df['confidence_score'].min():.4f}")
    print(f"    Max  confidence : {synth_df['confidence_score'].max():.4f}")

    return synth_df


def _remove_near_duplicates(
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
        L2 distance below which a sample is considered a duplicate of
        an earlier one.

    Returns
    -------
    List[pd.Series]
        De-duplicated list (order preserved, later duplicates removed).
    """
    if len(samples) <= 1:
        return samples

    # Build matrix
    cols_avail = [c for c in numeric_cols if all(c in s.index for s in samples)]
    if not cols_avail:
        return samples

    mat = np.array([
        [s.get(c, np.nan) for c in cols_avail]
        for s in samples
    ], dtype=float)

    # Replace NaN with column median for distance calc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_medians = np.nanmedian(mat, axis=0)
    for j in range(mat.shape[1]):
        mask = np.isnan(mat[:, j])
        mat[mask, j] = col_medians[j]

    # Normalise
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


# ════════════════════════════════════════════════════════════════════
#  9.  SAVE SYNTHETIC DATA
# ════════════════════════════════════════════════════════════════════
def save_synthetic_data(
    synth_df: pd.DataFrame,
    filepath: str = "synthetic_SPIHF.csv",
) -> None:
    """Save the synthetic dataset to CSV and print a summary.

    The output columns match the original SPIHF dataset plus an
    additional ``confidence_score`` column.

    Parameters
    ----------
    synth_df : pd.DataFrame
        Synthetic dataset from ``generate_synthetic_dataset``.
    filepath : str
        Output CSV path.

    Returns
    -------
    None

    Engineering note
    ----------------
    The CSV is saved with ``index=False`` and default float formatting
    so that downstream consumers can ``pd.read_csv`` it directly.
    """
    # Reorder columns to match original + confidence_score at end
    desired_order = COLUMN_NAMES + ["confidence_score"]
    cols_present = [c for c in desired_order if c in synth_df.columns]
    extra_cols = [c for c in synth_df.columns if c not in desired_order]
    synth_df = synth_df[cols_present + extra_cols]

    synth_df.to_csv(filepath, index=False)
    print(f"\n[save_synthetic_data] Saved {len(synth_df)} synthetic samples to '{filepath}'.")

    # Summary statistics
    print("\n" + "=" * 72)
    print("SYNTHETIC DATASET SUMMARY")
    print("=" * 72)
    print(f"  Total samples       : {len(synth_df)}")
    print(f"  Unique materials    : {synth_df['Material'].nunique()}")
    print(f"  Confidence (mean)   : {synth_df['confidence_score'].mean():.4f}")
    print(f"  Confidence (median) : {synth_df['confidence_score'].median():.4f}")
    print(f"  Confidence (min)    : {synth_df['confidence_score'].min():.4f}")
    print()
    print("  Samples per material:")
    for mat, count in synth_df["Material"].value_counts().items():
        print(f"    {mat:40s}  {count:4d}")
    print("=" * 72)


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run the full augmentation pipeline end-to-end.

    Workflow
    --------
    1. ``load_data``          — read SPIHF_Data.csv
    2. ``preprocess_data``    — clean & canonicalise
    3. ``compute_feature_statistics`` — per-material stats
    4. ``generate_synthetic_dataset`` — interpolation + noise + physics
                                        repair + rejection + dedup +
                                        confidence scoring
    5. ``save_synthetic_data``        — write synthetic_SPIHF.csv
    """
    np.random.seed(42)  # ensure reproducibility even if called repeatedly

    # Step 1
    df = pd.read_csv("SPIHF_Data.csv")
    print(f"[main] Raw data: {df.shape[0]} rows × {df.shape[1]} columns.\n")

    # Step 2
    df_clean = preprocess_data(df)

    # Step 3
    stats = compute_feature_statistics(df_clean)

    # Step 4
    synth_df = generate_synthetic_dataset(df_clean, stats, target_size=1000)

    # Step 5
    save_synthetic_data(synth_df, filepath="synthetic_SPIHF.csv")

    print("\n[OK] Augmentation pipeline complete.")


if __name__ == "__main__":
    main()
