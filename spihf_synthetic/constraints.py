"""
constraints.py
==============
Engineering constraint checks and sample repair for SPIHF synthetic data.

All physics-informed bounds (metallurgical, geometric, process) are
centralised here so that both the augmentation pipeline and downstream
consumers can validate samples against the same rule set.

Functions
---------
check_strength_constraints    UTS ≥ YS, Hollomon consistency, hardening limits.
check_geometry_constraints    Thickness, HER, flange height, roughness, angle.
check_formability_constraints Elongation bounds, R-value range.
check_process_constraints     Feed rate, tool speed, step depth, stages.
repair_sample                 Apply all soft corrections to a single sample.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from spihf_synthetic.config import (
    ANGLE_LIMITS,
    GEOMETRY_LIMITS,
    MATERIAL_PROPERTY_LIMITS,
    PROCESS_LIMITS,
)


# ═══════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _get(sample: pd.Series, col: str) -> Optional[float]:
    """Retrieve a value from *sample*, returning ``None`` if NaN."""
    v = sample.get(col, np.nan)
    return v if pd.notna(v) else None


def _set(sample: pd.Series, col: str, val: float) -> None:
    """Set *col* in *sample* if the column exists."""
    if col in sample.index:
        sample[col] = val


# ═══════════════════════════════════════════════════════════════════════
#  STRENGTH CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

def check_strength_constraints(sample: pd.Series) -> pd.Series:
    """Enforce metallurgical strength constraints.

    Corrections applied (in order):

    - UTS ≥ YS (by definition of ultimate vs. yield strength).
    - Strength coefficient k ≥ UTS (Hollomon σ = kεⁿ; at necking
      σ = UTS, ε < 1 ⟹ k ≥ UTS for n > 0).
    - 0.01 ≤ n ≤ 1.0 (strain-hardening exponent physical range).
    - UTS > 0, YS > 0.

    Parameters
    ----------
    sample : pd.Series
        A single synthetic sample.

    Returns
    -------
    pd.Series
        Corrected sample (mutated in-place and returned).
    """
    uts = _get(sample, "UTS (MPa)")
    ys = _get(sample, "YS (MPa)")
    k = _get(sample, "Strength Coefficient (k in MPa)")
    n = _get(sample, "Strain hardening coefficient (n)")

    # UTS ≥ YS
    if uts is not None and ys is not None and uts < ys:
        _set(sample, "UTS (MPa)", ys)
        _set(sample, "YS (MPa)", uts)
        uts, ys = ys, uts

    # Hollomon consistency: k ≥ UTS
    if k is not None:
        if uts is not None and k < uts:
            _set(sample, "Strength Coefficient (k in MPa)", uts * 1.05)
        if k <= 0:
            _set(sample, "Strength Coefficient (k in MPa)", abs(k) + 1.0)

    # Hardening exponent bounds
    n_lo, n_hi = MATERIAL_PROPERTY_LIMITS["Strain hardening coefficient (n)"]
    if n is not None:
        _set(sample, "Strain hardening coefficient (n)",
             float(np.clip(n, n_lo, n_hi)))

    # Positive strengths
    if uts is not None:
        _set(sample, "UTS (MPa)", max(uts, 1.0))
    if ys is not None:
        _set(sample, "YS (MPa)", max(ys, 1.0))

    return sample


# ═══════════════════════════════════════════════════════════════════════
#  GEOMETRY CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

def check_geometry_constraints(sample: pd.Series) -> pd.Series:
    """Enforce geometric and forming-output constraints.

    Corrections applied:

    - Thickness > 0.05 mm.
    - Precut dimension > 1.0 mm.
    - HER ≥ 1.0.
    - Flange height > 0.1 mm.
    - Roughness ≥ 0.001 µm.
    - Final angle ∈ [1°, 180°].
    - Sine-law thinning bound: ``t_min ≥ t₀·sin(α)·0.5``
      (allows ~30 % extra thinning beyond sine law).
    - Minimum thickness ≤ initial thickness.

    Parameters
    ----------
    sample : pd.Series
        A single synthetic sample.

    Returns
    -------
    pd.Series
        Corrected sample.
    """
    t0 = _get(sample, "Thickness (mm)")
    precut = _get(sample, "Precut dimensions (diameter/side length) mm")
    her = _get(sample, "HER")
    fh = _get(sample, "Flange Height (mm)")
    angle = _get(sample, "Final angle after the final stage (degrees)")
    tmin = _get(sample, "Minimum thickness (after final stage, mm)")

    # Check roughness under both possible column names
    rough_col = "Roughness (µm)"
    rough = _get(sample, rough_col)
    if rough is None:
        rough_col = "Roughness (um)"
        rough = _get(sample, rough_col)

    if t0 is not None:
        t0 = max(t0, GEOMETRY_LIMITS["Thickness (mm)"])
        _set(sample, "Thickness (mm)", t0)

    if precut is not None:
        _set(sample, "Precut dimensions (diameter/side length) mm",
             max(precut, GEOMETRY_LIMITS["Precut dimensions (diameter/side length) mm"]))

    if her is not None:
        _set(sample, "HER", max(her, GEOMETRY_LIMITS["HER"]))

    if fh is not None:
        _set(sample, "Flange Height (mm)",
             max(fh, GEOMETRY_LIMITS["Flange Height (mm)"]))

    if rough is not None:
        _set(sample, rough_col, max(rough, 0.001))

    a_lo, a_hi = ANGLE_LIMITS
    if angle is not None:
        angle = float(np.clip(angle, a_lo, a_hi))
        _set(sample, "Final angle after the final stage (degrees)", angle)

    if tmin is not None:
        tmin = max(tmin, GEOMETRY_LIMITS["Minimum thickness (after final stage, mm)"])
        # Sine-law thinning bound
        if t0 is not None and angle is not None:
            angle_rad = np.radians(angle)
            sine_limit = t0 * np.sin(angle_rad)
            if tmin < sine_limit * 0.5:
                tmin = sine_limit * 0.7
        # Cannot exceed original thickness
        if t0 is not None:
            tmin = min(tmin, t0)
        _set(sample, "Minimum thickness (after final stage, mm)", tmin)

    return sample


# ═══════════════════════════════════════════════════════════════════════
#  FORMABILITY CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

def check_formability_constraints(sample: pd.Series) -> pd.Series:
    """Enforce material-formability constraints.

    Corrections applied:

    - R-value (Lankford) ∈ [0.1, 5.0].
    - Total elongation ∈ [0.5, 99.0] %.

    Parameters
    ----------
    sample : pd.Series
        A single synthetic sample.

    Returns
    -------
    pd.Series
        Corrected sample.
    """
    r = _get(sample, "Anisotropic (R Value)")
    elong = _get(sample, "Total Strain/Elongation (%)")

    r_lo, r_hi = MATERIAL_PROPERTY_LIMITS["Anisotropic (R Value)"]
    if r is not None:
        _set(sample, "Anisotropic (R Value)", float(np.clip(r, r_lo, r_hi)))

    e_lo, e_hi = MATERIAL_PROPERTY_LIMITS["Total Strain/Elongation (%)"]
    if elong is not None:
        _set(sample, "Total Strain/Elongation (%)",
             float(np.clip(elong, e_lo, e_hi)))

    return sample


# ═══════════════════════════════════════════════════════════════════════
#  PROCESS CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

def check_process_constraints(sample: pd.Series) -> pd.Series:
    """Enforce process-parameter constraints.

    Corrections applied:

    - Feed rate ≥ 1.0 mm/min.
    - Tool speed ≥ 0 rpm.
    - Step depth ≥ 0.01 mm.
    - Number of stages ≥ 1 (integer).
    - Lubricant flag rounded to {0, 1}.

    Parameters
    ----------
    sample : pd.Series
        A single synthetic sample.

    Returns
    -------
    pd.Series
        Corrected sample.
    """
    feed = _get(sample, "Feed rate (mm/min)")
    speed = _get(sample, "Tool speed (rpm)")
    step = _get(sample, "Step depth (mm)")
    stages = _get(sample, "No of stages")
    lub = _get(sample, "Is lubricant used?")

    if feed is not None:
        lo, _ = PROCESS_LIMITS["Feed rate (mm/min)"]
        _set(sample, "Feed rate (mm/min)", max(feed, lo))

    if speed is not None:
        lo, _ = PROCESS_LIMITS["Tool speed (rpm)"]
        _set(sample, "Tool speed (rpm)", max(speed, lo))

    if step is not None:
        lo, _ = PROCESS_LIMITS["Step depth (mm)"]
        _set(sample, "Step depth (mm)", max(step, lo))

    if stages is not None:
        _set(sample, "No of stages", max(int(round(stages)), 1))

    if lub is not None:
        _set(sample, "Is lubricant used?", int(round(np.clip(lub, 0, 1))))

    return sample


# ═══════════════════════════════════════════════════════════════════════
#  COMBINED REPAIR
# ═══════════════════════════════════════════════════════════════════════

def repair_sample(sample: pd.Series) -> pd.Series:
    """Apply all physics-informed constraint checks to a single sample.

    This is the single entry point that the augmentation pipeline calls
    after interpolation and noise injection.  It composes:

    1. ``check_strength_constraints``
    2. ``check_formability_constraints``
    3. ``check_process_constraints``
    4. ``check_geometry_constraints``

    Parameters
    ----------
    sample : pd.Series
        A single synthetic sample (post-interpolation + noise).

    Returns
    -------
    pd.Series
        Physics-corrected sample.

    Notes
    -----
    These corrections act as a *projection onto the feasible manifold*
    in parameter space.  They are intentionally conservative (wide
    bounds) so that the synthetic dataset retains the statistical
    diversity introduced by interpolation and noise while remaining
    physically realisable.
    """
    s = sample.copy()
    s = check_strength_constraints(s)
    s = check_formability_constraints(s)
    s = check_process_constraints(s)
    s = check_geometry_constraints(s)
    return s
