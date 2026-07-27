"""
config.py
=========
Central configuration for the ``spihf_synthetic`` package.

All global constants, column definitions, physics limits, and tunable
hyper-parameters are defined here.  Every other module imports from
this file — nothing is hard-coded elsewhere.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════

RANDOM_SEED: int = 42
"""Master random seed for full reproducibility."""

# ═══════════════════════════════════════════════════════════════════════
#  AUGMENTATION HYPER-PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

NOISE_LEVEL: Tuple[float, float] = (0.005,0.015)
"""(low, high) fraction of within-group std used for Gaussian noise."""

CONFIDENCE_THRESHOLD: float = 0.40
"""Minimum confidence score for rejection sampling."""

NUM_SYNTHETIC_SAMPLES: int = 1000
"""Target number of synthetic samples to generate."""

MIN_SAMPLES_FOR_INTERPOLATION: int = 2
"""Minimum real samples in a material group to enable SMOTE interpolation."""

MAX_ATTEMPTS_MULTIPLIER: int = 10
"""Maximum generation attempts = required × this multiplier."""

DUPLICATE_THRESHOLD: float = 0.005
"""Normalised L2 distance below which two synthetic samples are duplicates."""

ALPHA_RANGE: Tuple[float, float] = (0.3,0.7)
"""SMOTE interpolation coefficient α bounds."""

# ═══════════════════════════════════════════════════════════════════════
#  COLUMN DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

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
"""Canonical column names (order matches the raw CSV after renaming)."""

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
"""Numeric features used for interpolation, perturbation, and validation."""

CATEGORICAL_FEATURES: List[str] = [
    "Material",
    "Precut Shape (circle/square/etc)",
]
"""Categorical columns carried through unchanged during interpolation."""

# Validation uses 'um' in column name (synthetic CSV encodes µ differently).
NUMERIC_FEATURES_VALIDATION: List[str] = [
    c.replace("µm", "um") for c in NUMERIC_FEATURES
]
"""Numeric features with ASCII roughness column name for validation/viz."""

SHORT_LABELS: Dict[str, str] = {
    "Thickness (mm)": "Thickness",
    "Precut dimensions (diameter/side length) mm": "Precut dim.",
    "Total Strain/Elongation (%)": "Elongation %",
    "UTS (MPa)": "UTS",
    "YS (MPa)": "YS",
    "Strength Coefficient (k in MPa)": "Strength k",
    "Strain hardening coefficient (n)": "n (hardening)",
    "Anisotropic (R Value)": "R-value",
    "Is lubricant used?": "Lubricant",
    "Feed rate (mm/min)": "Feed rate",
    "Tool speed (rpm)": "Tool speed",
    "Step depth (mm)": "Step depth",
    "No of stages": "Stages",
    "HER": "HER",
    "Flange Height (mm)": "Flange Height",
    "Roughness (um)": "Roughness",
    "Roughness (µm)": "Roughness",
    "Minimum thickness (after final stage, mm)": "Min thickness",
    "Final angle after the final stage (degrees)": "Final angle",
}
"""Short labels for axis readability in plots and tables."""

# ═══════════════════════════════════════════════════════════════════════
#  RAW-TO-CANONICAL COLUMN MAPPING
# ═══════════════════════════════════════════════════════════════════════

RAW_TO_CANONICAL: Dict[str, str] = {
    "Precut dimensions (diameter/ side length) mm":
        "Precut dimensions (diameter/side length) mm",
    "Precut Shape (circle/ square/etc)":
        "Precut Shape (circle/square/etc)",
    "Total Strain/ Elongation (%)":
        "Total Strain/Elongation (%)",
    "Step depth(mm)": "Step depth (mm)",
    "Final angle after the final stage, degrees":
        "Final angle after the final stage (degrees)",
}
"""Mapping from raw CSV column names to canonical names."""

# ═══════════════════════════════════════════════════════════════════════
#  PHYSICS LIMITS
# ═══════════════════════════════════════════════════════════════════════

PROCESS_LIMITS: Dict[str, Tuple[float, float]] = {
    "Feed rate (mm/min)":   (1.0, 50_000.0),
    "Tool speed (rpm)":     (0.0, 100_000.0),
    "Step depth (mm)":      (0.01, 20.0),
    "No of stages":         (1.0, 20.0),
}
"""Hard physical bounds for process parameters (min, max)."""

ANGLE_LIMITS: Tuple[float, float] = (1.0, 180.0)
"""Final wall-angle bounds in degrees."""

MATERIAL_PROPERTY_LIMITS: Dict[str, Tuple[float, float]] = {
    "UTS (MPa)":                          (1.0, 5_000.0),
    "YS (MPa)":                           (1.0, 5_000.0),
    "Strength Coefficient (k in MPa)":    (1.0, 10_000.0),
    "Strain hardening coefficient (n)":   (0.01, 1.0),
    "Anisotropic (R Value)":              (0.1, 5.0),
    "Total Strain/Elongation (%)":        (0.5, 99.0),
}
"""Physical bounds for material-property features."""

GEOMETRY_LIMITS: Dict[str, float] = {
    "Thickness (mm)":                                 0.05,
    "Precut dimensions (diameter/side length) mm":    1.0,
    "HER":                                            1.0,
    "Flange Height (mm)":                             0.1,
    "Roughness (µm)":                                 0.001,
    "Minimum thickness (after final stage, mm)":      0.01,
}
"""Minimum values for geometric / forming-output features."""

# ═══════════════════════════════════════════════════════════════════════
#  PHYSICS CORRELATION PAIRS  (for validation)
# ═══════════════════════════════════════════════════════════════════════

PHYSICS_CORRELATION_PAIRS: List[Tuple[str, str, int]] = [
    ("HER", "Flange Height (mm)", +1),
    ("Step depth (mm)", "Roughness (um)", +1),
    ("Step depth (mm)", "Minimum thickness (after final stage, mm)", -1),
    ("No of stages", "HER", +1),
    ("Is lubricant used?", "Roughness (um)", -1),
    ("Total Strain/Elongation (%)", "HER", +1),
    ("Anisotropic (R Value)", "HER", 0),
]
"""
Domain-specific correlation pairs from SPIHF process physics.

Each tuple is ``(feature_a, feature_b, expected_direction_sign)``.
+1 = expected positive correlation, -1 = expected negative, 0 = data-dependent.
"""

# ═══════════════════════════════════════════════════════════════════════
#  ENGINEERING RELATIONSHIP PAIRS  (for visualisation)
# ═══════════════════════════════════════════════════════════════════════

ENGINEERING_PAIRS: List[Tuple[str, str, str, str]] = [
    ("HER", "Flange Height (mm)",
     "Hole Expansion Ratio (HER)", "Flange Height (mm)"),
    ("Step depth (mm)", "Roughness (um)",
     "Step Depth (mm)", "Surface Roughness (µm)"),
    ("No of stages", "HER",
     "Number of Stages", "HER"),
    ("Thickness (mm)", "Minimum thickness (after final stage, mm)",
     "Initial Thickness (mm)", "Minimum Thickness (mm)"),
    ("Total Strain/Elongation (%)", "HER",
     "Total Elongation (%)", "HER"),
    ("Anisotropic (R Value)", "HER",
     "Lankford R-value", "HER"),
]
"""Engineering relationship pairs for scatter-plot visualisation."""

# ═══════════════════════════════════════════════════════════════════════
#  MATERIAL ALIAS GROUPS
# ═══════════════════════════════════════════════════════════════════════

MATERIAL_ALIAS_GROUPS: List[Tuple[str, List[str]]] = [
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
"""Material alias groups for canonicalisation during preprocessing."""

# ═══════════════════════════════════════════════════════════════════════
#  UNIT-STRING SUFFIXES FOR CLEANING
# ═══════════════════════════════════════════════════════════════════════

UNIT_SUFFIXES: List[str] = [
    " mm/min", " rpm clockwise", " rpm", " mm/cycle",
    " mm", " µm", " um", "°", "° ",
]
"""Embedded unit strings to strip from numeric cells during cleaning."""
