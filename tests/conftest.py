"""
conftest.py
===========
Shared pytest fixtures for the SPIHF synthetic-data test suite.

Provides small, deterministic DataFrames and Series that mirror the
structure of the real SPIHF dataset without requiring the CSV file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_real_df() -> pd.DataFrame:
    """A minimal realistic SPIHF DataFrame (10 rows, 2 materials).

    Covers two material groups with enough rows for interpolation,
    near-duplicate detection, and per-material statistics.
    """
    np.random.seed(42)
    data = {
        "Material": (["DC01"] * 5) + (["AA6061-T6"] * 5),
        "Thickness (mm)": [1.0, 1.2, 0.8, 1.1, 0.9, 1.6, 1.5, 1.4, 1.7, 1.6],
        "Precut dimensions (diameter/side length) mm": [
            60, 65, 55, 62, 58, 80, 75, 70, 85, 82,
        ],
        "Precut Shape (circle/square/etc)": ["circle"] * 10,
        "Total Strain/Elongation (%)": [28, 30, 25, 27, 29, 17, 18, 16, 19, 17],
        "UTS (MPa)": [310, 320, 305, 315, 312, 310, 315, 308, 320, 312],
        "YS (MPa)": [180, 190, 175, 185, 182, 276, 280, 270, 285, 278],
        "Strength Coefficient (k in MPa)": [
            530, 540, 520, 535, 528, 400, 410, 395, 415, 405,
        ],
        "Strain hardening coefficient (n)": [
            0.22, 0.24, 0.20, 0.23, 0.21, 0.17, 0.18, 0.16, 0.19, 0.17,
        ],
        "Anisotropic (R Value)": [1.6, 1.7, 1.5, 1.65, 1.55, 0.85, 0.9, 0.8, 0.95, 0.87],
        "Is lubricant used?": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        "Feed rate (mm/min)": [
            1000, 1200, 800, 1100, 900, 1500, 1400, 1300, 1600, 1500,
        ],
        "Tool speed (rpm)": [0, 500, 0, 200, 100, 1000, 0, 500, 800, 1000],
        "Step depth (mm)": [0.5, 0.3, 0.4, 0.5, 0.3, 0.2, 0.3, 0.2, 0.25, 0.2],
        "No of stages": [1, 2, 1, 3, 1, 1, 2, 1, 1, 2],
        "HER": [1.33, 1.45, 1.28, 1.50, 1.35, 1.46, 1.55, 1.40, 1.60, 1.48],
        "Flange Height (mm)": [15, 18, 12, 20, 14, 23, 25, 21, 27, 24],
        "Roughness (µm)": [1.2, 0.9, 1.5, 1.0, 1.3, 0.8, 0.7, 0.9, 0.6, 0.8],
        "Minimum thickness (after final stage, mm)": [
            0.7, 0.8, 0.5, 0.75, 0.6, 1.1, 1.0, 0.9, 1.2, 1.1,
        ],
        "Final angle after the final stage (degrees)": [
            90, 85, 88, 90, 87, 90, 90, 88, 90, 90,
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def sample_dc01_df(sample_real_df: pd.DataFrame) -> pd.DataFrame:
    """Just the DC01 material rows from the fixture."""
    return sample_real_df[sample_real_df["Material"] == "DC01"].reset_index(drop=True)


@pytest.fixture()
def sample_series(sample_real_df: pd.DataFrame) -> pd.Series:
    """A single DC01 row as a pd.Series."""
    return sample_real_df.iloc[0].copy()


@pytest.fixture()
def numeric_cols() -> list[str]:
    """The numeric columns used throughout the package."""
    from spihf_synthetic.config import NUMERIC_FEATURES
    return NUMERIC_FEATURES
