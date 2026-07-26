"""
test_augmentation.py
====================
Unit tests for spihf_synthetic.augmentation.

Covers interpolation, noise injection, and the core generation pipeline
at a unit level (small fixture data, fast execution).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spihf_synthetic.augmentation import (
    apply_gaussian_noise,
    compute_feature_statistics,
    generate_interpolated_sample,
    generate_synthetic_dataset,
    preprocess_data,
)
from spihf_synthetic.config import NUMERIC_FEATURES


# ─── Interpolation ─────────────────────────────────────────────────────

class TestInterpolation:
    """Tests for ``generate_interpolated_sample``."""

    def test_interpolated_values_between_parents(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """Interpolated numeric values must lie between the two parent rows."""
        np.random.seed(42)
        df = sample_real_df[sample_real_df["Material"] == "DC01"]
        row_i = df.iloc[0]
        row_j = df.iloc[1]

        result = generate_interpolated_sample(row_i, row_j)

        for feat in NUMERIC_FEATURES:
            if feat not in result.index:
                continue
            vi, vj = row_i[feat], row_j[feat]
            if pd.notna(vi) and pd.notna(vj):
                lo, hi = min(vi, vj), max(vi, vj)
                assert lo <= result[feat] <= hi, (
                    f"{feat}: {result[feat]} not in [{lo}, {hi}]"
                )

    def test_categorical_inherited_from_row_i(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """Categorical features (Material, Precut Shape) come from row_i."""
        df = sample_real_df[sample_real_df["Material"] == "DC01"]
        row_i = df.iloc[0]
        row_j = df.iloc[1]

        result = generate_interpolated_sample(row_i, row_j)
        assert result["Material"] == row_i["Material"]
        assert result["Precut Shape (circle/square/etc)"] == row_i[
            "Precut Shape (circle/square/etc)"
        ]

    def test_alpha_range_respected(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """When alpha_low == alpha_high, result is deterministic."""
        df = sample_real_df[sample_real_df["Material"] == "DC01"]
        row_i = df.iloc[0]
        row_j = df.iloc[1]

        # α = 1.0 → result should equal row_i for all numeric features
        result = generate_interpolated_sample(
            row_i, row_j, alpha_low=1.0, alpha_high=1.0
        )
        for feat in NUMERIC_FEATURES:
            if feat in result.index and pd.notna(row_i.get(feat)):
                assert abs(result[feat] - row_i[feat]) < 1e-10, (
                    f"{feat} not equal to row_i at alpha=1.0"
                )


# ─── Gaussian noise ────────────────────────────────────────────────────

class TestGaussianNoise:
    """Tests for ``apply_gaussian_noise``."""

    def test_noise_magnitude_small(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """Noise should be small relative to the feature values."""
        np.random.seed(42)
        df = sample_real_df[sample_real_df["Material"] == "DC01"]
        stats = compute_feature_statistics(df)
        mat_stats = stats["DC01"]
        sample = df.iloc[0].copy()

        noisy = apply_gaussian_noise(sample, mat_stats)

        for feat in NUMERIC_FEATURES:
            if feat in ("Is lubricant used?", "No of stages"):
                continue
            if feat not in sample.index or pd.isna(sample[feat]):
                continue
            original = sample[feat]
            perturbed = noisy[feat]
            if abs(original) > 1e-12:
                pct_change = abs(perturbed - original) / abs(original) * 100
                # Noise should be < 50% of value (very generous bound)
                assert pct_change < 50, (
                    f"{feat}: {pct_change:.1f}% change is too large"
                )

    def test_lubricant_and_stages_untouched(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """Lubricant flag and stages should NOT receive noise."""
        np.random.seed(42)
        df = sample_real_df[sample_real_df["Material"] == "DC01"]
        stats = compute_feature_statistics(df)
        mat_stats = stats["DC01"]
        sample = df.iloc[0].copy()

        noisy = apply_gaussian_noise(sample, mat_stats)
        assert noisy["Is lubricant used?"] == sample["Is lubricant used?"]
        assert noisy["No of stages"] == sample["No of stages"]


# ─── Feature statistics ───────────────────────────────────────────────

class TestFeatureStatistics:
    """Tests for ``compute_feature_statistics``."""

    def test_returns_dict_per_material(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """Should return one entry per unique material."""
        stats = compute_feature_statistics(sample_real_df)
        materials = sample_real_df["Material"].unique()
        for mat in materials:
            assert mat in stats
            assert "mean" in stats[mat]
            assert "std" in stats[mat]
            assert "min" in stats[mat]
            assert "max" in stats[mat]

    def test_stats_are_per_material(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """DC01 and AA6061-T6 should have different statistics."""
        stats = compute_feature_statistics(sample_real_df)
        dc01_uts_mean = stats["DC01"]["mean"]["UTS (MPa)"]
        aa6061_uts_mean = stats["AA6061-T6"]["mean"]["UTS (MPa)"]
        # These are different materials, means should differ
        assert dc01_uts_mean != aa6061_uts_mean


# ─── Full pipeline (small scale) ──────────────────────────────────────

class TestGenerateSyntheticDataset:
    """Tests for ``generate_synthetic_dataset`` at small scale."""

    def test_generates_requested_count(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """Pipeline should produce approximately the requested sample count."""
        np.random.seed(42)
        stats = compute_feature_statistics(sample_real_df)
        synth = generate_synthetic_dataset(
            sample_real_df, stats, target_size=20
        )
        # Should be close to 20 (exact depends on rejection sampling)
        assert len(synth) >= 10
        assert len(synth) <= 30

    def test_confidence_column_present(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """Synthetic data must include a confidence_score column."""
        np.random.seed(42)
        stats = compute_feature_statistics(sample_real_df)
        synth = generate_synthetic_dataset(
            sample_real_df, stats, target_size=10
        )
        assert "confidence_score" in synth.columns
        assert synth["confidence_score"].notna().all()

    def test_materials_preserved(
        self, sample_real_df: pd.DataFrame
    ) -> None:
        """Synthetic data should only contain materials from the real data."""
        np.random.seed(42)
        stats = compute_feature_statistics(sample_real_df)
        synth = generate_synthetic_dataset(
            sample_real_df, stats, target_size=20
        )
        real_mats = set(sample_real_df["Material"].unique())
        synth_mats = set(synth["Material"].unique())
        assert synth_mats.issubset(real_mats)
