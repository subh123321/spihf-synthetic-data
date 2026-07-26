"""
test_confidence.py
==================
Unit tests for spihf_synthetic.confidence.

Verifies that the three confidence components (range, distance,
correlation) and their weighted combination behave correctly under
normal and edge-case inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spihf_synthetic.confidence import (
    compute_distribution_score,
    compute_mahalanobis_score,
    compute_physics_score,
    compute_total_confidence,
)


# ─── Range score ───────────────────────────────────────────────────────

class TestMahalanobisScore:
    """Tests for ``compute_mahalanobis_score`` (range-based scoring)."""

    def test_real_sample_scores_high(
        self,
        sample_dc01_df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> None:
        """A sample drawn from the real data should score near 1.0."""
        sample = sample_dc01_df.iloc[0]
        score = compute_mahalanobis_score(
            sample, sample_dc01_df, numeric_cols
        )
        assert score >= 0.8, f"Real sample scored only {score:.2f}"

    def test_extreme_sample_scores_low(
        self,
        sample_dc01_df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> None:
        """A sample with extreme values should score low."""
        extreme = sample_dc01_df.iloc[0].copy()
        for col in numeric_cols:
            if col in extreme.index and pd.notna(extreme[col]):
                extreme[col] = extreme[col] * 100  # 100x out of range
        score = compute_mahalanobis_score(
            extreme, sample_dc01_df, numeric_cols
        )
        assert score < 0.5, f"Extreme sample scored {score:.2f}"

    def test_empty_material_returns_zero(self, numeric_cols: list[str]) -> None:
        """If material_data is empty, score should be 0.0."""
        sample = pd.Series({"UTS (MPa)": 300.0, "YS (MPa)": 200.0})
        empty_df = pd.DataFrame()
        score = compute_mahalanobis_score(sample, empty_df, numeric_cols)
        assert score == 0.0


# ─── Distance score ───────────────────────────────────────────────────

class TestPhysicsScore:
    """Tests for ``compute_physics_score`` (nearest-neighbour distance)."""

    def test_identical_sample_scores_one(
        self,
        sample_dc01_df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> None:
        """A sample identical to a real row should score exp(0) = 1.0."""
        sample = sample_dc01_df.iloc[0]
        score = compute_physics_score(sample, sample_dc01_df, numeric_cols)
        assert score > 0.95, f"Identical sample scored only {score:.2f}"

    def test_far_sample_scores_lower(
        self,
        sample_dc01_df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> None:
        """A sample far from all real data should score lower."""
        far = sample_dc01_df.iloc[0].copy()
        for col in numeric_cols:
            if col in far.index and pd.notna(far[col]):
                far[col] = far[col] * 50
        score_far = compute_physics_score(far, sample_dc01_df, numeric_cols)
        score_close = compute_physics_score(
            sample_dc01_df.iloc[0], sample_dc01_df, numeric_cols
        )
        assert score_far < score_close


# ─── Distribution / correlation score ─────────────────────────────────

class TestDistributionScore:
    """Tests for ``compute_distribution_score``."""

    def test_returns_float_in_range(
        self,
        sample_dc01_df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> None:
        """Score should always be in [0, 1]."""
        sample = sample_dc01_df.iloc[0]
        score = compute_distribution_score(
            sample, sample_dc01_df, numeric_cols
        )
        assert 0.0 <= score <= 1.0

    def test_insufficient_data_returns_half(
        self, numeric_cols: list[str]
    ) -> None:
        """With < 3 rows, should return 0.5 (neutral)."""
        tiny_df = pd.DataFrame({"UTS (MPa)": [300], "YS (MPa)": [200]})
        sample = tiny_df.iloc[0]
        score = compute_distribution_score(sample, tiny_df, numeric_cols)
        assert score == 0.5


# ─── Total confidence ─────────────────────────────────────────────────

class TestTotalConfidence:
    """Tests for ``compute_total_confidence``."""

    def test_score_bounded(
        self,
        sample_dc01_df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> None:
        """Total score must be in [0, 1]."""
        sample = sample_dc01_df.iloc[0]
        score = compute_total_confidence(
            sample, sample_dc01_df, numeric_cols
        )
        assert 0.0 <= score <= 1.0

    def test_real_sample_above_threshold(
        self,
        sample_dc01_df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> None:
        """A real sample should pass the default rejection threshold (0.25)."""
        sample = sample_dc01_df.iloc[0]
        score = compute_total_confidence(
            sample, sample_dc01_df, numeric_cols
        )
        assert score > 0.25, f"Real sample scored {score:.2f}, below threshold"

    def test_weights_sum_to_one_by_default(self) -> None:
        """Default weights should sum to 1.0."""
        from inspect import signature
        sig = signature(compute_total_confidence)
        w_range = sig.parameters["weight_range"].default
        w_dist = sig.parameters["weight_distance"].default
        w_corr = sig.parameters["weight_correlation"].default
        assert abs((w_range + w_dist + w_corr) - 1.0) < 1e-10
