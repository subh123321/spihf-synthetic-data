"""
test_constraints.py
====================
Unit tests for spihf_synthetic.constraints.

Verifies that every constraint function correctly repairs physically
impossible synthetic samples while leaving valid samples untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spihf_synthetic.constraints import (
    check_formability_constraints,
    check_geometry_constraints,
    check_process_constraints,
    check_strength_constraints,
    repair_sample,
)


# ─── Strength constraints ──────────────────────────────────────────────

class TestStrengthConstraints:
    """Tests for ``check_strength_constraints``."""

    def test_swaps_when_uts_less_than_ys(self, sample_series: pd.Series) -> None:
        """If UTS < YS, they should be swapped."""
        s = sample_series.copy()
        s["UTS (MPa)"] = 100.0
        s["YS (MPa)"] = 200.0
        result = check_strength_constraints(s)
        assert result["UTS (MPa)"] >= result["YS (MPa)"]

    def test_uts_ys_correct_order_unchanged(self, sample_series: pd.Series) -> None:
        """If UTS >= YS already, values should remain the same."""
        s = sample_series.copy()
        s["UTS (MPa)"] = 400.0
        s["YS (MPa)"] = 250.0
        result = check_strength_constraints(s)
        assert result["UTS (MPa)"] == 400.0
        assert result["YS (MPa)"] == 250.0

    def test_hollomon_k_corrected_when_less_than_uts(
        self, sample_series: pd.Series
    ) -> None:
        """Strength coefficient k should be >= UTS."""
        s = sample_series.copy()
        s["UTS (MPa)"] = 500.0
        s["Strength Coefficient (k in MPa)"] = 300.0
        result = check_strength_constraints(s)
        assert result["Strength Coefficient (k in MPa)"] >= 500.0

    def test_hardening_exponent_clipped(self, sample_series: pd.Series) -> None:
        """Strain hardening exponent n must be in [0.01, 1.0]."""
        s = sample_series.copy()
        s["Strain hardening coefficient (n)"] = -0.5
        result = check_strength_constraints(s)
        assert result["Strain hardening coefficient (n)"] >= 0.01

        s2 = sample_series.copy()
        s2["Strain hardening coefficient (n)"] = 2.5
        result2 = check_strength_constraints(s2)
        assert result2["Strain hardening coefficient (n)"] <= 1.0

    def test_positive_strengths(self, sample_series: pd.Series) -> None:
        """UTS and YS must be > 0."""
        s = sample_series.copy()
        s["UTS (MPa)"] = -10.0
        s["YS (MPa)"] = -5.0
        result = check_strength_constraints(s)
        assert result["UTS (MPa)"] >= 1.0
        assert result["YS (MPa)"] >= 1.0


# ─── Geometry constraints ──────────────────────────────────────────────

class TestGeometryConstraints:
    """Tests for ``check_geometry_constraints``."""

    def test_thickness_minimum(self, sample_series: pd.Series) -> None:
        """Thickness must be >= 0.05 mm."""
        s = sample_series.copy()
        s["Thickness (mm)"] = 0.001
        result = check_geometry_constraints(s)
        assert result["Thickness (mm)"] >= 0.05

    def test_her_minimum(self, sample_series: pd.Series) -> None:
        """HER must be >= 1.0."""
        s = sample_series.copy()
        s["HER"] = 0.5
        result = check_geometry_constraints(s)
        assert result["HER"] >= 1.0

    def test_angle_clipped(self, sample_series: pd.Series) -> None:
        """Final angle must be in [1, 180] degrees."""
        s = sample_series.copy()
        s["Final angle after the final stage (degrees)"] = 200.0
        result = check_geometry_constraints(s)
        assert result["Final angle after the final stage (degrees)"] <= 180.0

        s2 = sample_series.copy()
        s2["Final angle after the final stage (degrees)"] = -5.0
        result2 = check_geometry_constraints(s2)
        assert result2["Final angle after the final stage (degrees)"] >= 1.0

    def test_min_thickness_leq_initial(self, sample_series: pd.Series) -> None:
        """Minimum thickness after forming cannot exceed initial thickness."""
        s = sample_series.copy()
        s["Thickness (mm)"] = 1.0
        s["Minimum thickness (after final stage, mm)"] = 2.0
        s["Final angle after the final stage (degrees)"] = 90.0
        result = check_geometry_constraints(s)
        assert (
            result["Minimum thickness (after final stage, mm)"]
            <= result["Thickness (mm)"]
        )


# ─── Process constraints ──────────────────────────────────────────────

class TestProcessConstraints:
    """Tests for ``check_process_constraints``."""

    def test_feed_rate_minimum(self, sample_series: pd.Series) -> None:
        """Feed rate must be >= 1.0 mm/min."""
        s = sample_series.copy()
        s["Feed rate (mm/min)"] = -100.0
        result = check_process_constraints(s)
        assert result["Feed rate (mm/min)"] >= 1.0

    def test_tool_speed_non_negative(self, sample_series: pd.Series) -> None:
        """Tool speed must be >= 0 rpm."""
        s = sample_series.copy()
        s["Tool speed (rpm)"] = -500.0
        result = check_process_constraints(s)
        assert result["Tool speed (rpm)"] >= 0.0

    def test_stages_integer_and_positive(self, sample_series: pd.Series) -> None:
        """Number of stages must be >= 1 and integer."""
        s = sample_series.copy()
        s["No of stages"] = 2.7
        result = check_process_constraints(s)
        assert result["No of stages"] == 3
        assert isinstance(result["No of stages"], (int, np.integer))

    def test_stages_minimum_one(self, sample_series: pd.Series) -> None:
        """Number of stages must be >= 1 even for near-zero input."""
        s = sample_series.copy()
        s["No of stages"] = 0.1
        result = check_process_constraints(s)
        assert result["No of stages"] >= 1

    def test_lubricant_binary(self, sample_series: pd.Series) -> None:
        """Lubricant flag must be rounded to {0, 1}."""
        s = sample_series.copy()
        s["Is lubricant used?"] = 0.7
        result = check_process_constraints(s)
        assert result["Is lubricant used?"] in (0, 1)


# ─── Formability constraints ──────────────────────────────────────────

class TestFormabilityConstraints:
    """Tests for ``check_formability_constraints``."""

    def test_r_value_clipped(self, sample_series: pd.Series) -> None:
        """Lankford R-value must be in [0.1, 5.0]."""
        s = sample_series.copy()
        s["Anisotropic (R Value)"] = -1.0
        result = check_formability_constraints(s)
        assert result["Anisotropic (R Value)"] >= 0.1

        s2 = sample_series.copy()
        s2["Anisotropic (R Value)"] = 10.0
        result2 = check_formability_constraints(s2)
        assert result2["Anisotropic (R Value)"] <= 5.0

    def test_elongation_clipped(self, sample_series: pd.Series) -> None:
        """Elongation must be in [0.5, 99.0]%."""
        s = sample_series.copy()
        s["Total Strain/Elongation (%)"] = -5.0
        result = check_formability_constraints(s)
        assert result["Total Strain/Elongation (%)"] >= 0.5


# ─── Combined repair ──────────────────────────────────────────────────

class TestRepairSample:
    """Tests for the combined ``repair_sample`` function."""

    def test_valid_sample_unchanged(self, sample_series: pd.Series) -> None:
        """A physically valid sample should pass through repair with minimal change."""
        result = repair_sample(sample_series)
        # Core values should be very close (floating-point clip is OK)
        assert abs(result["UTS (MPa)"] - sample_series["UTS (MPa)"]) < 1e-6

    def test_multiple_violations_all_fixed(self, sample_series: pd.Series) -> None:
        """A sample with multiple violations should have ALL of them repaired."""
        s = sample_series.copy()
        s["UTS (MPa)"] = 100.0
        s["YS (MPa)"] = 300.0       # UTS < YS
        s["HER"] = 0.3              # HER < 1.0
        s["No of stages"] = 0.2     # Stages < 1
        s["Feed rate (mm/min)"] = -50  # Negative feed rate

        result = repair_sample(s)
        assert result["UTS (MPa)"] >= result["YS (MPa)"]
        assert result["HER"] >= 1.0
        assert result["No of stages"] >= 1
        assert result["Feed rate (mm/min)"] >= 1.0

    def test_repair_returns_copy(self, sample_series: pd.Series) -> None:
        """repair_sample should not mutate the original Series."""
        original = sample_series.copy()
        original["UTS (MPa)"] = 50.0
        original["YS (MPa)"] = 200.0
        original_uts = original["UTS (MPa)"]

        _ = repair_sample(original)
        # Original should be unchanged
        assert original["UTS (MPa)"] == original_uts
