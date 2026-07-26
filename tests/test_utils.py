"""
test_utils.py
=============
Unit tests for spihf_synthetic.utils.

Covers material alias mapping, near-duplicate removal, formatting
helpers, JSON sanitisation, and column harmonisation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spihf_synthetic.utils import (
    build_material_map,
    detect_outliers_iqr,
    fmt_float,
    fmt_pct,
    remove_near_duplicates,
    sanitise_for_json,
    short_label,
)


# ─── Material alias mapping ───────────────────────────────────────────

class TestBuildMaterialMap:
    """Tests for ``build_material_map``."""

    def test_returns_non_empty_dict(self) -> None:
        """Map should contain entries."""
        mat_map = build_material_map()
        assert isinstance(mat_map, dict)
        assert len(mat_map) > 0

    def test_known_aliases_resolve(self) -> None:
        """Known aliases should map to their canonical name."""
        mat_map = build_material_map()
        assert mat_map.get("Al 1050") == "AA1050"
        assert mat_map.get("DC01 Steel") == "DC01"
        assert mat_map.get("Al 6061") == "AA6061-T6"

    def test_canonical_maps_to_self(self) -> None:
        """Canonical names should map to themselves."""
        mat_map = build_material_map()
        assert mat_map.get("AA7075-O") == "AA7075-O"
        assert mat_map.get("DC01") == "DC01"


# ─── Near-duplicate removal ───────────────────────────────────────────

class TestRemoveNearDuplicates:
    """Tests for ``remove_near_duplicates``."""

    def test_removes_identical_duplicates(self) -> None:
        """Identical samples should be deduplicated to one."""
        cols = ["a", "b"]
        s = pd.Series({"a": 1.0, "b": 2.0})
        samples = [s.copy() for _ in range(5)]
        result = remove_near_duplicates(samples, cols, threshold=0.005)
        assert len(result) == 1

    def test_preserves_distinct_samples(self) -> None:
        """Sufficiently different samples should all be kept."""
        cols = ["a", "b"]
        samples = [
            pd.Series({"a": 0.0, "b": 0.0}),
            pd.Series({"a": 100.0, "b": 100.0}),
            pd.Series({"a": 50.0, "b": 50.0}),
        ]
        result = remove_near_duplicates(samples, cols, threshold=0.005)
        assert len(result) == 3

    def test_single_sample_returned(self) -> None:
        """A single-element list should be returned as-is."""
        s = pd.Series({"a": 1.0, "b": 2.0})
        result = remove_near_duplicates([s], ["a", "b"])
        assert len(result) == 1

    def test_empty_list_returned(self) -> None:
        """Empty input should return empty."""
        result = remove_near_duplicates([], ["a"])
        assert len(result) == 0


# ─── Formatting helpers ───────────────────────────────────────────────

class TestFormatting:
    """Tests for ``fmt_float``, ``fmt_pct``, ``short_label``."""

    def test_fmt_float_normal(self) -> None:
        assert fmt_float(3.14159, 2) == "3.14"

    def test_fmt_float_none(self) -> None:
        assert fmt_float(None, 4) == "N/A"

    def test_fmt_float_nan(self) -> None:
        assert fmt_float(float("nan"), 4) == "N/A"

    def test_fmt_pct(self) -> None:
        assert fmt_pct(85.123) == "85.12%"

    def test_short_label_known(self) -> None:
        assert short_label("UTS (MPa)") == "UTS"
        assert short_label("Thickness (mm)") == "Thickness"

    def test_short_label_unknown(self) -> None:
        """Unknown column names should be truncated to 20 chars."""
        long_name = "A" * 50
        assert len(short_label(long_name)) == 20


# ─── Outlier detection ────────────────────────────────────────────────

class TestDetectOutliersIqr:
    """Tests for ``detect_outliers_iqr``."""

    def test_no_outliers_in_uniform_data(self) -> None:
        """Uniformly distributed data should have few or no outliers."""
        s = pd.Series(np.arange(100, dtype=float))
        assert detect_outliers_iqr(s) == 0

    def test_detects_extreme_outlier(self) -> None:
        """A very extreme value should be flagged."""
        data = list(range(100))
        data.append(10_000)  # extreme outlier
        s = pd.Series(data, dtype=float)
        assert detect_outliers_iqr(s) >= 1

    def test_small_series_returns_zero(self) -> None:
        """Series with < 4 values should return 0 (can't compute IQR)."""
        s = pd.Series([1.0, 2.0, 3.0])
        assert detect_outliers_iqr(s) == 0


# ─── JSON sanitisation ────────────────────────────────────────────────

class TestSanitiseForJson:
    """Tests for ``sanitise_for_json``."""

    def test_numpy_int_to_int(self) -> None:
        assert isinstance(sanitise_for_json(np.int64(42)), int)

    def test_numpy_float_to_float(self) -> None:
        assert isinstance(sanitise_for_json(np.float64(3.14)), float)

    def test_nan_to_none(self) -> None:
        assert sanitise_for_json(float("nan")) is None

    def test_inf_to_none(self) -> None:
        assert sanitise_for_json(float("inf")) is None

    def test_nested_dict(self) -> None:
        obj = {"a": np.int64(1), "b": {"c": np.float64(2.5)}}
        result = sanitise_for_json(obj)
        assert result == {"a": 1, "b": {"c": 2.5}}

    def test_numpy_array_to_list(self) -> None:
        arr = np.array([1, 2, 3])
        result = sanitise_for_json(arr)
        assert result == [1, 2, 3]
