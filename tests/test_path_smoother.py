"""
tests/test_path_smoother.py
---------------------------
Unit tests for trajectory_control.path_smoother.smooth_path

Run with:
    pytest tests/test_path_smoother.py -v
"""

import sys
import os
import math
import pytest
import numpy as np

# Make the package importable without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajectory_control.path_smoother import smooth_path


# ─── Output shape ────────────────────────────────────────────────────────────

class TestOutputShape:
    """smooth_path must always return an (N, 2) numpy array."""

    def test_default_num_points(self):
        result = smooth_path([(0, 0), (1, 1)])
        assert result.shape == (400, 2), (
            "Default num_points should be 400"
        )

    def test_custom_num_points(self):
        for n in [10, 50, 200, 500]:
            result = smooth_path([(0, 0), (1, 0), (2, 1)], num_points=n)
            assert result.shape == (n, 2), (
                f"Expected shape ({n}, 2), got {result.shape}"
            )

    def test_returns_numpy_array(self):
        result = smooth_path([(0, 0), (1, 1)])
        assert isinstance(result, np.ndarray)

    def test_dtype_is_float(self):
        result = smooth_path([(0, 0), (1, 1)])
        assert np.issubdtype(result.dtype, np.floating)


# ─── Boundary conditions ─────────────────────────────────────────────────────

class TestBoundaryConditions:
    """The spline must pass through (or very close to) the first and last waypoints."""

    def test_starts_at_first_waypoint(self):
        wp = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0)]
        result = smooth_path(wp, num_points=200)
        assert result[0, 0] == pytest.approx(wp[0][0], abs=1e-3)
        assert result[0, 1] == pytest.approx(wp[0][1], abs=1e-3)

    def test_ends_at_last_waypoint(self):
        wp = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0)]
        result = smooth_path(wp, num_points=200)
        assert result[-1, 0] == pytest.approx(wp[-1][0], abs=1e-3)
        assert result[-1, 1] == pytest.approx(wp[-1][1], abs=1e-3)

    def test_minimum_two_waypoints(self):
        """Two waypoints is the minimum valid input."""
        result = smooth_path([(0, 0), (1, 0)], num_points=10)
        assert result.shape == (10, 2)

    def test_single_waypoint_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            smooth_path([(0, 0)])

    def test_empty_waypoints_raises(self):
        with pytest.raises((ValueError, IndexError)):
            smooth_path([])


# ─── Geometric correctness ────────────────────────────────────────────────────

class TestGeometricCorrectness:
    """The output curve must be geometrically reasonable."""

    def test_straight_line_stays_on_line(self):
        """Smoothing collinear waypoints should stay near the line y = 0."""
        wp = [(0, 0), (1, 0), (2, 0), (3, 0)]
        result = smooth_path(wp, num_points=100)
        # All y values should be essentially zero
        assert np.allclose(result[:, 1], 0.0, atol=1e-6), (
            "Collinear waypoints on y=0 should produce a straight line"
        )

    def test_x_values_are_monotonic_for_simple_s_curve(self):
        """For an S-curve with strictly increasing x, the smooth path x should
        also be monotonically increasing."""
        wp = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, -1.0), (4.0, 0.0)]
        result = smooth_path(wp, num_points=200)
        diffs = np.diff(result[:, 0])
        assert np.all(diffs >= -1e-6), (
            "x should not decrease for an S-curve with monotonically increasing x waypoints"
        )

    def test_output_within_bounding_box_of_waypoints(self):
        """The smooth curve should stay within a reasonable bounding box of the
        input waypoints (centripetal splines do not overshoot badly)."""
        wp = [(0, 0), (1, 2), (3, 1), (4, 3)]
        result = smooth_path(wp, num_points=300)
        pts = np.array(wp)
        margin = 0.5  # allow small overshoot
        assert result[:, 0].min() >= pts[:, 0].min() - margin
        assert result[:, 0].max() <= pts[:, 0].max() + margin
        assert result[:, 1].min() >= pts[:, 1].min() - margin
        assert result[:, 1].max() <= pts[:, 1].max() + margin

    def test_no_nan_or_inf(self):
        wp = [(0, 0), (1, 1), (2, -1), (3, 0)]
        result = smooth_path(wp, num_points=200)
        assert not np.any(np.isnan(result)), "Output contains NaN"
        assert not np.any(np.isinf(result)), "Output contains Inf"

    def test_symmetric_path_is_symmetric(self):
        """An S-curve that is symmetric about y=0 should produce a symmetric result."""
        wp = [(0, 0), (1, 1), (2, 0), (3, -1), (4, 0)]
        result = smooth_path(wp, num_points=200)
        # The path sampled at the midpoint should be near y=0
        mid = result[len(result) // 2]
        assert abs(mid[1]) < 0.15, (
            f"Midpoint y should be near 0 for symmetric path, got {mid[1]:.4f}"
        )

    def test_many_waypoints(self):
        """Should work for a large number of input waypoints."""
        wp = [(float(i), math.sin(i * 0.5)) for i in range(20)]
        result = smooth_path(wp, num_points=400)
        assert result.shape == (400, 2)
        assert not np.any(np.isnan(result))


# ─── Smoothness ───────────────────────────────────────────────────────────────

class TestSmoothness:
    """The output must be smoother than the raw waypoint sequence."""

    def test_output_smoother_than_input(self):
        """Measure total variation of y — the smoothed curve should have less
        high-frequency variation than linearly interpolating between waypoints."""
        wp = [(0, 0), (1, 2), (2, 0), (3, 2), (4, 0)]
        pts = np.array(wp)
        raw_tv = float(np.sum(np.abs(np.diff(pts[:, 1]))))

        result = smooth_path(wp, num_points=len(wp))
        smooth_tv = float(np.sum(np.abs(np.diff(result[:, 1]))))

        # The smooth path sampled at the same number of points should have
        # smaller or equal total variation
        assert smooth_tv <= raw_tv + 1e-6, (
            f"Smooth path (TV={smooth_tv:.3f}) is not smoother than raw (TV={raw_tv:.3f})"
        )

    def test_segment_lengths_are_roughly_uniform(self):
        """Evenly-sampled parameter should yield roughly uniform arc-length steps
        (within an order of magnitude, not identical)."""
        wp = [(0, 0), (1, 0.5), (2, 0), (3, -0.5), (4, 0)]
        result = smooth_path(wp, num_points=100)
        seg_lengths = np.sqrt(np.sum(np.diff(result, axis=0) ** 2, axis=1))
        ratio = seg_lengths.max() / (seg_lengths.min() + 1e-9)
        assert ratio < 20, (
            f"Segment length ratio max/min={ratio:.1f} is too large — path is not smooth"
        )
