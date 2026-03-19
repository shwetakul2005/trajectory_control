"""
tests/test_trajectory_generator.py
-----------------------------------
Unit tests for trajectory_control.trajectory_generator.generate_trajectory

Run with:
    pytest tests/test_trajectory_generator.py -v
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajectory_control.path_smoother import smooth_path
from trajectory_control.trajectory_generator import generate_trajectory, TrajectoryPoint


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def straight_path():
    """A straight horizontal path with 50 evenly-spaced points."""
    return smooth_path([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)], num_points=50)


@pytest.fixture
def s_curve_path():
    """Standard S-curve used in the package's params.yaml."""
    return smooth_path(
        [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, -1.0), (4.0, 0.0), (5.0, 1.0), (6.0, 0.0)],
        num_points=400
    )


# ─── Return type ─────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_list(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert isinstance(traj, list)

    def test_each_element_is_trajectory_point(self, straight_path):
        traj = generate_trajectory(straight_path)
        for pt in traj:
            assert isinstance(pt, TrajectoryPoint), (
                f"Expected TrajectoryPoint, got {type(pt)}"
            )

    def test_trajectory_point_fields_exist(self, straight_path):
        traj = generate_trajectory(straight_path)
        pt = traj[0]
        assert hasattr(pt, 'x')
        assert hasattr(pt, 'y')
        assert hasattr(pt, 't')
        assert hasattr(pt, 'v')

    def test_length_matches_path_length(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert len(traj) == len(straight_path)


# ─── Velocity profile ─────────────────────────────────────────────────────────

class TestVelocityProfile:
    MAX_VEL = 0.20
    ACCEL   = 0.08

    def test_all_velocities_nonnegative(self, s_curve_path):
        traj = generate_trajectory(s_curve_path, max_vel=self.MAX_VEL, accel=self.ACCEL)
        for pt in traj:
            assert pt.v >= 0.0, f"Negative velocity {pt.v} at point {pt}"

    def test_velocity_never_exceeds_max(self, s_curve_path):
        traj = generate_trajectory(s_curve_path, max_vel=self.MAX_VEL, accel=self.ACCEL)
        for pt in traj:
            assert pt.v <= self.MAX_VEL + 1e-6, (
                f"Velocity {pt.v:.4f} exceeds max_vel {self.MAX_VEL}"
            )

    def test_starts_slow(self, s_curve_path):
        """Robot should begin from near rest (trapezoidal profile: accel phase)."""
        traj = generate_trajectory(s_curve_path, max_vel=self.MAX_VEL, accel=self.ACCEL)
        assert traj[0].v < self.MAX_VEL * 0.15, (
            f"First point velocity {traj[0].v:.4f} should be near 0, not already at cruise"
        )

    def test_ends_slow(self, s_curve_path):
        """Robot should decelerate to near rest at the end."""
        traj = generate_trajectory(s_curve_path, max_vel=self.MAX_VEL, accel=self.ACCEL)
        assert traj[-1].v < self.MAX_VEL * 0.15, (
            f"Last point velocity {traj[-1].v:.4f} should be near 0, not still at cruise"
        )

    def test_reaches_cruise_in_middle(self, s_curve_path):
        """The mid-section of a long path should hit cruise speed."""
        traj = generate_trajectory(s_curve_path, max_vel=self.MAX_VEL, accel=self.ACCEL)
        mid = len(traj) // 2
        # Allow small tolerance for discretisation
        assert traj[mid].v >= self.MAX_VEL * 0.90, (
            f"Midpoint velocity {traj[mid].v:.4f} should be near max_vel {self.MAX_VEL}"
        )

    def test_custom_max_vel_respected(self, s_curve_path):
        for max_v in [0.10, 0.15, 0.22]:
            traj = generate_trajectory(s_curve_path, max_vel=max_v, accel=self.ACCEL)
            assert all(pt.v <= max_v + 1e-6 for pt in traj), (
                f"max_vel={max_v} violated"
            )


# ─── Timestamps ───────────────────────────────────────────────────────────────

class TestTimestamps:
    def test_first_timestamp_is_zero(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert traj[0].t == pytest.approx(0.0, abs=1e-4)

    def test_timestamps_monotonically_increasing(self, s_curve_path):
        traj = generate_trajectory(s_curve_path)
        for i in range(1, len(traj)):
            assert traj[i].t >= traj[i - 1].t, (
                f"Timestamp decreased at index {i}: {traj[i].t} < {traj[i-1].t}"
            )

    def test_total_time_is_positive(self, s_curve_path):
        traj = generate_trajectory(s_curve_path)
        assert traj[-1].t > 0.0

    def test_faster_speed_means_shorter_time(self):
        """Doubling max_vel on the same path should reduce total time."""
        path = smooth_path([(0, 0), (3, 0), (6, 0)], num_points=100)
        traj_slow = generate_trajectory(path, max_vel=0.10, accel=0.05)
        traj_fast = generate_trajectory(path, max_vel=0.20, accel=0.10)
        assert traj_fast[-1].t < traj_slow[-1].t, (
            "Higher max_vel should produce shorter total trajectory time"
        )


# ─── Spatial consistency ─────────────────────────────────────────────────────

class TestSpatialConsistency:
    def test_xy_coordinates_match_path(self, s_curve_path):
        """Trajectory x, y must exactly match the input smooth_path."""
        traj = generate_trajectory(s_curve_path)
        for i, pt in enumerate(traj):
            assert pt.x == pytest.approx(float(s_curve_path[i, 0]), abs=1e-4)
            assert pt.y == pytest.approx(float(s_curve_path[i, 1]), abs=1e-4)

    def test_no_nan_or_inf(self, s_curve_path):
        traj = generate_trajectory(s_curve_path)
        for pt in traj:
            assert not (pt.x != pt.x), f"NaN in x at {pt}"
            assert not (pt.y != pt.y), f"NaN in y at {pt}"
            assert not (pt.t != pt.t), f"NaN in t at {pt}"
            assert not (pt.v != pt.v), f"NaN in v at {pt}"


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_minimum_two_point_path(self):
        """Should work with the smallest valid path."""
        path = smooth_path([(0, 0), (1, 0)], num_points=5)
        traj = generate_trajectory(path)
        assert len(traj) == 5

    def test_raises_on_single_point_path(self):
        with pytest.raises(ValueError):
            generate_trajectory(np.array([[0.0, 0.0]]))

    def test_very_short_path_no_crash(self):
        """A very short path should not cause division by zero or NaN."""
        path = smooth_path([(0, 0), (0.01, 0)], num_points=10)
        traj = generate_trajectory(path, max_vel=0.20, accel=0.08)
        assert len(traj) == 10
        assert all(pt.v >= 0 for pt in traj)
