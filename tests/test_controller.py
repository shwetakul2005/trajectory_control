"""
tests/test_controller.py
------------------------
Unit tests for:
  - PurePursuitController  (find_lookahead_point, compute_velocity_commands)
  - CrossTrackErrorMonitor (compute)

Run with:
    pytest tests/test_controller.py -v
"""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajectory_control.path_smoother import smooth_path
from trajectory_control.trajectory_generator import generate_trajectory, TrajectoryPoint
from trajectory_control.controller import PurePursuitController, CrossTrackErrorMonitor


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_traj(waypoints, num_points=100, max_vel=0.20, accel=0.08):
    """Build a trajectory from raw waypoints."""
    path = smooth_path(waypoints, num_points=num_points)
    return generate_trajectory(path, max_vel=max_vel, accel=accel)


def make_line_traj(x_end=4.0, n=20):
    """Straight trajectory along the x-axis: (0,0) → (x_end, 0)."""
    return make_traj([(0.0, 0.0), (x_end / 2, 0.0), (x_end, 0.0)], num_points=n)


def fake_traj_points(pairs):
    """Build a list of TrajectoryPoint from (x, y) pairs, with dummy t and v."""
    return [
        TrajectoryPoint(x=float(x), y=float(y), t=float(i) * 0.1, v=0.2)
        for i, (x, y) in enumerate(pairs)
    ]


# ─── PurePursuitController: construction ─────────────────────────────────────

class TestControllerConstruction:
    def test_default_parameters(self):
        ctrl = PurePursuitController()
        assert ctrl.L > 0
        assert ctrl.max_vel > 0
        assert ctrl.max_omega > 0

    def test_custom_parameters_stored(self):
        ctrl = PurePursuitController(lookahead_dist=0.5, max_vel=0.15, max_omega=1.5)
        assert ctrl.L == pytest.approx(0.5)
        assert ctrl.max_vel == pytest.approx(0.15)
        assert ctrl.max_omega == pytest.approx(1.5)


# ─── find_lookahead_point ─────────────────────────────────────────────────────

class TestFindLookaheadPoint:
    L = 0.35

    def setup_method(self):
        self.ctrl = PurePursuitController(lookahead_dist=self.L)
        self.traj = make_line_traj(x_end=4.0, n=40)

    def test_returns_tuple_of_idx_and_point(self):
        idx, pt = self.ctrl.find_lookahead_point((0, 0, 0), self.traj, 0)
        assert isinstance(idx, int)
        assert isinstance(pt, tuple)
        assert len(pt) == 2

    def test_lookahead_point_at_least_L_away(self):
        pose = (0.0, 0.0, 0.0)
        idx, (gx, gy) = self.ctrl.find_lookahead_point(pose, self.traj, 0)
        dist = math.hypot(gx - pose[0], gy - pose[1])
        # Allow one-step tolerance (last point fallback)
        assert dist >= self.L - 0.05 or idx == len(self.traj) - 1

    def test_idx_never_goes_backward(self):
        """start_idx should prevent the robot from revisiting earlier points."""
        start = 10
        idx, _ = self.ctrl.find_lookahead_point(
            (self.traj[10].x, self.traj[10].y, 0), self.traj, start
        )
        assert idx >= start

    def test_returns_last_point_when_past_end(self):
        """If robot is at the end, it should return the last trajectory point."""
        last = self.traj[-1]
        idx, pt = self.ctrl.find_lookahead_point(
            (last.x, last.y, 0), self.traj, len(self.traj) - 1
        )
        assert idx == len(self.traj) - 1

    def test_lookahead_advances_with_robot(self):
        """Lookahead index should advance as the robot moves forward."""
        idx_a, _ = self.ctrl.find_lookahead_point((0.0, 0, 0), self.traj, 0)
        idx_b, _ = self.ctrl.find_lookahead_point((1.0, 0, 0), self.traj, idx_a)
        idx_c, _ = self.ctrl.find_lookahead_point((2.0, 0, 0), self.traj, idx_b)
        assert idx_b >= idx_a
        assert idx_c >= idx_b


# ─── compute_velocity_commands ────────────────────────────────────────────────

class TestComputeVelocityCommands:
    MAX_VEL   = 0.18
    MAX_OMEGA = 2.0

    def setup_method(self):
        self.ctrl = PurePursuitController(
            lookahead_dist=0.35,
            max_vel=self.MAX_VEL,
            max_omega=self.MAX_OMEGA
        )

    # --- Linear velocity ---

    def test_linear_velocity_is_positive(self):
        v, _ = self.ctrl.compute_velocity_commands((0, 0, 0), (1, 0))
        assert v > 0

    def test_linear_velocity_equals_max_vel(self):
        v, _ = self.ctrl.compute_velocity_commands((0, 0, 0), (2, 0))
        assert v == pytest.approx(self.MAX_VEL, abs=1e-6)

    # --- Goal directly ahead → zero angular velocity ---

    def test_goal_straight_ahead_gives_zero_omega(self):
        """Robot at origin facing +x, goal directly on +x → omega should be 0."""
        v, omega = self.ctrl.compute_velocity_commands(
            (0.0, 0.0, 0.0), (1.0, 0.0)
        )
        assert omega == pytest.approx(0.0, abs=1e-6)

    def test_robot_facing_goal_any_heading(self):
        """Robot facing exactly toward the goal should give near-zero omega."""
        goal = (3.0, 4.0)
        theta = math.atan2(goal[1], goal[0])  # angle from origin to goal
        v, omega = self.ctrl.compute_velocity_commands((0.0, 0.0, theta), goal)
        assert abs(omega) < 0.01

    # --- Goal to the left → positive omega (turn left) ---

    def test_goal_to_left_gives_positive_omega(self):
        """Robot at origin facing +x, goal to its left (+y direction) → omega > 0."""
        _, omega = self.ctrl.compute_velocity_commands(
            (0.0, 0.0, 0.0), (0.5, 0.5)
        )
        assert omega > 0, f"Expected positive omega for leftward goal, got {omega}"

    def test_goal_to_right_gives_negative_omega(self):
        """Robot at origin facing +x, goal to its right (-y direction) → omega < 0."""
        _, omega = self.ctrl.compute_velocity_commands(
            (0.0, 0.0, 0.0), (0.5, -0.5)
        )
        assert omega < 0, f"Expected negative omega for rightward goal, got {omega}"

    # --- Omega clamping ---

    def test_omega_clamped_to_max(self):
        """A goal almost directly behind the robot should not exceed max_omega."""
        _, omega = self.ctrl.compute_velocity_commands(
            (0.0, 0.0, 0.0), (0.0, 0.01)  # almost 90° left turn
        )
        assert abs(omega) <= self.MAX_OMEGA + 1e-6

    def test_omega_symmetric_for_symmetric_goals(self):
        """Goals equally to the left and right should produce equal-magnitude omegas."""
        _, omega_left  = self.ctrl.compute_velocity_commands((0,0,0), (0.5,  0.4))
        _, omega_right = self.ctrl.compute_velocity_commands((0,0,0), (0.5, -0.4))
        assert abs(omega_left) == pytest.approx(abs(omega_right), rel=1e-4)
        assert omega_left > 0
        assert omega_right < 0

    # --- Edge case: robot already at goal ---

    def test_goal_at_robot_position_returns_zero(self):
        """If goal coincides with the robot, commands should be (0, 0)."""
        v, omega = self.ctrl.compute_velocity_commands((1.0, 2.0, 0.5), (1.0, 2.0))
        assert v == pytest.approx(0.0, abs=1e-6)
        assert omega == pytest.approx(0.0, abs=1e-6)

    # --- Heading sensitivity ---

    def test_heading_matters(self):
        """Same goal, different headings should produce different omegas."""
        goal = (1.0, 0.5)
        _, omega_0   = self.ctrl.compute_velocity_commands((0,0,  0.0), goal)
        _, omega_90  = self.ctrl.compute_velocity_commands((0,0, math.pi/2), goal)
        assert omega_0 != pytest.approx(omega_90, abs=0.01), (
            "Heading change should affect angular velocity"
        )


# ─── CrossTrackErrorMonitor ───────────────────────────────────────────────────

class TestCrossTrackErrorMonitor:
    """Tests for signed cross-track error computation."""

    def _line_traj(self, length=4.0, n=20):
        """Straight trajectory along y=0 from (0,0) to (length, 0)."""
        xs = np.linspace(0, length, n)
        return [
            TrajectoryPoint(x=float(x), y=0.0, t=float(i) * 0.1, v=0.2)
            for i, x in enumerate(xs)
        ]

    # --- On path ---

    def test_robot_on_path_gives_zero_cte(self):
        traj = self._line_traj()
        cte = CrossTrackErrorMonitor.compute((2.0, 0.0, 0.0), traj)
        assert cte == pytest.approx(0.0, abs=1e-6)

    # --- Sign convention ---

    def test_robot_left_of_path_gives_positive_cte(self):
        """Robot above the x-axis path (y > 0) is to the left → positive CTE."""
        traj = self._line_traj()
        cte = CrossTrackErrorMonitor.compute((2.0, 0.5, 0.0), traj)
        assert cte > 0, f"Expected positive CTE for robot left of path, got {cte}"

    def test_robot_right_of_path_gives_negative_cte(self):
        """Robot below the x-axis path (y < 0) is to the right → negative CTE."""
        traj = self._line_traj()
        cte = CrossTrackErrorMonitor.compute((2.0, -0.5, 0.0), traj)
        assert cte < 0, f"Expected negative CTE for robot right of path, got {cte}"

    # --- Magnitude ---

    def test_cte_magnitude_matches_perpendicular_distance(self):
        """For a straight horizontal path, CTE magnitude = y offset."""
        traj = self._line_traj()
        for offset in [0.1, 0.3, 0.5, 1.0]:
            cte = CrossTrackErrorMonitor.compute((2.0, offset, 0.0), traj)
            assert abs(cte) == pytest.approx(offset, abs=0.05), (
                f"CTE magnitude {abs(cte):.4f} should match y-offset {offset}"
            )

    def test_cte_symmetric_around_path(self):
        """Equal offsets left and right should give equal-magnitude CTEs."""
        traj = self._line_traj()
        cte_left  = CrossTrackErrorMonitor.compute((2.0,  0.4, 0.0), traj)
        cte_right = CrossTrackErrorMonitor.compute((2.0, -0.4, 0.0), traj)
        assert abs(cte_left) == pytest.approx(abs(cte_right), abs=1e-6)

    # --- Larger offsets ---

    def test_larger_offset_gives_larger_cte(self):
        traj = self._line_traj()
        cte_small = CrossTrackErrorMonitor.compute((2.0, 0.2, 0.0), traj)
        cte_large = CrossTrackErrorMonitor.compute((2.0, 0.8, 0.0), traj)
        assert abs(cte_large) > abs(cte_small)

    # --- Non-degenerate trajectory ---

    def test_cte_on_s_curve_trajectory(self):
        """CTE should not be NaN or Inf on a realistic S-curve trajectory."""
        traj = make_traj(
            [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, -1.0), (4.0, 0.0)],
            num_points=100
        )
        cte = CrossTrackErrorMonitor.compute((2.0, 0.1, 0.0), traj)
        assert not math.isnan(cte), "CTE is NaN on S-curve trajectory"
        assert not math.isinf(cte), "CTE is Inf on S-curve trajectory"

    # --- Robot before path start and after path end ---

    def test_robot_before_path_start(self):
        traj = self._line_traj()
        cte = CrossTrackErrorMonitor.compute((-0.5, 0.3, 0.0), traj)
        assert not math.isnan(cte)

    def test_robot_after_path_end(self):
        traj = self._line_traj(length=4.0)
        cte = CrossTrackErrorMonitor.compute((5.0, 0.0, 0.0), traj)
        assert not math.isnan(cte)
