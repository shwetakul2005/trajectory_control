"""
tests/test_integration.py
--------------------------
End-to-end integration tests for the full trajectory pipeline:
    waypoints → smooth_path → generate_trajectory → PurePursuitController

These tests verify that all three modules work correctly together,
mimicking what the ROS2 node does at runtime.

Run with:
    pytest tests/test_integration.py -v
"""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajectory_control.path_smoother import smooth_path
from trajectory_control.trajectory_generator import generate_trajectory
from trajectory_control.controller import PurePursuitController, CrossTrackErrorMonitor


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    """Build the full pipeline for the default S-curve from params.yaml."""
    waypoints = [
        (0.0, 0.0), (1.0, 1.0), (2.0, 0.0),
        (3.0, -1.0), (4.0, 0.0), (5.0, 1.0), (6.0, 0.0)
    ]
    path = smooth_path(waypoints, num_points=400)
    traj = generate_trajectory(path, max_vel=0.20, accel=0.08)
    ctrl = PurePursuitController(lookahead_dist=0.35, max_vel=0.18, max_omega=2.0)
    return waypoints, path, traj, ctrl


# ─── Pipeline integrity ───────────────────────────────────────────────────────

class TestPipelineIntegrity:
    def test_full_pipeline_runs_without_error(self, pipeline):
        waypoints, path, traj, ctrl = pipeline
        assert len(traj) == 400

    def test_trajectory_inherits_path_shape(self, pipeline):
        waypoints, path, traj, ctrl = pipeline
        assert len(traj) == path.shape[0]

    def test_trajectory_starts_near_first_waypoint(self, pipeline):
        waypoints, path, traj, ctrl = pipeline
        first_wp = waypoints[0]
        assert traj[0].x == pytest.approx(first_wp[0], abs=0.05)
        assert traj[0].y == pytest.approx(first_wp[1], abs=0.05)

    def test_trajectory_ends_near_last_waypoint(self, pipeline):
        waypoints, path, traj, ctrl = pipeline
        last_wp = waypoints[-1]
        assert traj[-1].x == pytest.approx(last_wp[0], abs=0.05)
        assert traj[-1].y == pytest.approx(last_wp[1], abs=0.05)


# ─── Simulated control loop ───────────────────────────────────────────────────

class TestSimulatedControlLoop:
    """
    Simulate the 20 Hz control loop from main_node.py with a perfect
    (noise-free) kinematic model and verify the robot makes progress.
    """

    def _run_simulation(self, waypoints, steps=1200, dt=0.05,
                        max_vel=0.18, max_omega=2.0,
                        lookahead=0.35, goal_tol=0.15):
        """
        Run a simple unicycle simulation and return:
          (reached_goal, actual_path, cte_log, final_pose)
        """
        path = smooth_path(waypoints, num_points=400)
        traj = generate_trajectory(path, max_vel=0.20, accel=0.08)
        ctrl = PurePursuitController(
            lookahead_dist=lookahead, max_vel=max_vel, max_omega=max_omega
        )

        x, y, theta = 0.0, 0.0, 0.0
        traj_idx = 0
        actual_path = []
        cte_log = []
        reached_goal = False

        for _ in range(steps):
            # Check goal
            dist = math.hypot(x - traj[-1].x, y - traj[-1].y)
            if traj_idx >= len(traj) - 1 or dist < goal_tol:
                reached_goal = True
                break

            # Pure pursuit
            traj_idx, la_pt = ctrl.find_lookahead_point((x, y, theta), traj, traj_idx)
            v, omega = ctrl.compute_velocity_commands((x, y, theta), la_pt)

            # Perfect kinematic integration (noise-free)
            x     += v * math.cos(theta) * dt
            y     += v * math.sin(theta) * dt
            theta += omega * dt
            theta  = math.atan2(math.sin(theta), math.cos(theta))

            actual_path.append((x, y))
            cte = CrossTrackErrorMonitor.compute((x, y, theta), traj)
            cte_log.append(abs(cte))

        return reached_goal, actual_path, cte_log, (x, y, theta)

    def test_robot_reaches_goal_straight_line(self):
        """Robot should reach the end of a straight-line path."""
        reached, _, _, _ = self._run_simulation(
            [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        )
        assert reached, "Robot did not reach goal on straight path"

    def test_robot_reaches_goal_s_curve(self):
        """Robot should reach the end of the default S-curve."""
        reached, _, _, _ = self._run_simulation(
            [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0),
             (3.0, -1.0), (4.0, 0.0), (5.0, 1.0), (6.0, 0.0)]
        )
        assert reached, "Robot did not reach goal on S-curve"

    def test_mean_cte_is_small_on_straight_path(self):
        """Mean cross-track error on a straight path should be under 10 cm."""
        reached, _, cte_log, _ = self._run_simulation(
            [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        )
        assert reached, "Robot did not complete the path"
        mean_cte = sum(cte_log) / len(cte_log)
        assert mean_cte < 0.10, (
            f"Mean CTE {mean_cte:.4f} m is too large for straight path"
        )

    def test_mean_cte_is_acceptable_on_s_curve(self):
        """Mean cross-track error on the S-curve should stay under 15 cm."""
        reached, _, cte_log, _ = self._run_simulation(
            [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0),
             (3.0, -1.0), (4.0, 0.0), (5.0, 1.0), (6.0, 0.0)]
        )
        assert reached, "Robot did not complete the S-curve"
        mean_cte = sum(cte_log) / len(cte_log)
        assert mean_cte < 0.15, (
            f"Mean CTE {mean_cte:.4f} m is too large for S-curve"
        )

    def test_robot_makes_forward_progress(self):
        """Robot should always be moving forward — x should increase over time
        on a left-to-right path."""
        _, actual_path, _, _ = self._run_simulation(
            [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)], steps=200
        )
        assert len(actual_path) > 5
        xs = [p[0] for p in actual_path]
        assert xs[-1] > xs[0], "Robot did not make forward progress"

    def test_different_waypoint_configs(self):
        """Pipeline should work for simple, S-curve, and L-shaped paths."""
        configs = [
            [(0, 0), (2, 0), (4, 0)],                     # straight
            [(0, 0), (1, 1), (2, 0), (3, -1), (4, 0)],   # S-curve
            [(0, 0), (0, 2), (2, 2)],                      # L-shape
        ]
        for wp in configs:
            path = smooth_path(wp, num_points=200)
            traj = generate_trajectory(path, max_vel=0.18, accel=0.08)
            ctrl = PurePursuitController()
            idx, pt = ctrl.find_lookahead_point((0, 0, 0), traj, 0)
            assert idx >= 0
            assert pt is not None


# ─── Parameterisation robustness ─────────────────────────────────────────────

class TestParameterisationRobustness:
    """Verify the pipeline works across different parameter combinations."""

    def _check_velocity_params(self, max_vel, accel):
        path = smooth_path([(0, 0), (3, 0), (6, 0)], num_points=100)
        traj = generate_trajectory(path, max_vel=max_vel, accel=accel)
        assert all(pt.v <= max_vel + 1e-6 for pt in traj)
        assert traj[-1].t > 0

    def test_velocity_params_slow(self):
        self._check_velocity_params(0.10, 0.04)

    def test_velocity_params_default(self):
        self._check_velocity_params(0.18, 0.08)

    def test_velocity_params_fast(self):
        self._check_velocity_params(0.22, 0.10)

    def _check_lookahead(self, lookahead):
        path = smooth_path([(0, 0), (2, 0), (4, 0)], num_points=80)
        traj = generate_trajectory(path)
        ctrl = PurePursuitController(lookahead_dist=lookahead)
        idx, pt = ctrl.find_lookahead_point((0, 0, 0), traj, 0)
        v, omega = ctrl.compute_velocity_commands((0, 0, 0), pt)
        assert v > 0
        assert not math.isnan(omega)

    def test_lookahead_short(self):
        self._check_lookahead(0.20)

    def test_lookahead_default(self):
        self._check_lookahead(0.35)

    def test_lookahead_long(self):
        self._check_lookahead(0.50)

    def _check_path_density(self, num_points):
        path = smooth_path(
            [(0, 0), (1, 1), (2, 0), (3, -1), (4, 0)],
            num_points=num_points
        )
        traj = generate_trajectory(path)
        assert len(traj) == num_points
        assert traj[0].t == pytest.approx(0.0, abs=1e-4)

    def test_path_density_sparse(self):
        self._check_path_density(20)

    def test_path_density_medium(self):
        self._check_path_density(100)

    def test_path_density_dense(self):
        self._check_path_density(400)
