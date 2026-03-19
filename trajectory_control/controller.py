"""
controller.py
-------------
Task 3: Pure Pursuit trajectory tracking controller.

Algorithm:
  At each control cycle (20 Hz):
  1. Find a 'lookahead point' on the trajectory — the first point
     that is at least L metres ahead of the robot.
  2. Transform the lookahead point into the robot's local frame
     (rotate by -theta so the robot always faces forward along x).
  3. The robot must travel an arc to reach the lookahead point.
     Arc curvature:  k = 2 * y_local / dist²
     Angular vel:    ω = v * k
     Linear vel:     v = constant (or from trajectory profile)

Why Pure Pursuit over PID:
  - Geometric — directly computes the arc connecting robot to goal.
  - Single intuitive tuning parameter: lookahead distance L.
  - Naturally handles large heading errors without integral windup.
  - Industry standard for mobile robot path tracking.

Tuning the lookahead distance L:
  - L too small: robot oscillates around the path (overcorrects).
  - L too large:  robot cuts corners and misses tight turns.
  - Rule of thumb: L ≈ 2–4 × robot wheelbase.
  - Turtlebot3 Burger wheelbase = 0.16 m → L ≈ 0.30–0.50 m.
"""

import math
from .trajectory_generator import TrajectoryPoint


class PurePursuitController:
    """
    Pure Pursuit geometric controller for differential drive robot.

    Attributes:
        lookahead_dist : L in metres — how far ahead the robot looks.
        max_vel        : maximum linear velocity command (m/s).
        max_omega      : maximum angular velocity command (rad/s).
    """

    def __init__(self,
                 lookahead_dist: float = 0.35,
                 max_vel:        float = 0.18,
                 max_omega:      float = 2.0):
        self.L         = lookahead_dist
        self.max_vel   = max_vel
        self.max_omega = max_omega

    def find_lookahead_point(self,
                             pose:        tuple[float, float, float],
                             trajectory:  list[TrajectoryPoint],
                             start_idx:   int
                             ) -> tuple[int, tuple[float, float]]:
        """
        Scan forward from start_idx to find the first trajectory
        point at least self.L metres from the robot.

        Args:
            pose       : (x, y, theta) — current robot pose in metres/rad
            trajectory : list of TrajectoryPoint
            start_idx  : index to start scanning from (prevents
                         the robot from backtracking)

        Returns:
            (new_idx, (goal_x, goal_y))
            new_idx is the updated progress index along trajectory.
        """
        rx, ry, _ = pose

        for i in range(start_idx, len(trajectory)):
            dist = math.hypot(trajectory[i].x - rx,
                              trajectory[i].y - ry)
            if dist >= self.L:
                return i, (trajectory[i].x, trajectory[i].y)

        # Past the end of trajectory — return last point
        last = trajectory[-1]
        return len(trajectory) - 1, (last.x, last.y)

    def compute_velocity_commands(self,
                                  pose:            tuple[float, float, float],
                                  lookahead_point: tuple[float, float],
                                  ref_speed:       float = None
                                  ) -> tuple[float, float]:
        """
        Core Pure Pursuit formula.

        Transforms the lookahead point into robot frame, then
        computes the arc curvature and resulting velocity commands.

        Args:
            pose            : (x, y, theta) current robot pose
            lookahead_point : (gx, gy) goal point in world frame
            ref_speed       : desired linear speed from the trajectory
                              velocity profile (m/s). When provided,
                              this replaces the fixed max_vel so the
                              robot honours the trapezoidal accel/decel
                              profile computed in Task 2. Clamped to
                              [0, max_vel] for safety.
                              If None, falls back to max_vel (original
                              behaviour, backwards-compatible).

        Returns:
            (v, omega) — linear (m/s) and angular (rad/s) velocity.
            These map directly to geometry_msgs/Twist:
                cmd.linear.x  = v
                cmd.angular.z = omega
        """
        rx, ry, theta = pose
        gx, gy        = lookahead_point

        # --- Transform goal to robot frame ---
        # Rotation by -theta aligns robot heading with x-axis
        dx =  (gx - rx) * math.cos(theta) + (gy - ry) * math.sin(theta)
        dy = -(gx - rx) * math.sin(theta) + (gy - ry) * math.cos(theta)

        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0

        # --- Arc curvature k = 2 * lateral_offset / dist² ---
        # Derivation: the robot and lookahead point lie on a circle.
        # The lateral offset dy is the chord's perpendicular distance.
        curvature = 2.0 * dy / (dist ** 2)

        # Use trajectory profile speed if supplied, else fall back to max_vel.
        # Clamp to [0, max_vel] so a bad profile value can never exceed limits.
        if ref_speed is not None:
            v = max(0.0, min(self.max_vel, ref_speed))
        else:
            v = self.max_vel

        omega = v * curvature

        # Clamp to physical limits
        omega = max(-self.max_omega, min(self.max_omega, omega))

        return v, omega


class CrossTrackErrorMonitor:
    """
    Computes cross-track error (CTE) — the perpendicular distance
    from the robot to the nearest point on the reference trajectory.

    Used for:
      - Performance evaluation and logging
      - Generating the CTE plot for the submission
      - NOT used for control (Pure Pursuit handles correction geometrically)

    Sign convention:
      Positive CTE = robot is to the LEFT  of the path direction.
      Negative CTE = robot is to the RIGHT of the path direction.
    """

    @staticmethod
    def compute(pose:       tuple[float, float, float],
                trajectory: list[TrajectoryPoint]) -> float:
        """
        Args:
            pose       : (x, y, theta) current robot pose
            trajectory : reference trajectory

        Returns:
            Signed cross-track error in metres.
        """
        rx, ry, _ = pose
        min_dist   = float('inf')
        sign       = 1.0

        for i in range(1, len(trajectory)):
            ax, ay = trajectory[i - 1].x, trajectory[i - 1].y
            bx, by = trajectory[i].x,     trajectory[i].y

            # Project robot position onto segment [A, B]
            seg_sq = (bx - ax) ** 2 + (by - ay) ** 2
            if seg_sq < 1e-9:
                continue
            t_proj = max(0.0, min(1.0,
                ((rx - ax) * (bx - ax) + (ry - ay) * (by - ay)) / seg_sq))

            # Closest point on segment
            cx = ax + t_proj * (bx - ax)
            cy = ay + t_proj * (by - ay)

            dist = math.hypot(rx - cx, ry - cy)
            if dist < min_dist:
                min_dist = dist
                # Cross product gives sign: which side of path is robot on
                cross = (bx - ax) * (ry - ay) - (by - ay) * (rx - ax)
                sign  = 1.0 if cross >= 0.0 else -1.0

        return sign * min_dist