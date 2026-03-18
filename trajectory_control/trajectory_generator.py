"""
trajectory_generator.py
-----------------------
Task 2: Convert a geometric path (x, y points) into a
        time-parameterised trajectory (x, y, t, v).

Key concepts:
  Arc-length parameterisation:
    Decouples speed from path geometry. Without it, closely-spaced
    path points would produce slow motion and wide-spaced points
    would produce fast motion regardless of the desired speed.

  Trapezoidal velocity profile:
    - Phase 1 (acceleration): v increases from 0 to max_vel
    - Phase 2 (cruise):       v = max_vel
    - Phase 3 (deceleration): v decreases from max_vel to 0
    The velocity-vs-time graph looks like a trapezoid.
    This is standard in industrial motion control because it
    respects actuator torque limits and avoids jerk.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """A single point in the time-parameterised trajectory."""
    x: float   # position x (metres)
    y: float   # position y (metres)
    t: float   # time when robot should be at this point (seconds)
    v: float   # desired linear speed at this point (m/s)


def generate_trajectory(smooth_path:  np.ndarray,
                        max_vel:      float = 0.20,
                        accel:        float = 0.08
                        ) -> list[TrajectoryPoint]:
    """
    Convert a smooth path into a time-stamped trajectory with
    trapezoidal velocity profile.

    Args:
        smooth_path : (N, 2) numpy array from smooth_path()
        max_vel     : cruise speed in m/s.
                      Turtlebot3 Burger hardware limit = 0.22 m/s.
        accel       : acceleration and deceleration magnitude (m/s²).

    Returns:
        List of TrajectoryPoint — the deliverable:
            trajectory = [(x0,y0,t0,v0), (x1,y1,t1,v1), ...]

    Design note:
        Time stamps are computed by integrating dt = ds / v(s)
        along the arc length s. This is exact for any velocity
        profile shape.
    """
    if len(smooth_path) < 2:
        raise ValueError("smooth_path must have at least 2 points.")

    # --- Step 1: Compute arc length at each path point ---
    diffs    = np.diff(smooth_path, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    arc_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total    = arc_lens[-1]

    # --- Step 2: Trapezoidal profile ---
    # Acceleration distance: derived from v² = 2*a*s → s = v²/(2a)
    accel_dist = min((max_vel ** 2) / (2.0 * accel), total / 2.0)

    def speed_at(s: float) -> float:
        """Returns the desired speed at arc length s."""
        if s < accel_dist:
            # Accelerating: v = sqrt(2 * a * s)
            return float(np.sqrt(max(2.0 * accel * s, 1e-4)))
        elif s > total - accel_dist:
            # Decelerating: mirror of acceleration phase
            remaining = total - s
            return float(np.sqrt(max(2.0 * accel * remaining, 1e-4)))
        else:
            # Cruising
            return max_vel

    # --- Step 3: Integrate time stamps ---
    # dt = ds / v  →  t[i] = sum of ds/v over all previous segments
    trajectory: list[TrajectoryPoint] = []
    time = 0.0

    for i, s in enumerate(arc_lens):
        v = min(speed_at(s), max_vel)

        if i > 0:
            ds    = arc_lens[i] - arc_lens[i - 1]
            time += ds / max(v, 0.01)          # guard against div/zero

        trajectory.append(TrajectoryPoint(
            x = round(float(smooth_path[i, 0]), 4),
            y = round(float(smooth_path[i, 1]), 4),
            t = round(time, 4),
            v = round(v,    4),
        ))

    return trajectory
