"""
path_smoother.py
----------------
Task 1: Smooth a list of discrete 2D waypoints into a continuous
        C2 curve using centripetal cubic spline interpolation.

Why cubic spline:
  - C2 continuity means position, velocity AND acceleration are
    continuous — no jerky motion or wheel slip on a real robot.

Why centripetal parameterisation:
  - Parameter t is proportional to sqrt(chord length) between points.
  - This prevents cusps and self-intersections at sharp turns,
    which uniform parameterisation cannot guarantee.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def smooth_path(waypoints: list[tuple[float, float]],
                num_points: int = 400) -> np.ndarray:
    """
    Fit a centripetal cubic spline through waypoints.

    Args:
        waypoints  : list of (x, y) tuples in metres,
                     e.g. [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]
        num_points : number of points on the output smooth curve.
                     Higher = smoother-looking, but more memory.

    Returns:
        np.ndarray of shape (num_points, 2)
        Each row is [x, y] in metres.

    Raises:
        ValueError: if fewer than 2 waypoints are provided.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints to smooth.")

    pts = np.array(waypoints, dtype=float)

    # --- Centripetal parameterisation ---
    # t[i] = cumulative sum of sqrt(chord_length) between points
    # This maps parameter t to arc-like distance, preventing cusps.
    diffs  = np.diff(pts, axis=0)                        # shape (N-1, 2)
    chords = np.sqrt((diffs ** 2).sum(axis=1))           # chord lengths
    t      = np.concatenate([[0.0], np.cumsum(np.sqrt(chords))])
    t      = t / t[-1]                                   # normalise to [0,1]

    # --- Fit independent splines for x(t) and y(t) ---
    spline_x = CubicSpline(t, pts[:, 0])
    spline_y = CubicSpline(t, pts[:, 1])

    # --- Sample at evenly-spaced parameter values ---
    t_samples = np.linspace(0.0, 1.0, num_points)
    smooth_x  = spline_x(t_samples)
    smooth_y  = spline_y(t_samples)

    return np.column_stack([smooth_x, smooth_y])
    """
    if __name__ == "__main__":
    waypoints = [(0, 0), (1, 2), (3, 1), (4, 3)]

    smooth = smooth_path(waypoints)

    print("Smoothed points:")
    print(smooth[:10])  # print first 10 points
    """
