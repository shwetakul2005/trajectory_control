#!/usr/bin/env python3
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, "/home/ubuntu/Desktop/ros2_ws/src/trajectory_control")
from trajectory_control.path_smoother import smooth_path
from trajectory_control.trajectory_generator import generate_trajectory

RESULTS  = os.path.expanduser("~/Desktop/ros2_ws/results")
WAYPOINTS = [(0.0,0.0),(1.0,1.0),(2.0,0.0),(3.0,-1.0),(4.0,0.0),(5.0,1.0),(6.0,0.0)]

try:
    cte = list(csv.reader(open(f"{RESULTS}/cte.csv")))
    vel = list(csv.reader(open(f"{RESULTS}/velocity.csv")))
except FileNotFoundError as e:
    print(f"ERROR: {e}"); exit(1)

# Regenerate full smooth path and trajectory from waypoints
path = smooth_path(WAYPOINTS, num_points=400)
traj = generate_trajectory(path, max_vel=0.20, accel=0.08)

tx = [p.x for p in traj]
ty = [p.y for p in traj]
tt = [p.t for p in traj]
tv = [p.v for p in traj]
ct = [float(r[0]) for r in cte]
ce = [float(r[1]) for r in cte]
vt = [float(r[0]) for r in vel]
vv = [float(r[1]) for r in vel]

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Trajectory Control Results - Differential Drive Robot", fontsize=14)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: Full smooth path vs raw waypoints
ax1 = fig.add_subplot(gs[0, :])
wx = [p[0] for p in WAYPOINTS]
wy = [p[1] for p in WAYPOINTS]
ax1.plot(wx, wy, "o--", color="#378ADD", markersize=10,
         linewidth=1.5, label="Raw waypoints", zorder=3)
for i,(x,y) in enumerate(WAYPOINTS):
    ax1.text(x+0.05, y+0.08, str(i+1), fontsize=9,
             color="#378ADD", fontweight="bold")
ax1.plot(tx, ty, "-", color="#1D9E75", linewidth=2.5,
         label="Smooth trajectory (cubic spline)", zorder=2)
ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
ax1.set_title("Task 1 and 2: Path Smoothing and Trajectory Generation")
ax1.legend(fontsize=9); ax1.set_aspect("equal"); ax1.grid(alpha=0.3)

# Plot 2: Velocity profile
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(tt, tv, "--", color="#B4B2A9", linewidth=1.5,
         label="Reference (trapezoidal)")
ax2.plot(vt, vv, "-", color="#378ADD", linewidth=2, label="Actual")
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Speed (m/s)")
ax2.set_title("Task 2: Trapezoidal Velocity Profile")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# Plot 3: Cross-track error
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(ct, ce, color="#E24B4A", linewidth=1.8, label="CTE")
if ce:
    mean_cte = np.mean(ce)
    ax3.axhline(mean_cte, color="#888780", linestyle="--",
                label=f"Mean = {mean_cte:.4f} m")
    print(f"Mean cross-track error: {mean_cte:.4f} m")
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("|CTE| (m)")
ax3.set_title("Task 3: Cross-Track Error (Tracking Performance)")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

plt.savefig(f"{RESULTS}/submission_plots.png", dpi=150, bbox_inches="tight")
print(f"Saved to {RESULTS}/submission_plots.png")
plt.show()
