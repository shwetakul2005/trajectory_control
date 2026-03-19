#!/usr/bin/env python3
"""
main_node.py
------------
ROS2 node that ties all three pipeline stages together
and controls the Turtlebot3 in Gazebo simulation.
"""

import rclpy
import math
import csv
import os

from rclpy.node               import Node
from geometry_msgs.msg        import Twist, PoseStamped
from nav_msgs.msg             import Path, Odometry
from std_msgs.msg             import Float32
from visualization_msgs.msg   import Marker, MarkerArray

from .path_smoother           import smooth_path
from .trajectory_generator    import generate_trajectory, TrajectoryPoint
from .controller              import PurePursuitController, CrossTrackErrorMonitor

class TrajectoryControlNode(Node):
    """
    Main ROS2 node for trajectory tracking.
    """

    def __init__(self):
        super().__init__('trajectory_control_node')

        # ── Declare Parameters ──────────────────────────────────────
        self.declare_parameter('waypoints_x', [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        self.declare_parameter('waypoints_y', [0.0, 0.5, 0.3, 0.9, 0.5, 0.0])
        self.declare_parameter('lookahead_dist', 0.35)
        self.declare_parameter('max_vel', 0.18)
        self.declare_parameter('max_omega', 2.0)
        self.declare_parameter('control_hz', 20)
        self.declare_parameter('goal_tolerance', 0.12)
        self.declare_parameter('max_vel_traj', 0.20)
        self.declare_parameter('accel', 0.08)
        self.declare_parameter('num_path_points', 400)

        # ── Retrieve Parameters ─────────────────────────────────────
        wx = self.get_parameter('waypoints_x').value
        wy = self.get_parameter('waypoints_y').value
        
        # Zip waypoints ensuring matching lengths
        if len(wx) == len(wy) and len(wx) > 1:
            waypoints = list(zip(wx, wy))
        else:
            self.get_logger().warn("Invalid waypoints in params. Using defaults.")
            waypoints = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.3), (1.5, 0.9), (2.0, 0.5), (2.5, 0.0)]
        
        self.hz = self.get_parameter('control_hz').value
        self.goal_tol = self.get_parameter('goal_tolerance').value

        # ── Publishers ──────────────────────────────────────────────
        self.cmd_pub      = self.create_publisher(Twist, '/cmd_vel', 10)
        self.smooth_pub   = self.create_publisher(Path, '/smooth_path', 10)
        self.actual_pub   = self.create_publisher(Path, '/actual_path', 10)
        self.ref_traj_pub = self.create_publisher(Path, '/ref_trajectory', 10)
        self.cte_pub      = self.create_publisher(Float32, '/cross_track_error', 10)
        self.wp_marker_pub= self.create_publisher(MarkerArray, '/waypoints_marker', 10)

        # ── Subscriber ──────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)

        # ── Internal state ───────────────────────────────────────────
        self._pose:        tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._traj_idx:    int   = 0
        self._actual_path: list  = []
        self._cte_log:     list  = []
        self._vel_log:     list  = []
        self._done:        bool  = False

        # ── Build path and trajectory (Tasks 1 & 2) ─────────────────
        self.get_logger().info('Building smooth path...')
        path = smooth_path(waypoints, num_points=self.get_parameter('num_path_points').value)

        self.get_logger().info('Generating trajectory...')
        self._traj = generate_trajectory(
            path, 
            max_vel=self.get_parameter('max_vel_traj').value, 
            accel=self.get_parameter('accel').value
        )

        self._ctrl = PurePursuitController(
            lookahead_dist=self.get_parameter('lookahead_dist').value,
            max_vel=self.get_parameter('max_vel').value,
            max_omega=self.get_parameter('max_omega').value,
        )

        self.get_logger().info(
            f'Ready. {len(self._traj)} trajectory points. '
            f'Estimated duration: {self._traj[-1].t:.1f}s'
        )

        # Publish reference paths once for RViz visualisation
        self._publish_smooth_path(path)
        self._publish_ref_trajectory()
        self._publish_waypoints(waypoints)

        # ── Control loop (Task 3) ──────────────────────────────
        self.create_timer(1.0 / self.hz, self._control_loop)

    # ── Callbacks ────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # Quaternion → yaw (rotation around Z axis)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw       = math.atan2(siny_cosp, cosy_cosp)

        self._pose = (p.x, p.y, yaw)

    def _control_loop(self) -> None:
        if self._done:
            return

        # Check goal reached
        dist_to_goal = math.hypot(
            self._pose[0] - self._traj[-1].x,
            self._pose[1] - self._traj[-1].y,
        )
        if self._traj_idx >= len(self._traj) - 1 or dist_to_goal < self.goal_tol:
            self._stop_robot()
            self._done = True
            self._save_logs()
            self.get_logger().info(
                f'Goal reached! Mean CTE: {self._mean_cte():.4f} m'
            )
            return

        # Find lookahead point and compute commands
        self._traj_idx, la_pt = self._ctrl.find_lookahead_point(
            self._pose, self._traj, self._traj_idx)

        ref_speed = self._traj[self._traj_idx].v
v, omega = self._ctrl.compute_velocity_commands(self._pose, la_pt, ref_speed=ref_speed)

        # Publish velocity command
        cmd           = Twist()
        cmd.linear.x  = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

        # Compute and publish cross-track error
        cte = CrossTrackErrorMonitor.compute(self._pose, self._traj)
        self._cte_log.append(abs(cte))
        self._vel_log.append(v)
        self._actual_path.append(self._pose[:2])

        cte_msg       = Float32()
        cte_msg.data  = float(abs(cte))
        self.cte_pub.publish(cte_msg)

        # Publish actual path for RViz
        self._publish_actual_path()

    # ── Helper methods ────────────────────────────────────────────────

    def _stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def _mean_cte(self) -> float:
        if not self._cte_log:
            return 0.0
        return sum(self._cte_log) / len(self._cte_log)

    def _publish_smooth_path(self, path) -> None:
        msg = self._make_path_msg(
            [(float(p[0]), float(p[1])) for p in path])
        self.smooth_pub.publish(msg)

    def _publish_ref_trajectory(self) -> None:
        msg = self._make_path_msg(
            [(p.x, p.y) for p in self._traj])
        self.ref_traj_pub.publish(msg)

    def _publish_actual_path(self) -> None:
        msg = self._make_path_msg(self._actual_path[-500:])
        self.actual_pub.publish(msg)

    def _make_path_msg(self, points: list) -> Path:
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in points:
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.header.stamp = msg.header.stamp
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0  # Crucial: Without a valid quaternion, RViz refuses to draw the path line!
            msg.poses.append(ps)
        return msg

    def _publish_waypoints(self, waypoints: list) -> None:
        msg = MarkerArray()
        for i, (x, y) in enumerate(waypoints):
            item = Marker()
            item.header.frame_id = 'odom'
            item.ns = 'waypoints'
            item.id = i
            item.type = Marker.CYLINDER
            item.action = Marker.ADD
            item.pose.position.x = float(x)
            item.pose.position.y = float(y)
            item.pose.position.z = 0.05
            item.scale.x = 0.1
            item.scale.y = 0.1
            item.scale.z = 0.1
            item.color.a = 1.0
            item.color.r = 1.0
            item.color.g = 1.0
            item.color.b = 0.0
            msg.markers.append(item)
        self.wp_marker_pub.publish(msg)

    def _save_logs(self) -> None:
        log_dir = os.path.expanduser('~/trajectory_results')
        os.makedirs(log_dir, exist_ok=True)

        # CTE log
        with open(os.path.join(log_dir, 'cte.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(
                [[round(i / self.hz, 3), v] for i, v in enumerate(self._cte_log)])

        # Velocity log
        with open(os.path.join(log_dir, 'velocity.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(
                [[round(i / self.hz, 3), v] for i, v in enumerate(self._vel_log)])

        # Full trajectory
        with open(os.path.join(log_dir, 'trajectory.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['i', 'x', 'y', 't', 'v'])
            for i, p in enumerate(self._traj):
                w.writerow([i, p.x, p.y, p.t, p.v])

        self.get_logger().info(f'Logs successfully saved to {log_dir}')

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
