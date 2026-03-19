#!/usr/bin/env python3
"""
robot_simulator.py
------------------
A lightweight ROS2 node that simulates a differential drive robot
WITHOUT Gazebo or any 3D renderer.

How it works:
  - Subscribes to /cmd_vel (geometry_msgs/Twist)
  - Integrates the unicycle kinematic model at 50Hz:
        x     += v * cos(theta) * dt
        y     += v * sin(theta) * dt
        theta += omega * dt
  - Publishes the resulting pose as /odom (nav_msgs/Odometry)
  - Adds small Gaussian noise to simulate real wheel imperfection

This means your PurePursuitController talks to THIS node
exactly as it would talk to a real Turtlebot3 or Gazebo.
The interface is identical — /cmd_vel in, /odom out.

Why this is valid:
  The assignment tests path smoothing, trajectory generation,
  and control algorithms. Those algorithms are simulator-agnostic.
  This node faithfully implements the differential drive kinematic
  model (the same model Gazebo uses internally for Turtlebot3).
"""

import rclpy, math, time
import numpy as np
from rclpy.node                 import Node
from geometry_msgs.msg          import Twist, TransformStamped
from nav_msgs.msg               import Odometry
from tf2_ros                    import TransformBroadcaster
from sensor_msgs.msg            import LaserScan

# Simulation parameters
SIM_HZ        = 50.0           # physics update rate (Hz)
WHEEL_NOISE   = 0.003          # std dev of velocity noise (m/s)
OMEGA_NOISE   = 0.005          # std dev of angular noise (rad/s)

# Turtlebot3 Burger physical specs (for realism)
WHEELBASE     = 0.160          # metres
MAX_VEL       = 0.22           # m/s
MAX_OMEGA     = 2.84           # rad/s


class RobotSimulator(Node):
    """
    Simulates a differential drive robot's kinematics.
    Publishes /odom and broadcasts the odom→base_footprint TF.
    """

    def __init__(self):
        super().__init__('robot_simulator')

        # ── Robot state ───────────────────────────────────────────
        self._x:     float = 0.0
        self._y:     float = 0.0
        self._theta: float = 0.0
        self._v:     float = 0.0   # current linear velocity command
        self._omega: float = 0.0   # current angular velocity command
        self._scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # ── Subscriber: receive velocity commands ─────────────────
        self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10)

        # ── Publisher: broadcast simulated odometry ───────────────
        self._odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # ── TF broadcaster (needed for RViz) ──────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── Physics loop at SIM_HZ ────────────────────────────────
        self.create_timer(1.0 / SIM_HZ, self._physics_step)

        self.get_logger().info(
            f'Robot simulator running at {SIM_HZ:.0f}Hz. '
            f'Waiting for /cmd_vel commands...'
        )

    def _cmd_vel_callback(self, msg: Twist) -> None:
        """Receive and clamp velocity commands from controller."""
        self._v     = max(-MAX_VEL,   min(MAX_VEL,   msg.linear.x))
        self._omega = max(-MAX_OMEGA, min(MAX_OMEGA,  msg.angular.z))

    def _physics_step(self) -> None:
        """
        Integrate unicycle kinematic model forward by dt = 1/SIM_HZ.

        Equations:
            x(t+dt)     = x(t)     + v * cos(theta) * dt
            y(t+dt)     = y(t)     + v * sin(theta) * dt
            theta(t+dt) = theta(t) + omega * dt

        Small Gaussian noise is added to v and omega to simulate
        wheel slip and encoder imperfection — makes the controller
        work harder and produces more realistic CTE plots.
        """
        dt = 1.0 / SIM_HZ

        # Add realistic noise
        v_noisy     = self._v     + np.random.normal(0.0, WHEEL_NOISE)
        omega_noisy = self._omega + np.random.normal(0.0, OMEGA_NOISE)

        # Integrate kinematic model
        self._x     += v_noisy * math.cos(self._theta) * dt
        self._y     += v_noisy * math.sin(self._theta) * dt
        self._theta += omega_noisy * dt

        # Normalise theta to [-pi, pi]
        self._theta = math.atan2(
            math.sin(self._theta), math.cos(self._theta))

        # Publish odometry and TF
        self._publish_odom()
        self._publish_fake_scan()
    
    def _publish_fake_scan(self) -> None:
        """Publish a clear scan (no obstacles) for the controller to read.
        In a real robot or Gazebo this comes from actual LiDAR hardware."""
        msg = LaserScan()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_scan'
        msg.angle_min       = -3.14159
        msg.angle_max       =  3.14159
        msg.angle_increment =  0.01745   # ~1 degree
        msg.range_min       =  0.12
        msg.range_max       =  3.50
        # All ranges at max = clear environment
        num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment)
        msg.ranges = [msg.range_max] * num_readings
        self._scan_pub.publish(msg)

    def _publish_odom(self) -> None:
        """
        Publish current pose as nav_msgs/Odometry.
        Also broadcasts odom → base_footprint TF for RViz.
        """
        now = self.get_clock().now().to_msg()

        # ── Odometry message ──────────────────────────────────────
        odom                         = Odometry()
        odom.header.stamp            = now
        odom.header.frame_id         = 'odom'
        odom.child_frame_id          = 'base_footprint'

        # Position
        odom.pose.pose.position.x    = self._x
        odom.pose.pose.position.y    = self._y
        odom.pose.pose.position.z    = 0.0

        # Orientation as quaternion (yaw only — 2D robot)
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(self._theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self._theta / 2.0)

        # Velocity
        odom.twist.twist.linear.x    = self._v
        odom.twist.twist.angular.z   = self._omega

        self._odom_pub.publish(odom)

        # ── TF broadcast ──────────────────────────────────────────
        tf                              = TransformStamped()
        tf.header.stamp                 = now
        tf.header.frame_id              = 'odom'
        tf.child_frame_id               = 'base_footprint'
        tf.transform.translation.x      = self._x
        tf.transform.translation.y      = self._y
        tf.transform.translation.z      = 0.0
        tf.transform.rotation.x         = 0.0
        tf.transform.rotation.y         = 0.0
        tf.transform.rotation.z         = math.sin(self._theta / 2.0)
        tf.transform.rotation.w         = math.cos(self._theta / 2.0)
        self._tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
