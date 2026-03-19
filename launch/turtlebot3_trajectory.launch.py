import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('trajectory_control')
    config = os.path.join(pkg_dir, 'config', 'params.yaml')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'turtlebot3_trajectory.rviz')

    sim = Node(
        package='trajectory_control',
        executable='robot_simulator',
        name='robot_simulator',
        output='screen'
    )
    
    ctrl = TimerAction(
        period=1.0,
        actions=[
            Node(
                package='trajectory_control',
                executable='main_node',
                name='trajectory_control_node',
                parameters=[config],
                output='screen'
            )
        ]
    )
    
    rviz = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([sim, ctrl, rviz])
