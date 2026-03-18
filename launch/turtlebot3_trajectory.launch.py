from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
def generate_launch_description():
    sim = Node(package='trajectory_control',executable='robot_simulator',name='robot_simulator',output='screen')
    ctrl = TimerAction(period=1.0,actions=[Node(package='trajectory_control',executable='main_node',name='trajectory_control_node',output='screen')])
    rviz = TimerAction(period=2.0,actions=[Node(package='rviz2',executable='rviz2',name='rviz2',output='screen')])
    return LaunchDescription([sim,ctrl,rviz])
