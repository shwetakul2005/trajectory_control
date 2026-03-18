import os
base = "/home/ubuntu/Desktop/ros2_ws/src/trajectory_control"
os.makedirs(base+"/launch", exist_ok=True)
os.makedirs(base+"/config", exist_ok=True)
open(base+"/setup.py","w").write("""from setuptools import setup
from glob import glob
import os
package_name = 'trajectory_control'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/'+package_name]),
        ('share/'+package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'), glob('launch/*.py')),
        (os.path.join('share',package_name,'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Trajectory tracking robot',
    license='MIT',
    entry_points={'console_scripts': ['main_node = trajectory_control.main_node:main','robot_simulator = trajectory_control.robot_simulator:main']},
)
""")
open(base+"/launch/turtlebot3_trajectory.launch.py","w").write("""from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
def generate_launch_description():
    sim = Node(package='trajectory_control',executable='robot_simulator',name='robot_simulator',output='screen')
    ctrl = TimerAction(period=1.0,actions=[Node(package='trajectory_control',executable='main_node',name='trajectory_control_node',output='screen')])
    rviz = TimerAction(period=2.0,actions=[Node(package='rviz2',executable='rviz2',name='rviz2',output='screen')])
    return LaunchDescription([sim,ctrl,rviz])
""")
print("ALL FILES WRITTEN OK")
print("setup.py:", open(base+"/setup.py").read()[:50])
print("launch:", os.listdir(base+"/launch"))
