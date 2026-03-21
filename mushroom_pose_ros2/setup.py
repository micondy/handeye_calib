from setuptools import setup

package_name = 'mushroom_pose_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/mushroom_pose.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='winnnnnn',
    maintainer_email='winnnnnn@local',
    description='Mushroom pose pipeline integration for lododo arm grasping.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mushroom_pose_node = mushroom_pose_ros2.mushroom_pose_node:main',
        ],
    },
)
