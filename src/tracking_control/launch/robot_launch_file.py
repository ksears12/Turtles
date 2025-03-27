from launch import LaunchDescription
from launch_ros.actions import Node 
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    object_detection_pkg = 'object_detection'
    tracking_pkg = 'tracking_control'
    obj_detection_package_path = get_package_share_directory(object_detection_pkg)
    tracking_package_path = get_package_share_directory(tracking_pkg)
    cam_package_path = get_package_share_directory('astra_camera')
    yahboomcar_package_path = get_package_share_directory('yahboomcar_bringup')
    
    obj_detection_node = Node(
        package=object_detection_pkg,
        executable='color_obj_detection',
        name='color_obj_detection_node',
        parameters=[
            {'color_low': [0, 50, 150]},{'color_high': [100, 255, 255]}, {'object_size_min':200}
        ],
        output="screen"
    )

    goal_detection_node = Node(
        package=object_detection_pkg,
        executable='color_goal_detection',
        name='color_goal_detection_node',
        parameters=[
            {'color_low': [0, 150, 50]},{'color_high': [120, 255, 255]}, {'object_size_min':200}
        ],
        output="screen"
    )

    tracking_control_node = Node(
        package=tracking_pkg,
        executable='tracking_node',
        name='tracking_node',
        output="screen"
    )

    motor_control_node = Node(
        package=yahboomcar_package_path,
        executable='Mcnamu_driver_X3',
        name='motor_control_node',
        output="screen"
    )

    astra_camera_launch = IncludeLaunchDescription(XMLLaunchDescriptionSource(
        [os.path.join(cam_package_path, 'launch'),
         '/astra_pro.launch.xml'])
    )
    yahboomcar_brinup_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        [os.path.join(yahboomcar_package_path, 'launch'),
         '/yahboomcar_bringup_X3_launch.py'])
    )
    
    return LaunchDescription([
        obj_detection_node,
        goal_detection_node,
        tracking_control_node,
        astra_camera_launch,
        yahboomcar_brinup_launch
    ])
