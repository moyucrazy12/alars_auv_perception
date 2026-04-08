from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = 'alars_auv_perception'

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='M350'
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true'
    )
    model_file_arg = DeclareLaunchArgument(
        'model_file',
        default_value='yolo_model_4cls.pt'
    )

    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=''
    )
    bag_rate_arg = DeclareLaunchArgument(
        'bag_rate',
        default_value='1.0'
    )
    bag_loop_arg = DeclareLaunchArgument(
        'bag_loop',
        default_value='false'
    )
    bag_start_paused_arg = DeclareLaunchArgument(
        'bag_start_paused',
        default_value='false'
    )

    # Optional: remap image topic from bag -> detector expected topic
    # Example:
    #   /camera/image_raw:=/M350/camera/image_raw
    bag_remap_arg = DeclareLaunchArgument(
        'bag_remap',
        default_value=''
    )

    namespace = LaunchConfiguration('namespace')
    device = LaunchConfiguration('device')
    use_sim_time = LaunchConfiguration('use_sim_time')
    model_file = LaunchConfiguration('model_file')

    bag_path = LaunchConfiguration('bag_path')
    bag_rate = LaunchConfiguration('bag_rate')
    bag_loop = LaunchConfiguration('bag_loop')
    bag_start_paused = LaunchConfiguration('bag_start_paused')
    bag_remap = LaunchConfiguration('bag_remap')

    detection_config = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'parameters',
        'detection_parameters.yaml'
    ])

    model_path = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'models',
        model_file
    ])

    detector_node = Node(
        package=package_name,
        executable='alars_yolo_detector',
        namespace=namespace,
        output='screen',
        parameters=[
            detection_config,
            {
                'namespace': namespace,
                'device': device,
                'use_sim_time': use_sim_time,
                'model_path': model_path,
            }
        ],
    )

    # Basic rosbag play
    # If your bag topic names already match what the detector expects,
    # leave bag_remap empty.
    bag_play_cmd = [
        'ros2', 'bag', 'play',
        bag_path,
        '--clock',
        '--rate', bag_rate,
    ]

    bag_play = ExecuteProcess(
        cmd=bag_play_cmd,
        output='screen',
    )

    # Optional helpers for loop / start paused / remap
    bag_play_loop = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play',
            bag_path,
            '--clock',
            '--rate', bag_rate,
            '--loop',
        ],
        condition=IfCondition(bag_loop),
        output='screen',
    )

    bag_play_paused = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play',
            bag_path,
            '--clock',
            '--rate', bag_rate,
            '--start-paused',
        ],
        condition=IfCondition(bag_start_paused),
        output='screen',
    )

    return LaunchDescription([
        namespace_arg,
        device_arg,
        use_sim_time_arg,
        model_file_arg,
        bag_path_arg,
        bag_rate_arg,
        bag_loop_arg,
        bag_start_paused_arg,
        bag_remap_arg,

        LogInfo(msg=['[Launch] namespace = ', namespace]),
        LogInfo(msg=['[Launch] device = ', device]),
        LogInfo(msg=['[Launch] use_sim_time = ', use_sim_time]),
        LogInfo(msg=['[Launch] model path = ', model_path]),
        LogInfo(msg=['[Launch] bag path = ', bag_path]),
        LogInfo(msg=['[Launch] bag rate = ', bag_rate]),
        LogInfo(msg=['[Launch] bag loop = ', bag_loop]),
        LogInfo(msg=['[Launch] bag start paused = ', bag_start_paused]),

        detector_node,

        # Use only one of these in practice.
        # Default normal play:
        bag_play,

        # Optional variants:
        # bag_play_loop,
        # bag_play_paused,
    ])