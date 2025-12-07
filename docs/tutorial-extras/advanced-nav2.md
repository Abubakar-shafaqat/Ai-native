---
sidebar_position: 2
---

# 2. Advanced Navigation with Nav2

Nav2 (introduced in Chapter 13) is the powerful ROS 2 navigation stack that enables mobile robots to autonomously navigate complex environments. This tutorial builds upon basic navigation concepts, guiding you through setting up Nav2 to perform mapping, localization, and goal-driven navigation in a simulated environment using a predefined robot model.

## 2.1 Understanding the Nav2 Workflow

Nav2 operates using a modular architecture with several key components working together. For goal-driven navigation, the typical workflow is:

1.  **Mapping:** Creating a 2D occupancy grid map of the environment (often using SLAM algorithms).
2.  **Localization:** Determining the robot's pose within that map (e.g., using AMCL).
3.  **Path Planning:** Generating a global path to the goal and a local trajectory to follow it while avoiding obstacles.
4.  **Control:** Sending velocity commands to the robot to execute the planned trajectory.

**Why Advanced Nav2?** While basic navigation might simply involve moving from point A to B, advanced Nav2 configuration allows for:
*   **Dynamic Obstacle Avoidance:** Handling moving obstacles.
*   **Complex Environments:** Navigating cluttered or narrow spaces.
*   **Recovery Behaviors:** Responding gracefully to unexpected situations (e.g., getting stuck).
*   **Parameter Tuning:** Optimizing navigation performance for specific robots and tasks.

## 2.2 Setting Up Nav2 for a Simulated Robot

For this tutorial, we will assume you have a differential drive robot model running in Gazebo (from Tutorial 3) that publishes `/odom`, `/imu`, and `/scan` topics (as seen in Tutorial 4).

### 2.2.1 Install Nav2 Packages

Nav2 is installed as part of the ROS 2 desktop full installation. If you installed a minimal ROS 2 setup, ensure you install Nav2:

1.  Open a terminal.
2.  Source your ROS 2 environment: `source /opt/ros/<ROS2_DISTRO>/setup.bash`
3.  Install Nav2: `sudo apt install ros-<ROS2_DISTRO>-navigation2 ros-<ROS2_DISTRO>-nav2-bringup`

### 2.2.2 Create a Map

Before autonomous navigation, your robot needs a map of its environment. We'll use a simulated environment (e.g., a simple warehouse world) and `slam_toolbox` to create a map.

1.  **Terminal 1:** Launch your Gazebo world with the robot. (e.g., `ros2 launch ros_gz_sim_demos diff_drive_robot.launch.py`). This launch might also start `ros_gz_bridge` to publish sensor data.
2.  **Terminal 2:** Launch `slam_toolbox` (used for Simultaneous Localization and Mapping - SLAM): `ros2 launch slam_toolbox online_async_launch.py params_file:=/path/to/my/slam_toolbox_params.yaml`
    *   *Note: You would typically create a `slam_toolbox_params.yaml` file in your package's config directory for specific settings. For a simple start, `slam_toolbox` may work with default parameters if your robot publishes `/scan` and `/tf` correctly.*
3.  **Terminal 3:** Launch RViz 2: `rviz2 -d /path/to/my/nav2_rviz_config.rviz`
    *   *Note: Create an RViz config for Nav2. In RViz, add "Map" (topic `/map`), "LaserScan" (topic `/scan`), "RobotModel", and "TF" displays.*
4.  **Drive the robot:** Use your keyboard or `ros2 topic pub` (as in Tutorial 4) to manually drive your robot around the Gazebo world. As the robot moves, you should see the map being built in the "Map" display in RViz 2.
5.  **Save the map:** Once you've explored the area, save the map: `ros2 run nav2_map_server map_saver_cli -f my_map`
    *   This creates `my_map.yaml` and `my_map.pgm` in your current directory. Move these to your `my_robot_bringup/maps` directory.

:::info Diagram Placeholder
**Diagram 2.1: Mapping with SLAM Toolbox Workflow**
A flowchart showing:
-   "Simulated Robot in Gazebo" -> "LiDAR Sensor (publishes /scan)"
-   "LiDAR Sensor" + "Robot Odometry (publishes /odom and /tf)" -> "SLAM Toolbox Node"
-   "SLAM Toolbox Node" -> "Map Topic (/map)" + "Localized Pose Topic (/tf)"
-   "Map Topic" -> "RViz 2 Map Display"
-   "Save Map CLI Tool" -> "my_map.pgm & my_map.yaml"
:::

## 2.3 Running Nav2 for Autonomous Navigation

Now that you have a map, you can use Nav2 to perform autonomous navigation.

### 2.3.1 Create a Nav2 Bringup Launch File

Create a launch file (e.g., `nav2_bringup.launch.py`) in your `my_robot_bringup/launch` directory. This file will bring up your robot, the Nav2 stack, and load your created map.

    *This launch file would:*
    *   Include your robot's Gazebo launch.
    *   Include the main Nav2 launch file (`nav2_bringup.launch.py` from the `nav2_bringup` package).
    *   Pass parameters to Nav2, including the path to your saved map (`my_map.yaml`).
    *   Set various Nav2 parameters (e.g., global planner, local planner, controller types) via a YAML config file (e.g., `nav2_params.yaml`).

### 2.3.2 Launch Nav2 and Send a Goal

1.  **Terminal 1:** Launch your comprehensive Nav2 setup: `ros2 launch my_robot_bringup nav2_bringup.launch.py`
    *   This should open Gazebo with your robot, load your map into Nav2, and start all the Nav2 nodes.
2.  **Terminal 2:** Launch RViz 2 (if not already launched by the previous command) with your Nav2 config.
    *   In RViz 2, ensure you have "Map," "RobotModel," "TF," and "Nav2" plugins (e.g., Global Plan, Local Plan, Costmap) added and configured.
3.  **Set Initial Pose:** In RViz 2, use the "2D Pose Estimate" tool (usually an arrow icon) to tell Nav2 where your robot is on the map. Click and drag an arrow on the map to indicate the robot's approximate position and orientation.
4.  **Set Navigation Goal:** Use the "2D Goal Pose" tool (another arrow icon) to set a target location for your robot on the map.

Your robot should now start autonomously navigating towards the goal, planning its path, avoiding obstacles, and updating its localization in RViz 2.

:::info Diagram Placeholder
**Diagram 2.2: Nav2 Autonomous Navigation Flow**
A flowchart showing:
-   "User sets 2D Pose Estimate in RViz 2" -> "AMCL Node (Localization)"
-   "AMCL Node" + "Map Server Node" -> "Global Planner"
-   "Global Planner" -> "Global Path"
-   "Global Path" + "Sensor Data" -> "Local Planner (Controller)"
-   "Local Planner" -> "Robot Base Controller (Velocity Commands)" -> "Simulated Robot in Gazebo"
-   "User sets 2D Goal Pose in RViz 2" -> "Nav2 Behavior Tree" (orchestrates the planning)
:::

## 2.4 Nav2 Parameter Tuning and Advanced Features

Nav2 is highly configurable. Its performance heavily depends on parameter tuning to match your robot's kinematics, sensor characteristics, and the environment.

### 2.4.1 Key Parameters to Tune

*   **Costmap Parameters:** `inflation_radius`, `obstacle_range`, `raytrace_range`. These affect how obstacles are perceived and how far the robot tries to avoid them.
*   **Global Planner Parameters:** `tolerance`, `allow_unknown`. These affect path optimality and willingness to explore.
*   **Local Planner (Controller) Parameters:** `min_vel_x`, `max_vel_x`, `max_rot_vel`, `acc_lim_x`, `acc_lim_theta`, `yaw_goal_tolerance`, `xy_goal_tolerance`. These directly control the robot's speed, acceleration, and precision in reaching the goal.
*   **Recovery Behaviors:** Nav2's behavior tree allows for custom recovery strategies. You can enable/disable default ones (e.g., `spin_in_place`, `backup`) or create your own.

**Possible Pitfalls:**
*   **Over-tuning:** Too aggressive parameters can make the robot unstable or prone to collisions.
*   **Under-tuning:** Too conservative parameters can make the robot slow or unable to navigate tight spaces.
*   **Odometry Drift:** If your odometry (from wheel encoders) is drifting significantly, Nav2's localization (AMCL) will struggle. Ensure good sensor fusion (as in Tutorial 1) for a robust `odom` input.

**Further Resources:**
*   Official Nav2 Documentation: [link to docs.ros.org/en/latest/p/nav2/](https://docs.ros.org/en/latest/p/nav2/)
*   Nav2 Configuration Guide: [link to Nav2 specific config tuning guides]

Mastering Nav2's configuration allows you to tailor its behavior for optimal performance across a wide range of autonomous navigation tasks for your physical AI robot.
