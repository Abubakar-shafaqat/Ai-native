---
sidebar_position: 3
---

# 3. Running Your First Robot Models in Gazebo

Now that you have Gazebo installed and can launch a basic world, let's bring a robot into the simulation. This tutorial will guide you through spawning a simple robot model and visualizing its state.

## 3.1 Understanding Robot Description Formats

Before we can spawn a robot, it's essential to understand how robots are digitally described. ROS 2 primarily uses **URDF (Unified Robot Description Format)**, which defines the robot's physical structure, joints, and visual/collision properties (as discussed in Chapter 8). Gazebo often uses **SDF (Simulation Description Format)**, which is more comprehensive and can describe entire worlds. For ROS 2, URDF models are typically converted to SDF for use in Gazebo.

:::tip Diagram Placeholder
**Diagram 3.1: Robot Description Workflow**
A flowchart showing:
-   "Robot Design (CAD)" -> "URDF File (.urdf)"
-   "URDF File" -> "Conversion Tool" -> "SDF File (.sdf)"
-   "SDF File" -> "Gazebo Simulator"
-   "URDF File" -> "robot_state_publisher" -> "RViz2"
Arrows indicate the flow of description from design to simulation and visualization.
:::

## 3.2 Spawning a Simple Robot in Gazebo

For this tutorial, we will use a pre-existing simple differential drive robot, commonly available as part of ROS 2 examples or generic robot packages. We will spawn this robot into an empty Gazebo world using a ROS 2 launch file.

### 3.2.1 Create a ROS 2 Workspace and Package

If you haven't already, create a simple ROS 2 workspace and a basic package to store your launch files.

1.  Open a terminal.
2.  Create a workspace: `mkdir -p ~/ros2_ws/src`
3.  Navigate to the source directory: `cd ~/ros2_ws/src`
4.  Create a new ROS 2 Python package: `ros2 pkg create my_robot_bringup --build-type ament_python`
5.  Navigate back to your workspace root: `cd ~/ros2_ws`
6.  Build the empty package: `colcon build`
7.  Source your workspace: `source install/setup.bash`

### 3.2.2 Launch a Robot in Gazebo

Now, we'll create a launch file to bring up Gazebo with a simple robot. This typically involves launching Gazebo itself, and then using `ros_gz_sim` to spawn a URDF model.

1.  Navigate to your package's launch directory: `cd ~/ros2_ws/src/my_robot_bringup/launch`
2.  Create a new Python launch file (e.g., `spawn_diff_drive.launch.py`). *Content for this file is shown conceptually below, as direct code is avoided.*

    *   This launch file would:
        *   Include the Gazebo simulation launch (from `ros_gz_sim`).
        *   Define the URDF content for a simple differential drive robot (or load it from an existing `robot_description` parameter).
        *   Use the `ros_gz_sim` `create` entity service to spawn the robot's SDF representation into Gazebo.

3.  Run the launch file: `ros2 launch my_robot_bringup spawn_diff_drive.launch.py`

You should see a Gazebo window open with your robot model present in the world. It will likely just be sitting there, as we haven't commanded its motion yet.

:::info Diagram Placeholder
**Diagram 3.2: Spawning Robot in Gazebo Workflow**
A conceptual diagram showing:
-   "ROS 2 Launch File" (Python script icon).
-   This launch file's output goes to:
    -   "Gazebo Sim" (launches simulator).
    -   "ros_gz_sim create service" (spawns robot model).
-   "Robot Model" appears in "Gazebo 3D Viewport."
:::

## 3.3 Visualizing Robot State with RViz 2

While Gazebo provides a 3D visualization, RViz 2 is an indispensable tool in ROS 2 for visualizing sensor data, robot states, and planning information. We will use `robot_state_publisher` to display our robot in RViz 2.

### 3.3.1 Understanding `robot_state_publisher`

The `robot_state_publisher` is a ROS 2 node that reads the robot's URDF description and its current joint states (typically published by either hardware drivers or a simulation) and then publishes the 3D transformations (TF messages) between all the robot's links. RViz 2 uses these transformations to display the robot's correct pose and configuration.

### 3.3.2 Launch RViz 2 and Visualize Your Robot

1.  In a **new terminal**, ensure your workspace is sourced: `source ~/ros2_ws/install/setup.bash`
2.  Launch RViz 2: `rviz2`
3.  In RViz 2:
    *   **Add** a "RobotModel" display.
    *   Ensure the "Fixed Frame" is set to the robot's base frame (e.g., `base_link`).
4.  In another **new terminal**, ensure your workspace is sourced.
5.  Launch the `robot_state_publisher` node. *This would typically be included in your `spawn_diff_drive.launch.py` file, but for demonstration, we assume it's launched separately for now.*

    *   This node needs the robot's URDF description. The URDF can be passed as a parameter or loaded into the `robot_description` ROS 2 parameter server.

Once `robot_state_publisher` is running and publishing TF transforms, your robot model should appear in RViz 2. Initially, its joints might not move. For movement, we'd need a node publishing joint states (e.g., from simulation or hardware) and commands.

:::info Diagram Placeholder
**Diagram 3.3: Robot Visualization with RViz 2**
A diagram showing:
-   "Gazebo Sim" -> "Joint State ROS 2 Topic" (from simulated robot).
-   "Joint State ROS 2 Topic" -> "robot_state_publisher Node."
-   "robot_state_publisher Node" -> "TF Transforms ROS 2 Topic."
-   "TF Transforms ROS 2 Topic" -> "RViz 2" (displaying the robot's 3D model).
-   A conceptual screenshot of RViz 2 showing a robot model.
:::

You have successfully spawned a robot in Gazebo and visualized its static representation in RViz 2! This sets the stage for controlling your robot and reading its sensors.
