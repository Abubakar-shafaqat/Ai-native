---
sidebar_position: 2
---

# 2. Basics of Gazebo Simulation

Gazebo is a powerful 3D simulator for robotics, allowing you to develop and test robot algorithms in a virtual environment. This tutorial will guide you through installing Gazebo and launching a basic simulation world.

## 2.1 Installing Gazebo and ROS 2 Bridge

Before launching Gazebo, ensure it's installed and connected to your ROS 2 environment. We'll use the `ros_gz_sim` bridge for integration.

### 2.1.1 Install Gazebo Sim (Fortress)

We'll install Gazebo Sim, specifically the Fortress distribution which is compatible with ROS 2 Iron Irwini (as a common choice).

1.  Open a terminal (ensure your ROS 2 environment is sourced from the previous tutorial).
2.  Add the Gazebo repository: `sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-noble-amd64 noble main" > /etc/apt/sources.list.d/gazebo-latest.list'`
3.  Add the GPG key: `wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -`
4.  Update your package list: `sudo apt update`
5.  Install Gazebo Fortress: `sudo apt install gazebo-fortress`

### 2.1.2 Install ROS 2 to Gazebo Bridge

Next, install the bridge packages that allow ROS 2 to communicate with Gazebo.

1.  Install the necessary bridge packages (replace `IRON` with your ROS 2 distribution, e.g., `humble` or `iron`): `sudo apt install ros-<ROS2_DISTRO>-ros-gz-sim ros-<ROS2_DISTRO>-ros-gz-sim-demos`

These steps ensure Gazebo and its ROS 2 bridge are ready on your system.

:::tip Diagram Placeholder
**Diagram 2.1: ROS 2 + Gazebo Integration Overview**
A block diagram showing:
-   **Block 1: ROS 2 Environment** (with `ros2` commands, nodes)
-   **Block 2: ROS-Gazebo Bridge** (`ros_gz_sim`)
-   **Block 3: Gazebo Sim** (physics, rendering, `gz` commands)
Arrows indicate bidirectional communication between ROS 2 and Gazebo through the bridge.
:::

## 2.2 Launching Your First Gazebo World

Now, let's launch a simple empty world in Gazebo. This will verify your installation and show you the basic Gazebo user interface.

### 2.2.1 Launch an Empty Gazebo World

You can launch Gazebo directly using the `gz` command or through a ROS 2 launch file for better integration. We'll start with the direct approach.

1.  Open a terminal.
2.  Source your ROS 2 environment: `source /opt/ros/<ROS2_DISTRO>/setup.bash`
3.  Launch an empty Gazebo world: `gz sim empty.sdf`

A Gazebo window should appear, showing an empty 3D environment with a ground plane. You can use your mouse to navigate the view:
*   **Orbit:** Left-click and drag.
*   **Pan:** Right-click and drag.
*   **Zoom:** Scroll wheel.

### 2.2.2 Understanding the Gazebo UI

The Gazebo GUI provides several panels and tools for interacting with your simulation:

*   **3D Viewport:** The main area where you visualize your world, robots, and objects.
*   **World Tree (Left Panel):** Lists all entities present in your world (e.g., `ground_plane`, `sun`). You can select entities here to view their properties.
*   **Entity Properties (Right Panel):** Shows detailed information and allows you to modify properties of selected entities (e.g., position, scale, physics properties).
*   **Toolbar (Top):** Contains tools for manipulating objects (move, rotate, scale), adding primitive shapes (box, sphere, cylinder), and controlling the simulation (play/pause).

Familiarize yourself with these controls; they will be essential for creating and debugging your robotic simulations.

:::info Diagram Placeholder
**Diagram 2.2: Gazebo GUI Layout**
A screenshot-like diagram of the Gazebo GUI. Label the main components:
-   "3D Viewport" (center)
-   "World Tree" (left panel)
-   "Entity Properties" (right panel)
-   "Toolbar" (top)
-   Highlight controls for orbiting, panning, and zooming.
:::

## 2.3 Exploring Gazebo Assets

Gazebo comes with a rich set of pre-built models and environments. You can easily add these to your simulation.

### 2.3.1 Adding a Simple Object

Let's add a simple box to our empty world.

1.  With your `empty.sdf` world running, locate the "Insert" tab in the left-hand "World Tree" panel.
2.  Expand the `GazeboModels` list.
3.  Click and drag a "Cube" model from the list directly into your 3D viewport.
4.  You can then use the move and rotate tools (from the top toolbar) to position your cube.

You've successfully installed Gazebo and launched your first world with an object! This foundation is crucial for the next steps in bringing robots into your simulation.
