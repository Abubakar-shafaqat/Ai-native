---
sidebar_position: 4
---

# 4. Simple Sensor Readings and Actuations

With your robot model running in Gazebo and its state visualizable in RViz 2, the next crucial step is to interact with it: reading its simulated sensor data and sending commands to control its movements. This tutorial will demonstrate basic sensor subscription and actuator publishing using ROS 2 command-line tools.

## 4.1 Understanding ROS 2 Topics for Sensors and Actuators

As introduced in Chapter 4, ROS 2 uses a publish/subscribe model for asynchronous data exchange. Simulated sensors in Gazebo publish their data to specific ROS 2 topics, and actuators (like robot wheels or joints) subscribe to command topics to receive instructions.

### 4.1.1 Common Sensor Topics

When a robot model is spawned in Gazebo with `ros_gz_sim` bridge, it typically publishes various sensor data. Common topics include:

*   `/scan`: LiDAR or 2D laser scan data (type: `sensor_msgs/msg/LaserScan`).
*   `/odom`: Odometry data, representing the robot's pose and velocity relative to its starting point (type: `nav_msgs/msg/Odometry`).
*   `/tf`: Transform frames, providing the relationships between different parts of the robot and the world (type: `tf2_msgs/msg/TFMessage`).
*   `/camera/image_raw`: Raw image data from a simulated camera (type: `sensor_msgs/msg/Image`).

You can discover the active topics in your running ROS 2 system using `ros2 topic list`.

### 4.1.2 Common Actuator Command Topics

For mobile robots, the primary way to command movement is often through a `geometry_msgs/msg/Twist` message published to a velocity command topic, commonly `/cmd_vel`.

*   `/cmd_vel`: Publishes linear (forward/backward) and angular (turning) velocities for a differential drive robot.

## 4.2 Reading Sensor Data

Let's read some sensor data from your robot running in Gazebo. Ensure your Gazebo simulation with your robot model is still running from the previous tutorial.

### 4.2.1 Inspecting Available Topics

1.  Open a new terminal.
2.  Source your ROS 2 environment: `source /opt/ros/<ROS2_DISTRO>/setup.bash`
3.  List all active topics: `ros2 topic list`
    *   You should see topics like `/scan`, `/odom`, `/tf`, and potentially others depending on your robot model's sensors.
4.  To see the type of message published on a topic: `ros2 topic info /scan` (replace `/scan` with any topic you want to inspect).

### 4.2.2 Subscribing to Sensor Topics

Now, let's view the actual data being published on a sensor topic.

1.  In the same terminal, echo the contents of the `/odom` topic: `ros2 topic echo /odom`
    *   You will see a continuous stream of JSON-like messages, each containing the robot's estimated position (x, y, z), orientation (quaternion), and linear/angular velocities. Since the robot is currently static, these values should remain constant (or close to zero).
2.  Try echoing the `/scan` topic: `ros2 topic echo /scan`
    *   This will show you the simulated laser scan data, including ranges to detected objects around the robot. If your robot is in an empty world, the ranges might be consistently high (to the edge of the world).

:::info Diagram Placeholder
**Diagram 4.1: Sensor Data Flow**
A flowchart showing:
-   "Simulated LiDAR in Gazebo" -> "ROS 2 Topic `/scan`"
-   "Simulated Odometry in Gazebo" -> "ROS 2 Topic `/odom`"
-   "ROS 2 Topic `/scan`" -> "Terminal running `ros2 topic echo /scan`" (showing data output)
-   "ROS 2 Topic `/odom`" -> "Terminal running `ros2 topic echo /odom`" (showing data output)
:::

## 4.3 Sending Actuator Commands

Now, let's make your robot move by publishing velocity commands.

### 4.3.1 Publishing a Twist Message

We'll publish a `geometry_msgs/msg/Twist` message to the `/cmd_vel` topic. This message contains two vectors: `linear` for forward/backward and side-to-side motion, and `angular` for rotational motion.

1.  Open a new terminal.
2.  Source your ROS 2 environment: `source /opt/ros/<ROS2_DISTRO>/setup.bash`
3.  Publish a forward velocity command: `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'`
    *   Your robot in Gazebo should move forward briefly. The `--once` flag means it publishes the message only once.
4.  Publish a continuous rotational command (keep this terminal open): `ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}'`
    *   Your robot should now continuously rotate in Gazebo. To stop it, press `Ctrl+C` in the terminal where you published the command, and then publish a zero velocity: `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'`

### 4.3.2 Observing Robot Motion and Odometry

While your robot is moving, observe its behavior in Gazebo. If you still have the terminal echoing `/odom` open, you will now see its `pose.pose.position.x` and `pose.pose.orientation.z` values changing as it moves and rotates.

:::info Diagram Placeholder
**Diagram 4.2: Actuator Command Flow**
A flowchart showing:
-   "Terminal running `ros2 topic pub /cmd_vel`" -> "ROS 2 Topic `/cmd_vel`"
-   "ROS 2 Topic `/cmd_vel`" -> "Simulated Actuators in Gazebo"
-   "Simulated Actuators in Gazebo" -> "Robot Movement in 3D Viewport."
-   A conceptual screenshot of the Gazebo 3D Viewport showing the robot moving.
:::

You have successfully learned how to inspect sensor data and command your robot's basic movements using ROS 2 command-line tools. This forms the basis for developing more sophisticated control algorithms.
