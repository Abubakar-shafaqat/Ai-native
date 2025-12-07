---
sidebar_position: 5
---

# Chapter 9: Writing ROS 2 Packages (Python)

## Part A: ROS 2 Package Structure

### I. Understanding Workspace and Packages

In ROS 2, a **workspace** serves as the central directory where you manage and build multiple ROS 2 packages. It's essentially a collection of source packages that you want to develop or modify together. This hierarchical organization allows for efficient compilation and dependency management. Within a workspace, each **package** is a self-contained unit designed to provide a specific set of functionalities, such as sensor drivers, navigation algorithms, or control interfaces. This modular approach promotes reusability, simplifies testing, and facilitates collaborative development among different teams or individuals working on distinct aspects of a complex robotic project. A well-structured package clearly defines its purpose, dependencies, and executables, ensuring it can be easily integrated into larger systems.

:::info Diagram Placeholder
**Diagram 5.1: ROS 2 Workspace and Package Hierarchy**
A tree-like diagram showing a `colcon_ws` (workspace) directory at the top. Inside, there's a `src` folder. Within `src`, multiple package folders are shown (e.g., `my_robot_driver`, `my_navigation_pkg`, `my_perception_pkg`), each containing their own `package.xml`, `setup.py`, and source code directories. Arrows should indicate that `colcon build` operates on the whole workspace.
:::

### II. `package.xml` and `setup.py`

Every ROS 2 Python package requires two crucial configuration files: `package.xml` and `setup.py`.

*   **`package.xml`:** This manifest file provides essential metadata about the package. It defines the package's name, version, description, maintainers, license, and, most importantly, its dependencies. Dependencies are categorized into `build_depends` (for packages needed at compile time) and `exec_depends` (for packages needed at runtime). Proper declaration of these dependencies ensures that `colcon` (the ROS 2 build tool) can correctly resolve and install all necessary components before building or running your package.
*   **`setup.py`:** This Python script is used by `setuptools` to define how the Python components of your package are built, installed, and what executables they provide. It specifies which Python modules belong to the package, any data files to be installed, and entry points for executable scripts (console_scripts). For ROS 2 Python packages, `setup.py` plays a critical role in making your Python nodes discoverable and runnable by the ROS 2 system.

Together, these files ensure that your package is properly integrated into the ROS 2 ecosystem, its dependencies are met, and its functionalities are accessible.

:::tip Reflection Question
Why is it important to clearly declare all dependencies in `package.xml`? What problems might arise if a dependency is missing or incorrect?
:::

### III. Creating a New Python Package

Creating a new ROS 2 Python package is typically done using the `ros2 pkg create` command-line tool. This command streamlines the process by generating the basic directory structure and template files (`package.xml`, `setup.py`) automatically.

When creating a package, you specify its name and, crucially for Python development, the `--build-type ament_python` argument. This build type configures `colcon` to use `setuptools` for building and installing your Python code. After creation, you will find a basic structure including `package.xml` and `setup.py`, and a directory for your Python source code. You will then populate this source directory with your Python scripts that implement ROS 2 nodes. This structured approach helps maintain consistency across different packages and ensures that your development adheres to ROS 2 best practices.

:::bulb Quiz Idea
**Quiz:** Which command would you use to create a new ROS 2 Python package named `my_robot_controller`?
a) `ros2 create package my_robot_controller --language python`
b) `ros2 pkg new my_robot_controller --build-type ament_python`
c) `ros2 pkg create my_robot_controller --build-type ament_python`
d) `colcon create python_package my_robot_controller`
*Correct Answer: c) `ros2 pkg create my_robot_controller --build-type ament_python`*
:::

## Part B: Implementing ROS 2 Nodes in Python

### I. Basic Node Creation (Initialization, Spin)

Every ROS 2 node, regardless of its function, follows a fundamental structure in Python. The process begins with importing the `rclpy` library, which provides the Python client library for ROS 2. A node is then initialized by calling `rclpy.init()`, followed by the creation of a `Node` object, typically by subclassing `rclpy.node.Node`. Inside the node's constructor, you define its name. After the node is created, `rclpy.spin()` is called. This function keeps the node alive and continuously processes callbacks for incoming messages, service requests, or timer events. Without `spin()`, the node would simply initialize and then exit. Finally, `rclpy.shutdown()` is called upon termination to gracefully release all ROS 2 resources. This boilerplate ensures that your Python node can participate in the ROS 2 graph and interact with other nodes.

:::info Diagram Placeholder
**Diagram 5.2: Lifecycle of a Basic ROS 2 Python Node**
A simple flowchart showing:
1. `rclpy.init()`
2. `Node = rclpy.node.Node("node_name")`
3. `rclpy.spin(node)` (with a loop indicating continuous execution)
4. `node.destroy_node()`
5. `rclpy.shutdown()`
:::

### II. Publishers and Subscribers in Python

Implementing the Publish/Subscribe communication model in Python nodes involves creating objects that handle message transmission and reception.

*   **Publishers:** To publish data, a node first creates a `Publisher` object using `self.create_publisher()`. This method takes three main arguments: the message type (e.g., `String`, `Twist`), the topic name (e.g., `/robot/cmd_vel`), and the Quality of Service (QoS) profile. The QoS profile defines communication policies such as reliability and history. Once the publisher is created, the node can periodically construct messages of the specified type and call the publisher's `publish()` method to send them to the topic. Often, a `Timer` is used to trigger publishing at regular intervals.
*   **Subscribers:** To receive data, a node creates a `Subscriber` object using `self.create_subscription()`. This also requires the message type, the topic name, and a **callback function**. The callback function is a method within your node that will be automatically invoked by `rclpy` every time a new message arrives on the subscribed topic. The received message object is passed as an argument to this callback, where your node can then process the data. Like publishers, subscribers also specify a QoS profile.

This pattern allows for robust and asynchronous data flow between different parts of your robotic system.

:::tip Interactive Element Suggestion
**Interactive Code Example Walkthrough (Conceptual):** An interactive step-by-step walkthrough of a conceptual Python script for a simple publisher/subscriber pair. Users could see annotations explaining each line's purpose in creating the publisher/subscriber and handling the callback.
:::

### III. Service Servers and Clients in Python

Implementing the Request/Reply model for services in Python nodes involves distinct roles for service servers and clients.

*   **Service Servers:** A node acts as a service server by calling `self.create_service()`. This method requires the service type (e.g., `AddTwoInts`), the service name (e.g., `/calculator/add`), and a **callback function**. This callback function is executed whenever a client sends a request to this service. The callback receives the request message as an argument, performs the necessary computation or action, and returns a response message of the specified service type. The service server then sends this response back to the client.
*   **Service Clients:** A node acts as a service client by calling `self.create_client()`. This requires the service type and the service name. Once the client object is created, it can construct a request message and then call the client's `call_async()` method to send the request. Since service calls are asynchronous, `call_async()` returns a future object. The client node must then "wait" for this future to complete, typically by adding a callback to the future or spinning until the future is done, to retrieve the response. This prevents the client node from blocking the entire ROS 2 execution while waiting for the service response.

Services are fundamental for synchronized operations where a specific action needs to be performed, and the outcome is essential before proceeding.