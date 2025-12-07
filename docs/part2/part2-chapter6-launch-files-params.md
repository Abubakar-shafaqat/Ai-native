---
sidebar_position: 6
---

# Chapter 10: ROS 2 Launch Files and Parameters

## Part A: Orchestrating ROS 2 Applications with Launch Files

### I. The Purpose of Launch Files

ROS 2 Launch files are XML or Python scripts that serve as a declarative way to define and execute a collection of ROS 2 nodes and other processes. Their primary purpose is to simplify the startup and configuration of complex robotic applications, which often involve dozens of interconnected nodes. Instead of launching each node individually with specific command-line arguments, a single launch file can define all the required nodes, set their parameters, map topic names, remap node names, and even execute external commands. This provides a centralized and reproducible way to manage the runtime environment of a robotic system, ensuring that all components are initialized correctly and consistently, which is crucial for development, testing, and deployment.

:::tip Reflection Question
Consider a scenario where you have a mobile robot with separate nodes for motor control, LiDAR data processing, and obstacle avoidance. Without a launch file, what manual steps would you have to take to start and configure this robot's software? How could this become problematic in a real-world application?
:::

### II. Basic Launch File Elements (Nodes, Parameters)

ROS 2 Launch files are built using a set of fundamental elements that allow you to define the structure and behavior of your robotic application.

*   **`Node` Element:** The most common element is the `<node>` tag (in XML) or `Node()` action (in Python). This element specifies an executable ROS 2 node to be launched. You define its package name, executable name, and optionally provide a custom node name (to avoid conflicts or clarify purpose). You can also specify arguments to pass to the executable.
*   **`Parameter` Element:** Launch files are an ideal place to set parameters for your nodes. The `<param>` tag (in XML) or `DeclareLaunchArgument()` and `SetParametersFromFile()` actions (in Python) allow you to load parameter values from YAML files or set individual parameters directly. This ensures that nodes start with a predefined configuration, making it easy to change settings without recompiling code.

By combining these basic elements, you can quickly define how individual nodes are started and their initial configuration, forming the backbone of your robotic system's software architecture.

:::info Diagram Placeholder
**Diagram 6.1: Basic Launch File Execution**
A diagram showing a single "Launch File" box. Inside it, smaller boxes represent `Node A`, `Node B`, each with associated `Parameters`. Arrows emanate from the Launch File box to indicate that it "launches" these nodes with their specified configurations.
:::

### III. Advanced Launch Features (Groups, Conditionals, Events)

ROS 2 Launch files offer powerful advanced features for managing complex scenarios, enabling flexible and adaptable system deployments.

*   **Groups:** The `<group>` tag (XML) or `GroupAction()` (Python) allows you to logically group multiple nodes and actions, applying common attributes like remappings or namespaces to all elements within the group. This is particularly useful for managing multiple instances of the same robot or organizing subsystems.
*   **Conditionals:** Launch files support conditional execution using `<if>` / `<unless>` (XML) or `IfCondition()` / `UnlessCondition()` (Python). This allows you to launch nodes or set parameters based on the value of a launch argument, environment variable, or other conditions. For example, you might conditionally launch a simulation-specific node only when a `use_sim_time` argument is true.
*   **Events and Event Handlers:** ROS 2 Launch provides an event system to react to various occurrences, such as a node starting, a node exiting, or a shutdown request. `RegisterEventHandler()` (Python) allows you to define actions to be taken when specific events occur, such as logging a message or restarting a failed node. This enhances the robustness and self-healing capabilities of your robotic applications.

:::bulb Quiz Idea
**Quiz:** You want to launch a set of navigation nodes only when operating in a simulated environment. Which advanced launch feature would be most appropriate for this task?
a) Groups
b) Conditionals
c) Event Handlers
d) Parameters
*Correct Answer: b) Conditionals*
:::

## Part B: Dynamic Configuration with ROS 2 Parameters

### I. Declaring and Using Parameters in Nodes

ROS 2 parameters provide a structured way for nodes to expose their internal configuration values, allowing them to be dynamically adjusted without modifying and recompiling the source code. Within a Python node (using `rclpy`), parameters are typically declared in the node's constructor using `self.declare_parameter()`. When declaring a parameter, you specify its name (e.g., `"max_velocity"`), its default value, and optionally a description. After declaration, the node can retrieve the current value of a parameter using `self.get_parameter("param_name").value`. Parameters can be of various types, including integers, floats, booleans, strings, and lists of these types. This mechanism promotes flexible and adaptive node behavior, allowing a single node implementation to be used in diverse operational contexts by simply changing its parameter values.

:::info Diagram Placeholder
**Diagram 6.2: Node with Declared Parameters**
Shows a "ROS 2 Node" box. Inside, there are smaller boxes representing "Parameter A (default=X)", "Parameter B (default=Y)". An arrow points into the node labeled "Parameter API" through which external entities can interact with these parameters.
:::

### II. Overriding Parameters from Launch Files

One of the most common and powerful uses of ROS 2 parameters is overriding their default values directly from launch files. This provides a centralized and convenient way to configure an entire robotic system's behavior without directly touching individual node codebases. In a Python launch file, you can use `DeclareLaunchArgument()` to make a parameter configurable from the command line when running the launch file. More commonly, you can use `SetParametersFromFile()` to load a YAML file containing parameter key-value pairs, or directly pass parameters to a node using `parameters=[{'param_name': 'value'}]` within the `Node()` action. This allows developers to create different launch configurations (e.g., "fast_mode.launch.py", "safety_mode.launch.py") that set distinct parameter values for the same set of nodes, dramatically increasing the flexibility and reusability of their robotic software.

:::tip Interactive Element Suggestion
**Interactive Table:** A simple table where users can toggle between different "Launch File Configurations" (e.g., "High Speed," "Low Speed," "Debugging"), and see how `max_velocity`, `lidar_range`, or `log_level` parameters change for a hypothetical navigation node.
:::

### III. Dynamic Parameter Updates and `rqt_reconfigure`

Beyond setting parameters at launch time, ROS 2 also supports dynamic parameter updates during runtime. This means that a node can react to changes in its parameter values without being restarted. Nodes can register a callback function using `self.add_on_set_parameters_callback()` that is invoked whenever one of its declared parameters is modified. This enables advanced behaviors such as dynamically adjusting a controller's gains, changing the frequency of a sensor's output, or switching between different algorithms based on external commands or environmental conditions. The `rqt_reconfigure` tool (part of `rqt_console`) provides a graphical user interface (GUI) to easily inspect and modify a node's exposed parameters in real-time. This dynamic capability is invaluable during the debugging, tuning, and operational phases of robotic system development, offering unparalleled flexibility to adapt system behavior on the fly.