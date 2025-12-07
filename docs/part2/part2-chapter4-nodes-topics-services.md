---
sidebar_position: 4
---

# Chapter 8: ROS 2 Communication Fundamentals

## Part A: ROS 2 Communication Fundamentals

### I. Understanding Nodes

A **Node** is the fundamental unit of computation in ROS 2. Conceptually, it is an executable process that performs a specific, well-defined task within the robotic system. For instance, a robot might have a node responsible for reading data from its LiDAR sensor, another for controlling its motors, and yet another for navigating based on a map. Nodes are designed to be modular and loosely coupled, meaning they can be developed, tested, and deployed independently. This modularity greatly simplifies the development and debugging of complex robotic applications, as well as promoting code reusability. A single robotic application typically consists of many nodes running concurrently, communicating with each other to achieve higher-level functionalities.

:::tip Reflection Question
Imagine a simple mobile robot that needs to avoid obstacles. What distinct, small tasks would this robot need to perform? How could each of these tasks be encapsulated into a separate ROS 2 node?
:::

### II. The Publish/Subscribe Model and Topics

The **Publish/Subscribe (Pub/Sub) model** is the primary method for asynchronous data streaming in ROS 2. It's used for continuous, one-to-many communication where nodes transmit data without needing to know which other nodes are receiving it, and receiving nodes get data without needing to know which nodes are sending it.

*   **Topics:** A Topic is a named channel over which nodes exchange messages. When a node wants to share data, it "publishes" a message to a specific topic. Any other node interested in that type of data can "subscribe" to the same topic to receive messages. This allows for flexible and scalable communication. For example, a "camera_node" might publish image messages to a `/camera/image_raw` topic, and an "object_detection_node" would subscribe to this topic to process the images.

Messages on topics are typed, meaning they adhere to a predefined data structure (e.g., `sensor_msgs/msg/Image` for image data). This ensures that all publishers and subscribers on a given topic understand the format of the data being exchanged. The pub/sub model is ideal for streams of data that are continuously updated, like sensor readings, joint states, or odometry information.

:::info Diagram Placeholder
**Diagram 4.1: Publish/Subscribe Communication**
Illustrates two Publisher nodes (e.g., "LiDAR Sensor Node", "Camera Node") sending data to two different Topics (e.g., "/scan", "/image_raw"). Two Subscriber nodes (e.g., "Mapper Node", "Object Detector Node") are shown subscribing to these topics. Arrows indicate data flow from Publishers to Topics and from Topics to Subscribers. The topics are central, but communication is decentralized via DDS.
:::

:::bulb Quiz Idea
**Quiz:** A node that continuously reports the robot's current position to all other interested nodes would use which ROS 2 communication mechanism?
a) Services
b) Actions
c) Topics
d) Parameters
*Correct Answer: c) Topics*
:::

### III. Request/Reply with Services

While topics are excellent for continuous data streams, sometimes a node needs to request a specific piece of information or trigger an action from another node and then wait for a response. For these synchronous, one-to-one interactions, ROS 2 provides **Services**.

*   **Services:** A Service defines a pair of message types: a request message and a response message. One node acts as a "Service Server," offering a specific service. Other nodes act as "Service Clients," sending requests to the server and blocking (or uses a future in asynchronous clients) until they receive a response. For example, a "map_server_node" might offer a service called `/get_map` that, when requested by a client (e.g., a "navigation_node"), returns the current map of the environment.

Services are well-suited for operations that occur infrequently, require an immediate result, and involve a specific request-response interaction. Examples include triggering a robot to perform a specific action, querying a database for a piece of information, or reconfiguring a sensor.

:::info Diagram Placeholder
**Diagram 4.2: Service Communication**
Shows a "Service Client Node" sending a "Request" message to a "Service Server Node" via a named Service (e.g., "/calibrate_robot"). The Service Server processes the request and sends back a "Response" message to the client. This clearly illustrates the synchronous, request-reply nature.
:::

## Part B: Advanced Communication Patterns

### I. Actions for Long-Running Tasks

For tasks that are typically long-running, interruptible, and provide periodic feedback on their progress, ROS 2 introduces **Actions**. Actions combine features of both topics and services, providing a more structured way to manage complex, goal-oriented behaviors.

An Action consists of three parts:

*   **Goal:** The client sends a goal to the action server (e.g., "Move robot to position X, Y").
*   **Feedback:** The action server provides continuous feedback to the client about the progress of the goal (e.g., "Robot is currently at X', Y' and 50% complete").
*   **Result:** Once the goal is completed (successfully or with failure), the action server sends a final result to a client (e.g., "Robot reached target position successfully").

Crucially, actions can be preempted or canceled by the client if circumstances change. This makes them ideal for tasks like complex navigation (e.g., "navigate to kitchen"), manipulating objects over a period, or executing sequences of movements.

:::info Diagram Placeholder
**Diagram 4.3: Action Communication Flow**
Illustrates a "Action Client Node" sending a "Goal" to an "Action Server Node". The Action Server continuously sends "Feedback" to the Client, and finally sends a "Result" once the Goal is achieved or failed. An optional arrow from Client to Server indicating "Cancel Goal" or "Preempt" should also be shown.
:::

:::tip Interactive Element Suggestion
**Interactive Diagram:** An interactive sequence diagram where users can click "Send Goal," "Get Feedback," "Get Result," and "Cancel Goal" to see how the messages flow between an Action Client and an Action Server for a robot navigation task.
:::

### II. Messages and Interfaces

At the core of all ROS 2 communication (topics, services, and actions) are **Messages**. Messages are simple data structures used to transmit information between nodes. Each message has a specific type, defined in `.msg` files, which specifies the data fields and their types (e.g., `std_msgs/msg/String`, `geometry_msgs/msg/Point`).

*   **Interfaces:** The definition files for messages (`.msg`), services (`.srv`), and actions (`.action`) are collectively known as **Interfaces**. These interface definitions are language-agnostic, allowing ROS 2 to automatically generate source code for different programming languages (like C++ and Python) that conform to these definitions. This ensures interoperability and consistent data handling across the entire ROS 2 ecosystem. By defining clear interfaces, developers can ensure that all nodes expecting a certain type of data receive it in a predictable format, promoting modularity and robust integration.

### III. Parameters for Dynamic Configuration

**Parameters** provide a mechanism for dynamically configuring the behavior of nodes at runtime, without needing to recompile the code. Instead of hardcoding values directly into a node's source code, parameters allow developers to expose configurable settings that can be changed through the ROS 2 command-line tools or programmatically by other nodes.

*   **How they work:** A node declares which parameters it exposes (e.g., `speed_limit`, `detection_threshold`). These parameters have names, types (integer, float, string, boolean), and default values. Other nodes or users can then query the current value of a parameter, set a new value, or even be notified when a parameter's value changes. This is particularly useful for fine-tuning algorithms, adjusting sensor settings, or switching between different operational modes without restarting the entire system. For example, a "navigation_node" might have a `max_velocity` parameter that can be adjusted on the fly to control the robot's speed.

:::info Diagram Placeholder
**Diagram 4.4: Parameter Interaction**
Shows a "Node A" (e.g., "Navigation Node") exposing a "Parameter" (e.g., `max_speed`). A "Node B" (e.g., "User Interface Node") or the "ROS 2 CLI" is shown setting/getting the parameter value. This highlights the dynamic configuration aspect.
:::