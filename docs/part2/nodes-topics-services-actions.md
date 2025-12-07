---
sidebar_position: 2
title: Nodes, Topics, Services, and Actions
---

# Chapter 7: Nodes, Topics, Services, and Actions

## Part A: Introduction to ROS 2 Communication Primitives

### I. Understanding ROS 2 Communication Primitives

ROS 2 provides a set of communication primitives that enable different software components (nodes) to interact and exchange data within a robotic system. These primitives are crucial for building modular, distributed, and scalable robot applications.

### II. Nodes: The Building Blocks

As introduced in the ROS 2 Architecture chapter, a **Node** is an executable process that performs a specific task. Each node should ideally be single-purpose and follow the Unix philosophy of "do one thing and do it well." This modularity allows for easier debugging, testing, and replacement of components.

*   **Creating a Node:** Nodes are typically implemented in C++ or Python using the `rclcpp` or `rclpy` client libraries, respectively.
*   **Node Lifecycle:** ROS 2 introduces a managed lifecycle for nodes, allowing them to transition through states like `unconfigured`, `inactive`, `active`, and `finalized`. This enables robust system startup and shutdown procedures.

## Part B: Asynchronous and Synchronous Communication

### I. Topics: Asynchronous Data Streaming

**Topics** are the most common way for nodes to exchange continuous streams of data. They implement a publish-subscribe communication model.

*   **Publishers:** A node that sends messages to a topic is called a **publisher**. Publishers create messages of a specific message type and broadcast them to a named topic.
*   **Subscribers:** A node that receives messages from a topic is called a **subscriber**. Subscribers listen for messages on a named topic and process them as they arrive.
*   **Message Types:** Messages transmitted over topics must conform to predefined data structures called **message types**. These types are defined using `.msg` files, which then generate language-specific headers/classes. Examples include `std_msgs/msg/String`, `sensor_msgs/msg/LaserScan`, `geometry_msgs/msg/Twist`.
*   **Decoupling:** Publishers and subscribers are decoupled; they don't need to know about each other's existence directly. The ROS 2 middleware handles message routing.

### II. Services: Synchronous Request-Reply

**Services** are used for synchronous communication, where a client node sends a request to a server node and waits for a single response.

*   **Service Servers:** A node that offers a service is a **service server**. It processes incoming requests and sends back a response.
*   **Service Clients:** A node that requests a service is a **service client**. It sends a request message and blocks (or uses a future in asynchronous clients) until it receives a response.
*   **Service Types:** Like topics, services also use predefined data structures called **service types**, defined in `.srv` files. A service type defines both the request and response message structures.
*   **Use Cases:** Services are ideal for immediate queries or single actions, such as "get current robot pose," "turn on/off a sensor," or "perform a single inverse kinematics calculation."

### III. Actions: Long-Running Goal-Oriented Tasks

**Actions** provide a framework for long-running, interruptible, and feedback-rich goal execution. They combine aspects of topics (for feedback) and services (for requests and results).

*   **Action Servers:** A node that provides an action is an **action server**. It receives goals, executes them, sends periodic feedback, and ultimately returns a result. It can also be preempted (cancelled) by a client.
*   **Action Clients:** A node that requests an action is an **action client**. It sends a goal to an action server, can monitor the goal's progress via feedback, and can cancel the goal.
*   **Action Types:** Action types are defined in `.action` files, specifying the goal, feedback, and result message structures.
*   **Use Cases:** Actions are suitable for tasks like "navigate to a waypoint," "grasp an object," or "perform a complex manipulation sequence," where the task might take a significant amount of time and requires progress monitoring or cancellation.

By judiciously using these communication primitives, developers can design complex robotic systems that are robust, flexible, and easy to maintain.
