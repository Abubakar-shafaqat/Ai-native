---
sidebar_position: 1
title: ROS 2 Architecture
---

# Chapter 6: Core Concepts of ROS 2 Architecture

## Part A: Introduction to ROS 2

### I. Introduction to ROS 2

The Robot Operating System (ROS) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms. ROS 2 is the latest generation, re-engineered to address the needs of modern robotics, including multi-robot systems, real-time control, and embedded systems.

### II. Why ROS 2?

ROS 2 was developed to overcome limitations of ROS 1 in areas such as:

*   **Real-time Performance:** Improved determinism and low-latency communication.
*   **Multi-robot Systems:** Better support for distributed systems and inter-robot communication.
*   **Security:** Enhanced security features for communication.
*   **Embedded Systems:** More flexible resource management and compatibility with various operating systems.
*   **Production Readiness:** Designed with industrial applications in mind.

## Part B: Core Concepts of ROS 2 Architecture

ROS 2's architecture is built around several key concepts that facilitate modularity, reusability, and robust communication between different components of a robot system.

### I. Nodes

A **Node** is an executable process that performs computations. In ROS 2, a robot system is typically composed of many nodes, each responsible for a specific function (e.g., one node for camera driver, another for localization, a third for path planning). Nodes are designed to be small, single-purpose, and loosely coupled.

### II. Topics

**Topics** are the primary mechanism for asynchronous, many-to-many, publish-subscribe communication in ROS 2. Nodes publish messages to topics, and other nodes subscribe to those topics to receive the messages. This allows for data streams (e.g., sensor data, motor commands) to flow throughout the robot system.

### III. Services

**Services** provide a synchronous, request-reply communication mechanism between nodes. One node acts as a service server, offering a particular service, and other nodes can act as service clients, sending requests and waiting for a response. Services are suitable for tasks that require an immediate, one-time result, like querying a robot's state or triggering a specific action.

### IV. Actions

**Actions** are a higher-level communication mechanism designed for long-running, goal-oriented tasks. They provide preemptable, feedback-enabled communication between an action client and an action server. For example, a "navigate to goal" action might continuously send feedback about the robot's progress and allow the client to cancel the goal mid-way.

### V. ROS 2 Graph

The **ROS 2 Graph** is the network of ROS 2 nodes processing data. It represents how all the different executable processes in the robot system are connected and communicate via topics, services, and actions. Visualizing this graph (e.g., using tools like `rqt_graph`) is essential for understanding and debugging complex robot behaviors.

## Part C: Data Distribution Service (DDS)

### I. Data Distribution Service (DDS)

A fundamental difference from ROS 1, ROS 2 is built on top of **DDS (Data Distribution Service)**. DDS is an open international standard for publish-subscribe communication designed for real-time, high-performance, and scalable systems. ROS 2 leverages DDS for its underlying communication, enabling features like quality of service (QoS) settings, discovery, and transport capabilities.

### II. Quality of Service (QoS)

ROS 2's QoS policies allow developers to fine-tune the communication behavior of topics, services, and actions to meet specific application requirements. Examples include:

*   **Reliability:** Guarantees message delivery, even if it requires retransmissions.
*   **Durability:** Controls whether late-joining subscribers receive previously published messages.
*   **History:** Determines how many messages a publisher keeps and how many a subscriber receives.
*   **Liveliness:** Monitors whether other entities are active and communicating.

Understanding the ROS 2 architecture and its underlying DDS framework is crucial for developing robust and efficient robotic applications.
