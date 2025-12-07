---
sidebar_position: 1
title: Gazebo and URDF/SDF
---

# Chapter 14: Robot Description Formats: URDF and SDF

## Part A: Robot Simulation with Gazebo

### I. Robot Simulation with Gazebo

Gazebo is a powerful 3D robot simulator that accurately and efficiently simulates robots in complex indoor and outdoor environments. It offers the ability to accurately simulate populations of robots, sensors, and objects in a high-fidelity virtual world. Gazebo is widely used in robotics research and development, particularly with ROS/ROS 2, due to its robust physics engine, rendering capabilities, and extensive model library.

### II. Key Features of Gazebo

*   **Physics Engine:** Supports multiple high-performance physics engines (ODE, Bullet, Simbody, DART) for realistic rigid body dynamics.
*   **High-Quality Graphics:** Realistic rendering of environments and robots.
*   **Sensor Simulation:** Accurate simulation of various sensors (cameras, LiDAR, IMUs, force sensors).
*   **Plugin Architecture:** Extensible through plugins for custom robot control, sensor models, and world interactions.
*   **Command-Line Tools:** Tools for controlling and interacting with simulations programmatically.

## Part B: Robot Description Formats: URDF and SDF

To simulate a robot in Gazebo (or other simulators), you need a way to describe its physical properties, joints, sensors, and actuators. ROS 2 primarily uses two XML-based formats for this purpose: URDF and SDF.

### I. URDF (Unified Robot Description Format)

URDF is an XML format for describing all elements of a robot. It is designed for representing the kinematic and dynamic properties of a single robot in a tree-like structure.

*   **Links:** Represent rigid bodies of the robot (e.g., base, arm segments, wheels).
*   **Joints:** Define the connections between links, specifying their type (revolute, prismatic, fixed) and motion limits.
*   **Visuals:** Describe the graphical appearance of links.
*   **Collisions:** Define the collision geometry for physics simulation.
*   **Inertial Properties:** Specify mass and inertia tensors for dynamic calculations.

**Limitations of URDF:**
*   Cannot describe closed kinematic chains (loops).
*   Cannot specify arbitrary attachments or environmental objects.
*   Primarily for single robots, not entire worlds.

### II. SDF (Simulation Description Format)

SDF is a more comprehensive XML format designed for describing environments, objects, and robots for use in simulators like Gazebo. It is a superset of URDF capabilities and can describe everything in a simulation world, including multiple robots, static objects, terrain, lights, and sensors.

*   **World Tag:** The top-level tag for an SDF file, containing all elements of a simulation.
*   **Model Tag:** Describes a collection of links and joints, representing a robot or an object. SDF models can have multiple root links and describe closed loops.
*   **Joints, Links, Visuals, Collisions, Inertials:** Similar concepts to URDF but with more flexibility and features.
*   **Sensors:** Can be directly attached to links within a model or standalone in the world.
*   **Lights:** Definition of light sources within the simulation.
*   **Plugins:** Integration of custom C++ or Python code to extend simulation behavior.

### III. Converting URDF to SDF

Gazebo can load URDF files, which it internally converts to SDF for simulation. However, for complex world descriptions or features not supported by URDF (like lights or environmental objects), direct SDF creation is necessary. Many tools exist to convert URDF to SDF, or to generate SDF from scratch.

Understanding and correctly utilizing URDF and SDF are fundamental skills for anyone building robot simulations.
