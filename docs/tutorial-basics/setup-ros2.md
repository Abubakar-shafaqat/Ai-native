---
sidebar_position: 1
---

# 1. Setting Up Your ROS 2 Environment

This guide will walk you through the essential steps to set up a functional ROS 2 environment on your system, which is the foundational software framework for developing robotic applications.

## 1.1 Choose Your ROS 2 Distribution

ROS 2 comes in various distributions (like versions). For this tutorial, we recommend using the latest long-term support (LTS) release for stability. As of your context date (December 2025), a stable LTS release like "Iron Irwini" or a newer one would be appropriate. This guide assumes an Ubuntu Linux environment, which is the most common operating system for ROS 2 development.

:::tip Diagram Placeholder
**Diagram 1.1: ROS 2 Distribution Timeline (Conceptual)**
A simple timeline showing various ROS 2 distributions (e.g., Foxy, Galactic, Humble, Iron) with their release dates and LTS labels. Highlight the recommended LTS release.
:::

## 1.2 Install ROS 2 Packages

The installation process for ROS 2 involves adding the ROS 2 repositories to your system and then using the package manager to install the necessary components.

### 1.2.1 Configure ROS 2 Repositories

First, ensure your system's package list is up-to-date and install necessary tools:

1.  Open a terminal.
2.  Update system packages: `sudo apt update && sudo apt upgrade`
3.  Install `curl` and `software-properties-common`: `sudo apt install curl software-properties-common`
4.  Add the ROS 2 GPG key: `sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg`
5.  Add the ROS 2 repository to your sources list (replace `IRON` with your chosen distribution, e.g., `humble` or `iron`): `echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.d/ros2.list > /dev/null`

### 1.2.2 Install ROS 2 Desktop Install

With the repositories configured, you can now install the full ROS 2 Desktop environment, which includes ROS 2 base packages, `ros_comm` (ROS communications core), `rqt_` tools, and `rviz2`.

1.  Update your package cache: `sudo apt update`
2.  Install the desktop full package: `sudo apt install ros-<ROS2_DISTRO>-desktop-full` (e.g., `sudo apt install ros-iron-desktop-full`)

## 1.3 Environment Setup

After installation, you need to configure your shell environment so that ROS 2 commands are accessible.

### 1.3.1 Source the ROS 2 Setup Script

Each time you open a new terminal, you need to "source" the ROS 2 setup script to add ROS 2 tools and libraries to your path.

1.  Source the setup script: `source /opt/ros/<ROS2_DISTRO>/setup.bash` (e.g., `source /opt/ros/iron/setup.bash`)
2.  To make this permanent, add it to your `~/.bashrc` file: `echo "source /opt/ros/<ROS2_DISTRO>/setup.bash" >> ~/.bashrc`
    *   Then `source ~/.bashrc` to apply the changes immediately.

### 1.3.2 Verify Installation

To confirm your ROS 2 installation is successful, you can run a simple demo:

1.  Open a new terminal (or source your `~/.bashrc`).
2.  Run the `talker` node: `ros2 run demo_nodes_cpp talker`
3.  Open a second terminal.
4.  Source the ROS 2 setup script again.
5.  Run the `listener` node: `ros2 run demo_nodes_py listener`

You should see messages being published by the `talker` and received by the `listener`, confirming that your ROS 2 communication is working.

:::info Diagram Placeholder
**Diagram 1.2: ROS 2 Talker-Listener Verification**
Illustrates two terminal windows side-by-side.
-   **Terminal 1:** `ros2 run demo_nodes_cpp talker` command, showing "Publishing: 'Hello World: 1'" messages.
-   **Terminal 2:** `ros2 run demo_nodes_py listener` command, showing "I heard: 'Hello World: 1'" messages.
An arrow should visually connect the output of the talker to the input of the listener.
:::

## 1.4 Install Colcon (ROS 2 Build Tools)

`colcon` is the primary build tool used in ROS 2 to compile multiple packages efficiently. While often installed with `desktop-full`, it's good practice to ensure it's present.

1.  Install `colcon` and its necessary extensions: `sudo apt install python3-colcon-common-extensions`

Your ROS 2 development environment is now set up! You are ready to start building and experimenting with robotic applications.
