---
sidebar_position: 3
title: ROS 2 Development with Python
---

# Chapter 11: ROS 2 Development with Python

## Part A: Introduction to Python in ROS 2

### I. Why Python for ROS 2?

Python is a popular choice for ROS 2 development due to its simplicity, readability, extensive libraries, and rapid prototyping capabilities. While C++ is often used for performance-critical components, Python (`rclpy`) is excellent for:

*   High-level logic and task coordination.
*   Interfacing with AI/ML libraries (e.g., TensorFlow, PyTorch, OpenCV).
*   Scripting and data analysis.
*   Developing user interfaces and visualization tools.

### II. Setting Up Your Python Environment

Before diving into ROS 2 Python development, ensure you have:

*   **ROS 2 Distribution:** Installed a compatible ROS 2 distribution (e.g., Foxy, Galactic, Humble, Iron).
*   **Python:** ROS 2 typically supports specific Python versions (e.g., Python 3.8+).
*   **`rclpy`:** The Python client library for ROS 2, usually installed as part of the ROS 2 distribution.
*   **`colcon`:** The build system used by ROS 2, which orchestrates the compilation of packages.

## Part B: Creating and Building Python Packages

### I. Creating a ROS 2 Python Package

A ROS 2 package organizes your source code, build scripts, and other related files.

1.  **Create Workspace:**
    ```bash
    mkdir -p ros2_ws/src
    cd ros2_ws/src
    ```
2.  **Create Package:**
    ```bash
    ros2 pkg create --build-type ament_python my_python_pkg --dependencies rclpy std_msgs
    ```
    This command creates a new Python package named `my_python_pkg` with `rclpy` and `std_msgs` as dependencies.

### II. Package Structure

A typical `ament_python` package structure includes:

```
my_python_pkg/
├── package.xml                 # Package metadata (dependencies, author, etc.)
├── setup.py                    # Python build script
├── setup.cfg                   # Python build configuration
├── resource/
│   └── my_python_pkg           # Marker file for ament_python
├── my_python_pkg/              # Python source code directory (same name as package)
│   ├── __init__.py
│   └── my_node.py              # Your node's Python script
└── test/
    └── test_my_node.py
```

## Part C: Writing Your First ROS 2 Python Node

### I. Writing Your First ROS 2 Python Node

Let's create a simple "talker" node that publishes a "Hello World" message.

1.  **Edit `my_python_pkg/my_python_pkg/my_node.py`:**
    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class Talker(Node):

        def __init__(self):
            super().__init__('talker')
            self.publisher_ = self.create_publisher(String, 'topic', 10)
            timer_period = 0.5  # seconds
            self.timer = self.create_timer(timer_period, self.timer_callback)
            self.i = 0

        def timer_callback(self):
            msg = String()
            msg.data = 'Hello World: %d' % self.i
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)
            self.i += 1

    def main(args=None):
        rclpy.init(args=args)
        talker = Talker()
        rclpy.spin(talker)
        talker.destroy_node()
        rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

### II. Register the Executable in `setup.py`

You need to tell `colcon` how to find and install your Python executable. Add an `entry_points` section to your `setup.py` (or modify the existing one):
```python
from setuptools import find_packages, setup

package_name = 'my_python_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A simple ROS 2 Python package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_python_pkg.my_node:main', # This line registers your executable
        ],
    },
)
```

### III. Build the Package

Navigate back to your workspace root (`ros2_ws/`) and build:
```bash
cd ros2_ws
colcon build
```

### IV. Source the Setup Files

After building, you need to source the setup files to make your package executables available in your shell:
```bash
# On Linux/macOS
source install/setup.bash
# On Windows
call install/setup.bat
```

### V. Run the Node

```bash
ros2 run my_python_pkg talker
```

This basic workflow demonstrates how to create, build, and run a simple ROS 2 Python node. Mastering these fundamentals is key to building more complex robotic behaviors.
