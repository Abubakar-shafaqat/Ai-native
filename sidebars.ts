import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro', // The intro.md file

    {
      type: 'category',
      label: 'Chapter 1: Introduction to Physical AI',
      link: {
        type: 'doc',
        id: 'part1/chap1',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Defining Physical and Embodied Intelligence',
          href: '/docs/part1/chap1#part-a-defining-physical-and-embodied-intelligence',
        },
        {
          type: 'link',
          label: 'Part B: Core Technical Pillars',
          href: '/docs/part1/chap1#part-b-core-technical-pillars',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 2: AI in Robotics (Overview)',
      link: {
        type: 'doc',
        id: 'part1/ai-in-robotics-overview',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: The Role of AI in Modern Robotics',
          href: '/docs/part1/ai-in-robotics-overview#part-a-the-role-of-ai-in-modern-robotics',
        },
        {
          type: 'link',
          label: 'Part B: Historical Context and Future Outlook',
          href: '/docs/part1/ai-in-robotics-overview#part-b-historical-context-and-future-outlook',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 3: Introduction to Robotics',
      link: {
        type: 'doc',
        id: 'part1/introduction-to-robotics',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Fundamentals of Robotics',
          href: '/docs/part1/introduction-to-robotics#part-a-fundamentals-of-robotics',
        },
        {
          type: 'link',
          label: 'Part B: Context and Applications',
          href: '/docs/part1/introduction-to-robotics#part-b-context-and-applications',
        },
        {
          type: 'link',
          label: 'Part C: The Evolution to Intelligent Robots',
          href: '/docs/part1/introduction-to-robotics#part-c-the-evolution-to-intelligent-robots',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 4: Sensors & Perception',
      link: {
        type: 'doc',
        id: 'part1/part1-chapter2-sensors-perception',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Overview of Sensors',
          href: '/docs/part1/part1-chapter2-sensors-perception#part-a-overview-of-sensors',
        },
        {
          type: 'link',
          label: 'Part B: Perception Systems',
          href: '/docs/part1/part1-chapter2-sensors-perception#part-b-perception-systems',
        },
        {
          type: 'link',
          label: 'Part C: Applications in Physical AI',
          href: '/docs/part1/part1-chapter2-sensors-perception#part-c-applications-in-physical-ai',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 5: Sensing and Perception',
      link: {
        type: 'doc',
        id: 'part1/sensing-and-perception',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: The Role of Sensors in Robotics',
          href: '/docs/part1/sensing-and-perception#part-a-the-role-of-sensors-in-robotics',
        },
        {
          type: 'link',
          label: 'Part B: Principles of Perception',
          href: '/docs/part1/sensing-and-perception#part-b-principles-of-perception',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 6: Core Concepts of ROS 2 Architecture',
      link: {
        type: 'doc',
        id: 'part2/ros2-architecture',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Introduction to ROS 2',
          href: '/docs/part2/ros2-architecture#part-a-introduction-to-ros-2',
        },
        {
          type: 'link',
          label: 'Part B: Core Concepts of ROS 2 Architecture',
          href: '/docs/part2/ros2-architecture#part-b-core-concepts-of-ros-2-architecture',
        },
        {
          type: 'link',
          label: 'Part C: Data Distribution Service (DDS)',
          href: '/docs/part2/ros2-architecture#part-c-data-distribution-service-dds',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 7: ROS 2 Architecture',
      link: {
        type: 'doc',
        id: 'part2/part2-chapter3-ros2-architecture',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Core Concepts of ROS 2',
          href: '/docs/part2/part2-chapter3-ros2-architecture#part-a-core-concepts-of-ros-2',
        },
        {
          type: 'link',
          label: 'Part B: Key Architectural Components',
          href: '/docs/part2/part2-chapter3-ros2-architecture#part-b-key-architectural-components',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 6: Core Concepts of ROS 2 Architecture',
      link: {
        type: 'doc',
        id: 'part2/ros2-architecture',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Introduction to ROS 2',
          href: '/docs/part2/ros2-architecture#part-a-introduction-to-ros-2',
        },
        {
          type: 'link',
          label: 'Part B: Core Concepts of ROS 2 Architecture',
          href: '/docs/part2/ros2-architecture#part-b-core-concepts-of-ros-2-architecture',
        },
        {
          type: 'link',
          label: 'Part C: Data Distribution Service (DDS)',
          href: '/docs/part2/ros2-architecture#part-c-data-distribution-service-dds',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 7: ROS 2 Architecture',
      link: {
        type: 'doc',
        id: 'part2/part2-chapter3-ros2-architecture',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Core Concepts of ROS 2',
          href: '/docs/part2/part2-chapter3-ros2-architecture#part-a-core-concepts-of-ros-2',
        },
        {
          type: 'link',
          label: 'Part B: Key Architectural Components',
          href: '/docs/part2/part2-chapter3-ros2-architecture#part-b-key-architectural-components',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 8: Nodes, Topics, Services, and Actions',
      link: {
        type: 'doc',
        id: 'part2/nodes-topics-services-actions',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Introduction to ROS 2 Communication Primitives',
          href: '/docs/part2/nodes-topics-services-actions#part-a-introduction-to-ros-2-communication-primitives',
        },
        {
          type: 'link',
          label: 'Part B: Asynchronous and Synchronous Communication',
          href: '/docs/part2/nodes-topics-services-actions#part-b-asynchronous-and-synchronous-communication',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 9: ROS 2 Communication Fundamentals',
      link: {
        type: 'doc',
        id: 'part2/part2-chapter4-nodes-topics-services',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: ROS 2 Communication Fundamentals',
          href: '/docs/part2/part2-chapter4-nodes-topics-services#part-a-ros-2-communication-fundamentals',
        },
        {
          type: 'link',
          label: 'Part B: Advanced Communication Patterns',
          href: '/docs/part2/part2-chapter4-nodes-topics-services#part-b-advanced-communication-patterns',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 10: Writing ROS 2 Packages (Python)',
      link: {
        type: 'doc',
        id: 'part2/part2-chapter5-python-packages',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: ROS 2 Package Structure',
          href: '/docs/part2/part2-chapter5-python-packages#part-a-ros-2-package-structure',
        },
        {
          type: 'link',
          label: 'Part B: Implementing ROS 2 Nodes in Python',
          href: '/docs/part2/part2-chapter5-python-packages#part-b-implementing-ros-2-nodes-in-python',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 11: ROS 2 Launch Files and Parameters',
      link: {
        type: 'doc',
        id: 'part2/part2-chapter6-launch-files-params',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Orchestrating ROS 2 Applications with Launch Files',
          href: '/docs/part2/part2-chapter6-launch-files-params#part-a-orchestrating-ros-2-applications-with-launch-files',
        },
        {
          type: 'link',
          label: 'Part B: Dynamic Configuration with ROS 2 Parameters',
          href: '/docs/part2/part2-chapter6-launch-files-params#part-b-dynamic-configuration-with-ros-2-parameters',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 12: ROS 2 Development with Python',
      link: {
        type: 'doc',
        id: 'part2/ros2-development-with-python',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Introduction to Python in ROS 2',
          href: '/docs/part2/ros2-development-with-python#part-a-introduction-to-python-in-ros-2',
        },
        {
          type: 'link',
          label: 'Part B: Creating and Building Python Packages',
          href: '/docs/part2/ros2-development-with-python#part-b-creating-and-building-python-packages',
        },
        {
          type: 'link',
          label: 'Part C: Writing Your First ROS 2 Python Node',
          href: '/docs/part2/ros2-development-with-python#part-c-writing-your-first-ros-2-python-node',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 13: Gazebo Simulation Setup',
      link: {
        type: 'doc',
        id: 'part3/part3-chapter7-gazebo-setup',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Introduction to Robotic Simulation with Gazebo',
          href: '/docs/part3/part3-chapter7-gazebo-setup#part-a-introduction-to-robotic-simulation-with-gazebo',
        },
        {
          type: 'link',
          label: 'Part B: Setting Up a Basic Simulation Environment',
          href: '/docs/part3/part3-chapter7-gazebo-setup#part-b-setting-up-a-basic-simulation-environment',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 14: Robot Description Formats: URDF and SDF',
      link: {
        type: 'doc',
        id: 'part3/gazebo-and-urdf-sdf',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Robot Simulation with Gazebo',
          href: '/docs/part3/gazebo-and-urdf-sdf#part-a-robot-simulation-with-gazebo',
        },
        {
          type: 'link',
          label: 'Part B: Robot Description Formats: URDF and SDF',
          href: '/docs/part3/gazebo-and-urdf-sdf#part-b-robot-description-formats-urdf-and-sdf',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 15: URDF and SDF for Robot Models',
      link: {
        type: 'doc',
        id: 'part3/part3-chapter8-urdf-sdf',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Describing Robot Kinematics and Dynamics',
          href: '/docs/part3/part3-chapter8-urdf-sdf#part-a-describing-robot-kinematics-and-dynamics',
        },
        {
          type: 'link',
          label: 'Part B: Building and Using Robot Models',
          href: '/docs/part3/part3-chapter8-urdf-sdf#part-b-building-and-using-robot-models',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 16: Physics and Sensor Simulation',
      link: {
        type: 'doc',
        id: 'part3/part3-chapter9-physics-sensors',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Principles of Physics Simulation in Robotics',
          href: '/docs/part3/part3-chapter9-physics-sensors#part-a-principles-of-physics-simulation-in-robotics',
        },
        {
          type: 'link',
          label: 'Part B: Simulating Robotic Sensors',
          href: '/docs/part3/part3-chapter9-physics-sensors#part-b-simulating-robotic-sensors',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 17: Unity for Robot Visualization',
      link: {
        type: 'doc',
        id: 'part3/part3-chapter10-unity-visualization',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Advanced Visualization in Robotics',
          href: '/docs/part3/part3-chapter10-unity-visualization#part-a-advanced-visualization-in-robotics',
        },
        {
          type: 'link',
          label: 'Part B: Integrating ROS 2 with Unity',
          href: '/docs/part3/part3-chapter10-unity-visualization#part-b-integrating-ros-2-with-unity',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 18: Physics and Sensors in Simulation',
      link: {
        type: 'doc',
        id: 'part3/physics-and-sensors-in-simulation',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Realistic Robot Behavior: Physics Engines',
          href: '/docs/part3/physics-and-sensors-in-simulation#part-a-realistic-robot-behavior-physics-engines',
        },
        {
          type: 'link',
          label: 'Part B: Simulating Sensors for Perception',
          href: '/docs/part3/physics-and-sensors-in-simulation#part-b-simulating-sensors-for-perception',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 19: Unity/Isaac Sim for Robotics',
      link: {
        type: 'doc',
        id: 'part3/unity-isaac-sim-for-robotics',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Advanced Robot Simulation with Game Engines',
          href: '/docs/part3/unity-isaac-sim-for-robotics#part-a-advanced-robot-simulation-with-game-engines',
        },
        {
          type: 'link',
          label: 'Part B: NVIDIA Isaac Sim',
          href: '/docs/part3/unity-isaac-sim-for-robotics#part-b-nvidia-isaac-sim',
        },
      ],
    },    {
      type: 'category',
      label: 'Chapter 20: Localization and Mapping (SLAM)',
      link: {
        type: 'doc',
        id: 'part4/localization-and-mapping-slam',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Understanding SLAM',
          href: '/docs/part4/localization-and-mapping-slam#part-a-understanding-slam',
        },
        {
          type: 'link',
          label: 'Part B: Advanced Topics in SLAM',
          href: '/docs/part4/localization-and-mapping-slam#part-b-advanced-topics-in-slam',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 21: Navigation (Path Planning & Control)',
      link: {
        type: 'doc',
        id: 'part4/navigation-path-planning-control',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Autonomous Robot Navigation',
          href: '/docs/part4/navigation-path-planning-control#part-a-autonomous-robot-navigation',
        },
        {
          type: 'link',
          label: 'Part B: Motion Control',
          href: '/docs/part4/navigation-path-planning-control#part-b-motion-control',
        },
        {
          type: 'link',
          label: 'Part C: The ROS 2 Navigation Stack (Nav2)',
          href: '/docs/part4/navigation-path-planning-control#part-c-the-ros-2-navigation-stack-nav2',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 22: Isaac Sim',
      link: {
        type: 'doc',
        id: 'part4/part4-chapter11-isaac-sim',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Next-Generation Robotics Simulation',
          href: '/docs/part4/part4-chapter11-isaac-sim#part-a-next-generation-robotics-simulation',
        },
        {
          type: 'link',
          label: 'Part B: Developing with Isaac Sim',
          href: '/docs/part4/part4-chapter11-isaac-sim#part-b-developing-with-isaac-sim',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 23: Isaac ROS and Hardware Acceleration',
      link: {
        type: 'doc',
        id: 'part4/part4-chapter12-isaac-ros',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Accelerating ROS 2 with NVIDIA Hardware',
          href: '/docs/part4/part4-chapter12-isaac-ros#part-a-accelerating-ros-2-with-nvidia-hardware',
        },
        {
          type: 'link',
          label: 'Part B: Deep Learning Inference and Sensor Processing',
          href: '/docs/part4/part4-chapter12-isaac-ros#part-b-deep-learning-inference-and-sensor-processing',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 24: Nav2 and Reinforcement Learning',
      link: {
        type: 'doc',
        id: 'part4/part4-chapter13-nav2-rl',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Autonomous Navigation with Nav2',
          href: '/docs/part4/part4-chapter13-nav2-rl#part-a-autonomous-navigation-with-nav2',
        },
        {
          type: 'link',
          label: 'Part B: Reinforcement Learning for Navigation',
          href: '/docs/part4/part4-chapter13-nav2-rl#part-b-reinforcement-learning-for-navigation',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 25: Sim-to-Real Transfer',
      link: {
        type: 'doc',
        id: 'part4/part4-chapter14-sim-to-real',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Bridging the Reality Gap',
          href: '/docs/part4/part4-chapter14-sim-to-real#part-a-bridging-the-reality-gap',
        },
        {
          type: 'link',
          label: 'Part B: Techniques for Successful Transfer',
          href: '/docs/part4/part4-chapter14-sim-to-real#part-b-techniques-for-successful-transfer',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 26: Reinforcement Learning for Robotics',
      link: {
        type: 'doc',
        id: 'part4/reinforcement-learning-for-robotics',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Reinforcement Learning (RL) in Robotics',
          href: '/docs/part4/reinforcement-learning-for-robotics#part-a-reinforcement-learning-rl-in-robotics',
        },
        {
          type: 'link',
          label: 'Part B: Challenges and Applications of RL in Robotics',
          href: '/docs/part4/reinforcement-learning-for-robotics#part-b-challenges-and-applications-of-rl-in-robotics',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 27: Humanoid Kinematics',
      link: {
        type: 'doc',
        id: 'part5/part5-chapter15-kinematics',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Fundamentals of Robot Kinematics',
          href: '/docs/part5/part5-chapter15-kinematics#part-a-fundamentals-of-robot-kinematics',
        },
        {
          type: 'link',
          label: 'Part B: Kinematics for Humanoid Robots',
          href: '/docs/part5/part5-chapter15-kinematics#part-b-kinematics-for-humanoid-robots',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 28: Humanoid Kinematics & Dynamics',
      link: {
        type: 'doc',
        id: 'part5/humanoid-kinematics-dynamics',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Understanding Humanoid Robot Motion',
          href: '/docs/part5/humanoid-kinematics-dynamics#part-a-understanding-humanoid-robot-motion',
        },
        {
          type: 'link',
          label: 'Part B: Key Concepts for Humanoid Dynamics',
          href: '/docs/part5/humanoid-kinematics-dynamics#part-b-key-concepts-for-humanoid-dynamics',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 29: Locomotion & Balance',
      link: {
        type: 'doc',
        id: 'part5/locomotion-balance',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Humanoid Locomotion: The Challenge of Bipedalism',
          href: '/docs/part5/locomotion-balance#part-a-humanoid-locomotion-the-challenge-of-bipedalism',
        },
        {
          type: 'link',
          label: 'Part B: Navigation in Humanoid Robotics',
          href: '/docs/part5/locomotion-balance#part-b-navigation-in-humanoid-robotics',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 30: Manipulation & Grasping',
      link: {
        type: 'doc',
        id: 'part5/manipulation-grasping',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Robotic Manipulation: Interacting with the World',
          href: '/docs/part5/manipulation-grasping#part-a-robotic-manipulation-interacting-with-the-world',
        },
        {
          type: 'link',
          label: 'Part B: Challenges in Humanoid Manipulation',
          href: '/docs/part5/manipulation-grasping#part-b-challenges-in-humanoid-manipulation',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 31: Conversational Robotics',
      link: {
        type: 'doc',
        id: 'part6/part6-chapter18-conversational-robotics',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Foundations of Human-Robot Dialogue',
          href: '/docs/part6/part6-chapter18-conversational-robotics#part-a-foundations-of-human-robot-dialogue',
        },
        {
          type: 'link',
          label: 'Part B: Integrating Language with Embodied Action',
          href: '/docs/part6/part6-chapter18-conversational-robotics#part-b-integrating-language-with-embodied-action',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 32: Whisper Voice Commands',
      link: {
        type: 'doc',
        id: 'part6/part6-chapter19-voice-interfaces',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Advancements in Speech Recognition for Robotics',
          href: '/docs/part6/part6-chapter19-voice-interfaces#part-a-advancements-in-speech-recognition-for-robotics',
        },
        {
          type: 'link',
          label: 'Part B: Implementing Whisper for Robot Control',
          href: '/docs/part6/part6-chapter19-voice-interfaces#part-b-implementing-whisper-for-robot-control',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 33: LLM to ROS Action Planning',
      link: {
        type: 'doc',
        id: 'part6/part6-chapter20-llm-to-ros-planning',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Large Language Models in Robotics',
          href: '/docs/part6/part6-chapter20-llm-to-ros-planning#part-a-large-language-models-in-robotics',
        },
        {
          type: 'link',
          label: 'Part B: Architectures for LLM-Driven Robot Control',
          href: '/docs/part6/part6-chapter20-llm-to-ros-planning#part-b-architectures-for-llm-driven-robot-control',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 34: Conversational AI for Robots',
      link: {
        type: 'doc',
        id: 'part6/conversational-ai-for-robots',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: The Rise of Conversational Robots',
          href: '/docs/part6/conversational-ai-for-robots#part-a-the-rise-of-conversational-robots',
        },
        {
          type: 'link',
          label: 'Part B: Challenges and LLMs in Conversational Robotics',
          href: '/docs/part6/conversational-ai-for-robots#part-b-challenges-and-llms-in-conversational-robotics',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 35: Voice Interfaces and Speech Recognition',
      link: {
        type: 'doc',
        id: 'part6/voice-interfaces-and-speech-recognition',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Voice as a Natural Human-Robot Interface',
          href: '/docs/part6/voice-interfaces-and-speech-recognition#part-a-voice-as-a-natural-human-robot-interface',
        },
        {
          type: 'link',
          label: 'Part B: Whisper: A Powerful ASR Model',
          href: '/docs/part6/voice-interfaces-and-speech-recognition#part-b-whisper-a-powerful-asr-model',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 36: LLM to ROS Action Planning',
      link: {
        type: 'doc',
        id: 'part6/llm-to-ros-action-planning',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Bridging the Gap: Large Language Models and Robot Action',
          href: '/docs/part6/llm-to-ros-action-planning#part-a-bridging-the-gap-large-language-models-and-robot-action',
        },
        {
          type: 'link',
          label: 'Part B: LLM to ROS Action Planning Pipeline',
          href: '/docs/part6/llm-to-ros-action-planning#part-b-llm-to-ros-action-planning-pipeline',
        },
      ],
    },
    {
      type: 'category',
      label: 'Chapter 37: Full Humanoid Autonomy Pipeline',
      link: {
        type: 'doc',
        id: 'capstone/capstone-chapter21-full-autonomy-pipeline',
      },
      items: [
        {
          type: 'link',
          label: 'Part A: Architecture for Humanoid Autonomy',
          href: '/docs/capstone/capstone-chapter21-full-autonomy-pipeline#part-a-architecture-for-humanoid-autonomy',
        },
        {
          type: 'link',
          label: 'Part B: Future of Humanoid AI',
          href: '/docs/capstone/capstone-chapter21-full-autonomy-pipeline#part-b-future-of-humanoid-ai',
        },
      ],
    },
    
    // Tutorial categories
    {
      type: 'category',
      label: 'Tutorial - Basics',
      link: {
        type: 'generated-index',
        description: 'Fundamental skills and concepts in Physical AI and Humanoid Robotics, including ROS 2 setup, Gazebo, and Isaac Sim basics.',
      },
      items: [{type: 'autogenerated', dirName: 'tutorial-basics'}],
    },
    {
      type: 'category',
      label: 'Tutorial - Extras',
      link: {
        type: 'generated-index',
        description: 'Intermediate to advanced experiments and integrations in Physical AI and Humanoid Robotics.',
      },
      items: [{type: 'autogenerated', dirName: 'tutorial-extras'}],
    },
  ],
};

export default sidebars;