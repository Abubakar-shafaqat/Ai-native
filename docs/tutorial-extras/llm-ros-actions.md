---
sidebar_position: 4
---

# 4. Integrating LLM Commands with ROS 2 Actions

Large Language Models (LLMs) (discussed in Chapter 20) are revolutionizing how robots interpret high-level, natural language instructions. This tutorial guides you through the conceptual setup of an architecture that allows an LLM to decompose a human command into a sequence of ROS 2 actions (Chapter 4) that your robot can execute. We'll focus on the "LLM as a Task Decomposer" paradigm.

**Learning Objective:** Understand how to expose robot capabilities to an LLM, how the LLM generates a plan, and how that plan can be executed by a ROS 2 system.

## 4.1 Understanding the LLM-ROS 2 Action Pipeline

The core idea is to treat the LLM as a sophisticated natural language interface and task planner. It takes a human instruction, identifies available robot "tools" (ROS 2 services, actions, topics), and orchestrates them to fulfill the request.

### 4.1.1 Key Components

1.  **Human Input:** Natural language command (e.g., "Go to the kitchen and grab a soda from the fridge"). This could come from a voice interface (Chapter 19).
2.  **LLM Interface:** A ROS 2 node or Python script that communicates with the LLM API (local or cloud-based). This interface feeds the human instruction and a description of the robot's capabilities (its "API schema") to the LLM.
3.  **LLM (Large Language Model):** Processes the input, generates a sequence of "function calls" (representing robot actions) that would achieve the human's goal.
4.  **Action Executor:** A ROS 2 node that receives the LLM's generated function calls, validates them, and translates them into actual ROS 2 service calls or action goals (Chapter 4).
5.  **Robot Capabilities (ROS 2 Services/Actions):** Pre-programmed, robust ROS 2 services and actions that perform specific, low-level robot tasks (e.g., `navigate_to` service, `grasp_object` action).

**Possible Pitfalls:**
*   **LLM Hallucinations:** The LLM might generate invalid or non-existent function calls. The Action Executor must validate these.
*   **Safety:** The LLM does not inherently understand physical safety. Safety checks must be external to the LLM.
*   **Latency:** Cloud-based LLMs can introduce significant latency.

## 4.2 Exposing Robot Capabilities to the LLM (Tool Description)

For an LLM to generate robot actions, it needs to know what actions the robot can perform. This is done by providing the LLM with a "tool description" or "function schema."

### 4.2.1 Define Your Robot's API

Assume your robot has the following ROS 2 capabilities:

*   **Navigation:** A ROS 2 action `navigate_to` that takes a `location` (string) as a goal.
*   **Manipulation:** A ROS 2 service `grasp_object` that takes an `object_name` (string) as a request and returns success/failure.
*   **Speech Output:** A ROS 2 topic `speak` that takes a `text` (string) to announce something.

You would conceptually describe these to the LLM (often in JSON schema format) like this:

```json
[
  {
    "name": "navigate_to",
    "description": "Navigate the robot to a specified named location.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The named destination, e.g., 'kitchen', 'office', 'charging_station'."
        }
      },
      "required": ["location"]
    }
  },
  {
    "name": "grasp_object",
    "description": "Attempt to grasp a specified object in the robot's current view.",
    "parameters": {
      "type": "object",
      "properties": {
        "object_name": {
          "type": "string",
          "description": "The name of the object to grasp, e.g., 'soda can', 'book', 'mug'."
        }
      },
      "required": ["object_name"]
    }
  },
  {
    "name": "speak",
    "description": "Make the robot say a given text aloud.",
    "parameters": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "The text for the robot to speak."
        }
      },
      "required": ["text"]
    }
  }
]
```
**Explanation:** This schema clearly defines the function names, what they do, and what parameters they require. The LLM uses this information to determine which function(s) to call and with what arguments based on the user's input.

## 4.3 LLM Plan Generation and Execution

Now, let's trace how a human command becomes a robot action.

### 4.3.1 Human Instruction to LLM Function Call

1.  **Human:** "Robot, please go to the kitchen and get me a coke from the fridge."
2.  **LLM Interface:** This instruction, along with the tool descriptions, is sent to the LLM.
3.  **LLM's Response:** The LLM, understanding the available tools, might generate a series of function calls:

    ```json
    [
      {
        "function_call": {
          "name": "navigate_to",
          "arguments": {
            "location": "kitchen"
          }
        }
      },
      {
        "function_call": {
          "name": "grasp_object",
          "arguments": {
            "object_name": "soda can"
          }
        }
      },
      {
        "function_call": {
          "name": "speak",
          "arguments": {
            "text": "I have the soda. Returning now."
          }
        }
      }
    ]
    ```
    **Explanation:** The LLM has successfully decomposed the high-level request into a logical sequence of three distinct robot actions, using the provided API. It also inferred "coke" as "soda can."

### 4.3.2 Action Execution with ROS 2

The generated function calls are then passed to the ROS 2 Action Executor.

1.  **Action Executor:** Receives the LLM's JSON output.
2.  **Validation:** It first validates each function call against the known schema to ensure it's a legitimate command and all required parameters are present.
3.  **ROS 2 Invocation:**
    *   For `navigate_to("kitchen")`: The executor sends a goal to the `navigate_to` ROS 2 action server. It waits for the navigation to complete (success or failure).
    *   For `grasp_object("soda can")`: If navigation succeeds, it calls the `grasp_object` ROS 2 service with "soda can" as the request. It waits for the service response.
    *   For `speak("I have the soda...")`: If grasping succeeds, it publishes "I have the soda. Returning now." to the `speak` ROS 2 topic.
4.  **Feedback to LLM (Optional but Recommended):** The executor can send the success/failure status of each action back to the LLM. If an action fails, the LLM could then attempt to re-plan or ask the human for clarification.

:::info Diagram Placeholder
**Diagram 4.1: LLM-ROS 2 Action Execution Flow**
A flowchart showing:
-   "LLM-Generated Function Calls (JSON)" -> "ROS 2 Action Executor Node"
-   "ROS 2 Action Executor Node" -> "Validate Function Call"
-   If Valid -> "Invoke ROS 2 Action/Service (e.g., `navigate_to` Action, `grasp_object` Service)"
-   "ROS 2 Action/Service" -> "Robot Control System (e.g., Nav2, Manipulation Node)"
-   "Robot Control System" -> "Robot Execution (in Sim/Real)"
-   "Robot Execution Status" -> "ROS 2 Action Executor Node" -> "Feedback to LLM (for replanning)"
:::

## 4.4 Advanced Considerations and Pitfalls

### 4.4.1 State Management and Context

The LLM is stateless by default. For multi-turn conversations or tasks requiring context, the LLM Interface needs to manage the dialogue history and robot's current state (e.g., its location, what it's currently holding) and provide this information in subsequent prompts to the LLM. This "context window" allows for more nuanced and continuous interaction.

### 4.4.2 Human-in-the-Loop Validation

Given the LLM's potential for errors or unsafe suggestions, a human-in-the-loop (HITL) system is often crucial. This could involve:
*   **Plan Preview:** The robot first describes the LLM-generated plan to the human for approval before execution.
*   **Emergency Stop:** Human override capability at any point.
*   **Clarification:** The LLM asking the human for clarification if its confidence in a plan is low.

### 4.4.3 Integrating Perception Feedback

The current setup assumes successful execution of actions. However, real-world robots often fail. Integrating perception feedback (e.g., "did I actually grasp the soda?") is vital. If perception detects a failure, the LLM can be prompted to re-plan based on the new observation.

**Further Resources:**
*   OpenAI Function Calling Documentation: [link]
*   ROS 2 Action Client/Server Tutorials: [link]
*   Research papers on LLMs in Robotics (e.g., SayCan, Google Robotics Transformers): [links to conceptual papers]

By intelligently integrating LLMs with ROS 2 actions, you can create robots that are not only capable of understanding complex commands but can also robustly translate them into meaningful physical interactions, moving towards a new era of intuitive human-robot collaboration.
