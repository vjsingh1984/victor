# Victor Architecture Map

This document outlines the high-level architecture of the Victor framework, focusing on the core components and their interactions.

## Core Components

The framework is designed around a set of modular, decoupled components that provide a flexible and extensible platform for building AI agents.

### 1. Agent (`victor/framework/agent.py`)

- **Role:** The primary, user-facing entry point for interacting with the framework.
- **Design:** A facade that simplifies the creation and use of agents, abstracting away the underlying complexity of the orchestration and tool systems.
- **Key Abstractions:** `Agent`, `ChatSession`.
- **Interaction:** Users typically create an `Agent` instance and use its `run()`, `stream()`, or `chat()` methods to interact with the LLM. It delegates all the real work to the `AgentOrchestrator`.

### 2. AgentOrchestrator (Referenced, not directly read)

- **Role:** The central "brain" of a single agent. It manages the conversation loop, tool execution, and state management for a single agent instance.
- **Design:** It coordinates the interaction between the language model provider, the tool system, and the state management components. It's the component that actually executes the "ReAct" (Reason-Act) loop.
- **Interaction:** The `Agent` class is a wrapper around an `AgentOrchestrator`.

### 3. Workflow Engine (`victor/framework/workflow_engine.py`)

- **Role:** A high-level facade for executing complex, multi-step, and potentially multi-agent workflows.
- **Design:** It provides a unified API for running workflows defined in YAML or programmatically using the `StateGraph` DSL. It uses a set of `Coordinator` classes to handle different workflow types and features (e.g., `YAMLWorkflowCoordinator`, `GraphExecutionCoordinator`, `HITLCoordinator`).
- **Interaction:** Used for tasks that require more structure than a single ReAct loop, such as a sequence of analysis, coding, and testing steps.

### 4. StateGraph (`victor/framework/graph.py`)

- **Role:** The low-level orchestration engine that powers the `WorkflowEngine`.
- **Design:** A LangGraph-inspired implementation for building cyclic, stateful graphs. It provides a declarative way to define workflows as a set of nodes (functions) and edges (transitions).
- **Key Abstractions:** `StateGraph`, `CompiledGraph`, `Node`, `Edge`.
- **Interaction:** Developers can use `StateGraph` to define complex workflows programmatically. The `WorkflowEngine` compiles these graphs into `CompiledGraph` objects for execution.

### 5. Tool System (`victor/framework/tools.py`)

- **Role:** Defines the set of capabilities that an agent can use.
- **Design:** Based on `ToolSet` and `ToolCategory`, it provides a flexible way to configure the tools available to an agent. The system is designed to be extensible, allowing for the addition of custom tools.
- **Key Abstractions:** `ToolSet`, `ToolCategory`.
- **Interaction:** A `ToolSet` is provided when an `Agent` is created, and the `AgentOrchestrator` uses this set to execute tool calls requested by the LLM.

### 6. Verticals (e.g., `victor/coding`, `victor/research`)

- **Role:** Specialized, domain-specific applications built on top of the core framework.
- **Design:** Verticals package together pre-defined configurations, including specialized toolsets, system prompts, and workflows, to create tailored assistants for specific tasks (e.g., a `CodingAssistant` or a `ResearchAssistant`).
- **Interaction:** A vertical is passed to `Agent.create()` to instantiate a pre-configured agent, simplifying the process of creating specialized agents.

## Data Flows

### Simple Agent Interaction (ReAct Loop)

1.  `User` -> `Agent.run(prompt)`
2.  `Agent` -> `AgentOrchestrator.execute(prompt)`
3.  `AgentOrchestrator` -> `LLMProvider.invoke(prompt)`
4.  `LLMProvider` -> `LLM`
5.  `LLM` -> `AgentOrchestrator` (response may contain a tool call)
6.  `AgentOrchestrator` -> `ToolExecutor.execute(tool_call)` (using the configured `ToolSet`)
7.  `ToolExecutor` -> `AgentOrchestrator` (with tool result)
8.  `AgentOrchestrator` -> `LLMProvider.invoke(prompt_with_tool_result)`
9.  (Repeat 3-8 until task is complete)
10. `AgentOrchestrator` -> `Agent` (with final result)
11. `Agent` -> `User`

### Complex Workflow Execution

1.  `User` -> `WorkflowEngine.execute_yaml(workflow_path, initial_state)`
2.  `WorkflowEngine` -> `YAMLWorkflowCoordinator.execute()`
3.  `YAMLWorkflowCoordinator` -> `UnifiedWorkflowCompiler.compile_yaml()` (compiles YAML into a `StateGraph`)
4.  `UnifiedWorkflowCompiler` -> `StateGraph.compile()` (creates a `CompiledGraph`)
5.  `YAMLWorkflowCoordinator` -> `CompiledGraph.invoke(initial_state)`
6.  `CompiledGraph` executes the nodes and edges of the graph, managing the state and calling node functions.
7.  `CompiledGraph` -> `User` (with final state)
