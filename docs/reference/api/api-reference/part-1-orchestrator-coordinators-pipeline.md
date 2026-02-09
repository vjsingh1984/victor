# Victor AI 0.5.0 API Reference - Part 1

**Part 1 of 2:** AgentOrchestrator, Coordinators, and Intelligent Pipeline

---

## Navigation

- **[Part 1: Orchestrator, Coordinators, Pipeline](#)** (Current)
- [Part 2: Provider, Tools, Conversation, Streaming](part-2-provider-tools-conversation-streaming.md)
- [**Complete Reference](../API_REFERENCE.md)**

---

> **Note**: This legacy API documentation is retained for reference. For current docs, see `docs/reference/api/`.

Complete API reference documentation for Victor AI's public interfaces.

**Table of Contents**
- [AgentOrchestrator](#agentorchestrator)
- [Coordinators](#coordinators)
- [Intelligent Pipeline](#intelligent-pipeline)
- [Provider Management](#provider-management) *(in Part 2)*
- [Tool System](#tool-system) *(in Part 2)*
- [Conversation Management](#conversation-management) *(in Part 2)*
- [Streaming](#streaming) *(in Part 2)*
- [Vertical Base](#vertical-base) *(in Part 2)*

---

## AgentOrchestrator

The main facade for Victor AI's agent capabilities.

### Overview

`AgentOrchestrator` is a facade pattern implementation that coordinates all agent operations through specialized coordinators. It provides a unified interface for:

- Multi-turn conversations
- Tool execution and budgeting
- Provider management and switching
- State tracking and persistence
- Streaming and batch responses
- Multi-agent team coordination

### Architecture

```
AgentOrchestrator (Facade)
├── ConversationCoordinator     - Message history, context tracking
├── ToolExecutionCoordinator    - Tool validation, execution, budgeting
├── PromptCoordinator           - System prompt assembly
├── StateCoordinator            - Conversation stage management
├── ProviderCoordinator         - Provider lifecycle and switching
├── StreamingCoordinator        - Response processing for streaming
├── SearchCoordinator           - Semantic and keyword search
├── TeamCoordinator             - Multi-agent coordination
├── CheckpointCoordinator       - State persistence
├── MetricsCoordinator          - Observability and metrics
├── EvaluationCoordinator       - Response validation
├── ResponseCoordinator         - Response formatting
```

[Content continues through Intelligent Pipeline...]


**Reading Time:** 1 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Provider, Tools, Conversation, Streaming](part-2-provider-tools-conversation-streaming.md)**
