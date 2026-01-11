# Victor Architecture Overview

This document provides a comprehensive overview of Victor's architecture, covering high-level design, core components, key patterns, data flow, and extension points.

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Core Components](#core-components)
- [Key Patterns](#key-patterns)
- [Data Flow](#data-flow)
- [Extension Points](#extension-points)
- [Design Principles](#design-principles)
- [Related Documentation](#related-documentation)

---

## High-Level Architecture

Victor is a provider-agnostic AI coding assistant supporting 21 LLM providers with 55+ specialized tools across 6 domain verticals. The architecture follows a layered design with clear separation of concerns.

### System Architecture Diagram

```
+-------------------------------------------------------------------------+
|                              CLIENTS                                      |
|   +----------+    +-----------+    +-----------+    +------------+       |
|   | CLI/TUI  |    | VS Code   |    |MCP Server |    | API Server |       |
|   +----+-----+    | (HTTP)    |    +-----+-----+    +-----+------+       |
|        |          +-----+-----+          |                |              |
+--------|----------------|----------------|----------------|---------------+
         |                |                |                |
         v                v                v                v
+-------------------------------------------------------------------------+
|                     AGENT ORCHESTRATOR (Facade)                          |
|                                                                          |
|   Delegates to:                                                          |
|   +---------------------+  +---------------+  +-------------------+      |
|   |ConversationController|  | ToolPipeline |  |StreamingController|      |
|   +---------------------+  +---------------+  +-------------------+      |
|   +---------------+  +---------------+  +-----------------+              |
|   |ProviderManager|  | ToolRegistrar |  |  TaskAnalyzer   |              |
|   +---------------+  +---------------+  +-----------------+              |
+-----------------------------+--------------------------------------------+
                              |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
+----------------+   +----------------+   +-------------------+
|   PROVIDERS    |   |     TOOLS      |   |    WORKFLOWS      |
|      (21)      |   |      (55+)     |   |    StateGraph     |
|                |   |                |   |    + YAML         |
| - Anthropic    |   | - File Ops     |   |                   |
| - OpenAI       |   | - Git          |   | +---------------+ |
| - Google       |   | - Shell        |   | |UnifiedCompiler| |
| - Ollama       |   | - Web          |   | +---------------+ |
| - DeepSeek     |   | - Search       |   |                   |
| - 16 more...   |   | - Analysis     |   | +---------------+ |
+----------------+   +----------------+   | |WorkflowEngine | |
                                          | +---------------+ |
         +--------------------------------+-------------------+
         |
         v
+-------------------------------------------------------------------------+
|                           VERTICALS (6)                                  |
|                                                                          |
|   +----------+  +----------+  +------+  +-------------+  +----------+   |
|   |  Coding  |  | DevOps   |  | RAG  |  |DataAnalysis |  | Research |   |
|   +----------+  +----------+  +------+  +-------------+  +----------+   |
|   +------------+                                                         |
|   | Benchmark  |                                                         |
|   +------------+                                                         |
+-------------------------------------------------------------------------+
```

### Mermaid Diagram

```mermaid
flowchart TB
    subgraph Clients["CLIENTS"]
        CLI["CLI/TUI"]
        HTTP["HTTP API"]
        MCP["MCP Server"]
        VSCODE["VS Code Extension"]
    end

    subgraph Orchestrator["AGENT ORCHESTRATOR (Facade)"]
        ORC["AgentOrchestrator"]
        CC["ConversationController"]
        TP["ToolPipeline"]
        SC["StreamingController"]
        PM["ProviderManager"]
        TR["ToolRegistrar"]
    end

    subgraph Providers["PROVIDERS (21)"]
        ANT["Anthropic"]
        OAI["OpenAI"]
        GGL["Google"]
        OLL["Ollama"]
        MORE["..."]
    end

    subgraph Tools["TOOLS (55+)"]
        FILE["File Ops"]
        GIT["Git"]
        SHELL["Shell"]
        WEB["Web"]
        SEARCH["Search"]
    end

    subgraph Workflows["WORKFLOWS"]
        SG["StateGraph DSL"]
        YAML["YAML Workflows"]
        UC["UnifiedCompiler"]
    end

    subgraph Verticals["VERTICALS (6)"]
        COD["Coding"]
        DEV["DevOps"]
        RAG["RAG"]
        DATA["Data Analysis"]
        RES["Research"]
        BENCH["Benchmark"]
    end

    Clients --> Orchestrator
    ORC --> CC
    ORC --> TP
    ORC --> SC
    ORC --> PM
    ORC --> TR
    Orchestrator --> Providers
    Orchestrator --> Tools
    Orchestrator --> Workflows
    Orchestrator --> Verticals

    style Clients fill:#e0e7ff,stroke:#4f46e5
    style Orchestrator fill:#d1fae5,stroke:#10b981
    style Providers fill:#fef3c7,stroke:#f59e0b
    style Tools fill:#cffafe,stroke:#06b6d4
    style Workflows fill:#fce7f3,stroke:#ec4899
    style Verticals fill:#f3e8ff,stroke:#a855f7
```

---

## Core Components

### AgentOrchestrator

**Location:** `victor/agent/orchestrator.py`

The AgentOrchestrator is the central **Facade** that coordinates all other components. It acts as a thin coordination layer, delegating work to specialized components.

**Responsibilities:**
- High-level chat flow coordination
- Configuration loading and validation
- Session lifecycle management
- Provider/model switching hooks

**Key Methods:**
- `run()` - Main entry point for chat sessions
- `process_message()` - Handle single message processing
- `cancel()` - Graceful cancellation of operations

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(
    provider_name="anthropic",
    model="claude-sonnet-4-5",
    settings=settings,
)
result = await orchestrator.process_message("Analyze this code")
```

### ConversationController

**Location:** `victor/agent/conversation_controller.py`

Manages the conversation state, message history, and context tracking throughout a session.

**Responsibilities:**
- Message history management
- Context window tracking
- Stage-based state transitions
- Context retrieval for prompts

**Key Methods:**
- `add_message()` - Add message to history
- `get_context()` - Retrieve conversation context
- `get_stage()` - Get current conversation stage

### ToolPipeline

**Location:** `victor/agent/tool_pipeline.py`

Handles tool validation, selection, and execution with budget enforcement.

**Responsibilities:**
- Tool validation against schemas
- Tool execution coordination
- Budget tracking and enforcement
- Result processing and formatting

**Key Methods:**
- `execute()` - Execute a tool call
- `validate()` - Validate tool parameters
- `get_available_tools()` - List tools for current context

### ProviderManager

**Location:** `victor/agent/provider_manager.py`

Manages LLM provider initialization, switching, health checks, and fallback strategies.

**Responsibilities:**
- Provider initialization
- Mid-conversation provider switching
- Health monitoring and circuit breaker
- Fallback provider selection

**Key Methods:**
- `get_provider()` - Get current provider instance
- `switch()` - Switch to different provider/model
- `check_health()` - Provider health check

### ServiceProvider

**Location:** `victor/agent/service_provider.py`

Implements dependency injection for component management.

**Responsibilities:**
- Component registration and resolution
- Lifecycle management
- Singleton pattern enforcement

```python
from victor.agent.service_provider import ServiceProvider

provider = ServiceProvider()
tool_registry = provider.resolve(ToolRegistry)
```

### ToolRegistrar

**Location:** `victor/agent/tool_registrar.py`

Handles dynamic tool discovery, registration, and plugin integration.

**Responsibilities:**
- Default tool registration
- Plugin discovery and loading
- MCP (Model Context Protocol) integration
- Tool filtering and selection

---

## Key Patterns

### Facade Pattern (Orchestrator)

The AgentOrchestrator implements the Facade pattern, providing a simplified interface to the complex subsystem of components.

```
                    +-------------------+
     User Request   |                   |
    --------------> | AgentOrchestrator |
                    |     (Facade)      |
                    +--------+----------+
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
+------------------+ +---------------+ +-----------------+
|Conversation      | | ToolPipeline  | | ProviderManager |
|Controller        | |               | |                 |
+------------------+ +---------------+ +-----------------+
```

**Benefits:**
- Simplified client interaction
- Reduced coupling between subsystems
- Easy to swap implementations

### Protocol-Based Design (ISP Compliance)

Victor uses Python Protocols for interface segregation, ensuring components depend only on what they need.

```python
# victor/agent/subagents/protocols.py
class SubAgentContext(Protocol):
    """Minimal interface for SubAgent dependencies."""

    @property
    def settings(self) -> Any: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def tool_registry(self) -> Any: ...
```

**Key Protocols:**
- `SubAgentContext` - ISP-compliant SubAgent dependencies
- `CapabilityRegistryProtocol` - Capability discovery
- `OrchestratorVerticalProtocol` - Vertical integration
- `StepHandlerProtocol` - Handler substitutability

### YAML-First Configuration

Workflows and configurations prefer YAML for structure with Python escape hatches for complex logic.

```yaml
# victor/{vertical}/workflows/example.yaml
workflows:
  example_workflow:
    nodes:
      - id: start
        type: agent
        role: analyzer
        goal: "Analyze the input"
        next: [process]

      - id: process
        type: compute
        handler: process_data  # Python escape hatch
        next: [check_quality]

      - id: check_quality
        type: condition
        condition: "quality_check"  # Python function
        branches:
          "pass": complete
          "fail": retry
```

### Vertical Architecture

Self-contained domain modules that encapsulate tools, prompts, workflows, and configurations.

```
victor/{vertical}/
    __init__.py           # Vertical class definition
    assistant.py          # Main VerticalBase implementation
    safety.py             # Safety patterns
    prompts.py            # Domain-specific prompts
    escape_hatches.py     # YAML workflow conditions
    workflows/
        __init__.py
        workflow.yaml
```

---

## Data Flow

### Request Processing Flow

```
+--------+    +-------------+    +----------+    +-------+    +--------+
|  User  |--->| Orchestrator|--->| Provider |--->| LLM   |--->| Tools  |
+--------+    +-------------+    +----------+    +-------+    +--------+
                    |                                ^             |
                    |                                |             |
                    +--------------------------------+-------------+
                              Tool Results
```

### Detailed Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant TS as ToolSelector
    participant P as Provider
    participant LLM as LLM API
    participant T as Tools

    U->>O: User Query
    O->>TS: Select relevant tools
    TS-->>O: Tool list
    O->>P: Send query + tools
    P->>LLM: API call
    LLM-->>P: Tool call request
    P-->>O: Parse tool calls
    O->>T: Execute tool(s)
    T-->>O: Tool results
    O->>P: Send results
    P->>LLM: Continue
    LLM-->>P: Final response
    P-->>O: Response
    O-->>U: Display response
```

### Data Flow by Layer

| Layer | Input | Output | Components |
|-------|-------|--------|------------|
| **Client** | User input | Displayed response | CLI/TUI, HTTP, MCP |
| **Orchestrator** | Message | Structured response | Orchestrator, Controllers |
| **Provider** | Messages + Tools | Completion/Tool calls | ProviderManager, Adapters |
| **Tool** | Tool parameters | Execution result | ToolPipeline, Tools |
| **Vertical** | Context | Domain config | VerticalLoader, Extensions |

---

## Extension Points

### Custom Providers

Create new LLM providers by inheriting from `BaseProvider`.

**Location:** `victor/providers/base.py`

```python
from victor.providers.base import BaseProvider, ChatResponse

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    async def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        # Implementation
        ...

    async def stream_chat(self, messages, tools=None, **kwargs):
        # Streaming implementation
        ...

    def supports_tools(self) -> bool:
        return True
```

**Registration:**
```python
# victor/providers/registry.py
from victor.providers.registry import ProviderRegistry
ProviderRegistry.register("my_provider", MyProvider)
```

### Custom Tools

Create new tools by inheriting from `BaseTool`.

**Location:** `victor/tools/base.py`

```python
from victor.tools.base import BaseTool, CostTier, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    cost_tier = CostTier.LOW

    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input parameter"}
        },
        "required": ["input"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        # Implementation
        return ToolResult(success=True, data={"result": "..."})
```

### Custom Verticals

Create domain-specific assistants by inheriting from `VerticalBase`.

**Location:** `victor/core/verticals/base.py`

```python
from victor.core.verticals import VerticalBase

class SecurityAssistant(VerticalBase):
    name = "security"
    description = "Security analysis assistant"

    @classmethod
    def get_tools(cls) -> list[str]:
        return ["read", "grep", "shell", "web_search"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a security analyst..."
```

**External Plugin Registration:**
```toml
# pyproject.toml
[project.entry-points."victor.verticals"]
security = "victor_security:SecurityAssistant"
```

### Workflow Escape Hatches

Add Python logic to YAML workflows for complex conditions.

```python
# victor/{vertical}/escape_hatches.py

def quality_check(ctx: dict) -> str:
    """Escape hatch for quality checking."""
    score = ctx.get("quality_score", 0)
    if score >= 0.9:
        return "high_quality"
    elif score >= 0.5:
        return "acceptable"
    return "needs_improvement"

CONDITIONS = {
    "quality_check": quality_check,
}
```

---

## Design Principles

### SOLID Compliance

Victor's architecture adheres to SOLID principles:

| Principle | Implementation |
|-----------|----------------|
| **Single Responsibility (SRP)** | Each StepHandler handles one concern; Orchestrator is a thin facade |
| **Open/Closed (OCP)** | ExtensionHandlerRegistry for pluggable components; Plugin system for providers/tools |
| **Liskov Substitution (LSP)** | Protocol-based interfaces ensure substitutability |
| **Interface Segregation (ISP)** | Focused protocols like `SubAgentContext` |
| **Dependency Inversion (DIP)** | Protocol-first capability invocation |

### Provider Agnosticism

Victor supports 21 LLM providers through a unified interface:

- **Cloud Providers:** Anthropic, OpenAI, Google, Azure, AWS Bedrock, Cohere
- **Local Providers:** Ollama, LM Studio, vLLM (air-gapped capable)
- **Specialized:** Groq (fast inference), DeepSeek (thinking tags)

Switch providers mid-conversation without losing context:
```
/provider openai --model gpt-4o
```

### Air-Gapped Capability

When `airgapped_mode=True`:
- Only local providers available (Ollama, LM Studio, vLLM)
- No web tools (web_search, web_fetch disabled)
- Local embeddings for semantic search
- Full functionality without internet access

### Performance Optimizations

| Optimization | Impact | Location |
|--------------|--------|----------|
| **Lazy Tool Loading** | Faster startup | `victor/tools/composition/lazy.py` |
| **AOT Manifest Cache** | 50-100ms startup savings | `victor/core/aot_manifest.py` |
| **Extension Caching** | One-time initialization | `VerticalBase._get_cached_extension()` |
| **Two-Level Workflow Cache** | Definition + execution caching | `victor/workflows/unified_compiler.py` |
| **RL Cache Eviction** | Smart cache management | `victor/storage/cache/rl_eviction_policy.py` |

---

## Related Documentation

### Architecture Deep Dives
- [Component Details](../development/architecture/component-details.md) - Detailed component documentation
- [Deep Dive](../development/architecture/deep-dive.md) - Architecture deep dive with diagrams
- [Data Flow](../development/architecture/data-flow.md) - Request execution flow
- [State Machine](../development/architecture/state-machine.md) - Conversation stage management
- [Framework Integration](../development/architecture/framework-vertical-integration.md) - Vertical integration protocols

### Extension Guides
- [Vertical Development](../development/extending/verticals.md) - Creating custom verticals
- [Plugin Development](../development/extending/plugins.md) - Creating plugins

### Reference
- [Tool Catalog](../reference/tools/catalog.md) - Complete tool reference
- [Provider Comparison](../reference/providers/comparison.md) - Provider capabilities
- [Configuration Keys](../reference/configuration/keys.md) - All configuration options

---

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Architecture Pattern** | Facade with extracted components |
| **Provider Count** | 21 (cloud + local) |
| **Tool Count** | 55+ specialized tools |
| **Vertical Count** | 6 built-in (Coding, DevOps, RAG, Data Analysis, Research, Benchmark) |
| **Key Entry Point** | `AgentOrchestrator.process_message()` |
| **Configuration** | YAML profiles + Python settings |
| **Extension Mechanism** | Entry points (`victor.verticals`, `victor.providers`) |
| **Air-Gapped Support** | Yes (Ollama, LM Studio, vLLM) |
