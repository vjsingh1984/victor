# Victor Architecture Diagrams

This file contains Mermaid diagrams for Victor's architecture documentation.

## System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[CLI Mode]
        TUI[TUI Mode]
        API[HTTP API]
        MCP[MCP Server]
    end

    subgraph "Orchestration Layer"
        Orchestrator[AgentOrchestrator<br/>Facade Pattern]
        ConvController[ConversationController]
        ToolPipeline[ToolPipeline]
        StreamingController[StreamingController]
    end

    subgraph "Provider Layer"
        ProviderManager[ProviderManager]
        ProviderRegistry[ProviderRegistry<br/>21 Providers]
    end

    subgraph "Tool Layer"
        ToolRegistry[ToolRegistry<br/>55 Tools]
        ToolGraph[ToolExecutionGraph]
    end

    subgraph "Workflow Layer"
        WorkflowCompiler[UnifiedWorkflowCompiler<br/>YAML-first]
        StateGraph[StateGraph DSL]
        Scheduler[Workflow Scheduler]
    end

    subgraph "Vertical Layer"
        Coding[Coding Vertical]
        DevOps[DevOps Vertical]
        RAG[RAG Vertical]
        DataAnalysis[Data Analysis Vertical]
        Research[Research Vertical]
        Benchmark[Benchmark Vertical]
    end

    CLI --> Orchestrator
    TUI --> Orchestrator
    API --> Orchestrator
    MCP --> Orchestrator

    Orchestrator --> ConvController
    Orchestrator --> ToolPipeline
    Orchestrator --> StreamingController

    ConvController --> ProviderManager
    ToolPipeline --> ToolRegistry
    ToolPipeline --> ToolGraph

    ProviderManager --> ProviderRegistry
    ProviderRegistry --> Anthropic[Anthropic]
    ProviderRegistry --> OpenAI[OpenAI]
    ProviderRegistry --> Google[Google]
    ProviderRegistry --> Local[Local Providers]

    Orchestrator --> WorkflowCompiler
    WorkflowCompiler --> StateGraph
    StateGraph --> Scheduler

    Orchestrator --> Coding
    Orchestrator --> DevOps
    Orchestrator --> RAG
    Orchestrator --> DataAnalysis
    Orchestrator --> Research
    Orchestrator --> Benchmark

    style Orchestrator fill:#6366f1,color:#fff
    style ProviderRegistry fill:#10b981,color:#fff
    style ToolRegistry fill:#f59e0b,color:#fff
    style WorkflowCompiler fill:#8b5cf6,color:#fff
```

## Provider System Architecture

```mermaid
graph LR
    subgraph "Provider Abstraction"
        BaseProvider[BaseProvider<br/>Protocol]
        StreamingProvider[StreamingProvider<br/>Protocol]
        ToolCallingProvider[ToolCallingProvider<br/>Protocol]
    end

    subgraph "Provider Implementations"
        AnthropicProvider[AnthropicProvider]
        OpenAIProvider[OpenAIProvider]
        GoogleProvider[GoogleProvider]
        DeepSeekProvider[DeepSeekProvider]
        LocalProviders[Local Providers<br/>Ollama, LMStudio, vLLM]
    end

    subgraph "Adapters"
        AnthropicAdapter[AnthropicAdapter]
        OpenAIAdapter[OpenAIAdapter]
        UniversalAdapter[UniversalAdapter<br/>For non-tool-calling]
    end

    subgraph "Features"
        ToolCalling[Tool Calling]
        Streaming[Streaming]
        Vision[Vision Support]
        FunctionCalling[Function Calling]
    end

    BaseProvider --> AnthropicProvider
    BaseProvider --> OpenAIProvider
    BaseProvider --> GoogleProvider
    BaseProvider --> DeepSeekProvider
    BaseProvider --> LocalProviders

    AnthropicProvider --> AnthropicAdapter
    OpenAIProvider --> OpenAIAdapter
    LocalProviders --> UniversalAdapter

    AnthropicAdapter --> ToolCalling
    OpenAIAdapter --> FunctionCalling
    AnthropicProvider --> Streaming
    OpenAIProvider --> Streaming
    GoogleProvider --> Vision

    style BaseProvider fill:#6366f1,color:#fff
    style AnthropicProvider fill:#10b981,color:#fff
    style OpenAIProvider fill:#10b981,color:#fff
```

## Tool System Architecture

```mermaid
graph TB
    subgraph "Tool Abstraction"
        BaseTool[BaseTool<br/>Abstract Class]
        ToolDecorator[@tool<br/>Decorator]
    end

    subgraph "Tool Categories"
        FileOps[File Operations<br/>read, write, edit]
        CodeAnalysis[Code Analysis<br/>code_search, overview]
        Execution[Execution<br/>shell, bash]
        Git[Git Operations<br/>git_status, git_commit]
        Web[Web Tools<br/>web_search, web_fetch]
        RAG[RAG Tools<br/>rag_ingest, rag_search]
    end

    subgraph "Tool Properties"
        CostTier[Cost Tier<br/>FREE, LOW, MEDIUM, HIGH]
        AccessMode[Access Mode<br/>PUBLIC, RESTRICTED, PRIVATE]
        ApprovalRequired[Approval Required<br/>for destructive ops]
    end

    subgraph "Tool Selection"
        Keyword[Keyword<br/>Exact match]
        Semantic[Semantic<br/>Embedding-based]
        Hybrid[Hybrid<br/>70% semantic + 30% keyword]
    end

    BaseTool --> FileOps
    BaseTool --> CodeAnalysis
    BaseTool --> Execution
    BaseTool --> Git
    BaseTool --> Web
    BaseTool --> RAG

    FileOps --> CostTier
    FileOps --> AccessMode
    Execution --> ApprovalRequired

    ToolDecorator --> Hybrid

    style BaseTool fill:#6366f1,color:#fff
    style CostTier fill:#f59e0b,color:#fff
    style Hybrid fill:#10b981,color:#fff
```

## Workflow Execution Flow

```mermaid
graph TB
    Start([Start]) --> Load[Load YAML<br/>Workflow Definition]
    Load --> Validate[Validate YAML<br/>Schema Check]
    Validate --> Compile[Compile to<br/>StateGraph]

    Compile --> Checkpoint{Checkpoint<br/>Enabled?}

    Checkpoint -->|Yes| LoadState[Load State<br/>from Checkpoint]
    Checkpoint -->|No| InitState[Initialize<br/>Empty State]

    LoadState --> Execute
    InitState --> Execute[Execute<br/>Workflow]

    Execute --> Node{Node Type?}

    Node -->|Agent| AgentNode[Run LLM<br/>with Tools]
    Node -->|Compute| ComputeNode[Run Python<br/>Handler]
    Node -->|Condition| ConditionNode[Evaluate<br/>Condition]
    Node -->|Parallel| ParallelNode[Run branches<br/>concurrently]
    Node -->|Transform| TransformNode[Transform<br/>State]
    Node -->|HITL| HITLNode[Human Approval<br/>Required]

    AgentNode --> UpdateState
    ComputeNode --> UpdateState
    ConditionNode --> Branch
    ParallelNode --> Join
    TransformNode --> UpdateState
    HITLNode --> Decision

    Decision{Approved?}
    Decision -->|Yes| UpdateState
    Decision -->|No| [Stop/Halt]

    UpdateState[Update State]
    Branch --> UpdateState
    Join --> UpdateState

    UpdateState --> Checkpoint{Checkpoint<br/>Enabled?}
    Checkpoint -->|Yes| SaveState[Save State<br/>to Checkpoint]
    Checkpoint -->|No| Continue
    SaveState --> Continue

    Continue --> More{More<br/>Nodes?}
    More -->|Yes| Node
    More -->|No| End([End])

    style Compile fill:#8b5cf6,color:#fff
    style Execute fill:#10b981,color:#fff
    style HITLNode fill:#f59e0b,color:#fff
```

## Multi-Agent Coordination

```mermaid
graph TB
    subgraph "Team Formation"
        Formation{Team Formation<br/>Strategy}
    end

    subgraph "Team Types"
        Solo[Solo<br/>Single Agent]
        Hierarchy[Hierarchy<br/>Manager + Workers]
        Sequential[Sequential<br/>Agent Chain]
        Debate[Debate<br/>Peer Discussion]
        Consensus[Consensus<br/>Voting]
    end

    subgraph "Coordination"
        Coordinator[TeamCoordinator<br/>Unified Interface]
        Messenger[Messenger<br/>Message Passing]
        Router[Router<br/>Task Distribution]
    end

    subgraph "Agent Roles"
        Planner[Planner Agent<br/>Task Breakdown]
        Researcher[Researcher Agent<br/>Information Gathering]
        Coder[Coder Agent<br/>Implementation]
        Reviewer[Reviewer Agent<br/>Quality Check]
        Tester[Tester Agent<br/>Validation]
    end

    Formation --> Solo
    Formation --> Hierarchy
    Formation --> Sequential
    Formation --> Debate
    Formation --> Consensus

    Coordinator --> Messenger
    Coordinator --> Router

    Messenger --> Planner
    Messenger --> Researcher
    Messenger --> Coder
    Messenger --> Reviewer
    Messenger --> Tester

    Router --> Planner
    Router --> Researcher
    Router --> Coder

    style Coordinator fill:#6366f1,color:#fff
    style Formation fill:#8b5cf6,color:#fff
```

## Dependency Injection Container

```mermaid
graph TB
    subgraph "Service Container"
        Container[ServiceProvider<br/>DI Container]
    end

    subgraph "Core Services"
        Orchestrator[AgentOrchestrator]
        ProviderManager[ProviderManager]
        ToolRegistrar[ToolRegistrar]
        EventRegistry[EventRegistry]
    end

    subgraph "Optional Services"
        Embeddings[EmbeddingService]
        VectorStore[VectorStore]
        GraphStore[GraphStore]
    end

    subgraph "Lifecycle"
        Register[Register<br/>Singleton]
        Resolve[Resolve/<br/>Create]
        Dispose[Dispose/<br/>Cleanup]
    end

    Container --> Register
    Container --> Resolve
    Container --> Dispose

    Register --> Orchestrator
    Register --> ProviderManager
    Register --> ToolRegistrar
    Register --> EventRegistry

    Resolve --> Orchestrator
    Resolve --> ProviderManager
    Resolve --> Embeddings
    Resolve --> VectorStore

    Dispose --> EventRegistry
    Dispose --> VectorStore

    style Container fill:#6366f1,color:#fff
    style Register fill:#10b981,color:#fff
    style Resolve fill:#f59e0b,color:#fff
```

## Event System

```mermaid
graph LR
    subgraph "Event Producers"
        Agent[Agent Actions]
        Tool[Tool Execution]
        Workflow[Workflow Events]
        System[System Events]
    end

    subgraph "Event Bus"
        EventBus[EventBus<br/>Pub/Sub Pattern]
    end

    subgraph "Event Consumers"
        Logger[Event Logger]
        Metrics[Metrics Collector]
        UI[UI Updater]
        Debugger[Debug Handler]
    end

    subgraph "Event Types"
        ToolStart[ToolStartEvent]
        ToolEnd[ToolEndEvent]
        ToolError[ToolErrorEvent]
        LLMStart[LLMStartEvent]
        LLMEnd[LLMEndEvent]
        LLMError[LLMErrorEvent]
    end

    Agent --> EventBus
    Tool --> EventBus
    Workflow --> EventBus
    System --> EventBus

    EventBus --> Logger
    EventBus --> Metrics
    EventBus --> UI
    EventBus --> Debugger

    Tool --> ToolStart
    Tool --> ToolEnd
    Tool --> ToolError

    Agent --> LLMStart
    Agent --> LLMEnd
    Agent --> LLMError

    style EventBus fill:#6366f1,color:#fff
```

## Vertical Architecture

```mermaid
graph TB
    subgraph "Vertical Base"
        VerticalBase[VerticalBase<br/>Abstract Class]
    end

    subgraph "Vertical Components"
        Assistant[Assistant<br/>Main Interface]
        Tools[Tools<br/>Specialized]
        Workflows[Workflows<br/>YAML Definitions]
        Capabilities[Capabilities<br/>Prompt Enhancers]
        Prompts[Prompts<br/>Task Hints]
        Safety[Safety<br/>Validators]
    end

    subgraph "Vertical Instances"
        Coding[Coding<br/>13,452 LOC]
        DevOps[DevOps<br/>Infrastructure]
        RAG[RAG<br/>Retrieval]
        DataAnalysis[DataAnalysis<br/>Pandas/NumPy]
        Research[Research<br/>Web/Search]
        Benchmark[Benchmark<br/>SWE-bench]
    end

    VerticalBase --> Assistant
    VerticalBase --> Tools
    VerticalBase --> Workflows
    VerticalBase --> Capabilities
    VerticalBase --> Prompts
    VerticalBase --> Safety

    Assistant --> Coding
    Assistant --> DevOps
    Assistant --> RAG
    Assistant --> DataAnalysis
    Assistant --> Research
    Assistant --> Benchmark

    style VerticalBase fill:#6366f1,color:#fff
    style Coding fill:#10b981,color:#fff
    style RAG fill:#8b5cf6,color:#fff
```

## Configuration System

```mermaid
graph TB
    subgraph "Config Sources"
        Env[Environment<br/>Variables]
        ConfigYAML[config.yaml<br/>Project Level]
        ProfilesYAML[profiles.yaml<br/>User Level]
        CLAUDE[CLAUDE.md<br/>Context]
    end

    subgraph "Config Layers"
        Settings[Settings<br/>Pydantic Model]
        Profiles[Profile<br/>Manager]
        Loader[YAML<br/>Loader]
    end

    subgraph "Config Categories"
        Provider[Provider Settings<br/>API keys, models]
        Tool[Tool Settings<br/>Budget, selection]
        Workflow[Workflow Settings<br/>Cache, checkpoints]
        Safety[Safety Settings<br/>Approvals, air-gapped]
        Performance[Performance<br/>Timeouts, limits]
    end

    Env --> Settings
    ConfigYAML --> Loader
    ProfilesYAML --> Profiles
    CLAUDE --> Settings

    Settings --> Provider
    Settings --> Tool
    Settings --> Workflow
    Settings --> Safety
    Settings --> Performance

    Profiles --> Provider
    Profiles --> Tool

    style Settings fill:#6366f1,color:#fff
    style Provider fill:#10b981,color:#fff
    style Tool fill:#f59e0b,color:#fff
```

## Testing Architecture

```mermaid
graph TB
    subgraph "Test Levels"
        Unit[Unit Tests<br/>Fast, Isolated]
        Integration[Integration Tests<br/>Component Interaction]
        E2E[E2E Tests<br/>Full Workflows]
    end

    subgraph "Test Tools"
        Pytest[Pytest<br/>Test Runner]
        Mocks[Mocks<br/>unittest.mock]
        Respx[Respx<br/>HTTP Mocking]
        Fixtures[Fixtures<br/>Test Data]
    end

    subgraph "Test Markers"
        UnitMark[@pytest.mark.unit]
        IntegrationMark[@pytest.mark.integration]
        SlowMark[@pytest.mark.slow]
        WorkflowMark[@pytest.mark.workflows]
    end

    Unit --> Pytest
    Integration --> Pytest
    E2E --> Pytest

    Pytest --> UnitMark
    Pytest --> IntegrationMark
    Pytest --> SlowMark
    Pytest --> WorkflowMark

    Unit --> Mocks
    Integration --> Respx
    Integration --> Fixtures

    style Pytest fill:#6366f1,color:#fff
    style Unit fill:#10b981,color:#fff
    style Integration fill:#f59e0b,color:#fff
```

## Data Flow: Tool Execution

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant ToolSelector
    participant Tool as Tool
    participant Provider

    User->>Orchestrator: Request with tools
    Orchestrator->>ToolSelector: Select tools for query

    ToolSelector->>ToolSelector: Semantic search
    ToolSelector->>ToolSelector: Apply budget
    ToolSelector->>ToolSelector: Check permissions

    ToolSelector-->>Orchestrator: Selected tools

    loop For each tool
        Orchestrator->>Tool: execute()
        Tool->>Tool: Validate parameters
        Tool->>Tool: Check approval required
        Tool->>Tool: Execute operation
        Tool-->>Orchestrator: ToolResult
    end

    Orchestrator->>Provider: Format with tool results
    Provider-->>User: Final response

    Note over Orchestrator,Tool: Tools run in parallel when safe
```

## Data Flow: Workflow Execution

```mermaid
sequenceDiagram
    participant User
    participant Compiler
    participant Executor
    participant LLM as Provider
    participant Tools

    User->>Compiler: Load YAML workflow
    Compiler->>Compiler: Validate schema
    Compiler->>Compiler: Build StateGraph
    Compiler-->>User: Compiled workflow

    User->>Executor: Invoke with initial state
    Executor->>Executor: Load checkpoint (if exists)

    loop For each node in graph
        Executor->>Executor: Determine node type
        alt Agent node
            Executor->>LLM: Generate with state
            LLM-->>Executor: Response + tool calls
            Executor->>Tools: Execute tools
            Tools-->>Executor: Tool results
            Executor->>LLM: Continue with tool results
        end

        alt Compute node
            Executor->>Executor: Run handler function
        end

        alt Condition node
            Executor->>Executor: Evaluate condition
            Executor->>Executor: Route to next node
        end

        Executor->>Executor: Update state
        Executor->>Executor: Save checkpoint
    end

    Executor-->>User: Final state + results
```

---

## Usage in Documentation

To use these diagrams in Markdown files:

```markdown
## System Architecture

```mermaid
paste diagram code here
```
```

MkDocs with the Material theme automatically renders Mermaid diagrams.
