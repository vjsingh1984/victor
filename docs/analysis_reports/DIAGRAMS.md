# Victor Codebase Architecture Diagrams

Generated: 2026-01-10

## 1. System Architecture Mindmap

```mermaid
mindmap
  root((Victor AI))
    Core Infrastructure
      victor.core
        Bootstrap
        Container/DI
        Event Sourcing
        Repository Pattern
        CQRS
        Validation
      victor.config
        Settings
        Model Capabilities
        Provider Config
      victor.protocols
        Provider Adapters
        Grounding
        Quality
        Team Coordination
        Search Interfaces
    Agent Layer
      victor.agent
        Orchestrator [Facade]
        Conversation Controller
        Tool Pipeline
        Streaming Controller
        Provider Manager
        Tool Registrar
        Recovery
        RL Components
      victor.teams
        Team Formations
        Unified Coordinator
    Providers
      victor.providers
        Anthropic
        OpenAI
        Ollama
        Google
        +18 others
    Tools & Workflows
      victor.tools
        55 Specialized Tools
        Cache Manager
        Tool Selection
      victor.workflows
        Unified Compiler
        Scheduler
        Versioning
        Cache
    Domain Verticals
      victor.coding
        AST/Tree-sitter
        LSP Integration
        Code Review
        Test Generation
        Refactoring
      victor.devops
        Docker
        Terraform
        CI/CD Tools
      victor.rag
        Document Ingestion
        Vector Search
        Chunking
      victor.dataanalysis
        Pandas Integration
        Visualization
        Statistics
      victor.research
        Web Search
        Citations
        Synthesis
      victor.benchmark
        SWE-bench
        HumanEval
    UI Layer
      victor.ui
        TUI [Textual]
        CLI [Typer]
        Emoji/Icons
        Slash Commands
    Observability
      victor.observability
        Metrics
        Event Bus
        Tracing
    Storage
      victor.storage
        LanceDB
        SQLite
        Caching
```

## 2. High-Level Module Dependency Diagram

```mermaid
graph TB
    subgraph "Entry Points"
        CLI[victor.ui.cli]
        TUI[victor.ui.tui]
        API[victor.api]
    end

    subgraph "Orchestration Layer"
        ORCH[victor.agent.orchestrator]
        CC[Conversation Controller]
        TP[Tool Pipeline]
        SC[Streaming Controller]
        PM[Provider Manager]
    end

    subgraph "Provider Layer"
        PROV[victor.providers]
        ANT[Anthropic]
        OAI[OpenAI]
        OLL[Ollama]
        GGL[Google]
    end

    subgraph "Tool Layer"
        TOOLS[victor.tools]
        CODING_T[Coding Tools]
        DEVOPS_T[DevOps Tools]
        RAG_T[RAG Tools]
    end

    subgraph "Framework Layer"
        FW[victor.framework]
        SG[StateGraph DSL]
        WE[Workflow Engine]
        COORD[Coordinators]
    end

    subgraph "Domain Verticals"
        COD[victor.coding]
        DEV[victor.devops]
        RAG[victor.rag]
        DA[victor.dataanalysis]
        RES[victor.research]
    end

    subgraph "Core Infrastructure"
        CORE[victor.core]
        CONFIG[victor.config]
        PROTO[victor.protocols]
        STORE[victor.storage]
    end

    CLI --> ORCH
    TUI --> ORCH
    API --> ORCH

    ORCH --> CC
    ORCH --> TP
    ORCH --> SC
    ORCH --> PM

    PM --> PROV
    PROV --> ANT
    PROV --> OAI
    PROV --> OLL
    PROV --> GGL

    TP --> TOOLS
    TOOLS --> CODING_T
    TOOLS --> DEVOPS_T
    TOOLS --> RAG_T

    ORCH --> FW
    FW --> SG
    FW --> WE
    FW --> COORD

    CODING_T --> COD
    DEVOPS_T --> DEV
    RAG_T --> RAG

    COD --> CORE
    DEV --> CORE
    RAG --> CORE
    DA --> CORE
    RES --> CORE

    CORE --> CONFIG
    CORE --> PROTO
    CORE --> STORE
```

## 3. Use Case Diagram

```mermaid
graph LR
    subgraph "Users"
        DEV[Developer]
        DEVOPS_U[DevOps Engineer]
        ANALYST[Data Analyst]
        RESEARCHER[Researcher]
    end

    subgraph "Victor System"
        subgraph "Primary Use Cases"
            UC1[Code Assistance]
            UC2[Code Review]
            UC3[Test Generation]
            UC4[Refactoring]
            UC5[Documentation]
        end

        subgraph "DevOps Use Cases"
            UC6[Docker Management]
            UC7[CI/CD Configuration]
            UC8[Infrastructure as Code]
        end

        subgraph "RAG Use Cases"
            UC9[Document Search]
            UC10[Semantic Code Search]
            UC11[Knowledge Retrieval]
        end

        subgraph "Analysis Use Cases"
            UC12[Data Analysis]
            UC13[Visualization]
            UC14[Report Generation]
        end

        subgraph "Research Use Cases"
            UC15[Web Research]
            UC16[Citation Management]
            UC17[Synthesis]
        end

        subgraph "Workflow Use Cases"
            UC18[Custom Workflows]
            UC19[Multi-Agent Teams]
            UC20[Benchmarking]
        end
    end

    DEV --> UC1
    DEV --> UC2
    DEV --> UC3
    DEV --> UC4
    DEV --> UC5
    DEV --> UC10

    DEVOPS_U --> UC6
    DEVOPS_U --> UC7
    DEVOPS_U --> UC8

    ANALYST --> UC9
    ANALYST --> UC12
    ANALYST --> UC13
    ANALYST --> UC14

    RESEARCHER --> UC9
    RESEARCHER --> UC15
    RESEARCHER --> UC16
    RESEARCHER --> UC17

    DEV --> UC18
    DEV --> UC19
    DEV --> UC20
```

## 4. Agent Component Class Diagram

```mermaid
classDiagram
    class AgentOrchestrator {
        +ConversationController controller
        +ToolPipeline pipeline
        +StreamingController streaming
        +ProviderManager provider_manager
        +ToolRegistrar tool_registrar
        +chat()
        +stream_chat()
        +execute_tool()
    }

    class ConversationController {
        +ConversationMemory memory
        +ConversationState state
        +add_message()
        +get_history()
        +clear()
    }

    class ToolPipeline {
        +List~BaseTool~ tools
        +ToolSelector selector
        +execute()
        +select_tools()
    }

    class StreamingController {
        +stream()
        +handle_chunk()
        +finalize()
    }

    class ProviderManager {
        +BaseProvider current_provider
        +switch_provider()
        +get_provider()
    }

    class ToolRegistrar {
        +Dict tools
        +register()
        +get_tool()
        +list_tools()
    }

    class BaseProvider {
        <<abstract>>
        +name: str
        +chat()
        +stream_chat()
        +supports_tools()
    }

    class BaseTool {
        <<abstract>>
        +name: str
        +description: str
        +parameters: dict
        +cost_tier: CostTier
        +execute()
    }

    class ConversationMemory {
        +messages: List
        +add_message()
        +get_messages()
        +summarize()
    }

    AgentOrchestrator --> ConversationController
    AgentOrchestrator --> ToolPipeline
    AgentOrchestrator --> StreamingController
    AgentOrchestrator --> ProviderManager
    AgentOrchestrator --> ToolRegistrar
    ProviderManager --> BaseProvider
    ToolPipeline --> BaseTool
    ConversationController --> ConversationMemory
```

## 5. Framework Component Class Diagram

```mermaid
classDiagram
    class StateGraph {
        +Dict nodes
        +Dict edges
        +str entry_point
        +add_node()
        +add_edge()
        +add_conditional_edges()
        +compile()
    }

    class CompiledGraph {
        +StateGraph graph
        +invoke()
        +stream()
        +get_state()
    }

    class WorkflowEngine {
        +execute()
        +validate()
        +checkpoint()
    }

    class UnifiedWorkflowCompiler {
        +compile_workflow()
        +cache_definition()
        +cache_execution()
    }

    class BaseYAMLWorkflowProvider {
        <<abstract>>
        +load_workflow()
        +compile_workflow()
        +create_executor()
    }

    class TeamFormation {
        +str name
        +List agents
        +coordinator_style
        +create_coordinator()
    }

    class Agent {
        +str name
        +str role
        +List tools
        +run()
    }

    class Task {
        +str description
        +Agent agent
        +execute()
    }

    StateGraph --> CompiledGraph : compiles to
    WorkflowEngine --> CompiledGraph : executes
    UnifiedWorkflowCompiler --> StateGraph : builds
    BaseYAMLWorkflowProvider --> UnifiedWorkflowCompiler : uses
    TeamFormation --> Agent : contains
    Agent --> Task : executes
```

## 6. Provider System Class Diagram

```mermaid
classDiagram
    class BaseProvider {
        <<abstract>>
        +str name
        +chat(messages, tools) Response
        +stream_chat(messages, tools) AsyncIterator
        +supports_tools() bool
        +get_model_info() dict
    }

    class AnthropicProvider {
        +str name = "anthropic"
        +chat()
        +stream_chat()
        +supports_tools()
    }

    class OpenAIProvider {
        +str name = "openai"
        +chat()
        +stream_chat()
        +supports_tools()
    }

    class OllamaProvider {
        +str name = "ollama"
        +chat()
        +stream_chat()
        +supports_tools()
    }

    class GoogleProvider {
        +str name = "google"
        +chat()
        +stream_chat()
        +supports_tools()
    }

    class ProviderRegistry {
        +Dict providers
        +register()
        +get_provider()
        +list_providers()
    }

    class CircuitBreaker {
        +CircuitState state
        +int failure_count
        +call()
        +record_success()
        +record_failure()
    }

    BaseProvider <|-- AnthropicProvider
    BaseProvider <|-- OpenAIProvider
    BaseProvider <|-- OllamaProvider
    BaseProvider <|-- GoogleProvider
    ProviderRegistry --> BaseProvider
    BaseProvider --> CircuitBreaker : protected by
```

## 7. Tool System Class Diagram

```mermaid
classDiagram
    class BaseTool {
        <<abstract>>
        +str name
        +str description
        +dict parameters
        +CostTier cost_tier
        +execute(args) ToolResult
        +validate_args(args) bool
    }

    class CostTier {
        <<enumeration>>
        FREE
        LOW
        MEDIUM
        HIGH
    }

    class ToolResult {
        +bool success
        +Any data
        +str error
        +dict metadata
    }

    class ReadFileTool {
        +name = "read_file"
        +cost_tier = FREE
        +execute()
    }

    class WriteFileTool {
        +name = "write_file"
        +cost_tier = LOW
        +execute()
    }

    class ExecuteCodeTool {
        +name = "execute_code"
        +cost_tier = HIGH
        +execute()
    }

    class WebSearchTool {
        +name = "web_search"
        +cost_tier = MEDIUM
        +execute()
    }

    class ToolSelector {
        +str strategy
        +select_tools(query, tools) List~BaseTool~
    }

    class HybridToolSelector {
        +float semantic_weight
        +float keyword_weight
        +select_tools()
    }

    BaseTool <|-- ReadFileTool
    BaseTool <|-- WriteFileTool
    BaseTool <|-- ExecuteCodeTool
    BaseTool <|-- WebSearchTool
    BaseTool --> CostTier
    BaseTool --> ToolResult : produces
    ToolSelector <|-- HybridToolSelector
    ToolSelector --> BaseTool : selects
```

## 8. Coding Vertical Class Diagram

```mermaid
classDiagram
    class CodingAssistant {
        +CodebaseAnalyzer analyzer
        +ASTParser parser
        +LSPClient lsp
        +review_code()
        +generate_tests()
        +refactor()
    }

    class CodebaseAnalyzer {
        +analyze_structure()
        +find_dependencies()
        +get_symbols()
    }

    class ASTParser {
        +TreeSitter parser
        +parse()
        +get_symbols()
        +query()
    }

    class LSPClient {
        +get_diagnostics()
        +get_completions()
        +get_references()
    }

    class CodeReviewer {
        +review()
        +suggest_improvements()
        +check_style()
    }

    class TestGenerator {
        +generate_unit_tests()
        +generate_integration_tests()
        +get_coverage()
    }

    class Refactorer {
        +rename_symbol()
        +extract_method()
        +inline_variable()
    }

    class LanguageServer {
        <<interface>>
        +initialize()
        +shutdown()
        +textDocument_didOpen()
        +textDocument_completion()
    }

    CodingAssistant --> CodebaseAnalyzer
    CodingAssistant --> ASTParser
    CodingAssistant --> LSPClient
    CodingAssistant --> CodeReviewer
    CodingAssistant --> TestGenerator
    CodingAssistant --> Refactorer
    LSPClient --> LanguageServer
```

## 9. Storage and Persistence Class Diagram

```mermaid
classDiagram
    class StorageBackend {
        <<interface>>
        +save()
        +load()
        +delete()
        +exists()
    }

    class SQLiteBackend {
        +connection
        +save()
        +load()
        +query()
    }

    class LanceDBBackend {
        +table
        +add()
        +search()
        +delete()
    }

    class CacheManager {
        +Dict cache
        +int max_size
        +get()
        +set()
        +evict()
    }

    class EventStore {
        +append()
        +read()
        +get_stream()
    }

    class Repository {
        <<abstract>>
        +StorageBackend backend
        +get()
        +save()
        +delete()
        +list()
    }

    class ConversationRepository {
        +save_conversation()
        +load_conversation()
        +list_sessions()
    }

    class EmbeddingStore {
        +LanceDBBackend backend
        +add_embedding()
        +search_similar()
    }

    StorageBackend <|-- SQLiteBackend
    StorageBackend <|-- LanceDBBackend
    Repository --> StorageBackend
    Repository <|-- ConversationRepository
    EmbeddingStore --> LanceDBBackend
    CacheManager --> StorageBackend
    EventStore --> SQLiteBackend
```

## 10. Module Coupling Heatmap (Conceptual)

```mermaid
graph TB
    subgraph "High Coupling Zone"
        A1[victor.agent] --> A2[victor.tools]
        A1 --> A3[victor.providers]
        A1 --> A4[victor.core]
        A1 --> A5[victor.framework]
    end

    subgraph "Medium Coupling Zone"
        B1[victor.coding] --> A4
        B2[victor.workflows] --> A5
        B3[victor.ui] --> A1
        B4[victor.observability] --> A4
    end

    subgraph "Low Coupling Zone"
        C1[victor.devops] --> A4
        C2[victor.rag] --> A4
        C3[victor.dataanalysis] --> A4
        C4[victor.research] --> A4
    end

    style A1 fill:#ff6b6b
    style A2 fill:#ffa07a
    style A3 fill:#ffa07a
    style A4 fill:#ff6b6b
    style A5 fill:#ffa07a
    style B1 fill:#ffd93d
    style B2 fill:#ffd93d
    style B3 fill:#ffd93d
    style B4 fill:#ffd93d
    style C1 fill:#6bcf6b
    style C2 fill:#6bcf6b
    style C3 fill:#6bcf6b
    style C4 fill:#6bcf6b
```

## Legend

- **Red nodes**: High coupling (>500 cross-module edges)
- **Orange nodes**: Medium-high coupling (200-500 edges)
- **Yellow nodes**: Medium coupling (50-200 edges)
- **Green nodes**: Low coupling (<50 edges)

---

## Key Architectural Insights

1. **Single Giant Component**: 99.7% of nodes are in one weakly connected component, indicating high cohesion
2. **Central Hub**: `victor.agent` is the largest module (4,748 nodes) and primary orchestration point
3. **PageRank Leaders**: `emoji.get`, `event_sourcing.append`, `graph.items` are most referenced
4. **Vertical Independence**: Domain verticals (coding, devops, rag, dataanalysis, research) are relatively loosely coupled to each other
5. **Protocol Abstraction**: `victor.protocols` provides clean interface boundaries
