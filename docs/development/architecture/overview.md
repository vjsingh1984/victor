# Victor Architecture Overview

**Visual guide to Victor's system architecture and component relationships.**

## System Architecture

```mermaid
flowchart TB
    subgraph Clients["CLIENTS"]
        CLI["CLI/TUI"]
        HTTP["HTTP API"]
        MCP["MCP Server"]
    end

    subgraph Core["ORCHESTRATOR"]
        ORCH["AgentOrchestrator"]
        CONV["ConversationController"]
        TOOL["ToolPipeline"]
        STRM["StreamingController"]
    end

    subgraph Providers["PROVIDERS (22)"]
        ANT["Anthropic"]
        OAI["OpenAI"]
        OLL["Ollama"]
        MORE["..."]
    end

    subgraph Verticals["VERTICALS (9)"]
        COD["Coding"]
        DEV["DevOps"]
        RAG["RAG"]
        DAT["Data Analysis"]
        RES["Research"]
        SEC["Security"]
        IAC["IaC"]
        CLS["Classification"]
        BEN["Benchmark"]
    end

    subgraph Tools["TOOLS (33 modules)"]
        FILE["File Ops"]
        GIT["Git"]
        SHELL["Shell"]
        WEB["Web"]
        SEARCH["Search"]
    end

    subgraph Teams["TEAMS"]
        SEQ["Sequential"]
        PAR["Parallel"]
        HIER["Hierarchical"]
        PIPE["Pipeline"]
    end

    subgraph State["STATE (4 scopes)"]
        WFS["Workflow"]
        CONVS["Conversation"]
        TMS["Team"]
        GLS["Global"]
    end

    Clients --> Core
    Core --> Providers
    Core --> Verticals
    Core --> Tools
    Core --> Teams
    Core --> State

    style Clients fill:#e0e7ff,stroke:#4f46e5
    style Core fill:#d1fae5,stroke:#10b981
    style Providers fill:#fef3c7,stroke:#f59e0b
    style Verticals fill:#fce7f3,stroke:#ec4899
    style Tools fill:#cffafe,stroke:#06b6d4
    style Teams fill:#e0e7ff,stroke:#6366f1
    style State fill:#f3e8ff,stroke:#a855f7
```

## Request Flow

```
┌──────────┐     ┌─────────────────────────────────────────────────────────┐
│  User    │────▶│                   AgentOrchestrator                    │
└──────────┘     │  ┌─────────────────────────────────────────────────┐  │
                 │  │         VerticalIntegrationPipeline              │  │
                 │  │  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │  │
                 │  │  │   Tools   │  │ Prompts  │  │   Config     │  │  │
                 │  │  │  Handler  │  │ Handler  │  │   Handler    │  │  │
                 │  │  └───────────┘  └──────────┘  └──────────────┘  │  │
                 │  └─────────────────────────────────────────────────┘  │
                 └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │                      VerticalBase                       │
                 │  Coding | DevOps | RAG | DataAnalysis | Research       │
                 │  Security | IaC | Classification | Benchmark           │
                 └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │                    Provider                              │
                 │         (Anthropic | OpenAI | Ollama | ...)            │
                 └─────────────────────────────────────────────────────────┘
```

## Component Quick Reference

| Layer | Components | Purpose |
|-------|------------|---------|
| **Clients** | CLI, HTTP API, MCP Server, VS Code | User interaction |
| **Orchestrator** | AgentOrchestrator, Controllers | Coordinate execution |
| **Providers** | 22 LLM providers | Model abstraction |
| **Tools** | 33 tool modules | Capability execution |
| **Workflows** | StateGraph, YAML DSL | Multi-step processes |
| **Teams** | 4 formations (seq/par/hier/pipe) | Multi-agent coordination |
| **State** | 4 scopes (workflow/conv/team/global) | Unified state management |
| **Verticals** | 9 built-in + custom | Domain specialization |

## Verticals Overview

| Vertical | Use Case |
|----------|----------|
| **Coding** | Code analysis, refactoring, testing |
| **DevOps** | Docker, CI/CD, infrastructure |
| **RAG** | Document retrieval, vector search |
| **DataAnalysis** | Pandas, visualization, statistics |
| **Research** | Web search, synthesis, citations |
| **Security** | Vulnerability scanning, audit, compliance |
| **IaC** | Infrastructure as Code management |
| **Classification** | Text/data classification pipelines |
| **Benchmark** | Agent evaluation and benchmarking |

## Provider Support

| Provider | Tool Calling | Streaming | Air-Gapped |
|----------|--------------|-----------|------------|
| Anthropic | ✅ | ✅ | ❌ |
| OpenAI | ✅ | ✅ | ❌ |
| Google Gemini | ✅ | ✅ | ❌ |
| Ollama | ✅ | ✅ | ✅ |
| LM Studio | ✅ | ✅ | ✅ |
| vLLM | ✅ | ✅ | ✅ |
| ...and 16 more | | | |

## Tool Categories

| Category | Examples | Cost Tier |
|----------|----------|-----------|
| **Filesystem** | read, write, edit, grep | FREE |
| **Git** | status, log, diff, commit | LOW |
| **Execution** | shell, python, docker | MEDIUM |
| **Search** | code_search, symbol, refs | LOW |
| **Web** | web_search, web_fetch | MEDIUM |
| **Analysis** | review, test_gen | HIGH |

## Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **SRP** | Each StepHandler handles one concern |
| **OCP** | ExtensionHandlerRegistry for pluggable components |
| **LSP** | Protocol-based interfaces (SubAgentContext, CapabilityRegistry) |
| **ISP** | Focused protocols, minimal dependencies |
| **DIP** | Protocol-first capability invocation |

## Deep Links

- [Components Reference](components.md) - Detailed component documentation
- [Data Flow](data-flow.md) - Request execution and event flow
- [State Machine](state-machine.md) - Conversation stage management
- [Vertical Integration](framework-vertical-integration.md) - Extension protocols
