# Victor Architecture (Overview)

## System view

```text
User
  -> CLI/TUI/API
  -> Agent Orchestrator
  -> Provider Adapter (local or cloud)
  -> Tooling + Workflows
  -> Storage (sessions, profiles, embeddings)
```

## Core components

| Component | Responsibility | Primary entry points |
|-----------|----------------|----------------------|
| **Agent** | Public API for chat/run/stream | `victor/framework/agent.py` |
| **AgentOrchestrator** | Conversation loop and tool routing | `victor/agent/orchestrator.py` |
| **Providers** | LLM I/O, auth, retry, streaming | `victor/providers/` |
| **Tooling** | File, git, test, search, web tools | `victor/framework/tools.py` |
| **Workflow Engine** | YAML/graph workflows | `victor/framework/workflow_engine.py` |
| **StateGraph** | Execution runtime for workflows | `victor/framework/graph.py` |
| **Verticals** | Domain presets (coding, RAG, etc.) | `victor/verticals/` |
| **UI** | CLI/TUI and API server | `victor/ui/`, `victor/server/` |

## Data flows

**Single-agent run**

1. User prompt enters CLI/TUI/API.
2. Orchestrator calls provider, handles tool calls, updates state.
3. Tools execute; results feed back into the loop.
4. Final response returns to the user.

**Workflow execution**

1. Workflow YAML compiled to a StateGraph.
2. Nodes run in order with state transitions.
3. Outputs are merged and returned.

## Extension points

- **Providers**: add LLM backends via provider adapters.
- **Tools**: register new tool categories and implementations.
- **Verticals**: bundle tools, prompts, and workflows.
- **Workflows**: author YAML or programmatic graphs.

## Design principles

- Clear separation of orchestration, provider I/O, and tooling.
- Provider-independent context management.
- Opt-in complexity: workflows and teams when needed.
- **Two-layer coordinator architecture**: Application-specific vs framework-agnostic

## Coordinator Architecture

Victor employs a **two-layer coordinator architecture** that separates application-specific orchestration from
  framework-agnostic workflow infrastructure. This design follows SOLID principles and enables clear separation of
  concerns.

### Two-Layer Design

```text
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│                  victor/agent/coordinators/                     │
│                                                                   │
│  Manages AI agent conversation lifecycle and orchestration      │
│                                                                   │
│  • ChatCoordinator: LLM chat & streaming                       │
│  • ToolCoordinator: Tool validation & execution                 │
│  • ContextCoordinator: Context management                       │
│  • AnalyticsCoordinator: Session metrics                        │
│  • PromptCoordinator: System prompt building                    │
│  • SessionCoordinator: Session lifecycle                        │
│  • ProviderCoordinator: Provider switching                      │
│  • ModeCoordinator: Agent modes (build/plan/explore)            │
│  • ConfigCoordinator: Configuration loading                     │
│  • ToolSelectionCoordinator: Semantic tool selection            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ uses
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRAMEWORK LAYER                             │
│                 victor/framework/coordinators/                  │
│                                                                   │
│  Provides domain-agnostic workflow infrastructure                │
│                                                                   │
│  • YAMLWorkflowCoordinator: YAML workflow execution             │
│  • GraphExecutionCoordinator: StateGraph execution              │
│  • HITLCoordinator: Human-in-the-loop integration               │
│  • CacheCoordinator: Workflow caching                           │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Separation?

**Application Layer** (`victor/agent/coordinators/`):
- Manages Victor-specific business logic (conversation lifecycle, tool orchestration)
- Handles user interactions and session management
- Coordinates between multiple concerns (chat, tools, context, analytics)
- Changes when Victor's application requirements change

**Framework Layer** (`victor/framework/coordinators/`):
- Provides reusable workflow infrastructure (YAML execution, StateGraph runtime)
- Domain-agnostic (no Victor-specific logic)
- Used across all verticals (Coding, DevOps, RAG, DataAnalysis, Research)
- Can be used by external verticals and third-party packages

### Key Benefits

1. **Single Responsibility**: Each coordinator has one clear purpose
2. **Layered Architecture**: Application logic builds on framework foundation
3. **Reusability**: Framework coordinators work across all verticals
4. **Testability**: Coordinators can be tested independently
5. **Maintainability**: Clear boundaries reduce coupling

### Further Reading

- [Coordinator Architecture (Detailed)](coordinator_separation.md) - Complete explanation with examples
- [Architecture Diagrams](diagrams/coordinators.mmd) - Visual representations
- [Coordinator Quick Reference](coordinator_based_architecture.md) - Quick lookup guide

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 2 min
