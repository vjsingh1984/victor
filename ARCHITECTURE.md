# Victor Architecture (Overview)

## System view

```
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
