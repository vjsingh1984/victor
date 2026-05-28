mermaid
flowchart TB
    subgraph L1["L1 - Client Surface"]
        CLI["CLI / TUI\nvictor/ui/"]
        API["HTTP API\nvictor/integrations/api/"]
        MCP["MCP Server\nvictor/integrations/mcp/"]
        VSC["VS Code\nvscode-victor/"]
    end
    subgraph L2["L2 - Framework API"]
        AGENT["Agent\nvictor/framework/agent.py"]
        SG["StateGraph\nvictor/framework/graph.py"]
        WE["WorkflowEngine\nvictor/framework/workflow_engine.py"]
        TOOLS["Tool Registry\nvictor/framework/tools.py"]
    end
    subgraph L3["L3 - Runtime"]
        ORC["AgentOrchestrator\nvictor/agent/orchestrator.py"]
        SVC["6 Canonical Services\nvictor/agent/services/"]
        AL["AgenticLoop\nvictor/framework/agentic_loop.py"]
        EXC["ExecutionContext\nvictor/runtime/context.py"]
    end
    subgraph L4["L4 - Infrastructure"]
        PROV["Providers 24+\nvictor/providers/"]
        TMOD["Tools 34+\nvictor/tools/"]
        ST["State\nvictor/state/"]
        DB["Database\nvictor/core/database.py"]
        CFG["Config\nvictor/config/settings.py"]
    end
    L1 -->|"VictorClient"| L2
    L2 -->|"AgentFactory"| L3
    L3 -->|"Services"| L4
    style L1 fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b
    style L2 fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style L3 fill:#fef3c7,stroke:#f59e0b,color:#78350f
    style L4 fill:#d1fae5,stroke:#10b981,color:#064e3b