# Victor Framework Promotion Specification

## Executive Summary

This document identifies generic capabilities that should be promoted from vertical-specific implementations to the Victor framework core, ensuring all verticals benefit from shared, robust infrastructure. The analysis follows SOLID principles to ensure pluggability, extensibility, and alignment with modern AI agent framework best practices.

---

## Table of Contents

1. [Framework Architecture Analysis](#1-framework-architecture-analysis)
2. [Capabilities for Promotion](#2-capabilities-for-promotion)
3. [SOLID Principles Compliance](#3-solid-principles-compliance)
4. [Competitive Analysis](#4-competitive-analysis)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Design Patterns & Best Practices](#6-design-patterns--best-practices)

---

## 1. Framework Architecture Analysis

### 1.1 Current Layer Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           FRAMEWORK LAYER                                     │
│  victor/framework/                                                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Agent     │ │    Task     │ │   Tools     │ │   State     │            │
│  │  (facade)   │ │  (types)    │ │  (presets)  │ │ (observable)│            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
│                          │                                                    │
│                          ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │              VerticalIntegrationPipeline (9 steps)                  │      │
│  └────────────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PROTOCOL LAYER                                      │
│  victor/protocols/                                                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │  Provider   │ │  Grounding  │ │   Quality   │ │    LSP      │            │
│  │  Adapter    │ │  Strategy   │ │  Assessor   │ │   Types     │            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│                           VERTICAL LAYER                                      │
│  victor/verticals/                                                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Coding    │ │   DevOps    │ │    Data     │ │  Research   │            │
│  │  Assistant  │ │  Assistant  │ │  Analysis   │ │  Assistant  │            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│                            AGENT LAYER                                        │
│  victor/agent/                                                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │Orchestrator │ │    Tool     │ │  Streaming  │ │     RL      │            │
│  │  (facade)   │ │  Pipeline   │ │ Controller  │ │ Coordinator │            │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Strengths

| Component | Rating | Notes |
|-----------|--------|-------|
| Protocol Layer | ★★★★★ | Clean SOLID interfaces, 20+ provider adapters |
| RL System | ★★★★★ | 13 learners, event-driven, cross-vertical transfer |
| Framework API | ★★★★☆ | Golden 5-concept API, but escape hatches needed |
| Tool System | ★★★★☆ | 45+ tools, semantic selection, cost tiers |
| Vertical Integration | ★★★☆☆ | Good pipeline, but significant duplication |
| State Management | ★★★☆☆ | Conversation stages, but no LangGraph-style graphs |

### 1.3 Identified Issues

1. **Mode Configuration Duplication**: 4 verticals × ~150 lines = ~600 lines duplicated
2. **Tool Dependency Duplication**: 4 verticals × ~200 lines = ~800 lines duplicated
3. **Missing Graph-Based Workflows**: No DAG execution like LangGraph
4. **Limited Multi-Agent**: Teams exist but not as flexible as CrewAI crews
5. **No Human-in-the-Loop Protocol**: HITL nodes exist in YAML but no framework protocol

---

## 2. Capabilities for Promotion

### 2.1 HIGH PRIORITY - Immediate Promotion

#### 2.1.1 Unified Mode Configuration Factory

**Current State**: Each vertical implements custom ModeConfig dataclass
**Duplication**: ~600 lines across 4 verticals

**Promotion Target**: victor/framework/mode_config.py

```python
@dataclass
class UnifiedModeConfig:
    """Framework-level mode configuration."""
    name: str
    tool_budget: int
    max_iterations: int
    temperature: float = 0.7
    description: str = ""
    allowed_tools: Optional[Set[str]] = None
    allowed_stages: Optional[List[str]] = None  # For DevOps
    exploration_multiplier: float = 1.0
    sandbox_only: bool = False

class ModeConfigFactory:
    """Factory for creating mode configs from vertical data."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedModeConfig:
        """Create config from dictionary (YAML/JSON)."""

    @classmethod
    def create_preset(cls, preset: str) -> UnifiedModeConfig:
        """Create from preset name (fast, default, thorough, etc.)."""
```

**Benefits**:
- Single source of truth for mode structure
- Verticals define only data, not classes
- Enables cross-vertical mode sharing

---

#### 2.1.2 Tool Dependency Factory

**Current State**: Each vertical reimplements BaseToolDependencyProvider
**Duplication**: ~800 lines across 4 verticals

**Promotion Target**: victor/framework/tool_dependencies.py

```python
@dataclass
class ToolDependencySpec:
    """Declarative tool dependency specification."""
    dependencies: List[ToolDependency]
    transitions: Dict[str, List[Tuple[str, float]]]
    clusters: Dict[str, Set[str]]
    sequences: Dict[str, List[str]]
    required_tools: Set[str]
    optional_tools: Set[str]

class ToolDependencyFactory:
    """Factory for creating dependency providers from specs."""

    @classmethod
    def from_spec(cls, spec: ToolDependencySpec) -> BaseToolDependencyProvider:
        """Create provider from declarative spec."""

    @classmethod
    def from_yaml(cls, path: Path) -> BaseToolDependencyProvider:
        """Load from YAML configuration."""
```

**Benefits**:
- Eliminates 80% of tool dependency code in verticals
- Enables YAML-driven configuration
- Supports spec merging for composite verticals

---

#### 2.1.3 Graph-Based Workflow Engine

**Gap**: Victor has YAML workflows but no LangGraph-style DAG execution

**Promotion Target**: victor/framework/graph_engine.py

```python
class WorkflowNode(Protocol):
    """Protocol for workflow nodes."""

    @property
    def node_id(self) -> str: ...

    async def execute(self, state: WorkflowState) -> WorkflowState: ...

    def get_next_nodes(self, state: WorkflowState) -> List[str]: ...

class WorkflowGraph:
    """DAG-based workflow execution engine."""

    def add_node(self, node: WorkflowNode) -> None: ...
    def add_edge(self, from_node: str, to_node: str, condition: Optional[Callable] = None) -> None: ...
    def add_conditional_edges(self, from_node: str, router: Callable[[WorkflowState], str]) -> None: ...

    async def execute(self, initial_state: WorkflowState) -> WorkflowState:
        """Execute workflow with state transitions."""
```

**Competitive Alignment**: Matches LangGraph's graph-based architecture

---

### 2.2 MEDIUM PRIORITY - Phase 2 Promotion

#### 2.2.1 Human-in-the-Loop Protocol

**Current State**: HITL nodes in YAML workflows but no framework protocol

**Promotion Target**: victor/framework/hitl.py

```python
class HITLProtocol(Protocol):
    """Human-in-the-loop interaction protocol."""

    async def request_approval(
        self,
        action: str,
        context: Dict[str, Any],
        timeout: float = 300.0
    ) -> HITLResponse: ...

    async def request_input(
        self,
        prompt: str,
        input_type: HITLInputType,
        validation: Optional[Callable] = None
    ) -> Any: ...
```

---

#### 2.2.2 Multi-Agent Crew System

**Current State**: Teams exist but limited to coding vertical

**Promotion Target**: victor/framework/crews.py

```python
class AgentRole(Protocol):
    """Protocol for agent roles in crews."""

    @property
    def role_name(self) -> str: ...

    @property
    def capabilities(self) -> Set[str]: ...

    async def execute_task(self, task: Task, context: CrewContext) -> TaskResult: ...

class Crew:
    """Multi-agent crew for collaborative task execution."""

    def __init__(self, name: str, formation: CrewFormation):
        self.agents: Dict[str, AgentRole] = {}

    async def execute(self, task: Task) -> CrewResult:
        """Execute task with crew collaboration."""

class CrewFormation(Enum):
    SEQUENTIAL = "sequential"      # Agents work in order
    PARALLEL = "parallel"          # Agents work simultaneously
    HIERARCHICAL = "hierarchical"  # Manager delegates to workers
    CONSENSUS = "consensus"        # Agents vote on decisions
```

**Competitive Alignment**: Matches CrewAI's role-based architecture

---

#### 2.2.3 Unified Safety Scanner

**Promotion Target**: victor/framework/safety.py

```python
class SafetyScannerRegistry:
    """Registry of safety scanners for composition."""

    _scanners: Dict[str, SafetyScanner] = {}

    @classmethod
    def register(cls, name: str, scanner: SafetyScanner) -> None: ...

    @classmethod
    def create_composite(cls, *names: str) -> CompositeSafetyScanner: ...

class SafetyProfile:
    """Declarative safety profile for verticals."""

    def __init__(
        self,
        scanners: List[str],
        additional_patterns: Optional[List[SafetyPattern]] = None
    ): ...
```

---

## 3. SOLID Principles Compliance

### 3.1 Current Compliance Assessment

| Principle | Current | Target | Gap |
|-----------|---------|--------|-----|
| **Single Responsibility** | ★★★★☆ | ★★★★★ | Mode configs do too much |
| **Open/Closed** | ★★★★☆ | ★★★★★ | Some private attr writes |
| **Liskov Substitution** | ★★★★★ | ★★★★★ | Protocols well-designed |
| **Interface Segregation** | ★★★★☆ | ★★★★★ | Some protocols too large |
| **Dependency Inversion** | ★★★★☆ | ★★★★★ | Some concrete dependencies |

### 3.2 Recommended Improvements

#### 3.2.1 Single Responsibility

**Issue**: VerticalBase has 15+ methods
**Fix**: Extract into focused protocols

```python
# After: Focused protocols
class ToolProviderProtocol(Protocol):
    def get_tools(self) -> List[str]: ...

class PromptProviderProtocol(Protocol):
    def get_system_prompt(self) -> str: ...

class SafetyProviderProtocol(Protocol):
    def get_safety_profile(self) -> SafetyProfile: ...
```

#### 3.2.2 Open/Closed

**Issue**: Private attribute writes in VerticalIntegrationPipeline
**Fix**: Use protocol methods exclusively

```python
# After: Protocol-only
class OrchestratorVerticalProtocol(Protocol):
    def set_vertical_context(self, ctx: VerticalContext) -> None: ...

orchestrator.set_vertical_context(context)  # Always protocol method
```

#### 3.2.3 Interface Segregation

**Issue**: OrchestratorProtocol has 50+ methods
**Fix**: Split into focused sub-protocols

```python
class ConversationProtocol(Protocol):
    def add_message(self, msg: Message) -> None: ...

class ToolExecutionProtocol(Protocol):
    def execute_tool(self, name: str, args: Dict) -> ToolResult: ...

class OrchestratorProtocol(ConversationProtocol, ToolExecutionProtocol, ...):
    pass
```

---

## 4. Competitive Analysis

### 4.1 Framework Comparison Matrix

| Feature | Victor | LangGraph | CrewAI | AutoGen |
|---------|--------|-----------|--------|---------|
| **Architecture** | Protocol-First Layered | Graph-Based DAG | Role-Based Crews | Conversational |
| **State Management** | Conversation Stages | Explicit Graph State | Shared Crew Context | Chat History |
| **Multi-Agent** | Teams (Limited) | Nodes as Agents | Full Crew System | Dynamic Roles |
| **Tool System** | 45+ Tools, Semantic Selection | LangChain Tools | Tool Calling | Code Execution |
| **RL Learning** | 13 Learners, Cross-Vertical | None | None | None |
| **Provider Support** | 25+ Providers | Via LangChain | Limited | OpenAI Focus |
| **Air-Gapped Mode** | Full Support | Partial | None | None |
| **Vertical Domains** | 4 Verticals | None (Generic) | None (Generic) | None (Generic) |
| **Enterprise Features** | RBAC, Audit, Safety | Via LangSmith | Enterprise Plan | Limited |

### 4.2 Victor's Unique Differentiators

1. **Reinforcement Learning System**: Only framework with 13+ RL learners
2. **Cross-Vertical Transfer Learning**: Knowledge sharing between domains
3. **25+ Provider Support**: Most comprehensive provider coverage
4. **Air-Gapped Mode**: Enterprise security requirement
5. **Domain Verticals**: Pre-built assistants for coding, DevOps, data analysis
6. **Protocol-First Design**: SOLID-compliant extensibility

### 4.3 Competitive Gaps to Address

| Gap | Competitor Advantage | Victor Solution |
|-----|---------------------|-----------------|
| Graph Workflows | LangGraph DAG engine | Implement WorkflowGraph |
| Role-Based Crews | CrewAI crew system | Promote Teams to framework |
| Visual Debugging | LangSmith tracing | Enhance observability |
| Low-Code Builder | Langflow UI | Future: Victor Studio |

### 4.4 Strategic Positioning

```
                    HIGH FLEXIBILITY
                          │
         LangGraph        │         AutoGen
         (Graph Control)  │     (Conversational)
                          │
    ──────────────────────┼──────────────────────
                          │
         Victor           │         CrewAI
     (Protocol + RL)      │     (Role-Based)
                          │
                    HIGH STRUCTURE
```

**Victor's Position**: High structure with protocol-first design AND unique RL capabilities

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Sprint 1-2)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Unified Mode Config Factory | HIGH | 3 days | Eliminate 600 lines |
| Tool Dependency Factory | HIGH | 5 days | Eliminate 800 lines |
| Protocol Refactoring | MEDIUM | 3 days | Better extensibility |

### Phase 2: Competitive Features (Sprint 3-4)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Graph Workflow Engine | HIGH | 8 days | Match LangGraph |
| Crew System Promotion | HIGH | 5 days | Match CrewAI |
| HITL Protocol | MEDIUM | 3 days | Enterprise workflows |

### Phase 3: Polish (Sprint 5-6)

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Vertical Template Generator | LOW | 3 days | Developer experience |
| Transfer Learning API | LOW | 5 days | RL enhancement |
| Documentation Update | MEDIUM | 5 days | Adoption |

---

## 6. Design Patterns & Best Practices

### 6.1 Recommended Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Protocol First** | All public interfaces | WorkflowNode(Protocol) |
| **Factory** | Object creation | ModeConfigFactory |
| **Registry** | Dynamic lookup | SafetyScannerRegistry |
| **Composite** | Aggregation | CompositeSafetyScanner |
| **Strategy** | Pluggable algorithms | Tool selection strategies |
| **Observer** | Event dispatch | RL hooks system |
| **Facade** | Complex subsystems | AgentOrchestrator |
| **Template Method** | Extension skeleton | VerticalBase |

### 6.2 Anti-Patterns to Avoid

| Anti-Pattern | Current Occurrence | Fix |
|--------------|-------------------|-----|
| Private Attr Writes | VerticalIntegrationPipeline | Use protocol methods |
| God Class | AgentOrchestrator (100+ deps) | Continue extraction |
| Duplicate Code | Mode configs, tool deps | Use factories |
| Concrete Dependencies | Some direct imports | Use DI container |

---

## 7. Conclusion

### Key Recommendations

1. **Immediate**: Implement UnifiedModeConfig and ToolDependencyFactory to eliminate ~1400 lines of duplication

2. **Short-term**: Add WorkflowGraph for LangGraph-competitive DAG execution

3. **Medium-term**: Promote Teams to framework-level Crew system for CrewAI competitiveness

4. **Long-term**: Build Victor Studio for low-code agent building

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Vertical code duplication | ~40% | <10% |
| New vertical creation time | 2-3 days | 2-3 hours |
| Protocol coverage | 70% | 95% |
| Competitive feature parity | 60% | 90% |

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [DataCamp Framework Comparison](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)
- [Turing AI Frameworks Guide](https://www.turing.com/resources/ai-agent-frameworks)

---

*Document Version: 1.0*
*Last Updated: 2025-12-29*
