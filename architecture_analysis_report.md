# Victor Architecture Analysis: Post-Extraction State & Roadmap

## A. Current-State Architecture Map

The Victor ecosystem currently operates as a **Monolithic Core with Plugin Extensions**, where external verticals are loaded into the core runtime process but remain tightly coupled to internal framework implementation details.

### 1. Component Map

```mermaid
graph TD
    subgraph "Victor Core (victor-ai)"
        CoreRuntime[Orchestrator Runtime]
        ToolSystem[Tool System & Registry]
        WorkflowEngine[Workflow Engine]
        EventBus[Event Bus (CQRS)]
        
        subgraph "Integration Layer"
            VerticalRegistry[VerticalRegistry]
            ExtensionLoader[VerticalExtensionLoader]
            BaseClasses[VerticalBase (God Class)]
        end
    end

    subgraph "External Verticals"
        CodingVertical[victor-coding]
        ResearchVertical[victor-research]
        DevOpsVertical[victor-devops]
    end

    CodingVertical -->|Inherits| BaseClasses
    ResearchVertical -->|Inherits| BaseClasses
    DevOpsVertical -->|Inherits| BaseClasses

    CoreRuntime -->|Loads via entry_points| VerticalRegistry
    VerticalRegistry -->|Discovers| CodingVertical
    VerticalRegistry -->|Discovers| ResearchVertical
```

### 2. Execution Flow (Vertical Loading)

1.  **Discovery**: `VerticalRegistry.discover_external_verticals()` scans `sys.modules` and `entry_points(group="victor.verticals")`.
2.  **Validation**: Checks inheritance from `VerticalBase` and `VERTICAL_API_VERSION`.
3.  **Registration**: Valid classes are stored in `VerticalRegistry._registry`.
4.  **Activation**:
    *   User requests an agent with a vertical (e.g., `Agent.create(vertical="coding")`).
    *   `Agent` calls `VerticalRegistry.get("coding")`.
    *   `VerticalBase.get_config()` is called, triggering:
        *   `get_tools()`
        *   `get_system_prompt()`
        *   `get_extensions()` (lazy loads middleware, safety, etc.)
5.  **Runtime**: The `Orchestrator` uses the loaded `VerticalConfig` to configure the agent's context, tools, and system prompt.

### 3. Key Integration Files
*   **Registry**: `victor/core/verticals/base.py` (Central registry & base class)
*   **Loader**: `victor/core/verticals/extension_loader.py` (Lazy loading of extensions)
*   **Contract**: `victor/core/verticals/protocols/` (Emerging interfaces)
*   **Entry Points**: `pyproject.toml` in vertical repos (defines `victor.verticals` group)

---

## B. Decoupling Assessment & Findings

### 1. Critical Coupling Points (Severity: High)
*   **"God Class" Inheritance**: Verticals must inherit from `VerticalBase`, which is a concrete class in the core runtime (`victor.core`). This ties verticals to the *implementation* of the core, not just an interface.
    *   *Risk*: Changes to `VerticalBase` logic (e.g., caching, config generation) break all verticals.
    *   *Violation*: Dependency Inversion Principle (DIP). High-level modules (Verticals) depend on low-level details (Core implementation).
*   **Runtime Dependency**: Verticals depend on `victor-ai` package. This pulls in the entire framework (orchestrator, database, CLI) just to define a plugin.
    *   *Risk*: Bloated vertical environments; version conflicts if Core updates internal deps (e.g., `pydantic`, `sqlalchemy`).

### 2. Leaked Abstractions (Severity: Medium)
*   **ToolSet Leaking**: `VerticalBase.get_tools()` returns a `List[str]`, which creates a loose coupling to tool names. However, `VerticalConfig` uses `victor.framework.tools.ToolSet`, exposing internal tool configuration logic to verticals.
*   **Direct Imports**: Verticals likely import utilities like `LazyProperty` or `ServiceContainer` from `victor.core`, creating hidden dependencies on internal infrastructure.

### 3. Generic Capabilities in Verticals (Refactoring Candidates)
*   **Language Server Protocol (LSP)**: `victor-coding` implements LSP client logic. This is a generic "Language Intelligence" capability that could be useful for a "Data Analysis" vertical (SQL/Python analysis).
    *   *Recommendation*: Move LSP infrastructure to `victor.framework.intelligence`.
*   **Git Operations**: `victor-coding` likely has advanced Git handling.
    *   *Recommendation*: Promote to `victor.framework.scm` if reusable.

---

## C. SOLID Evaluation

| Principle | Status | Violation / Fix |
| :--- | :--- | :--- |
| **SRP** | ⚠️ | **Violation**: `VerticalBase` handles Metadata, Loading, Workflow, and Config. <br>**Fix**: Split into `VerticalDefinition` (Data), `VerticalLoader` (Service), and `VerticalRuntime` (Logic). |
| **OCP** | ✅ | **Good**: Extensions (Middleware, Safety) are loaded dynamically via `ExtensionLoader`, allowing behavior addition without modifying Core. |
| **LSP** | ⚠️ | **Violation**: `VerticalBase` overrides often change behavior (e.g., `get_tools` returning different types in past versions). <br>**Fix**: Enforce stricter Protocol definitions for all extension points. |
| **ISP** | ❌ | **Violation**: Verticals implement one massive `VerticalBase` interface. <br>**Fix**: Break down into `ToolProvider`, `PromptProvider`, `WorkflowProvider` interfaces. Verticals implement only what they need. |
| **DIP** | ❌ | **Violation**: Verticals depend on concrete `victor.core` classes. <br>**Fix**: Introduce `victor-sdk` (pure abstract interfaces). Core and Verticals both depend on SDK. |

---

## D. Target Architecture (The "Plugin Protocol")

### 1. New Artifact Structure
*   **`victor-sdk` (New Package)**:
    *   Contains *only* Protocols, Abstract Base Classes, and Data Models (Pydantic).
    *   **No runtime logic**. No heavy dependencies (numpy, pandas, torch).
    *   Release cycle: Slow, stable, strictly semantic versioned.
*   **`victor-runtime` (Renamed Core)**:
    *   Implements the Orchestrator, Database, CLI.
    *   Depends on `victor-sdk`.
*   **`victor-vertical-X` (Verticals)**:
    *   Depends *only* on `victor-sdk`.
    *   Implements interfaces defined in SDK.

### 2. The Bridge Pattern
Instead of inheritance (`class MyVertical(VerticalBase)`), use composition/registration:

```python
# In victor-sdk
class VerticalManifest(BaseModel):
    name: str
    version: str
    description: str

class ToolProvider(Protocol):
    def get_tools(self) -> List[ToolDefinition]: ...

# In victor-coding
manifest = VerticalManifest(name="coding", ...)
def get_tools(): return [...]

# Registration (declarative)
# pyproject.toml
# [project.entry-points."victor.plugins"]
# coding = "victor_coding:plugin_factory"
```

---

## E. Implementation Roadmap

### Phase 1: SDK Extraction (Weeks 1-3)
*   **Goal**: Create `victor-sdk` package.
*   **Steps**:
    1.  Copy `victor.core.verticals.protocols` and `victor.core.vertical_types` to new repo/package.
    2.  Create Abstract Base Classes (ABCs) in SDK to mirror `VerticalBase` (but pure abstract).
    3.  Publish `victor-sdk` (alpha).

### Phase 2: Core Refactoring (Weeks 4-6)
*   **Goal**: Make Core depend on SDK.
*   **Steps**:
    1.  Update `victor-ai` to depend on `victor-sdk`.
    2.  Refactor `VerticalRegistry` to accept objects implementing SDK protocols.
    3.  Add compatibility shim: `class VerticalBase(SdkVerticalBase): ...` to support existing verticals.

### Phase 3: Vertical Migration (Weeks 7-10)
*   **Goal**: Update Verticals to use SDK.
*   **Steps**:
    1.  Update `victor-coding` to depend on `victor-sdk`.
    2.  Change inheritance to SDK interfaces.
    3.  Remove `victor-ai` dependency from vertical.

### Phase 4: Runtime Separation (Week 11+)
*   **Goal**: Full process isolation (Optional/Advanced).
*   **Concept**: Load verticals in separate processes/containers communicating via gRPC/MCP (Model Context Protocol).

---

## F. Comparative Positioning

| Dimension | Victor | LangGraph | CrewAI | LangChain | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Modularity** | 8 | 9 | 6 | 5 | Victor's vertical system is cleaner than LC's "everything bucket" but trails LangGraph's node purity. |
| **Type Safety** | 9 | 8 | 5 | 4 | Heavy use of Protocols and Pydantic in Core; very strong typing. |
| **Plugin Arch** | 7 | 8 | 6 | 5 | Good entry_points usage, but heavily coupled to Core runtime currently. |
| **Observability** | 8 | 7 | 4 | 6 | Built-in CQRS event bus is superior for enterprise auditing. |
| **Scalability** | 7 | 8 | 5 | 4 | Async-first core is good; monolithic startup is the main bottleneck. |
| **Weighted Score**| **7.8** | **8.0** | **5.2** | **4.8** | **Strong architectural foundation, needs SDK decoupling.** |

*Weights: Modularity (30%), Type Safety (20%), Plugin Arch (20%), Observability (15%), Scalability (15%)*
