# Roadmap: Phased Improvements for the Victor Framework

This document outlines a strategic roadmap for the continued development of the Victor framework, focusing on solidifying the core, enhancing the ecosystem, and pushing the boundaries of agentic orchestration.

## Phase 1: Solidify the Core ✅ COMPLETED

**Objective:** Address the identified SOLID violations and improve the developer experience of the core framework. This phase is about building a rock-solid foundation for future growth.

**Status:** All key initiatives completed as of January 9, 2025

**Completed Initiatives:**

1.  **✅ Refactor `CompiledGraph.invoke`:**
    - **Status:** COMPLETE - Decomposed into focused helper classes
    - **Implementation:** Introduced specialized helpers for different concerns:
      - `IterationController`: iteration/recursion limits
      - `TimeoutManager`: timeout tracking
      - `InterruptHandler`: human-in-the-loop interrupts
      - `NodeExecutor`: node execution with COW optimization
      - `CheckpointManager`: state persistence
      - `GraphEventEmitter`: observability events
    - **Result:** Improved readability, maintainability, and testability. SRP compliance achieved.

2.  **✅ Decompose `GraphConfig`:**
    - **Status:** COMPLETE - ISP-compliant configuration
    - **Implementation:** Created focused config classes:
      - `ExecutionConfig`: execution limits
      - `CheckpointConfig`: state persistence
      - `InterruptConfig`: interrupt behavior
      - `PerformanceConfig`: performance optimizations
      - `ObservabilityConfig`: observability and eventing
    - **Result:** Interface Segregation Principle achieved. Clients depend only on configs they use.

3.  **✅ Improve Developer Documentation:**
    - **Status:** COMPLETE - Comprehensive documentation
    - **Implementation:**
      - ARCHITECTURE.md: High-level architecture overview
      - SCALABILITY_AND_PERFORMANCE.md: Honest performance assessment
      - API_KEYS.md: API key management guide
      - Streamlined docs for OSS release (removed 36 internal reports)
    - **Result:** Clear, structured, and honest documentation.

**Note:** RLCheckpointerAdapter remains in framework (correct placement). RL is framework infrastructure for RL-based learning, not a vertical.

## Phase 2: Enhance the Vertical Ecosystem

**Objective:** Make it easier for developers to build, share, and discover new verticals, turning the framework into a thriving ecosystem.

**Status:** In Progress (1 of 3 complete, 2 in progress as of January 9, 2025)

**Completed Initiatives:**

1.  **✅ Vertical Scaffolding Tool:**
    - **Status:** COMPLETE - Full-featured CLI tool available
    - **Implementation:** `victor vertical create` command
      ```bash
      victor vertical create security --description "Security analysis"
      victor vertical create analytics --service-provider
      ```
    - **Features:**
      - Jinja2-based templating system
      - Name validation (Python identifier, no reserved names)
      - Generates complete vertical structure:
        - `__init__.py` - Package initialization
        - `assistant.py` - Main vertical class (VerticalBase)
        - `safety.py` - Safety patterns
        - `prompts.py` - Task type hints
        - `mode_config.py` - Mode configurations
        - `service_provider.py` - DI container (optional)
      - Dry-run mode for preview
      - Force mode for overwriting
    - **Benefit:** Radically simplifies creating new verticals
    - **Templates:** Located in `victor/templates/vertical/`

**In Progress Initiatives:**

2.  **⏳ Vertical Registry and Discovery (50% Complete):**
    - **Status:** IN PROGRESS - Schema complete, CLI commands pending
    - **Completed (January 9, 2025):**
      - victor-vertical.toml specification with Pydantic validation
      - VerticalPackageMetadata schema (package_schema.py)
      - Package metadata validation: authors, dependencies, compatibility, security
      - Template for victor-vertical.toml files
    - **Pending:**
      - `victor vertical install` command (PyPI, git, local)
      - `victor vertical list/search` commands
      - Central registry client (GitHub-based MVP)
      - victor-registry repository setup
    - **Implementation:** Package metadata foundation complete. Next: CLI commands for install/list/search and registry integration.
    - **Benefit:** Creates a marketplace of ideas and reusable components, allowing users to easily find and install pre-built solutions for their needs.

3.  **⏳ Framework Enhancement Proposal Process (30% Complete):**
    - **Status:** IN PROGRESS - Process documentation complete, CLI tooling pending
    - **Completed (January 9, 2025):**
      - FEP-0001: Framework Enhancement Proposal process (meta-FEP)
      - FEP template (fep-0000-template.md) with all required sections
      - FEP README.md with complete process documentation
      - Governance model: rough consensus, 14-day review period
      - FEP types: Standards Track, Informational, Process
      - Numbering scheme and repository structure defined
    - **Pending:**
      - `victor fep create/validate/submit/list/view` CLI commands
      - FEP validation schema implementation
      - GitHub Actions workflow for FEP validation
      - Example FEPs (FEP-0002: Informational, FEP-0003: Standards Track)
      - Integration with CONTRIBUTING.md
    - **Implementation:** Process foundation established. Next: CLI tooling and examples.
    - **Benefit:** Ensures that the framework evolves based on the real-world needs of its users, while maintaining a high standard of quality and architectural coherence.

## Phase 3: Advanced Orchestration and Intelligence (12-18 Months)

**Objective:** Push the boundaries of what's possible with agentic workflows, moving from pre-defined graphs to more dynamic and intelligent orchestration.

**Key Initiatives:**

1.  **Dynamic Graph Generation:**
    - **Action:** Introduce the capability for an agent to dynamically generate or modify its own `StateGraph` at runtime.
    - **Implementation:** This would likely involve creating a "meta-agent" that takes a high-level goal and outputs a `StateGraph` definition, which is then compiled and executed.
    - **Benefit:** A major step towards more autonomous and adaptable agents that can create their own plans to solve novel problems.

2.  **Adaptive Orchestration:**
    - **Action:** Develop a "meta-orchestrator" that can observe the performance of different workflows and automatically optimize them.
    - **Implementation:** This could involve A/B testing different graph structures, dynamically adjusting toolsets based on success rates, or even using reinforcement learning to optimize the flow of control.
    - **Benefit:** Self-optimizing workflows that become more efficient and effective over time.

3.  **Hybrid Orchestration Model:**
    - **Action:** Explore a hybrid orchestration model that combines the explicit, predictable control of `StateGraph` with the dynamic, conversation-driven model of frameworks like AutoGen.
    - **Implementation:** This would allow for workflows that have a structured backbone but can also spawn ad-hoc, multi-agent "sub-teams" to solve specific sub-problems.
    - **Benefit:** The best of both worlds: the reliability of structured workflows and the flexibility of emergent, multi-agent collaboration.

4.  **Enhanced Observability and Debugging:**
    - **Action:** Create a rich, visual debugger for `StateGraph` workflows.
    - **Implementation:** A web-based UI that allows developers to step through the execution of a graph, inspect the full state at each node, set breakpoints, and visualize the flow of control.
    - **Benefit:** Dramatically improves the developer experience of building and debugging complex workflows, making the "inner workings" of the agent transparent and understandable.
