# Documentation Restructuring Prompt

**Purpose**: Analyze Victor codebase and restructure documentation to align with OSS best practices
**Execution Strategy**: Maximum parallelism via Task agents for expedited completion

---

## Master Prompt

```
You are tasked with analyzing the Victor AI coding assistant codebase and restructuring its documentation to align with industry-standard open-source software (OSS) repository practices.

Victor is an open-source AI coding assistant supporting 21 LLM providers with 55 specialized tools across 5 domain verticals (Coding, DevOps, RAG, Data Analysis, Research).

## Objectives

1. **Audit existing documentation** - Identify gaps, outdated content, and structural issues
2. **Align with OSS standards** - Restructure to match successful OSS projects (LangChain, FastAPI, etc.)
3. **Improve discoverability** - Ensure new contributors can onboard quickly
4. **Maximize parallelism** - Use Task agents concurrently for granular updates

## OSS Documentation Standards to Follow

### Root-Level Files (Required)
- README.md - Project overview, quickstart, badges, links
- CONTRIBUTING.md - How to contribute, PR process, code style
- CODE_OF_CONDUCT.md - Community guidelines
- LICENSE - Apache 2.0 (already exists)
- CHANGELOG.md - Version history with semantic versioning
- SECURITY.md - Security policy, vulnerability reporting
- CLAUDE.md - AI assistant context (Victor-specific, keep)

### Documentation Directory Structure
```
docs/
├── index.md                    # Landing page
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   ├── configuration.md
│   └── first-steps.md
├── user-guide/
│   ├── cli-usage.md
│   ├── tui-mode.md
│   ├── providers.md
│   ├── tools.md
│   └── workflows.md
├── verticals/
│   ├── coding.md
│   ├── devops.md
│   ├── rag.md
│   ├── data-analysis.md
│   └── research.md
├── architecture/
│   ├── overview.md
│   ├── agent-orchestrator.md
│   ├── provider-system.md
│   ├── tool-system.md
│   ├── workflow-engine.md
│   └── diagrams/
├── api-reference/
│   ├── providers/
│   ├── tools/
│   ├── workflows/
│   └── protocols/
├── development/
│   ├── setup.md
│   ├── testing.md
│   ├── code-style.md
│   ├── adding-providers.md
│   ├── adding-tools.md
│   └── adding-verticals.md
├── deployment/
│   ├── docker.md
│   ├── kubernetes.md
│   └── air-gapped.md
├── tutorials/
│   ├── build-custom-tool.md
│   ├── create-workflow.md
│   └── integrate-provider.md
└── reference/
    ├── cli-commands.md
    ├── configuration-options.md
    ├── environment-variables.md
    └── model-capabilities.md
```

## Execution Plan

Execute the following task groups IN PARALLEL using Task agents. Each group contains independent tasks that can run concurrently.
```

---

## Parallel Task Groups

### PHASE 1: Analysis (Run All in Parallel)

Launch these 6 agents simultaneously:

```python
# Task Group 1A: Documentation Inventory
Task(
    subagent_type="Explore",
    prompt="""
    Inventory all existing documentation in the Victor codebase.

    Search for:
    1. All .md files in the repository
    2. Docstrings in Python modules (sample top 20 modules)
    3. Inline comments with TODO/FIXME/DEPRECATED
    4. README files in subdirectories
    5. YAML files with documentation sections

    Output a structured inventory:
    - File path
    - Type (readme, guide, api-doc, inline)
    - Last modified date
    - Approximate word count
    - Key topics covered

    Save to: docs/analysis_reports/10_doc_inventory.json
    """,
    description="Inventory existing docs"
)

# Task Group 1B: Gap Analysis
Task(
    subagent_type="Explore",
    prompt="""
    Compare Victor's documentation against OSS best practices.

    Check for presence and quality of:
    1. Root files: README.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md
    2. Getting started guides
    3. API reference documentation
    4. Architecture diagrams
    5. Tutorial content
    6. Deployment guides

    For each category, rate:
    - EXISTS: Yes/No/Partial
    - QUALITY: 1-5 (if exists)
    - PRIORITY: Critical/High/Medium/Low

    Save to: docs/analysis_reports/10_doc_gaps.json
    """,
    description="Analyze documentation gaps"
)

# Task Group 1C: Code-to-Doc Coverage
Task(
    subagent_type="Explore",
    prompt="""
    Analyze code documentation coverage.

    For each major module, check:
    1. Module-level docstrings
    2. Class docstrings
    3. Public method docstrings
    4. Type hints presence
    5. Example usage in docstrings

    Modules to analyze:
    - victor/agent/*.py
    - victor/providers/*.py
    - victor/tools/*.py
    - victor/workflows/*.py
    - victor/framework/*.py

    Calculate coverage percentage per module.
    Identify top 10 undocumented public APIs.

    Save to: docs/analysis_reports/10_code_doc_coverage.json
    """,
    description="Analyze code doc coverage"
)

# Task Group 1D: README Quality Audit
Task(
    subagent_type="Explore",
    prompt="""
    Audit the main README.md for OSS best practices.

    Check for:
    1. Project badges (build status, coverage, version, license)
    2. One-liner description
    3. Key features list
    4. Installation instructions
    5. Quickstart code example
    6. Screenshots/GIFs of TUI
    7. Links to documentation
    8. Contributing section
    9. License section
    10. Acknowledgments/Credits

    Rate each section 1-5 and provide improvement suggestions.
    Compare against: LangChain, FastAPI, Hugging Face Transformers READMEs.

    Save to: docs/analysis_reports/10_readme_audit.json
    """,
    description="Audit README quality"
)

# Task Group 1E: Vertical Documentation Audit
Task(
    subagent_type="Explore",
    prompt="""
    Audit documentation for each vertical.

    For each vertical (coding, devops, rag, dataanalysis, research):
    1. Check for dedicated documentation
    2. List available tools with descriptions
    3. Document workflow definitions
    4. Identify example usage
    5. Check for integration guides

    Identify:
    - Verticals with good documentation
    - Verticals needing documentation
    - Cross-vertical documentation opportunities

    Save to: docs/analysis_reports/10_vertical_docs.json
    """,
    description="Audit vertical docs"
)

# Task Group 1F: API Surface Analysis
Task(
    subagent_type="Explore",
    prompt="""
    Analyze the public API surface that needs documentation.

    Identify all public interfaces:
    1. CLI commands (from typer app)
    2. Python API entry points
    3. HTTP API endpoints (if api server enabled)
    4. MCP server capabilities
    5. Configuration options

    For each interface:
    - Name and location
    - Current documentation status
    - Complexity (simple/medium/complex)
    - Usage frequency (based on code references)

    Save to: docs/analysis_reports/10_api_surface.json
    """,
    description="Analyze API surface"
)
```

---

### PHASE 2: Root Files Creation (Run All in Parallel)

Launch these 5 agents simultaneously after Phase 1:

```python
# Task 2A: Create CONTRIBUTING.md
Task(
    subagent_type="general-purpose",
    prompt="""
    Create a comprehensive CONTRIBUTING.md for Victor.

    Include sections:
    1. Welcome message and project overview
    2. Ways to contribute (code, docs, issues, discussions)
    3. Development setup
       - Prerequisites (Python 3.11+, pip, git)
       - Installation steps
       - Running tests
    4. Code style guidelines
       - Black formatting (line length 100)
       - Ruff linting
       - Type hints required
       - Google-style docstrings
    5. Pull request process
       - Branch naming (feature/, fix/, docs/)
       - Commit message format
       - PR template checklist
       - Review process
    6. Issue guidelines
       - Bug report template
       - Feature request template
    7. Testing requirements
       - Unit test coverage
       - Integration test markers
    8. Documentation contributions

    Reference CLAUDE.md for project specifics.
    Follow GitHub's recommended CONTRIBUTING.md structure.

    Write to: CONTRIBUTING.md
    """,
    description="Create CONTRIBUTING.md"
)

# Task 2B: Create CODE_OF_CONDUCT.md
Task(
    subagent_type="general-purpose",
    prompt="""
    Create CODE_OF_CONDUCT.md using Contributor Covenant v2.1.

    Customize for Victor project:
    1. Use Contributor Covenant 2.1 as base
    2. Add project-specific contact information
    3. Include enforcement guidelines
    4. Add examples of positive behavior
    5. Specify reporting mechanisms

    Contact: Vijaykumar Singh (project maintainer)

    Write to: CODE_OF_CONDUCT.md
    """,
    description="Create CODE_OF_CONDUCT.md"
)

# Task 2C: Create SECURITY.md
Task(
    subagent_type="general-purpose",
    prompt="""
    Create SECURITY.md for Victor.

    Include sections:
    1. Supported versions table
    2. Reporting vulnerabilities
       - Private disclosure process
       - Expected response time
       - What to include in report
    3. Security considerations
       - API key handling
       - Air-gapped mode
       - Sandboxed code execution
       - Provider credential storage
    4. Security best practices for users
    5. Known security limitations

    Write to: SECURITY.md
    """,
    description="Create SECURITY.md"
)

# Task 2D: Create CHANGELOG.md
Task(
    subagent_type="general-purpose",
    prompt="""
    Create CHANGELOG.md following Keep a Changelog format.

    Analyze git history to extract:
    1. Version tags and release dates
    2. Major features added per version
    3. Breaking changes
    4. Bug fixes
    5. Deprecations

    Use format:
    ## [Unreleased]
    ### Added
    ### Changed
    ### Deprecated
    ### Removed
    ### Fixed
    ### Security

    ## [0.5.1] - YYYY-MM-DD
    ...

    Go back at least 5 versions or 3 months of history.

    Write to: CHANGELOG.md
    """,
    description="Create CHANGELOG.md"
)

# Task 2E: Enhance README.md
Task(
    subagent_type="general-purpose",
    prompt="""
    Enhance README.md to match top OSS project standards.

    Current README exists - enhance it with:

    1. Add badges section at top:
       - Python version
       - License
       - Build status (placeholder)
       - Coverage (placeholder)
       - PyPI version

    2. Add eye-catching header with logo placeholder

    3. Ensure these sections exist/are enhanced:
       - One-liner tagline
       - Key features (bullet points with emojis)
       - Quick install: pip install victor-ai
       - Quickstart code example (3-5 lines)
       - TUI screenshot placeholder
       - Supported providers list
       - Documentation links
       - Contributing link
       - License

    4. Add comparison table vs other tools (optional)

    5. Add "Star History" section placeholder

    Read current README.md first, then enhance.
    Write to: README.md
    """,
    description="Enhance README.md"
)
```

---

### PHASE 3: Documentation Structure (Run All in Parallel)

Launch these 8 agents simultaneously:

```python
# Task 3A: Getting Started Docs
Task(
    subagent_type="general-purpose",
    prompt="""
    Create getting-started documentation.

    Create these files:

    1. docs/getting-started/installation.md
       - pip install
       - Optional dependencies
       - Verification steps

    2. docs/getting-started/quickstart.md
       - First run
       - Basic commands
       - Provider setup

    3. docs/getting-started/configuration.md
       - Config file locations
       - Environment variables
       - Settings overview

    Reference CLAUDE.md and existing configs for accuracy.
    Use clear, beginner-friendly language.
    Include code examples with expected output.

    Create directory if needed.
    """,
    description="Create getting-started docs"
)

# Task 3B: User Guide - CLI/TUI
Task(
    subagent_type="general-purpose",
    prompt="""
    Create user guide for CLI and TUI modes.

    Create:
    1. docs/user-guide/cli-usage.md
       - All CLI commands
       - Common flags
       - Examples

    2. docs/user-guide/tui-mode.md
       - TUI navigation
       - Keyboard shortcuts
       - Panel descriptions

    Extract CLI commands from victor/cli/*.py
    Document all Typer commands and options.
    """,
    description="Create CLI/TUI docs"
)

# Task 3C: Provider Documentation
Task(
    subagent_type="general-purpose",
    prompt="""
    Create comprehensive provider documentation.

    Create docs/user-guide/providers.md covering:

    1. Provider overview
    2. For each provider:
       - Setup instructions
       - Required environment variables
       - Supported models
       - Special features
       - Limitations

    Providers to document:
    - Anthropic (Claude)
    - OpenAI
    - Google (Gemini)
    - DeepSeek
    - Ollama (local)
    - LMStudio (local)
    - And others from victor/providers/

    Include provider switching mid-conversation feature.
    """,
    description="Create provider docs"
)

# Task 3D: Tools Documentation
Task(
    subagent_type="general-purpose",
    prompt="""
    Create tools documentation.

    Create docs/user-guide/tools.md covering:

    1. Tool system overview
    2. Tool categories:
       - File operations
       - Code analysis
       - Search tools
       - Shell/execution
       - Git operations
       - Web tools

    3. For each tool category:
       - Available tools
       - Parameters
       - Examples
       - Cost tiers

    Extract from victor/tools/*.py
    Include tool selection strategies (keyword, semantic, hybrid).
    """,
    description="Create tools docs"
)

# Task 3E: Workflow Documentation
Task(
    subagent_type="general-purpose",
    prompt="""
    Create workflow documentation.

    Create docs/user-guide/workflows.md covering:

    1. Workflow system overview
    2. YAML workflow syntax
    3. Node types:
       - agent
       - compute
       - condition
       - parallel
       - transform
       - hitl
    4. Escape hatches for complex logic
    5. Built-in workflows per vertical
    6. Creating custom workflows

    Extract from victor/workflows/*.yaml and CLAUDE.md.
    Include complete YAML examples.
    """,
    description="Create workflow docs"
)

# Task 3F: Architecture Overview
Task(
    subagent_type="general-purpose",
    prompt="""
    Create architecture documentation.

    Create docs/architecture/overview.md covering:

    1. High-level architecture diagram (ASCII art)
    2. Core components:
       - AgentOrchestrator (facade)
       - ConversationController
       - ToolPipeline
       - ProviderManager
       - ServiceProvider (DI)
    3. Data flow
    4. Extension points
    5. Design patterns used

    Reference CLAUDE.md architecture section.
    Include Mermaid diagrams where appropriate.
    """,
    description="Create architecture docs"
)

# Task 3G: Vertical-Specific Docs
Task(
    subagent_type="general-purpose",
    prompt="""
    Create documentation for each vertical.

    Create these files:
    1. docs/verticals/coding.md
    2. docs/verticals/devops.md
    3. docs/verticals/rag.md
    4. docs/verticals/data-analysis.md
    5. docs/verticals/research.md

    For each vertical include:
    - Purpose and use cases
    - Available tools
    - Workflows
    - Configuration options
    - Examples

    Extract from victor/{vertical}/ directories.
    """,
    description="Create vertical docs"
)

# Task 3H: Development Setup Guide
Task(
    subagent_type="general-purpose",
    prompt="""
    Create developer documentation.

    Create docs/development/setup.md covering:

    1. Prerequisites
    2. Clone and install (dev mode)
    3. Running tests
       - Unit tests
       - Integration tests
       - Coverage
    4. Code quality checks
       - Black
       - Ruff
       - Mypy
    5. Pre-commit hooks
    6. IDE setup (VS Code recommended settings)

    Create docs/development/testing.md covering:
    - Test structure
    - Fixtures
    - Markers
    - Mocking patterns

    Reference CLAUDE.md and pyproject.toml.
    """,
    description="Create dev docs"
)
```

---

### PHASE 4: API Reference Generation (Run All in Parallel)

Launch these 4 agents simultaneously:

```python
# Task 4A: Provider API Reference
Task(
    subagent_type="general-purpose",
    prompt="""
    Generate API reference for providers.

    Create docs/api-reference/providers/index.md with:
    1. BaseProvider interface
    2. All provider implementations
    3. Method signatures
    4. Parameters and return types
    5. Usage examples

    Extract from:
    - victor/providers/base.py
    - victor/providers/*.py

    Use consistent format for each provider.
    """,
    description="Generate provider API ref"
)

# Task 4B: Tools API Reference
Task(
    subagent_type="general-purpose",
    prompt="""
    Generate API reference for tools.

    Create docs/api-reference/tools/index.md with:
    1. BaseTool interface
    2. Tool registration
    3. All tool implementations
    4. Parameter schemas
    5. Return types

    Extract from:
    - victor/tools/base.py
    - victor/tools/*.py

    Include JSON schema for each tool's parameters.
    """,
    description="Generate tools API ref"
)

# Task 4C: Workflow API Reference
Task(
    subagent_type="general-purpose",
    prompt="""
    Generate API reference for workflows.

    Create docs/api-reference/workflows/index.md with:
    1. WorkflowGraph API
    2. UnifiedWorkflowCompiler API
    3. BaseYAMLWorkflowProvider API
    4. Node type specifications
    5. State management

    Extract from:
    - victor/workflows/*.py
    - victor/framework/workflows/*.py

    Include TypedDict state definitions.
    """,
    description="Generate workflow API ref"
)

# Task 4D: Protocols Reference
Task(
    subagent_type="general-purpose",
    prompt="""
    Generate API reference for protocols.

    Create docs/api-reference/protocols/index.md with:
    1. All Protocol definitions
    2. Interface contracts
    3. Implementation examples
    4. Type hierarchy

    Extract from:
    - victor/protocols/*.py

    Document:
    - ISemanticSearch
    - ITeamCoordinator
    - IAgent
    - LSP types
    - Provider adapters
    """,
    description="Generate protocols API ref"
)
```

---

### PHASE 5: Tutorials and Examples (Run All in Parallel)

Launch these 4 agents simultaneously:

```python
# Task 5A: Build Custom Tool Tutorial
Task(
    subagent_type="general-purpose",
    prompt="""
    Create tutorial for building custom tools.

    Create docs/tutorials/build-custom-tool.md:

    1. Introduction
    2. Tool anatomy
       - BaseTool inheritance
       - Required attributes
       - execute() method
    3. Step-by-step example: Building a "fetch-weather" tool
    4. Parameter validation
    5. Error handling
    6. Cost tier assignment
    7. Registration
    8. Testing the tool
    9. Complete code listing

    Make it practical and copy-paste ready.
    """,
    description="Create tool tutorial"
)

# Task 5B: Create Workflow Tutorial
Task(
    subagent_type="general-purpose",
    prompt="""
    Create tutorial for creating workflows.

    Create docs/tutorials/create-workflow.md:

    1. Introduction to workflows
    2. YAML syntax overview
    3. Step-by-step: Building a code review workflow
       - Define nodes
       - Add conditions
       - Handle errors
       - Add HITL approval
    4. Escape hatches for Python logic
    5. Testing workflows
    6. Deploying workflows
    7. Complete YAML listing

    Include both simple and complex examples.
    """,
    description="Create workflow tutorial"
)

# Task 5C: Integrate Provider Tutorial
Task(
    subagent_type="general-purpose",
    prompt="""
    Create tutorial for integrating new providers.

    Create docs/tutorials/integrate-provider.md:

    1. Provider architecture overview
    2. Step-by-step: Adding a new LLM provider
       - BaseProvider inheritance
       - Required methods
       - Tool calling adapters
       - Streaming support
    3. Model capabilities configuration
    4. Testing the provider
    5. Registration in ProviderRegistry
    6. Complete code listing

    Use a hypothetical "CustomLLM" provider as example.
    """,
    description="Create provider tutorial"
)

# Task 5D: Reference Documentation
Task(
    subagent_type="general-purpose",
    prompt="""
    Create reference documentation.

    Create:
    1. docs/reference/cli-commands.md
       - All victor CLI commands
       - Options and flags
       - Examples

    2. docs/reference/configuration-options.md
       - All settings.py options
       - YAML config format
       - Defaults

    3. docs/reference/environment-variables.md
       - All env vars
       - Descriptions
       - Defaults

    Extract from victor/cli/, victor/config/settings.py.
    """,
    description="Create reference docs"
)
```

---

### PHASE 6: Final Assembly and Validation (Sequential)

```python
# Task 6A: Create Documentation Index
Task(
    subagent_type="general-purpose",
    prompt="""
    Create main documentation index.

    Create docs/index.md with:
    1. Welcome message
    2. What is Victor?
    3. Key features
    4. Quick links to all sections
    5. Getting started path
    6. Version information

    Ensure all links are valid.
    """,
    description="Create docs index"
)

# Task 6B: Validate All Documentation
Task(
    subagent_type="Explore",
    prompt="""
    Validate all documentation created.

    Check:
    1. All markdown files are valid
    2. All internal links work
    3. Code examples are syntactically correct
    4. No placeholder text remains
    5. Consistent formatting
    6. No duplicate content

    Generate validation report.
    Save to: docs/analysis_reports/10_doc_validation.json
    """,
    description="Validate documentation"
)

# Task 6C: Generate Documentation Site Config
Task(
    subagent_type="general-purpose",
    prompt="""
    Create MkDocs configuration for documentation site.

    Create mkdocs.yml with:
    1. Site metadata
    2. Theme (material recommended)
    3. Navigation structure
    4. Plugins (search, etc.)
    5. Markdown extensions

    Ensure all docs are included in nav.
    """,
    description="Create MkDocs config"
)
```

---

## Execution Commands

### Run Phase 1 (Analysis) - 6 Parallel Agents
```bash
# Launch all 6 exploration agents in parallel
# These have no dependencies and can run concurrently
```

### Run Phase 2 (Root Files) - 5 Parallel Agents
```bash
# After Phase 1 completes, launch 5 content creation agents
# These are independent and can run concurrently
```

### Run Phase 3 (Structure) - 8 Parallel Agents
```bash
# Launch 8 documentation structure agents
# Each creates independent documentation sections
```

### Run Phase 4 (API Reference) - 4 Parallel Agents
```bash
# Launch 4 API reference generation agents
# Each handles a different module group
```

### Run Phase 5 (Tutorials) - 4 Parallel Agents
```bash
# Launch 4 tutorial creation agents
# Each creates independent tutorial content
```

### Run Phase 6 (Assembly) - Sequential
```bash
# Run index, validation, and config generation
# These depend on previous phases
```

---

## Parallelism Summary

| Phase | Agents | Dependencies | Est. Time (Parallel) |
|-------|--------|--------------|---------------------|
| Phase 1 | 6 | None | ~5 min |
| Phase 2 | 5 | Phase 1 | ~10 min |
| Phase 3 | 8 | Phase 2 | ~15 min |
| Phase 4 | 4 | Phase 3 | ~10 min |
| Phase 5 | 4 | Phase 3 | ~10 min |
| Phase 6 | 3 | All | ~5 min |

**Total Agents**: 30
**Sequential Time Estimate**: ~4-6 hours
**Parallel Time Estimate**: ~55 minutes (with max parallelism)
**Speedup**: ~5-6x

---

## Success Criteria

1. All root files exist (README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG)
2. Documentation structure matches OSS standards
3. All major features documented
4. Getting started guide complete
5. At least 3 tutorials created
6. API reference covers core modules
7. All internal links valid
8. Documentation site buildable with MkDocs
