# Vertical Development Guide - Part 2

**Part 2 of 4:** Chain Registry Best Practices and Persona Design Guidelines

---

## Navigation

- [Part 1: Capability & Middleware](part-1-capability-middleware.md)
- **[Part 2: Chain Registry & Personas](#)** (Current)
- [Part 3: Complete Workflow Example](part-3-complete-workflow-example.md)
- [Part 4: Appendix & Conclusion](part-4-appendix-conclusion.md)
- [**Complete Guide**](../VERTICAL_DEVELOPMENT_GUIDE.md)

---

## 3. Chain Registry Best Practices

The ChainRegistry provides versioned, discoverable storage for LCEL-composed tool chains.

### 3.1 Chain Registration

#### Basic Registration

```python
from victor.framework.chains.registry import ChainRegistry, get_chain_registry

registry = get_chain_registry()

# Register a chain
registry.register_chain(
    name="my_chain",
    version="0.5.0",
    chain=my_lcel_chain,
    category="editing",  # exploration, editing, analysis, testing, other
    description="Description of what this chain does",
    tags=["tag1", "tag2"],
    author="Your Name",
    deprecated=False,
)
```

#### Decorator Registration

```python
from victor.framework.chains import chain

@chain(
    name="safe_edit_chain",
    version="0.5.0",
    category="editing",
    description="Safe edit with verification",
    tags=["edit", "verification"],
)
def create_safe_edit_chain():
    """Create a safe edit chain."""
    from victor.tools.composition import RunnableSequence

    return (
        RunnableSequence()
        | read_file
        | validate_syntax
        | apply_edit
        | verify_result
    )
```

### 3.2 Namespace Organization

**By Vertical**:
```python
# victor/coding/chains.py
def register_coding_chains():
    registry = get_chain_registry()

    registry.register_chain(
        name="coding.analyze_function",
        version="0.5.0",
        chain=analyze_function_chain,
        category="analysis",
        description="Analyze a function's structure and behavior",
    )

    registry.register_chain(
        name="coding.safe_edit",
        version="0.5.0",
        chain=safe_edit_chain,
        category="editing",
        description="Edit code with validation and verification",
    )
```

**By Category**:
```python
# victor/framework/chains/exploration.py
EXPLORATION_CHAINS = {
    "explore_file": explore_file_chain,
    "search_codebase": search_codebase_chain,
    "find_symbols": find_symbols_chain,
}

def register_exploration_chains():
    registry = get_chain_registry()
    for name, chain in EXPLORATION_CHAINS.items():
        registry.register_chain(
            name=f"exploration.{name}",
            version="0.5.0",
            chain=chain,
            category="exploration",
        )
```

### 3.3 Semantic Versioning

Chains use SemVer for versioning:

```python
# MAJOR.MINOR.PATCH
# - MAJOR: Breaking changes to chain interface/output
# - MINOR: New features, backward compatible
# - PATCH: Bug fixes, internal changes

registry.register_chain(
    name="my_chain",
    version="2.1.3",  # Valid SemVer
    chain=my_chain,
    category="editing",
)
```

**Version retrieval**:
```python
# Get latest version
chain = registry.get_chain("my_chain")

# Get specific version
chain = registry.get_chain("my_chain", version="1.5.0")

# Get latest version string
version = registry.get_chain_version("my_chain")

# List all versions
versions = registry.list_chain_versions("my_chain")
# Returns: ["2.1.3", "2.1.2", "2.1.1", "2.0.0", "1.5.0", ...]
```

### 3.4 Lazy Loading Patterns

```python
class LazyChainRegistry:
    """Lazy-loading chain registry."""

    def __init__(self):
        self._factories: Dict[str, Callable[[], Runnable]] = {}
        self._loaded: Dict[str, Runnable] = {}

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Runnable],
    ) -> None:
        """Register a chain factory for lazy loading."""
        self._factories[name] = factory

    def get_chain(self, name: str) -> Optional[Runnable]:
        """Get chain, loading if necessary."""
        if name not in self._loaded:
            if name not in self._factories:
                return None
            self._loaded[name] = self._factories[name]()

        return self._loaded[name]
```

**Usage**:
```python
# Register factory
registry.register_factory("my_chain", lambda: create_complex_chain())

# Chain is only created when first accessed
chain = registry.get_chain("my_chain")  # Lazy creation
```

### 3.5 Metadata and Discovery

```python
from victor.framework.chains.registry import ChainMetadata

# Get metadata
metadata = registry.get_chain_metadata("my_chain")
print(f"Description: {metadata.description}")
print(f"Tags: {metadata.tags}")
print(f"Author: {metadata.author}")
print(f"Deprecated: {metadata.deprecated}")

# List chains by category
editing_chains = registry.list_chains(category="editing")
# Returns: ["safe_edit", "batch_edit", "refactor", ...]

# Search by tags
def find_chains_by_tag(tag: str) -> List[str]:
    """Find all chains with a specific tag."""
    matching = []
    for name in registry.list_chains():
        metadata = registry.get_chain_metadata(name)
        if tag in metadata.tags:
            matching.append(name)
    return matching
```

### 3.6 Real-World Example: Coding Chains

```python
# victor/coding/composed_chains.py
from victor.tools.composition import (
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
)

# Read and analyze file in parallel
analyze_file_chain = (
    RunnableParallel({
        "content": read_file,
        "symbols": extract_symbols,
        "structure": parse_ast,
    })
    | RunnableLambda(lambda x: {
        "summary": summarize_analysis(x),
        "issues": detect_issues(x),
    })
)

# Safe edit with verification
safe_edit_chain = (
    RunnableSequence()
    | read_file
    | validate_syntax
    | apply_edit
    | validate_syntax
    | RunnableLambda(lambda x: {
        "success": x.get("syntax_valid", False),
        "content": x.get("content"),
    })
)

# Register chains
from victor.framework.chains.registry import get_chain_registry

registry = get_chain_registry()
registry.register_chain(
    name="coding.analyze_file",
    version="0.5.0",
    chain=analyze_file_chain,
    category="analysis",
    description="Analyze file structure and detect issues",
    tags=["analysis", "ast", "validation"],
)

registry.register_chain(
    name="coding.safe_edit",
    version="0.5.0",
    chain=safe_edit_chain,
    category="editing",
    description="Edit file with syntax validation",
    tags=["edit", "validation", "syntax"],
)
```

---

## 4. Persona Design Guidelines

Personas define agent characteristics for multi-agent team collaboration.

### 4.1 Persona Structure

```python
from victor.framework.multi_agent import (
    PersonaTraits,
    CommunicationStyle,
    ExpertiseLevel,
)

persona = PersonaTraits(
    name="Security Auditor",
    role="security_reviewer",
    description="Identifies vulnerabilities and security issues in code",
    communication_style=CommunicationStyle.FORMAL,
    expertise_level=ExpertiseLevel.SPECIALIST,
    verbosity=0.5,
    strengths=["vulnerability_detection", "threat_modeling"],
    weaknesses=["performance_optimization"],
    preferred_tools=["static_analysis", "dependency_check"],
    risk_tolerance=0.2,  # Low risk tolerance
    creativity=0.3,       # Low creativity (follows protocols)
    custom_traits={
        "certification": "CISSP",
        "focus_areas": ["OWASP_TOP_10", "crypto"],
    },
)
```

### 4.2 Communication Styles

```python
from victor.framework.multi_agent import CommunicationStyle

CommunicationStyle.FORMAL
# Formal, professional communication with proper grammar

CommunicationStyle.CASUAL
# Relaxed, conversational tone

CommunicationStyle.TECHNICAL
# Precise, technical language (default for most agents)

CommunicationStyle.CONCISE
# Brief, to-the-point responses
```

**Choosing a style**:
- **FORMAL**: Documentation, client-facing output, compliance reports
- **CASUAL**: Internal collaboration, brainstorming, exploratory tasks
- **TECHNICAL**: Code reviews, technical analysis, debugging
- **CONCISE**: Status updates, summaries, quick checks

### 4.3 Expertise Levels

```python
from victor.framework.multi_agent import ExpertiseLevel

ExpertiseLevel.NOVICE
# Entry-level, needs guidance

ExpertiseLevel.INTERMEDIATE
# Working knowledge, practical experience

ExpertiseLevel.EXPERT
# Deep expertise, comprehensive understanding

ExpertiseLevel.SPECIALIST
# Highly specialized in narrow domain
```

**Level selection guide**:
- **NOVICE**: Training agents, learning mode, tutorial assistants
- **INTERMEDIATE**: General-purpose agents, routine tasks
- **EXPERT**: Domain specialists, complex problem solving
- **SPECIALIST**: Deep niche expertise (e.g., cryptography, ML ops)

### 4.4 Team Templates

```python
from victor.framework.multi_agent import (
    TeamTemplate,
    TeamTopology,
    TaskAssignmentStrategy,
)

# Define team structure
code_review_team = TeamTemplate(
    name="Code Review Team",
    description="Reviews code for quality, security, and performance",
    topology=TeamTopology.PIPELINE,  # Linear sequence
    assignment_strategy=TaskAssignmentStrategy.SKILL_MATCH,
    member_slots={
        "reviewer": 2,  # 2 code reviewers
        "security": 1,  # 1 security specialist
        "approver": 1,  # 1 approver
    },
    shared_context_keys=["file_path", "diff", "pr_url"],
    escalation_threshold=0.7,  # Escalate if confidence < 70%
    max_iterations=5,
)
```

**Topology types**:
- `HIERARCHY`: Manager delegates to subordinates
- `MESH`: Fully connected, any agent can communicate with any other
- `PIPELINE`: Sequential, output of one feeds into next
- `HUB_SPOKE`: Central coordinator with specialized workers

**Assignment strategies**:
- `ROUND_ROBIN`: Distribute tasks evenly
- `SKILL_MATCH`: Assign based on skills and expertise
- `LOAD_BALANCED`: Assign to least busy member

### 4.5 Registration Patterns

#### Direct Registration

```python
from victor.framework.multi_agent.persona_provider import PersonaProvider

provider = PersonaProvider()

# Register persona
provider.register_persona(
    "security_auditor",
    PersonaTraits(
        name="Security Auditor",
        role="security_reviewer",
        description="Identifies vulnerabilities",
        communication_style=CommunicationStyle.FORMAL,
        expertise_level=ExpertiseLevel.SPECIALIST,
    ),
)

# Retrieve persona
persona = provider.get_persona("security_auditor")
```

#### Template-Based Registration

```python
from victor.framework.multi_agent import PersonaTemplate

# Create base template
reviewer_template = PersonaTemplate(
    base_traits=PersonaTraits(
        name="Code Reviewer",
        role="reviewer",
        description="Reviews code for quality",
        communication_style=CommunicationStyle.TECHNICAL,
    ),
    overrides={
        "risk_tolerance": 0.5,
        "creativity": 0.5,
    },
)

# Create specialized personas
security_reviewer = reviewer_template.create(
    name="Security Reviewer",
    description="Reviews code for security vulnerabilities",
    expertise_level=ExpertiseLevel.SPECIALIST,
    strengths=["vulnerability_detection", "threat_modeling"],
    preferred_tools=["static_analysis", "dependency_check"],
)

performance_reviewer = reviewer_template.create(
    name="Performance Reviewer",
    description="Reviews code for performance issues",
    expertise_level=ExpertiseLevel.EXPERT,
    strengths=["profiling", "optimization"],
    preferred_tools=["profiler", "benchmark"],
)
```

#### Vertical Integration

```python
# victor/myvertical/personas.py
from victor.framework.multi_agent.persona_provider import PersonaProvider
from victor.core.verticals.base import VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"
    description = "My custom vertical"

    @classmethod
    def get_team_spec_provider(cls) -> Optional[Any]:
        """Return team specification provider."""
        from myvertical.teams import MyVerticalTeamProvider
        return MyVerticalTeamProvider()
```


**Reading Time:** 5 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


