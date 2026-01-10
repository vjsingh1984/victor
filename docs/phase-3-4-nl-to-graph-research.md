# Phase 3.4: Natural Language to Workflow Graph Generation - Research Document

**Status:** Research & Design Phase
**Created:** January 9, 2026
**Last Updated:** January 9, 2026
**Feasibility Score:** 7/10 (increased from initial 6/10 based on research)

## Executive Summary

This document presents comprehensive research on converting natural language descriptions into executable StateGraph workflows for Victor. The research covers LLM capabilities, intermediate representations, multi-stage generation architectures, validation strategies, and integration with existing Victor infrastructure.

**Key Finding:** Based on 2025 research and emerging tools, natural language to workflow generation is **feasible and practical** for Victor, with several proven approaches and mature tooling available. The feasibility score of **7/10** reflects strong technical viability with manageable risks.

## Table of Contents

1. [LLM Evaluation for Code Generation](#1-llm-evaluation-for-code-generation)
2. [Intermediate Representation Design](#2-intermediate-representation-design)
3. [Multi-Stage Generation Architecture](#3-multi-stage-generation-architecture)
4. [Validation and Safety](#4-validation-and-safety)
5. [Integration with Existing Systems](#5-integration-with-existing-systems)
6. [Risk Assessment and Mitigation](#6-risk-assessment-and-mitigation)
7. [MVP Scope Recommendation](#7-mvp-scope-recommendation)
8. [Implementation Estimate](#8-implementation-estimate)
9. [References and Sources](#9-references-and-sources)

---

## 1. LLM Evaluation for Code Generation

### 1.1 Current State of Structured Output (2025)

The LLM landscape has matured significantly in 2025, with robust support for structured output generation:

#### Claude (Anthropic)
- **Structured Outputs API** (released November 2025)
- Uses constrained decoding to guarantee JSON schema compliance
- Header: `anthropic-beta: structured-outputs-2025-11-13`
- Zero-error JSON generation with schema validation
- **Best for:** Complex nested structures, high reliability requirements
- **Documentation:** [Claude Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)

#### GPT-4 / GPT-4o (OpenAI)
- Native function calling with JSON Schema
- Strong track record with structured output
- Excellent at following complex schemas
- **Best for:** General-purpose structured generation

#### Gemini (Google)
- Native structured output support
- Good at code generation tasks
- **Best for:** Cost-effective alternative to Claude/GPT-4

### 1.2 Evaluation Matrix

| Model | JSON/YAML Generation | Complex Schemas | Hallucination Rate | Speed | Cost | Recommended |
|-------|---------------------|----------------|-------------------|-------|------|-------------|
| **Claude 3.5 Sonnet** | Excellent | Excellent | Very Low | Medium | Medium | **YES** (Primary) |
| **GPT-4o** | Excellent | Excellent | Low | Fast | High | YES (Backup) |
| **Gemini 1.5 Pro** | Good | Good | Medium | Fast | Low | YES (Cost-opt) |
| **Claude 3.5 Haiku** | Good | Fair | Low | Very Fast | Very Low | NO (Simple tasks) |
| **GPT-4o-mini** | Good | Fair | Medium | Very Fast | Very Low | NO (Too unreliable) |

### 1.3 Prompt Engineering Techniques for Structured Output

Based on 2025 research, the following techniques are proven effective:

1. **Schema-First Prompting**
   - Provide complete JSON Schema upfront
   - Use `additionalProperties: false` to constrain output
   - Include examples in schema

2. **Two-Stage Generation**
   - Stage 1: Generate high-level structure (nodes, edges)
   - Stage 2: Fill in details for each component

3. **Chain-of-Thought for Workflows**
   - Ask LLM to "think through" the workflow before generating
   - Extract reasoning to explain generation decisions

4. **Validation Feedback Loop**
   - Generate schema
   - Validate against constraints
   - Feed back errors for regeneration

### 1.4 Benchmarks and Quality Metrics

**StructEval** (May 2025) provides standardized evaluation for structured output:

- Evaluates non-renderable formats: JSON, YAML, CSV, TOML
- Evaluates renderable formats: HTML, Mermaid, Vega-Lite
- **Key Finding:** Top models (Claude 3.5, GPT-4o) achieve >95% validity on complex JSON schemas

**Sources:**
- [StructEval Benchmark](https://tiger-ai-lab.github.io/StructEval/)
- [Claude Structured Outputs Guide](https://medium.com/@meshuggah22/zero-error-json-with-claude-how-anthropics-structured-outputs-actually-work-in-real-code-789cde7aff13)
- [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf)

---

## 2. Intermediate Representation Design

### 2.1 Evaluation of Representations

We evaluated four intermediate representations for LLM comprehension and parsing complexity:

#### Option 1: Abstract Syntax Tree (AST)
**Pros:**
- Structured, type-safe
- Easy to validate
- Direct mapping to StateGraph

**Cons:**
- Complex for LLMs to generate correctly
- High token overhead
- Difficult to debug for humans

**Verdict:** ❌ Not recommended for NL-to-Graph

#### Option 2: Text-Based DSL
**Pros:**
- Human-readable and editable
- LLM-friendly (similar to code)
- Easy to debug

**Cons:**
- Requires custom parser
- Validation complexity
- Potential ambiguity

**Example:**
```
workflow research_task {
  node gather: agent(researcher)
  node analyze: agent(analyst)
  edge gather -> analyze
  entry: gather
}
```

**Verdict:** ⚠️ Possible, but adds complexity

#### Option 3: Structured JSON Schema (RECOMMENDED)
**Pros:**
- Native LLM support (all models handle JSON well)
- Direct schema validation (JSON Schema, Pydantic)
- Easy to parse and convert
- LLM-friendly format
- Industry standard

**Cons:**
- Verbose
- Less human-readable than YAML

**Verdict:** ✅ **RECOMMENDED** for LLM generation

#### Option 4: YAML with Schema
**Pros:**
- Most human-readable
- Compact
- Compatible with existing Victor YAML workflows
- LLMs handle YAML well

**Cons:**
- Slightly more complex validation than JSON
- Indentation sensitivity

**Verdict:** ✅ **RECOMMENDED** as secondary format (can convert from JSON)

### 2.2 Recommended Intermediate Representation

**Primary Format: JSON Schema with Validation**

```json
{
  "workflow_name": "deep_research",
  "description": "Conduct deep research on a topic",
  "nodes": [
    {
      "id": "search_literature",
      "type": "agent",
      "role": "researcher",
      "goal": "Search academic literature on {topic}",
      "tool_budget": 10,
      "allowed_tools": ["search", "read"],
      "output_key": "literature"
    },
    {
      "id": "synthesize",
      "type": "agent",
      "role": "analyst",
      "goal": "Synthesize findings from literature",
      "tool_budget": 5,
      "output_key": "synthesis"
    },
    {
      "id": "check_quality",
      "type": "condition",
      "condition": "quality_threshold",
      "branches": {
        "sufficient": "end",
        "needs_more": "search_literature"
      }
    }
  ],
  "edges": [
    {
      "source": "search_literature",
      "target": "synthesize",
      "type": "normal"
    },
    {
      "source": "synthesize",
      "target": "check_quality",
      "type": "normal"
    },
    {
      "source": "check_quality",
      "target": {
        "sufficient": "__end__",
        "needs_more": "search_literature"
      },
      "type": "conditional",
      "condition": "quality_threshold"
    }
  ],
  "entry_point": "search_literature",
  "metadata": {
    "max_iterations": 25,
    "timeout_seconds": 300,
    "vertical": "research"
  }
}
```

**Schema Validation (Pydantic):**

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Literal, Optional

class NodeConfig(BaseModel):
    id: str
    type: Literal["agent", "compute", "condition", "parallel", "transform", "team"]
    role: Optional[str] = None
    goal: Optional[str] = None
    tool_budget: Optional[int] = 10
    allowed_tools: Optional[List[str]] = None
    handler: Optional[str] = None
    output_key: Optional[str] = None
    condition: Optional[str] = None
    branches: Optional[Dict[str, str]] = None

class EdgeConfig(BaseModel):
    source: str
    target: Union[str, Dict[str, str]]
    type: Literal["normal", "conditional"]
    condition: Optional[str] = None

class WorkflowSchema(BaseModel):
    workflow_name: str
    description: str
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]
    entry_point: str
    metadata: Optional[Dict[str, any]] = {}
```

### 2.3 Integration with StateGraph.from_schema()

The JSON representation directly integrates with Victor's existing `StateGraph.from_schema()` method:

```python
# Generated JSON from LLM
workflow_json = llm.generate_workflow(description)

# Direct conversion to StateGraph
from victor.framework.graph import StateGraph

graph = StateGraph.from_schema(
    workflow_json,
    state_schema=AgentState,
    node_registry=node_registry,
    condition_registry=condition_registry
)

# Compile and execute
compiled = graph.compile()
result = await compiled.invoke(initial_state)
```

---

## 3. Multi-Stage Generation Architecture

### 3.1 Recommended: Four-Stage Pipeline

Based on research into multi-agent workflows and DSL generation, we recommend a **four-stage generation pipeline**:

#### Stage 1: Requirement Understanding (Extractor Agent)

**Goal:** Extract structured requirements from natural language

**Input:** User description (natural language)

**Output:** Structured requirement document

```json
{
  "primary_task": "research_topic",
  "task_type": "research",
  "subtasks": ["search_literature", "synthesize", "validate"],
  "dependencies": [
    {"from": "synthesize", "to": "search_literature"},
    {"from": "validate", "to": "synthesize"}
  ],
  "required_tools": ["search", "read"],
  "vertical": "research",
  "complexity": "medium"
}
```

**Prompt Strategy:**
- Ask LLM to identify task type, subtasks, and dependencies
- Extract required tools and capabilities
- Determine vertical (research, coding, devops, etc.)
- Assess complexity level

**Sources:**
- [Constructing Workflows from Natural Language with Multi-Agent Approach](https://aclanthology.org/2025.naacl-industry.3.pdf)

#### Stage 2: Workflow Design (Architect Agent)

**Goal:** Design graph structure (nodes, edges, conditions)

**Input:** Structured requirements from Stage 1

**Output:** High-level workflow graph

```json
{
  "workflow_structure": {
    "nodes": [
      {"id": "n1", "type": "agent", "purpose": "search"},
      {"id": "n2", "type": "agent", "purpose": "synthesize"},
      {"id": "n3", "type": "condition", "purpose": "check_quality"}
    ],
    "flow": "n1 -> n2 -> n3 -> [if good: end, else: n1]"
  }
}
```

**Prompt Strategy:**
- Given requirements, design optimal node sequence
- Identify where conditional logic is needed
- Determine parallel execution opportunities
- Add validation/verification steps

**Sources:**
- [Multi-Agent Reasoning with Automated Workflow](https://arxiv.org/html/2507.14393v1)

#### Stage 3: Code Generation (Builder Agent)

**Goal:** Generate full JSON schema for StateGraph

**Input:** Workflow structure from Stage 2

**Output:** Complete, validated JSON workflow schema

**Prompt Strategy:**
- Use structured output API (Claude) or function calling (GPT-4)
- Provide strict JSON Schema
- Include examples from existing Victor workflows
- Request inline comments explaining decisions

**Example Prompt:**
```
Given this workflow structure:
{structure_from_stage_2}

Generate a complete StateGraph JSON schema following these constraints:
- Use node_registry: {available_nodes}
- Use condition_registry: {available_conditions}
- Follow this JSON Schema: {strict_schema}
- Include all required fields (nodes, edges, entry_point)

Example workflow for reference:
{example_victor_workflow}
```

**Sources:**
- [LangGraph Reflection Researcher](https://github.com/junfanz1/LangGraph-Reflection-Researcher)
- [Claude Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

#### Stage 4: Validation and Refinement (Validator Agent)

**Goal:** Validate and refine generated workflow

**Input:** Generated JSON schema from Stage 3

**Output:** Validated schema or error feedback

**Validation Steps:**
1. **Schema Validation:** JSON Schema / Pydantic validation
2. **Graph Validation:** Check for unreachable nodes, infinite loops
3. **Type Validation:** Verify node types, edge types
4. **Security Validation:** Check for dangerous tool combinations
5. **Semantic Validation:** Ensure workflow makes logical sense

**Refinement Strategy:**
- If validation fails: Feed back specific errors to Stage 3
- Use "critic" LLM to suggest improvements
- Limit to 3 refinement iterations to avoid loops

**Sources:**
- [OWASP LLM Security 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf)
- [Hallucination Detection in LLMs](https://www.researchgate.net/publication/397333418_Hallucination_Detection_and_Mitigation_in_Large_Language_Models_A_Comprehensive_Review)

### 3.2 Alternative: Single-Stage Generation (Simpler)

For MVP, a **single-stage generation with validation** is feasible:

```python
async def generate_workflow(
    description: str,
    provider: BaseProvider,
    vertical: str
) -> StateGraph:
    """Generate workflow from natural language in one step."""

    # Prompt with examples and schema
    prompt = f"""
    Generate a StateGraph JSON schema for this task:
    {description}

    Vertical: {vertical}
    Available nodes: {get_available_nodes(vertical)}
    Available conditions: {get_available_conditions(vertical)}

    {get_example_workflows(vertical)}

    Generate valid JSON following this schema:
    {get_workflow_json_schema()}
    """

    # Generate with structured output
    workflow_json = await provider.generate_structured(
        prompt,
        schema=WorkflowSchema
    )

    # Validate
    validation = validate_workflow(workflow_json)
    if not validation.valid:
        # Feed back for refinement (up to 3 attempts)
        workflow_json = await refine_workflow(
            workflow_json,
            validation.errors,
            provider
        )

    # Convert to StateGraph
    return StateGraph.from_schema(
        workflow_json,
        node_registry=get_node_registry(vertical),
        condition_registry=get_condition_registry(vertical)
    )
```

**Verdict:** ✅ **Start with single-stage** for MVP, add multi-stage later for quality improvements.

---

## 4. Validation and Safety

### 4.1 Multi-Layer Validation Strategy

#### Layer 1: Schema Validation (JSON Schema / Pydantic)

**Purpose:** Ensure structural correctness

```python
from pydantic import ValidationError

def validate_schema(workflow_json: dict) -> ValidationResult:
    """Validate workflow JSON against schema."""
    try:
        workflow = WorkflowSchema(**workflow_json)
        return ValidationResult(valid=True, errors=[])
    except ValidationError as e:
        return ValidationResult(
            valid=False,
            errors=[str(err) for err in e.errors()]
        )
```

**Catches:**
- Missing required fields
- Type mismatches
- Invalid enum values
- Malformed JSON

#### Layer 2: Graph Validation

**Purpose:** Ensure executable graph structure

```python
def validate_graph_structure(workflow: WorkflowSchema) -> ValidationResult:
    """Validate graph structure for execution."""
    errors = []

    # Check all nodes reachable from entry point
    reachable = find_reachable_nodes(workflow)
    unreachable = set(n.id for n in workflow.nodes) - reachable
    if unreachable:
        errors.append(f"Unreachable nodes: {unreachable}")

    # Check for infinite loops without conditions
    cycles = find_cycles(workflow.edges)
    for cycle in cycles:
        if not has_condition_node_in_cycle(cycle, workflow):
            errors.append(f"Unconditional cycle detected: {cycle}")

    # Check all edge targets exist
    for edge in workflow.edges:
        if isinstance(edge.target, str):
            if edge.target != "__end__" and edge.target not in {n.id for n in workflow.nodes}:
                errors.append(f"Edge target not found: {edge.target}")
        elif isinstance(edge.target, dict):
            for branch, target in edge.target.items():
                if target != "__end__" and target not in {n.id for n in workflow.nodes}:
                    errors.append(f"Conditional target not found: {branch} -> {target}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors
    )
```

**Catches:**
- Unreachable nodes
- Infinite loops without conditions
- Broken references (edges to non-existent nodes)
- Orphaned nodes

#### Layer 3: Type Validation

**Purpose:** Ensure node types and configurations are valid

```python
def validate_node_types(workflow: WorkflowSchema) -> ValidationResult:
    """Validate node type configurations."""
    errors = []

    for node in workflow.nodes:
        if node.type == "agent":
            if not node.role:
                errors.append(f"Agent node '{node.id}' missing role")
            if not node.goal:
                errors.append(f"Agent node '{node.id}' missing goal")

        elif node.type == "compute":
            if not node.handler:
                errors.append(f"Compute node '{node.id}' missing handler")

        elif node.type == "condition":
            if not node.condition:
                errors.append(f"Condition node '{node.id}' missing condition function")
            if not node.branches:
                errors.append(f"Condition node '{node.id}' missing branches")

    return ValidationResult(valid=len(errors) == 0, errors=errors)
```

**Catches:**
- Missing required node properties
- Invalid type combinations
- Undefined handler/condition references

#### Layer 4: Security Validation

**Purpose:** Prevent dangerous workflows

Based on [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf):

```python
DANGEROUS_TOOL_COMBINATIONS = [
    {"file_write", "file_execute"},  # Code injection risk
    {"shell", "sudo"},               # Privilege escalation
    {"network", "eval"},             # RCE risk
]

def validate_security(workflow: WorkflowSchema) -> ValidationResult:
    """Validate workflow for security risks."""
    warnings = []
    errors = []

    # Check for dangerous tool combinations
    workflow_tools = set()
    for node in workflow.nodes:
        if node.allowed_tools:
            workflow_tools.update(node.allowed_tools)

    for dangerous_combo in DANGEROUS_TOOL_COMBINATIONS:
        if dangerous_combo.issubset(workflow_tools):
            errors.append(
                f"Dangerous tool combination detected: {dangerous_combo}. "
                "This combination poses security risks."
            )

    # Check for excessive tool budgets
    for node in workflow.nodes:
        if node.tool_budget and node.tool_budget > 50:
            warnings.append(
                f"Node '{node.id}' has high tool_budget ({node.tool_budget}). "
                "Consider limiting to prevent runaway execution."
            )

    # Check for infinite loops
    cycles = find_cycles(workflow.edges)
    if cycles:
        for cycle in cycles:
            cycle_length = len(cycle)
            if cycle_length > 5:
                warnings.append(
                    f"Long cycle detected (length {cycle_length}): {cycle}. "
                    "Ensure max_iterations is configured appropriately."
                )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
```

**Catches:**
- Dangerous tool combinations
- Excessive resource allocation
- Potential infinite loops
- Missing safety constraints

### 4.2 Human-in-the-Loop Review

For critical workflows or high-complexity generations, require human approval:

```python
def require_human_review(workflow: WorkflowSchema) -> bool:
    """Determine if workflow requires human review."""
    # Require review if:
    # - High complexity (>10 nodes)
    # - Dangerous tools present
    # - Conditional branching
    # - User is in "safe mode"

    if len(workflow.nodes) > 10:
        return True

    for node in workflow.nodes:
        if node.allowed_tools:
            if any(t in DANGEROUS_TOOLS for t in node.allowed_tools):
                return True

        if node.type == "condition":
            return True

    return False
```

### 4.3 Fallback Strategies

When generation fails after validation/refinement:

1. **Fallback to Template:** Match to pre-defined workflow template
2. **Simplify:** Ask LLM to generate simpler version
3. **Manual Intervention:** Prompt user to provide more details
4. **Partial Generation:** Generate valid partial workflow, let user complete

```python
async def fallback_to_template(description: str) -> StateGraph:
    """Fallback to closest template workflow."""
    # Find similar template
    template = find_similar_workflow_template(description)

    # Allow user to customize
    return customize_template(template, description)
```

---

## 5. Integration with Existing Systems

### 5.1 Integration Points

Victor has excellent infrastructure for this feature:

#### Existing API: StateGraph.from_schema()

**Location:** `victor/framework/graph.py` (line 1853)

**Features:**
- Accepts JSON schema or YAML string
- Validates structure
- Creates StateGraph instance
- Ready for compilation and execution

**Usage:**
```python
graph = StateGraph.from_schema(
    workflow_json,
    state_schema=AgentState,
    node_registry=node_registry,
    condition_registry=condition_registry
)
```

**Integration:** Direct plug-and-play for generated workflows

#### Existing Infrastructure: UnifiedWorkflowCompiler

**Location:** `victor/workflows/unified_compiler.py`

**Features:**
- Two-level caching (definition + execution)
- Compilation from YAML, JSON, or WorkflowDefinition
- Execution with checkpointing and observability

**Integration:** Use for compiling generated workflows

```python
compiler = UnifiedWorkflowCompiler(enable_caching=True)
compiled = compiler.compile_definition(generated_definition)
result = await compiled.invoke(initial_state)
```

#### Existing Infrastructure: WorkflowValidator

**Location:** `victor/workflows/validator.py`

**Features:**
- Workflow validation logic
- Error reporting
- Best practices checking

**Integration:** Extend for generated workflow validation

#### Existing Infrastructure: A/B Testing (Phase 3.3)

**Location:** `docs/phase-3-3-ab-testing-design.md`

**Features:**
- Experiment tracking
- Variant comparison
- Performance metrics

**Integration:** Compare different generation strategies

```python
# A/B test different generation prompts
experiment = create_ab_experiment(
    name="nl2graph_prompt_variants",
    variants=["prompt_v1", "prompt_v2", "prompt_v3"],
    metrics=["success_rate", "validation_errors", "execution_time"]
)

# Track which prompt generates better workflows
log_experiment_result(experiment, variant_id, metrics)
```

#### Existing Infrastructure: Optimization (Phase 3.3)

**Location:** `docs/phase-3-3-optimization-algorithms-design.md`

**Features:**
- Workflow optimization algorithms
- Performance tuning
- Resource allocation

**Integration:** Optimize generated workflows for efficiency

### 5.2 Proposed API Design

```python
# victor/workflows/nl_to_graph.py

from typing import Optional, Dict, Any
from victor.framework.graph import StateGraph
from victor.providers.base import BaseProvider
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

class WorkflowGenerator:
    """Generate workflows from natural language."""

    def __init__(
        self,
        provider: BaseProvider,
        vertical: str,
        compiler: Optional[UnifiedWorkflowCompiler] = None,
        enable_validation: bool = True,
        require_human_review: bool = False
    ):
        """Initialize workflow generator.

        Args:
            provider: LLM provider for generation
            vertical: Target vertical (research, coding, devops, etc.)
            compiler: Optional compiler for workflow execution
            enable_validation: Whether to validate generated workflows
            require_human_review: Whether to require human approval
        """
        self.provider = provider
        self.vertical = vertical
        self.compiler = compiler or UnifiedWorkflowCompiler(enable_caching=True)
        self.enable_validation = enable_validation
        self.require_human_review = require_human_review

        # Load registries
        self.node_registry = get_node_registry(vertical)
        self.condition_registry = get_condition_registry(vertical)

    async def generate_workflow(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        max_refinements: int = 3
    ) -> StateGraph:
        """Generate workflow from natural language description.

        Args:
            description: Natural language workflow description
            context: Additional context for generation
            max_refinements: Maximum validation refinement iterations

        Returns:
            StateGraph ready for compilation

        Raises:
            WorkflowGenerationError: If generation fails
        """
        # Stage 1: Generate JSON schema
        workflow_json = await self._generate_schema(
            description,
            context or {}
        )

        # Stage 2: Validate and refine
        for iteration in range(max_refinements):
            validation = self._validate_workflow(workflow_json)

            if validation.valid:
                break

            if iteration == max_refinements - 1:
                raise WorkflowGenerationError(
                    f"Failed to generate valid workflow after {max_refinements} attempts: "
                    f"{validation.errors}"
                )

            # Refine with feedback
            workflow_json = await self._refine_schema(
                workflow_json,
                validation.errors
            )

        # Stage 3: Convert to StateGraph
        graph = StateGraph.from_schema(
            workflow_json,
            node_registry=self.node_registry,
            condition_registry=self.condition_registry
        )

        # Stage 4: Human review if required
        if self.require_human_review and self._should_review(graph):
            graph = await self._request_human_review(graph, workflow_json)

        return graph

    async def refine_workflow(
        self,
        graph: StateGraph,
        feedback: str
    ) -> StateGraph:
        """Refine existing workflow based on feedback.

        Args:
            graph: Existing StateGraph to refine
            feedback: Natural language feedback for changes

        Returns:
            Refined StateGraph
        """
        # Convert graph to schema
        current_schema = graph.to_dict()

        # Generate refined version
        refined_schema = await self._generate_refined_schema(
            current_schema,
            feedback
        )

        # Validate and return
        return StateGraph.from_schema(
            refined_schema,
            node_registry=self.node_registry,
            condition_registry=self.condition_registry
        )

    async def _generate_schema(
        self,
        description: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate workflow JSON schema from description."""
        # Build prompt with examples and schema
        prompt = self._build_generation_prompt(description, context)

        # Use provider's structured output
        if hasattr(self.provider, 'generate_structured'):
            return await self.provider.generate_structured(
                prompt,
                schema=WorkflowSchema
            )
        else:
            # Fallback to prompt-based generation
            response = await self.provider.chat(
                prompt,
                response_format="json"
            )
            return json.loads(response.content)

    def _validate_workflow(self, workflow_json: Dict[str, Any]) -> ValidationResult:
        """Validate generated workflow."""
        if not self.enable_validation:
            return ValidationResult(valid=True, errors=[])

        # Layer 1: Schema validation
        schema_result = validate_schema(workflow_json)
        if not schema_result.valid:
            return schema_result

        # Parse to WorkflowSchema
        workflow = WorkflowSchema(**workflow_json)

        # Layer 2: Graph validation
        graph_result = validate_graph_structure(workflow)
        if not graph_result.valid:
            return graph_result

        # Layer 3: Type validation
        type_result = validate_node_types(workflow)
        if not type_result.valid:
            return type_result

        # Layer 4: Security validation
        security_result = validate_security(workflow)
        if not security_result.valid:
            return security_result

        return ValidationResult(
            valid=True,
            errors=[],
            warnings=security_result.warnings
        )

    def _build_generation_prompt(
        self,
        description: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for workflow generation."""
        examples = get_example_workflows(self.vertical)
        available_nodes = list(self.node_registry.keys())
        available_conditions = list(self.condition_registry.keys())

        return f"""
You are a workflow generation expert for Victor AI assistant.

Generate a StateGraph JSON schema for this task:
{description}

Context:
{json.dumps(context, indent=2)}

**Vertical:** {self.vertical}

**Available Node Types:**
- agent: LLM-powered agent with role and goal
- compute: Direct tool execution
- condition: Conditional branching logic
- parallel: Execute multiple branches concurrently
- transform: Data transformation
- team: Multi-agent team

**Available Node Functions:** {available_nodes}
**Available Conditions:** {available_conditions}

**Examples from {self.vertical} vertical:**
{json.dumps(examples, indent=2)}

**JSON Schema to Follow:**
{json.dumps(WorkflowSchema.model_json_schema(), indent=2)}

**Requirements:**
1. Generate valid JSON following the schema exactly
2. All nodes must be reachable from entry_point
3. Conditional edges must have condition function
4. Agent nodes must have role and goal
5. Compute nodes must have handler
6. Avoid infinite loops (use max_iterations)
7. Keep workflows simple (<10 nodes preferred)

Generate the workflow JSON:
"""

    async def _refine_schema(
        self,
        workflow_json: Dict[str, Any],
        errors: List[str]
    ) -> Dict[str, Any]:
        """Refine workflow based on validation errors."""
        prompt = f"""
Fix this workflow JSON based on validation errors:

**Current Workflow:**
{json.dumps(workflow_json, indent=2)}

**Validation Errors:**
{chr(10).join(f"- {err}" for err in errors)}

**Instructions:**
1. Fix all validation errors
2. Maintain the workflow's original intent
3. Ensure all required fields are present
4. Verify all node and edge references are valid
5. Output valid JSON only (no explanations)

Fixed workflow JSON:
"""

        response = await self.provider.chat(prompt)
        return json.loads(response.content)
```

### 5.3 CLI Integration

Add new CLI command for workflow generation:

```bash
# Generate workflow from natural language
victor workflow generate "Research AI trends in 2025" --vertical research

# Generate with interactive review
victor workflow generate "Deploy to production" --vertical devops --review

# Generate and execute
victor workflow generate "Bug fix for authentication" --vertical coding --execute

# Refine existing workflow
victor workflow refine my_workflow.yaml "Add validation step"

# Validate generated workflow
victor workflow validate generated_workflow.json
```

---

## 6. Risk Assessment and Mitigation

### 6.1 Risk Matrix

| Risk | Probability | Impact | Severity | Mitigation |
|------|-------------|--------|----------|------------|
| **LLM Hallucination (Invalid Workflows)** | High | High | **HIGH** | Multi-layer validation, refinement loops, human review |
| **Complexity Limits (Overly Complex Workflows)** | Medium | Medium | **MEDIUM** | Node count limits, complexity scoring, template fallback |
| **Cost Considerations (LLM API Costs)** | Medium | Low | **MEDIUM** | Caching, cheaper models for refinement, cost limits |
| **Quality Assurance (Poor Workflow Quality)** | Medium | High | **HIGH** | A/B testing, user feedback, optimization |
| **Security Issues (Dangerous Tool Combinations)** | Low | Critical | **HIGH** | Security validation, allowlists, human review |
| **Dependency on External LLMs** | Low | Medium | **LOW** | Provider abstraction, fallback models |

### 6.2 Mitigation Strategies

#### Mitigation 1: LLM Hallucination

**Problem:** LLM generates structurally valid but semantically incorrect workflows

**Strategies:**
1. **Multi-Layer Validation:** Schema, graph, type, security (Section 4.1)
2. **Refinement Loops:** Feed back errors up to 3 times
3. **Self-Consistency Checks:** Generate 3 variants, pick most consistent
4. **Human Review:** Require approval for high-complexity workflows
5. **Template Fallback:** Match to pre-defined templates on failure

**Sources:**
- [Hallucination Detection and Mitigation in LLMs](https://www.researchgate.net/publication/397333418_Hallucination_Detection_and_Mitigation_in_Large_Language_Models_A_Comprehensive_Review)
- [OWASP LLM Hallucination Guide](https://medium.com/@adnanmasood/llms-are-brilliant-and-breakable-why-hallucinations-prompt-injections-and-jailbreaks-demand-cdae33adadcd)

#### Mitigation 2: Complexity Limits

**Problem:** LLM generates overly complex workflows that are hard to debug

**Strategies:**
1. **Node Count Limits:** Cap at 10 nodes for MVP, 20 for advanced
2. **Complexity Scoring:** Rate workflow complexity, reject if too high
3. **Incremental Generation:** Generate simple workflow, offer to expand
4. **User Override:** Allow users to bypass limits with warning

```python
def calculate_complexity(workflow: WorkflowSchema) -> int:
    """Calculate workflow complexity score."""
    score = 0

    # Base: number of nodes
    score += len(workflow.nodes) * 2

    # Conditional edges add complexity
    for edge in workflow.edges:
        if edge.type == "conditional":
            score += 3

    # Parallel nodes add complexity
    for node in workflow.nodes:
        if node.type == "parallel":
            score += 5

    # Team nodes add complexity
    for node in workflow.nodes:
        if node.type == "team":
            score += 5

    return score

MAX_COMPLEXITY = 50  # Adjust based on testing

if calculate_complexity(workflow) > MAX_COMPLEXITY:
    return WorkflowTooComplexError(
        f"Workflow complexity ({score}) exceeds maximum ({MAX_COMPLEXITY}). "
        "Please simplify your request or break into smaller workflows."
    )
```

#### Mitigation 3: Cost Considerations

**Problem:** LLM API calls can be expensive for frequent generation

**Strategies:**
1. **Aggressive Caching:** Cache generated workflows by description hash
2. **Model Selection:** Use cheaper models (Claude Haiku, GPT-4o-mini) for refinement
3. **Generation Budgets:** Cap tokens per generation
4. **Cost Tracking:** Log costs, alert on thresholds
5. **Tiered Pricing:** Offer basic (cheap) and premium (expensive) generation

```python
# Cache workflow generation results
@lru_cache(maxsize=1000)
async def generate_workflow_cached(
    description: str,
    vertical: str
) -> StateGraph:
    """Cached workflow generation."""
    return await generate_workflow(description, vertical)

# Use cheaper model for refinement
refinement_provider = create_provider(
    name="claude-haiku",  # Cheaper than sonnet
    api_key=settings.claude_api_key
)
```

#### Mitigation 4: Quality Assurance

**Problem:** Generated workflows may have poor performance

**Strategies:**
1. **A/B Testing:** Compare generation strategies (Section 5.1, Phase 3.3)
2. **User Feedback:** Collect ratings on generated workflows
3. **Performance Metrics:** Track execution time, success rate, tool usage
4. **Optimization:** Use Phase 3.3 optimization algorithms to tune workflows
5. **Continuous Improvement:** Retrain prompts based on feedback

```python
# Log workflow execution for quality tracking
from victor.observability.experiment_tracking import log_experiment

log_experiment(
    name="nl2graph_generation",
    parameters={
        "description": description,
        "vertical": vertical,
        "model": provider.name
    },
    metrics={
        "success": result.success,
        "execution_time": result.duration,
        "node_count": len(workflow.nodes),
        "validation_errors": len(validation.errors)
    }
)
```

#### Mitigation 5: Security Issues

**Problem:** Generated workflows could execute dangerous operations

**Strategies:**
1. **Security Validation:** Check dangerous tool combinations (Section 4.1)
2. **Tool Allowlists:** Restrict which tools can be generated
3. **Sandboxing:** Execute workflows in isolated environment
4. **Human Review:** Require approval for dangerous workflows
5. **Audit Logging:** Log all generated workflows for security review

```python
# Restrict to safe tools for auto-generation
SAFE_TOOLS = {
    "research": ["search", "read", "summarize"],
    "coding": ["read", "write", "test"],
    "devops": ["docker", "k8s", "terraform"]
}

def validate_tools_allowed(workflow: WorkflowSchema, vertical: str) -> ValidationResult:
    """Ensure only safe tools are used."""
    allowed = SAFE_TOOLS.get(vertical, [])

    for node in workflow.nodes:
        if node.allowed_tools:
            for tool in node.allowed_tools:
                if tool not in allowed:
                    return ValidationResult(
                        valid=False,
                        errors=[f"Tool '{tool}' not allowed in auto-generated workflows. "
                                f"Allowed: {allowed}"]
                    )

    return ValidationResult(valid=True)
```

#### Mitigation 6: Dependency on External LLMs

**Problem:** Feature depends on external LLM providers

**Strategies:**
1. **Provider Abstraction:** Support multiple providers (Claude, GPT-4, Gemini)
2. **Fallback Models:** Use local models (Ollama) as backup
3. **Graceful Degradation:** Fall back to template matching if LLM unavailable
4. **Provider Rotation:** Distribute load across providers

```python
# Provider with fallback
async def generate_with_fallback(
    description: str,
    primary_provider: BaseProvider,
    fallback_provider: BaseProvider
) -> StateGraph:
    """Generate with provider fallback."""
    try:
        return await generate_workflow(description, primary_provider)
    except LLMError:
        logger.warning(f"Primary provider failed, using fallback")
        return await generate_workflow(description, fallback_provider)
```

---

## 7. MVP Scope Recommendation

### 7.1 MVP Definition (Minimum Viable Product)

**Goal:** Deliver a working NL-to-Graph feature with manageable risk and clear value.

**Timeline:** 4-6 weeks for MVP

**Scope:**

#### ✅ Include in MVP

1. **Single-Stage Generation**
   - Simple prompt-based generation
   - JSON output with structured validation
   - 1 refinement iteration (not 3)

2. **Two Verticals Only**
   - Research workflows (well-defined patterns)
   - Coding workflows (common use case)
   - DevOps and DataAnalysis in post-MVP

3. **Basic Validation**
   - Schema validation (Pydantic)
   - Graph validation (reachability, cycles)
   - Skip security validation for MVP (add in post-MVP)

4. **Limited Node Types**
   - agent nodes
   - compute nodes
   - condition nodes
   - Skip parallel, transform, team for MVP

5. **Template Fallback**
   - If generation fails, offer closest template
   - 3-5 templates per vertical

6. **CLI Integration**
   - `victor workflow generate` command
   - No interactive review (add post-MVP)

7. **Caching**
   - Simple in-memory cache by description hash
   - Persistent cache in post-MVP

8. **Cost Controls**
   - Max 3 LLM calls per generation (1 generate + 2 refinements)
   - Use Claude 3.5 Sonnet (primary) + Haiku (refinement)

9. **Basic Metrics**
   - Success rate
   - Generation time
   - Validation error rate

#### ❌ Exclude from MVP (Post-MVP)

1. Multi-stage generation (architect → builder → validator)
2. All 6 verticals (start with 2)
3. Security validation layer
4. Parallel, transform, team node types
5. Human-in-the-loop review workflow
6. A/B testing integration
7. Workflow optimization
8. Persistent cache storage
9. Web UI integration
10. Advanced refinement (3+ iterations)

### 7.2 Success Criteria

MVP is successful if:

1. **70% success rate:** 7 out of 10 generation attempts produce valid workflows
2. **<10 second latency:** Average generation time under 10 seconds
3. **Positive user feedback:** At least 5 users report usefulness
4. **Cost manageable:** <$0.50 per generation on average
5. **No security incidents:** No dangerous workflows executed

### 7.3 Post-MVP Roadmap

**Phase 1 (MVP):** Core generation with 2 verticals
- Timeline: 4-6 weeks
- Features: Single-stage, basic validation, template fallback

**Phase 2 (Quality):** Improve generation quality
- Timeline: +4 weeks
- Features: Multi-stage generation, all 6 verticals, all node types

**Phase 3 (Safety):** Add security and review
- Timeline: +3 weeks
- Features: Security validation, human review workflow, audit logging

**Phase 4 (Optimization):** Enhance performance
- Timeline: +3 weeks
- Features: A/B testing, optimization algorithms, persistent caching

---

## 8. Implementation Estimate

### 8.1 Lines of Code Estimate

| Component | LOC | Complexity |
|-----------|-----|------------|
| **Core Generation** | | |
| WorkflowGenerator class | 800 | Medium |
| Prompt templates | 300 | Low |
| Validation layers | 600 | Medium |
| Refinement logic | 400 | Medium |
| **Integration** | | |
| StateGraph.from_schema extension | 200 | Low |
| CLI commands | 400 | Low |
| Caching layer | 250 | Low |
| **Testing** | | |
| Unit tests | 1,200 | Medium |
| Integration tests | 600 | Medium |
| End-to-end tests | 400 | Low |
| **Documentation** | | |
| API docs | 300 | Low |
| User guide | 500 | Low |
| Examples | 400 | Low |
| **Total** | **6,350** | **Medium** |

### 8.2 Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| **Design & Setup** | 3 days | None |
| - API design | 1 day | |
| - Prompt engineering | 1 day | |
| - Test infrastructure | 1 day | |
| **Core Generation** | 10 days | Design |
| - Single-stage generation | 3 days | |
| - Validation layers | 3 days | |
| - Refinement logic | 2 days | |
| - Template fallback | 2 days | |
| **Integration** | 5 days | Core |
| - StateGraph integration | 2 days | |
| - CLI commands | 2 days | |
| - Caching layer | 1 day | |
| **Testing** | 5 days | Integration |
| - Unit tests | 2 days | |
| - Integration tests | 2 days | |
| - E2E tests | 1 day | |
| **Documentation** | 3 days | Testing |
| - API documentation | 1 day | |
| - User guide | 1 day | |
| - Examples | 1 day | |
| **Buffer** | 4 days | All |
| **Total** | **30 days** (~6 weeks) | |

### 8.3 Resource Estimate

**Developers:** 1-2 developers
**Duration:** 4-6 weeks
**Risk:** Medium (manageable with clear MVP scope)

**Key Dependencies:**
- LLM provider API access (Claude/GPT-4)
- Existing Victor infrastructure (StateGraph, UnifiedCompiler)
- Validation logic (Pydantic, JSON Schema)

---

## 9. References and Sources

### 9.1 Academic Research

1. **StructEval: Benchmarking LLMs' Capabilities to Generate Structured Output**
   - https://tiger-ai-lab.github.io/StructEval/
   - Published: May 2025
   - Key insights: Standardized evaluation for structured output

2. **A Survey on Code Generation with LLM-based Agents**
   - https://arxiv.org/html/2508.00083v1
   - Published: July 2025
   - Key insights: Comprehensive survey of LLM code generation

3. **Constructing Workflows from Natural Language with Multi-Agent Approach**
   - https://aclanthology.org/2025.naacl-industry.3.pdf
   - Published: NAACL 2025
   - Key insights: Multi-agent architecture for NL2Workflow

4. **Adaptive Multi-Agent Reasoning via Automated Workflow**
   - https://arxiv.org/html/2507.14393v1
   - Published: July 2025
   - Key insights: Iterative prompt refinement for agents

5. **LLM-based Agents Suffer from Hallucinations: A Survey**
   - https://arxiv.org/html/2509.18970v1
   - Published: September 2025
   - Key insights: Hallucination mitigation strategies

6. **Hallucination Detection and Mitigation in LLMs: Comprehensive Review**
   - https://www.researchgate.net/publication/397333418_Hallucination_Detection_and_Mitigation_in_Large_Language_Models_A_Comprehensive_Review
   - Published: November 2025
   - Key insights: Five primary detection methods

### 9.2 Tools and Frameworks

7. **Claude Structured Outputs**
   - https://platform.claude.com/docs/en/build-with-claude/structured-outputs
   - Released: November 2025
   - Key insights: Zero-error JSON generation

8. **Anthropic Advanced Tool Use**
   - https://www.anthropic.com/engineering/advanced-tool-use
   - Published: November 2025
   - Key insights: Programmatic tool calling

9. **LangGraph Documentation**
   - https://docs.langchain.com/oss/python/langgraph/workflows-agents
   - Key insights: Multi-agent workflow patterns

10. **LangGraph-Reflection-Researcher**
    - https://github.com/junfanz1/LangGraph-Reflection-Researcher
    - Key insights: Iterative refinement pattern

11. **MetaGPT: Multi-Agent Framework**
    - https://github.com/FoundationAgents/MetaGPT
    - Updated: February 2025
    - Key insights: Natural language programming

12. **StrictJSON Framework**
    - https://github.com/tanchongmin/strictjson
    - Key insights: Constrained LLM output

### 9.3 Security and Best Practices

13. **OWASP Top 10 for LLM Applications 2025**
    - https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf
    - Published: November 2024
    - Key insights: LLM security best practices

14. **Structural Trimming for Vulnerability Mitigation in Code LLMs**
    - https://openreview.net/pdf?id=dU4Y2sNfJ2
    - Key insights: Code LLM vulnerability mitigation

### 9.4 Architecture and Design Patterns

15. **Building AI Agents: Workflow-First vs. Code-First vs. Hybrid**
    - https://techcommunity.microsoft.com/blog/azurearchitectureblog/building-ai-agents-workflow-first-vs-code-first-vs-hybrid/4466788
    - Published: November 2025
    - Key insights: Adaptive workflow generation

16. **A Large Language Model Programming Workflow**
    - https://aclanthology.org/2025.acl-long.621.pdf
    - Key insights: Two-phase LLM programming workflow

17. **AIAP: A No-Code Workflow Builder for Non-Experts**
    - https://arxiv.org/html/2508.02470v1
    - Published: August 2025
    - Key insights: No-code workflow construction

18. **Chain-of-Programming (CoP)**
    - https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2509812
    - Key insights: Translating natural language to executable code

### 9.5 Victor Infrastructure

19. **Victor StateGraph Implementation**
    - `victor/framework/graph.py`
    - Key feature: `StateGraph.from_schema()` for dynamic graph generation

20. **Victor UnifiedWorkflowCompiler**
    - `victor/workflows/unified_compiler.py`
    - Key feature: Two-level caching and compilation

21. **Victor BaseYAMLWorkflowProvider**
    - `victor/framework/workflows/base_yaml_provider.py`
    - Key feature: YAML-first workflow architecture

22. **Phase 3.3: Optimization Algorithms**
    - `docs/phase-3-3-optimization-algorithms-design.md`
    - Key feature: Workflow optimization infrastructure

23. **Phase 3.3: A/B Testing**
    - `docs/phase-3-3-ab-testing-design.md`
    - Key feature: Experiment tracking and comparison

24. **Victor Roadmap**
    - `roadmap.md`
    - Key feature: Phase 3.4 positioning

---

## Appendix: Example Workflows

### A.1 Research Workflow Example

**Input Description:**
```
"I need to do deep research on the latest AI trends in 2025.
Search academic papers, synthesize findings, and create a summary."
```

**Generated JSON Schema:**
```json
{
  "workflow_name": "ai_trends_research",
  "description": "Deep research on AI trends in 2025",
  "nodes": [
    {
      "id": "search_papers",
      "type": "agent",
      "role": "researcher",
      "goal": "Search academic papers on AI trends in 2025",
      "tool_budget": 10,
      "allowed_tools": ["search", "read"],
      "output_key": "papers"
    },
    {
      "id": "synthesize",
      "type": "agent",
      "role": "analyst",
      "goal": "Synthesize findings from {papers} into summary",
      "tool_budget": 5,
      "output_key": "summary"
    },
    {
      "id": "check_depth",
      "type": "condition",
      "condition": "has_sufficient_sources",
      "branches": {
        "sufficient": "__end__",
        "insufficient": "search_papers"
      }
    }
  ],
  "edges": [
    {
      "source": "search_papers",
      "target": "synthesize",
      "type": "normal"
    },
    {
      "source": "synthesize",
      "target": "check_depth",
      "type": "normal"
    },
    {
      "source": "check_depth",
      "target": {
        "sufficient": "__end__",
        "insufficient": "search_papers"
      },
      "type": "conditional",
      "condition": "has_sufficient_sources"
    }
  ],
  "entry_point": "search_papers",
  "metadata": {
    "vertical": "research",
    "max_iterations": 10,
    "timeout_seconds": 300
  }
}
```

### A.2 Coding Workflow Example

**Input Description:**
```
"Fix the authentication bug where users can't log in after password reset.
First investigate the code, then implement a fix, and finally test it."
```

**Generated JSON Schema:**
```json
{
  "workflow_name": "auth_bug_fix",
  "description": "Fix authentication bug after password reset",
  "nodes": [
    {
      "id": "investigate",
      "type": "agent",
      "role": "researcher",
      "goal": "Investigate authentication code to find password reset bug",
      "tool_budget": 8,
      "allowed_tools": ["read", "search", "git_log"],
      "output_key": "findings"
    },
    {
      "id": "implement_fix",
      "type": "agent",
      "role": "developer",
      "goal": "Implement fix for authentication bug based on {findings}",
      "tool_budget": 10,
      "allowed_tools": ["write", "edit", "read"],
      "output_key": "fix"
    },
    {
      "id": "test_fix",
      "type": "agent",
      "role": "tester",
      "goal": "Test the authentication fix",
      "tool_budget": 5,
      "allowed_tools": ["test", "run_tests"],
      "output_key": "test_results"
    },
    {
      "id": "check_tests",
      "type": "condition",
      "condition": "tests_passed",
      "branches": {
        "passed": "__end__",
        "failed": "implement_fix"
      }
    }
  ],
  "edges": [
    {
      "source": "investigate",
      "target": "implement_fix",
      "type": "normal"
    },
    {
      "source": "implement_fix",
      "target": "test_fix",
      "type": "normal"
    },
    {
      "source": "test_fix",
      "target": "check_tests",
      "type": "normal"
    },
    {
      "source": "check_tests",
      "target": {
        "passed": "__end__",
        "failed": "implement_fix"
      },
      "type": "conditional",
      "condition": "tests_passed"
    }
  ],
  "entry_point": "investigate",
  "metadata": {
    "vertical": "coding",
    "max_iterations": 15,
    "timeout_seconds": 600
  }
}
```

---

## Conclusion

This research demonstrates that **natural language to workflow graph generation is feasible and practical** for Victor, with a recommended feasibility score of **7/10**. The 2025 LLM landscape offers mature structured output capabilities, proven multi-agent architectures, and robust validation strategies.

**Key Recommendations:**

1. **Start with MVP:** Focus on single-stage generation with 2 verticals (Research, Coding)
2. **Use JSON Intermediate:** Leverage existing JSON Schema support and Pydantic validation
3. **Multi-Layer Validation:** Implement schema, graph, type, and security validation layers
4. **Iterative Refinement:** Include 1-3 refinement iterations to fix validation errors
5. **Template Fallback:** Provide pre-defined templates when generation fails
6. **Risk Management:** Address hallucination, complexity, cost, quality, and security through mitigation strategies

**Next Steps:**

1. Review and approve this research document
2. Create detailed design specification based on MVP scope
3. Set up development environment and LLM provider access
4. Begin implementation following 6-week timeline
5. Conduct user testing with internal team
6. Iterate based on feedback before public release

**Estimated Timeline:** 4-6 weeks for MVP, +10 weeks for full feature (16 weeks total)

**Estimated Effort:** 6,350 LOC, 1-2 developers, manageable risk with clear scope

---

**Document Status:** Ready for Review
**Next Review:** Upon completion of design specification
