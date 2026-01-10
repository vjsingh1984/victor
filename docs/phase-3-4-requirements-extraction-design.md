# Phase 3.4: Requirements Extraction and Analysis System - Design Document

**Status:** Draft
**Created:** 2025-01-09
**Author:** Claude (Sonnet 4.5)
**Target:** Victor v0.6.0+

---

## Executive Summary

This document designs a **natural language workflow requirement extraction system** that enables users to describe workflows in plain English and automatically generate structured workflow definitions. The system leverages LLM-based extraction with rule-based validation and ambiguity resolution through interactive clarification.

**Key Problem:** Users want to create workflows but don't know YAML syntax or graph DSL. They describe workflows in natural language:
- "Research AI trends, summarize findings, and if there are more than 10 sources, create a report"
- "Deploy to staging, run tests, and if tests pass, deploy to production"
- "Analyze this codebase, find bugs, fix them, and run tests to verify"

**Our Solution:** Extract structured requirements from natural language using:
1. **LLM-based extraction** (primary) - Structured output via JSON schema
2. **Rule-based validation** - Verify and correct LLM outputs
3. **Ambiguity detection** - Identify missing/contradictory requirements
4. **Interactive clarification** - Ask users for missing information
5. **Requirement validation** - Ensure completeness before graph generation

---

## Table of Contents

1. [Requirement Categories](#1-requirement-categories)
2. [Extraction Techniques](#2-extraction-techniques)
3. [Ambiguity Resolution](#3-ambiguity-resolution)
4. [Requirement Validation](#4-requirement-validation)
5. [Structured Output Format](#5-structured-output-format)
6. [Interactive Clarification Mode](#6-interactive-clarification-mode)
7. [Integration with Graph Generation](#7-integration-with-graph-generation)
8. [Implementation Plan](#8-implementation-plan)
9. [MVP Feature List](#9-mvp-feature-list)

---

## 1. Requirement Categories

### 1.1 Functional Requirements

Define **what** the workflow should do:

```python
@dataclass
class FunctionalRequirements:
    """Core functional requirements for workflow tasks.

    Attributes:
        tasks: List of tasks to perform (in order)
        tools: Tools needed for each task
        inputs: Data inputs required (files, APIs, user input)
        outputs: Expected outputs (reports, code changes, data)
        success_criteria: Conditions for successful completion
    """

    tasks: List[TaskRequirement]
    tools: Dict[str, List[str]]  # task_id -> tool names
    inputs: List[InputRequirement]
    outputs: List[OutputRequirement]
    success_criteria: List[str]


@dataclass
class TaskRequirement:
    """Individual task requirement.

    Attributes:
        id: Unique task identifier
        description: What the task does
        task_type: Type of task (agent, compute, condition, transform)
        role: Agent role (if agent task)
        goal: Goal for the agent (if agent task)
        tools: Tools needed for this task
        dependencies: Other tasks this depends on
    """

    id: str
    description: str
    task_type: str  # agent, compute, condition, transform, parallel
    role: Optional[str] = None
    goal: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class InputRequirement:
    """Input requirement for workflow.

    Attributes:
        name: Input parameter name
        type: Data type (file, text, code, api_response)
        source: Where input comes from (user, file, api, previous_task)
        required: Whether input is required or optional
        validation: How to validate input (regex, schema, custom)
    """

    name: str
    type: str  # file, text, code, api_response, dict
    source: str  # user, file, api, task:{task_id}
    required: bool = True
    validation: Optional[str] = None


@dataclass
class OutputRequirement:
    """Output requirement from workflow.

    Attributes:
        name: Output name
        type: Output type (file, report, code_changes, data)
        destination: Where output goes (file, stdout, api, next_task)
        format: Output format (markdown, json, code, plain_text)
    """

    name: str
    type: str  # file, report, code_changes, data, console
    destination: str  # file, stdout, api, task:{task_id}
    format: str  # markdown, json, code, plain_text
```

### 1.2 Structural Requirements

Define **how** tasks are organized:

```python
@dataclass
class StructuralRequirements:
    """Execution structure requirements.

    Attributes:
        execution_order: Overall execution pattern (sequential, parallel, mixed)
        dependencies: Explicit dependencies between tasks
        branches: Conditional branching logic
        loops: Iterative/repetitive patterns
        joins: How parallel branches merge
    """

    execution_order: str  # sequential, parallel, mixed, conditional
    dependencies: Dict[str, List[str]]  # task_id -> dependent task IDs
    branches: List[BranchRequirement]
    loops: List[LoopRequirement]
    joins: Dict[str, str]  # parallel_task_id -> join_strategy


@dataclass
class BranchRequirement:
    """Conditional branch requirement.

    Attributes:
        condition_id: Unique branch identifier
        condition: Description of branch condition
        true_branch: Task ID if condition is true
        false_branch: Task ID if condition is false
        condition_type: Type of condition (quality_threshold, error_check, user_approval)
    """

    condition_id: str
    condition: str
    true_branch: str
    false_branch: str
    condition_type: str  # quality_threshold, error_check, user_approval, data_check


@dataclass
class LoopRequirement:
    """Loop/repetition requirement.

    Attributes:
        loop_id: Unique loop identifier
        task_to_repeat: Which task to repeat
        exit_condition: When to stop looping
        max_iterations: Maximum times to repeat
    """

    loop_id: str
    task_to_repeat: str
    exit_condition: str
    max_iterations: int = 3
```

### 1.3 Quality Requirements

Define **constraints and performance targets**:

```python
@dataclass
class QualityRequirements:
    """Quality and performance requirements.

    Attributes:
        max_duration_seconds: Maximum workflow execution time
        max_cost_tier: Maximum tool cost tier (FREE, LOW, MEDIUM, HIGH)
        accuracy_threshold: Minimum accuracy/success rate (0.0-1.0)
        max_tool_calls: Maximum total tool calls across workflow
        max_tokens: Maximum LLM tokens to consume
        retry_policy: How to handle failures (retry, fail_fast, continue)
    """

    max_duration_seconds: Optional[int] = None
    max_cost_tier: str = "MEDIUM"
    accuracy_threshold: Optional[float] = None
    max_tool_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    retry_policy: str = "retry"  # retry, fail_fast, continue, fallback


@dataclass
class TaskQualityRequirements:
    """Per-task quality requirements.

    Attributes:
        task_id: Task this applies to
        timeout_seconds: Task-specific timeout
        tool_budget: Max tool calls for this task
        allowed_tools: Specific tools allowed (empty = all)
        quality_threshold: Task-specific quality threshold
    """

    task_id: str
    timeout_seconds: Optional[int] = None
    tool_budget: Optional[int] = None
    allowed_tools: List[str] = field(default_factory=list)
    quality_threshold: Optional[float] = None
```

### 1.4 Context Requirements

Define **environment and domain context**:

```python
@dataclass
class ContextRequirements:
    """Context and environment requirements.

    Attributes:
        vertical: Domain vertical (coding, devops, research, rag, dataanalysis)
        subdomain: Specific subdomain (e.g., "bug_fix", "deployment", "fact_checking")
        environment: Execution environment (local, cloud, sandbox)
        user_preferences: User-specific preferences
        project_context: Project-specific information (repo, language, framework)
    """

    vertical: str  # coding, devops, research, rag, dataanalysis, benchmark
    subdomain: Optional[str] = None
    environment: str = "local"  # local, cloud, sandbox
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    project_context: Optional[ProjectContext] = None


@dataclass
class ProjectContext:
    """Project-specific context.

    Attributes:
        repo_path: Path to codebase repository
        primary_language: Main programming language
        framework: Framework being used (e.g., "FastAPI", "React")
        testing_framework: Test framework (e.g., "pytest", "jest")
        build_system: Build tool (e.g., "make", "npm", "cargo")
    """

    repo_path: Optional[str] = None
    primary_language: Optional[str] = None
    framework: Optional[str] = None
    testing_framework: Optional[str] = None
    build_system: Optional[str] = None
```

---

## 2. Extraction Techniques

### 2.1 LLM-Based Extraction (Primary)

Use LLM with structured output to extract all requirement categories:

```python
class WorkflowRequirementExtractor:
    """Extract workflow requirements from natural language using LLM.

    Uses structured output (JSON schema) to extract all requirement
    categories in a single LLM call.

    Example:
        extractor = WorkflowRequirementExtractor(orchestrator)
        requirements = await extractor.extract(
            "Analyze this codebase, find bugs, fix them, and run tests"
        )
        # requirements.functional.tasks -> [
        #     TaskRequirement(id="analyze", description="Analyze codebase"),
        #     TaskRequirement(id="fix", description="Fix bugs"),
        #     TaskRequirement(id="test", description="Run tests")
        # ]
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator
        self._schema = self._build_json_schema()

    async def extract(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRequirements:
        """Extract structured requirements from natural language.

        Args:
            description: Natural language workflow description
            context: Optional context (project info, user preferences)

        Returns:
            WorkflowRequirements with all extracted information
        """
        prompt = self._build_extraction_prompt(description, context)

        # Call LLM with structured output
        result = await self._call_llm_with_schema(prompt, self._schema)

        # Parse and validate
        requirements = self._parse_requirements(result)

        # Apply rule-based validation
        requirements = self._validate_with_rules(requirements)

        return requirements
```

**Extraction Prompt Template:**

```python
EXTRACTION_PROMPT = """
You are a workflow analysis expert. Extract structured workflow requirements
from the user's natural language description.

User Description:
{description}

Context:
{context}

Extract the following information:

1. **Functional Requirements**:
   - Tasks: What steps should be performed? (in order)
   - Tools: What tools are needed for each task?
   - Inputs: What data/files are needed?
   - Outputs: What should be produced?
   - Success Criteria: How do we know the workflow succeeded?

2. **Structural Requirements**:
   - Execution Order: Sequential, parallel, or conditional?
   - Dependencies: Which tasks depend on others?
   - Branches: Are there any conditional paths?
   - Loops: Are there any repetitive patterns?

3. **Quality Requirements**:
   - Performance Constraints: Max duration, max cost?
   - Quality Targets: Accuracy thresholds?
   - Resource Limits: Max tool calls, max tokens?

4. **Context Requirements**:
   - Domain: Which vertical (coding, devops, research, etc.)?
   - Environment: Local, cloud, or sandbox?
   - Preferences: Any user preferences mentioned?

Respond with a JSON object following this schema:
{schema}

Guidelines:
- Be specific but make reasonable assumptions if unclear
- If uncertain, set confidence score lower
- Extract conditional phrases ("if X then Y") as branches
- Detect parallel patterns ("do X and Y simultaneously")
- Identify success criteria explicitly stated or implied

Examples:

Input: "Analyze this Python codebase, find bugs, fix them, and run pytest to verify"
Output:
{{
  "functional": {{
    "tasks": [
      {{"id": "analyze", "description": "Analyze Python codebase", "task_type": "agent"}},
      {{"id": "find_bugs", "description": "Find bugs in code", "task_type": "agent", "dependencies": ["analyze"]}},
      {{"id": "fix", "description": "Fix identified bugs", "task_type": "agent", "dependencies": ["find_bugs"]}},
      {{"id": "test", "description": "Run pytest to verify fixes", "task_type": "compute", "dependencies": ["fix"]}}
    ],
    "tools": {{
      "analyze": ["code_search", "ast_analyzer"],
      "find_bugs": ["code_search", "linter"],
      "test": ["bash"]
    }},
    "success_criteria": ["All tests pass", "No critical bugs remain"]
  }},
  "structural": {{
    "execution_order": "sequential",
    "dependencies": {{"find_bugs": ["analyze"], "fix": ["find_bugs"], "test": ["fix"]}}
  }},
  "context": {{
    "vertical": "coding",
    "subdomain": "bug_fix",
    "project_context": {{"primary_language": "Python"}}
  }}
}}

Input: "Research AI trends from 5 sources, summarize, and if quality score > 0.8, create report"
Output:
{{
  "functional": {{
    "tasks": [
      {{"id": "research", "description": "Research AI trends from 5 sources", "task_type": "agent"}},
      {{"id": "summarize", "description": "Summarize findings", "task_type": "agent", "dependencies": ["research"]}},
      {{"id": "create_report", "description": "Create report", "task_type": "agent", "dependencies": ["summarize"]}}
    ],
    "success_criteria": ["Quality score > 0.8", "At least 5 sources cited"]
  }},
  "structural": {{
    "execution_order": "conditional",
    "branches": [
      {{
        "condition_id": "quality_check",
        "condition": "quality_score > 0.8",
        "true_branch": "create_report",
        "false_branch": "end",
        "condition_type": "quality_threshold"
      }}
    ]
  }}
}}
"""
```

### 2.2 Rule-Based Extraction (Fallback)

When LLM extraction fails or for validation:

```python
class RuleBasedRequirementExtractor:
    """Fallback extractor using rules and patterns.

    Uses keyword matching, regex patterns, and heuristics to extract
    requirements when LLM extraction is unavailable or failed.

    Example:
        extractor = RuleBasedRequirementExtractor()
        requirements = extractor.extract("Run tests and if they pass, deploy")
        # Detects: sequential execution with conditional branch
    """

    def __init__(self):
        self._patterns = self._load_patterns()

    def extract(self, description: str) -> WorkflowRequirements:
        """Extract requirements using rule-based patterns."""
        # Detect execution order
        execution_order = self._detect_execution_order(description)

        # Extract tasks using verb patterns
        tasks = self._extract_tasks(description)

        # Detect conditional keywords
        branches = self._detect_branches(description)

        # Extract tool mentions
        tools = self._extract_tools(description)

        return WorkflowRequirements(
            functional=FunctionalRequirements(tasks=tasks, tools=tools),
            structural=StructuralRequirements(
                execution_order=execution_order,
                branches=branches,
            ),
        )

    def _detect_execution_order(self, text: str) -> str:
        """Detect execution order from keywords."""
        parallel_keywords = ["and", "simultaneously", "in parallel", "concurrently"]
        conditional_keywords = ["if", "when", "unless", "otherwise", "else"]

        text_lower = text.lower()

        if any(kw in text_lower for kw in conditional_keywords):
            return "conditional"
        elif any(kw in text_lower for kw in parallel_keywords):
            return "parallel"
        else:
            return "sequential"

    def _extract_tasks(self, text: str) -> List[TaskRequirement]:
        """Extract tasks using verb-noun patterns."""
        # Pattern: verb + noun phrase
        # "analyze code", "run tests", "deploy to staging"
        import re

        # Common task verbs
        task_verbs = [
            "analyze",
            "deploy",
            "test",
            "fix",
            "research",
            "summarize",
            "create",
            "generate",
            "review",
            "build",
            "compile",
            "document",
        ]

        tasks = []
        for i, verb in enumerate(task_verbs):
            pattern = rf"\b{verb}\s+([^.!?]+)[.!?]?"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                tasks.append(
                    TaskRequirement(
                        id=f"task_{i}",
                        description=match.group(0).strip(),
                        task_type="agent",  # Default to agent
                    )
                )

        return tasks

    def _detect_branches(self, text: str) -> List[BranchRequirement]:
        """Detect conditional branches from text."""
        import re

        branches = []

        # Pattern: "if X then Y else Z"
        pattern = r"if\s+(.+?)\s+then\s+(.+?)(?:\s+else\s+(.+?))?(?:\.|$)"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for i, match in enumerate(matches):
            condition = match.group(1).strip()
            true_branch = match.group(2).strip()
            false_branch = match.group(3).strip() if match.group(3) else None

            branches.append(
                BranchRequirement(
                    condition_id=f"branch_{i}",
                    condition=condition,
                    true_branch=true_branch,
                    false_branch=false_branch or "end",
                    condition_type="data_check",
                )
            )

        return branches

    def _extract_tools(self, text: str) -> Dict[str, List[str]]:
        """Extract tool names from text."""
        # Known tool names (from tool registry)
        known_tools = [
            "bash",
            "code_search",
            "file_read",
            "web_search",
            "git",
            "pytest",
            "npm",
            "docker",
        ]

        tools = {}
        text_lower = text.lower()

        for tool in known_tools:
            if tool in text_lower:
                # Associate with all tasks (will be refined later)
                for task_id in ["task_0"]:  # Placeholder
                    tools.setdefault(task_id, []).append(tool)

        return tools
```

### 2.3 Hybrid Approach

Combine LLM and rule-based extraction:

```python
class HybridRequirementExtractor:
    """Combines LLM and rule-based extraction.

    Strategy:
    1. Use LLM for primary extraction (handles ambiguity well)
    2. Validate LLM output with rules (catch hallucinations)
    3. Use rules to fill missing fields (graceful degradation)
    4. Fallback to pure rule-based if LLM fails
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self._llm_extractor = WorkflowRequirementExtractor(orchestrator)
        self._rule_extractor = RuleBasedRequirementExtractor()

    async def extract(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRequirements:
        """Extract requirements using hybrid approach."""
        try:
            # Try LLM extraction first
            requirements = await self._llm_extractor.extract(description, context)

            # Validate with rules
            validation_errors = self._validate_with_rules(requirements)

            if validation_errors:
                # Use rules to correct errors
                requirements = self._correct_with_rules(
                    requirements, validation_errors
                )

            return requirements

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to rules")
            # Fallback to rule-based
            return self._rule_extractor.extract(description)

    def _validate_with_rules(self, requirements: WorkflowRequirements) -> List[str]:
        """Validate LLM output with rule-based checks."""
        errors = []

        # Check: Tasks must have descriptions
        for task in requirements.functional.tasks:
            if not task.description:
                errors.append(f"Task {task.id} missing description")

        # Check: Branches must have condition
        for branch in requirements.structural.branches:
            if not branch.condition:
                errors.append(f"Branch {branch.condition_id} missing condition")

        # Check: Tool names must be valid
        all_tools = []
        for tools in requirements.functional.tools.values():
            all_tools.extend(tools)

        # Would validate against tool registry
        # invalid_tools = [t for t in all_tools if t not in TOOL_REGISTRY]
        # if invalid_tools:
        #     errors.append(f"Invalid tools: {invalid_tools}")

        return errors

    def _correct_with_rules(
        self,
        requirements: WorkflowRequirements,
        errors: List[str],
    ) -> WorkflowRequirements:
        """Correct LLM output using rules."""
        # Apply rule-based corrections
        for error in errors:
            if "missing description" in error:
                # Extract task ID
                task_id = error.split()[1]
                # Use rule-based extraction to fill
                # ...

        return requirements
```

---

## 3. Ambiguity Resolution

### 3.1 Ambiguity Detection

Identify missing or unclear requirements:

```python
class AmbiguityDetector:
    """Detect ambiguities in extracted requirements.

    Finds:
    - Missing required fields
    - Contradictory requirements
    - Vague descriptions
    - Infeasible combinations
    """

    def detect(self, requirements: WorkflowRequirements) -> List[Ambiguity]:
        """Detect all ambiguities in requirements.

        Returns:
            List of Ambiguity objects sorted by severity
        """
        ambiguities = []

        # Check for missing functional requirements
        ambiguities.extend(self._check_functional_gaps(requirements))

        # Check for structural inconsistencies
        ambiguities.extend(self._check_structural_conflicts(requirements))

        # Check for vague descriptions
        ambiguities.extend(self._check_vagueness(requirements))

        # Check for infeasible constraints
        ambiguities.extend(self._check_feasibility(requirements))

        # Sort by severity
        ambiguities.sort(key=lambda a: a.severity, reverse=True)

        return ambiguities

    def _check_functional_gaps(
        self,
        requirements: WorkflowRequirements,
    ) -> List[Ambiguity]:
        """Check for missing functional requirements."""
        gaps = []

        # Check: At least one task
        if not requirements.functional.tasks:
            gaps.append(
                Ambiguity(
                    type="missing_tasks",
                    severity=10,
                    message="No tasks detected",
                    suggestion="What should this workflow do?",
                    field="functional.tasks",
                )
            )

        # Check: Tasks have tools or roles
        for task in requirements.functional.tasks:
            if task.task_type == "agent" and not task.role:
                gaps.append(
                    Ambiguity(
                        type="missing_role",
                        severity=7,
                        message=f"Task '{task.id}' has no agent role specified",
                        suggestion="What role should perform this task? (researcher, executor, planner, reviewer)",
                        field=f"functional.tasks.{task.id}.role",
                    )
                )

        # Check: Success criteria defined
        if not requirements.functional.success_criteria:
            gaps.append(
                Ambiguity(
                    type="missing_success_criteria",
                    severity=5,
                    message="No success criteria defined",
                    suggestion="How do we know the workflow succeeded?",
                    field="functional.success_criteria",
                )
            )

        return gaps

    def _check_structural_conflicts(
        self,
        requirements: WorkflowRequirements,
    ) -> List[Ambiguity]:
        """Check for structural inconsistencies."""
        conflicts = []

        # Check: Circular dependencies
        graph = self._build_dependency_graph(requirements)
        cycles = self._detect_cycles(graph)
        if cycles:
            conflicts.append(
                Ambiguity(
                    type="circular_dependency",
                    severity=9,
                    message=f"Circular dependency detected: {' -> '.join(cycles[0])}",
                    suggestion="Break the cycle by removing one dependency",
                    field="structural.dependencies",
                )
            )

        # Check: Parallel tasks with sequential dependencies
        if (
            requirements.structural.execution_order == "parallel"
            and requirements.structural.dependencies
        ):
            conflicts.append(
                Ambiguity(
                    type="contradictory_structure",
                    severity=6,
                    message="Parallel execution with dependencies may be inefficient",
                    suggestion="Consider sequential execution or remove dependencies",
                    field="structural.execution_order",
                )
            )

        # Check: Branch without condition
        for branch in requirements.structural.branches:
            if not branch.condition or branch.condition == "unknown":
                conflicts.append(
                    Ambiguity(
                        type="missing_condition",
                        severity=8,
                        message=f"Branch '{branch.condition_id}' has no condition",
                        suggestion="What condition should trigger this branch?",
                        field=f"structural.branches.{branch.condition_id}.condition",
                    )
                )

        return conflicts

    def _check_vagueness(
        self,
        requirements: WorkflowRequirements,
    ) -> List[Ambiguity]:
        """Check for vague descriptions."""
        vague = []

        # Vague task descriptions
        vague_words = ["something", "stuff", "things", "handle", "process"]
        for task in requirements.functional.tasks:
            words = task.description.lower().split()
            if any(v in words for v in vague_words):
                vague.append(
                    Ambiguity(
                        type="vague_description",
                        severity=4,
                        message=f"Task '{task.id}' has vague description",
                        suggestion="Be more specific about what this task does",
                        field=f"functional.tasks.{task.id}.description",
                    )
                )

        # Vague success criteria
        for criteria in requirements.functional.success_criteria:
            if len(criteria) < 10:
                vague.append(
                    Ambiguity(
                        type="vague_criteria",
                        severity=3,
                        message=f"Success criteria is too brief: '{criteria}'",
                        suggestion="Provide specific, measurable criteria",
                        field="functional.success_criteria",
                    )
                )

        return vague

    def _check_feasibility(
        self,
        requirements: WorkflowRequirements,
    ) -> List[Ambiguity]:
        """Check for infeasible constraints."""
        infeasible = []

        # Check: Too many tool calls for budget
        if requirements.quality.max_tool_calls:
            estimated_calls = len(requirements.functional.tasks) * 10  # Rough estimate
            if estimated_calls > requirements.quality.max_tool_calls:
                infeasible.append(
                    Ambiguity(
                        type="infeasible_constraint",
                        severity=7,
                        message=f"Estimated {estimated_calls} tool calls exceeds budget of {requirements.quality.max_tool_calls}",
                        suggestion=f"Increase tool budget to {estimated_calls} or reduce tasks",
                        field="quality.max_tool_calls",
                    )
                )

        # Check: Timeout too short for task count
        if requirements.quality.max_duration_seconds:
            min_seconds = len(requirements.functional.tasks) * 30  # 30s per task min
            if requirements.quality.max_duration_seconds < min_seconds:
                infeasible.append(
                    Ambiguity(
                        type="infeasible_constraint",
                        severity=8,
                        message=f"Timeout {requirements.quality.max_duration_seconds}s too short for {len(requirements.functional.tasks)} tasks",
                        suggestion=f"Increase timeout to at least {min_seconds}s",
                        field="quality.max_duration_seconds",
                    )
                )

        return infeasible


@dataclass
class Ambiguity:
    """An ambiguity detected in requirements.

    Attributes:
        type: Type of ambiguity (missing, conflict, vague, infeasible)
        severity: Severity score (1-10, 10 = critical)
        message: Human-readable description
        suggestion: Suggested resolution
        field: JSON path to ambiguous field
        options: Multiple choice options (if applicable)
    """

    type: str
    severity: int
    message: str
    suggestion: str
    field: str
    options: Optional[List[str]] = None
```

### 3.2 Resolution Strategies

Multiple strategies for resolving ambiguities:

```python
class AmbiguityResolver:
    """Resolve ambiguities in workflow requirements.

    Strategies:
    1. Interactive: Ask user for clarification (preferred)
    2. Assumptions: Make reasonable assumptions with confidence scores
    3. Defaults: Use template defaults for common patterns
    4. Examples: Request specific examples from user
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator

    async def resolve(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
        strategy: str = "interactive",
    ) -> WorkflowRequirements:
        """Resolve ambiguities using specified strategy.

        Args:
            requirements: Requirements with ambiguities
            ambiguities: List of detected ambiguities
            strategy: Resolution strategy (interactive, assumptions, defaults)

        Returns:
            Resolved requirements
        """
        if strategy == "interactive":
            return await self._resolve_interactively(requirements, ambiguities)
        elif strategy == "assumptions":
            return self._resolve_with_assumptions(requirements, ambiguities)
        elif strategy == "defaults":
            return self._resolve_with_defaults(requirements, ambiguities)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def _resolve_interactively(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
    ) -> WorkflowRequirements:
        """Resolve by asking user questions."""
        from rich.console import Console
        from rich.prompt import Prompt

        console = Console()

        # Sort by severity (handle critical first)
        ambiguities.sort(key=lambda a: a.severity, reverse=True)

        for ambiguity in ambiguities:
            console.print(f"\n[bold yellow]⚠ {ambiguity.message}[/bold yellow]")
            console.print(f"[dim]Suggestion: {ambiguity.suggestion}[/dim]")

            if ambiguity.options:
                # Multiple choice
                console.print("\nOptions:")
                for i, option in enumerate(ambiguity.options, 1):
                    console.print(f"  {i}. {option}")

                choice = Prompt.ask(
                    "Choose an option",
                    choices=[str(i) for i in range(1, len(ambiguity.options) + 1)],
                )
                value = ambiguity.options[int(choice) - 1]
            else:
                # Free-form input
                value = Prompt.ask("Your answer")

            # Apply resolution
            requirements = self._apply_resolution(
                requirements, ambiguity.field, value
            )

        return requirements

    def _resolve_with_assumptions(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
    ) -> WorkflowRequirements:
        """Resolve by making assumptions."""
        # Build prompt for LLM
        prompt = self._build_assumption_prompt(requirements, ambiguities)

        # Call LLM to generate assumptions
        assumptions = await self._generate_assumptions(prompt)

        # Apply assumptions with confidence scores
        for assumption in assumptions:
            if assumption.confidence > 0.7:
                requirements = self._apply_resolution(
                    requirements,
                    assumption.field,
                    assumption.value,
                )

        return requirements

    def _resolve_with_defaults(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
    ) -> WorkflowRequirements:
        """Resolve using template defaults."""
        defaults = self._load_defaults()

        for ambiguity in ambiguities:
            field = ambiguity.field

            # Look up default
            if field in defaults:
                requirements = self._apply_resolution(
                    requirements,
                    field,
                    defaults[field],
                )

        return requirements

    def _apply_resolution(
        self,
        requirements: WorkflowRequirements,
        field: str,
        value: Any,
    ) -> WorkflowRequirements:
        """Apply a resolution to the requirements."""
        # Parse field path (e.g., "functional.tasks.0.role")
        parts = field.split(".")

        if parts[0] == "functional":
            return self._apply_functional_resolution(requirements, parts[1:], value)
        elif parts[0] == "structural":
            return self._apply_structural_resolution(requirements, parts[1:], value)
        elif parts[0] == "quality":
            return self._apply_quality_resolution(requirements, parts[1:], value)
        elif parts[0] == "context":
            return self._apply_context_resolution(requirements, parts[1:], value)
        else:
            logger.warning(f"Unknown field path: {field}")
            return requirements
```

---

## 4. Requirement Validation

### 4.1 Validation Checks

Comprehensive validation before graph generation:

```python
class RequirementValidator:
    """Validate workflow requirements for completeness and consistency.

    Checks:
    - Completeness: All required info present?
    - Consistency: No contradictions?
    - Feasibility: Can this be implemented?
    - Specificity: Enough detail for generation?
    """

    def validate(
        self,
        requirements: WorkflowRequirements,
    ) -> ValidationResult:
        """Perform comprehensive validation.

        Returns:
            ValidationResult with errors, warnings, and recommendations
        """
        errors = []
        warnings = []
        recommendations = []

        # Completeness checks
        completeness_errors = self._check_completeness(requirements)
        errors.extend(completeness_errors)

        # Consistency checks
        consistency_errors = self._check_consistency(requirements)
        errors.extend(consistency_errors)

        # Feasibility checks
        feasibility_errors = self._check_feasibility(requirements)
        errors.extend(feasibility_errors)

        # Specificity checks
        specificity_warnings = self._check_specificity(requirements)
        warnings.extend(specificity_warnings)

        # Generate recommendations
        recommendations = self._generate_recommendations(requirements)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            score=self._compute_score(requirements, errors, warnings),
        )

    def _check_completeness(
        self,
        requirements: WorkflowRequirements,
    ) -> List[ValidationError]:
        """Check for missing required information."""
        errors = []

        # Must have at least one task
        if not requirements.functional.tasks:
            errors.append(
                ValidationError(
                    field="functional.tasks",
                    message="No tasks defined",
                    severity="critical",
                )
            )

        # Each task must have description
        for task in requirements.functional.tasks:
            if not task.description:
                errors.append(
                    ValidationError(
                        field=f"functional.tasks.{task.id}",
                        message=f"Task '{task.id}' missing description",
                        severity="critical",
                    )
                )

        # Agent tasks must have role
        for task in requirements.functional.tasks:
            if task.task_type == "agent" and not task.role:
                errors.append(
                    ValidationError(
                        field=f"functional.tasks.{task.id}.role",
                        message=f"Agent task '{task.id}' missing role",
                        severity="critical",
                    )
                )

        # Branches must have conditions
        for branch in requirements.structural.branches:
            if not branch.condition:
                errors.append(
                    ValidationError(
                        field=f"structural.branches.{branch.condition_id}",
                        message=f"Branch '{branch.condition_id}' missing condition",
                        severity="critical",
                    )
                )

        # Must have context (vertical)
        if not requirements.context.vertical:
            errors.append(
                ValidationError(
                    field="context.vertical",
                    message="Domain vertical not specified",
                    severity="warning",
                )
            )

        return errors

    def _check_consistency(
        self,
        requirements: WorkflowRequirements,
    ) -> List[ValidationError]:
        """Check for contradictions."""
        errors = []

        # Check for circular dependencies
        dependencies = requirements.structural.dependencies
        if dependencies:
            cycles = self._detect_cycles(dependencies)
            if cycles:
                errors.append(
                    ValidationError(
                        field="structural.dependencies",
                        message=f"Circular dependency: {' -> '.join(cycles[0])}",
                        severity="error",
                    )
                )

        # Check: Parallel execution with dependencies
        if (
            requirements.structural.execution_order == "parallel"
            and dependencies
        ):
            errors.append(
                ValidationError(
                    field="structural.execution_order",
                    message="Parallel execution with dependencies is inefficient",
                    severity="warning",
                )
            )

        # Check: Conflicting quality constraints
        if (
            requirements.quality.max_duration_seconds
            and len(requirements.functional.tasks) > 0
        ):
            min_time = len(requirements.functional.tasks) * 30  # 30s per task
            if requirements.quality.max_duration_seconds < min_time:
                errors.append(
                    ValidationError(
                        field="quality.max_duration_seconds",
                        message=f"Timeout ({requirements.quality.max_duration_seconds}s) "
                        f"too short for {len(requirements.functional.tasks)} tasks "
                        f"(minimum: {min_time}s)",
                        severity="error",
                    )
                )

        return errors

    def _check_feasibility(
        self,
        requirements: WorkflowRequirements,
    ) -> List[ValidationError]:
        """Check if requirements can be implemented."""
        errors = []

        # Check: Tools exist in registry
        all_tools = set()
        for tools in requirements.functional.tools.values():
            all_tools.update(tools)

        # Would validate against tool registry
        # missing_tools = [t for t in all_tools if t not in TOOL_REGISTRY]
        # if missing_tools:
        #     errors.append(
        #         ValidationError(
        #             field="functional.tools",
        #             message=f"Tools not found in registry: {missing_tools}",
        #             severity="warning",
        #         )
        #     )

        # Check: Too many tasks for single workflow
        if len(requirements.functional.tasks) > 20:
            errors.append(
                ValidationError(
                    field="functional.tasks",
                    message=f"Too many tasks ({len(requirements.functional.tasks)}), "
                    "consider splitting into multiple workflows",
                    severity="warning",
                )
            )

        # Check: Complex structures
        if len(requirements.structural.branches) > 5:
            errors.append(
                ValidationError(
                    field="structural.branches",
                    message=f"Too many branches ({len(requirements.structural.branches)}), "
                    "workflow may be hard to debug",
                    severity="info",
                )
            )

        return errors

    def _check_specificity(
        self,
        requirements: WorkflowRequirements,
    ) -> List[ValidationError]:
        """Check if requirements are specific enough."""
        warnings = []

        # Check for vague task descriptions
        vague_patterns = [
            r"\bsomething\b",
            r"\bstuff\b",
            r"\bthings\b",
            r"\bhandle it\b",
            r"\bprocess\b",
        ]

        for task in requirements.functional.tasks:
            for pattern in vague_patterns:
                if re.search(pattern, task.description, re.IGNORECASE):
                    warnings.append(
                        ValidationError(
                            field=f"functional.tasks.{task.id}.description",
                            message=f"Task '{task.id}' has vague description",
                            severity="warning",
                        )
                    )
                    break

        # Check: Success criteria measurable
        for criteria in requirements.functional.success_criteria:
            # Check for measurable indicators (numbers, specific outcomes)
            if not any(
                indicator in criteria.lower()
                for indicator in ["%", "pass", "fail", "score", "error", "success"]
            ):
                warnings.append(
                    ValidationError(
                        field="functional.success_criteria",
                        message=f"Success criteria may not be measurable: '{criteria}'",
                        severity="info",
                    )
                )

        return warnings


@dataclass
class ValidationResult:
    """Result of requirement validation.

    Attributes:
        is_valid: Whether requirements are valid for generation
        errors: Critical issues that must be fixed
        warnings: Non-critical issues
        recommendations: Suggestions for improvement
        score: Overall quality score (0.0-1.0)
    """

    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    recommendations: List[str]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "recommendations": self.recommendations,
        }


@dataclass
class ValidationError:
    """A validation error or warning.

    Attributes:
        field: Field that failed validation (JSON path)
        message: Human-readable error message
        severity: Severity level (critical, error, warning, info)
        suggestion: Optional suggestion for fixing
    """

    field: str
    message: str
    severity: str
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }
```

### 4.2 Validation Feedback

Present validation results clearly to users:

```python
class ValidationFeedbackPresenter:
    """Present validation results in user-friendly format.

    Formats:
    - Console output (rich tables)
    - JSON (for API)
    - Markdown (for docs)
    """

    def present_console(self, result: ValidationResult) -> None:
        """Present validation results in console."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Overall status
        if result.is_valid:
            console.print(
                f"[bold green]✓ Requirements Valid[/bold green] "
                f"(score: {result.score:.2f})"
            )
        else:
            console.print(
                f"[bold red]✗ Validation Failed[/bold red] "
                f"(score: {result.score:.2f})"
            )

        # Errors table
        if result.errors:
            console.print("\n[bold red]Errors[/bold red]")
            errors_table = Table(show_header=True)
            errors_table.add_column("Field", style="cyan")
            errors_table.add_column("Severity", style="red")
            errors_table.add_column("Message")

            for error in result.errors:
                errors_table.add_row(
                    error.field,
                    error.severity,
                    error.message,
                )

            console.print(errors_table)

        # Warnings table
        if result.warnings:
            console.print("\n[bold yellow]Warnings[/bold yellow]")
            warnings_table = Table(show_header=True)
            warnings_table.add_column("Field", style="cyan")
            warnings_table.add_column("Severity", style="yellow")
            warnings_table.add_column("Message")

            for warning in result.warnings:
                warnings_table.add_row(
                    warning.field,
                    warning.severity,
                    warning.message,
                )

            console.print(warnings_table)

        # Recommendations
        if result.recommendations:
            console.print("\n[bold blue]Recommendations[/bold blue]")
            for i, rec in enumerate(result.recommendations, 1):
                console.print(f"  {i}. {rec}")
```

---

## 5. Structured Output Format

### 5.1 Complete Schema

```python
@dataclass
class WorkflowRequirements:
    """Complete workflow requirements extracted from natural language.

    This is the primary output of the requirement extraction system.
    It contains all information needed to generate a workflow graph.

    Attributes:
        description: Original natural language description
        functional: Functional requirements (tasks, tools, I/O)
        structural: Structural requirements (order, dependencies, branches)
        quality: Quality requirements (performance, constraints)
        context: Context requirements (domain, environment)
        confidence_scores: Confidence scores for each section (0.0-1.0)
        metadata: Extraction metadata
    """

    description: str
    functional: FunctionalRequirements
    structural: StructuralRequirements
    quality: QualityRequirements
    context: ContextRequirements
    confidence_scores: Dict[str, float]
    metadata: ExtractionMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "description": self.description,
            "functional": self.functional.to_dict(),
            "structural": self.structural.to_dict(),
            "quality": self.quality.to_dict(),
            "context": self.context.to_dict(),
            "confidence_scores": self.confidence_scores,
            "metadata": self.metadata.to_dict(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ExtractionMetadata:
    """Metadata about the extraction process.

    Attributes:
        extraction_method: Method used (llm, rules, hybrid)
        model: LLM model used (if applicable)
        extraction_time: Time taken for extraction
        ambiguity_count: Number of ambiguities detected
        resolution_strategy: How ambiguities were resolved
        confidence: Overall confidence score (0.0-1.0)
    """

    extraction_method: str  # llm, rules, hybrid
    model: Optional[str] = None
    extraction_time: float = 0.0
    ambiguity_count: int = 0
    resolution_strategy: str = "interactive"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extraction_method": self.extraction_method,
            "model": self.model,
            "extraction_time": self.extraction_time,
            "ambiguity_count": self.ambiguity_count,
            "resolution_strategy": self.resolution_strategy,
            "confidence": self.confidence,
        }
```

### 5.2 JSON Schema for LLM

```python
def build_requirements_schema() -> Dict[str, Any]:
    """Build JSON schema for LLM structured output.

    This schema is passed to the LLM to ensure it returns valid
    workflow requirements in the expected format.
    """
    return {
        "type": "object",
        "properties": {
            "functional": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "description": {"type": "string"},
                                "task_type": {
                                    "type": "string",
                                    "enum": [
                                        "agent",
                                        "compute",
                                        "condition",
                                        "transform",
                                        "parallel",
                                    ],
                                },
                                "role": {
                                    "type": "string",
                                    "enum": [
                                        "researcher",
                                        "executor",
                                        "planner",
                                        "reviewer",
                                        "writer",
                                    ],
                                },
                                "goal": {"type": "string"},
                                "tools": {"type": "array", "items": {"type": "string"}},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["id", "description", "task_type"],
                        },
                    },
                    "tools": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "inputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "source": {"type": "string"},
                                "required": {"type": "boolean"},
                            },
                        },
                    },
                    "outputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "destination": {"type": "string"},
                                "format": {"type": "string"},
                            },
                        },
                    },
                    "success_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["tasks", "success_criteria"],
            },
            "structural": {
                "type": "object",
                "properties": {
                    "execution_order": {
                        "type": "string",
                        "enum": ["sequential", "parallel", "mixed", "conditional"],
                    },
                    "dependencies": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "branches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "condition_id": {"type": "string"},
                                "condition": {"type": "string"},
                                "true_branch": {"type": "string"},
                                "false_branch": {"type": "string"},
                                "condition_type": {"type": "string"},
                            },
                        },
                    },
                    "loops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "loop_id": {"type": "string"},
                                "task_to_repeat": {"type": "string"},
                                "exit_condition": {"type": "string"},
                                "max_iterations": {"type": "integer"},
                            },
                        },
                    },
                },
            },
            "quality": {
                "type": "object",
                "properties": {
                    "max_duration_seconds": {"type": "integer"},
                    "max_cost_tier": {
                        "type": "string",
                        "enum": ["FREE", "LOW", "MEDIUM", "HIGH"],
                    },
                    "accuracy_threshold": {"type": "number"},
                    "max_tool_calls": {"type": "integer"},
                    "max_tokens": {"type": "integer"},
                    "retry_policy": {
                        "type": "string",
                        "enum": ["retry", "fail_fast", "continue", "fallback"],
                    },
                },
            },
            "context": {
                "type": "object",
                "properties": {
                    "vertical": {
                        "type": "string",
                        "enum": [
                            "coding",
                            "devops",
                            "research",
                            "rag",
                            "dataanalysis",
                            "benchmark",
                        ],
                    },
                    "subdomain": {"type": "string"},
                    "environment": {
                        "type": "string",
                        "enum": ["local", "cloud", "sandbox"],
                    },
                    "user_preferences": {"type": "object"},
                    "project_context": {
                        "type": "object",
                        "properties": {
                            "repo_path": {"type": "string"},
                            "primary_language": {"type": "string"},
                            "framework": {"type": "string"},
                            "testing_framework": {"type": "string"},
                            "build_system": {"type": "string"},
                        },
                    },
                },
                "required": ["vertical"],
            },
        },
        "required": ["functional", "structural", "quality", "context"],
    }
```

---

## 6. Interactive Clarification Mode

### 6.1 Conversational Interface

Engage users in dialogue to resolve ambiguities:

```python
class InteractiveClarifier:
    """Interactive clarification system for workflow requirements.

    Engages user in conversation to:
    - Ask about missing information
    - Confirm assumptions
    - Resolve conflicts
    - Refine vague descriptions

    Example:
        clarifier = InteractiveClarifier(orchestrator)
        requirements = await clarifier.clarify(initial_requirements)
        # Asks user questions interactively
        # Returns refined requirements
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator
        self._question_generator = QuestionGenerator()
        self._response_parser = ResponseParser()

    async def clarify(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
    ) -> WorkflowRequirements:
        """Clarify requirements through interactive dialogue.

        Args:
            requirements: Initial requirements with ambiguities
            ambiguities: Detected ambiguities

        Returns:
            Refined requirements with ambiguities resolved
        """
        from rich.console import Console
        from rich.prompt import Prompt

        console = Console()

        console.print(
            "\n[bold cyan]🔍 Clarifying Workflow Requirements[/bold cyan]\n"
        )

        # Sort ambiguities by priority
        prioritized = self._prioritize_ambiguities(ambiguities)

        for ambiguity in prioritized:
            # Generate question
            question = self._question_generator.generate(ambiguity)

            # Present question
            console.print(f"[bold yellow]Question:[/bold yellow] {question.text}")

            if question.options:
                # Multiple choice
                console.print("\nOptions:")
                for i, option in enumerate(question.options, 1):
                    console.print(f"  {i}. {option}")

                choice = Prompt.ask(
                    "Choose an option",
                    choices=[str(i) for i in range(1, len(question.options) + 1)],
                    default=question.default,
                )
                answer = question.options[int(choice) - 1]
            else:
                # Free-form input
                answer = Prompt.ask("Your answer", default=question.default or "")

            # Parse answer and apply
            requirements = self._apply_answer(requirements, ambiguity, answer)

            console.print("[dim]✓ Applied[/dim]\n")

        # Show final requirements for confirmation
        self._show_summary(requirements, console)

        confirmed = Prompt.ask(
            "\n[bold]Generate workflow from these requirements?[/bold]",
            choices=["y", "n"],
            default="y",
        )

        if confirmed.lower() != "y":
            # Allow further refinement
            console.print("\n[dim]You can refine requirements further:[/dim]")
            # ...

        return requirements

    def _prioritize_ambiguities(
        self,
        ambiguities: List[Ambiguity],
    ) -> List[Ambiguity]:
        """Prioritize ambiguities by importance.

        Priority order:
        1. Critical (missing required fields)
        2. High (conflicts, infeasible constraints)
        3. Medium (vague descriptions)
        4. Low (nice-to-have improvements)
        """
        # Sort by severity score
        return sorted(ambiguities, key=lambda a: a.severity, reverse=True)

    def _show_summary(
        self,
        requirements: WorkflowRequirements,
        console: Console,
    ) -> None:
        """Show summary of extracted requirements."""
        console.print("\n[bold cyan]📋 Requirements Summary[/bold cyan]\n")

        # Tasks
        console.print("[bold]Tasks:[/bold]")
        for i, task in enumerate(requirements.functional.tasks, 1):
            deps = (
                f" (after: {', '.join(task.dependencies)})"
                if task.dependencies
                else ""
            )
            console.print(f"  {i}. {task.description}{deps}")

        # Structure
        console.print(f"\n[bold]Structure:[/bold] {requirements.structural.execution_order}")

        if requirements.structural.branches:
            console.print("[bold]Branches:[/bold]")
            for branch in requirements.structural.branches:
                console.print(f"  - If {branch.condition} → {branch.true_branch}")

        # Constraints
        if requirements.quality.max_duration_seconds:
            console.print(
                f"\n[bold]Constraints:[/bold] "
                f"{requirements.quality.max_duration_seconds}s timeout, "
                f"{requirements.quality.max_cost_tier} cost tier"
            )

        # Success criteria
        if requirements.functional.success_criteria:
            console.print("\n[bold]Success Criteria:[/bold]")
            for criteria in requirements.functional.success_criteria:
                console.print(f"  - {criteria}")


@dataclass
class Question:
    """A question to ask the user.

    Attributes:
        text: Question text
        options: Multiple choice options (None = free-form)
        default: Default answer
        field: Field this question resolves
    """

    text: str
    options: Optional[List[str]] = None
    default: Optional[str] = None
    field: str = ""


class QuestionGenerator:
    """Generate clarifying questions from ambiguities."""

    def generate(self, ambiguity: Ambiguity) -> Question:
        """Generate a question for an ambiguity.

        Args:
            ambiguity: The ambiguity to clarify

        Returns:
            Question to ask user
        """
        # Map ambiguity types to question templates
        templates = {
            "missing_tasks": Question(
                text="What should this workflow do?",
                default="Not specified",
            ),
            "missing_role": Question(
                text="What role should perform the '{task_id}' task?",
                options=[
                    "researcher",
                    "executor",
                    "planner",
                    "reviewer",
                    "writer",
                ],
                default="executor",
            ),
            "missing_condition": Question(
                text="What condition should trigger this branch?",
                default="success",
            ),
            "vague_description": Question(
                text="Can you be more specific about what this task does?",
                default="Use current description",
            ),
            "infeasible_constraint": Question(
                text="{message}",
                options=["Adjust constraint", "Reduce tasks", "Ignore warning"],
                default="Adjust constraint",
            ),
        }

        template = templates.get(ambiguity.type)

        if template:
            # Customize template
            text = template.text.format(**ambiguity.__dict__)
            return Question(
                text=text,
                options=template.options,
                default=template.default,
                field=ambiguity.field,
            )
        else:
            # Generic question
            return Question(
                text=f"{ambiguity.message}. {ambiguity.suggestion}",
                field=ambiguity.field,
            )
```

### 6.2 Iterative Refinement

Allow users to iteratively refine requirements:

```python
class IterativeRefiner:
    """Allow iterative refinement of workflow requirements.

    Users can:
    - Review extracted requirements
    - Edit specific fields
    - Add missing information
    - Remove incorrect information
    - Re-run extraction with new context
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator
        self._extractor = HybridRequirementExtractor(orchestrator)

    async def refine(
        self,
        requirements: WorkflowRequirements,
        user_feedback: str,
    ) -> WorkflowRequirements:
        """Refine requirements based on user feedback.

        Args:
            requirements: Current requirements
            user_feedback: User's feedback/edits

        Returns:
            Refined requirements
        """
        # Parse user feedback to determine what to change
        changes = self._parse_feedback(user_feedback)

        # Apply changes
        for change in changes:
            requirements = self._apply_change(requirements, change)

        # Re-validate
        validator = RequirementValidator()
        result = validator.validate(requirements)

        if not result.is_valid:
            # Show errors and allow further refinement
            # ...

        return requirements

    def _parse_feedback(self, feedback: str) -> List[RequirementChange]:
        """Parse user feedback into structured changes."""
        import re

        changes = []

        # Pattern: "change field X to Y"
        pattern = r"change\s+(\S+)\s+to\s+(.+)"
        matches = re.finditer(pattern, feedback, re.IGNORECASE)

        for match in matches:
            field = match.group(1)
            value = match.group(2)
            changes.append(
                RequirementChange(
                    type="edit",
                    field=field,
                    value=value,
                )
            )

        # Pattern: "add task: ..."
        pattern = r"add\s+task\s*:\s*(.+)"
        matches = re.finditer(pattern, feedback, re.IGNORECASE)

        for match in matches:
            description = match.group(1)
            changes.append(
                RequirementChange(
                    type="add",
                    field="functional.tasks",
                    value=description,
                )
            )

        return changes


@dataclass
class RequirementChange:
    """A change to apply to requirements.

    Attributes:
        type: Type of change (add, edit, remove)
        field: Field to change
        value: New value
    """

    type: str  # add, edit, remove
    field: str
    value: Any
```

---

## 7. Integration with Graph Generation

### 7.1 End-to-End Pipeline

Complete flow from natural language to workflow graph:

```python
class NaturalLanguageWorkflowPipeline:
    """End-to-end pipeline from natural language to workflow graph.

    Pipeline stages:
    1. Extract requirements from natural language
    2. Validate requirements
    3. Clarify ambiguities (if needed)
    4. Re-validate after clarification
    5. Generate workflow graph
    6. Compile graph for execution

    Example:
        pipeline = NaturalLanguageWorkflowPipeline(orchestrator)

        # Simple: Auto-resolve ambiguities
        graph = await pipeline.generate(
            "Analyze code, find bugs, fix them, run tests"
        )

        # Interactive: Ask user for clarification
        graph = await pipeline.generate_interactive(
            "Research AI and create report if quality is high"
        )
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self._orchestrator = orchestrator
        self._tool_registry = tool_registry or get_tool_registry()

        # Initialize components
        self._extractor = HybridRequirementExtractor(orchestrator)
        self._validator = RequirementValidator()
        self._clarifier = InteractiveClarifier(orchestrator)
        self._graph_generator = None  # Will be imported from phase 3.5

    async def generate(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        resolve_ambiguities: str = "assumptions",
    ) -> "CompiledGraph":
        """Generate workflow graph from natural language.

        Args:
            description: Natural language workflow description
            context: Optional context (project info, preferences)
            resolve_ambiguities: How to resolve ambiguities
                - "assumptions": Make reasonable assumptions
                - "defaults": Use template defaults
                - "fail": Fail if ambiguities found

        Returns:
            CompiledGraph ready for execution
        """
        logger.info(f"Generating workflow from: {description}")

        # Stage 1: Extract requirements
        logger.info("Stage 1: Extracting requirements...")
        requirements = await self._extractor.extract(description, context)

        # Stage 2: Validate requirements
        logger.info("Stage 2: Validating requirements...")
        validation_result = self._validator.validate(requirements)

        if not validation_result.is_valid:
            logger.error(f"Validation failed: {validation_result.errors}")
            raise ValueError(
                f"Requirements validation failed: {validation_result.errors}"
            )

        # Stage 3: Detect ambiguities
        logger.info("Stage 3: Detecting ambiguities...")
        detector = AmbiguityDetector()
        ambiguities = detector.detect(requirements)

        if ambiguities:
            logger.info(f"Found {len(ambiguities)} ambiguities")

            if resolve_ambiguities == "assumptions":
                resolver = AmbiguityResolver(self._orchestrator)
                requirements = await resolver.resolve(
                    requirements, ambiguities, strategy="assumptions"
                )
            elif resolve_ambiguities == "defaults":
                resolver = AmbiguityResolver(self._orchestrator)
                requirements = await resolver.resolve(
                    requirements, ambiguities, strategy="defaults"
                )
            elif resolve_ambiguities == "fail":
                raise ValueError(f"Ambiguities detected: {ambiguities}")

        # Stage 4: Re-validate after clarification
        logger.info("Stage 4: Re-validating...")
        validation_result = self._validator.validate(requirements)

        if not validation_result.is_valid:
            logger.error(f"Re-validation failed: {validation_result.errors}")
            raise ValueError(f"Requirements still invalid: {validation_result.errors}")

        # Stage 5: Generate workflow graph
        logger.info("Stage 5: Generating workflow graph...")
        workflow_def = self._generate_workflow_definition(requirements)

        # Stage 6: Compile graph
        logger.info("Stage 6: Compiling graph...")
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler(
            orchestrator=self._orchestrator,
            tool_registry=self._tool_registry,
        )

        compiled_graph = compiler.compile_definition(workflow_def)

        logger.info("✓ Workflow generated successfully")
        return compiled_graph

    async def generate_interactive(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> "CompiledGraph":
        """Generate workflow graph with interactive clarification.

        Like generate(), but asks user questions to resolve ambiguities
        instead of making assumptions.

        Args:
            description: Natural language workflow description
            context: Optional context

        Returns:
            CompiledGraph ready for execution
        """
        # Stage 1-2: Extract and validate
        requirements = await self._extractor.extract(description, context)
        validation_result = self._validator.validate(requirements)

        if not validation_result.is_valid:
            # Show errors and abort
            self._show_validation_errors(validation_result)
            raise ValueError("Requirements validation failed")

        # Stage 3: Interactive clarification
        detector = AmbiguityDetector()
        ambiguities = detector.detect(requirements)

        if ambiguities:
            requirements = await self._clarifier.clarify(requirements, ambiguities)

        # Stage 4-6: Same as non-interactive
        return await self._generate_from_requirements(requirements)

    def _generate_workflow_definition(
        self,
        requirements: WorkflowRequirements,
    ) -> "WorkflowDefinition":
        """Generate WorkflowDefinition from requirements.

        This is the bridge between requirements and graph generation.
        Maps requirement categories to workflow definition structure.

        Note: This is a placeholder. The actual implementation will be
        provided by the graph generation system (Phase 3.5).
        """
        from victor.workflows.definition import (
            WorkflowDefinition,
            AgentNode,
            ComputeNode,
            ConditionNode,
        )

        # Convert tasks to nodes
        nodes = []
        for task in requirements.functional.tasks:
            if task.task_type == "agent":
                node = AgentNode(
                    id=task.id,
                    name=task.description,
                    role=task.role or "executor",
                    goal=task.goal or task.description,
                    tool_budget=task.tool_budget or 15,
                )
            elif task.task_type == "compute":
                node = ComputeNode(
                    id=task.id,
                    name=task.description,
                    tools=task.tools,
                )
            # ... other node types

            nodes.append(node)

        # Build workflow definition
        return WorkflowDefinition(
            name="generated_workflow",
            description=requirements.description,
            nodes=nodes,
            # ... other fields
        )
```

### 7.2 CLI Command Integration

Add CLI command for natural language workflow generation:

```bash
# victor/ui/commands/workflow.py

@app.command()
def generate(
    description: str = Argument(..., help="Natural language workflow description"),
    interactive: bool = Option(False, "--interactive", "-i", help="Interactive mode"),
    output: Optional[Path] = Option(None, "--output", "-o", help="Output YAML file"),
):
    """Generate a workflow from natural language description.

    Examples:
        victor workflow generate "Analyze code, find bugs, fix them, run tests"
        victor workflow generate "Research AI and report if quality > 0.8" -i
        victor workflow generate "Deploy to staging and production" -o workflow.yaml
    """
    import asyncio

    from victor.core.bootstrap import bootstrap_orchestrator

    orchestrator = bootstrap_orchestrator()
    pipeline = NaturalLanguageWorkflowPipeline(orchestrator)

    if interactive:
        graph = asyncio.run(pipeline.generate_interactive(description))
    else:
        graph = asyncio.run(pipeline.generate(description))

    # Save to file if requested
    if output:
        # Export as YAML
        yaml_content = export_workflow_as_yaml(graph)
        output.write_text(yaml_content)
        typer.echo(f"✓ Workflow saved to {output}")
    else:
        typer.echo("✓ Workflow generated successfully")
        typer.echo(f"  Nodes: {len(graph.get_graph_schema()['nodes'])}")
```

---

## 8. Implementation Plan

### 8.1 Module Structure

```
victor/workflows/generation/
├── __init__.py
├── extractor.py          # LLM-based requirement extraction
├── requirements.py       # Requirement dataclasses and schemas
├── rule_extractor.py     # Rule-based fallback extraction
├── hybrid_extractor.py   # Hybrid LLM + rules
├── ambiguity.py          # Ambiguity detection
├── clarifier.py          # Interactive clarification
├── validator.py          # Requirement validation
└── pipeline.py           # End-to-end pipeline
```

### 8.2 Implementation Effort Estimates

| Module | LOC | Time (hours) | Dependencies |
|--------|-----|--------------|--------------|
| **requirements.py** | 500 | 8 | dataclasses, typing |
| **extractor.py** | 600 | 12 | orchestrator, providers |
| **rule_extractor.py** | 400 | 8 | regex, patterns |
| **hybrid_extractor.py** | 300 | 6 | extractor, rule_extractor |
| **ambiguity.py** | 500 | 10 | requirements |
| **clarifier.py** | 400 | 8 | ambiguity, rich/prompts |
| **validator.py** | 500 | 10 | requirements |
| **pipeline.py** | 400 | 8 | all above |
| **tests/** | 1200 | 16 | pytest, fixtures |
| **integration/** | 400 | 8 | end-to-end tests |
| **TOTAL** | **4800** | **94** | |

**Timeline:**
- Week 1: requirements.py, extractor.py (20 hours)
- Week 2: rule_extractor.py, hybrid_extractor.py, ambiguity.py (26 hours)
- Week 3: clarifier.py, validator.py (18 hours)
- Week 4: pipeline.py, integration, tests (30 hours)

### 8.3 Dependencies

**Internal:**
- `victor.agent.orchestrator` - LLM access
- `victor.tools.registry` - Tool validation
- `victor.workflows.definition` - WorkflowDefinition
- `victor.workflows.unified_compiler` - Graph compilation
- `victor.framework.graph` - CompiledGraph

**External:**
- `pydantic` - JSON schema validation (optional, can use dataclasses)
- `rich` - Interactive CLI prompts
- `pytest` - Testing
- `pytest-asyncio` - Async test support

### 8.4 Testing Strategy

**Unit Tests:**
- Test extraction with various input patterns
- Test ambiguity detection logic
- Test validation rules
- Mock LLM responses for reproducibility

**Integration Tests:**
- Test end-to-end pipeline with real LLM
- Test interactive clarification flow
- Test graph generation from requirements
- Test CLI command

**Test Fixtures:**
- Sample natural language descriptions
- Expected requirement outputs
- Mock orchestrator with canned responses

---

## 9. MVP Feature List

### 9.1 Minimum Viable Extraction System

**Core Features (Must Have):**

1. **Requirement Schema** ✅
   - [x] FunctionalRequirements (tasks, tools, I/O)
   - [x] StructuralRequirements (order, dependencies)
   - [x] QualityRequirements (constraints)
   - [x] ContextRequirements (vertical, environment)

2. **LLM-Based Extraction** ✅
   - [x] Structured output via JSON schema
   - [x] Prompt template with examples
   - [x] Extract all requirement categories
   - [x] Handle common workflow patterns

3. **Rule-Based Validation** ✅
   - [x] Validate task descriptions
   - [x] Validate tool names
   - [x] Detect circular dependencies
   - [x] Check constraint feasibility

4. **Ambiguity Detection** ✅
   - [x] Detect missing required fields
   - [x] Detect contradictory requirements
   - [x] Detect vague descriptions
   - [x] Prioritize by severity

5. **Requirement Validation** ✅
   - [x] Completeness checks
   - [x] Consistency checks
   - [x] Feasibility checks
   - [x] Validation feedback

6. **Basic Pipeline** ✅
   - [x] Extract → Validate → Generate
   - [x] Handle ambiguities with assumptions
   - [x] Generate WorkflowDefinition
   - [x] Compile to graph

**Nice-to-Have Features (Should Have):**

7. **Interactive Clarification** ⚠️
   - [ ] Ask users for missing info
   - [ ] Present assumptions for confirmation
   - [ ] Multiple choice questions
   - [ ] Iterative refinement

8. **Rule-Based Extraction Fallback** ⚠️
   - [ ] Keyword pattern matching
   - [ ] Task extraction from verbs
   - [ ] Branch detection from conditionals
   - [ ] Tool name extraction

9. **CLI Integration** ⚠️
   - [ ] `victor workflow generate` command
   - [ ] Interactive mode flag
   - [ ] YAML output option
   - [ ] Progress indicators

**Future Enhancements (Won't Have in MVP):**

10. **Advanced Features** ❌
    - [ ] Learn from user corrections
    - [ ] Template-based extraction
    - [ ] Multi-turn refinement
    - [ ] Export requirements as JSON

11. **Observability** ❌
    - [ ] Extraction metrics
    - [ ] Ambiguity statistics
    - [ ] Confidence tracking
    - [ ] Performance profiling

### 9.2 MVP Success Criteria

**Functional Requirements:**
- ✅ Extract requirements from 80% of common workflow patterns
- ✅ Generate valid WorkflowDefinition in 90% of cases
- ✅ Validation catches 95% of critical issues
- ✅ Interactive mode resolves 80% of ambiguities

**Quality Requirements:**
- ✅ Extraction time < 10 seconds (avg)
- ✅ Validation time < 2 seconds
- ✅ Confidence scores correlate with actual accuracy (>0.7)
- ✅ Unit test coverage > 80%

**Usability Requirements:**
- ✅ Clear error messages with suggestions
- ✅ Interactive mode is intuitive
- ✅ Support for 5+ verticals (coding, devops, research, etc.)
- ✅ CLI command works with no config

---

## 10. Architecture Diagrams

### 10.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT                                   │
│  Natural Language: "Analyze code, find bugs, fix them, test"    │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              REQUIREMENT EXTRACTION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  LLM-Based Extractor │◄────►│  Rule-Based Extractor │        │
│  │  (Primary)           │      │  (Fallback)          │        │
│  └──────────┬───────────┘      └──────────┬───────────┘        │
│             │                              │                     │
│             └──────────┬───────────────────┘                     │
│                        ▼                                         │
│            ┌─────────────────────┐                              │
│            │ Hybrid Extractor    │                              │
│            │ (Combine & Validate)│                              │
│            └──────────┬──────────┘                              │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│               WorkflowRequirements                                │
│  {                                                               │
│    functional: {tasks, tools, inputs, outputs},                 │
│    structural: {execution_order, dependencies, branches},        │
│    quality: {max_duration, max_cost, ...},                      │
│    context: {vertical, environment, ...}                        │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              AMBIGUITY DETECTION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ Functional Gaps │  │ Structural      │  │ Vague          │ │
│  │ (missing tasks, │  │ Conflicts       │  │ Descriptions   │ │
│  │  tools, roles)  │  │ (circular deps) │  │ (something,    │ │
│  └─────────────────┘  └─────────────────┘  │  stuff)         │ │
│                                            └────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           AMBIGUITY RESOLUTION STRATEGIES                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Interactive ───► Ask user questions (preferred)                 │
│  Assumptions ───► Make reasonable guesses                        │
│  Defaults ──────► Use template defaults                          │
│  Fail ───────────► Abort with error                              │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               REQUIREMENT VALIDATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Completeness │  │ Consistency  │  │ Feasibility  │        │
│  │ (all fields) │  │ (no conflicts)│ │ (can execute)│        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ ValidationResult                                       │    │
│  │ {is_valid, errors[], warnings[], recommendations[]}    │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              WORKFLOW GENERATION (Phase 3.5)                     │
│                                                                   │
│  Requirements ──► WorkflowDefinition ──► CompiledGraph           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Data Flow

```
Natural Language Input
        │
        ▼
┌───────────────────┐
│ Extract Requirements│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Detect Ambiguities│
└─────────┬─────────┘
          │
     ┌────┴────┐
     │         │
     ▼         ▼
┌─────────┐ ┌─────────────┐
│Ambiguities│ │No Ambiguities│
└────┬────┘ └──────┬──────┘
     │             │
     ▼             ▼
┌─────────────┐  │
│ Resolve     │  │
└──────┬──────┘  │
       │         │
       └────┬────┘
            ▼
     ┌──────────────┐
     │ Validate     │
     └──────┬───────┘
            │
       ┌────┴────┐
       │         │
       ▼         ▼
  ┌────────┐ ┌─────────┐
  │ Valid  │ │ Invalid │
  └───┬────┘ └────┬────┘
      │           │
      ▼           ▼
  ┌────────┐  ┌──────────┐
  │Generate│  │Show Errors│
  │ Graph  │  │& Abort   │
  └───┬────┘  └──────────┘
      │
      ▼
CompiledGraph
```

---

## 11. Example Usage

### 11.1 Simple Sequential Workflow

**Input:**
```python
description = """
Analyze this Python codebase for bugs using static analysis,
then fix the bugs you find, and finally run pytest to verify the fixes.
"""
```

**Extracted Requirements:**
```json
{
  "functional": {
    "tasks": [
      {
        "id": "analyze",
        "description": "Analyze Python codebase for bugs",
        "task_type": "agent",
        "role": "researcher",
        "goal": "Find bugs using static analysis tools",
        "tools": ["code_search", "linter", "ast_analyzer"]
      },
      {
        "id": "fix",
        "description": "Fix identified bugs",
        "task_type": "agent",
        "role": "executor",
        "goal": "Fix all bugs found in analysis",
        "dependencies": ["analyze"]
      },
      {
        "id": "test",
        "description": "Run pytest to verify fixes",
        "task_type": "compute",
        "tools": ["bash"],
        "dependencies": ["fix"]
      }
    ],
    "success_criteria": [
      "All tests pass",
      "No critical bugs remain",
      "Zero linter errors"
    ]
  },
  "structural": {
    "execution_order": "sequential",
    "dependencies": {
      "fix": ["analyze"],
      "test": ["fix"]
    }
  },
  "quality": {
    "max_cost_tier": "MEDIUM",
    "retry_policy": "retry"
  },
  "context": {
    "vertical": "coding",
    "subdomain": "bug_fix",
    "project_context": {
      "primary_language": "Python",
      "testing_framework": "pytest"
    }
  }
}
```

**Generated Workflow:**
```yaml
workflows:
  bug_fix_workflow:
    description: "Analyze, fix, and verify bugs"
    nodes:
      - id: analyze
        type: agent
        role: researcher
        goal: "Find bugs using static analysis"
        tool_budget: 20
        allowed_tools: [code_search, linter, ast_analyzer]
        next: [fix]

      - id: fix
        type: agent
        role: executor
        goal: "Fix all bugs found in analysis"
        tool_budget: 30
        next: [test]

      - id: test
        type: compute
        tools: [bash]
        input_mapping:
          command: "pytest"
        next: []
```

### 11.2 Conditional Workflow

**Input:**
```python
description = """
Research recent AI trends from at least 5 academic sources,
summarize the findings. If the quality score is above 0.8,
create a comprehensive report, otherwise just provide a brief summary.
"""
```

**Extracted Requirements:**
```json
{
  "functional": {
    "tasks": [
      {
        "id": "research",
        "description": "Research AI trends from 5+ sources",
        "task_type": "agent",
        "role": "researcher"
      },
      {
        "id": "summarize",
        "description": "Summarize findings",
        "task_type": "agent",
        "role": "writer",
        "dependencies": ["research"]
      },
      {
        "id": "create_report",
        "description": "Create comprehensive report",
        "task_type": "agent",
        "role": "writer",
        "dependencies": ["summarize"]
      },
      {
        "id": "brief_summary",
        "description": "Provide brief summary",
        "task_type": "agent",
        "role": "writer",
        "dependencies": ["summarize"]
      }
    ],
    "success_criteria": [
      "At least 5 sources cited",
      "Quality score checked"
    ]
  },
  "structural": {
    "execution_order": "conditional",
    "dependencies": {
      "summarize": ["research"],
      "create_report": ["summarize"],
      "brief_summary": ["summarize"]
    },
    "branches": [
      {
        "condition_id": "quality_check",
        "condition": "quality_score > 0.8",
        "true_branch": "create_report",
        "false_branch": "brief_summary",
        "condition_type": "quality_threshold"
      }
    ]
  },
  "context": {
    "vertical": "research",
    "subdomain": "literature_review"
  }
}
```

### 11.3 Parallel Workflow

**Input:**
```python
description = """
Deploy the application to staging and production environments
in parallel, then run smoke tests on both. If both deployments
succeed and tests pass, send notification to Slack.
"""
```

**Extracted Requirements:**
```json
{
  "functional": {
    "tasks": [
      {
        "id": "deploy_staging",
        "description": "Deploy to staging",
        "task_type": "compute",
        "tools": ["kubectl", "helm"]
      },
      {
        "id": "deploy_production",
        "description": "Deploy to production",
        "task_type": "compute",
        "tools": ["kubectl", "helm"]
      },
      {
        "id": "test_staging",
        "description": "Run smoke tests on staging",
        "task_type": "compute",
        "tools": ["pytest", "curl"],
        "dependencies": ["deploy_staging"]
      },
      {
        "id": "test_production",
        "description": "Run smoke tests on production",
        "task_type": "compute",
        "tools": ["pytest", "curl"],
        "dependencies": ["deploy_production"]
      },
      {
        "id": "notify",
        "description": "Send Slack notification",
        "task_type": "compute",
        "tools": ["slack_notifier"],
        "dependencies": ["test_staging", "test_production"]
      }
    ]
  },
  "structural": {
    "execution_order": "parallel",
    "branches": [
      {
        "condition_id": "deployment_success",
        "condition": "staging_success && production_success",
        "true_branch": "notify",
        "false_branch": "end",
        "condition_type": "data_check"
      }
    ]
  },
  "context": {
    "vertical": "devops",
    "subdomain": "deployment"
  }
}
```

---

## 12. Future Enhancements

### 12.1 Learning from Corrections

Track user corrections to improve extraction:

```python
class ExtractionLearner:
    """Learn from user corrections to improve extraction.

    Maintains a history of:
    - Initial extraction
    - User corrections
    - Final requirements
    - Extraction success metrics

    Uses this to:
    - Improve prompts
    - Add custom patterns
    - Train domain-specific models
    """

    def __init__(self):
        self._correction_history = []

    def record_correction(
        self,
        initial_requirements: WorkflowRequirements,
        user_corrections: List[RequirementChange],
        final_requirements: WorkflowRequirements,
    ) -> None:
        """Record a user correction for learning."""
        self._correction_history.append(
            {
                "initial": initial_requirements,
                "corrections": user_corrections,
                "final": final_requirements,
                "timestamp": time.time(),
            }
        )

    def suggest_improvements(self) -> List[str]:
        """Suggest prompt improvements based on corrections."""
        # Analyze correction history
        # Find common patterns
        # Suggest prompt changes
        pass
```

### 12.2 Template-Based Extraction

Use templates for common workflow patterns:

```python
class WorkflowTemplateExtractor:
    """Extract requirements using workflow templates.

    Predefined templates for common patterns:
    - Bug fix workflow
    - Deployment pipeline
    - Research workflow
    - Data processing pipeline
    """

    def __init__(self):
        self._templates = self._load_templates()

    def extract_from_template(
        self,
        description: str,
        template_name: str,
    ) -> WorkflowRequirements:
        """Extract using a specific template."""
        template = self._templates[template_name]

        # Fill template slots from description
        # Return structured requirements
        pass
```

---

## 13. Conclusion

This design document presents a comprehensive **requirement extraction and analysis system** for natural language workflow descriptions. The system:

### Key Strengths

1. **Comprehensive Extraction**: Captures functional, structural, quality, and context requirements
2. **Hybrid Approach**: Combines LLM intelligence with rule-based reliability
3. **Ambiguity Resolution**: Interactive clarification with multiple strategies
4. **Robust Validation**: Completeness, consistency, and feasibility checks
5. **User-Friendly**: Clear error messages and suggestions
6. **Extensible**: Easy to add new requirement types and validation rules

### Implementation Roadmap

- **Week 1**: Core schema and LLM extraction
- **Week 2**: Rule-based extraction and ambiguity detection
- **Week 3**: Validation and clarification
- **Week 4**: Pipeline integration and testing

### Success Metrics

- 80%+ extraction accuracy on common patterns
- < 10 second extraction time
- 95%+ validation coverage
- Positive user feedback on clarity

### Next Steps

1. Review and approve this design
2. Create detailed task breakdown
3. Begin implementation with requirements.py
4. Set up testing infrastructure
5. Iterate based on feedback

---

**Document Version:** 1.0
**Last Updated:** 2025-01-09
**Status:** Ready for Review
