# Team Nodes Guide - Part 4

**Part 4 of 4:** Complete Examples and Additional Resources

---

## Navigation

- [Part 1: Overview & Formation](part-1-overview-formation.md)
- [Part 2: Recursion & Configuration](part-2-recursion-configuration.md)
- [Part 3: Best Practices & Error Handling](part-3-best-practices-errors.md)
- **[Part 4: Complete Examples](#)** (Current)
- [**Complete Guide**](../team_nodes.md)

---

## Complete Examples

### Example: Multi-Stage Code Review Workflow

```yaml
workflows:
  comprehensive_review:
    description: "Multi-stage code review with specialized teams"

    metadata:
      version: "0.5.0"
      vertical: "coding"

    execution:
      max_recursion_depth: 3
      max_timeout_seconds: 1200

    nodes:
      # Stage 1: Automated analysis
      - id: automated_checks
        type: compute
        name: "Automated Checks"
        tools: [shell]
        inputs:
          commands:
            - "ruff check ."
            - "mypy ."
            - "pytest tests/"
        output: auto_results
        next: [initial_review]

      # Stage 2: Initial human review
      - id: initial_review
        type: team
        name: "Initial Review Team"
        goal: |
          Review the code changes:
          {{diff_summary}}

          Automated results: {{auto_results}}

          Provide initial feedback on:
          1. Code quality
          2. Potential bugs
          3. Security issues
          4. Performance concerns
        team_formation: parallel
        timeout_seconds: 300
        total_tool_budget: 75
        output_key: review_feedback
        members:
          - id: security_reviewer
            role: reviewer
            goal: "Check for security vulnerabilities and issues"
            tool_budget: 25
            tools: [read, grep]
            expertise: [security, authentication, authorization]
            backstory: |
              Security engineer with OWASP certification.
              Expert in identifying vulnerabilities and
              secure coding practices.

          - id: quality_reviewer
            role: reviewer
            goal: "Check code quality and maintainability"
            tool_budget: 25
            tools: [read, grep]
            expertise: [code-quality, design-patterns]
            backstory: |
              Senior developer focused on code quality,
              maintainability, and best practices.

          - id: logic_reviewer
            role: reviewer
            goal: "Check logic correctness and potential bugs"
            tool_budget: 25
            tools: [read, grep]
            expertise: [logic, algorithms, debugging]
            backstory: |
              Experienced developer with strong focus on
              logic correctness and edge cases.
        next: [decide_changes]

      # Stage 3: Decide if changes needed
      - id: decide_changes
        type: condition
        name: "Decision Point"
        condition: "needs_changes"
        branches:
          "true": implementation_team
          "false": final_approval

      # Stage 4a: Implementation team (if changes needed)
      - id: implementation_team
        type: team
        name: "Implementation Team"
        goal: |
          Implement requested changes based on feedback:
          {{review_feedback}}
        team_formation: pipeline
        timeout_seconds: 600
        total_tool_budget: 125
        output_key: changes_result
        members:
          - id: planner
            role: planner
            goal: |
              Plan the implementation approach:
              1. Review all feedback
              2. Prioritize changes
              3. Create implementation plan
              4. Identify risks
            tool_budget: 25
            tools: [read, grep, overview]
            backstory: |
              Technical planner experienced in breaking
              down feedback into actionable implementation plans.

          - id: developer
            role: executor
            goal: |
              Implement the planned changes:
              1. Follow implementation plan
              2. Write clean, tested code
              3. Update documentation
              4. Run tests
            tool_budget: 75
            tools: [read, write, grep, shell]
            backstory: |
              Full-stack developer focused on quality
              implementation and testing.

          - id: verifier
            role: reviewer
            goal: |
              Verify the implementation:
              1. Check all feedback addressed
              2. Run test suite
              3. Validate changes
              4. Report any issues
            tool_budget: 25
            tools: [read, grep, shell]
            backstory: |
              QA specialist with attention to detail
              and passion for quality.
        next: [final_review]

      # Stage 4b: Final approval (if no changes)
      - id: final_approval
        type: agent
        role: planner
        goal: "Approve the changes - no issues found"
        tool_budget: 10
        next: [complete]

      # Stage 5: Final review after changes
      - id: final_review
        type: team
        name: "Final Review Team"
        goal: |
          Final review of implemented changes:
          {{changes_result}}

          Ensure all feedback has been addressed
          and no regressions introduced.
        team_formation: consensus
        timeout_seconds: 300
        total_tool_budget: 75
        output_key: final_result
        members:
          - id: reviewer_1
            role: reviewer
            goal: "Verify changes address all feedback"
            tool_budget: 25
            tools: [read, grep]

          - id: reviewer_2
            role: reviewer
            goal: "Check for regressions and new issues"
            tool_budget: 25
            tools: [read, grep]

          - id: reviewer_3
            role: reviewer
            goal: "Final approval check"
            tool_budget: 25
            tools: [read, grep]
        next: [complete]

      # Stage 6: Complete
      - id: complete
        type: transform
        name: "Mark Complete"
        transform: "status = approved"
        next: []
```

### Example: Research and Implementation Workflow

```yaml
workflows:
  research_and_implement:
    description: "Research new technology and implement POC"

    metadata:
      version: "0.5.0"

    nodes:
      # Phase 1: Research team (parallel)
      - id: research_phase
        type: team
        name: "Research Team"
        goal: |
          Research {{technology}} thoroughly:

          Research Areas:
          1. Core concepts and architecture
          2. Best practices and patterns
          3. Integration options
          4. Potential pitfalls and limitations
          5. Community and ecosystem

          Provide comprehensive findings with examples.
        team_formation: parallel
        timeout_seconds: 900  # 15 minutes
        total_tool_budget: 150
        output_key: research_findings
        members:
          - id: concept_researcher
            role: researcher
            goal: "Research core concepts, architecture, and design principles"
            tool_budget: 50
            tools: [web_search, read]
            backstory: |
              Technology researcher with 10 years experience
              analyzing software architectures and patterns.
            expertise: [research, architecture-analysis, design-patterns]

          - id: best_practices_researcher
            role: researcher
            goal: "Find best practices, patterns, and anti-patterns"
            tool_budget: 50
            tools: [web_search, read]
            backstory: |
              Researcher focused on identifying proven practices
              and common pitfalls in technology adoption.
            expertise: [best-practices, patterns, anti-patterns]

          - id: integration_researcher
            role: researcher
            goal: "Research integration approaches and ecosystem"
            tool_budget: 50
            tools: [web_search, read]
            backstory: |
              Integration specialist with experience in
              system integration and API design.
            expertise: [integration, system-design, apis]
        next: [synthesis]

      # Phase 2: Synthesize research
      - id: synthesis
        type: agent
        role: planner
        goal: |
          Synthesize research findings into comprehensive guide:

          Research Findings: {{research_findings}}

          Create guide covering:
          1. Architecture overview with diagrams
          2. Implementation recommendations
          3. Code examples and patterns
          4. Integration plan with options
          5. Risk assessment and mitigation

          Output should be actionable and clear.
        tool_budget: 30
        tools: [write]
        output: guide
        next: [implementation_team]

      # Phase 3: Implementation team (sequential)
      - id: implementation_team
        type: team
        name: "Implementation Team"
        goal: |
          Implement POC based on research guide:
          {{guide}}

          Create working proof-of-concept demonstrating
          key concepts and integration points.
        team_formation: sequential
        timeout_seconds: 1200  # 20 minutes
        total_tool_budget: 200
        output_key: implementation_result
        members:
          - id: architect
            role: planner
            goal: |
              Design POC architecture:
              1. Define component structure
              2. Plan data flow and interfaces
              3. Identify integration points
              4. Create architecture documentation
            tool_budget: 40
            tools: [write]
            backstory: |
              Software architect specializing in POC design
              and rapid prototyping.
            expertise: [architecture, design, prototyping]

          - id: developer
            role: executor
            goal: |
              Implement core functionality:
              1. Set up project structure
              2. Implement key components
              3. Add error handling
              4. Create sample usage
            tool_budget: 120
            tools: [read, write, grep, shell]
            backstory: |
              Full-stack developer experienced in rapid
              prototyping and POC development.
            expertise: [implementation, debugging, prototyping]

          - id: documenter
            role: writer
            goal: |
              Create documentation:
              1. API documentation
              2. Usage examples
              3. Setup instructions
              4. Integration guide
            tool_budget: 40
            tools: [read, write]
            backstory: |
              Technical writer specializing in developer
              documentation and guides.
            expertise: [documentation, technical-writing]
        next: [testing_team]

      # Phase 4: Testing team (parallel)
      - id: testing_team
        type: team
        name: "Testing Team"
        goal: |
          Test POC implementation comprehensively:
          {{implementation_result}}

          Validate functionality, integration, and robustness.
        team_formation: parallel
        timeout_seconds: 600
        total_tool_budget: 100
        output_key: test_results
        members:
          - id: functional_tester
            role: reviewer
            goal: "Test functional requirements and core features"
            tool_budget: 50
            tools: [read, grep, shell]
            backstory: |
              QA engineer focused on functional testing
              and validation.

          - id: integration_tester
            role: reviewer
            goal: "Test integration points and interfaces"
            tool_budget: 50
            tools: [read, grep, shell]
            backstory: |
              Integration testing specialist with experience
              in API and component integration testing.
        next: [finalize]

      # Phase 5: Finalize
      - id: finalize
        type: agent
        role: writer
        goal: |
          Prepare final report:
          Research: {{research_findings}}
          Guide: {{guide}}
          Implementation: {{implementation_result}}
          Testing: {{test_results}}

          Create comprehensive report including:
          1. Executive summary
          2. Technical deep-dive
          3. Implementation details
          4. Test results and findings
          5. Recommendations

          Save report to file.
        tool_budget: 20
        tools: [write]
        next: []
```

## Additional Resources

### Related Documentation

- [Multi-Agent Teams Guide](../guides/MULTI_AGENT_TEAMS.md) - Team coordination patterns and standalone teams
- [Workflow DSL Guide](../guides/workflow-development/dsl.md) - Python workflow API and StateGraph
- [Workflow Examples](../guides/workflow-development/examples.md) - More workflow examples
- [Workflow User Guide](../user-guide/workflows.md) - General workflow documentation

### API Reference

#### TeamNodeWorkflow

```python
@dataclass
class TeamNodeWorkflow(WorkflowNode):
    """Node that spawns an ad-hoc multi-agent team.

    Attributes:
        id: Unique node identifier
        name: Human-readable name
        goal: Overall goal for the team
        team_formation: How to organize the team (sequential, parallel, etc.)
        members: List of team member configurations
        timeout_seconds: Maximum execution time (None = no limit)
        max_iterations: Maximum team iterations (default: 50)
        total_tool_budget: Total tool calls budget (default: 100)
        merge_strategy: How to merge team state (default: "dict")
        merge_mode: Conflict resolution mode (default: "team_wins")
        output_key: Context key for result (default: "team_result")
        continue_on_error: Continue workflow on failure (default: true)
    """
```

#### TeamFormation

```python
class TeamFormation(str, Enum):
    """Team organization patterns."""
    SEQUENTIAL = "sequential"      # Chain execution with context passing
    PARALLEL = "parallel"          # Simultaneous execution with aggregation
    PIPELINE = "pipeline"          # Output handoff between stages
    HIERARCHICAL = "hierarchical"  # Manager-worker coordination
    CONSENSUS = "consensus"        # Agreement-based decision making
```

#### RecursionContext

```python
@dataclass
class RecursionContext:
    """Tracks recursion depth for nested execution.

    Thread-safe for concurrent access.

    Attributes:
        current_depth: Current nesting level (0 = top-level)
        max_depth: Maximum allowed nesting level (default: 3)
        execution_stack: Stack trace of execution entries

    Methods:
        enter(type, id): Enter a nested level (raises if max exceeded)
        exit(): Exit a nested level
        can_nest(levels): Check if nesting is possible
        get_depth_info(): Get current depth information
    """
```

### Implementation Files

- `victor/workflows/recursion.py` - Recursion tracking implementation
- `victor/workflows/team_node_runner.py` - Team node execution
- `victor/framework/workflows/nodes.py` - TeamNode class definition
- `victor/workflows/yaml_loader.py` - YAML parsing (lines 1050-1106)
- `victor/teams/` - Team coordination infrastructure

### See Also

- [Team Configuration System](../teams/team_templates.md) - Team specification architecture
- [State Merging Guide](../teams/collaboration.md) - State merge strategies
- [Error Handling Guide](../guides/RESILIENCE.md) - Error handling patterns

---

*Last Updated: 2026-01-20*

---

**Last Updated:** February 01, 2026
**Reading Time:** 9 minutes
