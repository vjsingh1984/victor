# YAML Workflow Examples

**Version**: 1.0
**Last Updated**: 2026-04-07

---

## Example 1: Feature Development Workflow

Complete end-to-end feature development workflow with planning, implementation, testing, and review phases.

```yaml
workflows:
  feature_development:
    description: "End-to-end feature development workflow"
    metadata:
      version: "1.0"
      mode: build
      author: "development-team"
    nodes:
      # Phase 1: Requirements
      - id: gather_requirements
        type: agent
        role: business_analyst
        goal: "Gather and clarify requirements for the new feature"
        tool_budget: 15
        tools:
          - read
          - grep
          - code_search
        output: requirements
        next: [design_solution]
      
      # Phase 2: Design
      - id: design_solution
        type: agent
        role: architect
        goal: "Design technical solution for the feature"
        input: requirements
        tool_budget: 20
        tools:
          - read
          - write
          - grep
          - symbols
        output: design
        next: [review_design]
      
      # Design Review
      - id: review_design
        type: hitl
        hitl_type: approval
        prompt: "Review the technical design. Approve to proceed to implementation, or reject to revise."
        timeout: 600
        fallback: continue
        options:
          - approve: "Approve design - proceed to implementation"
          - reject: "Reject design - return to requirements gathering"
          - modify: "Request modifications - provide feedback"
        next: [implement]
      
      # Phase 3: Implementation
      - id: implement
        type: agent
        role: developer
        goal: "Implement the feature according to approved design"
        input: design
        tool_budget: 40
        tools:
          - read
          - write
          - edit
          - grep
          - code_search
          - shell
        output: implementation
        next: [write_tests]
      
      - id: write_tests
        type: agent
        role: test_engineer
        goal: "Write comprehensive tests for the implemented feature"
        input: implementation
        tool_budget: 25
        tools:
          - read
          - write
          - grep
        output: tests
        next: [run_tests]
      
      - id: run_tests
        type: agent
        role: qa_runner
        goal: "Run tests and check for failures"
        input: tests
        tool_budget: 15
        tools:
          - shell
          - grep
          - read
        error_strategy: continue
        output: test_results
        next: [check_test_results]
      
      - id: check_test_results
        type: condition
        condition: "all_tests_passed"
        input: test_results
        branches:
          true: code_review
          false: fix_tests
      
      - id: fix_tests
        type: agent
        role: developer
        goal: "Fix failing tests"
        input: test_results
        tool_budget: 20
        tools:
          - read
          - edit
          - write
          - shell
        next: [run_tests]
      
      # Phase 4: Code Review
      - id: code_review
        type: agent
        role: code_reviewer
        goal: "Review the implemented feature and tests"
        input: implementation
        tool_budget: 20
        tools:
          - read
          - grep
          - code_search
          - symbols
        output: review_feedback
        next: [review_decision]
      
      - id: review_decision
        type: condition
        condition: "review_score >= 0.8"
        input: review_feedback
        branches:
          true: approve_merge
          false: request_changes
      
      - id: request_changes
        type: agent
        role: developer
        goal: "Make changes based on review feedback"
        input: review_feedback
        tool_budget: 25
        tools:
          - read
          - edit
          - write
        next: [code_review]
      
      # Phase 5: Documentation
      - id: approve_merge
        type: agent
        role: technical_writer
        goal: "Write/update documentation for the feature"
        input: implementation
        tool_budget: 15
        tools:
          - read
          - write
          - grep
        output: documentation
        next: [final_check]
      
      - id: final_check
        type: agent
        role: qa_lead
        goal: "Final quality check before merge"
        input: documentation
        tool_budget: 10
        tools:
          - read
          - grep
          - shell
        output: ready_to_merge
```

---

## Example 2: Code Review Workflow

Automated code review workflow with parallel analysis and human approval.

```yaml
workflows:
  code_review:
    description: "Automated code review with parallel analysis"
    metadata:
      version: "1.0"
      mode: build
      tags: ["review", "quality"]
    nodes:
      # Initial analysis
      - id: setup_review
        type: agent
        role: review_coordinator
        goal: "Setup code review context and identify files to review"
        tool_budget: 10
        tools:
          - read
          - grep
          - glob
          - ls
        output: review_context
        next: [parallel_review]
      
      # Parallel analysis by different reviewers
      - id: parallel_review
        type: parallel
        nodes:
          - id: static_analysis
            type: agent
            role: static_analyzer
            goal: "Check code style, patterns, and static issues"
            tool_budget: 15
            tools:
              - read
              - grep
              - code_search
            output: static_findings
          
          - id: security_scan
            type: agent
            role: security_scanner
            goal: "Scan for security vulnerabilities and issues"
            tool_budget: 15
            tools:
              - read
              - grep
              - code_search
            output: security_findings
          
          - id: performance_check
            type: agent
            role: performance_analyzer
            goal: "Check for performance issues and anti-patterns"
            tool_budget: 12
            tools:
              - read
              - grep
              - symbols
            output: performance_findings
          
          - id: test_coverage
            type: agent
            role: coverage_analyzer
            goal: "Check test coverage and identify gaps"
            tool_budget: 10
            tools:
              - read
              - grep
              - shell
            output: coverage_findings
        join: merge_findings
        wait_for: all
        timeout: 300
        next: [generate_report]
      
      # Merge all findings
      - id: merge_findings
        type: transform
        transform: merge_results
        input: [static_findings, security_findings, performance_findings, coverage_findings]
        output: review_report
        next: [generate_report]
      
      - id: generate_report
        type: agent
        role: report_generator
        goal: "Generate comprehensive review report with all findings"
        input: review_report
        tool_budget: 15
        tools:
          - read
          - write
        output: final_report
        next: [review_decision]
      
      # Review decision
      - id: review_decision
        type: condition
        condition: "critical_issues == 0"
        input: final_report
        branches:
          true: human_review
          false: require_fixes
      
      - id: require_fixes
        type: agent
        role: issue_reporter
        goal: "Create issue report with required fixes"
        input: final_report
        tool_budget: 10
        tools:
          - write
        output: issue_list
        next: [end]
      
      # Human review checkpoint
      - id: human_review
        type: hitl
        hitl_type: approval
        prompt: "Review the automated analysis report. Approve to merge, or reject to require fixes."
        timeout: 600
        fallback: continue
        options:
          - approve: "Approve - no critical issues"
          - reject: "Reject - requires fixes"
          - modify: "Request changes - provide feedback"
        next: [merge_decision]
      
      - id: merge_decision
        type: condition
        condition: "approved"
        branches:
          true: merge
          false: request_changes
      
      - id: request_changes
        type: agent
        role: coordinator
        goal: "Coordinate requested changes with development team"
        tool_budget: 5
        next: [end]
      
      - id: merge
        type: agent
        role: integrator
        goal: "Merge approved changes"
        tool_budget: 10
        tools:
          - shell
          - write
        output: merge_result
```

---

## Example 3: Bug Investigation Workflow

Systematic bug investigation workflow for debugging complex issues.

```yaml
workflows:
  bug_investigation:
    description: "Systematic bug investigation and resolution workflow"
    metadata:
      version: "1.0"
      mode: explore
      tags: ["debugging", "investigation"]
    nodes:
      # Phase 1: Understand the Bug
      - id: understand_bug
        type: agent
        role: bug_analyst
        goal: "Understand the bug report and reproduction steps"
        tool_budget: 10
        tools:
          - read
          - grep
        output: bug_context
        next: [reproduce_bug]
      
      # Phase 2: Reproduce Bug
      - id: reproduce_bug
        type: agent
        role: qa_tester
        goal: "Reproduce the bug to confirm it exists"
        input: bug_context
        tool_budget: 15
        tools:
          - read
          - shell
          - grep
        error_strategy: continue
        output: reproduction_result
        next: [locate_source]
      
      # Phase 3: Locate Source
      - id: locate_source
        type: agent
        role: code_detective
        goal: "Locate the source code responsible for the bug"
        input: reproduction_result
        tool_budget: 25
        tools:
          - read
          - grep
          - code_search
          - symbols
          - git_log
        output: source_location
        next: [analyze_root_cause]
      
      # Phase 4: Root Cause Analysis
      - id: analyze_root_cause
        type: agent
        role: senior_developer
        goal: "Analyze the code to understand root cause"
        input: source_location
        tool_budget: 20
        tools:
          - read
          - grep
          - symbols
          - code_search
        output: root_cause_analysis
        next: [propose_fix]
      
      # Phase 5: Propose Fix
      - id: propose_fix
        type: agent
        role: solution_architect
        goal: "Propose a fix for the bug"
        input: root_cause_analysis
        tool_budget: 15
        tools:
          - read
          - write
        output: fix_proposal
        next: [review_fix]
      
      # Fix Review
      - id: review_fix
        type: hitl
        hitl_type: approval
        prompt: "Review the proposed fix. Approve to implement, or reject to revise."
        timeout: 600
        fallback: continue
        options:
          - approve: "Approve - implement the fix"
          - reject: "Reject - needs revision"
          - modify: "Request changes - provide feedback"
        next: [implement_fix]
      
      # Phase 6: Implement Fix
      - id: implement_fix
        type: agent
        role: developer
        goal: "Implement the approved fix"
        input: fix_proposal
        tool_budget: 30
        tools:
          - read
          - edit
          - write
          - shell
        output: fix_implementation
        next: [verify_fix]
      
      # Phase 7: Verify Fix
      - id: verify_fix
        type: agent
        role: qa_tester
        goal: "Verify the fix resolves the bug without regressions"
        input: fix_implementation
        tool_budget: 20
        tools:
          - shell
          - grep
          - read
        output: verification_result
        next: [check_verification]
      
      - id: check_verification
        type: condition
        condition: "bug_fixed and no_regressions"
        input: verification_result
        branches:
          true: document_resolution
          false: iterate
      
      # Iterate if needed
      - id: iterate
        type: agent
        role: bug_fixer
        goal: "Refine the fix based on verification feedback"
        input: verification_result
        tool_budget: 20
        tools:
          - read
          - edit
          - write
        next: [verify_fix]
      
      # Document resolution
      - id: document_resolution
        type: agent
        role: technical_writer
        goal: "Document the bug, root cause, and resolution"
        input: verification_result
        tool_budget: 15
        tools:
          - read
          - write
        output: bug_report
        next: [end]
```

---

## Example 4: Multi-Agent Team Workflow

Multi-agent workflow with specialized roles and parallel execution.

```yaml
workflows:
  multi_agent_analysis:
    description: "Multi-agent team workflow for comprehensive code analysis"
    metadata:
      version: "1.0"
      mode: explore
      team: analysis_team
    nodes:
      # Team formation using TeamSpecRegistry
      - id: form_team
        type: agent
        role: team_coordinator
        goal: "Form specialized analysis team"
        tool_budget: 5
        output: team_config
        next: [parallel_analysis]
      
      # Parallel analysis by team members
      - id: parallel_analysis
        type: parallel
        nodes:
          - id: security_audit
            type: agent
            role: security_auditor
            vertical: security
            goal: "Perform security audit of the codebase"
            tool_budget: 25
            tools:
              - read
              - grep
              - code_search
            output: security_report
          
          - id: performance_audit
            type: agent
            role: performance_auditor
            vertical: coding
            goal: "Analyze performance characteristics"
            tool_budget: 20
            tools:
              - read
              - grep
              - symbols
            output: performance_report
          
          - id: quality_audit
            type: agent
            role: quality_auditor
            vertical: coding
            goal: "Check code quality and adherence to standards"
            tool_budget: 20
            tools:
              - read
              - grep
              - code_search
            output: quality_report
          
          - id: documentation_audit
            type: agent
            role: documentation_reviewer
            vertical: coding
            goal: "Review documentation completeness"
            tool_budget: 15
            tools:
              - read
              - grep
            output: documentation_report
        join: synthesize_results
        wait_for: all
        timeout: 600
        next: [synthesize_results]
      
      # Synthesize all reports
      - id: synthesize_results
        type: agent
        role: lead_analyst
        goal: "Synthesize findings from all team members into comprehensive report"
        tool_budget: 25
        tools:
          - read
          - write
        input: [security_report, performance_report, quality_report, documentation_report]
        output: comprehensive_report
        next: [team_review]
      
      # Team review
      - id: team_review
        type: agent
        role: team_lead
        goal: "Review comprehensive analysis report as a team"
        input: comprehensive_report
        tool_budget: 15
        tools:
          - read
          - write
        output: team_feedback
        next: [final_approval]
      
      # Final approval
      - id: final_approval
        type: hitl
        hitl_type: approval
        prompt: "Review the comprehensive analysis report. Approve to finalize, or request revisions."
        timeout: 900
        fallback: continue
        options:
          - approve: "Approve - report is complete"
          - revise: "Request revisions - provide feedback"
        next: [publish_report]
      
      # Publish report
      - id: publish_report
        type: agent
        role: report_publisher
        goal: "Format and publish the final analysis report"
        input: comprehensive_report
        tool_budget: 10
        tools:
          - read
          - write
        output: published_report
        next: [end]
```

---

## Example 5: Documentation Generation Workflow

Automated documentation generation workflow.

```yaml
workflows:
  documentation_generation:
    description: "Generate comprehensive documentation from code"
    metadata:
      version: "1.0"
      mode: explore
      tags: ["documentation", "automation"]
    nodes:
      # Analyze codebase structure
      - id: analyze_structure
        type: agent
        role: code_analyzer
        goal: "Analyze the codebase structure and identify components to document"
        tool_budget: 20
        tools:
          - read
          - grep
          - glob
          - ls
        output: structure_analysis
        next: [extract_docstrings]
      
      # Extract docstrings
      - id: extract_docstrings
        type: agent
        role: documentation_extractor
        goal: "Extract all docstrings and type hints from the code"
        tool_budget: 25
        tools:
          - read
          - grep
          - ast
        output: extracted_docs
        next: [generate_api_docs]
      
      # Generate API documentation
      - id: generate_api_docs
        type: agent
        role: api_documentation_writer
        goal: "Generate API documentation from extracted information"
        input: extracted_docs
        tool_budget: 30
        tools:
          - read
          - write
        output: api_docs
        next: [generate_user_guides]
      
      # Generate user guides
      - id: generate_user_guides
        type: agent
        role: user_guide_writer
        goal: "Generate user guides and examples"
        input: structure_analysis
        tool_budget: 25
        tools:
          - read
          - write
        output: user_guides
        next: [generate_examples]
      
      # Generate examples
      - id: generate_examples
        type: agent
        role: example_generator
        goal: "Generate usage examples for major components"
        input: extracted_docs
        tool_budget: 20
        tools:
          - read
          - write
        output: examples
        next: [review_documentation]
      
      # Review documentation
      - id: review_documentation
        type: agent
        role: documentation_reviewer
        goal: "Review generated documentation for completeness and accuracy"
        input: [api_docs, user_guides, examples]
        tool_budget: 15
        tools:
          - read
          - grep
        output: review_feedback
        next: [document_approval]
      
      # Documentation approval
      - id: document_approval
        type: hitl
        hitl_type: approval
        prompt: "Review the generated documentation. Approve to publish, or request revisions."
        timeout: 600
        fallback: continue
        options:
          - approve: "Approve - documentation is ready"
          - revise: "Request revisions - provide feedback"
        next: [publish_documentation]
      
      # Publish documentation
      - id: publish_documentation
        type: agent
        role: documentation_publisher
        goal: "Format and publish the documentation"
        input: [api_docs, user_guides, examples]
        tool_budget: 15
        tools:
          - read
          - write
          - shell
        output: published_docs
        next: [end]
```

---

## Example 6: CI/CD Pipeline Workflow

Continuous integration and deployment workflow.

```yaml
workflows:
  cicd_pipeline:
    description: "Automated CI/CD pipeline for testing and deployment"
    metadata:
      version: "1.0"
      mode: build
      tags: ["cicd", "automation"]
    nodes:
      # Pre-flight checks
      - id: preflight
        type: agent
        role: ci_validator
        goal: "Run pre-flight checks (linting, formatting, type checking)"
        tool_budget: 10
        tools:
          - shell
        error_strategy: stop
        output: preflight_results
        next: [run_tests]
      
      # Run test suite
      - id: run_tests
        type: agent
        role: test_runner
        goal: "Run full test suite"
        tool_budget: 30
        tools:
          - shell
          - read
        error_strategy: continue
        output: test_results
        next: [check_tests]
      
      # Check test results
      - id: check_tests
        type: condition
        condition: "all_tests_passed"
        input: test_results
        branches:
          true: build
          false: report_failure
      
      - id: report_failure
        type: agent
        role: failure_reporter
        goal: "Generate failure report and notify team"
        input: test_results
        tool_budget: 10
        tools:
          - read
          - write
          - shell
        next: [end]
      
      # Build artifacts
      - id: build
        type: agent
        role: build_engineer
        goal: "Build deployment artifacts"
        tool_budget: 20
        tools:
          - shell
          - read
        output: build_artifacts
        next: [security_scan]
      
      # Security scan
      - id: security_scan
        type: agent
        role: security_scanner
        goal: "Run security scan on build artifacts"
        input: build_artifacts
        tool_budget: 15
        tools:
          - shell
          - read
        error_strategy: continue
        output: scan_results
        next: [check_security]
      
      # Check security
      - id: check_security
        type: condition
        condition: "no_critical_vulnerabilities"
        input: scan_results
        branches:
          true: deploy
          false: report_security_issues
      
      - id: report_security_issues
        type: agent
        role: security_reporter
        goal: "Generate security report and notify team"
        input: scan_results
        tool_budget: 10
        tools:
          - read
          - write
        next: [end]
      
      # Deploy
      - id: deploy
        type: agent
        role: deployment_agent
        goal: "Deploy artifacts to production"
        input: build_artifacts
        tool_budget: 15
        tools:
          - shell
          - read
        error_strategy: stop
        output: deployment_result
        next: [verify_deployment]
      
      # Verify deployment
      - id: verify_deployment
        type: agent
        role: deployment_verifier
        goal: "Verify deployment was successful"
        input: deployment_result
        tool_budget: 10
        tools:
          - shell
          - read
        output: verification_results
        next: [check_deployment]
      
      - id: check_deployment
        type: condition
        condition: "deployment_successful"
        input: verification_results
        branches:
          true: success
          false: rollback
      
      - id: rollback
        type: agent
        role: deployment_agent
        goal: "Rollback deployment due to verification failure"
        tool_budget: 10
        tools:
          - shell
        next: [end]
      
      - id: success
        type: agent
        role: notifier
        goal: "Notify team of successful deployment"
        tool_budget: 5
        tools:
          - read
          - write
        next: [end]
```

---

## Usage

### Running Workflows

```python
from victor.framework import Agent

# Create agent with workflow
agent = await Agent.create()

# Run workflow
result = await agent.run_workflow(
    workflow_path="path/to/workflow.yaml",
    workflow_name="feature_development",
    initial_state={
        "feature_name": "user authentication",
        "files": ["auth.py", "user.py"]
    }
)
```

### Loading Custom Workflows

```python
from victor.workflows.yaml_loader import YAMLWorkflowLoader

# Load workflow
loader = YAMLWorkflowLoader()
workflow = loader.load("path/to/workflow.yaml")

# Access specific workflow
feature_workflow = workflow["feature_development"]

# Use with agent
agent = await Agent.create()
result = await agent.run_workflow(
    workflow_definition=feature_workflow,
    initial_state={"project_root": "/path/to/project"}
)
```

---

## Next Steps

1. ✅ Review these examples
2. 📖 Study the syntax guide
3. 💡 Adapt patterns for your use cases
4. 🚀 Create your own workflows
5. 🧪 Test workflows in safe environment

---

## Contributing Examples

Have a great workflow example? Contribute it!

1. Add to this file
2. Follow the pattern above
3. Include description and metadata
4. Test your workflow before submitting

---

**Need More Examples?**
- Check `examples/workflows/` directory
- See mode-based workflows for common patterns
- Review TeamSpecRegistry for team configuration examples
