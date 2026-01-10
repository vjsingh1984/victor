# Human-in-the-Loop (HITL) Workflows Guide

This guide covers Victor's HITL infrastructure for human approval and oversight in agent workflows.

## Overview

HITL workflows enable:

- **Approval Gates**: Require human approval before critical actions
- **Review Checkpoints**: Insert review points in multi-step workflows
- **Override Capability**: Allow humans to modify agent decisions
- **Audit Trail**: Track all human interventions

## Quick Start

```python
from victor.framework.workflow_engine import WorkflowEngine
from victor.workflows.hitl import CLIHITLHandler

# Create engine with HITL handler
engine = WorkflowEngine(
    hitl_handler=CLIHITLHandler()
)

# Execute workflow - will pause at HITL nodes
result = await engine.execute_yaml(
    "workflow_with_approval.yaml",
    initial_state={"task": "Deploy to production"}
)
```

## HITL Node Types

### Approval Node

Requires explicit approval to continue:

```yaml
nodes:
  - id: approve_deploy
    type: hitl
    hitl_type: approval
    prompt: "Approve deployment to production?"
    context_keys:
      - changes_summary
      - test_results
    timeout: 300  # 5 minute timeout
    on_timeout: reject  # or "approve", "skip"
    next:
      approved: deploy
      rejected: rollback
```

### Review Node

Presents information for review, continues automatically:

```yaml
nodes:
  - id: review_changes
    type: hitl
    hitl_type: review
    prompt: "Review the following changes"
    context_keys:
      - diff
      - affected_files
    next: [continue_workflow]
```

### Input Node

Requests specific input from human:

```yaml
nodes:
  - id: get_config
    type: hitl
    hitl_type: input
    prompt: "Enter deployment configuration"
    input_schema:
      type: object
      properties:
        environment:
          type: string
          enum: [staging, production]
        replicas:
          type: integer
          minimum: 1
          maximum: 10
    output: deployment_config
    next: [deploy]
```

### Override Node

Allows human to modify agent decision:

```yaml
nodes:
  - id: override_decision
    type: hitl
    hitl_type: override
    prompt: "Agent recommends: {recommendation}. Override?"
    context_keys:
      - recommendation
      - reasoning
    allow_edit: true
    next: [execute_decision]
```

## HITL Handlers

### CLI Handler

Interactive terminal prompts:

```python
from victor.workflows.hitl import CLIHITLHandler

handler = CLIHITLHandler(
    timeout=300,
    default_action="reject",
)

engine = WorkflowEngine(hitl_handler=handler)
```

### TUI Handler

Rich terminal interface:

```python
from victor.workflows.hitl import TUIHITLHandler

handler = TUIHITLHandler(
    show_context=True,
    syntax_highlight=True,
)
```

### API Handler

HTTP-based approvals (for web apps):

```python
from victor.workflows.hitl import APIHITLHandler

handler = APIHITLHandler(
    webhook_url="https://my-app.com/approvals",
    poll_interval=5.0,
)
```

### Slack Handler

Slack-based approvals:

```python
from victor.workflows.hitl import SlackHITLHandler

handler = SlackHITLHandler(
    bot_token="xoxb-...",
    channel="#approvals",
    mention_users=["U123ABC"],
)
```

### Custom Handler

Implement your own:

```python
from victor.workflows.hitl import HITLHandler, HITLRequest, HITLResponse

class MyHandler(HITLHandler):
    async def request_approval(
        self,
        request: HITLRequest
    ) -> HITLResponse:
        # Your logic here
        user_response = await my_approval_system.request(
            prompt=request.prompt,
            context=request.context,
        )

        return HITLResponse(
            approved=user_response.approved,
            feedback=user_response.comments,
            modified_context=user_response.edits,
        )
```

## HITL Coordinator

Programmatic HITL control:

```python
from victor.framework.coordinators import HITLCoordinator

coordinator = HITLCoordinator(
    handler=my_handler,
    timeout=300,
    auto_approve_on_timeout=False,
)

# Request approval
response = await coordinator.request_approval(
    node_id="approve_deploy",
    prompt="Deploy to production?",
    context={
        "changes": change_summary,
        "tests_passed": True,
    }
)

if response.approved:
    await deploy()
else:
    logger.info(f"Rejected: {response.feedback}")
```

## Workflow Examples

### Deployment Pipeline with Approval

```yaml
name: deploy_pipeline
description: Deployment with human approval

nodes:
  - id: run_tests
    type: agent
    role: tester
    goal: Run all tests
    output: test_results
    next: [check_tests]

  - id: check_tests
    type: condition
    condition: tests_passed
    branches:
      "true": build
      "false": notify_failure

  - id: build
    type: agent
    role: builder
    goal: Build deployment artifacts
    output: build_info
    next: [approve_deploy]

  - id: approve_deploy
    type: hitl
    hitl_type: approval
    prompt: |
      Ready to deploy:
      - Tests: {test_results.summary}
      - Build: {build_info.version}

      Approve deployment?
    timeout: 600
    on_timeout: reject
    next:
      approved: deploy
      rejected: cancel

  - id: deploy
    type: agent
    role: deployer
    goal: Deploy to production
    next: [verify]

  - id: verify
    type: agent
    role: verifier
    goal: Verify deployment health
    next: [complete]
```

### Code Review Workflow

```yaml
name: code_review
description: Code review with human oversight

nodes:
  - id: analyze_code
    type: agent
    role: analyzer
    goal: Analyze code changes for issues
    output: analysis
    next: [ai_review]

  - id: ai_review
    type: agent
    role: reviewer
    goal: Provide review comments
    output: review_comments
    next: [human_review]

  - id: human_review
    type: hitl
    hitl_type: review
    prompt: |
      AI Review Summary:
      {review_comments.summary}

      Key Issues:
      {review_comments.issues}

      Please review and add any additional comments.
    allow_edit: true
    output: final_review
    next: [submit_review]

  - id: submit_review
    type: compute
    handler: submit_pr_review
    inputs:
      review: $ctx.final_review
    next: [end]
```

### Interactive Debugging

```yaml
name: interactive_debug
description: Debugging with human guidance

nodes:
  - id: analyze_error
    type: agent
    role: debugger
    goal: Analyze the error and identify potential causes
    output: analysis
    next: [propose_fix]

  - id: propose_fix
    type: agent
    role: fixer
    goal: Propose a fix based on analysis
    output: proposed_fix
    next: [review_fix]

  - id: review_fix
    type: hitl
    hitl_type: override
    prompt: |
      Proposed Fix:
      {proposed_fix.code}

      Reasoning:
      {proposed_fix.reasoning}

      Accept, modify, or provide alternative?
    allow_edit: true
    output: approved_fix
    next: [apply_fix]

  - id: apply_fix
    type: agent
    role: implementer
    goal: Apply the approved fix
    input: $ctx.approved_fix
    next: [verify_fix]
```

## Timeout Handling

Configure timeout behavior:

```yaml
# In HITL node
timeout: 300  # seconds
on_timeout: reject  # Options: approve, reject, skip, escalate

# Escalate to different handler
on_timeout: escalate
escalate_to: manager_approval
```

Programmatic timeout handling:

```python
from victor.workflows.hitl import TimeoutPolicy

policy = TimeoutPolicy(
    default_timeout=300,
    on_timeout="reject",
    escalation_chain=[
        ("team_lead", 600),
        ("manager", 1800),
        ("auto_reject", 0),
    ],
)

coordinator = HITLCoordinator(
    handler=handler,
    timeout_policy=policy,
)
```

## Request History

Track HITL interactions:

```python
from victor.workflows.hitl import HITLHistory

history = HITLHistory()

# Record a request
history.record(
    request_id="req-123",
    node_id="approve_deploy",
    prompt="Deploy to production?",
    response=response,
    responder="user@example.com",
    response_time=45.2,
)

# Query history
pending = history.get_pending()
recent = history.get_recent(limit=10)
by_user = history.get_by_responder("user@example.com")

# Export for audit
history.export_to_json("hitl_audit.json")
```

## Event Integration

HITL events for observability:

```python
from victor.observability.event_bus import get_event_bus, EventCategory

bus = get_event_bus()

bus.subscribe(EventCategory.LIFECYCLE, lambda e:
    if e.event_type.startswith("hitl_"):
        print(f"HITL: {e.event_type} - {e.data}")
)

# Events emitted:
# - hitl_requested
# - hitl_approved
# - hitl_rejected
# - hitl_timeout
# - hitl_escalated
# - hitl_override
```

## Best Practices

### 1. Provide Clear Context

```yaml
# Good - clear, specific prompt with context
prompt: |
  Deploy version {build_info.version} to {environment}?

  Changes since last deploy:
  {changes_summary}

  Test Results:
  - Unit: {test_results.unit}
  - Integration: {test_results.integration}

# Avoid - vague prompt
prompt: "Continue?"
```

### 2. Set Appropriate Timeouts

```yaml
# Quick decisions
- id: simple_approval
  timeout: 60
  on_timeout: skip

# Critical decisions
- id: production_deploy
  timeout: 3600  # 1 hour
  on_timeout: reject
```

### 3. Use Escalation Chains

```python
policy = TimeoutPolicy(
    escalation_chain=[
        ("developer", 300),
        ("team_lead", 900),
        ("manager", 3600),
    ]
)
```

### 4. Enable Audit Logging

```python
handler = CLIHITLHandler(
    audit_log="/var/log/victor/hitl_audit.log",
    log_context=True,
)
```

### 5. Handle Rejections Gracefully

```yaml
next:
  approved: continue
  rejected: handle_rejection

- id: handle_rejection
  type: agent
  goal: Notify stakeholders and clean up
```

## Testing HITL Workflows

### Auto-Approve for Tests

```python
from victor.workflows.hitl import AutoApproveHandler

# Always approve for testing
handler = AutoApproveHandler()

engine = WorkflowEngine(hitl_handler=handler)
```

### Mock Handler

```python
from victor.workflows.hitl import MockHITLHandler

handler = MockHITLHandler(
    responses={
        "approve_deploy": HITLResponse(approved=True),
        "review_changes": HITLResponse(approved=True, feedback="LGTM"),
    }
)
```

### Record/Replay

```python
from victor.workflows.hitl import RecordingHandler, ReplayHandler

# Record real interactions
recording_handler = RecordingHandler(
    delegate=CLIHITLHandler(),
    record_file="hitl_recording.json",
)

# Replay in tests
replay_handler = ReplayHandler(
    replay_file="hitl_recording.json",
)
```

## Troubleshooting

### Workflow Stuck at HITL

1. Check timeout setting
2. Verify handler is configured
3. Check for handler errors in logs

### Approval Not Received

1. Verify notification channel (Slack, email, etc.)
2. Check handler connectivity
3. Review webhook URLs

### Context Not Showing

1. Verify `context_keys` match state keys
2. Check state contains expected data
3. Enable debug logging

## Related Resources

- [Workflow DSL Guide](workflow-development/dsl.md) - Workflow syntax
- [Workflow Scheduler Guide](WORKFLOW_SCHEDULER.md) - Scheduled workflows
- [Observability Guide](OBSERVABILITY.md) - HITL event monitoring
