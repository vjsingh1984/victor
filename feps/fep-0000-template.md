# FEP-{number}: {Title}

- **FEP**: {number}
- **Title**: {Brief, descriptive title}
- **Type**: Standards Track / Informational / Process
- **Status**: Draft / Review / Accepted / Rejected / Deferred / Withdrawn
- **Authors**: {Name} <{email}> (@{github})
- **Created**: {YYYY-MM-DD}
- **Modified**: {YYYY-MM-DD}

---

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)

---

## Summary

{~200-word executive summary. Should be understandable by someone unfamiliar with the details. Include:

- What problem does this FEP solve?
- What is the proposed solution at a high level?
- Who will be affected by this change?
- What is the impact on backward compatibility?
}

## Motivation

### Problem Statement

{Detailed description of the problem this FEP addresses. Include:

- Current limitations or pain points
- Real-world use cases or scenarios
- Why this matters for the Victor ecosystem
- Metrics or evidence if available (e.g., "this issue appears in 3 active verticals")
}

### Goals

{Specific, measurable goals this FEP aims to achieve:

1. {Goal 1}
2. {Goal 2}
3. {Goal 3}
}

### Non-Goals

{What this FEP explicitly does NOT address:

- {Out of scope item 1}
- {Out of scope item 2}
}

## Proposed Change

### High-Level Design

{Architecture diagrams, flow charts, or high-level descriptions}

```python
# Example code showing the proposed API
from victor.framework import NewFeature

feature = NewFeature(
    parameter="value",
    # ...
)
```

### Detailed Specification

{Technical details broken into subsections}

#### {Section Title}

{Detailed technical specification:

- API definitions
- Protocol changes
- Configuration changes
- Error handling
- Edge cases
}

#### {Another Section}

{Continue with subsections as needed}

### API Changes

{If applicable, show:

- New APIs (with signatures)
- Modified APIs (show before/after)
- Deprecated APIs
}

```python
# Before
old_function(param: str) -> None:
    pass

# After
new_function(param: str, enable_feature: bool = True) -> None:
    pass
```

### Configuration Changes

{If applicable, describe new or modified configuration:

```yaml
# New config option in config.yaml
new_feature:
  enabled: true
  timeout: 30
```

### Dependencies

{List any new dependencies or version updates:

- New dependencies: `library >= 1.0`
- Version bumps: `existing-lib >= 2.0`
}

## Benefits

{Quantitative and qualitative benefits:}

### For Framework Users

- {Benefit 1}: {Impact}
- {Benefit 2}: {Impact}

### For Vertical Developers

- {Benefit 1}: {Impact}
- {Benefit 2}: {Impact}

### For the Ecosystem

- {Benefit 1}: {Impact}
- {Benefit 2}: {Impact}

## Drawbacks and Alternatives

### Drawbacks

{Honest assessment of downsides:

- {Drawback 1}: {Mitigation strategy}
- {Drawback 2}: {Mitigation strategy}
}

### Alternatives Considered

{Why this approach vs. alternatives:}

1. **Alternative 1: {Name}**
   - Description: {Brief description}
   - Pros: {Pros}
   - Cons: {Cons}
   - Why rejected: {Reason}

2. **Alternative 2: {Name}**
   - Description: {Brief description}
   - Pros: {Pros}
   - Cons: {Cons}
   - Why rejected: {Reason}

## Unresolved Questions

{Open questions that need discussion:}

- **Question 1**: {Question} (Proposed answer: {Initial thoughts})
- **Question 2**: {Question} (Proposed answer: {Initial thoughts})

## Implementation Plan

{Phased implementation approach:}

### Phase 1: Foundation ({Duration})

- [ ] {Task 1}
- [ ] {Task 2}
- [ ] {Task 3}

**Deliverable**: {What will be delivered}

### Phase 2: Integration ({Duration})

- [ ] {Task 1}
- [ ] {Task 2}

**Deliverable**: {What will be delivered}

### Phase 3: Rollout ({Duration})

- [ ] {Task 1}
- [ ] {Task 2}

**Deliverable**: {What will be delivered}

### Testing Strategy

{How this will be tested:}

- Unit tests: {Coverage goal}
- Integration tests: {Scenarios}
- Backward compatibility tests: {Approach}
- Performance tests: {Benchmarks}

### Rollout Plan

{How this will be deployed:}

- Feature flags: {If applicable}
- Gradual rollout: {Phasing}
- Documentation updates: {What needs updating}

## Migration Path

{For breaking changes, provide clear migration guide:}

### From Old API to New API

{Step-by-step migration:}

1. **Step 1**: {Description}
   ```python
   # Code example
   ```

2. **Step 2**: {Description}
   ```python
   # Code example
   ```

### Deprecation Timeline

- {Version}: Feature introduced with deprecation warning
- {Version}: Old API removed (estimated {date})

## Compatibility

### Backward Compatibility

- **Breaking change**: Yes/No
- **Migration required**: Yes/No
- **Deprecation period**: {Duration}

### Version Compatibility

- Minimum Python version: {X.Y}
- Minimum dependency versions: {List}
- Platform compatibility: {List}

### Vertical Compatibility

{Impact on existing verticals:}

- Built-in verticals: {Impact and required changes}
- External verticals: {Impact and migration guide}

## References

- [Related FEP-{number}](link)
- [GitHub Issue #{number}](link)
- [Discussion](link)
- [Relevant documentation](link)
- [Inspired by](link)

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
