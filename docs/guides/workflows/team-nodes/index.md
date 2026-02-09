# Team Nodes Guide

Complete guide for using team nodes in Victor workflows.

## Team Node Workflow

```mermaid
graph TB
    A[Start: User Request] --> B[Create Team Node]
    B --> C[Configure Members]
    C --> D{Formation Type}

    D -->|parallel| E[Execute Members<br/>Concurrently]
    D -->|pipeline| F[Execute Members<br/>In Sequence]
    D -->|recursive| G[Nested Teams<br/>with Depth Tracking]

    E --> H[Aggregate Results]
    F --> H
    G --> H

    H --> I{Depth Limit?}
    I -->|No| J[Return Final Result]
    I -->|Yes| K[Continue Recursion]
    K --> B

    J --> L[End]

    style B fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#f3e5f5
    style H fill:#e8f5e9
```

---

## Quick Summary

Team nodes enable multi-agent collaboration with:
- Parallel and pipeline execution patterns
- Recursive team formation
- Depth tracking and control
- Flexible member configuration

---

## Guide Parts

### [Part 1: Overview & Formation](part-1-overview-formation.md)
- Overview and When to Use Team Nodes
- Quick Start
- YAML Syntax and Configuration
- Team Formation Types

### [Part 2: Recursion & Configuration](part-2-recursion-configuration.md)
- Recursion Depth Tracking
- Member Configuration
- Configuration Examples

### [Part 3: Best Practices & Errors](part-3-best-practices-errors.md)
- Best Practices
- Error Handling

### [Part 4: Complete Examples](part-4-complete-examples.md)
- Complete Examples
- Additional Resources

---

**Reading Time:** 1 min
**Last Updated:** February 01, 2026
