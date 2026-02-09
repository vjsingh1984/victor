# Dynamic Skills Guide

Guide to Victor AI's dynamic skills system for runtime tool discovery, composition, and chaining.

---

## Quick Summary

Victor AI's dynamic skills system enables:
- **Skill Discovery** - Runtime tool discovery with semantic matching
- **Skill Composition** - Combine multiple tools into cohesive skills
- **Skill Chaining** - Automatic planning and execution of multi-step workflows
- **Adaptation** - Learning and adaptation from skill execution

---

## Guide Parts

### [Part 1: Discovery, Composition, Chaining](part-1-discovery-composition-chaining-adaptation.md)
- What are Dynamic Skills?
- Skill Discovery
- Skill Composition
- Skill Chaining
- Adaptation and Learning

### [Part 2: Best Practices, Troubleshooting, Examples](part-2-best-practices-troubleshooting-examples.md)
- Best Practices
- Troubleshooting
- Examples

---

## Quick Start

```python
from victor.skills import discover_skill, execute_skill

# Discover skill for task
skill = await discover_skill(
    task="analyze python code for bugs"
)

# Execute skill
result = await execute_skill(
    skill=skill,
    code="my_script.py",
    severity="medium"
)
```text

---

## Related Documentation

- [Skill API Reference](../../reference/skills/README.md)
- [Tool Registry](../../reference/tools/README.md)
- [MCP Integration](../../integrations/MCP_INTEGRATION.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 10 min (all parts)
