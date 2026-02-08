# Vertical Creation Guide

Guide to creating custom verticals for Victor AI.

---

## Quick Summary

This guide covers creating custom verticals for Victor AI:

- **Overview** - What are verticals and when to create them
- **Quick Start** - Create a vertical in 5 minutes
- **Template Structure** - Understanding the vertical template
- **Creating from Scratch** - Building a vertical step-by-step
- **Extracting Templates** - Learning from existing verticals
- **Template Validation** - Testing and validating your vertical
- **Customizing** - Adapting templates to your needs
- **Best Practices** - Patterns and anti-patterns
- **Examples** - Sample vertical implementations
- **Advanced Topics** - Advanced customization options
- **Troubleshooting** - Common issues and solutions

---

## Guide Parts

### [Part 1: Vertical Fundamentals](part-1-vertical-fundamentals.md)
- Overview
- Quick Start
- Template Structure
- Creating a Vertical from Scratch
- Customizing Generated Verticals

### [Part 2: Advanced & Examples](part-2-advanced-examples.md)
- Extracting Templates from Existing Verticals
- Template Validation
- Best Practices
- Examples
- Advanced Topics
- Troubleshooting
- Next Steps
- Related Documentation

---

## Quick Start

**1. Create a vertical using the CLI:**
```bash
victor vertical create myvertical --description "My custom vertical"
```

**2. Or manually:**
```bash
mkdir -p victor/myvertical
touch victor/myvertical/__init__.py
touch victor/myvertical/assistant.py
```

**3. Implement your vertical:**
```python
from victor.verticals.base import VerticalBase

class MyVertical(VerticalBase):
    @property
    def name(self) -> str:
        return "myvertical"

    def get_tools(self):
        return [my_custom_tool]

    def get_system_prompt(self):
        return "You are a specialized assistant for..."
```

---

## What are Verticals?

Verticals are domain-specific specializations that extend Victor AI's capabilities:

- **Coding Vertical** - Code analysis, refactoring, testing
- **DevOps Vertical** - Docker, Kubernetes, CI/CD
- **RAG Vertical** - Document search, retrieval, Q&A
- **Data Analysis Vertical** - Pandas, visualization, statistics
- **Research Vertical** - Literature search, analysis tools

**When to create a vertical:**
- You have a specialized domain with unique tools
- You need custom system prompts for a domain
- You want to package domain-specific workflows
- You need custom tool selection logic

---

## Related Documentation

- [Vertical Architecture](../../../architecture/VERTICALS.md)
- [Tool Development](../tools/TOOL_DEVELOPMENT.md)
- [Workflow Development](../workflows/README.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 20 min (all parts)
