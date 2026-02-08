# Vertical Creation Guide

Guide to creating new verticals in Victor using the template-based scaffolding system.

---

## Quick Summary

Victor's vertical template system provides a declarative way to define and generate vertical implementations:
- **Reduces duplication**: 65-70% less code
- **Ensures consistency**: All verticals follow the same structure
- **Speeds up development**: Create a new vertical in minutes
- **Maintains best practices**: Generated code follows Victor's patterns

---

## Guide Parts

### [Part 1: Templates & Creation](part-1-overview-customization.md)
- Overview
- Quick Start
- Template Structure
- Creating a Vertical from Scratch
- Extracting Templates from Existing Verticals
- Template Validation
- Customizing Generated Verticals

### [Part 2: Best Practices, Examples, Advanced](part-2-best-practices-examples-advanced.md)
- Best Practices
- Examples (Simple Analysis, Data Science)
- Advanced Topics (Custom Tools, Prompts, Modes)
- Troubleshooting

---

## Quick Start

```bash
# 1. Create a template YAML
cat > my_vertical.yaml << 'EOF'
metadata:
  name: security
  description: Security analysis and vulnerability detection
  version: "0.5.0"

tools:
  - read
  - grep
  - security_scan

system_prompt: |
  You are a security analysis assistant...
EOF

# 2. Generate the vertical
victor vertical create my_vertical.yaml

# 3. Use your new vertical
victor chat --vertical security
```

---

## Related Documentation

- [Vertical Architecture](../../architecture/VERTICALS.md)
- [Template Reference](../reference/TEMPLATE_REFERENCE.md)
- [Creating Tools](./CREATING_TOOLS.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 15 min (all parts)
