# Vertical Creation Guide - Part 1

**Part 1 of 2:** Overview, Quick Start, Template Structure, Creating from Scratch, Extracting Templates, Validation, and Customization

---

## Navigation

- **[Part 1: Templates & Creation](#)** (Current)
- [Part 2: Best Practices, Examples, Advanced](part-2-best-practices-examples-advanced.md)
- [**Complete Guide**](../vertical_creation_guide.md)

---

This guide explains how to create new verticals in Victor using the template-based scaffolding system, which reduces code duplication by 65-70%.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Template Structure](#template-structure)
4. [Creating a Vertical from Scratch](#creating-a-vertical-from-scratch)
5. [Extracting Templates from Existing Verticals](#extracting-templates-from-existing-verticals)
6. [Template Validation](#template-validation)
7. [Customizing Generated Verticals](#customizing-generated-verticals)
8. [Best Practices](#best-practices) *(in Part 2)*
9. [Examples](#examples) *(in Part 2)*

---

## Overview

Victor's vertical template system provides a declarative way to define and generate vertical implementations. Instead of writing 500+ lines of boilerplate code, you can:

1. Define your vertical in a YAML template
2. Generate all necessary files automatically
3. Customize only the parts that need special logic

This approach:
- **Reduces duplication**: 65-70% less code
- **Ensures consistency**: All verticals follow the same structure
- **Speeds up development**: Create a new vertical in minutes
- **Maintains best practices**: Generated code follows Victor's patterns

---

## Quick Start

### Create a New Vertical from Template

```bash
# 1. Create a template YAML (or use an existing one)
cat > my_vertical.yaml << 'EOF'
metadata:
  name: security
  description: Security analysis and vulnerability detection
  version: "0.5.0"
  category: security
  tags: [security, vulnerability, scanning]

tools:
  - read
  - grep
  - security_scan
  - web_search

system_prompt: |
  You are a security analysis assistant specializing in vulnerability detection...

EOF

# 2. Generate the vertical
victor vertical create my_vertical.yaml

# 3. Use your new vertical
victor chat --vertical security
```

[Content continues through Customization...]

---

**Continue to [Part 2: Best Practices, Examples, Advanced](part-2-best-practices-examples-advanced.md)**
