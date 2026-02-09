# Victor AI Documentation Standards

**Version:** 1.0
**Last Updated:** January 31, 2026
**Applies to:** All documentation in `docs/` directory

## Table of Contents

1. [Writing Principles](#writing-principles)
2. [Content Types (Diátaxis Framework)](#content-types-diátaxis-framework)
3. [File Size Limits](#file-size-limits)
4. [Formatting Guidelines](#formatting-guidelines)
5. [Code Examples](#code-examples)
6. [Diagrams](#diagrams)
7. [Quality Checklist](#quality-checklist)
8. [Templates](#templates)

---

## Writing Principles

### Clarity First

- **Write for your audience:** Beginners need simple language; experts need technical depth
- **Be concise:** Remove unnecessary words. Every sentence should add value
- **Use active voice:** "Install Victor" not "Victor should be installed"
- **Avoid jargon:** Explain technical terms or provide links to definitions

### Scannability

- **Front-load important information:** Put key points first
- **Use bold headings:** Make sections easy to scan
- **Short paragraphs:** Max 2-3 sentences per paragraph
- **Bullet points:** Use lists for steps, options, examples (max 7 items)

### Accuracy

- **Test all code examples:** Ensure they work before documenting
- **Verify commands:** Run commands to confirm output
- **Update frequently:** Review and update docs with each release
- **Link check:** Verify all links work

---

## Content Types (Diátaxis Framework)

Victor follows the **Diátaxis framework** for documentation:

### 1. Tutorials

**Purpose:** Learning-oriented, step-by-step lessons

**Characteristics:**
- Goal: Complete a project or learn a skill
- Style: Lesson, step-by-step
- Audience: Beginners
- Structure: Linear, ordered steps

**Example:** [Beginner Journey](../journeys/beginner.md)

**Template:** [Tutorial Template](#tutorial-template)

### 2. How-to Guides

**Purpose:** Problem-oriented, practical solutions

**Characteristics:**
- Goal: Solve a specific problem
- Style: Practical, focused
- Audience: Users with some knowledge
- Structure: Problem → Solution → Examples

**Example:** [Creating Workflows](../guides/workflows/)

**Template:** [How-to Template](#how-to-template)

### 3. Reference

**Purpose:** Information-oriented, look up technical details

**Characteristics:**
- Goal: Find specific information quickly
- Style: Formal, structured
- Audience: All users (varied expertise)
- Structure: Organized for search and scanning

**Example:** [API Reference](../reference/api/)

**Template:** [Reference Template](#reference-template)

### 4. Explanation

**Purpose:** Understanding-oriented, concepts and context

**Characteristics:**
- Goal: Understand concepts and context
- Style: Explanatory, discussion
- Audience: Intermediate to advanced
- Structure: Concept → Details → Implications

**Example:** [Architecture Overview](../architecture/overview.md)

**Template:** [Explanation Template](#explanation-template)

---

## File Size Limits

To maintain readability and findability:

| Content Type | Max Lines | Target Reading Time |
|--------------|-----------|---------------------|
| **Quick start guides** | 300 lines | 10-15 minutes |
| **How-to guides** | 500 lines | 20-30 minutes |
| **Reference pages** | 700 lines | Scannable |
| **Architecture docs** | 800 lines | 30-45 minutes |
| **Tutorials** | 600 lines | 30-40 minutes |

**If content exceeds limits:**
- Split into multiple focused files
- Use subdirectories for related content
- Add links between related sections

---

## Formatting Guidelines

### Headings

```markdown
# Main Title (H1) - One per file

## Section (H2) - Major sections

### Subsection (H3) - Sub-sections

#### Detail (H4) - Rarely used
```text

**Rules:**
- Start with H1 (document title)
- Use H2 for major sections
- Use H3 for sub-sections
- Avoid H4-H5 unless absolutely necessary
- Use sentence case for headings (not title case)

### Emphasis

```markdown
**Bold** for emphasis, key terms, UI elements
*Italic* for variables, placeholders
`Code` for inline code, commands, file names
```

**Examples:**
- Click the **Run** button
- Replace `YOUR_API_KEY` with your actual key
- The `victor chat` command starts the interface

### Lists

**Bullet Lists:**
- Use for unordered items
- Max 7 items per list
- Start each item with a capital letter
- Use parallel structure (all nouns, all verbs, etc.)

**Numbered Lists:**
- Use for ordered steps
- Complete steps in order
- Include expected outcomes

**Nested Lists:**
- Use for hierarchical information
- Max 2 levels deep

### Links

```markdown
[Link text](path/to/file.md)
[External link](https://example.com)
[Section link](#heading-id)
```text

**Rules:**
- Use descriptive link text (not "click here")
- For internal links, use relative paths
- For external links, include https://
- Test all links before committing

### Tables

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

**Rules:**
- Use for structured data
- Include headers
- Max 5 columns for readability
- Left-align text, right-align numbers

### Callouts

```markdown
**Note:** Information worth noting
**Tip:** Helpful suggestion
**Warning:** Cautionary advice
**Important:** Critical information
```text

---

## Code Examples

### Formatting

Use fenced code blocks with syntax highlighting:

````markdown
```python
def example_function():
    """Example function."""
    return "Hello, World!"
```
````

### Language Specification

Specify language for syntax highlighting:

| Language | Specifier |
|----------|-----------|
| Python | `python` |
| Bash/Shell | `bash` |
| YAML | `yaml` |
| JSON | `json` |
| Markdown | `markdown` |
| JavaScript | `javascript` |
| TypeScript | `typescript` |

### Code Block Requirements

Every code block must include:

1. **Syntax highlighting:** Specify language
2. **Explanation:** Describe what code does
3. **Context:** When to use this code
4. **Expected output:** What to expect

**Example:**

```python
# Create a custom tool
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    """My custom tool."""

    def execute(self, param: str) -> Dict[str, Any]:
        """Execute the tool."""
        # Tool implementation here
        return {"success": True, "data": param}
```text

This code defines a custom tool. Use this template when creating tools that extend Victor's capabilities.

### Testing Code Examples

All code examples must be tested before documentation:

```bash
# Test Python code
python -m pyexamples/my_example.py

# Test shell commands
bash -c examples/command.sh

# Test YAML syntax
yamllint config/example.yaml
```

---

## Diagrams

### When to Use Diagrams

Add a diagram when:
- Explaining complex flows
- Showing system architecture
- Illustrating user journeys
- Visualizing data structures

**Rule of thumb:** One diagram per 500 words

### Diagram Types

| Type | When to Use | Tool |
|------|-------------|------|
| **Flowchart** | Process flows, decision trees | Mermaid |
| **Sequence** | Interactions over time | Mermaid |
| **Architecture** | System structure | Mermaid |
| **State Machine** | State transitions | Mermaid |
| **Class/ER** | Data models | PlantUML |

### Diagram Standards

- **Max 20 nodes** per diagram
- **Max 5 colors** (Victor standard palette)
- **Alt text** required for accessibility
- **Source files** committed (.mmd, .puml)
- **Clear labels** on all nodes and edges

**Victor Color Palette:**
- Green: Success/Positive (`#2e7d32`)
- Blue: Information (`#1565c0`)
- Orange: Warning (`#e65100`)
- Red: Error (`#c62828`)
- Purple: Special (`#6a1b9a`)

### Embedding Diagrams

```markdown
## Title

Description of diagram.

```mermaid
graph TB
    A[Start] --> B[End]
```text

**Figure 1:** Diagram description for accessibility.
```

See [Diagram Standards](../diagrams/README.md) for detailed guidelines.

---

## Quality Checklist

Every documentation file must meet these criteria before merging:

### Content Requirements

- [ ] **Clear purpose statement** in first paragraph
- [ ] **Target audience** specified
- [ ] **Reading time estimate** included
- [ ] **Code examples** tested and working (where applicable)
- [ ] **Diagrams** included where helpful
- [ ] **Next steps/related links** at end
- [ ] **Last updated date** in footer

### Formatting Requirements

- [ ] **File size** within limits for content type
- [ ] **Headings** follow hierarchy (H1 → H2 → H3)
- [ ] **Links** tested and working
- [ ] **Code blocks** have syntax highlighting
- [ ] **Tables** properly formatted
- [ ] **Lists** use parallel structure
- [ ] **No broken references** to other docs

### Style Requirements

- [ ] **Active voice** used throughout
- [ ] **Short paragraphs** (max 2-3 sentences)
- [ ] **Bold headings** for scannability
- [ ] **Plain language** (minimal jargon)
- [ ] **Consistent terminology**
- [ ] **No typos or grammatical errors**

### Technical Requirements

- [ ] **CI/CD checks** passing (lint, links, spell check)
- [ ] **Filenames** use kebab-case
- [ ] **Images** have alt text
- [ ] **Diagrams** have source files committed

---

## Templates

### Tutorial Template

```markdown
# Tutorial Title

**Time Commitment:** X minutes
**Target Audience:** [Beginners/Intermediate/Advanced]
**Prerequisites:** [List prerequisites]

## Learning Objectives

By the end of this tutorial, you will be able to:
- [ ] Objective 1
- [ ] Objective 2
- [ ] Objective 3

## Overview

[Brief description of what you'll build/learn]

## Step 1: Title

[Step content with explanation]

**Code Example:**
```python
[code here]
```text

[Explanation of what code does]

## Step 2: Title

[Continue steps...]

## Summary

[Recap what was learned]

## Next Steps

- [ ] Next tutorial
- [ ] Related guide
- [ ] Reference documentation

---

**Last Updated:** [Date]
**Reading Time:** [Time]
**Related:** [Link to related docs]
```

### How-to Template

```markdown
# How to [Do Something]

**Problem:** [Description of problem]
**Solution:** [Brief summary of solution]
**Time:** X minutes

## Overview

[Context and when to use this solution]

## Prerequisites

- [ ] Prerequisite 1
- [ ] Prerequisite 2

## Method 1: Title

[Step-by-step instructions]

**Example:**
```bash
[command or code]
```text

**Expected Output:**
```
[what you should see]
```text

## Method 2: Title (Alternative)

[Alternative approach]

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Issue 1 | Fix 1 |
| Issue 2 | Fix 2 |

## See Also

- [Related guide](link.md)
- [Reference](link.md)

---

**Last Updated:** [Date]
```

### Reference Template

```markdown
# [Component/API] Reference

**Version:** X.X
**Status:** [Stable/Experimental/Deprecated]

## Overview

[Brief description of component]

## API

### Method/Function Name

**Signature:**
```python
function_name(param1: type, param2: type) -> return_type
```text

**Description:**
[Detailed description]

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| param1 | type | Description |
| param2 | type | Description |

**Returns:**
- `type`: Description

**Raises:**
- `Exception`: When and why

**Example:**
```python
# Usage example
result = function_name("value")
```

## Configuration

[Configuration options if applicable]

## See Also

- [Related component](link.md)
- [Usage guide](link.md)

---

**Last Updated:** [Date]
```text

### Explanation Template

```markdown
# [Concept] Explained

**Target Audience:** [Intermediate/Advanced]
**Related Concepts:** [Link to related concepts]

## Overview

[High-level description of concept]

## How It Works

[Detailed explanation]

### Key Components

1. **Component 1:** Description
2. **Component 2:** Description
3. **Component 3:** Description

## Architecture

```mermaid
[Diagram showing concept]
```

## Use Cases

| Use Case | Description | Example |
|----------|-------------|---------|
| Case 1 | Description | Example |
| Case 2 | Description | Example |

## Trade-offs

| Advantage | Disadvantage |
|-----------|--------------|
| Pro 1 | Con 1 |
| Pro 2 | Con 2 |

## Related Concepts

- [Concept 1](link.md)
- [Concept 2](link.md)

## Further Reading

- [External resource 1](url)
- [External resource 2](url)

---

**Last Updated:** [Date]
**Reading Time:** X minutes
```text

---

## CI/CD Checks

Documentation is automatically validated by CI/CD:

### Markdown Lint

```bash
# Run locally
markdownlint '**/*.md'

# CI/CD check
- name: Lint markdown
  run: markdownlint docs/
```

### Link Validation

```bash
# Run locally
lychee docs/ --verbose

# CI/CD check
- name: Check links
  run: lychee docs/ --verbose
```text

### Spell Check

```bash
# Run locally
codespell docs/

# CI/CD check
- name: Check spelling
  run: codespell docs/
```

### Diagram Render

```bash
# Run locally
mmdc -i docs/diagrams/example.mmd -o /tmp/test.svg

# CI/CD check
- name: Render diagrams
  run: find docs/diagrams -name "*.mmd" -exec mmdc -i {} -o {}.svg \;
```text

---

## Contributing

When contributing documentation:

1. **Choose the right content type:** Tutorial, How-to, Reference, or Explanation
2. **Use the appropriate template:** See templates above
3. **Follow writing principles:** Clarity, scannability, accuracy
4. **Meet quality checklist:** All items checked
5. **Pass CI/CD checks:** Lint, links, spell check
6. **Get review:** Submit PR for review

See [Contributing Guide](../contributing/) for full contribution workflow.

---

**Last Updated:** January 31, 2026
**Maintained by:** Victor AI Documentation Team
