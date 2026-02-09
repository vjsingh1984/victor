# Documentation Templates

This directory contains templates for different types of documentation content following the Diátaxis framework.

## Template Overview

| Template | Purpose | When to Use |
|----------|---------|-------------|
| [Tutorial](tutorial.md) | Learning-oriented lessons | Teaching step-by-step |
| [How-to Guide](how-to.md) | Problem-oriented solutions | Solving specific problems |
| [Reference](reference.md) | Information-oriented reference | Looking up technical details |
| [Explanation](explanation.md) | Understanding-oriented concepts | Explaining ideas and context |

## Diátaxis Framework

Victor AI documentation follows the **Diátaxis framework**, which categorizes documentation into four types:

```text
                    Tutorials
                        │
                        │ Learning
                        │
      Explanation ──────┼─────── How-to Guides
         Understanding │         Practical
                        │
                   Information
                     Reference
```

### Content Types

1. **Tutorials** (Learning-oriented)
   - Goal: Complete a project or learn a skill
   - Style: Lesson, step-by-step
   - Audience: Beginners
   - Example: [Beginner Journey](../journeys/beginner.md)

2. **How-to Guides** (Problem-oriented)
   - Goal: Solve a specific problem
   - Style: Practical, focused
   - Audience: Users with some knowledge
   - Example: [Creating Workflows](../guides/workflows/)

3. **Reference** (Information-oriented)
   - Goal: Look up technical details
   - Style: Formal, structured
   - Audience: All users
   - Example: [API Reference](../reference/api/)

4. **Explanation** (Understanding-oriented)
   - Goal: Understand concepts and context
   - Style: Explanatory, discussion
   - Audience: Intermediate to advanced
   - Example: [Architecture Overview](../architecture/overview.md)

## Using Templates

### 1. Choose the Right Template

Consider your goal:
- **Teaching a skill?** → Use Tutorial template
- **Solving a problem?** → Use How-to template
- **Documenting an API?** → Use Reference template
- **Explaining a concept?** → Use Explanation template

### 2. Copy Template

```bash
# Copy template to new location
cp docs/templates/how-to.md docs/guides/my-guide.md

# Or copy from template content
cat docs/templates/tutorial.md > docs/tutorials/my-tutorial.md
```text

### 3. Customize Template

Replace placeholders with your content:
- **Metadata:** Update time, audience, prerequisites
- **Content:** Replace with your actual content
- **Examples:** Add relevant code examples
- **Links:** Update cross-references

### 4. Review Against Standards

Check against [Documentation Standards](../STANDARDS.md):
- File size within limits
- All quality checklist items met
- Code examples tested
- Diagrams have alt text
- Links tested

## Template Features

Each template includes:

### Header Section
- **Title:** Clear, descriptive title
- **Metadata:** Time, audience, difficulty, prerequisites
- **Learning objectives:** What readers will learn

### Body Section
- **Structured content:** Appropriate for content type
- **Code examples:** With syntax highlighting and explanations
- **Diagrams:** Where helpful (Mermaid.js)
- **Tables:** For structured information

### Footer Section
- **Summary:** Key takeaways
- **Next steps:** Related documentation
- **See also:** Cross-references
- **Metadata:** Last updated, reading time

## Template Customization

### Adding Sections

Add sections as needed for your content:

```markdown
## Custom Section

Content for custom section.

### Subsection

Subsection content.
```

### Diagrams

Add diagrams using Mermaid.js:

```markdown
## Architecture

```mermaid
graph TB
    A[Start] --> B[End]
```text

**Figure 1:** Description of diagram.
```

### Code Examples

Add code examples with syntax highlighting:

```markdown
**Example:**
```python
def example_function():
    """Example function."""
    return "Hello, World!"
```text

**Explanation:** This code does X, Y, Z.
```

### Callouts

Use callouts for emphasis:

```markdown
**Note:** Information worth noting

**Tip:** Helpful suggestion

**Warning:** Cautionary advice

**Important:** Critical information
```text

## Quick Reference

### Tutorial Template
- **Purpose:** Learning-oriented lessons
- **Length:** 300-600 lines (10-40 min read)
- **Structure:** Linear steps, objectives, summary
- **Audience:** Beginners

### How-to Template
- **Purpose:** Problem-oriented solutions
- **Length:** 300-500 lines (15-30 min read)
- **Structure:** Problem → Solution → Examples
- **Audience:** Users with some knowledge

### Reference Template
- **Purpose:** Information-oriented reference
- **Length:** Up to 700 lines (scannable)
- **Structure:** Organized for search, API docs
- **Audience:** All users (varied expertise)

### Explanation Template
- **Purpose:** Understanding-oriented concepts
- **Length:** 400-800 lines (30-45 min read)
- **Structure:** Concept → Details → Implications
- **Audience:** Intermediate to advanced

## Examples

### Tutorial Example
- [Beginner Journey](../journeys/beginner.md)
- [Developer Journey](../journeys/developer.md)

### How-to Example
- [Creating Workflows](../guides/workflows/)
- [Creating Tools](../guides/tutorials/CREATING_TOOLS.md)

### Reference Example
- [API Reference](../reference/api/)
- [Configuration Reference](../reference/configuration/)

### Explanation Example
- [Architecture Overview](../architecture/overview.md)
- [Design Patterns](../architecture/patterns/)

## See Also

- [Documentation Standards](../STANDARDS.md)
- [Contributing Guide](../contributing/)
- [Diagram Standards](../diagrams/README.md)
- [Writing Guide](../contributing/documentation-guide.md)

---

**Reading Time:** 3 min
**Last Updated:** January 31, 2026
**Maintained by:** Victor AI Documentation Team
