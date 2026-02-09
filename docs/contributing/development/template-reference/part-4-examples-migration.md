# Template Reference - Part 4

**Part 4 of 4:** Complete Example, Migration, and Best Practices

---

## Navigation

- [Part 1: Structure & Extensions](part-1-structure-extensions.md)
- [Part 2: Workflows & Config](part-2-workflows-config.md)
- [Part 3: Inheritance & Validation](part-3-inheritance-validation.md)
- **[Part 4: Examples & Migration](#)** (Current)
- [**Complete Reference**](../template_reference.md)

---

```yaml
# complete_vertical_template.yaml
metadata:
  name: complete_vertical
  description: "Complete example vertical"
  version: "0.5.0"
  author: "Your Name"
  license: "Apache-2.0"
  category: development
  tags:
    - example
    - complete
  provider_hints:
    preferred_models:
      - claude-sonnet-4-5
    min_context: 128000
    supports_tools: true
  evaluation_criteria:
    - success_rate
    - quality_score

tools:
  - read_file
  - write_file
  - edit_file
  - grep
  - ls

system_prompt: |
  You are {vertical_name}, an expert assistant.

  Your capabilities:
  - Read and analyze files
  - Write and edit code
  - Search for information

  Guidelines:
  1. Be thorough in your analysis
  2. Provide clear explanations
  3. Verify your work

stages:
  INITIAL:
    name: INITIAL
    description: "Understanding the request"
    tools: [read_file, ls]
    keywords: [what, how, explain]
    next_stages: [EXECUTION]

  EXECUTION:
    name: EXECUTION
    description: "Performing the task"
    tools: [read_file, write_file, edit_file]
    keywords: [do, create, modify]
    next_stages: [COMPLETION]

  COMPLETION:
    name: COMPLETION
    description: "Task complete"
    tools: []
    keywords: [done, complete]
    next_stages: []

extensions:
  middleware:
    - name: logging_middleware
      class_name: LoggingMiddleware
      module: victor.complete_vertical.middleware
      enabled: true

  safety_patterns:
    - name: dangerous_delete
      pattern: "rm -rf .*"
      description: "Recursive delete"
      severity: critical
      category: commands

  prompt_hints:
    - task_type: create
      hint: "[CREATE] Plan before implementing..."
      tool_budget: 15
      priority_tools: [read_file, write_file]

workflows:
  - name: example_workflow
    file: workflows/example.yaml
    description: "Example workflow"
    enabled: true

teams:
  - name: example_team
    display_name: "Example Team"
    description: "Example team formation"
    formation: parallel
    communication_style: structured
    max_iterations: 3
    roles:
      - name: analyst
        display_name: "Analyst"
        description: "Analyzes requests"
        persona: "You are an analyst..."
        tool_categories: [analysis]
        capabilities: []

capabilities:
  - name: example_capability
    type: workflow
    description: "Example capability"
    enabled: true
    handler: victor.complete_vertical.capabilities:ExampleHandler
    config:
      setting: value

custom_config:
  example_setting: "value"
  grounding_rules: |
    Base responses on tool output.
  system_prompt_section: |
    Follow best practices.

file_templates: {}
```text

## Migration from Legacy Templates

If you have templates from older versions:

1. **Update metadata structure**:
   ```yaml
   # Old
   name: my_vertical
   description: "Description"

   # New
   metadata:
     name: my_vertical
     description: "Description"
   ```

2. **Rename extensions**:
   ```yaml
   # Old
   middleware: []

   # New
   extensions:
     middleware: []
```text

3. **Update stage structure**:
   ```yaml
   # Old
   stages:
     - name: INITIAL
       tools: []

   # New
   stages:
     INITIAL:
       name: INITIAL
       tools: []
   ```

See [Migration Tool Guide](migration_tool_guide.md) for automated migration.

## Best Practices Summary

1. **Use descriptive names** for templates, stages, and roles
2. **Provide clear descriptions** for all components
3. **Start with base template** and extend as needed
4. **Validate templates** before generation
5. **Use template inheritance** to avoid duplication
6. **Document custom configurations** with comments
7. **Keep prompts concise** and focused
8. **Define clear stage transitions**
9. **Specify appropriate tool budgets**
10. **Add safety patterns** for dangerous operations

## Troubleshooting

### Common Errors

#### Missing Required Field

```bash
❌ Validation error: Missing required field: metadata.version
```text

**Solution**: Add all required metadata fields.

#### Invalid Protocol

```bash
❌ Validation error: Invalid protocol: NotARealProtocol
```

**Solution**: Use valid protocol names from reference.

#### Workflow File Not Found

```bash
❌ Validation error: Workflow file not found: workflows/missing.yaml

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 2 min
**Last Updated:** February 08, 2026**
