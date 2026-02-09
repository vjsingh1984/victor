# Common Workflows Guide - Part 2

**Part 2 of 2:** Documentation, Debugging, Multi-Provider, Automation, and Best Practices

---

## Navigation

- [Part 1: Development Workflows](part-1-development-workflows.md)
- **[Part 2: Advanced Workflows](#)** (Current)
- [**Complete Guide](../COMMON_WORKFLOWS.md)**

---
## Debugging Workflows

### Workflow 13: Bug Diagnosis

**Goal:** Diagnose and fix bugs.

**Steps:**

```bash
# 1. Describe the bug
victor chat "I'm experiencing this bug:
[Bug description]
[Error message]
[Steps to reproduce]
[Expected behavior]
[Actual behavior]"

# 2. Analyze the code
victor chat "Analyze the relevant code to identify the root cause:
- Examine the error stack trace
- Review the code at the error location
- Check related code that might be involved"

# 3. Propose fixes
victor chat "Propose fixes for the bug and:
- Explain the root cause
- Provide code fix
- Suggest tests to prevent regression"

# 4. Verify fix
victor chat "Apply the fix and verify it:
- Explains what was changed
- Runs tests to ensure no regressions
- Verifies the bug is fixed"
```text

**Expected Output:**
- Root cause analysis
- Proposed fixes
- Implemented fix
- Verification tests

### Workflow 14: Performance Debugging

**Goal:** Debug performance issues.

**Steps:**

```bash
# 1. Profile the code
victor chat "Help me profile this slow function:
[Function code]
Identify performance bottlenecks"

# 2. Analyze bottlenecks
victor chat "Analyze the profiling results:
- Identify the slowest operations
- Explain why they're slow
- Suggest optimizations"

# 3. Implement optimizations
victor chat "Implement performance optimizations:
- Apply suggested optimizations
- Maintain code readability
- Add comments explaining changes"

# 4. Benchmark improvements
victor chat "Benchmark before and after performance:
- Measure execution time
- Calculate speedup factor
- Verify correctness maintained"
```

**Expected Output:**
- Profiling analysis
- Bottleneck identification
- Optimized code
- Benchmark results

## Multi-Provider Workflows

### Workflow 15: Cost-Optimized Development

**Goal:** Minimize costs while maintaining quality.

**Steps:**

```bash
# 1. Brainstorm with free model
victor chat --provider ollama "Brainstorm 5 different approaches to implement:
[Feature description]
Focus on creativity and exploration"

# 2. Plan with cheap model
victor chat --provider openai --model gpt-3.5-turbo "Create detailed implementation plan for approach 3:
- Break down into steps
- Identify potential issues
- Suggest testing strategy"

# 3. Implement with fast model
victor chat --provider openai --model gpt-3.5-turbo "Implement the feature following the plan:
- Write clean, production-ready code
- Add comments and documentation
- Follow best practices"

# 4. Review with premium model
victor chat --provider anthropic "Review the implementation for:
- Code quality
- Security issues
- Performance optimization
- Edge cases
- Bug potential"

# 5. Generate tests with free model
victor chat --provider ollama "Generate comprehensive tests for the implementation"
```text

**Expected Output:**
- Multiple design approaches
- Detailed implementation plan
- Production-ready code
- Quality review report
- Test suite

**Cost Savings:**
- Traditional: 100% (using premium model for everything)
- Multi-provider: ~10% (strategic provider selection)
- Savings: 90%

### Workflow 16: Provider Selection Guide

**Goal:** Choose the right provider for each task.

| Task | Best Provider | Reason |
|------|--------------|--------|
| Brainstorming | Ollama (Free) | Creativity, no cost |
| Implementation | GPT-3.5 (Cheap) | Fast, good code gen |
| Code Review | Claude (Premium) | Best reasoning |
| Documentation | Ollama (Free) | Good enough, saves cost |
| Debugging | Claude (Premium) | Excellent analysis |
| Refactoring | GPT-4 (Premium) | Complex reasoning |
| Testing | Ollama (Free) | Straightforward |
| Long Context | Gemini 1.5 Pro | 1M token context |

**Example Workflow:**

```bash
# Step 1: Brainstorm (Free - $0.00)
victor chat --provider ollama "Brainstorm API design for user management"

# Step 2: Plan (Cheap - $0.01)
victor chat --provider openai --model gpt-3.5-turbo "Create detailed implementation plan"

# Step 3: Implement (Cheap - $0.02)
victor chat --provider openai --model gpt-3.5-turbo "Implement the API endpoints"

# Step 4: Review (Premium - $0.05)
victor chat --provider anthropic "Review implementation for quality and security"

# Step 5: Test (Free - $0.00)
victor chat --provider ollama "Generate comprehensive tests"

# Total: $0.08 vs $0.15 (all premium) = 47% savings
```

## Workflow Automation

### Create Reusable Workflows

Save time by creating reusable workflow scripts:

**File: `workflows/feature.sh`**
```bash
#!/bin/bash
FEATURE_NAME=$1

# Brainstorm
victor chat --provider ollama \
  "Brainstorm approaches for $FEATURE_NAME" \
  > brainstorm.md

# Plan
victor chat --provider openai --model gpt-3.5-turbo \
  "Create detailed plan for $FEATURE_NAME based on brainstorm.md" \
  > plan.md

# Implement
victor chat --provider openai --model gpt-3.5-turbo \
  "Implement $FEATURE_NAME following plan.md" \
  > implementation.md

# Review
victor chat --provider anthropic \
  "Review implementation.md for quality" \
  > review.md

# Test
victor chat --provider ollama \
  "Generate tests for $FEATURE_NAME" \
  > tests.md
```text

**Usage:**
```bash
bash workflows/feature.sh "user authentication system"
```

## Best Practices

### 1. Start with Exploration

```bash
# Always explore first
victor chat --mode explore "Explore different approaches for..."
```text

### 2. Use Appropriate Modes

```bash
# PLAN for design
victor chat --mode plan "Design the system"

# BUILD for implementation
victor chat --mode build "Implement the feature"

# EXPLORE for research
victor chat --mode explore "Research options"
```

### 3. Iterate Gradually

```bash
# Don't try to do everything at once
victor chat "Step 1: Create basic structure"
victor chat "Step 2: Add error handling"
victor chat "Step 3: Add logging"
victor chat "Step 4: Add tests"
```text

### 4. Always Test

```bash
# Generate tests for every feature
victor chat "Generate tests for the code above"
```

### 5. Document Continuously

```bash
# Document as you go
victor chat "Add docstrings to all functions"
victor chat "Generate README for this module"
```text

## Conclusion

These workflows provide a starting point for using Victor AI effectively. Adapt them to your specific needs and
  development style. The key is to be iterative,
  use the right provider for each task, and always prioritize code quality.

Happy coding! 🚀

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
