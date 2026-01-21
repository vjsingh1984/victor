# Victor AI Recipe Book

**Copy-paste ready solutions for common tasks**

---

## Table of Contents

1. [Code Review Recipes](#code-review-recipes)
2. [Documentation Recipes](#documentation-recipes)
3. [Testing Recipes](#testing-recipes)
4. [Refactoring Recipes](#refactoring-recipes)
5. [Debugging Recipes](#debugging-recipes)
6. [Git Recipes](#git-recipes)
7. [API Development Recipes](#api-development-recipes)
8. [Data Analysis Recipes](#data-analysis-recipes)
9. [Research Recipes](#research-recipes)
10. [Automation Recipes](#automation-recipes)

---

## Code Review Recipes

### Recipe 1: Comprehensive Code Review

**Goal**: Thoroughly review code for quality, security, and performance.

```bash
victor chat --mode plan "
Perform a comprehensive code review of the current directory:

1. Security Analysis
   - Check for SQL injection vulnerabilities
   - Look for XSS vulnerabilities
   - Review authentication/authorization
   - Check for hardcoded secrets
   - Validate input sanitization

2. Performance Analysis
   - Identify inefficient algorithms
   - Check for N+1 queries
   - Review memory usage patterns
   - Look for resource leaks

3. Code Quality
   - Check SOLID principles adherence
   - Review naming conventions
   - Assess code complexity
   - Check for code duplication

4. Best Practices
   - PEP 8 compliance for Python
   - Error handling completeness
   - Logging appropriateness
   - Type hints coverage

Provide a prioritized list of issues with severity ratings (Critical/High/Medium/Low)
and specific code examples for fixes.
"
```

### Recipe 2: Security-Focused Review

**Goal**: Focus exclusively on security vulnerabilities.

```bash
victor chat --provider anthropic --model claude-sonnet-4-5 "
Review this codebase for security vulnerabilities:

Check for:
- OWASP Top 10 vulnerabilities
- Injection attacks (SQL, NoSQL, OS, LDAP)
- Authentication and authorization flaws
- Sensitive data exposure
- Cryptographic failures
- Insecure dependencies
- Security misconfigurations

For each vulnerability found:
1. Explain the risk
2. Show the vulnerable code
3. Provide a secure alternative
4. Explain the fix

Focus on actionable recommendations with code examples.
"
```

### Recipe 3: Performance Review

**Goal**: Identify performance bottlenecks and optimization opportunities.

```bash
victor chat "
Analyze this code for performance issues:

1. Algorithm Analysis
   - Identify inefficient algorithms (O(n²) or worse)
   - Suggest more efficient alternatives
   - Show Big O complexity comparison

2. Database Operations
   - Check for N+1 query problems
   - Suggest appropriate indexes
   - Recommend query optimizations

3. Memory Usage
   - Identify memory leaks
   - Suggest memory-efficient alternatives
   - Recommend caching strategies

4. I/O Operations
   - Identify blocking I/O
   - Suggest async alternatives
   - Recommend batching strategies

Provide before/after code examples with performance metrics.
"
```

### Recipe 4: Pull Request Review

**Goal**: Review a specific pull request or branch.

```bash
victor chat "
Review the changes in this pull request:

git diff main..feature-branch

Focus on:
1. Logic correctness
2. Edge cases handling
3. Error handling
4. Test coverage
5. Documentation updates
6. Breaking changes
7. Backward compatibility

Provide:
- Summary of changes
- List of concerns (if any)
- Suggested improvements
- Approval decision (LGTM/Request Changes)
"
```

---

## Documentation Recipes

### Recipe 5: Generate API Documentation

**Goal**: Generate comprehensive API documentation from code.

```bash
victor chat "
Generate comprehensive API documentation for this module:

For each function/class:
1. Add Google-style docstring with:
   - Brief description
   - Detailed description
   - Args section with types and descriptions
   - Returns section with type and description
   - Raises section with exceptions
   - Examples section with usage

2. Create API reference document:
   - Overview
   - Installation/Setup
   - Quick start guide
   - API endpoints/functions
   - Request/response examples
   - Error handling
   - Rate limiting (if applicable)

3. Generate usage examples for common scenarios

Output in Markdown format ready for documentation site.
"
```

### Recipe 6: Add Type Hints and Docstrings

**Goal**: Add type hints and docstrings to existing code.

```bash
victor chat "
Update this Python code with:

1. Type Hints
   - Add type hints to all function parameters
   - Add return type annotations
   - Use typing module for complex types (List, Dict, Optional, etc.)
   - Create TypeAlias for complex types
   - Use TypedDict for data models

2. Docstrings
   - Add Google-style docstrings to all functions
   - Include parameter descriptions
   - Include return value descriptions
   - Include raised exceptions
   - Add usage examples

3. Enable mypy validation
   - Ensure code passes mypy strict mode
   - Fix any type errors

Maintain existing functionality - only add types and documentation.
"
```

### Recipe 7: Generate README

**Goal**: Create a comprehensive README for a project.

```bash
victor chat "
Generate a comprehensive README.md for this project:

Include:
1. Project title and brief description
2. Features list
3. Installation instructions
4. Quick start guide
5. Usage examples
6. Configuration options
7. API documentation link
8. Contributing guidelines
9. License information
10. Changelog/Version history

Use badges where appropriate:
- Build status
- Version
- License
- Python version
- Coverage

Make it beginner-friendly and comprehensive.
"
```

### Recipe 8: Generate Architecture Documentation

**Goal**: Document system architecture and design decisions.

```bash
victor chat "
Generate architecture documentation for this project:

1. System Overview
   - High-level architecture diagram (ASCII)
   - Main components and their responsibilities
   - Data flow between components

2. Design Decisions
   - Technology choices with rationale
   - Trade-offs considered
   - Alternatives evaluated

3. Component Details
   - Each component's purpose
   - Interfaces/contracts
   - Dependencies

4. Deployment Architecture
   - Production setup
   - Scalability considerations
   - High availability design

5. Development Workflow
   - Local development setup
   - Testing strategy
   - CI/CD pipeline

Include diagrams using ASCII art or Mermaid syntax.
"
```

---

## Testing Recipes

### Recipe 9: Generate Unit Tests

**Goal**: Generate comprehensive unit tests for code.

```bash
victor chat "
Generate comprehensive unit tests for this code using pytest:

For each function/class:
1. Test happy path (normal operation)
2. Test edge cases
3. Test error conditions
4. Test boundary conditions
5. Test with various input types

Requirements:
- Use pytest framework
- Use fixtures for setup/teardown
- Use parametrize for data-driven tests
- Use mocks for external dependencies
- Aim for 90%+ code coverage
- Follow AAA pattern (Arrange, Act, Assert)
- Include descriptive test names
- Add docstrings to tests explaining what is tested

Organize tests by:
- test/unit/ for unit tests
- test/integration/ for integration tests
- test/fixtures/ for shared fixtures

Run pytest with coverage report and show results.
"
```

### Recipe 10: Generate Integration Tests

**Goal**: Create integration tests for the full system.

```bash
victor chat "
Generate integration tests for this application:

Test scenarios:
1. End-to-end user workflows
2. Database interactions
3. API integrations
4. External service calls
5. File I/O operations

Requirements:
- Use pytest with pytest-asyncio for async tests
- Use testcontainers for real database/services
- Clean up test data after each test
- Use fixtures for test setup
- Mock external APIs that have rate limits
- Test error scenarios (network failures, timeouts)
- Include performance assertions

Organize tests by feature/module.
"
```

### Recipe 11: Test-Driven Development (TDD)

**Goal**: Write tests first, then implementation.

```bash
victor chat "
Help me practice Test-Driven Development for: [feature description]

Step 1: Write failing tests first
- Create test file with comprehensive test cases
- Define the interface/contract
- Run tests and confirm they fail

Step 2: Write minimal implementation
- Implement just enough to pass tests
- Don't add extra features yet

Step 3: Refactor
- Improve code quality while keeping tests green
- Extract common patterns
- Apply design patterns

Step 4: Repeat
- Add more failing tests for new features
- Implement to pass
- Refactor

Guide me through each step, waiting for my confirmation before proceeding.
"
```

### Recipe 12: Property-Based Testing

**Goal**: Generate property-based tests using Hypothesis.

```bash
victor chat "
Generate property-based tests for this code using Hypothesis:

Identify properties that should always hold true:
1. Invariants (things that never change)
2. Idempotence (operation can be applied multiple times)
3. Commutativity (order doesn't matter)
4. Associativity (grouping doesn't matter)
5. Round-trip properties (serialize/deserialize)

For each property:
- Write a Hypothesis test
- Define strategies for generating test data
- Include specific examples in docstring
- Explain what property is being tested

Example:
```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    assert add(a, b) == add(b, a)
```
"
```

---

## Refactoring Recipes

### Recipe 13: Apply SOLID Principles

**Goal**: Refactor code to follow SOLID principles.

```bash
victor chat "
Refactor this code to follow SOLID principles:

1. Single Responsibility Principle
   - Identify classes/functions with multiple responsibilities
   - Split into focused, single-purpose components
   - Each class should have one reason to change

2. Open/Closed Principle
   - Identify code that needs modification for extensions
   - Use abstraction (interfaces/base classes)
   - Make it open for extension, closed for modification

3. Liskov Substitution Principle
   - Check inheritance relationships
   - Ensure subclasses can replace base classes
   - Fix any violations

4. Interface Segregation Principle
   - Identify bloated interfaces
   - Split into focused, role-specific interfaces
   - Clients should only depend on what they use

5. Dependency Inversion Principle
   - Depend on abstractions, not concretions
   - Use dependency injection
   - Invert dependencies using protocols/interfaces

For each principle:
1. Identify violations in the code
2. Explain why it's a violation
3. Provide refactored code
4. Explain the benefits

Maintain all existing functionality and tests.
"
```

### Recipe 14: Extract to Design Patterns

**Goal**: Refactor code to use appropriate design patterns.

```bash
victor chat "
Refactor this code to use appropriate design patterns:

Analyze the code and suggest patterns that would improve:
- Maintainability
- Extensibility
- Testability
- Readability

Common patterns to consider:
- Creational: Factory, Builder, Singleton, Prototype
- Structural: Adapter, Bridge, Composite, Decorator, Facade, Proxy
- Behavioral: Strategy, Observer, Command, Chain of Responsibility, State

For each suggested pattern:
1. Explain the problem it solves
2. Show the before/after code
3. Explain the benefits
4. Note any trade-offs

Implement the most beneficial patterns.
"
```

### Recipe 15: Improve Error Handling

**Goal**: Add comprehensive error handling.

```bash
victor chat "
Improve error handling in this code:

1. Add Specific Exceptions
   - Create custom exception classes
   - Use descriptive exception names
   - Include relevant context in exceptions

2. Handle Expected Errors
   - Use try/except for recoverable errors
   - Handle specific exceptions, not bare except
   - Provide helpful error messages
   - Log errors appropriately

3. Validate Inputs
   - Add input validation at function boundaries
   - Raise ValueError with helpful messages
   - Use pydantic for complex validation

4. Use Context Managers
   - Ensure resources are cleaned up
   - Handle cleanup errors

5. Follow EAFP vs LBYL
   - Prefer EAFP (Easier to Ask Forgiveness than Permission)
   - Use LBYL (Look Before You Leap) when appropriate

Example:
```python
# Before
def divide(a, b):
    return a / b

# After
def divide(a: float, b: float) -> float:
    \"\"\"Divide two numbers.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient

    Raises:
        ValueError: If b is zero
        TypeError: If inputs are not numbers
    \"\"\"
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError(\"Both arguments must be numbers\")
    if b == 0:
        raise ValueError(\"Cannot divide by zero\")
    return a / b
```
"
```

### Recipe 16: Extract to Functions/Classes

**Goal**: Break down long functions into smaller, focused units.

```bash
victor chat "
Refactor this code by extracting smaller functions/classes:

Analysis:
1. Identify long functions (>20 lines)
2. Find repeated logic
3. Locate complex conditional logic
4. Spot mixed abstraction levels

Refactoring steps:
1. Extract meaningful blocks into named functions
2. Use descriptive function names (verbs)
3. Each function should do one thing well
4. Functions should be at the same abstraction level
5. Extract related data and methods into classes

For each extraction:
- Show the original code
- Show the refactored code
- Explain the improvement
- Ensure all tests still pass

Follow the 'Extract Method' refactoring pattern.
"
```

---

## Debugging Recipes

### Recipe 17: Debug Failing Tests

**Goal**: Debug and fix failing test cases.

```bash
victor chat "
Help me debug failing tests:

Run the failing test and:
1. Analyze the error message
2. Examine the test code
3. Check the implementation code
4. Identify the root cause
5. Suggest fixes
6. Verify the fix

Test output:
[paste pytest output]

Debugging approach:
- Read error message carefully
- Check assertions
- Verify test setup
- Examine test data
- Check for side effects
- Look for timing issues
- Verify environment/state

Provide step-by-step debugging guidance.
"
```

### Recipe 18: Debug Production Issue

**Goal**: Debug issues reported from production.

```bash
victor chat "
Help me debug this production issue:

Issue description:
[paste issue details]

Error logs:
[paste error logs]

Environment:
- Production vs development differences
- Configuration values
- Data volumes
- Concurrent users

Debugging strategy:
1. Reproduce locally if possible
2. Add logging to trace execution
3. Check for race conditions
4. Verify assumptions
5. Check resource constraints
6. Review recent changes
7. Check for edge cases

Help me:
- Identify potential causes
- Suggest debugging steps
- Add appropriate logging
- Create a fix
- Add tests to prevent regression
"
```

### Recipe 19: Memory Leak Debugging

**Goal**: Debug and fix memory leaks.

```bash
victor chat "
Help me debug a memory leak:

Symptoms:
[describe memory usage pattern]

Debugging approach:
1. Use memory profiler to identify hotspots
2. Check for:
   - Unclosed resources (files, connections)
   - Circular references
   - Global variables growing unbounded
   - Caches without limits
   - Event listeners not removed
   - Static variables accumulating data

3. Add memory profiling code:
```python
import tracemalloc

tracemalloc.start()
# ... code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

4. Generate object referrer graphs
5. Check for common leak patterns
6. Implement fixes
7. Verify with long-running tests

Provide specific fixes for identified issues.
"
```

### Recipe 20: Performance Debugging

**Goal**: Debug and fix performance issues.

```bash
victor chat "
Help me debug performance issues:

Symptoms:
[describe performance problem - slow response, high CPU, etc.]

Profiling approach:
1. Profile the code using cProfile:
```bash
python -m cProfile -o profile.stats your_script.py
python -m pstats profile.stats
> stats 10  # Top 10 functions
> callers slow_function  # Who calls it
```

2. Use line profiler for hotspot analysis:
```python
@profile
def slow_function():
    # ...
```

3. Check for:
   - Inefficient algorithms (nested loops)
   - Repeated expensive operations
   - Missing caching
   - Blocking I/O
   - Database N+1 queries
   - Unnecessary data copying

4. Provide before/after timing comparisons

5. Suggest specific optimizations with code examples

Focus on the slowest functions first (Pareto principle).
"
```

---

## Git Recipes

### Recipe 21: Generate Commit Message

**Goal**: Generate a conventional commit message.

```bash
victor chat "
Generate a conventional commit message for these changes:

git diff

Requirements:
- Use Conventional Commits format
- Format: <type>(<scope>): <description>
- Types: feat, fix, docs, style, refactor, test, chore
- Include body with details
- Include breaking changes section if applicable
- Keep description under 72 characters
- Use imperative mood ('add' not 'added')
- Explain why, not what

Example:
```
feat(auth): add JWT token refresh mechanism

Implement automatic token refresh to improve user experience.
Tokens are refreshed 5 minutes before expiration.

- Add refresh endpoint
- Implement token rotation
- Update token validation logic

Closes #123
```
"
```

### Recipe 22: Review Git History

**Goal**: Analyze git history for insights.

```bash
victor chat "
Analyze the git history of this repository:

git log --oneline --graph --all -50

Provide:
1. Development activity summary
   - Most active contributors
   - Commit frequency patterns
   - Branching strategy used

2. Code evolution insights
   - Major feature additions
   - Refactoring efforts
   - Bug fix patterns

3. Identify:
   - Potential technical debt indicators
   - Commit message quality issues
   - Branching patterns
   - Release patterns

4. Recommendations for:
   - Improving commit practices
   - Better branching strategy
   - Code review process
"
```

### Recipe 23: Resolve Merge Conflicts

**Goal**: Get help resolving merge conflicts.

```bash
victor chat "
Help me resolve merge conflicts:

Conflict details:
[paste git conflict output]

Conflicting file:
[paste file with conflict markers]

For each conflict:
1. Analyze both sides of the conflict
2. Understand the intent of each change
3. Suggest a resolution strategy:
   - Keep both changes
   - Choose one side
   - Combine both changes
   - Manual resolution needed
4. Provide the resolved code
5. Explain the resolution

Best practices:
- Preserve the intent of both changes
- Test after resolution
- Communicate with contributors if unsure
- Use merge tools for complex conflicts
"
```

---

## API Development Recipes

### Recipe 24: Design REST API

**Goal**: Design a RESTful API from requirements.

```bash
victor chat "
Design a RESTful API for: [requirements]

Provide:

1. API Specification
   - Endpoint list with methods
   - Request/response schemas
   - Status codes
   - Authentication requirements

2. OpenAPI/Swagger specification:
```yaml
openapi: 3.0.0
info:
  title: API Name
  version: 1.0.0
paths:
  /users:
    get:
      summary: List users
      responses:
        '200':
          description: Success
```

3. Implementation guide for FastAPI:
   - Route definitions
   - Pydantic models
   - Dependencies
   - Error handlers

4. Best practices:
   - RESTful naming conventions
   - Versioning strategy
   - Pagination
   - Filtering and sorting
   - HATEOAS links (optional)

5. Security considerations
"
```

### Recipe 25: Add API Authentication

**Goal**: Add authentication to an API.

```bash
victor chat "
Add authentication to this FastAPI application:

Requirements:
- JWT-based authentication
- User registration
- Login/logout
- Token refresh
- Password reset
- Role-based access control

Implementation:
1. Database models for users
2. Password hashing (bcrypt)
3. JWT token generation/validation
4. Protected route decorators
5. Refresh token mechanism
6. Password reset flow

Security best practices:
- Use strong password hashing
- Implement rate limiting
- Store tokens securely
- Set appropriate token expiration
- Use HTTPS only
- Implement CORS properly
- Validate and sanitize inputs

Provide complete, production-ready code with tests.
"
```

### Recipe 26: Add API Rate Limiting

**Goal**: Add rate limiting to prevent abuse.

```bash
victor chat "
Add rate limiting to this FastAPI application:

Requirements:
- Rate limit by user/API key
- Different limits for different tiers
- Redis-backed for distributed systems
- Graceful degradation
- Rate limit headers in response

Implementation using slowapi:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api")
@limiter.limit("60/minute")
async def endpoint():
    return {"message": "Hello"}
```

Provide:
1. Rate limit strategies
2. Configuration options
3. Redis integration
4. Custom limit functions
5. Response headers
6. Error handling
7. Testing strategy
"
```

---

## Data Analysis Recipes

### Recipe 27: Exploratory Data Analysis

**Goal**: Perform exploratory data analysis with pandas.

```bash
victor chat "
Perform exploratory data analysis on this dataset:

File: [dataset.csv]

Generate Python code using pandas/seaborn/matplotlib to:

1. Load and inspect data
   - Data shape and types
   - Missing values
   - Basic statistics

2. Univariate analysis
   - Distribution plots for numeric columns
   - Value counts for categorical columns
   - Outlier detection

3. Bivariate analysis
   - Correlation matrix heatmap
   - Scatter plots for relationships
   - Box plots by category

4. Multivariate analysis
   - Pair plots
   - Faceted visualizations

5. Summary report
   - Key insights
   - Data quality issues
   - Recommendations

Provide Jupyter notebook-ready code with explanations.
"
```

### Recipe 28: Data Cleaning Pipeline

**Goal**: Create a data cleaning pipeline.

```bash
victor chat "
Create a data cleaning pipeline for: [dataset description]

Common cleaning tasks:
1. Handle missing values
   - Drop columns with too many missing
   - Impute numeric columns (mean/median/mode)
   - Flag imputed values

2. Remove duplicates
   - Identify duplicate rows
   - Decide on strategy (keep first/last)
   - Report duplicates removed

3. Fix data types
   - Convert to correct types
   - Parse dates
   - Clean strings

4. Handle outliers
   - Detect outliers (IQR, z-score)
   - Cap/floor/remove outliers
   - Document decisions

5. Standardize formats
   - Phone numbers
   - Addresses
   - Names

6. Validate data
   - Business rules
   - Data constraints
   - Referential integrity

Provide reusable Python functions for each task.
"
```

### Recipe 29: Create Data Visualization Dashboard

**Goal**: Create an interactive visualization dashboard.

```bash
victor chat "
Create an interactive data visualization dashboard:

Data: [describe data]

Use Streamlit for rapid dashboard development:

Features:
1. Data upload interface
2. Interactive filters
3. Multiple chart types:
   - Line charts for trends
   - Bar charts for comparisons
   - Scatter plots for relationships
   - Heatmaps for correlations
   - Maps for geospatial data

4. Layout:
   - Sidebar for controls
   - Main area for visualizations
   - Tabs for different views

5. Export functionality:
   - Download charts as PNG
   - Export data as CSV

Provide complete streamlit app code with:
- Configuration
- Styling
- Performance optimizations
"
```

---

## Research Recipes

### Recipe 30: Research Best Practices

**Goal**: Research and document best practices for a topic.

```bash
victor chat "
Research and summarize best practices for: [topic]

Provide:
1. Overview of the topic
2. Key principles
3. Common patterns
4. Anti-patterns to avoid
5. Industry standards
6. Tool recommendations
7. Example implementations
8. References and further reading

Use web search to find current information from:
- Official documentation
- Reputable blogs
- Stack Overflow consensus
- Research papers
- Industry experts

Organize findings in a structured Markdown document.
"
```

### Recipe 31: Compare Technologies

**Goal**: Compare different technologies/libraries.

```bash
victor chat "
Compare these technologies for [use case]:
- Option A: [technology A]
- Option B: [technology B]
- Option C: [technology C]

Provide:
1. Feature comparison table
2. Performance benchmarks (if available)
3. Learning curve
4. Community support
5. Documentation quality
6. Maintenance status
7. Use cases for each
8. Pros and cons
9. Recommendation with rationale

Research current information and provide specific examples.
"
```

### Recipe 32: Create Learning Roadmap

**Goal**: Create a structured learning path for a technology.

```bash
victor chat "
Create a comprehensive learning roadmap for: [technology/skill]

Include:
1. Prerequisites
2. Learning phases:
   - Phase 1: Fundamentals (1-2 weeks)
   - Phase 2: Intermediate (2-4 weeks)
   - Phase 3: Advanced (4-8 weeks)
   - Phase 4: Mastery (ongoing)

3. For each phase:
   - Learning objectives
   - Recommended resources (free/paid)
   - Hands-on projects
   - Practice exercises
   - Assessment criteria

4. Projects:
   - Beginner project ideas
   - Intermediate project ideas
   - Advanced project ideas
   - Portfolio-worthy projects

5. Common pitfalls and how to avoid them
6. Time estimates
7. Next steps after completion

Provide links to high-quality resources.
"
```

---

## Automation Recipes

### Recipe 33: Automate Repetitive Tasks

**Goal**: Create automation scripts for repetitive tasks.

```bash
victor chat "
Create an automation script for: [describe repetitive task]

Task requirements:
- Input/output specifications
- Error handling
- Logging
- Configuration
- Idempotency (can run multiple times safely)

Provide:
1. Python script using click/CLEO for CLI
2. Configuration file (YAML/TOML)
3. Logging setup
4. Error handling
5. Progress indicators
6. Documentation (README)
7. Installation instructions
8. Usage examples

Make it:
- Easy to use
- Well documented
- Robust
- Maintainable
- Testable
"
```

### Recipe 34: Create CI/CD Pipeline

**Goal**: Set up automated testing and deployment.

```bash
victor chat "
Create a CI/CD pipeline for this project:

Requirements:
- Run tests on every commit
- Build and deploy on main branch
- Run security scans
- Generate coverage reports

GitHub Actions workflow:
```yaml
name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      - name: Run tests
        run: pytest
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

Provide complete workflow files for:
- Linting and formatting checks
- Security scanning
- Dependency updates
- Automated releases
"
```

### Recipe 35: Schedule Automated Tasks

**Goal**: Set up scheduled jobs for periodic tasks.

```bash
victor chat "
Create scheduled tasks for: [describe periodic tasks]

Options to provide:
1. Cron jobs (Linux/Mac)
2. Windows Task Scheduler
3. Python APScheduler
4. GitHub Actions scheduled workflows
5. Cloud cron (AWS EventBridge, Google Cloud Scheduler)

For each approach:
- Configuration example
- Setup instructions
- Monitoring/logging
- Error handling
- Retry logic

Best practices:
- Idempotency
- Timezone handling
- Overlap prevention
- Failure notifications
- Run history tracking

Provide production-ready examples.
"
```

---

## Bonus Recipes

### Recipe 36: Migrate Code Base

**Goal**: Migrate code from one framework/version to another.

```bash
victor chat "
Create a migration plan for: [source] to [destination]

Example: Flask to FastAPI, Python 3.8 to 3.12, etc.

Provide:
1. Pre-migration checklist
2. Step-by-step migration guide
3. Code transformation examples
4. Breaking changes and how to handle
5. Testing strategy
6. Rollback plan
7. Post-migration tasks

For each step:
- What to do
- Why it's necessary
- How to verify
- Common pitfalls

Make migration incremental and reversible.
"
```

### Recipe 37: Optimize Database Queries

**Goal**: Identify and fix slow database queries.

```bash
victor chat "
Optimize database queries in this code:

1. Identify slow queries:
   - Add query logging
   - Use EXPLAIN ANALYZE
   - Check for N+1 problems

2. Optimization strategies:
   - Add appropriate indexes
   - Use query optimization (JOIN vs subquery)
   - Implement pagination
   - Cache frequently accessed data
   - Use bulk operations
   - Denormalize if needed

3. Before/after comparisons:
   - Query execution time
   - Database load
   - Response time improvements

Provide specific SQL and code optimizations.
"
```

### Recipe 38: Set Up Monitoring

**Goal**: Implement application monitoring and alerting.

```bash
victor chat "
Set up monitoring for this application:

Metrics to collect:
1. Application metrics:
   - Request rate
   - Response time
   - Error rate
   - Active users

2. System metrics:
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network I/O

3. Business metrics:
   - Conversions
   - User engagement
   - Revenue

Tools:
- Prometheus for metrics collection
- Grafana for visualization
- AlertManager for alerting

Implementation:
```python
from prometheus_client import Counter, Histogram, start_http_server

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.get("/api")
@request_duration.time()
def endpoint():
    request_count.inc()
    return {"message": "Hello"}
```

Provide:
- Complete setup instructions
- Dashboard configurations
- Alert rules
- Best practices
"
```

---

## Quick Reference

### Common Patterns

**File Reading:**
```bash
victor chat "Read and summarize: filename.py"
```

**Code Generation:**
```bash
victor chat "Generate a Python class for: [description]"
```

**Debugging:**
```bash
victor chat "Debug this error: [paste error]"
```

**Testing:**
```bash
victor chat "Write tests for: [file/function]"
```

**Documentation:**
```bash
victor chat "Add docstrings to: [file]"
```

**Refactoring:**
```bash
victor chat "Refactor this following SOLID principles: [paste code]"
```

---

## Tips for Best Results

1. **Be Specific**: Provide context, requirements, and constraints
2. **Show Examples**: Include sample input/output
3. **Iterate**: Refine prompts based on results
4. **Use Appropriate Modes**: plan for analysis, build for implementation
5. **Leverage Providers**: Switch providers based on task
6. **Save Good Workflows**: Reuse effective prompts and workflows
7. **Review Outputs**: Always review and test generated code
8. **Version Control**: Commit before major refactors

---

## Conclusion

These recipes provide copy-paste ready solutions for common tasks. Modify them to fit your specific needs and context.

**Need more recipes?** Check out:
- [User Guide](docs/user-guide/index.md)
- [Examples Directory](examples/)
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)

---

**Version**: 0.5.1
**Last Updated**: January 20, 2026
