# Victor Developer Tools (Phase 4)

Comprehensive developer tools for improving code quality and productivity in the Victor AI coding assistant project.

## Overview

Phase 4 introduces 6 production-ready developer tools designed to automate quality checks, improve code quality, and streamline the development workflow.

## Tools

### 1. Protocol Conformance Checker

**Script**: `scripts/check_protocol_conformance.py`

**Purpose**: Verify that verticals and components correctly implement their declared protocols.

**Usage**:
```bash
# Check all verticals
python scripts/check_protocol_conformance.py --all-verticals

# Check specific vertical
python scripts/check_protocol_conformance.py --vertical victor/coding

# Check specific protocol
python scripts/check_protocol_conformance.py --vertical victor/coding --protocol ToolProvider

# JSON output
python scripts/check_protocol_conformance.py --all-verticals --json

# Verbose output
python scripts/check_protocol_conformance.py --all-verticals -v
```

**Exit Codes**:
- `0`: All checks passed
- `1`: Violations found
- `2`: Error occurred

**What it checks**:
- Missing protocol methods
- Signature mismatches
- Extra methods (warnings)
- Parameter type compatibility

**Example Output**:
```
✗ NON-COMPLIANT: CodingAssistant -> ToolProvider
================================================================================
Missing Methods (1):
  - get_tool_categories

Signature Mismatches (0):

Extra Methods (2):
  - custom_method

Detailed Violations:
  [ERROR] CodingAssistant -> ToolProvider - Missing required method: get_tool_categories
    Suggestion: Add method: get_tool_categories(self) -> List[str]
```

---

### 2. Vertical Linter

**Script**: `scripts/lint_vertical.py`

**Purpose**: Comprehensive linting for vertical code quality beyond standard linters.

**Usage**:
```bash
# Lint all verticals
python scripts/lint_vertical.py --all-verticals

# Lint specific vertical
python scripts/lint_vertical.py victor/coding

# Auto-fix issues where possible
python scripts/lint_vertical.py victor/coding --fix

# JSON output
python scripts/lint_vertical.py --all-verticals --json
```

**What it checks**:
- **Naming Conventions**: PascalCase classes, snake_case functions/variables
- **Documentation**: Module/class/function docstrings, type hints
- **Protocol Conformance**: Implements required methods
- **Code Style**: Import ordering, line length
- **Security**: Hardcoded secrets, eval() usage
- **Architecture**: Required attributes, proper base classes
- **Best Practices**: Cyclomatic complexity, function length

**Example Output**:
```
VERTICAL LINT REPORT: coding
================================================================================
Files checked: 15
Errors: 3
Warnings: 12
Info: 28

NAMING
--------------------------------------------------------------------------------
[WARNING] victor/coding/my_module.py:10 - Class name should be PascalCase: badClass
  → Use PascalCase for class names

[INFO] victor/coding/my_module.py:20 - Variable name should be snake_case: BadVariable
  → Use snake_case for variable names

DOCUMENTATION
--------------------------------------------------------------------------------
[WARNING] victor/coding/assistant.py:1 - Missing module docstring
  → Add a docstring at the top of the file: """Module description."""

SECURITY
--------------------------------------------------------------------------------
[ERROR] victor/coding/config.py:15 - Hardcoded password
  → Use environment variables or configuration files
```

---

### 3. Configuration Validator

**Script**: `scripts/validate_config.py`

**Purpose**: Validate YAML configuration files for modes, capabilities, and teams.

**Usage**:
```bash
# Validate all configs
python scripts/validate_config.py --all-configs

# Validate specific config
python scripts/validate_config.py victor/config/modes/coding_modes.yaml

# Validate by type
python scripts/validate_config.py --type mode
```

**What it validates**:
- YAML syntax
- Required fields
- Value constraints (enums, ranges, types)
- Cross-file references

**Supported Config Types**:
- `mode`: Mode configurations
- `capability`: Capability definitions
- `team`: Team specifications

**Example Output**:
```
✓ VALID: victor/config/modes/coding_modes.yaml
================================================================================
No issues found

✗ INVALID: victor/config/teams/coding_teams.yaml
================================================================================
[ERROR] victor/config/teams/coding_teams.yaml::teams[0].formation - Invalid value 'invalid_formation'
  → Must be one of: ['pipeline', 'parallel', 'sequential', 'hierarchical', 'consensus']

[ERROR] victor/config/teams/coding_teams.yaml::teams[0].roles[0].persona - Missing required field: persona
  → Add persona to role
```

---

### 4. Performance Profiler

**Script**: `scripts/profile_coordinators.py`

**Purpose**: Profile coordinator performance to identify bottlenecks.

**Usage**:
```bash
# Profile all coordinators
python scripts/profile_coordinators.py --all-coordinators

# Profile specific coordinator
python scripts/profile_coordinators.py --coordinator ToolCoordinator

# Generate flamegraph
python scripts/profile_coordinators.py --coordinator ToolCoordinator --output profile.html

# Set iterations
python scripts/profile_coordinators.py --coordinator ToolCoordinator --iterations 20

# JSON output
python scripts/profile_coordinators.py --all-coordinators --json-output profiling.json
```

**What it measures**:
- Execution time (ms)
- Memory usage (MB)
- CPU percentage
- Function call counts
- Bottleneck identification

**Example Output**:
```
Profiling ToolCoordinator...
================================================================================
Execution Time: 45.23ms
Memory Usage: 125.45MB
CPU Usage: 12.3%
Total Function Calls: 1,234

Top Functions by Cumulative Time:
--------------------------------------------------------------------------------
  tool_selector.select_tools: 234 calls
  tool_executor.execute: 189 calls
  cache.get: 567 calls

Bottlenecks Identified:
--------------------------------------------------------------------------------
  ⚠ Slow function: select_and_execute (0.234s)
  ⚠ High call count in cache operations

Optimization Suggestions:
--------------------------------------------------------------------------------
  💡 Use tool selection caching to reduce repeated lookups
  💡 Implement lazy loading for tool metadata
  💡 Batch tool executions where possible

Flamegraph saved to: profiling_results/ToolCoordinator_flamegraph.html
```

**Requirements**:
```bash
pip install psutil
```

---

### 5. Test Coverage Reporter

**Script**: `scripts/coverage_report.py`

**Purpose**: Enhanced coverage reports with trends and goal enforcement.

**Usage**:
```bash
# Generate coverage report
python scripts/coverage_report.py --format html

# Filter by coordinators
python scripts/coverage_report.py --coordinators

# Filter by vertical
python scripts/coverage_report.py --vertical coding

# Check coverage goals
python scripts/coverage_report.py --check-goals

# Save trends
python scripts/coverage_report.py --save-trends coverage_trends.json

# Custom output
python scripts/coverage_report.py --output my_report.html
```

**What it provides**:
- Overall coverage percentage
- Component breakdown
- Missing lines highlighted
- Coverage trends over time
- Goal compliance checking

**Coverage Goals**:
- Coordinators: 80%
- Protocols: 85%
- Framework: 85%
- Tools: 75%
- Providers: 70%

**Example Output**:
```
Running coverage analysis for: victor
================================================================================
✓ COMPLIANT: victor
================================================================================
Total Statements: 15,234
Covered: 12,567
Missing: 2,667
Coverage: 82.5%

Component Breakdown:
--------------------------------------------------------------------------------
  ✓ victor.protocols.tool_selector: 92.3% (245/265)
  ✓ victor.agent.coordinators.tool_coordinator: 85.1% (423/497)
  ✗ victor.tools.custom_tool: 45.2% (34/75)
      Missing lines: [10, 15, 23, 45, 67, 89, 101, ...]

COVERAGE GOALS
================================================================================
coordinators: ✓ MET (goal: 80%)
protocols: ✓ MET (goal: 85%)
tools: ✗ NOT MET (goal: 75%)

HTML report available at: htmlcov/index.html
```

**Requirements**:
```bash
pip install pytest-cov
```

---

### 6. Documentation Generator

**Script**: `scripts/generate_docs.py`

**Purpose**: Generate comprehensive documentation from source code.

**Usage**:
```bash
# Generate all documentation
python scripts/generate_docs.py --all --output docs/generated

# Generate API docs for specific module
python scripts/generate_docs.py --type api --module victor.protocols --output docs/api/

# Generate coordinator docs
python scripts/generate_docs.py --type coordinators --format markdown

# Generate protocol docs
python scripts/generate_docs.py --type protocols

# Generate vertical docs
python scripts/generate_docs.py --type verticals
```

**What it generates**:
- API documentation from type hints and docstrings
- Protocol documentation
- Coordinator documentation
- Vertical documentation
- Usage examples

**Output Formats**:
- `markdown`: Markdown files (default)
- `html`: HTML files

**Example Output**:
```
Generating protocol docs...
  Wrote: docs/generated/protocols.md

Generating coordinator docs...
  Wrote: docs/generated/coordinators.md

Generating vertical docs...
  Wrote: docs/generated/verticals.md

Generating API docs for: victor.protocols
  Wrote: docs/generated/victor_protocols.md

Documentation generated in: docs/generated
```

---

## Installation

All tools are included with Victor. No additional installation required.

### CLI Entry Points

After installing Victor, tools are available as CLI commands:

```bash
# Protocol conformance
victor-check-protocol --all-verticals

# Vertical linting
victor-lint-vertical --all-verticals

# Config validation
victor-validate-config --all-configs

# Coordinator profiling
victor-profile-coordinators --all-coordinators

# Coverage reporting
victor-coverage-report --format html

# Documentation generation
victor-generate-docs --all
```

## Makefile Targets

Convenience targets are available in the Makefile:

```bash
# Run all developer tools
make dev-tools

# Individual tools
make check-protocol
make lint-vertical
make validate-config
make profile-coordinators
make coverage-report
make generate-docs
```

## CI/CD Integration

See [CI/CD Integration Guide](developer_tools_cicd.md) for detailed instructions on integrating these tools into your CI/CD pipeline.

### GitHub Actions Example

```yaml
- name: Run Developer Tools
  run: |
    make dev-tools
```

## Testing

Tests for developer tools are located in:

```
tests/unit/tools/
├── test_check_protocol_conformance.py
└── test_lint_vertical.py
```

Run tests:

```bash
pytest tests/unit/tools/ -v
```

## Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure Victor is installed in development mode:
```bash
pip install -e .
```

### Issue: Permission denied when running scripts

**Solution**: Make scripts executable:
```bash
chmod +x scripts/*.py
```

### Issue: Coverage report not generated

**Solution**: Install pytest-cov:
```bash
pip install pytest-cov
```

### Issue: Profiling tools not available

**Solution**: Install psutil:
```bash
pip install psutil
```

## Best Practices

1. **Run frequently**: Integrate into pre-commit hooks and CI/CD
2. **Fix issues promptly**: Address violations as they're found
3. **Monitor trends**: Track coverage and performance over time
4. **Update documentation**: Regenerate docs after API changes
5. **Review reports**: Check detailed reports for insights

## Contributing

When adding new developer tools:

1. Create script in `scripts/`
2. Add tests in `tests/unit/tools/`
3. Add CLI entry point in `pyproject.toml`
4. Add Makefile target
5. Update documentation
6. Add CI/CD integration

## Support

For issues or questions:

1. Check tool help: `python scripts/<tool>.py --help`
2. Review this documentation
3. Open an issue on GitHub

## References

- [Phase 4 Task Documentation](../fep-XXXX-test.md)
- [CI/CD Integration Guide](developer_tools_cicd.md)
- [Victor Architecture](../CLAUDE.md)

## License

Apache License 2.0

## Authors

Vijaykumar Singh <singhvjd@gmail.com>

## Version

Phase 4 - Released 2025-01-18
