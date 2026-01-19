# Developer Tools CI/CD Integration

This document describes how to integrate the Phase 4 Developer Tools into CI/CD pipelines.

## Overview

The Phase 4 Developer Tools provide automated quality checks that can be integrated into GitHub Actions, GitLab CI, or other CI/CD systems.

## Tools

1. **Protocol Conformance Checker** (`check_protocol_conformance.py`)
2. **Vertical Linter** (`lint_vertical.py`)
3. **Configuration Validator** (`validate_config.py`)
4. **Performance Profiler** (`profile_coordinators.py`)
5. **Test Coverage Reporter** (`coverage_report.py`)
6. **Documentation Generator** (`generate_docs.py`)

## GitHub Actions Integration

### Pre-commit Checks

Create `.github/workflows/dev-tools-check.yml`:

```yaml
name: Developer Tools Check

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  dev-tools:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Check protocol conformance
        run: |
          python scripts/check_protocol_conformance.py --all-verticals
        continue-on-error: true

      - name: Lint verticals
        run: |
          python scripts/lint_vertical.py --all-verticals
        continue-on-error: true

      - name: Validate configurations
        run: |
          python scripts/validate_config.py --all-configs

      - name: Generate coverage report
        run: |
          python scripts/coverage_report.py --format html --check-goals

      - name: Upload coverage reports
        uses: actions/upload-artifact@v3
        with:
          name: coverage-reports
          path: |
            htmlcov/
            coverage.json
```

### Nightly Checks

Create `.github/workflows/nightly-dev-tools.yml`:

```yaml
name: Nightly Developer Tools

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
  workflow_dispatch:

jobs:
  full-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run all developer tools
        run: |
          make dev-tools

      - name: Profile coordinators
        run: |
          python scripts/profile_coordinators.py --all-coordinators

      - name: Generate documentation
        run: |
          python scripts/generate_docs.py --all --output docs/generated

      - name: Upload profiling results
        uses: actions/upload-artifact@v3
        with:
          name: profiling-results
          path: profiling_results/

      - name: Upload documentation
        uses: actions/upload-artifact@v3
        with:
          name: generated-docs
          path: docs/generated/
```

## GitLab CI Integration

### .gitlab-ci.yml

```yaml
stages:
  - quality
  - test
  - profile

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip
    - venv/

before_script:
  - python -m venv venv
  - source venv/bin/activate
  - pip install -e ".[dev]"

# Developer Tools Checks
dev-tools:
  stage: quality
  script:
    - python scripts/check_protocol_conformance.py --all-verticals
    - python scripts/lint_vertical.py --all-verticals
    - python scripts/validate_config.py --all-configs
  artifacts:
    reports:
      junit: dev-tools-report.xml
  allow_failure: true

# Coverage Report
coverage:
  stage: test
  script:
    - python scripts/coverage_report.py --format html --check-goals
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    paths:
      - htmlcov/
      - coverage.json
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

# Performance Profiling
profile:
  stage: profile
  script:
    - python scripts/profile_coordinators.py --all-coordinators --json-output profiling.json
  artifacts:
    paths:
      - profiling_results/
    reports:
      metrics: profiling.json
  only:
    - schedules
```

## Pre-commit Hooks

### .pre-commit-config.yaml

```yaml
repos:
  - repo: local
    hooks:
      - id: check-protocol
        name: Check Protocol Conformance
        entry: python scripts/check_protocol_conformance.py --all-verticals
        language: system
        pass_filenames: false
        always_run: true

      - id: lint-verticals
        name: Lint Verticals
        entry: python scripts/lint_vertical.py --all-verticals
        language: system
        pass_filenames: false
        always_run: true

      - id: validate-config
        name: Validate Configs
        entry: python scripts/validate_config.py --all-configs
        language: system
        pass_filenames: false
        always_run: true
```

## Exit Codes

All tools use the following exit codes:

- `0`: Success / All checks passed
- `1`: Issues found / Non-compliant
- `2`: Error occurred

This allows CI/CD systems to detect failures:

```bash
python scripts/check_protocol_conformance.py --all-verticals
if [ $? -ne 0 ]; then
  echo "Protocol conformance check failed"
  exit 1
fi
```

## JSON Output

For CI/CD integration, tools support JSON output:

```bash
# Protocol conformance
python scripts/check_protocol_conformance.py --all-verticals --json

# Coverage report
python scripts/coverage_report.py --format json

# Profiling
python scripts/profile_coordinators.py --all-coordinators --json-output profiling.json
```

## Artifacts and Reports

### Coverage Reports

- **HTML**: `htmlcov/index.html` - Interactive HTML report
- **JSON**: `coverage.json` - Machine-readable coverage data
- **Trends**: `coverage_trends.json` - Historical coverage data

### Profiling Results

- **HTML**: `profiling_results/{coordinator}_flamegraph.html`
- **JSON**: `profiling_results/profiling_{timestamp}.json`

### Documentation

- **Markdown**: `docs/generated/*.md`
- **HTML**: `docs/generated/*.html` (if --format html)

## Docker Integration

### Dockerfile for CI

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e ".[dev]"

# Copy source
COPY . .

# Run developer tools
CMD ["python", "scripts/check_protocol_conformance.py", "--all-verticals"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  dev-tools:
    build: .
    volumes:
      - .:/app
      - ./reports:/app/reports
    command: |
      sh -c "
        python scripts/check_protocol_conformance.py --all-verticals &&
        python scripts/lint_vertical.py --all-verticals &&
        python scripts/validate_config.py --all-configs
      "
```

## Notifications

### Slack Integration

Add to GitHub Actions:

```yaml
- name: Notify Slack on failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "Developer tools check failed for ${{ github.repository }}",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "Developer tools check failed\n*Repository:* ${{ github.repository }}\n*Branch:* ${{ github.ref }}\n*Commit:* ${{ github.sha }}"
            }
          }
        ]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

## Best Practices

1. **Run frequently**: Integrate into pull request checks
2. **Fail fast**: Use exit codes to stop CI on critical failures
3. **Artifacts**: Save reports and logs for debugging
4. **Trends**: Track coverage and performance over time
5. **Notifications**: Alert team on critical failures
6. **Caching**: Cache dependencies to speed up CI

## Troubleshooting

### Common Issues

**Issue**: Protocol conformance check fails in CI but passes locally

**Solution**: Ensure all dependencies are installed:
```yaml
- name: Install dependencies
  run: |
    pip install -e ".[dev]"
    pip install -r requirements-dev.txt
```

**Issue**: Profiling takes too long in CI

**Solution**: Limit profiling to specific coordinators:
```bash
python scripts/profile_coordinators.py --coordinator ToolCoordinator --iterations 5
```

**Issue**: Coverage report not generated

**Solution**: Ensure pytest-cov is installed:
```bash
pip install pytest-cov
```

## Maintenance

### Updating Tools

When tools are updated:

1. Update version in `pyproject.toml`
2. Update CI/CD configurations
3. Test locally before committing
4. Monitor CI results

### Adding New Tools

To add a new developer tool:

1. Create script in `scripts/`
2. Add test in `tests/unit/tools/`
3. Add CLI entry point in `pyproject.toml`
4. Add Makefile target
5. Update CI/CD configurations

## Support

For issues or questions about developer tools:

1. Check tool documentation: `python scripts/<tool>.py --help`
2. Review logs in CI/CD artifacts
3. Open an issue on GitHub

## References

- [Phase 4 Task Documentation](../../fep-XXXX-test.md)
- [Developer Tools README](../developer_tools/)
- [CI/CD Best Practices](https://docs.github.com/en/actions)
