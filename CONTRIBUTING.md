<div align="center">

# Contributing to Victor

</div>

Thank you for your interest in contributing to Victor!

## Contribution Workflow

```mermaid
graph LR
    A[Fork] --> B[Branch]
    B --> C[Code]
    C --> D[Test]
    D --> E[Lint]
    E --> F[PR]
    F --> G[Review]
    G --> H[Merge]
```

---

## Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/victor.git
cd victor

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# 3. Create branch
git checkout -b feature/your-feature-name
```

---

## Commands Reference

| Task | Command |
|------|---------|
| Run tests | `pytest` |
| Single test | `pytest tests/unit/test_X.py::test_name` |
| Coverage | `pytest --cov --cov-report=html` |
| Format | `black victor tests` |
| Lint | `ruff check victor tests` |
| Type check | `mypy victor` |
| All checks | `black . && ruff check . && mypy victor && pytest` |

---

## Adding Components

### New Provider

1. Create `victor/providers/my_provider.py`
2. Inherit from `BaseProvider`
3. Implement: `chat()`, `stream_chat()`, `supports_tools()`
4. Register in `ProviderRegistry`
5. Add tests in `tests/unit/providers/`
6. Update documentation

### New Tool

1. Create `victor/tools/my_tool.py`
2. Inherit from `BaseTool`
3. Define `name`, `description`, `parameters`, `cost_tier`
4. Implement `execute()` method
5. Add tests
6. Run `python scripts/generate_tool_catalog.py`

---

## Code Style

| Requirement | Details |
|-------------|---------|
| Type hints | All public APIs |
| Docstrings | Google style |
| Line length | 100 chars (black enforced) |
| I/O | Async/await |
| HTTP mocking | Use `respx` |

---

## PR Checklist

| Check | Command | Required |
|-------|---------|:--------:|
| Tests pass | `pytest` | Yes |
| Formatted | `black --check .` | Yes |
| No lint errors | `ruff check .` | Yes |
| Types valid | `mypy victor` | Yes |
| Docs updated | Manual | If applicable |

---

## PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
```

---

## Areas We Need Help With

| Priority | Areas |
|----------|-------|
| **High** | Provider integrations, tool capabilities, performance |
| **Medium** | Examples, integration tests, CI/CD, Docker |
| **Good First** | Bug fixes, doc typos, test coverage |

---

## Questions?

- **Bugs/Features**: [GitHub Issues](https://github.com/vjsingh1984/victor/issues)
- **Questions**: [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions)

All contributors will be credited in CONTRIBUTORS.md and release notes.

Thank you for contributing to Victor!
