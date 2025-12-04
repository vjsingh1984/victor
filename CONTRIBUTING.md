<div align="center">

![Victor Logo](./assets/victor-logo.png)

# Contributing to Victor

</div>

Thank you for your interest in contributing to Victor! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Be respectful, constructive, and collaborative. We're building something useful together.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/victor.git
cd victor
```

### 2. Set Up Development Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 2.5 Repository Layout

- Active code lives under `victor/` (providers, tools, orchestrator, CLI).
- Prefer `docs/` and `docker/` as canonical homes for documentation.

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Development Workflow

### Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/unit/test_providers.py::test_ollama_provider

# With coverage
pytest --cov --cov-report=html
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black victor tests

# Lint
ruff check victor tests

# Type check
mypy victor

# Run all checks
black . && ruff check . && mypy victor && pytest
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code before commits:

```bash
pip install pre-commit
pre-commit install
```

## Contribution Guidelines

### Adding a New Provider

1. Create a new file in `victor/providers/`
2. Inherit from `BaseProvider`
3. Implement required methods: `chat()`, `stream()`, `supports_tools()`
4. Add configuration to example profiles
5. Write tests in `tests/unit/providers/`
6. Update documentation

Example:

```python
from victor.providers.base import BaseProvider

class MyProvider(BaseProvider):
    async def chat(self, messages, **kwargs):
        # Implementation
        pass
```

### Adding a New Tool

1. Create tool in `victor/tools/`
2. Inherit from `BaseTool`
3. Define JSON schema
4. Implement `execute()` method
5. Add tests
6. Update documentation

### Code Style

- Use type hints for all function signatures
- Write docstrings for public APIs (Google style)
- Keep functions focused and small
- Prefer composition over inheritance
- Use async/await for I/O operations

Example:

```python
async def fetch_completion(
    self,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> CompletionResponse:
    """Fetch completion from the LLM provider.

    Args:
        prompt: The input prompt
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate

    Returns:
        CompletionResponse with generated text

    Raises:
        ProviderError: If the API request fails
    """
    # Implementation
```

### Testing Guidelines

- Write tests for all new features
- Aim for >80% code coverage
- Use pytest fixtures for common setup
- Mock external API calls
- Test error cases and edge conditions

Example test structure:

```python
import pytest
from victor.providers.ollama_provider import OllamaProvider

@pytest.fixture
async def ollama_provider():
    return OllamaProvider(base_url="http://localhost:11434")

@pytest.mark.asyncio
async def test_ollama_chat(ollama_provider, respx_mock):
    # Mock the API response
    respx_mock.post("http://localhost:11434/api/chat").mock(
        return_value={"message": {"content": "Hello!"}}
    )

    response = await ollama_provider.chat([
        {"role": "user", "content": "Hi"}
    ])

    assert response.content == "Hello!"
```

### Documentation

- Update README.md for new features
- Add docstrings to public APIs
- Create examples in `examples/` directory
- Update DESIGN.md for architectural changes

## Pull Request Process

### Before Submitting

1. Ensure all tests pass: `pytest`
2. Check code quality: `black . && ruff check .`
3. Update documentation if needed
4. Add entry to CHANGELOG.md (create if needed)

### PR Description Template

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
- [ ] No breaking changes (or documented)
```

### Review Process

1. Maintainers will review your PR
2. Address feedback and comments
3. Once approved, PR will be merged
4. Your contribution will be credited

## Areas We Need Help With

### High Priority
- Additional provider integrations (Cohere, Mistral, etc.)
- Enhanced tool calling capabilities
- Performance optimizations
- Documentation improvements

### Medium Priority
- More example use cases
- Integration tests
- CI/CD improvements
- Docker support

### Good First Issues
- Bug fixes
- Documentation typos
- Small feature enhancements
- Test coverage improvements

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## Recognition

All contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in project documentation

Thank you for contributing to Victor!
