<div align="center">

# Contributor Quick Start Guide

**Start contributing to Victor AI in 10 minutes**

[![Contributors](https://img.shields.io/github/contributors/vjsingh1984/victor)](https://github.com/vjsingh1984/victor/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/vjsingh1984/victor)](https://github.com/vjsingh1984/victor/issues)

</div>

---

## Welcome to the Victor AI Community!

Thank you for your interest in contributing to Victor AI! This guide will help you set up your development environment and make your first contribution in just 10 minutes.

### What You'll Learn

- How to set up your development environment
- How to run tests
- How to make a pull request
- Contribution guidelines and best practices

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10 or higher**
- **Git** installed and configured
- **GitHub account**
- **Basic Python knowledge**
- **Familiarity with terminal/command line**

### Check Your Tools

```bash
python --version  # Should be 3.10+
git --version     # Should be 2.0+
```

---

## Step 1: Set Up Development Environment (3 minutes)

### Fork and Clone

1. **Fork the repository** on GitHub:
   - Visit https://github.com/vjsingh1984/victor
   - Click "Fork" in the top-right corner

2. **Clone your fork**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/victor.git
   cd victor
   ```

3. **Add upstream remote**:

   ```bash
   git remote add upstream https://github.com/vjsingh1984/victor.git
   ```

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Install Dependencies

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Verify installation
victor --version
```

---

## Step 2: Run Tests (2 minutes)

### Run All Tests

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests (requires additional setup)
pytest tests/integration -v

# Run all tests
pytest -v
```

### Run Specific Test

```bash
# Run a specific test file
pytest tests/unit/test_providers.py -v

# Run a specific test
pytest tests/unit/test_providers.py::test_anthropic_provider -v
```

### Run with Coverage

```bash
# Run tests with coverage report
pytest --cov=victor --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Step 3: Make Your First Contribution (5 minutes)

### Choose a Good First Issue

1. **Visit the issues page**:
   - https://github.com/vjsingh1984/victor/issues

2. **Filter for good first issues**:
   - Use labels: `good first issue`, `help wanted`, `beginner-friendly`

3. **Claim the issue**:
   - Comment on the issue: "I'd like to work on this"
   - Wait for assignment

### Create a Branch

```bash
# Update your local repository
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/your-feature-name

# Or for a bug fix
git checkout -b fix/bug-description
```

### Make Your Changes

1. **Edit the code**:

   ```bash
   # Open your favorite editor
   code .
   # Or use vim, nano, etc.
   ```

2. **Follow code style**:
   - Use type hints for public APIs
   - Write docstrings (Google style)
   - Follow PEP 8 guidelines

3. **Add tests**:
   - Write unit tests for new functionality
   - Ensure all tests pass

### Run Linters and Formatters

```bash
# Format code with black
black victor tests

# Check code style with ruff
ruff check victor tests

# Run type checking with mypy
mypy victor
```

### Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add support for new provider

- Implement XYZProvider class
- Add configuration for XYZ API
- Add unit tests for provider functionality
- Update documentation

Closes #123"
```

### Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### Create Pull Request

1. **Visit your fork on GitHub**
2. **Click "Compare & pull request"**
3. **Fill in the PR template**:
   - Title: Clear description of changes
   - Description: What you changed and why
   - References: Link to related issues
4. **Click "Create pull request"**

---

## Contribution Guidelines

### Code Style

- **Type Hints**: Required for all public APIs
- **Docstrings**: Google-style docstrings
- **Line Length**: 100 characters (black enforced)
- **Import Style**: Group imports: stdlib, third-party, local
- **Naming**: snake_case for functions/variables, PascalCase for classes

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

**Example**:
```
feat(providers): add Groq provider support

- Implement GroqProvider with chat completion
- Add configuration for Groq API
- Add unit tests for Groq integration
- Update provider documentation

Closes #456
```

### Testing Requirements

- **Unit Tests**: Required for new functionality
- **Integration Tests**: For complex features
- **Coverage**: Maintain >80% code coverage
- **Test Naming**: `test_<function>_<scenario>`

### Documentation

- **README**: Update if adding user-facing features
- **Docstrings**: Required for all public APIs
- **Type Hints**: Required for all public APIs
- **Changelog**: Add entry for user-facing changes

---

## Development Workflow

### Typical Workflow

```bash
# 1. Update from upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and test
# Edit code...
pytest tests/unit -v
black victor tests
ruff check victor tests

# 4. Commit changes
git add .
git commit -m "feat: add my feature"

# 5. Push to fork
git push origin feature/my-feature

# 6. Create pull request on GitHub
```

### Keeping Your Fork Synced

```bash
# Add upstream (if not already added)
git remote add upstream https://github.com/vjsingh1984/victor.git

# Fetch from upstream
git fetch upstream

# Merge upstream changes into your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

---

## Common Development Tasks

### Add a New Provider

1. Create provider class in `victor/providers/`
2. Inherit from `BaseProvider`
3. Implement required methods: `chat()`, `stream_chat()`, `supports_tools()`
4. Add configuration in `victor/config/model_capabilities.yaml`
5. Register in `ProviderRegistry`
6. Add tests in `tests/unit/providers/`
7. Update documentation

### Add a New Tool

1. Create tool class in `victor/tools/`
2. Inherit from `BaseTool`
3. Define: `name`, `description`, `parameters`, `cost_tier`, `execute()`
4. Register in tool registry
5. Add tests in `tests/unit/tools/`
6. Update documentation

### Fix a Bug

1. Reproduce the bug
2. Add test case that fails
3. Fix the bug
4. Ensure test passes
5. Check for related issues
6. Update documentation if needed

### Update Documentation

1. Edit documentation in `docs/`
2. Follow existing documentation style
3. Update links and references
4. Test documentation builds locally
5. Submit PR with `docs` label

---

## Testing Tips

### Run Tests Quickly

```bash
# Skip slow tests
pytest -m "not slow" -v

# Run specific test file
pytest tests/unit/test_providers.py -v

# Run specific test
pytest tests/unit/test_providers.py::test_anthropic_provider -v

# Stop on first failure
pytest -x -v

# Run with verbose output
pytest -vv
```

### Debug Tests

```bash
# Run with pdb debugger
pytest --pdb

# Show print statements
pytest -s -v

# Run with coverage
pytest --cov=victor --cov-report=term-missing
```

### Test Organization

```
tests/
├── unit/           # Unit tests (fast, isolated)
│   ├── providers/  # Provider tests
│   ├── tools/      # Tool tests
│   └── agents/     # Agent tests
├── integration/    # Integration tests (slower)
└── smoke/          # Smoke tests (quick validation)
```

---

## Code Review Process

### Before Submitting PR

- [ ] All tests pass (`pytest -v`)
- [ ] Code formatted (`black victor tests`)
- [ ] No lint errors (`ruff check victor tests`)
- [ ] Type checking passes (`mypy victor`)
- [ ] Documentation updated
- [ ] Changelog updated (for user-facing changes)
- [ ] Commit messages follow conventions

### During Review

1. **Address feedback**:
   - Make requested changes
   - Respond to comments
   - Push updates to branch

2. **Update PR**:
   - Mark conversations as resolved
   - Add "Fixes #XXX" to close related issues

3. **Be patient**:
   - Reviewers may take time
   - Ask for clarification if needed

### After Merge

1. **Delete branch**:
   ```bash
   git branch -d feature/my-feature
   git push origin --delete feature/my-feature
   ```

2. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

3. **Celebrate!** 🎉

---

## Getting Help

### Documentation

- [Developer Onboarding](./DEVELOPER_ONBOARDING.md) - Complete developer guide
- [Development Setup](./development/setup.md) - Environment setup
- [Code Style](./development/code-style.md) - Coding standards
- [Testing Guide](./development/testing.md) - Testing guidelines

### Community

- [GitHub Issues](https://github.com/vjsingh1984/victor/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/vjsingh1984/victor/discussions) - Community discussions
- [Contributing Guide](../CONTRIBUTING.md) - Contribution guidelines

### Quick Questions

- Check existing [Discussions](https://github.com/vjsingh1984/victor/discussions)
- Search [Issues](https://github.com/vjsingh1984/victor/issues) for similar problems
- Read [FAQ](./user-guide/faq.md)

---

## Resources

### Development Tools

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **Mypy**: Static type checker
- **Pytest**: Testing framework
- **Pre-commit**: Git hooks for code quality

### Learning Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PEP 8 Style Guide](https://pep8.org/)

### Victor Architecture

- [Architecture Overview](./architecture/overview.md) - System design
- [Component Reference](./architecture/COMPONENT_REFERENCE.md) - Component docs
- [Design Patterns](./architecture/DESIGN_PATTERNS.md) - SOLID patterns

---

## Next Steps

### For First-Time Contributors

1. ✅ Complete this guide
2. ✅ Set up development environment
3. ✅ Run tests successfully
4. ✅ Make your first PR
5. 📚 Read [Developer Onboarding](./DEVELOPER_ONBOARDING.md)

### For Regular Contributors

1. 📖 Explore [Architecture Documentation](./architecture/README.md)
2. 🔧 Build [Extensions](./extensions/README.md)
3. 📝 Review [Design Patterns](./architecture/DESIGN_PATTERNS.md)
4. 🚀 Contribute to [Roadmap](./roadmap/future_roadmap.md)

### For Advanced Contributors

1. 🏗️ Review [SOLID Refactoring](./SOLID_ARCHITECTURE_REFACTORING_REPORT.md)
2. ⚡ Explore [Performance Optimization](./performance/README.md)
3. 🔐 Review [Security Best Practices](./SECURITY_BEST_PRACTICES.md)
4. 📊 Analyze [Benchmarks](./performance/BENCHMARK_SUMMARY.md)

---

## Code of Conduct

Please be respectful and constructive:

- Be welcoming to newcomers
- Focus on what is best for the community
- Show empathy towards other community members
- Respect different opinions and viewpoints

See [Code of Conduct](../CODE_OF_CONDUCT.md) for details.

---

## License

By contributing to Victor AI, you agree that your contributions will be licensed under the **Apache License 2.0**.

---

## Thank You! 🎉

Thank you for contributing to Victor AI! Every contribution helps make Victor better for everyone.

**Ready to start?**

[Find an Issue](https://github.com/vjsingh1984/victor/issues) •
[Read the Docs](./INDEX.md) •
[Join Discussions](https://github.com/vjsingh1984/victor/discussions)

---

<div align="center">

**[Back to Documentation Index](./INDEX.md)**

</div>
