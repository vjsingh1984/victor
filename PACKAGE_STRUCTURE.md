# Package Structure & Installation Guide

## Why This Package Structure?

The current structure (`victor/` at the root) is a valid and commonly used Python package layout. Let me explain the different approaches:

### Current Structure (Recommended for this project)

```
victor/
├── victor/           # Package directory (importable)
│   ├── __init__.py
│   ├── agent/
│   ├── providers/
│   ├── tools/
│   └── ...
├── tests/                 # Tests outside package
├── examples/              # Examples
├── pyproject.toml         # Build configuration
└── README.md
```

**Advantages:**
- ✅ Simple and intuitive structure
- ✅ Package name matches directory name
- ✅ Easy to understand for contributors
- ✅ Standard setuptools discovery works automatically
- ✅ Tests are separate from package (good for distribution)

### Alternative: src/ Layout

```
victor/
├── src/
│   └── victor/       # Package inside src/
│       ├── __init__.py
│       └── ...
├── tests/
└── pyproject.toml
```

**When to use:**
- When you want to absolutely ensure tests run against installed package
- When you have multiple packages in one repo
- When you want extra isolation during development

**Why we don't use it here:**
- Our structure is already clean and works well
- Extra nesting adds complexity without clear benefit
- Standard structure is more familiar to contributors

## Understanding `pip install -e ".[dev]"`

Let's break down this command:

### `pip install`
Standard pip installation command

### `-e` (Editable Mode)
- Installs the package in "editable" or "development" mode
- Creates a symlink to your source code instead of copying it
- Changes to source code are immediately available without reinstalling
- Essential for active development

**Example:**
```bash
# Without -e: Code is copied to site-packages
pip install .
# Changes to code require reinstall: pip install . (again)

# With -e: Code stays in place
pip install -e .
# Changes to code are immediately available!
```

### `.` (Current Directory)
- Tells pip to install the package from the current directory
- Looks for `pyproject.toml` or `setup.py`

### `[dev]` (Extra Dependencies)
- Installs optional "dev" dependencies group
- Defined in `pyproject.toml` under `[project.optional-dependencies]`
- Includes testing, linting, formatting tools

**In our `pyproject.toml`:**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
    "mypy>=1.10",
    # ... etc
]
```

### Full Command Breakdown

```bash
pip install -e ".[dev]"
```

Translates to:
1. Install the package from current directory
2. In editable mode (symlink, not copy)
3. Include the "dev" extra dependencies
4. Quote it because square brackets are special in shell

## Installation Scenarios

### 1. Development Installation (Recommended)

```bash
pip install -e ".[dev]"
```

**Use when:**
- Actively developing CodingAgent
- Writing new features or fixing bugs
- Running tests
- Contributing to the project

**What happens:**
- Package is symlinked to current directory
- All dev tools installed (pytest, ruff, mypy)
- Code changes take effect immediately
- `victor` command available globally

### 2. User Installation (from PyPI - future)

```bash
pip install victor
```

**Use when:**
- Just want to use CodingAgent
- Don't need development tools
- Installing from PyPI

**What happens:**
- Package copied to site-packages
- Only runtime dependencies installed
- Can't edit code easily

### 3. Local Installation (without editable)

```bash
pip install .
```

**Use when:**
- Testing the installation process
- Want to test as end-user would
- Creating a distribution

**What happens:**
- Package copied to site-packages
- Changes require reinstall
- Cleaner but less convenient for dev

## How Import Works

After installation, you can import from anywhere:

```python
from victor import AgentOrchestrator
from victor.providers import OllamaProvider
from victor.tools.bash import BashTool
```

This works because:
1. `pyproject.toml` defines the package name as "victor"
2. Installation creates `victor` in Python's import path
3. The directory structure matches the import structure

## Package Discovery

### How setuptools finds the package

In `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
include = ["victor*"]
exclude = ["tests*", "docs*"]
```

This tells setuptools:
- Include: `victor` and all subpackages
- Exclude: `tests` and `docs` from distribution

**Result:** Only the `victor/` directory is packaged and installed

## Entry Points (CLI Commands)

```toml
[project.scripts]
victor = "victor.ui.cli:app"
ca = "victor.ui.cli:app"
```

This creates two commands:
- `victor` → runs `app` from `victor/ui/cli.py`
- `ca` → same, just a shorter alias

After installation, these commands are available globally in your terminal.

## Development Workflow

### 1. Initial Setup

```bash
# Clone repo
git clone <repo>
cd victor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 2. Making Changes

```python
# Edit code in victor/providers/new_provider.py
class NewProvider(BaseProvider):
    # ... your changes
```

```bash
# Test immediately (no reinstall needed!)
victor  # Your changes are live
python -c "from victor.providers.new_provider import NewProvider"  # Works!
```

### 3. Running Tests

```bash
# All dependencies already installed with [dev]
pytest
pytest --cov
pytest -v tests/unit/
```

### 4. Code Quality

```bash
# All tools available from [dev] extras
ruff check victor
mypy victor
```

## Why Not Use src/?

The src/ layout debate:

### Pros of src/ layout:
- Prevents accidentally importing from source instead of installed package
- Forces you to test against installed version
- Extra safety for complex projects

### Why we don't need it:
- Our project structure is already clean
- Editable installs are explicitly for development
- Extra nesting adds complexity
- Standard layout is more familiar
- Tests can still verify proper packaging

### When to reconsider:
- If we add multiple packages to the repo
- If we have import conflicts
- If community strongly prefers it

## Distribution

### Building the package:

```bash
# Install build tools
pip install build

# Build distribution
python -m build

# Creates:
# dist/victor-0.1.0-py3-none-any.whl
# dist/victor-0.1.0.tar.gz
```

### What gets included:

Only the `victor/` directory:
```
victor-0.1.0/
├── victor/
│   ├── __init__.py
│   ├── agent/
│   ├── providers/
│   ├── tools/
│   └── ...
├── pyproject.toml
├── README.md
└── LICENSE
```

Tests, examples, and docs are excluded (configured in pyproject.toml).

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'victor'"

**Solution:**
```bash
# Make sure you installed it
pip install -e ".[dev]"

# Check it's installed
pip list | grep victor

# Verify virtual environment is activated
which python  # Should point to venv/bin/python
```

### Issue: Changes not taking effect

**Solution:**
```bash
# Reinstall in editable mode
pip install -e ".[dev]"

# Or check if you accidentally did non-editable install
pip list -v | grep victor
# Should show: /path/to/victor (editable)
```

### Issue: "command not found: victor"

**Solution:**
```bash
# Reinstall to create entry points
pip install -e ".[dev]"

# Or verify PATH includes scripts directory
echo $PATH | grep venv

# Try with full path
python -m victor.ui.cli
```

## Best Practices

### For Development:
1. ✅ Always use `pip install -e ".[dev]"`
2. ✅ Use virtual environments
3. ✅ Keep package code in `victor/`
4. ✅ Keep tests in `tests/`
5. ✅ Run tests before committing

### For Distribution:
1. ✅ Exclude tests from package
2. ✅ Include README, LICENSE
3. ✅ Specify all dependencies correctly
4. ✅ Test installation in clean environment
5. ✅ Version properly (semver)

## Summary

**Current structure is great because:**
- Simple and standard
- Easy for contributors
- Works perfectly with editable installs
- No unnecessary complexity

**Use `pip install -e ".[dev]"` because:**
- Changes take effect immediately
- Includes all development tools
- Standard development practice
- Recommended by Python Packaging Authority

**No need for src/ because:**
- Our structure is already clean
- Package isolation is good enough
- Would add complexity without clear benefit
- Current structure is more intuitive

The package is production-ready and follows modern Python best practices!
