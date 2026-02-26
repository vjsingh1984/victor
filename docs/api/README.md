# Victor API Documentation

This directory contains the API reference documentation for the Victor AI Framework, built with Sphinx.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
# From the project root
pip install -e ".[sphinx-docs]"

# Or install from the docs directory
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
# From the docs directory
cd docs
make html

# Or on Windows
make.bat html
```

The built documentation will be in `docs/_build/html/`.

### Live Reload During Development

```bash
# From the docs directory
cd docs
make live

# Or on Windows
make.bat live
```

This will serve the documentation at `http://127.0.0.1:8000` and automatically rebuild when files change.

### Clean Build Directory

```bash
cd docs
make clean
```

## API Reference Structure

| File | Description |
|------|-------------|
| `agent.rst` | Agent API - creating and running agents |
| `stategraph.rst` | StateGraph - workflow engine API |
| `workflows.rst` | Workflows - YAML workflows and WorkflowEngine |
| `tools.rst` | Tools - tool system and built-in tools |
| `providers.rst` | Providers - LLM provider integrations |

## Sphinx Extensions

The documentation uses the following Sphinx extensions:

- **autodoc** - Automatic documentation from docstrings
- **napoleon** - Support for Google and NumPy style docstrings
- **autodoc-typehints** - Type hints in documentation
- **viewcode** - Links to highlighted source code
- **intersphinx** - Links to external documentation (Python, asyncio)

## Writing Documentation

### Adding a New API Reference

1. Create a new `.rst` file in this directory
2. Use Sphinx autodoc directives:

```rst
ClassName
=========

.. autoclass:: victor.module.ClassName
   :members:
   :undoc-members:
   :show-inheritance:
```

3. Add the file to the `toctree` in `index.rst`

### Docstring Format

Use Google style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: If param1 is empty
    """
    pass
```

## Next Steps

- Complete type hints on public APIs
- Add more examples to API reference
- Generate automatic API docs with sphinx-apidoc
- Add Jupyter notebook tutorials

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Furo Theme](https://pradyunsg.me/furo/)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
