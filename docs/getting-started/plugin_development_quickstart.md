# Quick Start: Plugin Development

## What You'll Learn

In this guide, you'll learn how to:
- Create a custom workflow compiler plugin
- Register plugins for custom URI schemes
- Load workflows from custom sources
- Test and validate plugins

## Prerequisites

- Victor installed: `pip install victor-ai`
- Python 3.10+
- Basic understanding of Python classes and protocols

## Step 1: Understand Plugin Architecture

Victor plugins implement the `WorkflowCompilerPlugin` protocol:

```python
from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin

class MyPlugin(WorkflowCompilerPlugin):
    def compile(self, source: str, *, workflow_name: str = None, validate: bool = True):
        # Load and compile workflow
        pass

    def validate_source(self, source: str) -> bool:
        # Validate source before loading
        pass

    def get_cache_key(self, source: str) -> str:
        # Generate cache key for source
        pass
```

## Step 2: Create Your First Plugin

Create `my_plugin.py`:

```python
import json
from pathlib import Path
from typing import Any, Optional
from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin

class JsonCompilerPlugin(WorkflowCompilerPlugin):
    """Plugin that loads workflows from JSON files."""

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile workflow from JSON file or string."""

        # Load JSON
        if Path(source).exists():
            with open(source, 'r') as f:
                data = json.load(f)
        else:
            data = json.loads(source)

        # Validate
        if validate:
            if 'workflows' not in data:
                raise ValueError("Missing 'workflows' key")

        # Convert to YAML and delegate to YAML compiler
        import yaml
        from victor.workflows.create import create_compiler

        yaml_str = yaml.dump(data)
        yaml_compiler = create_compiler('yaml://')
        return yaml_compiler.compile(yaml_str, workflow_name=workflow_name)

    def validate_source(self, source: str) -> bool:
        """Check if source is valid JSON."""
        try:
            json.loads(source)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def get_cache_key(self, source: str) -> str:
        """Generate cache key."""
        import hashlib
        return hashlib.md5(source.encode()).hexdigest()
```

## Step 3: Use Your Plugin

Create `use_plugin.py`:

```python
import asyncio
from victor.workflows.create import create_compiler
from my_plugin import JsonCompilerPlugin

async def main():
    # Register and create compiler
    compiler = create_compiler(
        'json://workflow.json',
        plugin_class=JsonCompilerPlugin
    )

    # Compile workflow
    workflow = compiler.compile('workflow.json', validate=True)

    # Execute workflow (if you have an agent)
    # result = await workflow.invoke({'query': 'test'})

    print("✅ Workflow compiled successfully!")

asyncio.run(main())
```

## Step 4: Create Example Workflow

Create `workflow.json`:

```json
{
  "workflows": {
    "simple_task": {
      "metadata": {
        "name": "Simple Task",
        "description": "A simple one-step workflow",
        "version": "1.0.0"
      },
      "nodes": [
        {
          "id": "start",
          "type": "agent",
          "role": "assistant",
          "goal": "Answer the user's question",
          "tool_budget": 5,
          "next": []
        }
      ]
    }
  }
}
```

Run it:

```bash
python use_plugin.py
```

## Step 5: Advanced Plugin with Options

Create a plugin with initialization options:

```python
class DatabaseCompilerPlugin(WorkflowCompilerPlugin):
    """Plugin that loads workflows from a database."""

    def __init__(self, db_path: str, table: str = "workflows"):
        self.db_path = db_path
        self.table = table
        self._connection = None

    def compile(self, source: str, *, workflow_name: str = None, validate: bool = True):
        """Load workflow from database."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT definition FROM {self.table} WHERE name = ?",
            (source,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise FileNotFoundError(f"Workflow '{source}' not found")

        # Parse and compile
        import yaml
        from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

        workflow_def = yaml.safe_load(row[0])
        compiler = YAMLToStateGraphCompiler(orchestrator=None)

        return compiler.compile(workflow_def, workflow_name=workflow_name)

    def get_cache_key(self, source: str) -> str:
        """Include database info in cache key."""
        import hashlib
        cache_string = f"{self.db_path}:{self.table}:{source}"
        return hashlib.md5(cache_string.encode()).hexdigest()
```

Use it:

```python
compiler = create_compiler(
    'db://my_workflow',
    plugin_class=DatabaseCompilerPlugin,
    db_path='workflows.db',
    table='workflows'
)

workflow = compiler.compile('my_workflow')
```

## Step 6: Register Plugin Permanently

Register via entry points in `pyproject.toml`:

```toml
[project.entry-points."victor.workflow_plugins"]
my_plugin = "my_package.my_plugin:MyPlugin"
json_plugin = "my_package.json_plugin:JsonCompilerPlugin"
```

Then use without specifying plugin_class:

```python
compiler = create_compiler('json://workflow.json')
workflow = compiler.compile('workflow.json')
```

## Common Plugin Patterns

### Pattern 1: Remote Source

```python
class HttpCompilerPlugin(WorkflowCompilerPlugin):
    """Load workflows from HTTP endpoints."""

    def compile(self, source: str, *, workflow_name: str = None, validate: bool = True):
        import urllib.request

        with urllib.request.urlopen(source, timeout=30) as response:
            content = response.read().decode('utf-8')

        # Delegate to YAML compiler
        from victor.workflows.create import create_compiler
        yaml_compiler = create_compiler('yaml://')
        return yaml_compiler.compile(content, workflow_name=workflow_name)
```

### Pattern 2: Custom Format

```python
class TomlCompilerPlugin(WorkflowCompilerPlugin):
    """Load workflows from TOML files."""

    def compile(self, source: str, *, workflow_name: str = None, validate: bool = True):
        import tomli
        import yaml

        # Load TOML
        with open(source, 'rb') as f:
            data = tomli.load(f)

        # Convert to YAML format
        yaml_str = yaml.dump(data)

        # Delegate to YAML compiler
        from victor.workflows.create import create_compiler
        yaml_compiler = create_compiler('yaml://')
        return yaml_compiler.compile(yaml_str, workflow_name=workflow_name)
```

### Pattern 3: Validation

```python
class ValidatedPlugin(WorkflowCompilerPlugin):
    """Plugin with schema validation."""

    def __init__(self, schema: dict):
        self.schema = schema

    def validate_source(self, source: str) -> bool:
        """Validate against JSON Schema."""
        try:
            import json
            from jsonschema import validate

            data = json.loads(source)
            validate(instance=data, schema=self.schema)
            return True
        except (json.JSONDecodeError, Exception):
            return False
```

## Testing Your Plugin

Create `test_plugin.py`:

```python
import pytest
from my_plugin import JsonCompilerPlugin

def test_plugin_compile():
    """Test basic compilation."""
    plugin = JsonCompilerPlugin()

    # Test with JSON string
    json_str = '{"workflows": {"test": {"nodes": []}}}'
    result = plugin.compile(json_str, validate=False)
    assert result is not None

def test_plugin_validation():
    """Test validation."""
    plugin = JsonCompilerPlugin()

    # Valid JSON
    assert plugin.validate_source('{"test": true}') is True

    # Invalid JSON
    assert plugin.validate_source('invalid') is False

def test_plugin_cache_key():
    """Test cache key generation."""
    plugin = JsonCompilerPlugin()

    key1 = plugin.get_cache_key('source')
    key2 = plugin.get_cache_key('source')

    # Same source should produce same key
    assert key1 == key2
```

Run tests:

```bash
pytest test_plugin.py -v
```

## Troubleshooting

### Issue: Plugin not found

```python
# Make sure plugin is in Python path
import sys
sys.path.append('/path/to/plugin/directory')

# Or install as package
pip install -e /path/to/plugin/package
```

### Issue: Compilation fails

```python
try:
    workflow = compiler.compile(source)
except ValueError as e:
    print(f"Validation error: {e}")
except FileNotFoundError as e:
    print(f"Source not found: {e}")
except Exception as e:
    print(f"Compilation error: {e}")
```

### Issue: Cache not working

```python
# Make sure get_cache_key returns consistent values
plugin = MyPlugin()

key1 = plugin.get_cache_key('test')
key2 = plugin.get_cache_key('test')

assert key1 == key2, "Cache keys should be consistent"
```

## Best Practices

### 1. Use Descriptive URI Schemes

```python
# Good
'json://'      # Clear purpose
's3://'        # Clear purpose
'db-postgres://'# Specific database

# Avoid
'custom://'    # Too generic
```

### 2. Validate Early

```python
def compile(self, source: str, *, validate: bool = True):
    # Validate before processing
    if validate and not self.validate_source(source):
        raise ValueError(f"Invalid source: {source}")

    # Proceed with compilation
    ...
```

### 3. Provide Clear Errors

```python
# Good
raise FileNotFoundError(
    f"Workflow '{source}' not found in S3 bucket '{self.bucket}'. "
    f"Check the bucket name and region."
)

# Avoid
raise Exception("Not found")
```

### 4. Document Usage

```python
class MyPlugin(WorkflowCompilerPlugin):
    """My custom workflow compiler plugin.

    Usage:
        compiler = create_compiler('my-scheme://source', plugin_class=MyPlugin)
        workflow = compiler.compile('source')

    Configuration:
        - option1: Description (default: value)
        - option2: Description (default: value)

    Raises:
        FileNotFoundError: If source doesn't exist
        ValueError: If source is invalid
    """
```

## Next Steps

1. **Read full documentation**: `docs/features/plugin_development.md`
2. **See complete examples**:
   - `examples/plugins/json_compiler_plugin.py`
   - `examples/plugins/s3_compiler_plugin.py`
   - `examples/custom_plugin.py`
3. **Explore workflow system**: `docs/user-guide/workflows.md`
4. **Build advanced plugins**: Add validation, caching, error handling

## Summary

✅ You learned:
- How to create a custom plugin
- How to register and use plugins
- Common plugin patterns
- How to test plugins
- Best practices for plugin development

You're now ready to create custom workflow compiler plugins!
