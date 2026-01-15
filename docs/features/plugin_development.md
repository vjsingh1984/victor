# Plugin Development Guide

## Overview

Victor's plugin system allows you to extend workflow compilation capabilities by creating custom compiler plugins. Plugins enable loading workflows from various sources (JSON, S3, databases, custom URIs) while maintaining a consistent interface.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Workflow Source                          │
│  (YAML, JSON, S3, Database, Custom URI, etc.)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              WorkflowCompilerPlugin                         │
│  - compile(source, workflow_name, validate)                 │
│  - validate_source(source)                                  │
│  - get_cache_key(source)                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              CompiledGraphProtocol                          │
│  - invoke(state)                                            │
│  - stream(state)                                            │
│  - get_graph()                                              │
└─────────────────────────────────────────────────────────────┘
```

## Base Plugin Interface

All plugins must inherit from `WorkflowCompilerPlugin`:

```python
from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin
from typing import Any, Optional

class MyPlugin(WorkflowCompilerPlugin):
    """Custom workflow compiler plugin."""

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile workflow from source.

        Args:
            source: Workflow source (file path, URI, string, etc.)
            workflow_name: Name of workflow to compile
            validate: Whether to validate before compilation

        Returns:
            CompiledGraphProtocol instance

        Raises:
            ValueError: If source is invalid or validation fails
            FileNotFoundError: If source file doesn't exist
        """
        # Your implementation here
        pass

    def validate_source(self, source: str) -> bool:
        """Validate source before compilation.

        Args:
            source: Workflow source to validate

        Returns:
            True if valid, False otherwise
        """
        # Your validation logic here
        return True

    def get_cache_key(self, source: str) -> str:
        """Generate cache key for source.

        Args:
            source: Workflow source

        Returns:
            Cache key string
        """
        # Your cache key generation here
        return source
```

## Creating a Custom Plugin

### Example 1: JSON Workflow Plugin

Create a plugin that loads workflows from JSON files:

```python
import json
from pathlib import Path
from typing import Any, Dict, Optional

from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin

class JsonCompilerPlugin(WorkflowCompilerPlugin):
    """JSON workflow compiler plugin."""

    supported_schemes = ['json', 'json+file']

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile JSON workflow."""
        # Load JSON
        data = self._load_json(source)

        # Validate if requested
        if validate:
            self._validate_json(data)

        # Convert to Victor's workflow format
        workflow_def = self._convert_to_workflow_def(data)

        # Compile using Victor's YAML compiler
        from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

        compiler = YAMLToStateGraphCompiler(
            orchestrator=None,
            orchestrators=None,
        )

        return compiler.compile(workflow_def, workflow_name=workflow_name)

    def _load_json(self, source: str) -> Dict:
        """Load JSON from file or string."""
        if Path(source).exists():
            with open(source, 'r') as f:
                return json.load(f)
        else:
            return json.loads(source)

    def _validate_json(self, data: Dict) -> None:
        """Validate JSON structure."""
        required_keys = ['workflows']

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        # Validate workflows section
        if not isinstance(data['workflows'], dict):
            raise ValueError("'workflows' must be a dictionary")

        # Validate each workflow
        for workflow_name, workflow_def in data['workflows'].items():
            if not isinstance(workflow_def, dict):
                raise ValueError(f"Workflow '{workflow_name}' must be a dictionary")

            if 'nodes' not in workflow_def:
                raise ValueError(f"Workflow '{workflow_name}' missing 'nodes' key")

    def validate_source(self, source: str) -> bool:
        """Validate JSON source."""
        try:
            json.loads(source)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def get_cache_key(self, source: str) -> str:
        """Generate cache key for JSON source."""
        import hashlib
        return hashlib.md5(source.encode()).hexdigest()
```

### Example 2: Database Plugin

Create a plugin that loads workflows from a database:

```python
from typing import Any, Optional
import sqlite3

from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin

class DatabaseCompilerPlugin(WorkflowCompilerPlugin):
    """Database workflow compiler plugin."""

    def __init__(self, db_path: str, table: str = "workflows"):
        """Initialize database plugin.

        Args:
            db_path: Path to SQLite database
            table: Table name containing workflows
        """
        self.db_path = db_path
        self.table = table

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile workflow from database."""
        # Query database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT definition FROM {self.table} WHERE name = ?",
            (source,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise FileNotFoundError(f"Workflow '{source}' not found in database")

        # Parse workflow definition
        import yaml
        workflow_def = yaml.safe_load(row[0])

        # Compile using YAML compiler
        from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

        compiler = YAMLToStateGraphCompiler(
            orchestrator=None,
            orchestrators=None,
        )

        return compiler.compile(workflow_def, workflow_name=workflow_name)

    def get_cache_key(self, source: str) -> str:
        """Generate cache key including database info."""
        import hashlib
        cache_string = f"{self.db_path}:{self.table}:{source}"
        return hashlib.md5(cache_string.encode()).hexdigest()
```

### Example 3: HTTP Plugin

Create a plugin that loads workflows from HTTP endpoints:

```python
from typing import Any, Optional
import urllib.request

from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin

class HttpCompilerPlugin(WorkflowCompilerPlugin):
    """HTTP workflow compiler plugin."""

    supported_schemes = ['http', 'https']

    def __init__(self, timeout: int = 30, headers: Optional[dict] = None):
        """Initialize HTTP plugin.

        Args:
            timeout: Request timeout in seconds
            headers: Optional HTTP headers
        """
        self.timeout = timeout
        self.headers = headers or {}

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile workflow from HTTP endpoint."""
        # Fetch workflow from HTTP
        req = urllib.request.Request(source, headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            content = response.read().decode('utf-8')

        # Compile using YAML compiler
        from victor.workflows.create import create_compiler

        yaml_compiler = create_compiler('yaml://')
        return yaml_compiler.compile(
            content,
            workflow_name=workflow_name,
            validate=validate,
        )

    def validate_source(self, source: str) -> bool:
        """Validate HTTP URL."""
        return source.startswith(('http://', 'https://'))

    def get_cache_key(self, source: str) -> str:
        """Generate cache key for URL."""
        import hashlib
        return hashlib.md5(source.encode()).hexdigest()
```

## Registering Plugins

### Method 1: Direct Registration

```python
from victor.workflows.compiler_registry import WorkflowCompilerRegistry
from my_plugin import MyPlugin

# Create registry
registry = WorkflowCompilerRegistry()

# Register plugin by URI scheme
registry.register('my-scheme', MyPlugin)

# Use plugin
compiler = registry.create('my-scheme://')
workflow = compiler.compile('my-source')
```

### Method 2: Via create_compiler()

```python
from victor.workflows.create import create_compiler
from my_plugin import MyPlugin

# Register and create in one step
compiler = create_compiler(
    'my-scheme://my-source',
    plugin_class=MyPlugin,
    plugin_options={'option1': 'value1'}
)

workflow = compiler.compile('my-source')
```

### Method 3: Entry Points (Package Distribution)

Register plugins via entry points in `pyproject.toml`:

```toml
[project.entry-points."victor.workflow_plugins"]
my_plugin = "my_package.my_plugin:MyPlugin"
json_plugin = "my_package.json_plugin:JsonCompilerPlugin"
```

## Plugin URI Schemes

Plugins are identified by URI schemes:

```python
# File-based schemes
'yaml://'       # YAML files (built-in)
'json://'       # JSON files (custom)

# Remote schemes
'http://'       # HTTP endpoints (custom)
'https://'      # HTTPS endpoints (custom)
's3://'         # S3 buckets (custom)

# Database schemes
'db://'         # Database queries (custom)
'mongodb://'    # MongoDB (custom)

# Custom schemes
'my-scheme://'  # Custom plugin
```

## Plugin Configuration

### Initialization Options

Pass configuration options to plugins:

```python
class MyPlugin(WorkflowCompilerPlugin):
    def __init__(
        self,
        option1: str,
        option2: int = 10,
        **options
    ):
        self.option1 = option1
        self.option2 = option2
        self.options = options

# Register with options
registry.register('my-scheme', MyPlugin)

# Create with options
compiler = registry.create(
    'my-scheme://',
    option1='value1',
    option2=20,
    custom_option='custom'
)
```

### Settings Integration

Configure plugins via settings:

```yaml
# config.yaml
workflow_plugins:
  my_plugin:
    enabled: true
    options:
      option1: value1
      option2: 20
```

```python
from victor.config.settings import Settings

settings = Settings(config_file='config.yaml')
plugin_config = settings.workflow_plugins.get('my_plugin', {})

compiler = MyPlugin(**plugin_config.get('options', {}))
```

## Advanced Features

### Custom Validation

Implement sophisticated validation logic:

```python
import json
from jsonschema import validate, ValidationError

class SchemaValidatedPlugin(WorkflowCompilerPlugin):
    """Plugin with JSON Schema validation."""

    def __init__(self, schema: dict):
        self.schema = schema

    def validate_source(self, source: str) -> bool:
        """Validate against JSON Schema."""
        try:
            data = json.loads(source)
            validate(instance=data, schema=self.schema)
            return True
        except (json.JSONDecodeError, ValidationError):
            return False
```

### Caching Strategies

Implement custom caching:

```python
import hashlib
import time

class TTLCachePlugin(WorkflowCompilerPlugin):
    """Plugin with TTL-based cache invalidation."""

    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl
        self.cache = {}

    def get_cache_key(self, source: str) -> str:
        """Generate cache key with timestamp."""
        key = hashlib.md5(source.encode()).hexdigest()
        timestamp = int(time.time())
        return f"{key}:{timestamp}"

    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        _, timestamp = cache_key.split(':')
        age = int(time.time()) - int(timestamp)
        return age < self.cache_ttl
```

### Lazy Loading

Implement lazy loading for large workflows:

```python
class LazyLoadingPlugin(WorkflowCompilerPlugin):
    """Plugin that lazy-loads workflow definitions."""

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Create lazy-loading wrapper."""
        # Don't load immediately
        return LazyWorkflow(
            source=source,
            workflow_name=workflow_name,
            validate=validate,
            plugin=self
        )

    def _load_workflow(self, source: str):
        """Actually load the workflow (called later)."""
        # Loading logic here
        pass

class LazyWorkflow:
    """Lazy-loading workflow wrapper."""

    def __init__(self, source: str, workflow_name: str, validate: bool, plugin):
        self.source = source
        self.workflow_name = workflow_name
        self.validate = validate
        self.plugin = plugin
        self._compiled = None

    def invoke(self, state: dict):
        """Load and invoke workflow."""
        if self._compiled is None:
            self._compiled = self.plugin._load_workflow(self.source)
        return self._compiled.invoke(state)
```

## Error Handling

### Graceful Degradation

```python
class RobustPlugin(WorkflowCompilerPlugin):
    """Plugin with fallback behavior."""

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile with fallback."""
        try:
            # Try primary method
            return self._compile_primary(source, workflow_name, validate)
        except Exception as e:
            logger.warning(f"Primary method failed: {e}")
            try:
                # Try fallback method
                return self._compile_fallback(source, workflow_name, validate)
            except Exception as e2:
                # Raise with helpful message
                raise RuntimeError(
                    f"Failed to compile workflow '{source}': "
                    f"Primary: {e}, Fallback: {e2}"
                ) from e2
```

### Validation Errors

```python
class ValidatedPlugin(WorkflowCompilerPlugin):
    """Plugin with detailed validation."""

    def validate_source(self, source: str) -> bool:
        """Validate with detailed error messages."""
        errors = []

        # Check 1: Source is not empty
        if not source:
            errors.append("Source is empty")

        # Check 2: Source is valid YAML/JSON
        try:
            import yaml
            yaml.safe_load(source)
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML: {e}")

        # Check 3: Required fields
        data = yaml.safe_load(source)
        required = ['workflows', 'metadata']
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if errors:
            raise ValueError(
                f"Validation failed for '{source}':\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return True
```

## Testing Plugins

### Unit Tests

```python
import pytest
from my_plugin import MyPlugin

def test_plugin_compile():
    """Test basic compilation."""
    plugin = MyPlugin()
    result = plugin.compile('{"workflows": {}}')
    assert result is not None

def test_plugin_validation():
    """Test validation."""
    plugin = MyPlugin()
    assert plugin.validate_source('{"workflows": {}}') is True
    assert plugin.validate_source('invalid') is False

def test_plugin_cache_key():
    """Test cache key generation."""
    plugin = MyPlugin()
    key1 = plugin.get_cache_key('source')
    key2 = plugin.get_cache_key('source')
    assert key1 == key2  # Same source, same key
```

### Integration Tests

```python
import pytest
from victor.workflows.create import create_compiler

@pytest.mark.asyncio
async def test_plugin_workflow_execution():
    """Test workflow execution with plugin."""
    compiler = create_compiler('my-scheme://test', plugin_class=MyPlugin)
    workflow = compiler.compile('test-workflow')
    result = await workflow.invoke({'input': 'test'})
    assert result is not None
```

## Best Practices

### 1. URI Scheme Design

Use descriptive, unique URI schemes:

```python
# Good
'json://'      # Clear purpose
's3://'        # Clear purpose
'db-postgres://'# Specific database

# Avoid
'custom://'    # Too generic
'plugin1://'   # Not descriptive
```

### 2. Error Messages

Provide clear, actionable error messages:

```python
# Good
raise FileNotFoundError(
    f"Workflow '{source}' not found in S3 bucket '{self.bucket}'. "
    f"Check the bucket name and region."
)

# Avoid
raise Exception("Not found")
```

### 3. Documentation

Document plugin usage thoroughly:

```python
class MyPlugin(WorkflowCompilerPlugin):
    """My custom workflow compiler plugin.

    Loads workflows from custom source.

    Usage:
        from victor.workflows.create import create_compiler

        compiler = create_compiler('my-scheme://source', plugin_class=MyPlugin)
        workflow = compiler.compile('source')
        result = await workflow.invoke({'input': 'data'})

    Configuration:
        - option1: Description (default: value)
        - option2: Description (default: value)

    Raises:
        FileNotFoundError: If source doesn't exist
        ValueError: If source is invalid
    """
```

### 4. Caching

Implement smart caching:

```python
def get_cache_key(self, source: str) -> str:
    """Include version/timestamp in cache key."""
    import hashlib
    metadata = self._get_metadata(source)
    cache_string = f"{source}:{metadata['version']}"
    return hashlib.md5(cache_string.encode()).hexdigest()
```

### 5. Validation

Validate inputs early:

```python
def compile(self, source: str, *, workflow_name: Optional[str] = None, validate: bool = True):
    """Validate inputs before processing."""
    if not source:
        raise ValueError("Source cannot be empty")

    if validate and not self.validate_source(source):
        raise ValueError(f"Invalid source: {source}")

    # Proceed with compilation
    ...
```

## Examples

See complete examples:
- `/Users/vijaysingh/code/codingagent/examples/plugins/json_compiler_plugin.py`
- `/Users/vijaysingh/code/codingagent/examples/plugins/s3_compiler_plugin.py`

## Related Documentation

- [Workflow System](../user-guide/workflows.md)
- [Creating Workflows](../tutorials/create-workflow.md)
- [Compiler API](../api-reference/workflows.md)
