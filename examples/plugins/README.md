# Victor Plugin Examples

This directory contains example workflow compiler plugins that demonstrate how to extend Victor's workflow system.

## Available Examples

### 1. JSON Compiler Plugin

**File**: `json_compiler_plugin.py`

Load workflows from JSON files instead of YAML.

```python
from victor.workflows.compiler_registry import WorkflowCompilerRegistry
from examples.plugins.json_compiler_plugin import JsonCompilerPlugin

registry = WorkflowCompilerRegistry()
registry.register('json', JsonCompilerPlugin)

compiler = registry.create('json://')
compiled = compiler.compile('workflow.json')
result = await compiled.invoke({'input': 'data'})
```

### 2. S3 Compiler Plugin

**File**: `s3_compiler_plugin.py`

Load workflows from AWS S3 buckets.

**Prerequisites**:
- `pip install boto3`
- AWS credentials configured
- S3 bucket access

```python
from victor.workflows.create import create_compiler

compiler = create_compiler(
    's3://my-bucket/workflows/deep_research.yaml',
    bucket='my-bucket',
    region='us-west-2'
)
compiled = compiler.compile('workflows/deep_research.yaml')
result = await compiled.invoke({'query': 'AI trends'})
```

## Creating Your Own Plugin

See [PLUGIN_DEVELOPMENT_GUIDE.md](../../../PLUGIN_DEVELOPMENT_GUIDE.md) for complete documentation on creating custom plugins.

## Quick Template

```python
from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin
from typing import Any, Dict, List, Optional

class MyPlugin(WorkflowCompilerPlugin):
    """My custom compiler plugin."""

    supported_schemes = ['my-scheme']

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile workflow from source."""
        # Your compilation logic here
        pass

    @property
    def supported_schemes(self) -> List[str]:
        return self.supported_schemes
```

## Testing Plugins

```python
# Test your plugin
from victor.workflows.compiler_registry import WorkflowCompilerRegistry
from my_plugin import MyPlugin

registry = WorkflowCompilerRegistry()
registry.register('test', MyPlugin)

compiler = registry.create('test://')
compiled = compiler.compile('test_workflow')
```

## Contributing

To add your plugin as an example:

1. Create a new file in this directory
2. Follow the naming convention: `{scheme}_compiler_plugin.py`
3. Include comprehensive docstrings
4. Add usage examples to this README
5. Ensure it passes tests

## Support

For questions or issues:
- See [PLUGIN_DEVELOPMENT_GUIDE.md](../../../PLUGIN_DEVELOPMENT_GUIDE.md)
- Check examples in this directory
- Report issues on GitHub

---

*Plugin Examples: January 9, 2025*
*Version: v0.6.0*
