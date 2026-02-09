# Vertical Development Guide - Part 4

**Part 4 of 4:** Appendix and Conclusion

---

## Navigation

- [Part 1: Capability & Middleware](part-1-capability-middleware.md)
- [Part 2: Chain Registry & Personas](part-2-chain-registry-personas.md)
- [Part 3: Complete Workflow Example](part-3-complete-workflow-example.md)
- **[Part 4: Appendix & Conclusion](#)** (Current)
- [**Complete Guide**](../VERTICAL_DEVELOPMENT_GUIDE.md)

---

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

```python
from victor.myvertical import MyVerticalAssistant

# Create assistant
assistant = MyVerticalAssistant()

# Get configuration
config = assistant.get_config()

# Use with orchestrator
orchestrator = AgentOrchestrator(config=config)
```text

## Workflows

- **analysis**: Comprehensive analysis of X
- **workflow2**: Description

## Capabilities

- `my_capability`: Description

## Personas

- **Specialist**: X domain specialist
- **Reviewer**: Quality assurance reviewer
```

### 5.5 Integration Checklist

- [ ] VerticalBase subclass created
- [ ] Capability provider implemented
- [ ] Middleware created (if needed)
- [ ] Personas defined (if multi-agent)
- [ ] Teams configured (if multi-agent)
- [ ] LCEL chains created (if needed)
- [ ] Workflow handlers implemented
- [ ] Escape hatches defined
- [ ] YAML workflows created
- [ ] Workflow provider created
- [ ] Vertical registered in VerticalRegistry
- [ ] Unit tests written
- [ ] Documentation created
- [ ] Integration tests pass

---

## Appendix: Common Patterns

### A. Vertical Registration Pattern

```python
# In vertical's __init__.py
from victor.myvertical.assistant import MyVerticalAssistant

# Auto-register on import
from victor.core.verticals import VerticalRegistry
VerticalRegistry.register(MyVerticalAssistant)

__all__ = ["MyVerticalAssistant"]
```text

### B. Capability Delegation Pattern

```python
# Delegate to framework capability
from victor.framework.capabilities.privacy import configure_data_privacy as framework_privacy

def configure_my_privacy(orchestrator: Any, **kwargs: Any) -> None:
    """Delegate to framework privacy capability."""
    framework_privacy(orchestrator, **kwargs)
```

### C. Middleware Composition Pattern

```python
class CompositeMiddleware(MiddlewareProtocol):
    """Combines multiple middleware."""

    def __init__(self, *middlewares: MiddlewareProtocol):
        self._middlewares = list(middlewares)

    async def before_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MiddlewareResult:
        for middleware in self._middlewares:
            result = await middleware.before_tool_call(tool_name, arguments)
            if not result.should_proceed:
                return result
            if result.modified_arguments:
                arguments = result.modified_arguments

        return MiddlewareResult(should_proceed=True)
```text

### D. Chain Factory Pattern

```python
def create_analysis_chain(
    *,
    include_validation: bool = True,
    parallel: bool = False,
) -> Runnable:
    """Factory for creating analysis chains."""
    if parallel:
        return _create_parallel_chain()
    elif include_validation:
        return _create_validated_chain()
    else:
        return _create_basic_chain()
```

---

## Conclusion

This guide provides comprehensive patterns and examples for creating new verticals in Victor AI. By following these
  patterns,
  you can create well-structured, maintainable, and consistent verticals that integrate seamlessly with the framework.

For more information:
- See existing vertical implementations: `victor/{coding,research,devops,dataanalysis,rag}/`
- Review test files for examples: `tests/unit/{vertical}/`
- Consult framework documentation: `victor/framework/`

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 5 minutes
