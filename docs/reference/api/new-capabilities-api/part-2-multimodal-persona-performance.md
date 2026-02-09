# New Capabilities API Reference - Part 2

**Part 2 of 2:** Multimodal, Persona, Performance, and Configuration APIs

---

## Navigation

- [Part 1: Planning, Memory, Skills](part-1-planning-memory-skills.md)
- **[Part 2: Multimodal, Persona, Performance](#)** (Current)
- [**Complete Reference**](../NEW_CAPABILITIES_API.md)

---

> **Note**: This legacy API documentation is retained for reference. For current docs, see `docs/reference/api/`.

## Multimodal APIs

### ImageProcessor

```python
class ImageProcessor:
    """Process and understand images."""

    async def analyze_image(
        self,
        image_uri: str,
        query: str = "Describe this image"
    ) -> ImageAnalysisResult:
        """Analyze image content.

        Args:
            image_uri: URI or path to image
            query: Query for image analysis

        Returns:
            ImageAnalysisResult: Analysis result
        """
        ...
```

## Persona APIs

### PersonaManager

```python
class PersonaManager:
    """Manage agent personas."""

    async def load_persona(
        self,
        persona_name: str
    ) -> Persona:
        """Load persona by name.

        Args:
            persona_name: Name of persona to load

        Returns:
            Persona: Loaded persona configuration
        """
        ...
```

## Performance APIs

### PerformanceMonitor

```python
class PerformanceMonitor:
    """Monitor and optimize performance."""

    async def get_metrics(
        self
    ) -> PerformanceMetrics:
        """Get current performance metrics.

        Returns:
            PerformanceMetrics: Current metrics
        """
        ...
```

## Configuration APIs

### ConfigurationManager

```python
class ConfigurationManager:
    """Manage system configuration."""

    async def update_config(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Update system configuration.

        Args:
            config: Configuration updates
        """
        ...
```

## Type Definitions

### ExecutionPlan

```python
@dataclass
class ExecutionPlan:
    """Execution plan for goal achievement."""

    goal: str
    steps: List[ExecutionStep]
    dependencies: Dict[str, List[str]]
    estimated_duration: Optional[float] = None
```

### MemoryEntry

```python
@dataclass
class MemoryEntry:
    """Memory entry representation."""

    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime
```

## Migration Notes

For migration from legacy APIs to new capability system, see:
- [Migration Guide](../../../architecture/MIGRATION_GUIDES.md)
- [Capability System](../CAPABILITIES.md)

## Additional Resources

- [Architecture Overview](../../../architecture/README.md)
- [Configuration Reference](../CONFIGURATION_REFERENCE.md)
- [API Reference](../API_REFERENCE.md)

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 1 min
**Last Updated:** February 01, 2026
