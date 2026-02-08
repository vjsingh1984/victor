# Step Handler Examples - Part 2

**Part 2 of 3:** Safety & Security and Configuration Management

---

## Navigation

- [Part 1: Basic through Workflow](part-1-basic-middleware-workflow.md)
- **[Part 2: Safety & Config](#)** (Current)
- [Part 3: Advanced & Testing](part-3-advanced-testing.md)
- [**Complete Guide**](../step_handler_examples.md)

---

## Safety & Security

### Example 11: Safety Pattern Validation

Validate safety patterns before application:

```python
from typing import List, Any

class SafetyValidationHandler(BaseStepHandler):
    """Validate safety patterns."""

    @property
    def name(self) -> str:
        return "safety_validation"

    @property
    def order(self) -> int:
        return 28  # Before safety application (30)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate safety patterns."""
        # Get safety extensions
        safety_extensions = vertical.get_safety_extensions()

        if not safety_extensions:
            return

        # Collect and validate patterns
        all_patterns = []
        invalid_count = 0

        for ext in safety_extensions:
            patterns = self._get_extension_patterns(ext)
            for pattern in patterns:
                if self._is_valid_pattern(pattern):
                    all_patterns.append(pattern)
                else:
                    invalid_count += 1

        # Report validation results
        if invalid_count > 0:
            result.add_warning(f"Found {invalid_count} invalid patterns")

        # Store validated patterns in context
        context.apply_validated_safety_patterns(all_patterns)

        result.add_info(f"Validated {len(all_patterns)} safety patterns")

    def _get_extension_patterns(self, extension: Any) -> List[Any]:
        """Get patterns from extension."""
        patterns = []
        if hasattr(extension, "get_bash_patterns"):
            patterns.extend(extension.get_bash_patterns())
        if hasattr(extension, "get_file_patterns"):
            patterns.extend(extension.get_file_patterns())
        return patterns

    def _is_valid_pattern(self, pattern: Any) -> bool:
        """Check if pattern is valid."""
        if not hasattr(pattern, "pattern"):
            return False
        if not hasattr(pattern, "type"):
            return False
        return True
```

### Example 12: Strict Mode Enforcement

Enable strict validation based on settings:

```python
class StrictModeHandler(BaseStepHandler):
    """Enable strict validation mode."""

    @property
    def name(self) -> str:
        return "strict_mode"

    @property
    def order(self) -> int:
        return 32  # After safety (30)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Enable strict mode if configured."""
        # Check if strict mode should be enabled
        if not self._should_enable_strict(orchestrator):
            result.add_info("Strict mode not enabled")
            return

        # Enable via capability
        if _check_capability(orchestrator, "strict_safety"):
            _invoke_capability(orchestrator, "strict_safety", True)
            result.add_info("Enabled strict safety mode")
        else:
            result.add_warning("Strict safety capability not available")

    def _should_enable_strict(self, orchestrator: Any) -> bool:
        """Check if strict mode should be enabled."""
        if hasattr(orchestrator, "settings"):
            return getattr(orchestrator.settings, "strict_mode", False)
        return False
```

---

## Configuration Management

### Example 13: Mode Configuration Application

Apply mode configurations with validation:

```python
from typing import Dict, Any

class ModeConfigurationHandler(BaseStepHandler):
    """Apply mode configurations."""

    @property
    def name(self) -> str:
        return "mode_configuration"

    @property
    def order(self) -> int:
        return 42  # Part of config step (40)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply mode configurations."""
        # Get mode provider from vertical
        provider = self._get_mode_provider(vertical)

        if provider is None:
            result.add_info("No mode configuration provider")
            return

        # Get mode configs
        mode_configs = provider.get_mode_configs()
        default_mode = provider.get_default_mode()
        default_budget = provider.get_default_tool_budget()

        # Validate mode configs
        if not self._validate_mode_configs(mode_configs):
            result.add_error("Invalid mode configurations")
            return

        # Apply to context
        context.apply_mode_configs(mode_configs, default_mode, default_budget)

        # Apply to orchestrator via capability
        if _check_capability(orchestrator, "mode_configs"):
            _invoke_capability(orchestrator, "mode_configs", mode_configs)

        result.mode_configs_count = len(mode_configs)
        result.add_info(f"Applied {len(mode_configs)} mode configs")

    def _get_mode_provider(self, vertical: Type[VerticalBase]) -> Optional[Any]:
        """Get mode provider from vertical."""
        if hasattr(vertical, "get_mode_config_provider"):
            return vertical.get_mode_config_provider()
        return None

    def _validate_mode_configs(self, configs: Dict[str, Any]) -> bool:
        """Validate mode configurations."""
        # Check for required modes
        if not configs:
            return False

        # Validate each mode config
        for name, config in configs.items():
            if not hasattr(config, "exploration"):
                return False
            if not hasattr(config, "tool_budget_multiplier"):
                return False

        return True
```

### Example 14: Stage Configuration

Configure conversation stages:

```python
class StageConfigurationHandler(BaseStepHandler):
    """Configure conversation stages."""

    @property
    def name(self) -> str:
        return "stage_configuration"

    @property
    def order(self) -> int:
        return 41  # Part of config step (40)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply stage configuration."""
        # Get stages from vertical
        stages = vertical.get_stages()

        if not stages:
            result.add_info("No custom stages")
            return

        # Validate stages
        if not self._validate_stages(stages):
            result.add_error("Invalid stage configuration")
            return

        # Apply to context
        context.apply_stages(stages)

        # Log stage names
        stage_names = list(stages.keys())
        result.add_info(f"Applied {len(stages)} stages: {stage_names}")

    def _validate_stages(self, stages: Dict[str, Any]) -> bool:
        """Validate stage configuration."""
        # Check for required stages
        required = {"INITIAL", "EXECUTING", "COMPLETION"}
        return required.issubset(set(stages.keys()))
```

---

**Continue to [Part 3: Advanced Patterns & Testing](part-3-advanced-testing.md)**
