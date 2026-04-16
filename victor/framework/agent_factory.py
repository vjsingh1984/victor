"""Unified AgentFactory — single authority for all agent creation paths.

Replaces 4 inconsistent initialization paths with one Facade:
- CLI chat → FrameworkShim → from_settings (deprecated)
- CLI tools → direct from_settings (no vertical)
- API server → direct from_settings (no profile)
- Agent.create() → inline creation

All entry points now use:
    factory = AgentFactory(settings, profile=..., vertical=..., ...)
    orchestrator = await factory.create()
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from victor.core.errors import VictorError

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class InitializationError(VictorError):
    """Structured error from agent initialization with user guidance.

    Attributes:
        stage: Which step failed (profile/credentials/provider/vertical/bootstrap)
        suggestions: List of commands or actions the user can try
        run_command: Primary fix command (e.g., "victor doctor")
    """

    def __init__(
        self,
        stage: str,
        message: str,
        suggestions: Optional[List[str]] = None,
        run_command: Optional[str] = None,
    ):
        self.stage = stage
        self.suggestions = suggestions or []
        self.run_command = run_command
        super().__init__(message)


class AgentFactory:
    """Unified agent creation — single authority for all entry points.

    Applies Facade pattern to consolidate initialization:
    1. Validate configuration (fail-fast with guidance)
    2. Bootstrap DI container with vertical
    3. Create orchestrator via from_settings
    4. Apply vertical configuration
    5. Wire observability
    6. Apply overrides (tool budget, iterations)

    Example:
        factory = AgentFactory(settings, profile="groq-fast", vertical="coding")
        orchestrator = await factory.create()
    """

    def __init__(
        self,
        settings: "Settings",
        profile: str = "default",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        vertical: Optional[Union[Type, str]] = None,
        thinking: bool = False,
        session_id: Optional[str] = None,
        enable_observability: bool = True,
        tool_budget: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        self._settings = settings
        self._profile = profile
        self._provider = provider
        self._model = model
        self._thinking = thinking
        self._session_id = session_id or str(uuid.uuid4())
        self._enable_observability = enable_observability
        self._tool_budget = tool_budget
        self._max_iterations = max_iterations
        self._orchestrator: Optional["AgentOrchestrator"] = None

        # Resolve vertical from string or class
        self._vertical = self._resolve_vertical(vertical)

    async def create(self) -> "AgentOrchestrator":
        """Create a fully configured orchestrator.

        Returns:
            Configured AgentOrchestrator instance.

        Raises:
            InitializationError: With stage, suggestions, and run_command.
        """
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.core.bootstrap import ensure_bootstrapped

        # Step 1: Pre-flight validation
        issues = self.validate()
        errors = [i for i in issues if i.get("severity") == "error"]
        if errors:
            raise InitializationError(
                stage=errors[0].get("stage", "validation"),
                message=errors[0].get("message", "Validation failed"),
                suggestions=errors[0].get("suggestions", []),
                run_command="victor doctor",
            )

        # Step 2: Bootstrap container with vertical
        try:
            vertical_name = self._vertical.name if self._vertical else None
            ensure_bootstrapped(self._settings, vertical=vertical_name)
        except Exception as e:
            raise InitializationError(
                stage="bootstrap",
                message=str(e),
                suggestions=["victor doctor", "victor init"],
                run_command="victor doctor",
            ) from e

        # Step 3: Create orchestrator
        try:
            self._orchestrator = await AgentOrchestrator.from_settings(
                self._settings,
                profile_name=self._profile,
                thinking=self._thinking,
            )
        except ValueError as e:
            # Profile not found, credentials missing, etc.
            raise InitializationError(
                stage="profile",
                message=str(e),
                suggestions=["victor profile list", "victor init"],
                run_command="victor init",
            ) from e
        except Exception as e:
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise InitializationError(
                    stage="credentials",
                    message=str(e),
                    suggestions=[
                        "Check API key environment variables",
                        "victor config validate",
                    ],
                    run_command="victor doctor",
                ) from e
            raise InitializationError(
                stage="orchestrator",
                message=str(e),
                suggestions=["victor doctor"],
            ) from e

        # Step 4: Apply vertical configuration
        if self._vertical:
            try:
                self._apply_vertical()
            except Exception as e:
                logger.warning(f"Vertical application failed: {e}")

        # Step 5: Wire observability
        if self._enable_observability:
            self._wire_observability()

        # Step 6: Apply overrides
        if self._tool_budget is not None and hasattr(self._orchestrator, "unified_tracker"):
            self._orchestrator.unified_tracker.set_tool_budget(
                self._tool_budget, user_override=True
            )
        if self._max_iterations is not None and hasattr(self._orchestrator, "unified_tracker"):
            self._orchestrator.unified_tracker.set_max_iterations(
                self._max_iterations, user_override=True
            )

        # Step 7: Initialize skill matcher
        await self._initialize_skill_matcher()

        logger.info(
            "AgentFactory created orchestrator: profile=%s, vertical=%s, session=%s",
            self._profile,
            self._vertical.name if self._vertical else None,
            self._session_id,
        )

        return self._orchestrator

    def validate(self) -> List[Dict[str, Any]]:
        """Pre-flight validation without creating orchestrator.

        Returns list of issues, each with: stage, severity, message, suggestions.
        """
        import os
        from pathlib import Path

        issues: List[Dict[str, Any]] = []

        # Check config directory
        config_dir = Path.home() / ".victor"
        if not config_dir.exists():
            issues.append(
                {
                    "stage": "config",
                    "severity": "warning",
                    "message": "No ~/.victor directory found. Run 'victor init' to set up.",
                    "suggestions": ["victor init"],
                }
            )

        # Check common API key env vars for the target provider
        provider = self._provider or "openai"
        key_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "xai": "XAI_API_KEY",
            "cohere": "COHERE_API_KEY",
        }
        if provider in key_vars and not os.environ.get(key_vars[provider]):
            issues.append(
                {
                    "stage": "credentials",
                    "severity": "warning",
                    "message": f"{key_vars[provider]} not set for provider '{provider}'",
                    "suggestions": [
                        f"export {key_vars[provider]}=...",
                        "victor doctor",
                    ],
                }
            )

        return issues

    def _resolve_vertical(self, vertical: Optional[Union[Type, str]]) -> Optional[Any]:
        """Resolve vertical from string name or class."""
        if vertical is None:
            return None
        if isinstance(vertical, str):
            try:
                from victor.core.verticals.base import VerticalRegistry

                return VerticalRegistry.get(vertical)
            except Exception:
                logger.debug(f"Vertical '{vertical}' not found in registry")
                return None
        return vertical

    def _apply_vertical(self) -> None:
        """Apply vertical configuration to the orchestrator."""
        if not self._vertical or not self._orchestrator:
            return
        try:
            from victor.framework.vertical_service import apply_vertical_configuration

            apply_vertical_configuration(self._orchestrator, self._vertical)
        except ImportError:
            logger.debug("vertical_service not available, skipping vertical application")
        except Exception as e:
            logger.warning(f"Vertical application failed: {e}")

    def _wire_observability(self) -> None:
        """Wire observability integration."""
        if not self._orchestrator:
            return
        try:
            from victor.framework._internal import setup_observability_integration

            setup_observability_integration(self._orchestrator, session_id=self._session_id)
        except Exception as e:
            logger.debug(f"Observability wiring failed: {e}")

    async def _initialize_skill_matcher(self) -> None:
        """Initialize skill auto-selection."""
        if not self._orchestrator:
            return
        try:
            from victor.framework.skills import SkillMatcher

            matcher = SkillMatcher()
            await matcher.initialize(self._orchestrator)
            self._orchestrator._skill_matcher = matcher
        except Exception as e:
            logger.debug(f"Skill matcher initialization skipped: {e}")
