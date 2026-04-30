"""Constraint activation service for unified constraint management.

Provides a single service for activating constraints across all
execution paths (legacy YAML, StateGraph, SubAgent spawning).

Research basis:
- SOLID principles — Single Responsibility for constraint activation
- Service layer pattern — Centralized constraint management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.tools.write_path_policy import WritePathPolicy
    from victor.workflows.definition import ConstraintsProtocol


@dataclass(frozen=True)
class ActivationResult:
    """Result of constraint activation.

    Attributes:
        success: True if activation succeeded
        write_path_policy: Activated write policy (if any)
        isolation_config: Isolation config from IsolationMapper (if any)
        error: Error message if activation failed
    """

    success: bool
    write_path_policy: Optional[WritePathPolicy]
    isolation_config: Optional[dict]
    error: Optional[str] = None


class ConstraintActivationService:
    """Service for activating task constraints.

    This service provides a single point for constraint activation,
    ensuring WritePathPolicy and isolation settings are applied
    consistently across all execution paths.

    Singleton pattern — use get_constraint_activator() to access.

    Thread-safe: Uses instance-level state isolation.
    """

    _instance: Optional[ConstraintActivationService] = None

    def __init__(self) -> None:
        self._active_policy: Optional[WritePathPolicy] = None
        self._active_constraints: Optional[ConstraintsProtocol] = None

    @classmethod
    def get_instance(cls) -> ConstraintActivationService:
        """Get the singleton instance.

        Returns:
            ConstraintActivationService instance
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.debug("ConstraintActivationService singleton created")
        return cls._instance

    def activate_constraints(
        self,
        constraints: Optional[ConstraintsProtocol],
        vertical: str = "coding",
    ) -> ActivationResult:
        """Activate constraints for the current execution context.

        This method:
        1. Maps constraints to isolation config via IsolationMapper
        2. Activates WritePathPolicy based on constraints
        3. Stores active constraints for reference

        Args:
            constraints: Task constraints from workflow/node
            vertical: Vertical name for default isolation

        Returns:
            ActivationResult with policy and isolation config
        """
        from victor.tools.write_path_policy import get_active_write_policy, set_active_write_policy

        # Store constraints
        self._active_constraints = constraints

        # If no constraints, use vertical default
        if constraints is None:
            logger.debug(f"No constraints provided, using vertical default: {vertical}")
            from victor.workflows.isolation import IsolationMapper

            isolation = IsolationMapper.get_vertical_default(vertical)

            # Set default policy
            default_policy = self._infer_policy_from_isolation(isolation)
            if default_policy:
                set_active_write_policy(default_policy)
                self._active_policy = default_policy

            return ActivationResult(
                success=True,
                write_path_policy=default_policy,
                isolation_config=isolation.to_dict() if isolation else None,
            )

        # Use IsolationMapper to map constraints to isolation
        try:
            from victor.workflows.isolation import IsolationMapper

            isolation = IsolationMapper.from_constraints(
                constraints=constraints,
                vertical=vertical,
            )

            # Get the activated policy (set by IsolationMapper)
            current_policy = get_active_write_policy()
            self._active_policy = current_policy

            logger.info(
                f"Constraints activated: policy={current_policy}, "
                f"isolation={isolation.to_dict() if isolation else None}"
            )

            return ActivationResult(
                success=True,
                write_path_policy=current_policy,
                isolation_config=isolation.to_dict() if isolation else None,
            )

        except Exception as e:
            error_msg = f"Constraint activation failed: {e}"
            logger.error(error_msg)
            return ActivationResult(
                success=False,
                write_path_policy=None,
                isolation_config=None,
                error=error_msg,
            )

    def deactivate_constraints(self) -> None:
        """Deactivate constraints (restore defaults).

        Called after task execution completes.
        """
        from victor.tools.write_path_policy import set_active_write_policy

        self._active_constraints = None
        self._active_policy = None
        set_active_write_policy(None)
        logger.debug("Constraints deactivated")

    def get_active_policy(self) -> Optional[WritePathPolicy]:
        """Get the currently active write policy.

        Returns:
            Active WritePathPolicy or None
        """
        return self._active_policy

    def get_active_constraints(self) -> Optional[ConstraintsProtocol]:
        """Get the currently active constraints.

        Returns:
            Active ConstraintsProtocol or None
        """
        return self._active_constraints

    def _infer_policy_from_isolation(
        self,
        isolation_config: Any,
    ) -> Optional[WritePathPolicy]:
        """Infer WritePathPolicy from IsolationConfig.

        Args:
            isolation_config: IsolationConfig instance

        Returns:
            Inferred WritePathPolicy or None
        """
        from victor.tools.write_path_policy import WritePathPolicy

        if not isolation_config:
            return None

        if isolation_config.filesystem_readonly:
            return WritePathPolicy.read_only()

        # If no explicit write_path_policy, infer from settings
        if hasattr(isolation_config, "write_path_policy"):
            return isolation_config.write_path_policy

        # Default to full access if not read-only
        return WritePathPolicy.full_access()


def get_constraint_activator() -> ConstraintActivationService:
    """Get the constraint activation service instance.

    Convenience function for accessing the singleton.

    Returns:
        ConstraintActivationService instance
    """
    return ConstraintActivationService.get_instance()
