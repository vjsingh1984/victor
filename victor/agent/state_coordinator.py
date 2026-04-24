# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deprecated shim for StateCoordinator compatibility types."""

from victor.agent.services.state_compat import (
    IStateCoordinator,
    StageTransition,
    StateCoordinator,
    StateCoordinatorConfig,
    create_state_coordinator,
)

__all__ = [
    "StateCoordinator",
    "StateCoordinatorConfig",
    "StageTransition",
    "IStateCoordinator",
    "create_state_coordinator",
]
