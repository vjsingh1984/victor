# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deprecated shim for streaming recovery compatibility types."""

from victor.agent.services.recovery_compat import (
    StreamingRecoveryContext,
    StreamingRecoveryCoordinator,
)

__all__ = [
    "StreamingRecoveryCoordinator",
    "StreamingRecoveryContext",
]
