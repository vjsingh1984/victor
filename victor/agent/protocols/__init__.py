# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manager/coordinator protocol interfaces.

This package provides Protocol-based interfaces for managers and coordinators,
following the Interface Segregation Principle (ISP).
"""

from victor.agent.protocols.manager_protocols import (
    IProviderHealthMonitor,
    IProviderSwitcher,
    IToolAdapterCoordinator,
    IProviderEventEmitter,
    IProviderClassificationStrategy,
    IMessageStore,
    IContextOverflowHandler,
    ISessionManager,
    IEmbeddingManager,
    IBudgetTracker,
    IMultiplierCalculator,
    IModeCompletionChecker,
    IToolCallClassifier,
)

__all__ = [
    "IProviderHealthMonitor",
    "IProviderSwitcher",
    "IToolAdapterCoordinator",
    "IProviderEventEmitter",
    "IProviderClassificationStrategy",
    "IMessageStore",
    "IContextOverflowHandler",
    "ISessionManager",
    "IEmbeddingManager",
    "IBudgetTracker",
    "IMultiplierCalculator",
    "IModeCompletionChecker",
    "IToolCallClassifier",
]
