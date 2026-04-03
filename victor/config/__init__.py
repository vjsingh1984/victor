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

"""Victor configuration system.

This module provides a streamlined configuration system for Victor that unifies
provider accounts, authentication, and settings.

Key Components:
- ProviderAccount: Unified account model
- AccountManager: Account management and resolution
- ConfigMigrator: Migration from old configuration format
- ConnectionValidator: Test provider connections

Example:
    from victor.config import get_account_manager

    manager = get_account_manager()
    account = manager.get_account("default")
    config = manager.resolve_provider_config(account)
"""

# New unified configuration system
from victor.config.accounts import (
    ProviderAccount,
    AuthConfig,
    AccountManager,
    VictorConfig,
    ConfigDefaults,
    get_account_manager,
    reset_account_manager,
)

from victor.config.resolution import (
    ProviderResolver,
    ResolutionContext,
    get_provider_resolver,
    resolve_provider_config,
    reset_provider_resolver,
)

from victor.config.migration import (
    ConfigMigrator,
    MigrationResult,
    check_migration_needed,
    run_migration,
    rollback_migration,
)

from victor.config.connection_validation import (
    ConnectionValidator,
    ValidationResult,
    ConnectionTestResult,
    validate_account_sync,
    validate_account_async,
    ping_provider,
)

# Legacy components (for backward compatibility)
from victor.config.settings import Settings, load_settings, get_settings
from victor.config.provider_config_registry import (
    ProviderConfigStrategy,
    ProviderConfigRegistry,
    get_provider_config_registry,
    register_provider_config,
    DefaultProviderConfig,
    DEFAULT_PROVIDER_ENDPOINTS,
)

__all__ = [
    # New unified configuration
    "ProviderAccount",
    "AuthConfig",
    "AccountManager",
    "VictorConfig",
    "ConfigDefaults",
    "get_account_manager",
    "reset_account_manager",
    # Resolution
    "ProviderResolver",
    "ResolutionContext",
    "get_provider_resolver",
    "resolve_provider_config",
    "reset_provider_resolver",
    # Migration
    "ConfigMigrator",
    "MigrationResult",
    "check_migration_needed",
    "run_migration",
    "rollback_migration",
    # Connection validation
    "ConnectionValidator",
    "ValidationResult",
    "ConnectionTestResult",
    "validate_account_sync",
    "validate_account_async",
    "ping_provider",
    # Legacy (backward compatibility)
    "Settings",
    "load_settings",
    "get_settings",
    "ProviderConfigStrategy",
    "ProviderConfigRegistry",
    "get_provider_config_registry",
    "register_provider_config",
    "DefaultProviderConfig",
    "DEFAULT_PROVIDER_ENDPOINTS",
]
