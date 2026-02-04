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

"""Secure API key management for Victor.

This module provides secure storage and retrieval of API keys:
- Keys stored in ~/.victor/api_keys.yaml (outside code repository)
- Only loads the key for the requested provider (isolation)
- Environment variables take precedence over file-based keys
- System keyring integration for secure persistent storage
- Graceful handling for providers without keys (Ollama, LMStudio)

Security Features:
- SEC-001: HOME directory validation against passwd database
- SEC-001: System keyring integration for encrypted storage
- SEC-004: Audit logging for all secret access attempts
- SEC-005: Atomic file creation with secure permissions (TOCTOU fix)
- SEC-006: Secure credential caching with memory clearing
- File has restricted permissions (0600) when created
- Keys are never logged or exposed in error messages
- Provider isolation prevents cross-provider key leakage

Configuration:
- Provider and service definitions are loaded from api_keys_registry.yaml
- This allows external configuration of supported providers/services
- The registry file is the single source of truth for key metadata

Resolution Order:
1. Environment variable (highest priority - for automation/CI)
2. System keyring (secure encrypted storage)
3. Keys file (fallback with audit logging)
"""

import ctypes
import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml


def _get_registry_path() -> Path:
    """Get the path to the API keys registry YAML file."""
    return Path(__file__).parent / "api_keys_registry.yaml"


def _load_registry() -> dict[str, Any]:
    """Load the API keys registry from YAML.

    Returns cached version if already loaded.
    Falls back to hardcoded defaults if file not found.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE

    registry_path = _get_registry_path()
    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                _REGISTRY_CACHE = yaml.safe_load(f) or {}
                return _REGISTRY_CACHE
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load registry: {e}")

    # Return empty dict - will use hardcoded fallbacks
    _REGISTRY_CACHE = {}
    return _REGISTRY_CACHE


def _build_env_vars_from_registry(section: str) -> dict[str, str]:
    """Build environment variable mapping from registry section.

    Args:
        section: 'providers' or 'services'

    Returns:
        Dict mapping name to env_var
    """
    registry = _load_registry()
    items = registry.get(section, {})

    result = {}
    for name, config in items.items():
        if isinstance(config, dict):
            env_var = config.get("env_var")
            if env_var:
                result[name] = env_var
                # Also add aliases
                for alias in config.get("aliases", []):
                    result[alias] = env_var

    return result


# Cache for loaded registry
_REGISTRY_CACHE: Optional[dict[str, Any]] = None

logger = logging.getLogger(__name__)


def _get_secure_keys_file() -> Path:
    """Get the secure path for API keys file.

    Uses secure home directory resolution to protect against
    HOME environment variable manipulation.
    """
    try:
        from victor.config.secure_paths import get_victor_dir

        return get_victor_dir() / "api_keys.yaml"
    except ImportError:
        # Fallback if secure_paths not available
        return Path.home() / ".victor" / "api_keys.yaml"


# Default location for API keys file (computed securely)
DEFAULT_KEYS_FILE = _get_secure_keys_file()

# Hardcoded fallback for provider environment variables
# These are used if api_keys_registry.yaml is not found or fails to load
_PROVIDER_ENV_VARS_FALLBACK: dict[str, str] = {
    # Premium API providers
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "grok": "XAI_API_KEY",  # Alias for xai
    "moonshot": "MOONSHOT_API_KEY",
    "kimi": "MOONSHOT_API_KEY",  # Alias for moonshot
    "deepseek": "DEEPSEEK_API_KEY",
    "zai": "ZAI_API_KEY",
    "zhipuai": "ZAI_API_KEY",  # Alias for zai
    "zhipu": "ZAI_API_KEY",  # Alias for zai
    # Free-tier providers
    "groqcloud": "GROQCLOUD_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    # Enterprise cloud providers
    "vertex": "GOOGLE_APPLICATION_CREDENTIALS",  # Service account JSON path or ADC
    "vertexai": "GOOGLE_APPLICATION_CREDENTIALS",  # Alias
    "azure": "AZURE_OPENAI_API_KEY",
    "azure-openai": "AZURE_OPENAI_API_KEY",  # Alias
    "bedrock": "AWS_ACCESS_KEY_ID",  # Uses AWS credentials chain
    "aws": "AWS_ACCESS_KEY_ID",  # Alias
    "huggingface": "HF_TOKEN",
    "hf": "HF_TOKEN",  # Alias
    "replicate": "REPLICATE_API_TOKEN",
}


def _get_provider_env_vars() -> dict[str, str]:
    """Get provider to environment variable mapping.

    Loads from registry if available, falls back to hardcoded values.
    """
    registry_vars = _build_env_vars_from_registry("providers")
    if registry_vars:
        return registry_vars
    return _PROVIDER_ENV_VARS_FALLBACK.copy()


# Provider to environment variable mapping (lazy-loaded from registry)
PROVIDER_ENV_VARS: dict[str, str] = _get_provider_env_vars()

# Providers that don't require API keys
LOCAL_PROVIDERS = {"ollama", "lmstudio", "vllm"}

# ============================================================================
# SERVICE API KEYS (External Data Services - not LLM providers)
# ============================================================================
# Hardcoded fallback for service environment variables
_SERVICE_ENV_VARS_FALLBACK: dict[str, str] = {
    # Market Data & Financial Services
    "finnhub": "FINNHUB_API_KEY",  # Stock data, news sentiment, analyst estimates
    "fred": "FRED_API_KEY",  # Federal Reserve Economic Data
    "alphavantage": "ALPHAVANTAGE_API_KEY",  # Stock/forex/crypto data
    "polygon": "POLYGON_API_KEY",  # Real-time & historical market data
    "tiingo": "TIINGO_API_KEY",  # Stock/crypto/forex data
    "iex": "IEX_API_KEY",  # IEX Cloud market data
    "quandl": "QUANDL_API_KEY",  # Financial/economic datasets (now Nasdaq Data Link)
    "nasdaq": "NASDAQ_API_KEY",  # Nasdaq Data Link
    # News & Sentiment
    "newsapi": "NEWSAPI_API_KEY",  # News aggregation
    "marketaux": "MARKETAUX_API_KEY",  # Financial news API
    # SEC & Regulatory
    "sec": "SEC_API_KEY",  # SEC EDGAR API (optional, for higher rate limits)
    # Other Data Services
    "openweather": "OPENWEATHER_API_KEY",  # Weather data
    "geocoding": "GEOCODING_API_KEY",  # Geocoding services
}


def _get_service_env_vars() -> dict[str, str]:
    """Get service to environment variable mapping.

    Loads from registry if available, falls back to hardcoded values.
    """
    registry_vars = _build_env_vars_from_registry("services")
    if registry_vars:
        return registry_vars
    return _SERVICE_ENV_VARS_FALLBACK.copy()


# Service to environment variable mapping (lazy-loaded from registry)
SERVICE_ENV_VARS: dict[str, str] = _get_service_env_vars()


def get_all_key_types() -> dict[str, dict[str, str]]:
    """Get all known key types (providers + services)."""
    return {
        "provider": PROVIDER_ENV_VARS,
        "service": SERVICE_ENV_VARS,
    }


# All known keys (providers + services) for comprehensive listing
ALL_KEY_TYPES = get_all_key_types()

# Keyring service name for Victor
KEYRING_SERVICE = "victor"

# Check if keyring is available
_KEYRING_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("keyring") is not None:
        _KEYRING_AVAILABLE = True
except ImportError:
    pass


def _get_key_from_keyring(provider: str) -> Optional[str]:
    """Get API key from system keyring.

    Uses the system's secure credential storage:
    - macOS: Keychain
    - Windows: Credential Manager
    - Linux: Secret Service (GNOME Keyring, KWallet)

    Args:
        provider: Provider name (e.g., "anthropic")

    Returns:
        API key if found, None otherwise
    """
    if not _KEYRING_AVAILABLE:
        return None

    try:
        import keyring  # type: ignore[import-not-found]

        key = keyring.get_password(KEYRING_SERVICE, f"{provider}_api_key")
        return key if key is not None else None
    except Exception as e:
        logger.debug(f"Keyring access failed for {provider}: {e}")
        return None


def _set_key_in_keyring(provider: str, key: str) -> bool:
    """Store API key in system keyring.

    Args:
        provider: Provider name
        key: API key to store

    Returns:
        True if successful, False otherwise
    """
    if not _KEYRING_AVAILABLE:
        logger.warning("Keyring not available. Install 'keyring' package for secure storage.")
        return False

    try:
        import keyring
        keyring.set_password(KEYRING_SERVICE, f"{provider}_api_key", key)
        logger.info(f"API key for {provider} stored in system keyring")
        return True
    except Exception as e:
        logger.warning(f"Failed to store key in keyring: {e}")
        return False


def _delete_key_from_keyring(provider: str) -> bool:
    """Delete API key from system keyring.

    Args:
        provider: Provider name

    Returns:
        True if successful, False otherwise
    """
    if not _KEYRING_AVAILABLE:
        return False

    try:
        import keyring
        keyring.delete_password(KEYRING_SERVICE, f"{provider}_api_key")
        return True
    except Exception as e:
        logger.debug(f"Failed to delete key from keyring: {e}")
        return False


class SecureString:
    """Memory-safe string storage with secure zeroing.

    This class provides a way to store sensitive strings that can be
    securely cleared from memory when no longer needed.

    Security:
        - Stores value as mutable bytearray
        - Provides secure zeroing on clear()
        - Automatically clears on deletion
    """

    def __init__(self, value: str):
        """Initialize with a string value."""
        self._value = bytearray(value.encode("utf-8"))
        self._length = len(self._value)
        self._cleared = False

    def get(self) -> str:
        """Get the string value."""
        if self._cleared:
            return ""
        return self._value.decode("utf-8")

    def clear(self) -> None:
        """Securely zero the memory."""
        if self._cleared:
            return

        # Zero out the bytearray
        for i in range(self._length):
            self._value[i] = 0

        # Force memory write if possible
        try:
            if self._length > 0:
                ctypes.memset(
                    ctypes.addressof(ctypes.c_char.from_buffer(self._value)),
                    0,
                    self._length,
                )
        except (ValueError, TypeError):
            pass  # Buffer may already be empty

        self._cleared = True

    def __del__(self) -> None:
        """Clear memory on deletion."""
        self.clear()

    def __bool__(self) -> bool:
        """Return True if contains a non-empty value."""
        return not self._cleared and self._length > 0


def _audit_log_secret_access(
    action: str,
    provider: str,
    source: str,
    success: bool,
    key_length: int = 0,
) -> None:
    """Log a secret access event for audit purposes.

    Args:
        action: Action being performed (request, loaded, denied)
        provider: Provider name
        source: Source of the key (environment, file, cache)
        success: Whether the operation succeeded
        key_length: Length of key (never log the actual key!)
    """
    # Always log at INFO level for audit trail
    logger.info(
        f"SECRET_ACCESS: action={action} provider={provider} "
        f"source={source} success={success} key_length={key_length}"
    )

    # Try to log to audit system if available
    try:
        from victor.core.security.audit import AuditManager, AuditEventType

        audit = AuditManager.get_instance()
        audit.log_event(
            event_type=AuditEventType.SECRET_ACCESS,
            action=action,
            details={
                "provider": provider,
                "source": source,
                "success": success,
                "key_length": key_length,
                "pid": os.getpid(),
                "uid": os.getuid() if hasattr(os, "getuid") else None,
            },
        )
    except (ImportError, AttributeError):
        pass  # Audit module not available
    except Exception as e:
        logger.debug(f"Could not log to audit system: {e}")


class APIKeyManager:
    """Manages API keys with secure storage and provider isolation.

    Security Features:
        - Validates HOME directory against passwd database
        - Audit logs all secret access attempts
        - Atomic file writes to prevent TOCTOU race conditions
        - Secure credential caching with memory clearing

    Usage:
        manager = APIKeyManager()
        key = manager.get_key("anthropic")  # Only loads anthropic key

        # Or use convenience function:
        key = get_api_key("deepseek")
    """

    def __init__(self, keys_file: Optional[Path] = None):
        """Initialize API key manager.

        Args:
            keys_file: Path to API keys file (default: ~/.victor/api_keys.yaml)
        """
        if keys_file is None:
            keys_file = _get_secure_keys_file()
        self.keys_file = keys_file
        # Use SecureString for cached credentials
        self._cache: dict[str, Optional[SecureString]] = {}

    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider.

        Resolution order (most secure first):
        1. Environment variable (highest priority - for automation/CI)
        2. System keyring (secure encrypted storage)
        3. Keys file (~/.victor/api_keys.yaml)
        4. None (for local providers or missing keys)

        All access attempts are audit logged.

        Args:
            provider: Provider name (e.g., "anthropic", "deepseek")

        Returns:
            API key string or None if not found/required
        """
        provider = provider.lower()

        # Local providers don't need keys
        if provider in LOCAL_PROVIDERS:
            _audit_log_secret_access(
                action="request",
                provider=provider,
                source="local_provider",
                success=True,
                key_length=0,
            )
            return None

        # Check cache first
        if provider in self._cache:
            cached = self._cache[provider]
            if cached:
                key = cached.get()
                _audit_log_secret_access(
                    action="cache_hit",
                    provider=provider,
                    source="cache",
                    success=True,
                    key_length=len(key),
                )
                return key
            return None

        # Priority 1: Environment variable (highest priority - for automation)
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var:
            env_key = os.environ.get(env_var)
            if env_key:
                self._cache[provider] = SecureString(env_key)
                _audit_log_secret_access(
                    action="loaded",
                    provider=provider,
                    source="environment",
                    success=True,
                    key_length=len(env_key),
                )
                return env_key

        # Priority 2: System keyring (secure encrypted storage)
        keyring_key = _get_key_from_keyring(provider)
        if keyring_key:
            self._cache[provider] = SecureString(keyring_key)
            _audit_log_secret_access(
                action="loaded",
                provider=provider,
                source="keyring",
                success=True,
                key_length=len(keyring_key),
            )
            return keyring_key

        # Priority 3: Keys file (least secure, with audit)
        file_key = self._load_key_from_file(provider)
        if file_key:
            self._cache[provider] = SecureString(file_key)
            _audit_log_secret_access(
                action="loaded",
                provider=provider,
                source="file",
                success=True,
                key_length=len(file_key),
            )
            return file_key

        # No key found
        self._cache[provider] = None
        _audit_log_secret_access(
            action="not_found",
            provider=provider,
            source="none",
            success=False,
            key_length=0,
        )
        return None

    def _load_key_from_file(self, provider: str) -> Optional[str]:
        """Load a single provider's key from the keys file.

        Only reads the specific provider's key to maintain isolation.
        Checks file permissions for security.

        Args:
            provider: Provider name

        Returns:
            API key or None
        """
        if not self.keys_file.exists():
            return None

        # Check file permissions (SEC-005)
        try:
            from victor.config.secure_paths import check_file_permissions

            if not check_file_permissions(self.keys_file):
                logger.warning(
                    f"API keys file {self.keys_file} has insecure permissions. "
                    f"Consider running: chmod 600 {self.keys_file}"
                )
        except ImportError:
            pass

        try:
            with open(self.keys_file, "r") as f:
                data = yaml.safe_load(f) or {}

            # Keys are stored under 'api_keys' section
            api_keys = data.get("api_keys", data)  # Support both formats

            # Get key for this provider only
            key = api_keys.get(provider)
            return key if key is not None else None

        except Exception as e:
            logger.warning(f"Failed to load API key from {self.keys_file}: {e}")
            return None

    def set_key(self, provider: str, key: str, use_keyring: bool = False) -> bool:
        """Save an API key.

        Args:
            provider: Provider name
            key: API key value
            use_keyring: If True, store in system keyring (more secure).
                        If False, store in keys file.

        Returns:
            True if saved successfully
        """
        if use_keyring:
            return self._set_key_to_keyring(provider, key)
        return self._set_key_to_file(provider, key)

    def _set_key_to_keyring(self, provider: str, key: str) -> bool:
        """Store API key in system keyring.

        Args:
            provider: Provider name
            key: API key value

        Returns:
            True if saved successfully
        """
        success = _set_key_in_keyring(provider, key)
        if success:
            self._cache[provider] = SecureString(key)
            _audit_log_secret_access(
                action="saved",
                provider=provider,
                source="keyring",
                success=True,
                key_length=len(key),
            )
        else:
            _audit_log_secret_access(
                action="save_failed",
                provider=provider,
                source="keyring",
                success=False,
                key_length=0,
            )
        return success

    def _set_key_to_file(self, provider: str, key: str) -> bool:
        """Save an API key to the keys file.

        Uses atomic file write to prevent TOCTOU race conditions.

        Args:
            provider: Provider name
            key: API key value

        Returns:
            True if saved successfully
        """
        try:
            # Ensure directory exists
            self.keys_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing keys
            existing_data: dict[str, Any] = {}
            if self.keys_file.exists():
                with open(self.keys_file, "r") as f:
                    existing_data = yaml.safe_load(f) or {}

            # Ensure api_keys section exists
            if "api_keys" not in existing_data:
                existing_data["api_keys"] = {}

            # Set the key
            existing_data["api_keys"][provider] = key

            # Serialize to YAML
            content = yaml.dump(existing_data, default_flow_style=False)

            # Use secure atomic write (SEC-005: TOCTOU fix)
            try:
                from victor.config.secure_paths import secure_create_file

                success = secure_create_file(
                    self.keys_file,
                    content,
                    mode=0o600,
                    atomic=True,
                )
                if not success:
                    raise OSError("secure_create_file failed")
            except ImportError:
                # Fallback: atomic write with secure open
                temp_path = self.keys_file.with_suffix(".yaml.tmp")
                fd = os.open(
                    temp_path,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    0o600,
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        f.write(content)
                except Exception:
                    os.close(fd)
                    raise
                os.rename(temp_path, self.keys_file)

            # Update cache with SecureString
            self._cache[provider] = SecureString(key)

            _audit_log_secret_access(
                action="saved",
                provider=provider,
                source="file",
                success=True,
                key_length=len(key),
            )

            logger.info(f"API key for {provider} saved to {self.keys_file}")
            return True

        except Exception as e:
            _audit_log_secret_access(
                action="save_failed",
                provider=provider,
                source="file",
                success=False,
                key_length=0,
            )
            logger.error(f"Failed to save API key: {e}")
            return False

    def list_configured_providers(self) -> list[str]:
        """List providers that have API keys configured.

        Checks all sources: environment, keyring, and file.

        Returns:
            List of provider names with keys
        """
        configured = []

        # Check environment variables
        for provider, env_var in PROVIDER_ENV_VARS.items():
            if os.environ.get(env_var):
                if provider not in configured:
                    configured.append(provider)

        # Check keyring for all known providers
        if _KEYRING_AVAILABLE:
            for provider in PROVIDER_ENV_VARS.keys():
                if provider not in configured:
                    keyring_key = _get_key_from_keyring(provider)
                    if keyring_key:
                        configured.append(provider)

        # Check keys file
        if self.keys_file.exists():
            try:
                with open(self.keys_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                api_keys = data.get("api_keys", data)
                for provider, key in api_keys.items():
                    if key and provider not in configured:
                        configured.append(provider)
            except Exception:
                pass

        return sorted(configured)

    def clear_cache(self) -> None:
        """Securely clear all cached keys from memory."""
        for secure_str in self._cache.values():
            if secure_str:
                secure_str.clear()
        self._cache.clear()

    def __del__(self) -> None:
        """Ensure cache is cleared on deletion."""
        self.clear_cache()


# Global instance for convenience
_manager: Optional[APIKeyManager] = None


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider (convenience function).

    Args:
        provider: Provider name

    Returns:
        API key or None
    """
    global _manager
    if _manager is None:
        _manager = APIKeyManager()
    return _manager.get_key(provider)


def set_api_key(provider: str, key: str, use_keyring: bool = False) -> bool:
    """Set API key for a provider (convenience function).

    Args:
        provider: Provider name
        key: API key value
        use_keyring: If True, store in system keyring (more secure)

    Returns:
        True if saved successfully
    """
    global _manager
    if _manager is None:
        _manager = APIKeyManager()
    return _manager.set_key(provider, key, use_keyring=use_keyring)


def is_keyring_available() -> bool:
    """Check if system keyring is available.

    Returns:
        True if keyring package is installed and functional
    """
    return _KEYRING_AVAILABLE


def set_api_key_in_keyring(provider: str, key: str) -> bool:
    """Store API key in system keyring (convenience function).

    This is the recommended way to store API keys as it uses
    the system's encrypted credential storage.

    Args:
        provider: Provider name (e.g., "anthropic", "openai")
        key: API key value

    Returns:
        True if saved successfully
    """
    return set_api_key(provider, key, use_keyring=True)


def delete_api_key_from_keyring(provider: str) -> bool:
    """Delete API key from system keyring.

    Args:
        provider: Provider name

    Returns:
        True if deleted successfully
    """
    return _delete_key_from_keyring(provider)


def get_configured_providers() -> list[str]:
    """Get list of providers with configured API keys.

    Returns:
        List of provider names
    """
    global _manager
    if _manager is None:
        _manager = APIKeyManager()
    return _manager.list_configured_providers()


def clear_api_key_cache() -> None:
    """Securely clear all cached API keys from memory.

    Call this when shutting down or when keys are no longer needed.
    """
    global _manager
    if _manager is not None:
        _manager.clear_cache()


def get_service_key(service: str) -> Optional[str]:
    """Get API key for an external service (convenience function).

    This is for non-LLM services like Finnhub, FRED, etc.

    Args:
        service: Service name (e.g., "finnhub", "fred")

    Returns:
        API key or None
    """
    service = service.lower()

    # Check if it's a known service
    if service not in SERVICE_ENV_VARS:
        logger.warning(
            f"Unknown service: {service}. Known services: {list(SERVICE_ENV_VARS.keys())}"
        )
        return None

    global _manager
    if _manager is None:
        _manager = APIKeyManager()

    # Priority 1: Environment variable
    env_var = SERVICE_ENV_VARS.get(service)
    if env_var:
        env_key = os.environ.get(env_var)
        if env_key:
            _audit_log_secret_access(
                action="loaded",
                provider=f"service:{service}",
                source="environment",
                success=True,
                key_length=len(env_key),
            )
            return env_key

    # Priority 2: System keyring
    keyring_key = _get_key_from_keyring(f"service_{service}")
    if keyring_key:
        _audit_log_secret_access(
            action="loaded",
            provider=f"service:{service}",
            source="keyring",
            success=True,
            key_length=len(keyring_key),
        )
        return keyring_key

    # Priority 3: Keys file (under 'services' section)
    if _manager.keys_file.exists():
        try:
            with open(_manager.keys_file, "r") as f:
                data = yaml.safe_load(f) or {}
            services = data.get("services", {})
            file_key = services.get(service)
            if file_key:
                _audit_log_secret_access(
                    action="loaded",
                    provider=f"service:{service}",
                    source="file",
                    success=True,
                    key_length=len(file_key),
                )
                return file_key if isinstance(file_key, str) else None
        except Exception:
            pass

    _audit_log_secret_access(
        action="not_found",
        provider=f"service:{service}",
        source="none",
        success=False,
        key_length=0,
    )
    return None


def set_service_key(service: str, key: str, use_keyring: bool = False) -> bool:
    """Set API key for an external service.

    Args:
        service: Service name (e.g., "finnhub", "fred")
        key: API key value
        use_keyring: If True, store in system keyring (more secure)

    Returns:
        True if saved successfully
    """
    service = service.lower()

    if service not in SERVICE_ENV_VARS:
        logger.warning(f"Unknown service: {service}")
        return False

    if use_keyring:
        success = _set_key_in_keyring(f"service_{service}", key)
        if success:
            _audit_log_secret_access(
                action="saved",
                provider=f"service:{service}",
                source="keyring",
                success=True,
                key_length=len(key),
            )
        return success

    # Store in file under 'services' section
    global _manager
    if _manager is None:
        _manager = APIKeyManager()

    try:
        _manager.keys_file.parent.mkdir(parents=True, exist_ok=True)

        existing_data: dict[str, Any] = {}
        if _manager.keys_file.exists():
            with open(_manager.keys_file, "r") as f:
                existing_data = yaml.safe_load(f) or {}

        if "services" not in existing_data:
            existing_data["services"] = {}

        existing_data["services"][service] = key

        content = yaml.dump(existing_data, default_flow_style=False)

        # Atomic write with secure permissions
        temp_path = _manager.keys_file.with_suffix(".yaml.tmp")
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
        except Exception:
            os.close(fd)
            raise
        os.rename(temp_path, _manager.keys_file)

        _audit_log_secret_access(
            action="saved",
            provider=f"service:{service}",
            source="file",
            success=True,
            key_length=len(key),
        )
        return True

    except Exception as e:
        logger.error(f"Failed to save service key: {e}")
        return False


def delete_service_key_from_keyring(service: str) -> bool:
    """Delete service API key from system keyring.

    Args:
        service: Service name

    Returns:
        True if deleted successfully
    """
    return _delete_key_from_keyring(f"service_{service}")


def get_configured_services() -> list[str]:
    """Get list of services with configured API keys.

    Returns:
        List of service names
    """
    configured = []

    # Check environment variables
    for service, env_var in SERVICE_ENV_VARS.items():
        if os.environ.get(env_var):
            configured.append(service)

    # Check keyring
    if _KEYRING_AVAILABLE:
        for service in SERVICE_ENV_VARS.keys():
            if service not in configured:
                keyring_key = _get_key_from_keyring(f"service_{service}")
                if keyring_key:
                    configured.append(service)

    # Check keys file
    global _manager
    if _manager is None:
        _manager = APIKeyManager()

    if _manager.keys_file.exists():
        try:
            with open(_manager.keys_file, "r") as f:
                data = yaml.safe_load(f) or {}
            services = data.get("services", {})
            for service, key in services.items():
                if key and service not in configured:
                    configured.append(service)
        except Exception:
            pass

    return sorted(configured)


class APIKeysProxy:
    """Proxy for accessing API keys as a dictionary.

    This provides backward compatibility for code that expects
    victor.config.api_keys.api_keys to be a dict-like object.

    Usage:
        keys = victor.config.api_keys.api_keys
        anthropic_key = keys.get("anthropic")
    """

    def __init__(self) -> None:
        """Initialize the API keys proxy."""
        self._manager: Optional[APIKeyManager] = None

    def _get_manager(self) -> APIKeyManager:
        """Get or create the API key manager."""
        if self._manager is None:
            self._manager = APIKeyManager()
        return self._manager

    def get(self, provider: str, default: Optional[str] = None) -> Optional[str]:
        """Get API key for a provider.

        Args:
            provider: Provider name
            default: Default value if not found

        Returns:
            API key or default value
        """
        key = self._get_manager().get_key(provider)
        return key if key is not None else default

    def __getitem__(self, provider: str) -> Optional[str]:
        """Get API key using dictionary syntax.

        Args:
            provider: Provider name

        Returns:
            API key or None if not found
        """
        return self._get_manager().get_key(provider)

    def __contains__(self, provider: str) -> bool:
        """Check if provider has a configured key.

        Args:
            provider: Provider name

        Returns:
            True if key is configured
        """
        return self._get_manager().get_key(provider) is not None

    def __repr__(self) -> str:
        """Return representation of configured keys."""
        configured = self._get_manager().list_configured_providers()
        return f"APIKeysProxy(configured={configured})"

    def keys(self) -> list[str]:
        """Return list of providers with configured keys."""
        return self._get_manager().list_configured_providers()

    def values(self) -> list[str]:
        """Return list of configured API key values."""
        result = []
        for provider in self._get_manager().list_configured_providers():
            key = self._get_manager().get_key(provider)
            if key:
                # Return masked value for security
                result.append(f"{key[:8]}...{key[-4:]}")
        return result

    def items(self) -> list[tuple[str, str]]:
        """Return list of (provider, masked_key) tuples."""
        result = []
        for provider in self._get_manager().list_configured_providers():
            key = self._get_manager().get_key(provider)
            if key:
                # Return masked value for security
                masked = f"{key[:8]}...{key[-4:]}"
                result.append((provider, masked))
        return result


# Module-level api_keys attribute for backward compatibility
# This provides dict-like access to configured API keys
api_keys = APIKeysProxy()


def create_api_keys_template() -> str:
    """Generate a template for the API keys file.

    Returns:
        YAML template string
    """
    return """# Victor API Keys Configuration
# Location: ~/.victor/api_keys.yaml
#
# Store your API keys here to keep them out of your code repository.
# This file should have restricted permissions (0600).
#
# ============================================================================
# RECOMMENDED: Use system keyring for secure encrypted storage
# ============================================================================
#
# Instead of storing keys in this file, use the system keyring:
#
#   victor keys --set anthropic --keyring    # Store in macOS Keychain / Windows Credential Manager
#   victor keys --set openai --keyring       # Store in Linux Secret Service (GNOME Keyring)
#   victor keys --migrate                    # Migrate all keys from this file to keyring
#
# Keyring provides OS-level encryption and is more secure than file storage.
#
# ============================================================================
# RESOLUTION ORDER (most secure to least)
# ============================================================================
#
# 1. Environment variables (for CI/CD and containers)
# 2. System keyring (encrypted OS storage)
# 3. This file (least secure, not recommended for production)
#
# Environment variables always take precedence. For example:
#   export ANTHROPIC_API_KEY=sk-...   # Overrides any other source
#

api_keys:
  # ============================================================================
  # Premium API Providers
  # ============================================================================
  anthropic: ""     # ANTHROPIC_API_KEY - Claude models (claude-opus-4, claude-sonnet-4)
  openai: ""        # OPENAI_API_KEY - GPT models (gpt-4o, gpt-4-turbo)
  google: ""        # GOOGLE_API_KEY - Gemini models (gemini-2.0-flash, gemini-1.5-pro)
  xai: ""           # XAI_API_KEY - Grok models (grok-2, grok-beta)
  moonshot: ""      # MOONSHOT_API_KEY - Kimi K2 models (kimi-k2-0711-preview)
  deepseek: ""      # DEEPSEEK_API_KEY - DeepSeek models (deepseek-chat, deepseek-reasoner)
  zai: ""           # ZAI_API_KEY - GLM models (glm-4.7, glm-4.6, glm-4.5) - ZhipuAI/智谱AI

  # ============================================================================
  # Free-Tier Providers (great for getting started)
  # ============================================================================
  groqcloud: ""     # GROQCLOUD_API_KEY - Ultra-fast LPU inference (free tier)
  cerebras: ""      # CEREBRAS_API_KEY - Fast inference (free tier)
  mistral: ""       # MISTRAL_API_KEY - 500K tokens/min free tier
  together: ""      # TOGETHER_API_KEY - $25 free credits
  openrouter: ""    # OPENROUTER_API_KEY - Unified gateway (free daily limits)
  fireworks: ""     # FIREWORKS_API_KEY - $1 free credits

  # ============================================================================
  # Enterprise Cloud Providers
  # ============================================================================
  # Google Cloud Vertex AI - Uses Application Default Credentials (ADC)
  # vertex: ""      # GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON)
  #                 # Or use: gcloud auth application-default login

  # Azure OpenAI - Requires endpoint URL as well
  # azure: ""       # AZURE_OPENAI_API_KEY
  #                 # Also set: AZURE_OPENAI_ENDPOINT (e.g., https://your-resource.openai.azure.com)

  # AWS Bedrock - Uses AWS credentials chain
  # bedrock: ""     # AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
  #                 # Or use: aws configure / IAM role

  # Hugging Face Inference API
  huggingface: ""   # HF_TOKEN - Access token from huggingface.co/settings/tokens

  # Replicate - Pay-per-second model hosting
  replicate: ""     # REPLICATE_API_TOKEN

  # ============================================================================
  # Local Providers (no API key required)
  # ============================================================================
  # ollama: (no key needed) - Run: ollama serve
  # lmstudio: (no key needed) - Run LMStudio desktop app
  # vllm: (no key needed) - Run: python -m vllm.entrypoints.openai.api_server

# ============================================================================
# EXTERNAL DATA SERVICES (non-LLM APIs)
# ============================================================================
# These are API keys for market data, financial data, and other external services.
# Use: victor keys --set-service finnhub --keyring

services:
  # Market Data & Financial Services
  finnhub: ""        # FINNHUB_API_KEY - Stock data, news sentiment, analyst estimates (finnhub.io)
  fred: ""           # FRED_API_KEY - Federal Reserve Economic Data (fred.stlouisfed.org)
  alphavantage: ""   # ALPHAVANTAGE_API_KEY - Stock/forex/crypto data (alphavantage.co)
  polygon: ""        # POLYGON_API_KEY - Real-time & historical market data (polygon.io)
  tiingo: ""         # TIINGO_API_KEY - Stock/crypto/forex data (tiingo.com)
  iex: ""            # IEX_API_KEY - IEX Cloud market data (iexcloud.io)
  quandl: ""         # QUANDL_API_KEY - Financial datasets, now Nasdaq Data Link
  nasdaq: ""         # NASDAQ_API_KEY - Nasdaq Data Link (data.nasdaq.com)

  # News & Sentiment
  newsapi: ""        # NEWSAPI_API_KEY - News aggregation (newsapi.org)
  marketaux: ""      # MARKETAUX_API_KEY - Financial news API (marketaux.com)

  # SEC & Regulatory
  sec: ""            # SEC_API_KEY - SEC EDGAR API (optional, for higher rate limits)

  # Other Data Services
  openweather: ""    # OPENWEATHER_API_KEY - Weather data (openweathermap.org)
  geocoding: ""      # GEOCODING_API_KEY - Geocoding services
"""
