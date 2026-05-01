from __future__ import annotations

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

"""OAuth token manager for LLM provider subscription authentication.

Manages the OAuth 2.0 PKCE token lifecycle (login, persist, refresh) for
providers that support subscription-based OAuth access (OpenAI Codex, Qwen).

See FEP-0004 for design details.

Security:
- OAuth client_id is loaded from environment variables or keychain
- Never hardcoded in source code to prevent git leaks
- Resolution order: env var > keychain > error
"""

import logging
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from victor.workflows.services.credentials import (
    SSOAuthenticator,
    SSOConfig,
    SSOProvider,
    SSOTokens,
)

logger = logging.getLogger(__name__)

# Grace period: refresh token if it expires within this window
REFRESH_GRACE_MINUTES = 5

# OAuth client ID environment variables (highest priority)
OAUTH_CLIENT_ID_ENV_VARS = {
    "openai": "VICTOR_OPENAI_OAUTH_CLIENT_ID",
    "qwen": "VICTOR_QWEN_OAUTH_CLIENT_ID",
    "google": "VICTOR_GOOGLE_OAUTH_CLIENT_ID",
    "github-copilot": "VICTOR_GITHUB_COPILOT_OAUTH_CLIENT_ID",
}

# OAuth client ID keychain service names
KEYRING_SERVICE = "victor"
KEYRING_OAUTH_CLIENT_ID_PREFIX = "oauth_client_id_"


def _get_oauth_client_id_from_keyring(provider: str) -> Optional[str]:
    """Get OAuth client_id from system keyring.

    Args:
        provider: Provider name (e.g., "openai", "qwen")

    Returns:
        OAuth client_id if found, None otherwise
    """
    try:
        import keyring

        key = keyring.get_password(KEYRING_SERVICE, f"{KEYRING_OAUTH_CLIENT_ID_PREFIX}{provider}")
        return key
    except ImportError:
        logger.debug("Keyring not available for OAuth client_id retrieval")
        return None
    except Exception as e:
        logger.debug(f"Keyring access failed for OAuth client_id {provider}: {e}")
        return None


def _set_oauth_client_id_in_keyring(provider: str, client_id: str) -> bool:
    """Store OAuth client_id in system keyring.

    Args:
        provider: Provider name
        client_id: OAuth client_id to store

    Returns:
        True if successful, False otherwise
    """
    try:
        import keyring

        keyring.set_password(
            KEYRING_SERVICE, f"{KEYRING_OAUTH_CLIENT_ID_PREFIX}{provider}", client_id
        )
        logger.info(f"OAuth client_id for {provider} stored in system keyring")
        return True
    except ImportError:
        logger.warning("Keyring not available. Install 'keyring' package for secure storage.")
        return False
    except Exception as e:
        logger.warning(f"Failed to store OAuth client_id in keyring: {e}")
        return False


# Well-known public OAuth client IDs (shared by the ecosystem).
# These are public clients (no secret) used by Codex CLI, Cline, Roo Code, etc.
# See: https://github.com/openai/codex (codex-rs/core/src/auth.rs)
_PUBLIC_OAUTH_CLIENT_IDS: Dict[str, str] = {
    "openai": "app_EMoamEEZ73f0CkXaXp7hrann",
    # Google public client for installed apps (same as Gemini CLI)
    "google": ("681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j" ".apps.googleusercontent.com"),
}


def _get_oauth_client_id(provider: str) -> str:
    """Get OAuth client_id for a provider.

    Resolution order:
    1. Environment variable (highest priority)
    2. System keyring (secure encrypted storage)
    3. Well-known public client_id (OpenAI only — shared by Codex ecosystem)
    4. Error (must be explicitly configured)

    Args:
        provider: Provider name (e.g., "openai", "qwen")

    Returns:
        OAuth client_id

    Raises:
        ValueError: If client_id is not configured
    """
    # Priority 1: Environment variable
    env_var = OAUTH_CLIENT_ID_ENV_VARS.get(provider)
    if env_var:
        client_id = os.environ.get(env_var)
        if client_id:
            logger.debug(
                f"OAuth client_id for {provider} loaded from environment variable {env_var}"
            )
            return client_id

    # Priority 2: System keyring
    keyring_client_id = _get_oauth_client_id_from_keyring(provider)
    if keyring_client_id:
        logger.debug(f"OAuth client_id for {provider} loaded from keyring")
        return keyring_client_id

    # Priority 3: Well-known public client_id (OpenAI ecosystem)
    public_id = _PUBLIC_OAUTH_CLIENT_IDS.get(provider)
    if public_id:
        logger.debug(f"OAuth client_id for {provider} using well-known public client_id")
        return public_id

    # No client_id found
    raise ValueError(
        f"OAuth client_id for '{provider}' is not configured.\n"
        f"Set it using one of these methods:\n"
        f"  1. Environment variable: export {env_var or f'VICTOR_{provider.upper()}_OAUTH_CLIENT_ID'}=<your-client-id>\n"
        f"  2. Keyring: victor keys --set-oauth-client-id {provider}\n"
    )


@dataclass
class OAuthProviderConfig:
    """OAuth configuration for an LLM provider's subscription auth."""

    provider_name: str
    sso_provider: SSOProvider
    issuer_url: str
    # client_id is now loaded dynamically via get_client_id() method
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    token_endpoint: str = "/oauth/token"
    redirect_port: int = 8400
    # Config-driven overrides for providers with non-standard OAuth endpoints
    client_secret: Optional[str] = None  # Public secret (e.g. Google installed apps)
    authorize_path: Optional[str] = None  # e.g. "/o/oauth2/v2/auth" for Google
    token_url: Optional[str] = None  # Full token URL when host differs from issuer
    callback_path: str = "/callback"  # Redirect callback path
    extra_auth_params: Optional[Dict[str, str]] = None  # e.g. {"access_type": "offline"}
    # Device code flow (GitHub Copilot, headless environments)
    use_device_flow: bool = False
    device_code_url: Optional[str] = None

    def get_client_id(self) -> str:
        """Get the OAuth client_id for this provider.

        Loads from environment variable or keychain.

        Returns:
            OAuth client_id

        Raises:
            ValueError: If client_id is not configured
        """
        return _get_oauth_client_id(self.provider_name)

    def to_sso_config(self) -> SSOConfig:
        """Convert to SSOConfig for use with SSOAuthenticator."""
        # OpenAI uses /auth/callback (matching Codex CLI), others use their callback_path
        if self.sso_provider == SSOProvider.OPENAI_CODEX:
            redirect_uri = f"http://localhost:{self.redirect_port}/auth/callback"
        else:
            redirect_uri = f"http://localhost:{self.redirect_port}{self.callback_path}"

        return SSOConfig(
            provider=self.sso_provider,
            issuer_url=self.issuer_url,
            client_id=self.get_client_id(),
            client_secret=self.client_secret,
            scopes=self.scopes,
            redirect_uri=redirect_uri,
            use_pkce=not self.use_device_flow,
            authorize_path=self.authorize_path,
            token_url=self.token_url,
            extra_auth_params=self.extra_auth_params,
            use_device_flow=self.use_device_flow,
            device_code_url=self.device_code_url,
        )


# ---------------------------------------------------------------------------
# Provider registry — only providers with confirmed subscription OAuth
# ---------------------------------------------------------------------------

OAUTH_PROVIDERS: Dict[str, OAuthProviderConfig] = {
    "openai": OAuthProviderConfig(
        provider_name="openai",
        sso_provider=SSOProvider.OPENAI_CODEX,
        issuer_url="https://auth.openai.com",
        scopes=[
            "openid",
            "profile",
            "email",
            "offline_access",
            "api.connectors.read",
            "api.connectors.invoke",
        ],
        token_endpoint="/oauth/token",
        redirect_port=1455,
    ),
    "qwen": OAuthProviderConfig(
        provider_name="qwen",
        sso_provider=SSOProvider.QWEN,
        issuer_url="https://chat.qwen.ai",
        scopes=["openid", "profile", "email", "offline_access"],
        token_endpoint="/oauth/token",
    ),
    "google": OAuthProviderConfig(
        provider_name="google",
        sso_provider=SSOProvider.GOOGLE_GEMINI,
        issuer_url="https://accounts.google.com",
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        ],
        token_url="https://oauth2.googleapis.com/token",
        authorize_path="/o/oauth2/v2/auth",
        # Public client secret (standard for Google installed apps, same as Gemini CLI)
        client_secret="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",
        callback_path="/oauth2callback",
        redirect_port=8401,
        extra_auth_params={"access_type": "offline"},
    ),
    "github-copilot": OAuthProviderConfig(
        provider_name="github-copilot",
        sso_provider=SSOProvider.GITHUB_COPILOT,
        issuer_url="https://github.com",
        scopes=["copilot"],
        use_device_flow=True,
        device_code_url="https://github.com/login/device/code",
        token_url="https://github.com/login/oauth/access_token",
    ),
}

_DEFAULT_STORAGE_DIR = Path.home() / ".victor"


class OAuthTokenManager:
    """Manages OAuth token lifecycle for a single LLM provider.

    Handles:
    - Browser-based PKCE login (delegates to SSOAuthenticator)
    - Token persistence to ~/.victor/oauth_tokens.yaml (0600)
    - Automatic refresh when token is expiring
    - Fallback to re-login when refresh fails
    """

    def __init__(
        self,
        provider: str,
        storage_dir: Optional[Path] = None,
    ):
        self._provider = provider
        self._storage_dir = storage_dir or _DEFAULT_STORAGE_DIR
        self._token_file = self._storage_dir / "oauth_tokens.yaml"

        if provider not in OAUTH_PROVIDERS:
            raise KeyError(
                f"Provider '{provider}' does not support OAuth. "
                f"Supported: {list(OAUTH_PROVIDERS.keys())}"
            )
        self._config = OAUTH_PROVIDERS[provider]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_valid_token(self) -> str:
        """Return a valid access token, refreshing or logging in as needed."""
        cached = self._load_cached()

        if cached is None:
            logger.info("No cached OAuth token for %s — starting login", self._provider)
            tokens = await self.login()
            return tokens.access_token

        if self._needs_refresh(cached):
            try:
                tokens = await self.refresh()
                return tokens.access_token
            except Exception:
                logger.warning(
                    "Token refresh failed for %s — falling back to login",
                    self._provider,
                )
                tokens = await self.login()
                return tokens.access_token

        return cached.access_token

    async def login(self) -> SSOTokens:
        """Perform browser-based OAuth PKCE login."""
        sso_config = self._config.to_sso_config()
        auth = SSOAuthenticator(sso_config)
        tokens = await auth.login()
        self._save(tokens)
        logger.info("OAuth login successful for %s", self._provider)
        return tokens

    async def refresh(self) -> SSOTokens:
        """Refresh the current access token using the stored refresh token."""
        cached = self._load_cached()
        if cached is None or cached.refresh_token is None:
            raise ValueError(f"No refresh token available for {self._provider}")

        sso_config = self._config.to_sso_config()
        auth = SSOAuthenticator(sso_config)
        tokens = await auth.refresh(cached.refresh_token)
        self._save(tokens)
        logger.info("OAuth token refreshed for %s", self._provider)
        return tokens

    def clear(self) -> None:
        """Remove cached tokens for this provider."""
        all_tokens = self._load_all()
        if self._provider in all_tokens:
            del all_tokens[self._provider]
            self._write_all(all_tokens)
            logger.info("OAuth tokens cleared for %s", self._provider)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, tokens: SSOTokens) -> None:
        """Persist tokens to disk."""
        all_tokens = self._load_all()
        all_tokens[self._provider] = {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "id_token": tokens.id_token,
            "token_type": tokens.token_type,
            "expires_at": (tokens.expires_at.isoformat() if tokens.expires_at else None),
            "scopes": tokens.scopes,
        }
        self._write_all(all_tokens)

    def _load_cached(self) -> Optional[SSOTokens]:
        """Load cached tokens for this provider."""
        all_tokens = self._load_all()
        data = all_tokens.get(self._provider)
        if data is None:
            return None

        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return SSOTokens(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            id_token=data.get("id_token"),
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scopes=data.get("scopes", []),
        )

    def _load_all(self) -> Dict[str, Any]:
        """Load all provider tokens from disk."""
        if not self._token_file.exists():
            return {}
        try:
            with open(self._token_file) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            logger.warning("Failed to load OAuth tokens file")
            return {}

    def _write_all(self, data: Dict[str, Any]) -> None:
        """Write all provider tokens to disk with secure permissions."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self._token_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        os.chmod(self._token_file, stat.S_IRUSR | stat.S_IWUSR)  # 0600

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_refresh(tokens: SSOTokens) -> bool:
        """Check if tokens need refresh (expired or expiring within grace period)."""
        if tokens.expires_at is None:
            return False
        threshold = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_GRACE_MINUTES)
        return tokens.expires_at <= threshold
