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


@dataclass
class OAuthProviderConfig:
    """OAuth configuration for an LLM provider's subscription auth."""

    provider_name: str
    sso_provider: SSOProvider
    issuer_url: str
    client_id: str
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    token_endpoint: str = "/oauth/token"
    redirect_port: int = 8400

    def to_sso_config(self) -> SSOConfig:
        """Convert to SSOConfig for use with SSOAuthenticator."""
        return SSOConfig(
            provider=self.sso_provider,
            issuer_url=self.issuer_url,
            client_id=self.client_id,
            scopes=self.scopes,
            redirect_uri=f"http://localhost:{self.redirect_port}/callback",
            use_pkce=True,
        )


# ---------------------------------------------------------------------------
# Provider registry — only providers with confirmed subscription OAuth
# ---------------------------------------------------------------------------

OAUTH_PROVIDERS: Dict[str, OAuthProviderConfig] = {
    "openai": OAuthProviderConfig(
        provider_name="openai",
        sso_provider=SSOProvider.OPENAI_CODEX,
        issuer_url="https://auth.openai.com",
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        scopes=["openid", "profile", "email", "offline_access"],
        token_endpoint="/oauth/token",
    ),
    "qwen": OAuthProviderConfig(
        provider_name="qwen",
        sso_provider=SSOProvider.QWEN,
        issuer_url="https://chat.qwen.ai",
        client_id="qwen-code",  # built-in client; override via QWEN_OAUTH_CLIENT_ID
        scopes=["openid", "profile", "email", "offline_access"],
        token_endpoint="/oauth/token",
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
            "expires_at": (
                tokens.expires_at.isoformat() if tokens.expires_at else None
            ),
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
