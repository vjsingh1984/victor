"""
Configuration management for Code Review Assistant.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ReviewConfig:
    """Configuration for code review process."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration."""
        self.config = config_dict or {}

    @property
    def max_complexity(self) -> int:
        """Maximum allowed cyclomatic complexity."""
        return self.config.get("review", {}).get("max_complexity", 10)

    @property
    def max_line_length(self) -> int:
        """Maximum line length."""
        return self.config.get("review", {}).get("max_line_length", 100)

    @property
    def severity_levels(self) -> list:
        """Severity levels to report."""
        return self.config.get("review", {}).get("severity", ["high", "medium", "low"])

    @property
    def enabled_checks(self) -> list:
        """Enabled check types."""
        return self.config.get("review", {}).get("checks", ["security", "style", "complexity", "quality"])

    @property
    def ignore_patterns(self) -> list:
        """File patterns to ignore."""
        return self.config.get("review", {}).get("ignore", [])

    @property
    def custom_rules(self) -> list:
        """Custom review rules."""
        return self.config.get("review", {}).get("rules", [])

    @property
    def provider_name(self) -> str:
        """LLM provider name."""
        return self.config.get("provider", {}).get("name", "anthropic")

    @property
    def provider_model(self) -> str:
        """LLM model name."""
        return self.config.get("provider", {}).get("model", "claude-sonnet-4-5")

    @property
    def provider_temperature(self) -> float:
        """LLM temperature."""
        return self.config.get("provider", {}).get("temperature", 0.0)

    @property
    def team_formation(self) -> str:
        """Default team formation."""
        return self.config.get("team", {}).get("formation", "parallel")

    @property
    def team_roles(self) -> list:
        """Default team roles."""
        return self.config.get("team", {}).get("roles", [])


def load_config(config_path: Optional[str] = None) -> ReviewConfig:
    """Load configuration from file.

    Args:
        config_path: Path to config file. If None, searches for .victor-review.yaml

    Returns:
        ReviewConfig instance
    """
    if config_path:
        config_file = Path(config_path)
    else:
        # Search for config file in current directory and parents
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents):
            config_file = parent / ".victor-review.yaml"
            if config_file.exists():
                break
        else:
            # No config file found, use defaults
            return ReviewConfig()

    if not config_file.exists():
        return ReviewConfig()

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    return ReviewConfig(config_dict)
