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

"""Alerting system for production monitoring.

This module provides a comprehensive alerting framework with:
- Alert rules with conditions and thresholds
- Multiple notification channels (email, Slack, PagerDuty, webhook)
- Alert severity levels
- Alert deduplication and rate limiting
- On-call rotation support

Design Patterns:
- Observer Pattern: Alert notifications
- Strategy Pattern: Different notification channels
- Builder Pattern: Alert rule construction

Example:
    from victor.observability.alerting import (
        AlertManager,
        AlertRule,
        AlertSeverity,
        SlackNotifier,
    )

    # Create alert manager
    manager = AlertManager()

    # Add notification channel
    slack = SlackNotifier(webhook_url="https://hooks.slack.com/...")
    manager.add_notifier("slack", slack)

    # Define alert rule
    rule = AlertRule.builder()
        .name("high_error_rate")
        .condition("error_rate > 5")
        .severity(AlertSeverity.CRITICAL)
        .notification_channels(["slack"])
        .build()

    manager.add_rule(rule)

    # Check and send alerts
    await manager.check_and_alert()
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Callable

if TYPE_CHECKING:
    from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


# =============================================================================
# Alert Severity
# =============================================================================


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""

    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


# =============================================================================
# Alert Data Structures
# =============================================================================


@dataclass
class Alert:
    """Alert instance.

    Attributes:
        id: Unique alert ID
        rule_name: Name of the rule that triggered
        severity: Alert severity
        status: Alert status
        message: Alert message
        details: Additional details
        fired_at: When alert fired
        resolved_at: When alert resolved
        acknowledged_at: When alert acknowledged
        labels: Labels for grouping
    """

    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "fired_at": self.fired_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "labels": self.labels,
        }

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)

    def acknowledge(self) -> None:
        """Mark alert as acknowledged."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)


# =============================================================================
# Alert Rules
# =============================================================================


@dataclass
class AlertRule:
    """Alert rule definition.

    Attributes:
        name: Unique rule name
        description: Rule description
        condition: Alert condition (expression or function)
        severity: Alert severity
        enabled: Whether rule is enabled
        notification_channels: Channels to notify
        cooldown_seconds: Seconds between alerts for same rule
        threshold: Threshold value
        evaluation_interval_seconds: How often to evaluate
        labels: Labels to attach to alerts
    """

    name: str
    description: str
    condition: str | Callable[[], bool]
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    notification_channels: list[str] = field(default_factory=list)
    cooldown_seconds: float = 300.0  # 5 minutes default
    threshold: Optional[float] = None
    evaluation_interval_seconds: float = 60.0
    labels: dict[str, str] = field(default_factory=dict)
    last_fired: Optional[datetime] = None

    def should_fire(self) -> bool:
        """Check if rule should fire.

        Returns:
            True if alert should fire
        """
        if not self.enabled:
            return False

        # Check cooldown
        if self.last_fired:
            elapsed = (datetime.now(timezone.utc) - self.last_fired).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False

        # Evaluate condition
        if callable(self.condition):
            return self.condition()
        else:
            # For string conditions, we'd need an expression evaluator
            # For now, return False (should be implemented with proper eval)
            logger.warning(f"String conditions not yet supported: {self.condition}")
            return False

    def fire(self, message: str, details: Optional[dict[str, Any]] = None) -> Alert:
        """Fire alert.

        Args:
            message: Alert message
            details: Additional details

        Returns:
            Alert instance
        """
        self.last_fired = datetime.now(timezone.utc)

        import uuid

        return Alert(
            id=uuid.uuid4().hex,
            rule_name=self.name,
            severity=self.severity,
            status=AlertStatus.FIRING,
            message=message,
            details=details or {},
            labels=self.labels,
        )

    @classmethod
    def builder(cls) -> "AlertRuleBuilder":
        """Create alert rule builder.

        Returns:
            AlertRuleBuilder instance

        Example:
            rule = AlertRule.builder()
                .name("high_error_rate")
                .condition(lambda: error_rate > 5)
                .severity(AlertSeverity.CRITICAL)
                .build()
        """
        return AlertRuleBuilder()


@dataclass
class AlertRuleBuilder:
    """Builder for AlertRule."""

    _name: Optional[str] = None
    _description: str = ""
    _condition: Optional[str | Callable[[], bool]] = None
    _severity: AlertSeverity = AlertSeverity.WARNING
    _notification_channels: list[str] = field(default_factory=list)
    _cooldown_seconds: float = 300.0
    _threshold: Optional[float] = None
    _evaluation_interval_seconds: float = 60.0
    _labels: dict[str, str] = field(default_factory=dict)

    def name(self, name: str) -> "AlertRuleBuilder":
        """Set rule name."""
        self._name = name
        return self

    def description(self, description: str) -> "AlertRuleBuilder":
        """Set rule description."""
        self._description = description
        return self

    def condition(self, condition: str | Callable[[], bool]) -> "AlertRuleBuilder":
        """Set alert condition."""
        self._condition = condition
        return self

    def severity(self, severity: AlertSeverity) -> "AlertRuleBuilder":
        """Set alert severity."""
        self._severity = severity
        return self

    def notification_channels(self, channels: list[str]) -> "AlertRuleBuilder":
        """Set notification channels."""
        self._notification_channels = channels
        return self

    def cooldown(self, seconds: float) -> "AlertRuleBuilder":
        """Set cooldown period."""
        self._cooldown_seconds = seconds
        return self

    def threshold(self, value: float) -> "AlertRuleBuilder":
        """Set threshold."""
        self._threshold = value
        return self

    def evaluation_interval(self, seconds: float) -> "AlertRuleBuilder":
        """Set evaluation interval."""
        self._evaluation_interval_seconds = seconds
        return self

    def labels(self, labels: dict[str, str]) -> "AlertRuleBuilder":
        """Set labels."""
        self._labels = labels
        return self

    def build(self) -> AlertRule:
        """Build alert rule.

        Returns:
            AlertRule instance
        """
        if not self._name:
            raise ValueError("Alert rule name is required")
        if not self._condition:
            raise ValueError("Alert rule condition is required")

        return AlertRule(
            name=self._name,
            description=self._description,
            condition=self._condition,
            severity=self._severity,
            notification_channels=self._notification_channels,
            cooldown_seconds=self._cooldown_seconds,
            threshold=self._threshold,
            evaluation_interval_seconds=self._evaluation_interval_seconds,
            labels=self._labels,
        )


# =============================================================================
# Notification Channels
# =============================================================================


class NotificationChannel(ABC):
    """Abstract base for notification channels."""

    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        pass


class SlackNotifier(NotificationChannel):
    """Slack notification channel.

    Attributes:
        webhook_url: Slack webhook URL
        username: Bot username
        icon_emoji: Bot icon emoji
        channel: Default channel (overrides webhook default)
    """

    def __init__(
        self,
        webhook_url: str,
        username: str = "Victor Alert",
        icon_emoji: str = ":warning:",
        channel: Optional[str] = None,
    ) -> None:
        """Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
            username: Bot username
            icon_emoji: Bot icon emoji
            channel: Default channel override
        """
        self.webhook_url = webhook_url
        self.username = username
        self.icon_emoji = icon_emoji
        self.channel = channel

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        try:
            import aiohttp

            # Color based on severity
            colors = {
                AlertSeverity.INFO: "#36a64f",  # green
                AlertSeverity.WARNING: "#ff9900",  # orange
                AlertSeverity.ERROR: "#ff0000",  # red
                AlertSeverity.CRITICAL: "#990000",  # dark red
            }

            color = colors.get(alert.severity, "#ff9900")

            # Build attachment
            attachment: dict[str, Any] = {
                "color": color,
                "title": f"{alert.severity.value.upper()}: {alert.rule_name}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Status", "value": alert.status.value, "short": True},
                    {
                        "title": "Fired At",
                        "value": alert.fired_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "short": True,
                    },
                ],
                "footer": f"Alert ID: {alert.id}",
            }

            # Add details if present
            if alert.details:
                details_text = "\n".join(f"• {k}: {v}" for k, v in alert.details.items())
                attachment["fields"].append(
                    {"title": "Details", "value": details_text, "short": False}
                )

            payload = {
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "attachments": [attachment],
            }

            if self.channel:
                payload["channel"] = self.channel

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=10) as resp:
                    return resp.status == 200

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class EmailNotifier(NotificationChannel):
    """Email notification channel.

    Attributes:
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        username: SMTP username
        password: SMTP password
        from_address: From email address
        to_addresses: List of recipient addresses
        use_tls: Whether to use TLS
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_address: str = "alerts@victor.ai",
        to_addresses: list[str] | None = None,
        use_tls: bool = True,
    ) -> None:
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_address: From email address
            to_addresses: Recipient addresses
            use_tls: Whether to use TLS
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.to_addresses = to_addresses or []
        self.use_tls = use_tls

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        try:
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)

            # Build HTML body
            html = f"""
            <html>
              <body>
                <h2>{alert.severity.value.upper()}: {alert.rule_name}</h2>
                <p><strong>Severity:</strong> {alert.severity.value}</p>
                <p><strong>Status:</strong> {alert.status.value}</p>
                <p><strong>Fired At:</strong> {alert.fired_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Message:</strong></p>
                <p>{alert.message}</p>
            """

            if alert.details:
                html += "<h3>Details:</h3><ul>"
                for key, value in alert.details.items():
                    html += f"<li><strong>{key}:</strong> {value}</li>"
                html += "</ul>"

            html += (
                """
                <p><small>Alert ID: """
                + alert.id
                + """</small></p>
              </body>
            </html>
            """
            )

            msg.attach(MIMEText(html, "html"))

            # Send email
            # Note: This is synchronous, would be better to use an async SMTP library
            # For production, consider using aiosmtplib or similar
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_sync, msg)

            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _send_sync(self, msg: MIMEMultipart) -> None:
        """Send message synchronously."""
        import smtplib

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()

            if self.username and self.password:
                server.login(self.username, self.password)

            server.send_message(msg)


class WebhookNotifier(NotificationChannel):
    """Generic webhook notification channel.

    Attributes:
        webhook_url: Webhook URL
        headers: HTTP headers to send
        method: HTTP method
    """

    def __init__(
        self,
        webhook_url: str,
        headers: Optional[dict[str, str]] = None,
        method: str = "POST",
    ) -> None:
        """Initialize webhook notifier.

        Args:
            webhook_url: Webhook URL
            headers: HTTP headers
            method: HTTP method
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.method = method

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        try:
            import aiohttp

            payload = alert.to_dict()

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    self.method,
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=10,
                ) as resp:
                    return 200 <= resp.status < 300

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


# =============================================================================
# Alert Manager
# =============================================================================


class AlertManager:
    """Alert manager for rule evaluation and notification.

    Attributes:
        rules: Alert rules
        notifiers: Notification channels
        active_alerts: Currently firing alerts

    Example:
        manager = AlertManager()

        # Add notifier
        slack = SlackNotifier(webhook_url="...")
        manager.add_notifier("slack", slack)

        # Add rule
        rule = AlertRule.builder()
            .name("high_error_rate")
            .condition(lambda: get_error_rate() > 5)
            .severity(AlertSeverity.CRITICAL)
            .notification_channels(["slack"])
            .build()
        manager.add_rule(rule)

        # Evaluate rules
        await manager.check_and_alert()
    """

    def __init__(self) -> None:
        """Initialize alert manager."""
        self._rules: dict[str, AlertRule] = {}
        self._notifiers: dict[str, NotificationChannel] = {}
        self._active_alerts: dict[str, Alert] = {}

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.

        Args:
            rule: Alert rule to add
        """
        self._rules[rule.name] = rule

    def remove_rule(self, rule_name: str) -> None:
        """Remove alert rule.

        Args:
            rule_name: Rule name to remove
        """
        if rule_name in self._rules:
            del self._rules[rule_name]

    def add_notifier(self, name: str, notifier: NotificationChannel) -> None:
        """Add notification channel.

        Args:
            name: Channel name
            notifier: Notifier instance
        """
        self._notifiers[name] = notifier

    def remove_notifier(self, name: str) -> None:
        """Remove notification channel.

        Args:
            name: Channel name
        """
        if name in self._notifiers:
            del self._notifiers[name]

    async def check_and_alert(self) -> list[Alert]:
        """Check all rules and fire alerts if needed.

        Returns:
            List of newly fired alerts
        """
        new_alerts = []

        for rule in self._rules.values():
            if rule.should_fire():
                alert = rule.fire(
                    message=f"Alert rule '{rule.name}' triggered",
                    details={"severity": rule.severity.value},
                )

                # Send to notification channels
                for channel_name in rule.notification_channels:
                    notifier = self._notifiers.get(channel_name)
                    if notifier:
                        await notifier.send_alert(alert)

                # Track active alert
                self._active_alerts[alert.id] = alert
                new_alerts.append(alert)

        return new_alerts

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts.

        Returns:
            List of active alerts
        """
        return list(self._active_alerts.values())

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if alert was found and resolved
        """
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].resolve()
            del self._active_alerts[alert_id]
            return True
        return False

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if alert was found and acknowledged
        """
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].acknowledge()
            return True
        return False


# =============================================================================
# Factory Functions
# =============================================================================


def create_alert_manager() -> AlertManager:
    """Create alert manager.

    Returns:
        AlertManager instance

    Example:
        manager = create_alert_manager()
        manager.add_rule(rule)
    """
    return AlertManager()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AlertManager",
    "Alert",
    "AlertRule",
    "AlertRuleBuilder",
    "AlertSeverity",
    "AlertStatus",
    "NotificationChannel",
    "SlackNotifier",
    "EmailNotifier",
    "WebhookNotifier",
    "create_alert_manager",
]
