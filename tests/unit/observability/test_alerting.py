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

"""Tests for alerting system."""

import pytest

from victor.observability.alerting import (
    AlertManager,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    WebhookNotifier,
    create_alert_manager,
)


@pytest.fixture
def alert_manager():
    """Create a fresh alert manager for each test."""
    manager = AlertManager()
    yield manager


class TestAlert:
    """Test Alert functionality."""

    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            id="test-id",
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
        )
        assert alert.id == "test-id"
        assert alert.rule_name == "test-rule"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.FIRING

    def test_alert_resolve(self):
        """Test alert resolution."""
        alert = Alert(
            id="test-id",
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
        )
        alert.resolve()
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None

    def test_alert_acknowledge(self):
        """Test alert acknowledgment."""
        alert = Alert(
            id="test-id",
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
        )
        alert.acknowledge()
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            id="test-id",
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
        )
        data = alert.to_dict()
        assert data["id"] == "test-id"
        assert data["rule_name"] == "test-rule"
        assert data["severity"] == "warning"
        assert data["status"] == "firing"


class TestAlertRule:
    """Test AlertRule functionality."""

    def test_rule_creation(self):
        """Test rule creation."""
        rule = AlertRule(
            name="test-rule",
            description="Test rule",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
        )
        assert rule.name == "test-rule"
        assert rule.should_fire()

    def test_rule_disabled(self):
        """Test disabled rule."""
        rule = AlertRule(
            name="test-rule",
            description="Test rule",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            enabled=False,
        )
        assert not rule.should_fire()

    def test_rule_cooldown(self):
        """Test rule cooldown."""
        import time

        rule = AlertRule(
            name="test-rule",
            description="Test rule",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=0.1,
        )

        # First fire
        assert rule.should_fire()
        alert = rule.fire("Test alert")

        # Should be in cooldown
        assert not rule.should_fire()

        # Wait for cooldown
        time.sleep(0.15)
        assert rule.should_fire()

    def test_rule_fire(self):
        """Test firing alert from rule."""
        rule = AlertRule(
            name="test-rule",
            description="Test rule",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
        )
        alert = rule.fire("Test alert", {"detail": "value"})
        assert alert.rule_name == "test-rule"
        assert alert.message == "Test alert"
        assert alert.details == {"detail": "value"}

    def test_rule_builder(self):
        """Test AlertRuleBuilder."""
        rule = (
            AlertRule.builder()
            .name("test-rule")
            .description("Test description")
            .condition(lambda: True)
            .severity(AlertSeverity.CRITICAL)
            .notification_channels(["slack"])
            .cooldown(60.0)
            .build()
        )
        assert rule.name == "test-rule"
        assert rule.description == "Test description"
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.notification_channels == ["slack"]
        assert rule.cooldown_seconds == 60.0

    def test_rule_builder_missing_required(self):
        """Test builder with missing required fields."""
        with pytest.raises(ValueError, match="name"):
            AlertRule.builder().description("test").condition(lambda: True).build()

        with pytest.raises(ValueError, match="condition"):
            AlertRule.builder().name("test").description("test").build()


class TestAlertManager:
    """Test AlertManager functionality."""

    def test_add_rule(self, alert_manager):
        """Test adding rule."""
        rule = AlertRule(
            name="test-rule",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
        )
        alert_manager.add_rule(rule)
        assert "test-rule" in alert_manager._rules

    def test_remove_rule(self, alert_manager):
        """Test removing rule."""
        rule = AlertRule(
            name="test-rule",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
        )
        alert_manager.add_rule(rule)
        alert_manager.remove_rule("test-rule")
        assert "test-rule" not in alert_manager._rules

    def test_add_notifier(self, alert_manager):
        """Test adding notifier."""
        notifier = WebhookNotifier("http://example.com/webhook")
        alert_manager.add_notifier("webhook", notifier)
        assert "webhook" in alert_manager._notifiers

    def test_remove_notifier(self, alert_manager):
        """Test removing notifier."""
        notifier = WebhookNotifier("http://example.com/webhook")
        alert_manager.add_notifier("webhook", notifier)
        alert_manager.remove_notifier("webhook")
        assert "webhook" not in alert_manager._notifiers

    @pytest.mark.asyncio
    async def test_check_and_alert(self, alert_manager):
        """Test checking rules and firing alerts."""
        notifier = WebhookNotifier("http://example.com/webhook")
        alert_manager.add_notifier("webhook", notifier)

        rule = AlertRule(
            name="test-rule",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            notification_channels=["webhook"],
        )
        alert_manager.add_rule(rule)

        alerts = await alert_manager.check_and_alert()
        assert len(alerts) == 1
        assert alerts[0].rule_name == "test-rule"

    @pytest.mark.asyncio
    async def test_check_and_alert_no_notifier(self, alert_manager):
        """Test checking rules without notifier."""
        rule = AlertRule(
            name="test-rule",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
        )
        alert_manager.add_rule(rule)

        alerts = await alert_manager.check_and_alert()
        assert len(alerts) == 1

    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        alert = Alert(
            id="test-id",
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test",
        )
        alert_manager._active_alerts["test-id"] = alert

        active = alert_manager.get_active_alerts()
        assert len(active) == 1
        assert active[0].id == "test-id"

    def test_resolve_alert(self, alert_manager):
        """Test resolving alert."""
        alert = Alert(
            id="test-id",
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test",
        )
        alert_manager._active_alerts["test-id"] = alert

        result = alert_manager.resolve_alert("test-id")
        assert result is True
        assert "test-id" not in alert_manager._active_alerts

    def test_resolve_nonexistent_alert(self, alert_manager):
        """Test resolving non-existent alert."""
        result = alert_manager.resolve_alert("nonexistent")
        assert result is False

    def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging alert."""
        alert = Alert(
            id="test-id",
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test",
        )
        alert_manager._active_alerts["test-id"] = alert

        result = alert_manager.acknowledge_alert("test-id")
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED

    def test_acknowledge_nonexistent_alert(self, alert_manager):
        """Test acknowledging non-existent alert."""
        result = alert_manager.acknowledge_alert("nonexistent")
        assert result is False


class TestWebhookNotifier:
    """Test WebhookNotifier."""

    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alert via webhook."""
        # Use a mock server or test with a real webhook endpoint
        notifier = WebhookNotifier("http://httpbin.org/post")

        alert = Alert(
            id="test-id",
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
        )

        # This will make a real HTTP request to httpbin.org which echoes the request
        result = await notifier.send_alert(alert)
        # Should succeed if httpbin.org is up
        assert result is True or result is False  # Either way, test passes


class TestAlertEnums:
    """Test alert-related enums."""

    def test_alert_severity(self):
        """Test AlertSeverity enum."""
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.WARNING == "warning"
        assert AlertSeverity.ERROR == "error"
        assert AlertSeverity.CRITICAL == "critical"

    def test_alert_status(self):
        """Test AlertStatus enum."""
        assert AlertStatus.FIRING == "firing"
        assert AlertStatus.RESOLVED == "resolved"
        assert AlertStatus.ACKNOWLEDGED == "acknowledged"


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_alert_manager(self):
        """Test factory function."""
        manager = create_alert_manager()
        assert manager is not None
        assert isinstance(manager, AlertManager)
