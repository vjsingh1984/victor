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

"""Tests for IaC Scanner Tool."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from victor.tools.iac_scanner_tool import IaCScannerTool
from victor.tools.base import CostTier
from victor.iac import (
    Category,
    IaCConfig,
    IaCFinding,
    IaCPlatform,
    IaCResource,
    ScanResult,
    Severity,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tool():
    """Create IaC scanner tool instance."""
    return IaCScannerTool()


@pytest.fixture
def sample_finding():
    """Create sample IaC finding."""
    return IaCFinding(
        rule_id="TF001",
        severity=Severity.HIGH,
        category=Category.SECRETS,
        message="Hardcoded AWS access key detected",
        description="AWS access keys should not be hardcoded",
        file_path=Path("main.tf"),
        line_number=15,
        resource_type="aws_access_key",
        remediation="Use environment variables or AWS Secrets Manager",
    )


@pytest.fixture
def sample_config():
    """Create sample IaC config."""
    return IaCConfig(
        platform=IaCPlatform.TERRAFORM,
        file_path=Path("main.tf"),
        resources=[
            IaCResource(
                resource_type="aws_s3_bucket",
                name="my-bucket",
                file_path=Path("main.tf"),
                line_number=1,
            )
        ],
    )


@pytest.fixture
def sample_scan_result(sample_config, sample_finding):
    """Create sample scan result."""
    return ScanResult(
        configs=[sample_config],
        findings=[sample_finding],
        files_scanned=5,
        total_resources=10,
        scan_duration_ms=150,
    )


@pytest.fixture
def sample_summary():
    """Create sample security summary."""
    return {
        "risk_score": 45,
        "files_scanned": 10,
        "total_resources": 25,
        "scan_duration_ms": 200,
        "by_severity": {
            "critical": 1,
            "high": 3,
            "medium": 5,
            "low": 2,
            "info": 0,
        },
        "by_category": {
            "secrets": 2,
            "permissions": 4,
            "encryption": 3,
            "network": 2,
        },
        "top_rules": [
            ("TF001", 5),
            ("K8S003", 3),
            ("DOCKER002", 2),
        ],
    }


# =============================================================================
# TOOL PROPERTIES TESTS
# =============================================================================


class TestIaCScannerToolProperties:
    """Tests for tool properties and metadata."""

    def test_tool_name(self, tool):
        """Test tool name uses canonical name from ToolNames."""
        from victor.tools.tool_names import ToolNames

        assert tool.name == ToolNames.IAC
        assert tool.name == "iac"  # Canonical short name

    def test_tool_description_contains_platforms(self, tool):
        """Test description mentions supported platforms."""
        assert "Terraform" in tool.description
        assert "Docker" in tool.description
        assert "K8s" in tool.description

    def test_tool_description_contains_detections(self, tool):
        """Test description mentions detection capabilities."""
        assert "secrets" in tool.description.lower()
        assert "IAM" in tool.description
        assert "encryption" in tool.description.lower()

    def test_cost_tier(self, tool):
        """Test cost tier is LOW."""
        assert tool.cost_tier == CostTier.LOW

    def test_parameters_schema(self, tool):
        """Test parameters schema structure."""
        assert tool.parameters["type"] == "object"
        assert "action" in tool.parameters["properties"]
        assert "file_path" in tool.parameters["properties"]
        assert "platform" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["action"]

    def test_action_enum_values(self, tool):
        """Test action enum has all expected values."""
        actions = tool.parameters["properties"]["action"]["enum"]
        assert "scan" in actions
        assert "scan_file" in actions
        assert "summary" in actions
        assert "detect" in actions

    def test_platform_enum_values(self, tool):
        """Test platform enum has expected values."""
        platforms = tool.parameters["properties"]["platform"]["enum"]
        assert "terraform" in platforms
        assert "docker" in platforms
        assert "kubernetes" in platforms
        assert "all" in platforms

    def test_min_severity_enum_values(self, tool):
        """Test min_severity enum has expected values."""
        severities = tool.parameters["properties"]["min_severity"]["enum"]
        assert "critical" in severities
        assert "high" in severities
        assert "medium" in severities
        assert "low" in severities

    def test_metadata_category(self, tool):
        """Test metadata category."""
        assert tool.metadata.category == "iac"

    def test_metadata_keywords(self, tool):
        """Test metadata keywords."""
        keywords = tool.metadata.keywords
        assert "iac" in keywords
        assert "terraform" in keywords
        assert "kubernetes" in keywords


# =============================================================================
# DETECT ACTION TESTS
# =============================================================================


class TestDetectAction:
    """Tests for the detect action."""

    @pytest.mark.asyncio
    async def test_detect_no_platforms(self, tool):
        """Test detection when no IaC platforms found."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_platforms.return_value = []
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is True
            assert "No IaC files detected" in result.output
            assert result.metadata["platforms"] == []

    @pytest.mark.asyncio
    async def test_detect_with_platforms(self, tool):
        """Test detection with platforms found."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_platforms.return_value = [
                IaCPlatform.TERRAFORM,
                IaCPlatform.DOCKER,
            ]
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is True
            assert "Detected IaC Platforms" in result.output
            assert "terraform" in result.metadata["platforms"]
            assert "docker" in result.metadata["platforms"]


# =============================================================================
# SUMMARY ACTION TESTS
# =============================================================================


class TestSummaryAction:
    """Tests for the summary action."""

    @pytest.mark.asyncio
    async def test_summary_basic(self, tool, sample_summary):
        """Test basic security summary."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_summary.return_value = sample_summary
            MockManager.return_value = mock_manager

            result = await tool.execute(action="summary")

            assert result.success is True
            assert "IaC Security Summary" in result.output
            assert "45" in result.output  # risk score

    @pytest.mark.asyncio
    async def test_summary_low_risk(self, tool):
        """Test summary with low risk score."""
        summary = {
            "risk_score": 10,
            "files_scanned": 5,
            "total_resources": 10,
            "scan_duration_ms": 100,
            "by_severity": {"critical": 0, "high": 0, "medium": 1, "low": 2, "info": 0},
            "by_category": {},
            "top_rules": [],
        }

        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_summary.return_value = summary
            MockManager.return_value = mock_manager

            result = await tool.execute(action="summary")

            assert result.success is True
            assert "🟢" in result.output  # low risk icon
            assert "Low Risk" in result.output

    @pytest.mark.asyncio
    async def test_summary_critical_risk(self, tool):
        """Test summary with critical risk score."""
        summary = {
            "risk_score": 90,
            "files_scanned": 5,
            "total_resources": 10,
            "scan_duration_ms": 100,
            "by_severity": {"critical": 5, "high": 10, "medium": 5, "low": 0, "info": 0},
            "by_category": {},
            "top_rules": [],
        }

        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_summary.return_value = summary
            MockManager.return_value = mock_manager

            result = await tool.execute(action="summary")

            assert result.success is True
            # In CI, emojis are disabled, so check for text version [!]
            assert ("🔴" in result.output or "[!]" in result.output)  # critical risk icon
            assert "Critical Risk" in result.output


# =============================================================================
# SCAN ACTION TESTS
# =============================================================================


class TestScanAction:
    """Tests for the scan action."""

    @pytest.mark.asyncio
    async def test_scan_all_platforms(self, tool, sample_scan_result):
        """Test scanning all platforms."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.scan.return_value = sample_scan_result
            MockManager.return_value = mock_manager

            result = await tool.execute(action="scan")

            assert result.success is True
            assert "IaC Security Scan Results" in result.output
            mock_manager.scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_specific_platform(self, tool, sample_scan_result):
        """Test scanning specific platform."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.scan.return_value = sample_scan_result
            MockManager.return_value = mock_manager

            result = await tool.execute(action="scan", platform="terraform")

            assert result.success is True
            call_args = mock_manager.scan.call_args
            assert call_args[1]["platforms"] == [IaCPlatform.TERRAFORM]

    @pytest.mark.asyncio
    async def test_scan_with_min_severity(self, tool, sample_scan_result):
        """Test scanning with minimum severity filter."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.scan.return_value = sample_scan_result
            MockManager.return_value = mock_manager

            result = await tool.execute(action="scan", min_severity="high")

            assert result.success is True

    @pytest.mark.asyncio
    async def test_scan_with_critical_findings(self, tool):
        """Test scan output includes critical findings."""
        finding = IaCFinding(
            rule_id="CRIT001",
            severity=Severity.CRITICAL,
            category=Category.SECRETS,
            message="Exposed credentials",
            description="Critical security issue",
            file_path=Path("main.tf"),
            line_number=10,
            remediation="Remove credentials immediately",
        )
        scan_result = ScanResult(
            configs=[],
            findings=[finding],
            files_scanned=1,
            total_resources=1,
            scan_duration_ms=50,
        )

        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.scan.return_value = scan_result
            MockManager.return_value = mock_manager

            result = await tool.execute(action="scan")

            # In CI, emojis are disabled, so check for text versions
            assert ("🔴" in result.output or "[!]" in result.output)  # critical severity
            assert "CRIT001" in result.output
            assert ("💡" in result.output or "*" in result.output)  # remediation hint


# =============================================================================
# SCAN FILE ACTION TESTS
# =============================================================================


class TestScanFileAction:
    """Tests for the scan_file action."""

    @pytest.mark.asyncio
    async def test_scan_file_success(self, tool, sample_finding):
        """Test successful file scan."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.scan_file.return_value = [sample_finding]
            MockManager.return_value = mock_manager

            result = await tool.execute(action="scan_file", file_path="main.tf")

            assert result.success is True
            assert "Security Findings" in result.output
            assert result.metadata["findings"]

    @pytest.mark.asyncio
    async def test_scan_file_no_findings(self, tool):
        """Test file scan with no findings."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.scan_file.return_value = []
            MockManager.return_value = mock_manager

            result = await tool.execute(action="scan_file", file_path="clean.tf")

            assert result.success is True
            assert "No security issues found" in result.output

    @pytest.mark.asyncio
    async def test_scan_file_missing_path(self, tool):
        """Test scan_file without file path."""
        result = await tool.execute(action="scan_file")

        assert result.success is False
        assert "file_path is required" in result.output


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        """Test handling unknown action."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            MockManager.return_value = AsyncMock()

            result = await tool.execute(action="invalid_action")

            assert result.success is False
            assert "Unknown action" in result.output

    @pytest.mark.asyncio
    async def test_exception_handling(self, tool):
        """Test exception handling in execute."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_platforms.side_effect = Exception("Scanner error")
            MockManager.return_value = mock_manager

            result = await tool.execute(action="detect")

            assert result.success is False
            assert "IaC scanning failed" in result.output
            assert "Scanner error" in result.error


# =============================================================================
# FORMATTING TESTS
# =============================================================================


class TestFormatMethods:
    """Tests for formatting helper methods."""

    def test_format_platforms_empty(self, tool):
        """Test formatting empty platforms list."""
        output = tool._format_platforms([])
        assert "No IaC files detected" in output

    def test_format_platforms_with_data(self, tool):
        """Test formatting platforms with data."""
        platforms = [IaCPlatform.TERRAFORM, IaCPlatform.DOCKER, IaCPlatform.KUBERNETES]
        output = tool._format_platforms(platforms)

        assert "Detected IaC Platforms" in output
        assert "🏗️" in output  # terraform icon
        assert "🐳" in output  # docker icon
        assert "☸️" in output  # kubernetes icon

    def test_format_summary_medium_risk(self, tool, sample_summary):
        """Test formatting summary with medium risk."""
        output = tool._format_summary(sample_summary)

        assert "IaC Security Summary" in output
        # Accept both emoji (🟡) and text ([M]) versions
        assert ("🟡" in output or "[M]" in output)  # medium risk icon
        assert "Medium Risk" in output
        assert "45/100" in output

    def test_format_summary_high_risk(self, tool):
        """Test formatting summary with high risk."""
        summary = {
            "risk_score": 65,
            "files_scanned": 10,
            "total_resources": 25,
            "scan_duration_ms": 200,
            "by_severity": {"critical": 0, "high": 5, "medium": 3, "low": 2, "info": 0},
            "by_category": {},
            "top_rules": [],
        }
        output = tool._format_summary(summary)

        # Accept both emoji (🟠) and text ([H]) versions
        assert ("🟠" in output or "[H]" in output)  # high risk icon
        assert "High Risk" in output

    def test_format_summary_includes_categories(self, tool, sample_summary):
        """Test formatting summary includes categories."""
        output = tool._format_summary(sample_summary)

        assert "By Category" in output
        assert "secrets" in output

    def test_format_summary_includes_top_rules(self, tool, sample_summary):
        """Test formatting summary includes top rules."""
        output = tool._format_summary(sample_summary)

        assert "Most Common Issues" in output
        assert "TF001" in output

    def test_format_scan_result_basic(self, tool, sample_scan_result):
        """Test formatting scan result."""
        output = tool._format_scan_result(sample_scan_result)

        assert "IaC Security Scan Results" in output
        assert "**Files:** 5" in output
        assert "**Resources:** 10" in output

    def test_format_scan_result_with_severities(self, tool):
        """Test formatting scan result with different severities."""
        findings = [
            IaCFinding(
                rule_id=f"RULE{i}",
                severity=severity,
                category=Category.SECRETS,
                message=f"Finding {i}",
                description="Description",
                file_path=Path("main.tf"),
                line_number=i,
            )
            for i, severity in enumerate(
                [
                    Severity.CRITICAL,
                    Severity.HIGH,
                    Severity.MEDIUM,
                    Severity.LOW,
                ]
            )
        ]
        scan_result = ScanResult(
            configs=[],
            findings=findings,
            files_scanned=1,
            total_resources=1,
            scan_duration_ms=100,
        )
        output = tool._format_scan_result(scan_result)

        # In CI, emojis are disabled, so check for text versions
        assert ("🔴" in output or "[!]" in output)  # critical
        assert ("🟠" in output or "[H]" in output)  # high
        assert ("🟡" in output or "[M]" in output)  # medium

    def test_format_findings_empty(self, tool):
        """Test formatting empty findings list."""
        output = tool._format_findings([], Path("test.tf"))
        assert "No security issues found" in output

    def test_format_findings_with_data(self, tool, sample_finding):
        """Test formatting findings with data."""
        output = tool._format_findings([sample_finding], Path("main.tf"))

        assert "Security Findings" in output
        assert "TF001" in output
        # Accept both emoji (🟠) and text ([H]) versions
        assert ("🟠" in output or "[H]" in output)  # high severity
        assert "Line: 15" in output
        # Accept both emoji (💡) and text ([R]) versions for remediation
        assert ("💡" in output or "[R]" in output or "Remediation" in output)

    def test_format_findings_different_severities(self, tool):
        """Test formatting findings with different severities."""
        findings = [
            IaCFinding(
                rule_id="RULE1",
                severity=Severity.CRITICAL,
                category=Category.SECRETS,
                message="Critical finding",
                description="",
                file_path=Path("test.tf"),
            ),
            IaCFinding(
                rule_id="RULE2",
                severity=Severity.INFO,
                category=Category.BEST_PRACTICE,
                message="Info finding",
                description="",
                file_path=Path("test.tf"),
            ),
        ]
        output = tool._format_findings(findings, Path("test.tf"))

        # In CI, emojis are disabled, so check for text versions
        assert ("🔴" in output or "[!]" in output)  # critical
        assert ("⚪" in output or "[?]" in output)  # info


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestIntegrationStyle:
    """Integration-style tests combining multiple aspects."""

    @pytest.mark.asyncio
    async def test_full_scan_workflow(self, tool, sample_summary, sample_scan_result):
        """Test a full scanning workflow."""
        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.detect_platforms.return_value = [
                IaCPlatform.TERRAFORM,
                IaCPlatform.DOCKER,
            ]
            mock_manager.get_summary.return_value = sample_summary
            mock_manager.scan.return_value = sample_scan_result
            MockManager.return_value = mock_manager

            # Step 1: Detect
            detect_result = await tool.execute(action="detect")
            assert detect_result.success is True
            assert len(detect_result.metadata["platforms"]) == 2

            # Step 2: Summary
            summary_result = await tool.execute(action="summary")
            assert summary_result.success is True

            # Step 3: Full scan
            scan_result = await tool.execute(action="scan")
            assert scan_result.success is True
            assert scan_result.metadata is not None

    @pytest.mark.asyncio
    async def test_kubernetes_specific_scan(self, tool):
        """Test Kubernetes-specific scan workflow."""
        k8s_finding = IaCFinding(
            rule_id="K8S001",
            severity=Severity.HIGH,
            category=Category.PERMISSIONS,
            message="Container running as root",
            description="Security best practice violation",
            file_path=Path("deployment.yaml"),
            line_number=20,
            remediation="Set runAsNonRoot: true",
        )
        scan_result = ScanResult(
            configs=[
                IaCConfig(
                    platform=IaCPlatform.KUBERNETES,
                    file_path=Path("deployment.yaml"),
                )
            ],
            findings=[k8s_finding],
            files_scanned=1,
            total_resources=5,
            scan_duration_ms=80,
        )

        with patch("victor.tools.iac_scanner_tool.IaCManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.scan.return_value = scan_result
            MockManager.return_value = mock_manager

            result = await tool.execute(action="scan", platform="kubernetes")

            assert result.success is True
            call_args = mock_manager.scan.call_args
            assert call_args[1]["platforms"] == [IaCPlatform.KUBERNETES]
