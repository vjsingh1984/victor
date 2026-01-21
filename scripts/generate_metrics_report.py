#!/usr/bin/env python
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

"""Generate production metrics report.

This script collects metrics from Victor AI and generates comprehensive reports
in various formats (JSON, CSV, Markdown).

Usage:
    python scripts/generate_metrics_report.py
    python scripts/generate_metrics_report.py --format json
    python scripts/generate_metrics_report.py --format csv --output metrics.csv
    python scripts/generate_metrics_report.py --format markdown --output report.md
    python scripts/generate_metrics_report.py --prometheus http://localhost:9090
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MetricData:
    """Container for metric data."""

    name: str
    value: float
    labels: Dict[str, str]
    timestamp: str
    description: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

    request_duration_p50: float
    request_duration_p95: float
    request_duration_p99: float
    tool_execution_p95: float
    provider_latency_p95: float
    memory_usage_bytes: float
    cpu_usage_percent: float
    request_rate: float


@dataclass
class FunctionalMetrics:
    """Functional metrics container."""

    tool_executions_total: int
    tool_success_rate: float
    provider_requests_total: int
    provider_success_rate: float
    vertical_usage_counts: Dict[str, int]
    workflow_executions_total: int


@dataclass
class BusinessMetrics:
    """Business metrics container."""

    total_requests: int
    active_users: int
    average_session_duration: float
    requests_per_user: float


@dataclass
class VerticalMetrics:
    """Vertical-specific metrics container."""

    coding: Dict[str, Any]
    rag: Dict[str, Any]
    devops: Dict[str, Any]
    dataanalysis: Dict[str, Any]
    research: Dict[str, Any]


@dataclass
class SecurityMetrics:
    """Security metrics container."""

    authorization_success_rate: float
    failed_authorizations: int
    security_test_pass_rate: float
    vulnerabilities_found: int


@dataclass
class MetricsReport:
    """Complete metrics report."""

    generated_at: str
    time_range: str
    performance: PerformanceMetrics
    functional: FunctionalMetrics
    business: BusinessMetrics
    verticals: VerticalMetrics
    security: SecurityMetrics


class MetricsReportGenerator:
    """Generate production metrics reports."""

    def __init__(
        self,
        prometheus_url: Optional[str] = None,
        metrics_file: Optional[str] = None,
    ):
        """Initialize report generator.

        Args:
            prometheus_url: Prometheus server URL (e.g., http://localhost:9090)
            metrics_file: Path to metrics file (if not using Prometheus)
        """
        self.prometheus_url = prometheus_url
        self.metrics_file = metrics_file

    def collect_from_prometheus(self, query: str) -> List[Dict[str, Any]]:
        """Collect metrics from Prometheus.

        Args:
            query: PromQL query

        Returns:
            List of metric results
        """
        try:
            import requests

            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data["status"] == "success":
                return data["data"]["result"]
            return []
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return []

    def collect_from_file(self) -> List[MetricData]:
        """Collect metrics from file.

        Returns:
            List of metric data
        """
        if not self.metrics_file or not os.path.exists(self.metrics_file):
            return []

        metrics = []
        with open(self.metrics_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Parse Prometheus text format
                    try:
                        if "{" in line:
                            # Metric with labels
                            name_part, value_part = line.split("{", 1)
                            labels_part, value_part = value_part.split("}", 1)
                            name = name_part.strip()
                            labels = self._parse_labels(labels_part)
                            value = float(value_part.strip())
                        else:
                            # Metric without labels
                            name, value = line.split()
                            labels = {}

                        metrics.append(
                            MetricData(
                                name=name,
                                value=value,
                                labels=labels,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                            )
                        )
                    except Exception:
                        continue

        return metrics

    def _parse_labels(self, labels_str: str) -> Dict[str, str]:
        """Parse Prometheus labels string.

        Args:
            labels_str: Labels string (e.g., 'key1="value1",key2="value2"')

        Returns:
            Dictionary of labels
        """
        labels = {}
        for pair in labels_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                labels[key.strip()] = value.strip('"')
        return labels

    def generate_performance_metrics(self) -> PerformanceMetrics:
        """Generate performance metrics report.

        Returns:
            PerformanceMetrics object
        """
        if self.prometheus_url:
            # Query Prometheus
            p50 = self._query_prometheus_value(
                'job:victor_request_duration:p50:5m'
            )
            p95 = self._query_prometheus_value(
                'job:victor_request_duration:p95:5m'
            )
            p99 = self._query_prometheus_value(
                'job:victor_request_duration:p99:5m'
            )
            tool_p95 = self._query_prometheus_value(
                'victor:tool_execution_duration:p95:5m'
            )
            provider_p95 = self._query_prometheus_value(
                'victor:provider_latency:p95:5m'
            )
            memory = self._query_prometheus_value(
                'victor_memory_usage_bytes'
            )
            cpu = self._query_prometheus_value(
                'victor:cpu_usage:avg:5m'
            )
            rate = self._query_prometheus_value(
                'job:victor_request_rate:5m'
            )
        else:
            # Use defaults
            p50 = p95 = p99 = tool_p95 = provider_p95 = memory = cpu = rate = 0.0

        return PerformanceMetrics(
            request_duration_p50=p50,
            request_duration_p95=p95,
            request_duration_p99=p99,
            tool_execution_p95=tool_p95,
            provider_latency_p95=provider_p95,
            memory_usage_bytes=memory,
            cpu_usage_percent=cpu,
            request_rate=rate,
        )

    def _query_prometheus_value(self, query: str) -> float:
        """Query Prometheus and return value.

        Args:
            query: PromQL query

        Returns:
            Metric value (or 0.0 if not found)
        """
        results = self.collect_from_prometheus(query)
        if results and len(results) > 0:
            try:
                return float(results[0].get("value", [0, "0"])[1])
            except (ValueError, IndexError):
                pass
        return 0.0

    def generate_functional_metrics(self) -> FunctionalMetrics:
        """Generate functional metrics report.

        Returns:
            FunctionalMetrics object
        """
        if self.prometheus_url:
            tool_total = self._query_prometheus_value(
                'sum(victor_tool_executions_total)'
            )
            tool_rate = self._query_prometheus_value(
                'sum(rate(victor_tool_executions_total[5m]))'
            )
            provider_total = self._query_prometheus_value(
                'sum(victor_provider_requests_total)'
            )
            workflow_total = self._query_prometheus_value(
                'sum(victor_workflow_executions_total)'
            )
        else:
            tool_total = provider_total = workflow_total = 0
            tool_rate = 0.0

        # Get vertical usage
        vertical_usage = {}
        if self.prometheus_url:
            results = self.collect_from_prometheus(
                'sum(victor_vertical_usage_total) by (vertical)'
            )
            for result in results:
                vertical = result.get("metric", {}).get("vertical", "unknown")
                value = float(result.get("value", [0, "0"])[1])
                vertical_usage[vertical] = int(value)

        return FunctionalMetrics(
            tool_executions_total=int(tool_total),
            tool_success_rate=tool_rate,
            provider_requests_total=int(provider_total),
            provider_success_rate=tool_rate,
            vertical_usage_counts=vertical_usage,
            workflow_executions_total=int(workflow_total),
        )

    def generate_business_metrics(self) -> BusinessMetrics:
        """Generate business metrics report.

        Returns:
            BusinessMetrics object
        """
        if self.prometheus_url:
            total_requests = self._query_prometheus_value(
                'victor_total_requests'
            )
            active_users = self._query_prometheus_value(
                'sum(victor_active_users)'
            )
            avg_session = self._query_prometheus_value(
                'victor:session_duration:avg:5m'
            )
            requests_per_user = self._query_prometheus_value(
                'victor:requests_per_user:5m'
            )
        else:
            total_requests = active_users = 0
            avg_session = requests_per_user = 0.0

        return BusinessMetrics(
            total_requests=int(total_requests),
            active_users=int(active_users),
            average_session_duration=avg_session,
            requests_per_user=requests_per_user,
        )

    def generate_vertical_metrics(self) -> VerticalMetrics:
        """Generate vertical-specific metrics report.

        Returns:
            VerticalMetrics object
        """
        if self.prometheus_url:
            # Coding
            coding_files = self._query_prometheus_value(
                'victor_coding_files_analyzed_total'
            )
            coding_loc = self._query_prometheus_value(
                'victor_coding_loc_reviewed_total'
            )
            coding_issues = self._query_prometheus_value(
                'victor_coding_issues_found_total'
            )

            # RAG
            rag_docs = self._query_prometheus_value(
                'victor_rag_documents_ingested_total'
            )
            rag_accuracy = self._query_prometheus_value(
                'victor:rag:search_accuracy:5m'
            )

            # DevOps
            devops_deployments = self._query_prometheus_value(
                'victor_devops_deployments_total'
            )
            devops_containers = self._query_prometheus_value(
                'sum(victor_devops_containers_managed)'
            )

            # DataAnalysis
            da_queries = self._query_prometheus_value(
                'victor_dataanalysis_queries_total'
            )
            da_viz = self._query_prometheus_value(
                'victor_dataanalysis_visualizations_total'
            )

            # Research
            research_searches = self._query_prometheus_value(
                'victor_research_searches_total'
            )
            research_citations = self._query_prometheus_value(
                'victor_research_citations_generated_total'
            )
        else:
            coding_files = coding_loc = coding_issues = 0
            rag_docs = 0.0
            rag_accuracy = 0.0
            devops_deployments = 0
            devops_containers = 0
            da_queries = 0
            da_viz = 0
            research_searches = 0
            research_citations = 0

        return VerticalMetrics(
            coding={
                "files_analyzed": int(coding_files),
                "loc_reviewed": int(coding_loc),
                "issues_found": int(coding_issues),
            },
            rag={
                "documents_ingested": int(rag_docs),
                "search_accuracy": rag_accuracy,
            },
            devops={
                "deployments": int(devops_deployments),
                "containers_managed": int(devops_containers),
            },
            dataanalysis={
                "queries": int(da_queries),
                "visualizations": int(da_viz),
            },
            research={
                "searches": int(research_searches),
                "citations": int(research_citations),
            },
        )

    def generate_security_metrics(self) -> SecurityMetrics:
        """Generate security metrics report.

        Returns:
            SecurityMetrics object
        """
        if self.prometheus_url:
            auth_rate = self._query_prometheus_value(
                'victor:authorization_success_rate:5m'
            )
            failed_auth = self._query_prometheus_value(
                'victor:failed_authorization_rate:5m'
            )
            test_pass = self._query_prometheus_value(
                'victor_security_test_pass_rate'
            )
            vulns = self._query_prometheus_value(
                'sum(victor_security_vulnerabilities_found_total)'
            )
        else:
            auth_rate = 1.0
            failed_auth = 0
            test_pass = 1.0
            vulns = 0

        return SecurityMetrics(
            authorization_success_rate=auth_rate,
            failed_authorizations=int(failed_auth),
            security_test_pass_rate=test_pass,
            vulnerabilities_found=int(vulns),
        )

    def generate_report(
        self,
        time_range: str = "5m",
    ) -> MetricsReport:
        """Generate complete metrics report.

        Args:
            time_range: Time range for metrics (e.g., 5m, 1h, 24h)

        Returns:
            MetricsReport object
        """
        return MetricsReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            time_range=time_range,
            performance=self.generate_performance_metrics(),
            functional=self.generate_functional_metrics(),
            business=self.generate_business_metrics(),
            verticals=self.generate_vertical_metrics(),
            security=self.generate_security_metrics(),
        )

    def save_json(
        self,
        report: MetricsReport,
        output_path: str,
    ) -> None:
        """Save report as JSON.

        Args:
            report: Metrics report
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"JSON report saved to {output_path}")

    def save_csv(
        self,
        report: MetricsReport,
        output_path: str,
    ) -> None:
        """Save report as CSV.

        Args:
            report: Metrics report
            output_path: Output file path
        """
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["Category", "Metric", "Value"])

            # Performance metrics
            for key, value in asdict(report.performance).items():
                writer.writerow(["Performance", key, value])

            # Functional metrics
            for key, value in asdict(report.functional).items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        writer.writerow(["Functional", f"{key}.{k}", v])
                else:
                    writer.writerow(["Functional", key, value])

            # Business metrics
            for key, value in asdict(report.business).items():
                writer.writerow(["Business", key, value])

            # Vertical metrics
            for vertical, metrics in asdict(report.verticals).items():
                for metric, value in metrics.items():
                    writer.writerow(["Verticals", f"{vertical}.{metric}", value])

            # Security metrics
            for key, value in asdict(report.security).items():
                writer.writerow(["Security", key, value])

        print(f"CSV report saved to {output_path}")

    def save_markdown(
        self,
        report: MetricsReport,
        output_path: str,
    ) -> None:
        """Save report as Markdown.

        Args:
            report: Metrics report
            output_path: Output file path
        """
        lines = [
            "# Victor AI Production Metrics Report",
            "",
            f"**Generated:** {report.generated_at}",
            f"**Time Range:** {report.time_range}",
            "",
            "## Performance Metrics",
            "",
        ]

        # Performance
        for key, value in asdict(report.performance).items():
            lines.append(f"- **{key}:** {value}")

        lines.extend([
            "",
            "## Functional Metrics",
            "",
        ])

        # Functional
        for key, value in asdict(report.functional).items():
            if isinstance(value, dict):
                lines.append(f"- **{key}:**")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"- **{key}:** {value}")

        lines.extend([
            "",
            "## Business Metrics",
            "",
        ])

        # Business
        for key, value in asdict(report.business).items():
            lines.append(f"- **{key}:** {value}")

        lines.extend([
            "",
            "## Vertical Metrics",
            "",
        ])

        # Verticals
        for vertical, metrics in asdict(report.verticals).items():
            lines.append(f"### {vertical.capitalize()}")
            for metric, value in metrics.items():
                lines.append(f"- **{metric}:** {value}")
            lines.append("")

        lines.extend([
            "## Security Metrics",
            "",
        ])

        # Security
        for key, value in asdict(report.security).items():
            lines.append(f"- **{key}:** {value}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Markdown report saved to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Victor AI production metrics report"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown", "all"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: metrics_report.<ext>)",
    )
    parser.add_argument(
        "--prometheus",
        help="Prometheus server URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--metrics-file",
        help="Path to metrics file (alternative to Prometheus)",
    )
    parser.add_argument(
        "--time-range",
        default="5m",
        help="Time range for metrics (default: 5m)",
    )

    args = parser.parse_args()

    # Initialize generator
    generator = MetricsReportGenerator(
        prometheus_url=args.prometheus or "http://localhost:9090",
        metrics_file=args.metrics_file,
    )

    # Generate report
    print("Generating metrics report...")
    report = generator.generate_report(time_range=args.time_range)

    # Save report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.format in ["json", "all"]:
        output = args.output or f"metrics_report_{timestamp}.json"
        generator.save_json(report, output)

    if args.format in ["csv", "all"]:
        output = args.output or f"metrics_report_{timestamp}.csv"
        generator.save_csv(report, output)

    if args.format in ["markdown", "all"]:
        output = args.output or f"metrics_report_{timestamp}.md"
        generator.save_markdown(report, output)

    print("\nReport generation complete!")


if __name__ == "__main__":
    main()
