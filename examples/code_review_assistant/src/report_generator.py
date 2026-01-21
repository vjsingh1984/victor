"""
Report generation for code review results.
"""

import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from jinja2 import Template


class ReportGenerator:
    """Generate review reports in various formats."""

    def __init__(self, format: str):
        """Initialize report generator.

        Args:
            format: Report format (html, json, markdown)
        """
        self.format = format

    def generate(self, results: Dict[str, Any], output_path: str):
        """Generate report.

        Args:
            results: Review results
            output_path: Path to output file
        """
        if self.format == "html":
            self._generate_html(results, output_path)
        elif self.format == "json":
            self._generate_json(results, output_path)
        elif self.format == "markdown":
            self._generate_markdown(results, output_path)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _generate_html(self, results: Dict[str, Any], output_path: str):
        """Generate HTML report."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Victor AI Code Review Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; margin-bottom: 10px; }
        .date { color: #666; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h3 { color: #666; font-size: 14px; margin-bottom: 10px; }
        .card .value { font-size: 32px; font-weight: bold; }
        .card.critical .value { color: #d32f2f; }
        .card.high .value { color: #f57c00; }
        .card.medium .value { color: #fbc02d; }
        .card.low .value { color: #1976d2; }
        .issues { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .issues h2 { margin-bottom: 20px; }
        .issue { padding: 15px; margin-bottom: 10px; border-left: 4px solid #ddd; background: #f9f9f9; }
        .issue.critical { border-left-color: #d32f2f; }
        .issue.high { border-left-color: #f57c00; }
        .issue.medium { border-left-color: #fbc02d; }
        .issue.low { border-left-color: #1976d2; }
        .issue-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
        .severity { font-weight: bold; text-transform: uppercase; font-size: 12px; padding: 2px 8px; border-radius: 4px; }
        .severity.critical { background: #ffebee; color: #d32f2f; }
        .severity.high { background: #fff3e0; color: #f57c00; }
        .severity.medium { background: #fffde7; color: #f9a825; }
        .severity.low { background: #e3f2fd; color: #1976d2; }
        .message { font-weight: 500; margin-bottom: 5px; }
        .location { color: #666; font-size: 14px; }
        .suggestion { color: #666; font-size: 14px; margin-top: 10px; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Victor AI Code Review Report</h1>
        <p class="date">Generated: {{ timestamp }}</p>

        <div class="summary">
            <div class="card">
                <h3>Files Analyzed</h3>
                <div class="value">{{ results.files_analyzed }}</div>
            </div>
            <div class="card critical">
                <h3>Critical</h3>
                <div class="value">{{ results.critical }}</div>
            </div>
            <div class="card high">
                <h3>High</h3>
                <div class="value">{{ results.high }}</div>
            </div>
            <div class="card medium">
                <h3>Medium</h3>
                <div class="value">{{ results.medium }}</div>
            </div>
            <div class="card low">
                <h3>Low</h3>
                <div class="value">{{ results.low }}</div>
            </div>
        </div>

        <div class="issues">
            <h2>Issues Found ({{ results.total_issues }})</h2>
            {% for issue in results.issues %}
            <div class="issue {{ issue.severity }}">
                <div class="issue-header">
                    <span class="severity {{ issue.severity }}">{{ issue.severity }}</span>
                    <span class="location">{{ issue.file }}:{{ issue.line }}</span>
                </div>
                <div class="message">{{ issue.message }}</div>
                {% if issue.suggestion %}
                <div class="suggestion">💡 {{ issue.suggestion }}</div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
        """

        template = Template(html_template)
        html = template.render(
            results=results,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        with open(output_path, "w") as f:
            f.write(html)

    def _generate_json(self, results: Dict[str, Any], output_path: str):
        """Generate JSON report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "files_analyzed": results["files_analyzed"],
                "total_issues": results["total_issues"],
                "critical": results["critical"],
                "high": results["high"],
                "medium": results["medium"],
                "low": results["low"],
            },
            "issues_by_type": results["issues_by_type"],
            "issues": results["issues"],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    def _generate_markdown(self, results: Dict[str, Any], output_path: str):
        """Generate Markdown report."""
        md = f"""# Victor AI Code Review Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **Files Analyzed:** {results['files_analyzed']}
- **Total Issues:** {results['total_issues']}

### Issues by Severity

- 🔴 **Critical:** {results['critical']}
- 🟠 **High:** {results['high']}
- 🟡 **Medium:** {results['medium']}
- 🔵 **Low:** {results['low']}

### Issues by Type

"""

        for issue_type, count in results["issues_by_type"].items():
            md += f"- **{issue_type}:** {count}\n"

        md += "\n## Issues\n\n"

        for issue in results["issues"]:
            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
            }.get(issue["severity"], "⚪")

            md += f"### {severity_emoji} {issue['severity'].upper()}: {issue['message']}\n\n"
            md += f"**Location:** `{issue['file']}:{issue['line']}`\n\n"
            md += f"**Category:** {issue['category']}\n\n"

            if issue.get("suggestion"):
                md += f"**Suggestion:** {issue['suggestion']}\n\n"

            md += "---\n\n"

        with open(output_path, "w") as f:
            f.write(md)
