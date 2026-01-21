"""
Multi-agent team review functionality using Victor AI's team coordination.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console

from victor.teams import TeamFormation, create_coordinator


console = Console()


class TeamReviewer:
    """Multi-agent team reviewer using Victor AI teams."""

    def __init__(self, orchestrator, config, formation: str = "parallel"):
        """Initialize team reviewer.

        Args:
            orchestrator: Victor AI agent orchestrator
            config: Review configuration
            formation: Team formation type
        """
        self.orchestrator = orchestrator
        self.config = config
        self.formation = TeamFormation(formation)

    async def review(self, target_path: Path, roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform team-based code review.

        Args:
            target_path: Path to code to review
            roles: Optional list of specific roles to include

        Returns:
            Team review results
        """
        # Define default team roles
        default_roles = self._get_default_roles()

        # Filter roles if specified
        if roles:
            team_roles = [r for r in default_roles if r["name"] in roles]
        else:
            team_roles = default_roles

        # Create team coordinator
        console.print(f"[cyan]Creating {self.formation.value} team with {len(team_roles)} agents...[/cyan]\n")

        coordinator = create_coordinator(
            formation=self.formation,
            roles=team_roles
        )

        # Prepare context for team
        context = {
            "target_path": str(target_path),
            "review_config": {
                "max_complexity": self.config.max_complexity,
                "severity_levels": self.config.severity_levels,
            }
        }

        # Execute team review
        console.print("[cyan]Agents analyzing code...[/cyan]\n")

        team_result = await coordinator.execute_task(
            task="Review the code for issues and improvements",
            context=context
        )

        # Aggregate results
        return self._aggregate_team_results(team_result, team_roles)

    def _get_default_roles(self) -> List[Dict[str, Any]]:
        """Get default team role definitions."""
        return [
            {
                "name": "security_reviewer",
                "display_name": "Security Reviewer",
                "description": "Focuses on security vulnerabilities and threats",
                "persona": "You are a security expert specializing in identifying vulnerabilities, "
                          "security flaws, and potential attack vectors in code. You check for SQL injection, "
                          "XSS, hardcoded secrets, insecure dependencies, and authentication issues.",
                "tool_categories": ["security", "analysis"],
                "capabilities": ["security_scan", "vulnerability_detection"],
            },
            {
                "name": "quality_reviewer",
                "display_name": "Quality Reviewer",
                "description": "Focuses on code quality and maintainability",
                "persona": "You are a code quality expert focusing on maintainability, readability, "
                          "and best practices. You check code complexity, duplication, naming conventions, "
                          "documentation, and adherence to SOLID principles.",
                "tool_categories": ["analysis", "metrics"],
                "capabilities": ["complexity_analysis", "style_check", "duplication_detection"],
            },
            {
                "name": "performance_reviewer",
                "display_name": "Performance Reviewer",
                "description": "Focuses on performance optimization",
                "persona": "You are a performance optimization expert. You identify performance bottlenecks, "
                          "inefficient algorithms, memory leaks, database query issues, and opportunities for "
                          "caching and optimization.",
                "tool_categories": ["analysis", "profiling"],
                "capabilities": ["profiling", "performance_analysis", "benchmarking"],
            },
        ]

    def _aggregate_team_results(self, team_result: Any, team_roles: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all team members."""
        # Extract individual agent results
        agent_results = {}
        all_findings = []

        for role in team_roles:
            role_name = role["name"]
            # In real implementation, extract results from team_result
            # For demo, we'll use placeholder data
            agent_results[role_name] = {
                "issue_count": len([f for f in all_findings if f.get("agent") == role_name]),
                "confidence": 0.85,
                "findings": [],
            }

        # Aggregate findings by severity
        aggregated = sorted(all_findings, key=lambda f: f.get("severity", "low"))

        return {
            "agents": agent_results,
            "aggregated_findings": aggregated,
            "total_findings": len(aggregated),
        }
