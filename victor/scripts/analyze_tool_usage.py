#!/usr/bin/env python3
"""
Analyze actual tool usage from ConversationStore and JSONL logs.

This script pulls tool call frequency from actual usage data to drive
data-driven tool tier assignments (FULL/COMPACT/STUB).

Database Consolidation: Reads from both consolidated databases:
- project.db: Project-specific tool usage (primary source)
- victor.db: Global tool usage across all projects (fallback)
- Legacy conversation.db: Included for migration compatibility

Usage:
    python -m victor.scripts.analyze_tool_usage --days 30
    python -m victor.scripts.analyze_tool_usage --output tool_tiers.yaml
"""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


def get_conversation_db_paths() -> List[Path]:
    """Get paths to conversation databases in priority order.

    Returns:
        List of database paths (project-specific first, then global)
    """
    paths = []

    # 1. Project-specific database (current directory)
    project_db = Path.cwd() / ".victor" / "project.db"
    if project_db.exists():
        paths.append(project_db)

    # 2. Legacy project-specific conversation database
    legacy_db = Path.cwd() / ".victor" / "conversation.db"
    if legacy_db.exists():
        paths.append(legacy_db)

    # 3. Global database
    global_db = Path.home() / ".victor" / "victor.db"
    if global_db.exists():
        paths.append(global_db)

    # 4. Legacy global database
    legacy_global_db = Path.home() / ".victor" / "conversation.db"
    if legacy_global_db.exists():
        paths.append(legacy_global_db)

    return paths


def analyze_tool_usage_from_db(days: int = 30) -> List[Tuple[str, int, int]]:
    """
    Pull tool call frequency from consolidated victor.db databases.

    Args:
        days: Number of days to look back

    Returns:
        List of (tool_name, call_count, session_count) tuples
    """
    db_paths = get_conversation_db_paths()

    if not db_paths:
        print("Warning: No conversation databases found.")
        print("Searched in:")
        print(f"  - {Path.cwd() / '.victor' / 'project.db'} (project-specific)")
        print(f"  - {Path.cwd() / '.victor' / 'conversation.db'} (legacy project)")
        print(f"  - {Path.home() / '.victor' / 'victor.db'} (global)")
        print(f"  - {Path.home() / '.victor' / 'conversation.db'} (legacy global)")
        print("Returning empty results.")
        return []

    all_results = {}

    for db_path in db_paths:
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Check if messages table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='messages'
                """)

                if not cursor.fetchone():
                    print(f"Note: messages table not found in {db_path}")
                    continue

                # Get tool calls from last N days
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()

                # Query tool usage from messages table
                cursor.execute(
                    """
                    SELECT
                        tool_name,
                        COUNT(*) as calls,
                        COUNT(DISTINCT session_id) as sessions
                    FROM messages
                    WHERE tool_name IS NOT NULL
                      AND timestamp > ?
                    GROUP BY tool_name
                    ORDER BY calls DESC
                """,
                    (cutoff,),
                )

                results = cursor.fetchall()

                # Merge results from multiple databases
                for row in results:
                    tool_name = row["tool_name"]
                    calls = row["calls"]
                    sessions = row["sessions"]

                    if tool_name in all_results:
                        # Aggregate data from multiple sources
                        existing = all_results[tool_name]
                        all_results[tool_name] = (
                            tool_name,
                            existing[1] + calls,  # Sum calls
                            max(existing[2], sessions),  # Max sessions (approximate)
                        )
                    else:
                        all_results[tool_name] = (tool_name, calls, sessions)

                print(f"✓ Analyzed {len(results)} tools from {db_path}")

        except sqlite3.Error as e:
            print(f"Warning: Error querying {db_path}: {e}")
            continue

    # Convert to sorted list
    sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
    return sorted_results


def analyze_jsonl_logs(days: int = 30) -> List[Tuple[str, int, int]]:
    """
    Analyze tool usage from JSONL log files.

    Args:
        days: Number of days to look back

    Returns:
        List of (tool_name, call_count, session_count) tuples
    """
    # Look for JSONL logs in .victor/logs
    logs_dir = Path.home() / ".victor" / "logs"

    if not logs_dir.exists():
        return []

    tool_calls = Counter()
    sessions = set()

    # Find recent JSONL files
    cutoff_date = datetime.now() - timedelta(days=days)

    for jsonl_file in logs_dir.glob("*.jsonl"):
        # Check file modification time
        mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime)
        if mtime < cutoff_date:
            continue

        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)

                        # Look for tool call events (multiple formats)
                        # Format 1: event_type == "tool_call" with tool_name at top level
                        if entry.get("event_type") == "tool_call":
                            tool_name = entry.get("tool_name")
                            if tool_name:
                                tool_calls[tool_name] += 1
                                session_id = entry.get("session_id")
                                if session_id:
                                    sessions.add(session_id)

                        # Format 2: event_type == "tool_result" with tool_name in data
                        elif entry.get("event_type") == "tool_result":
                            data = entry.get("data", {})
                            tool_name = data.get("tool_name")
                            if tool_name:
                                tool_calls[tool_name] += 1
                                session_id = entry.get("session_id")
                                if session_id:
                                    sessions.add(session_id)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"Warning: Error reading {jsonl_file}: {e}")
            continue

    total_sessions = len(sessions)
    return [(tool, count, total_sessions) for tool, count in tool_calls.most_common()]


def merge_usage_data(
    db_data: List[Tuple[str, int, int]], jsonl_data: List[Tuple[str, int, int]]
) -> List[Tuple[str, int, int, float]]:
    """
    Merge usage data from multiple sources.

    Args:
        db_data: Data from ConversationStore
        jsonl_data: Data from JSONL logs

    Returns:
        List of (tool_name, calls, sessions, frequency) tuples
    """
    merged = {}

    # Combine data from both sources
    for tool_name, calls, sessions in db_data:
        if tool_name not in merged:
            merged[tool_name] = {"calls": 0, "sessions": set()}
        merged[tool_name]["calls"] += calls
        merged[tool_name]["sessions"].add(sessions) if sessions else None

    for tool_name, calls, sessions in jsonl_data:
        if tool_name not in merged:
            merged[tool_name] = {"calls": 0, "sessions": set()}
        merged[tool_name]["calls"] += calls
        if sessions:
            merged[tool_name]["sessions"].update(range(sessions))  # Approximate

    # Calculate frequency
    total_sessions = (
        max((len(merged.get(tool, {}).get("sessions", set())) for tool in merged), default=1) or 1
    )

    result = []
    for tool_name, data in merged.items():
        calls = data["calls"]
        sessions = len(data["sessions"]) if data["sessions"] else 0
        frequency = sessions / total_sessions if total_sessions > 0 else 0.0
        result.append((tool_name, calls, sessions, frequency))

    # Sort by call count
    result.sort(key=lambda x: x[1], reverse=True)

    return result


def assign_tiers(usage_data: List[Tuple[str, int, int, float]]) -> Dict[str, str]:
    """
    Assign schema tiers based on usage frequency.

    Tiers:
    - FULL: Top decile (>50% of sessions or top 10% by frequency)
    - COMPACT: Top quartile (20-50% of sessions or top 25% by frequency)
    - STUB: Remainder (<20% of sessions)

    Args:
        usage_data: List of (tool_name, calls, sessions, frequency) tuples

    Returns:
        Dict mapping tool_name -> tier (FULL/COMPACT/STUB)
    """
    if not usage_data:
        return {}

    tiers = {}

    # Calculate thresholds
    frequencies = [freq for _, _, _, freq in usage_data]

    if frequencies:
        # Top decile (>50% frequency or top 10% by rank)
        high_freq_threshold = max(
            0.50,
            frequencies[int(len(frequencies) * 0.1)] if len(frequencies) >= 10 else frequencies[-1],
        )

        # Top quartile (>20% frequency or top 25% by rank)
        med_freq_threshold = max(
            0.20,
            frequencies[int(len(frequencies) * 0.25)] if len(frequencies) >= 4 else frequencies[-1],
        )

    for tool_name, calls, sessions, frequency in usage_data:
        if frequency >= high_freq_threshold or frequency >= 0.50:
            tiers[tool_name] = "FULL"
        elif frequency >= med_freq_threshold or frequency >= 0.20:
            tiers[tool_name] = "COMPACT"
        else:
            tiers[tool_name] = "STUB"

    return tiers


def generate_tiers_yaml(
    usage_data: List[Tuple[str, int, int, float]],
    tiers: Dict[str, str],
    days: int,
    output_path: Path = None,
) -> str:
    """
    Generate tool_tiers.yaml with provenance metadata.

    Args:
        usage_data: Tool usage statistics
        tiers: Tier assignments
        days: Number of days of data analyzed
        output_path: Optional output file path

    Returns:
        YAML content as string
    """
    from victor.core.yaml_utils import safe_dump, safe_load

    # Calculate provenance
    total_calls = sum(calls for _, calls, _, _ in usage_data)
    total_sessions = max(sessions for _, _, sessions, _ in usage_data) if usage_data else 0

    # Preserve hand-curated provider_tiers and any non-tool_tiers keys.
    # Only the tool_tiers section is regenerated from telemetry.
    existing: Dict = {}
    if output_path and output_path.exists():
        try:
            existing = safe_load(output_path.read_text()) or {}
        except Exception:
            existing = {}

    yaml_content = dict(existing)
    yaml_content["provenance"] = {
        "generated_at": datetime.now().isoformat(),
        "data_source": "project.db + victor.db + JSONL logs",
        "sample_size_days": days,
        "total_sessions": total_sessions,
        "total_tool_calls": total_calls,
        "tools_analyzed": len(usage_data),
    }
    yaml_content["tool_tiers"] = {"FULL": [], "COMPACT": [], "STUB": []}

    # Populate tiers
    full_tools = []
    compact_tools = []

    for tool_name, calls, sessions, frequency in usage_data:
        tier = tiers.get(tool_name, "STUB")

        tool_entry = {
            "name": tool_name,
            "calls_last_30d": calls,
            "sessions_last_30d": sessions,
            "frequency": round(frequency, 3),
            "tier": tier,
        }

        if tier == "FULL":
            full_tools.append(tool_entry)
        elif tier == "COMPACT":
            compact_tools.append(tool_entry)

    yaml_content["tool_tiers"]["FULL"] = [t["name"] for t in full_tools]
    yaml_content["tool_tiers"]["COMPACT"] = [t["name"] for t in compact_tools]
    yaml_content["tool_tiers"]["STUB"] = "*"  # Wildcard for all other tools

    # Add detailed breakdown
    yaml_content["tools"] = {"full": full_tools, "compact": compact_tools}

    yaml_str = safe_dump(yaml_content, default_flow_style=False)

    if output_path:
        output_path.write_text(yaml_str)
        print(f"Generated {output_path}")

    return yaml_str


def print_summary(usage_data: List[Tuple[str, int, int, float]], tiers: Dict[str, str]):
    """Print summary of tool usage analysis."""
    if not usage_data:
        print("No tool usage data found.")
        print("This could mean:")
        print("  - No tool calls recorded yet")
        print("  - Database not initialized")
        print("  - Tracking not enabled")
        return

    print(f"\n{'=' * 80}")
    print("TOOL USAGE ANALYSIS")
    print("=" * 80)

    print(f"\nTotal tools analyzed: {len(usage_data)}")
    print(f"Total tool calls: {sum(calls for _, calls, _, _ in usage_data)}")
    print(f"Total sessions: {max(sessions for _, _, sessions, _ in usage_data)}")

    # Count tiers
    full_count = sum(1 for t in tiers.values() if t == "FULL")
    compact_count = sum(1 for t in tiers.values() if t == "COMPACT")
    stub_count = sum(1 for t in tiers.values() if t == "STUB")

    print("\nTier assignments:")
    print(f"  FULL: {full_count} tools (top decile, >50% sessions)")
    print(f"  COMPACT: {compact_count} tools (top quartile, >20% sessions)")
    print(f"  STUB: {stub_count} tools (remainder)")

    print(f"\n{'=' * 80}")
    print("TOP 20 TOOLS BY USAGE")
    print("=" * 80)

    print(f"\n{'Tool':<30} {'Calls':>10} {'Sessions':>10} {'Freq':>10} {'Tier':>10}")
    print("-" * 80)

    for tool_name, calls, sessions, frequency in usage_data[:20]:
        tier = tiers.get(tool_name, "STUB")
        freq_pct = f"{frequency * 100:.1f}%"
        print(f"{tool_name:<30} {calls:>10} {sessions:>10} {freq_pct:>10} {tier:>10}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze tool usage and generate data-driven tier assignments"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML file path (default: victor/config/tool_tiers.yaml)",
    )
    parser.add_argument(
        "--print-only", action="store_true", help="Print summary only, don't generate YAML"
    )

    args = parser.parse_args()

    # Collect usage data
    print(f"Analyzing tool usage from last {args.days} days...")

    db_data = analyze_tool_usage_from_db(args.days)
    jsonl_data = analyze_jsonl_logs(args.days)

    # Merge data
    usage_data = merge_usage_data(db_data, jsonl_data)

    if not usage_data:
        print("No usage data found. Exiting.")
        return

    # Assign tiers
    tiers = assign_tiers(usage_data)

    # Print summary
    print_summary(usage_data, tiers)

    # Generate YAML
    if not args.print_only:
        output_path = Path(args.output) if args.output else Path("victor/config/tool_tiers.yaml")

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generate_tiers_yaml(usage_data, tiers, args.days, output_path)

        print(f"\nGenerated {output_path}")
        print("\nNext steps:")
        print("  1. Review the generated tier assignments")
        print("  2. Adjust manually if needed based on domain knowledge")
        print("  3. Run: victor/scripts/validate_tool_strategy.py")
        print("  4. Enable feature flag: VICTOR_TOOL_STRATEGY_V2=true")


if __name__ == "__main__":
    main()
