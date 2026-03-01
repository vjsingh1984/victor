"""Example: Exporting metrics to external systems.

This example demonstrates how to export metrics from ObservabilityManager
to external monitoring systems like Prometheus, DataDog, or custom endpoints.
"""

import json
import time
from victor.framework.observability import ObservabilityManager


def export_to_json(filename: str = "metrics.json"):
    """Export dashboard data to JSON file.

    Args:
        filename: Output filename
    """
    manager = ObservabilityManager.get_instance()
    data = manager.get_dashboard_data()

    with open(filename, "w") as f:
        json.dump(data.to_dict(), f, indent=2)

    print(f"Exported metrics to {filename}")


def export_to_prometheus_format():
    """Export metrics in Prometheus text format.

    Prometheus expects metrics in the format:
    metric_name{"label"="value"} value
    """
    manager = ObservabilityManager.get_instance()
    collection = manager.collect_metrics()

    for snapshot in collection.snapshots:
        for metric in snapshot.metrics:
            # Format: metric_name{labels} value
            labels_str = ""
            if hasattr(metric, 'labels') and metric.labels:
                labels = ",".join(
                    f'{label.key}="{label.value}"' for label in metric.labels
                )
                labels_str = f"{{{labels}}}"

            print(f"{metric.name}{labels_str} {metric.value}")


def export_to_custom_endpoint(url: str, api_key: str = None):
    """Export metrics to custom HTTP endpoint.

    Args:
        url: Endpoint URL
        api_key: Optional API key for authentication
    """
    import httpx

    manager = ObservabilityManager.get_instance()
    data = manager.get_dashboard_data()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = httpx.post(
        url,
        json=data.to_dict(),
        headers=headers,
        timeout=10.0,
    )

    response.raise_for_status()
    print(f"Exported metrics to {url}")


def create_metrics_summary():
    """Create a human-readable metrics summary."""
    manager = ObservabilityManager.get_instance()
    data = manager.get_dashboard_data()

    summary = f"""
Victor Metrics Summary
======================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data.timestamp))}

Sources
------
Total: {data.total_sources}
By Type:
"""

    for source_type, count in data.sources_by_type.items():
        summary += f"  {source_type}: {count}\n"

    summary += f"""
Cache Metrics
-------------
Hit Rate: {data.cache_metrics.get('hit_rate', 0):.1%}
Total Hits: {data.cache_metrics.get('total_hits', 0)}
Total Misses: {data.cache_metrics.get('total_misses', 0)}
Evictions: {data.cache_metrics.get('evictions', 0)}

Tool Metrics
------------
Total Calls: {data.tool_metrics.get('total_calls', 0)}
Success Rate: {data.tool_metrics.get('success_rate', 0):.1%}
Average Latency: {data.tool_metrics.get('average_latency', 0):.3f}s

System Metrics
--------------
Memory Usage: {data.system_metrics.get('memory_usage_bytes', 0) / 1024 / 1024:.1f} MB
CPU Usage: {data.system_metrics.get('cpu_usage_percent', 0):.1f}%

Alerts
------
"""

    if data.alerts:
        for alert in data.alerts:
            severity_icon = "❌" if alert["severity"] == "error" else "⚠️"
            summary += f'{severity_icon} {alert["type"]}: {alert["message"]}\n'
    else:
        summary += "No alerts\n"

    print(summary)


def export_historical_data(hours: float = 1.0, filename: str = "history.json"):
    """Export historical metrics data.

    Args:
        hours: Hours of history to export
        filename: Output filename
    """
    manager = ObservabilityManager.get_instance()
    history = manager.get_historical_data(hours=hours)

    data = {
        "hours": hours,
        "collections": [collection.to_dict() for collection in history],
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(history)} collections to {filename}")


if __name__ == "__main__":
    # Example 1: Export to JSON
    print("Example 1: Export to JSON")
    export_to_json()

    # Example 2: Prometheus format
    print("\nExample 2: Prometheus Format")
    export_to_prometheus_format()

    # Example 3: Human-readable summary
    print("\nExample 3: Metrics Summary")
    create_metrics_summary()

    # Example 4: Historical data
    print("\nExample 4: Historical Data")
    export_historical_data(hours=1.0)

    # Example 5: Custom endpoint (commented out - requires actual endpoint)
    # print("\nExample 5: Custom Endpoint")
    # export_to_custom_endpoint("https://api.example.com/metrics", api_key="your-key")
