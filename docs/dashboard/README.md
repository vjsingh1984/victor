# Coordinator Performance Dashboard

A comprehensive performance monitoring dashboard for the coordinator-based orchestrator architecture in Victor.

## Overview

The dashboard provides real-time visibility into:
- Coordinator execution times and latency
- Memory usage by coordinator
- Cache hit rates
- Error rates and tracking
- Analytics event counts
- Throughput metrics

## Features

- **Real-time Metrics**: Auto-refreshes every 30 seconds
- **Interactive Charts**: Built with Chart.js for responsive visualizations
- **Multi-format Export**: JSON and Prometheus format support
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Demo Mode**: Shows sample data when metrics are unavailable

## Quick Start

### 1. Install Dependencies

```bash
# Install additional dependencies if needed
pip install psutil
```

### 2. Generate Metrics

Run the metrics generator script to collect performance data:

```bash
# Basic usage - generates demo data
python scripts/generate_coordinator_metrics.py --sample-data

# Run benchmarks and collect metrics
python scripts/generate_coordinator_metrics.py --benchmark --iterations 100

# Custom output location
python scripts/generate_coordinator_metrics.py --output /path/to/metrics.json

# Export in Prometheus format
python scripts/generate_coordinator_metrics.py --format prometheus --output metrics.prom
```

### 3. View Dashboard

Open the dashboard in your browser:

```bash
# Option 1: Open directly
open docs/dashboard/coordinator_performance.html

# Option 2: Start a simple HTTP server
cd docs/dashboard
python -m http.server 8000
# Then navigate to http://localhost:8000/coordinator_performance.html
```

## Dashboard Components

### Overview Metrics

The top section shows key performance indicators:
- **Total Executions**: Number of coordinator executions
- **Active Coordinators**: Number of coordinators running
- **Error Rate**: Overall error percentage
- **Throughput**: Requests per second

### Performance Charts

#### 1. Execution Latency (p95)
Bar chart showing 95th percentile latency for each coordinator.

**What to look for**:
- Latency under 50ms is excellent
- Latency between 50-100ms is good
- Latency over 100ms needs investigation

#### 2. Throughput (Requests/sec)
Line chart showing throughput across coordinators.

**What to look for**:
- Higher is better
- Sudden drops may indicate performance issues
- Compare against baseline

#### 3. Memory Usage by Coordinator
Doughnut chart showing memory distribution.

**What to look for**:
- Identify memory-intensive coordinators
- Check for memory leaks (growing over time)
- Compare against expected usage

#### 4. Cache Hit Rates
Bar chart showing cache performance.

**What to look for**:
- Hit rate above 70% is good
- Hit rate above 90% is excellent
- Low hit rates may indicate cache tuning needed

### Coordinator Breakdown Table

Detailed metrics for each coordinator:

| Column | Description |
|--------|-------------|
| Coordinator | Name of the coordinator |
| Executions | Total number of executions |
| Avg Latency | Average execution time |
| p95 Latency | 95th percentile latency |
| Errors | Number of errors |
| Error Rate | Error percentage |
| Cache Hit Rate | Cache performance |
| Status | Health status (Healthy/Warning/Error) |

### Error Log

Shows recent errors across all coordinators with timestamps and error rates.

## Advanced Usage

### Collecting Metrics in Production

To collect metrics from a running Victor instance:

```python
from victor.observability.coordinator_metrics import (
    CoordinatorMetricsCollector,
    get_coordinator_metrics_collector
)

# Get the singleton collector
collector = get_coordinator_metrics_collector()

# Track coordinator execution
with collector.track_coordinator("ChatCoordinator"):
    result = await orchestrator.chat("Hello")

# Record cache metrics
collector.record_cache_hit("PromptCoordinator")
collector.record_cache_miss("ContextCoordinator")

# Record analytics events
collector.record_analytics_event("tool_call", count=5)

# Export metrics
json_metrics = collector.export_json(include_history=True)
prometheus_metrics = collector.export_prometheus()

# Save to file
with open("metrics.json", "w") as f:
    f.write(json_metrics)
```

### Automatic Metrics Collection

Integrate metrics collection into your coordinators:

```python
from victor.observability.coordinator_metrics import track_coordinator_metrics

class ChatCoordinator:
    def __init__(self):
        self.metrics = get_coordinator_metrics_collector()

    @track_coordinator_metrics(get_coordinator_metrics_collector(), "ChatCoordinator")
    async def process_chat(self, message: str):
        # Your chat logic here
        return await self._do_chat(message)
```

### Scheduling Periodic Metrics Collection

Create a cron job or scheduled task to collect metrics:

```bash
# Add to crontab (crontab -e)
# Collect metrics every 5 minutes
*/5 * * * * cd /path/to/victor && python scripts/generate_coordinator_metrics.py --output docs/dashboard/metrics.json
```

Or use systemd:

```ini
# /etc/systemd/system/victor-metrics.service
[Unit]
Description=Victor Metrics Collector
After=network.target

[Service]
Type=oneshot
User=victor
WorkingDirectory=/path/to/victor
ExecStart=/usr/bin/python scripts/generate_coordinator_metrics.py --output docs/dashboard/metrics.json

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/victor-metrics.timer
[Unit]
Description=Victor Metrics Collector Timer
Requires=victor-metrics.service

[Timer]
OnCalendar=*:0/5
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
```

Enable and start:
```bash
sudo systemctl enable victor-metrics.timer
sudo systemctl start victor-metrics.timer
```

## Interpreting Metrics

### Performance Baselines

Based on the orchestrator refactoring analysis:

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Coordinator overhead | < 10% | 10-15% | > 15% |
| P95 latency | < 100ms | 100-200ms | > 200ms |
| Error rate | < 1% | 1-5% | > 5% |
| Cache hit rate | > 70% | 50-70% | < 50% |
| Memory growth | < 5MB/hour | 5-10MB/hour | > 10MB/hour |

### Common Issues and Solutions

#### High Latency

**Symptoms**: P95 latency > 200ms

**Possible causes**:
- Slow coordinator implementations
- Database/network bottlenecks
- Large payload sizes
- Resource contention

**Solutions**:
1. Profile slow coordinators
2. Optimize database queries
3. Implement request batching
4. Scale resources

#### High Error Rate

**Symptoms**: Error rate > 5%

**Possible causes**:
- Dependency failures
- Resource exhaustion
- Configuration errors
- Network issues

**Solutions**:
1. Check error log for patterns
2. Verify dependencies
3. Check resource limits
4. Review configuration

#### Low Cache Hit Rate

**Symptoms**: Cache hit rate < 50%

**Possible causes**:
- Insufficient cache size
- Poor cache key design
- High data churn
- Inefficient eviction policy

**Solutions**:
1. Increase cache size
2. Review cache key strategy
3. Adjust TTL settings
4. Consider different eviction policies

#### Memory Growth

**Symptoms**: Memory growing > 10MB/hour

**Possible causes**:
- Memory leaks
- Unbounded collections
- Large object retention
- Inefficient data structures

**Solutions**:
1. Profile memory usage
2. Check for unbounded collections
3. Implement proper cleanup
4. Review object lifecycle

## Metrics Format Reference

### JSON Format

```json
{
  "timestamp": "2025-01-13T12:00:00Z",
  "overall_stats": {
    "total_executions": 5000,
    "total_coordinators": 13,
    "total_errors": 25,
    "overall_error_rate": 0.005,
    "uptime_seconds": 3600
  },
  "coordinators": [
    {
      "coordinator_name": "ChatCoordinator",
      "memory_bytes": 5242880,
      "memory_mb": 5.0,
      "cpu_percent": 2.5,
      "execution_count": 1500,
      "total_duration_ms": 22500.0,
      "error_count": 5,
      "cache_hits": 800,
      "cache_misses": 200,
      "cache_hit_rate": 0.8
    }
  ]
}
```

### Prometheus Format

```
victor_coordinator_executions_total{coordinator="ChatCoordinator"} 1500
victor_coordinator_duration_ms_total{coordinator="ChatCoordinator"} 22500.00
victor_coordinator_errors_total{coordinator="ChatCoordinator"} 5
victor_coordinator_memory_bytes{coordinator="ChatCoordinator"} 5242880
victor_coordinator_cache_hit_rate{coordinator="ChatCoordinator"} 0.8000
victor_coordinator_cache_hits_total{coordinator="ChatCoordinator"} 800
victor_coordinator_cache_misses_total{coordinator="ChatCoordinator"} 200
victor_coordinator_uptime_seconds 3600.00
victor_coordinator_total_executions 5000
```

## Integration with Monitoring Systems

### Prometheus

1. Generate metrics in Prometheus format:
```bash
python scripts/generate_coordinator_metrics.py --format prometheus --output /var/lib/victor/metrics.prom
```

2. Configure Prometheus scrape job:
```yaml
scrape_configs:
  - job_name: 'victor-coordinators'
    static_configs:
      - targets: ['localhost:9090']
    file_sd_configs:
      - files:
        - '/var/lib/victor/metrics.prom'
```

### Grafana

Import the dashboard into Grafana:

1. Add Prometheus data source
2. Create new dashboard
3. Add panels for:
   - Coordinator latency (p95, p99)
   - Error rate by coordinator
   - Cache hit rates
   - Memory usage
   - Throughput

Example Prometheus queries:

```promql
# P95 latency by coordinator
histogram_quantile(0.95, sum(rate(victor_coordinator_duration_ms_bucket[5m])) by (coordinator, le))

# Error rate
sum(rate(victor_coordinator_errors_total[5m])) by (coordinator) /
sum(rate(victor_coordinator_executions_total[5m])) by (coordinator)

# Cache hit rate
sum(rate(victor_coordinator_cache_hits_total[5m])) by (coordinator) /
(sum(rate(victor_coordinator_cache_hits_total[5m])) by (coordinator) +
 sum(rate(victor_coordinator_cache_misses_total[5m])) by (coordinator))
```

## Troubleshooting

### Dashboard Shows "Loading..."

**Problem**: Dashboard can't load metrics.json

**Solutions**:
1. Check file exists: `ls docs/dashboard/metrics.json`
2. Check file permissions
3. Generate metrics: `python scripts/generate_coordinator_metrics.py`
4. Check browser console for CORS errors (use HTTP server)

### Charts Not Rendering

**Problem**: Charts show blank or don't load

**Solutions**:
1. Check Chart.js CDN is accessible
2. Check browser console for JavaScript errors
3. Verify metrics.json is valid JSON
4. Try different browser

### Outdated Metrics

**Problem**: Dashboard shows old data

**Solutions**:
1. Click "Refresh Metrics" button
2. Check auto-refresh is enabled
3. Verify metrics generation is scheduled
4. Check script execution logs

### High Memory Usage in Collector

**Problem**: Metrics collector consuming too much memory

**Solutions**:
1. Reduce `max_history` parameter
2. Reset metrics periodically
3. Export and clear old data
4. Increase memory limits

## Best Practices

1. **Schedule Regular Collection**: Use cron or systemd for automated metrics collection
2. **Monitor Trends**: Look for patterns over time, not just snapshots
3. **Set Alerts**: Configure alerts for critical thresholds
4. **Archive Historical Data**: Keep metrics history for trend analysis
5. **Regular Cleanup**: Reset or archive old metrics to prevent memory growth
6. **Correlate with Logs**: Cross-reference metrics with application logs
7. **Baseline Performance**: Establish baselines during normal operations

## Development

### Running Locally

```bash
# Generate demo data
python scripts/generate_coordinator_metrics.py --sample-data --output docs/dashboard/metrics.json

# Start HTTP server
cd docs/dashboard
python -m http.server 8000

# Open browser
open http://localhost:8000/coordinator_performance.html
```

### Customizing Dashboard

Edit `docs/dashboard/coordinator_performance.html`:
- Modify chart configurations
- Add new metrics cards
- Change color schemes
- Adjust refresh intervals

### Adding New Metrics

1. Add metric to `CoordinatorMetricsCollector`
2. Export in JSON/Prometheus format
3. Update dashboard to display new metric
4. Regenerate metrics with new data

## Related Documentation

- [Orchestrator Refactoring Analysis](../metrics/orchestrator_refactoring_analysis.md)
- [Coordinator Architecture](../architecture/coordinator-based-orchestrator.md)
- [Performance Benchmarks](../development/performance-testing.md)

## Support

For issues or questions:
1. Check this README
2. Review troubleshooting section
3. Check existing GitHub issues
4. Create new issue with details

## License

Apache License 2.0 - See LICENSE file for details
