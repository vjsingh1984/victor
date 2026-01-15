# Quick Start: Coordinator Performance Dashboard

This guide will help you get the dashboard up and running in under 5 minutes.

## Prerequisites

- Python 3.10+
- Victor installed locally

## Step 1: Install Dependencies

```bash
# Install psutil for system metrics
pip install psutil

# Or install all dev dependencies
pip install -e ".[dev]"
```

## Step 2: Generate Initial Metrics

```bash
# Generate demo metrics
python scripts/generate_coordinator_metrics.py --sample-data

# Or run actual benchmarks (takes longer)
python scripts/generate_coordinator_metrics.py --benchmark --iterations 50
```

## Step 3: View Dashboard

**Option A: Open directly in browser**
```bash
open docs/dashboard/coordinator_performance.html
# On Linux: xdg-open docs/dashboard/coordinator_performance.html
```

**Option B: Start HTTP server (recommended)**
```bash
cd docs/dashboard
python -m http.server 8000
# Open: http://localhost:8000/coordinator_performance.html
```

## Step 4: Explore the Dashboard

You should see:
- **Overview metrics**: Total executions, active coordinators, error rate, throughput
- **Performance charts**: Latency, throughput, memory, cache hit rates
- **Coordinator table**: Detailed metrics for each coordinator
- **Error log**: Recent errors (if any)

The dashboard auto-refreshes every 30 seconds, or click "🔄 Refresh Metrics" to update manually.

## Next Steps

### Collect Real Metrics

To collect metrics from a running Victor instance:

```python
from victor.observability.coordinator_metrics import get_coordinator_metrics_collector

# Get the collector
collector = get_coordinator_metrics_collector()

# It will automatically track coordinators that use it
# Export metrics
with open("metrics.json", "w") as f:
    f.write(collector.export_json(include_history=True))
```

### Schedule Automatic Collection

**Using cron (Linux/Mac)**:
```bash
# Edit crontab
crontab -e

# Add line to collect every 5 minutes
*/5 * * * * cd /path/to/victor && python scripts/generate_coordinator_metrics.py --output docs/dashboard/metrics.json
```

**Using Task Scheduler (Windows)**:
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger to "Every 5 minutes"
4. Action: Run `python scripts/generate_coordinator_metrics.py --output docs/dashboard/metrics.json`

### Integration with Prometheus

```bash
# Generate Prometheus format metrics
python scripts/generate_coordinator_metrics.py --format prometheus --output /var/lib/node_exporter/textfile_collector/victor_metrics.prom
```

Add to Prometheus configuration:
```yaml
scrape_configs:
  - job_name: 'victor'
    file_sd_configs:
      - files:
        - '/var/lib/node_exporter/textfile_collector/victor_metrics.prom'
```

## Troubleshooting

### Dashboard shows "Loading..."
```bash
# Check metrics.json exists
ls docs/dashboard/metrics.json

# Regenerate metrics
python scripts/generate_coordinator_metrics.py --sample-data
```

### ModuleNotFoundError: No module named 'psutil'
```bash
pip install psutil
```

### Charts not rendering
1. Check browser console (F12) for errors
2. Ensure Chart.js CDN is accessible
3. Try different browser

### Want to run benchmarks without psutil?
The code handles missing psutil gracefully - memory/CPU metrics will show as 0.

```bash
# This will work even without psutil installed
python scripts/generate_coordinator_metrics.py --sample-data
```

## Demo Mode

If metrics.json is missing or invalid, the dashboard automatically shows demo data with realistic metrics. This is useful for:
- Testing dashboard functionality
- Demonstrating the dashboard
- Development without real metrics

## Support

- Full documentation: `docs/dashboard/README.md`
- Report issues: GitHub Issues
- Architecture: `docs/metrics/orchestrator_refactoring_analysis.md`

Happy monitoring! 📊
