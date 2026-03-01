# Victor Observability Documentation

Welcome to the Victor Observability documentation!

## Overview

Victor provides a comprehensive observability system for monitoring all aspects of the framework:

- **Metrics Collection**: Unified metrics from all components
- **Dashboard**: Real-time dashboard with CLI commands
- **Alerting**: Automatic alerts for common issues
- **Historical Data**: Time-series metrics with configurable retention
- **System Monitoring**: CPU, memory, and process metrics

## Quick Links

- [Observability Guide](OBSERVABILITY_GUIDE.md) - Complete user guide
- [API Reference](API.md) - Detailed API documentation
- [Examples](examples/) - Code examples and tutorials

## Features

### Unified Metrics Aggregation

Collect metrics from:
- Capabilities (access counts, errors)
- Caches (hit rates, evictions, size)
- Tools (calls, errors, latency)
- Coordinators (operations, errors, latency)
- Verticals (requests, errors, latency)
- System Resources (CPU, memory)

### Dashboard CLI

```bash
victor observability dashboard              # Show dashboard
victor observability dashboard --json       # JSON output
victor observability dashboard --watch      # Real-time updates
victor observability metrics [source]       # Specific source
victor observability history [hours]        # Historical data
victor observability stats                  # Statistics
```

### Automatic Alerting

- ⚠️ Low cache hit rate (<50%)
- ❌ High tool error rate (>10%)
- ⚠️ High memory usage (>80%)

### Performance

- <5% performance overhead
- Background metrics collection
- Efficient data structures

## Getting Started

### Installation

Observability is included with Victor:

```bash
pip install victor-ai
```

### Basic Usage

```python
from victor.framework.observability import ObservabilityManager

# Get the manager
manager = ObservabilityManager.get_instance()

# Register a metrics source
manager.register_source(my_component)

# Get dashboard data
dashboard_data = manager.get_dashboard_data()
print(f"Cache hit rate: {dashboard_data.cache_metrics['hit_rate']:.2%}")
```

### CLI Usage

```bash
# Show dashboard
victor observability dashboard

# Show as JSON
victor observability dashboard --json

# Watch mode
victor observability dashboard --watch --interval 5
```

## Documentation

- **[Observability Guide](OBSERVABILITY_GUIDE.md)**: Complete guide with examples
- **[API Reference](API.md)**: Detailed API documentation
- **[Examples](examples/)**: Code examples

## Support

For issues and questions:
- GitHub: https://github.com/vijayds/victor-ai/issues
- Documentation: https://victor-ai.readthedocs.io/

---

**Version**: 1.0
**Last Updated**: 2025-02-28
