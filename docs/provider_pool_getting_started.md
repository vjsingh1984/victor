# Provider Pool Getting Started Guide

## Overview

The Provider Pool system enables load balancing across multiple LLM provider instances for improved reliability, performance, and cost optimization. It automatically distributes requests using configurable strategies, monitors health, and handles failover.

## Features

- **Load Balancing Strategies**: Round-robin, least connections, adaptive, random
- **Health Monitoring**: Automatic health checks and failover
- **Connection Warmup**: Reduce cold start latency
- **Circuit Breaking**: Prevent cascading failures
- **Performance Metrics**: Track latency, error rates, and request distribution

## Quick Start

### 1. Enable Provider Pool via CLI

```bash
# Enable provider pool with default settings
victor chat --pool

# Customize pool size and load balancer
victor chat --pool --pool-size 5 --pool-load-balancer adaptive

# Disable warmup for faster startup
victor chat --pool --no-pool-warmup

# Customize health check interval
victor chat --pool --pool-health-check-interval 60
```

### 2. Configure via Settings

Add to your `~/.victor/config.yaml` or `.env` file:

```yaml
# config.yaml
enable_provider_pool: true
pool_size: 3
pool_load_balancer: adaptive
pool_enable_warmup: true
pool_warmup_concurrency: 3
pool_health_check_interval: 30
pool_max_retries: 3
pool_min_instances: 1
```

Or via environment variables:

```bash
export VICTOR_ENABLE_PROVIDER_POOL=true
export VICTOR_POOL_SIZE=3
export VICTOR_POOL_LOAD_BALANCER=adaptive
export VICTOR_POOL_ENABLE_WARMUP=true
```

### 3. Configure Multiple Endpoints

The provider pool automatically detects multiple endpoints for load balancing:

```yaml
# config.yaml
lmstudio_base_urls:
  - http://localhost:1234
  - http://localhost:1235
  - http://localhost:1236
```

Or for Ollama:

```bash
export OLLAMA_BASE_URL="http://localhost:11434,http://localhost:11435,http://localhost:11436"
```

## Load Balancing Strategies

### Adaptive (Recommended)

**Best for**: Production environments with varying performance

The adaptive strategy selects providers based on:
- Current latency (prefer lower latency)
- Error rate (prefer lower error rate)
- Active connections (prefer fewer connections)
- Recent performance trends

```yaml
pool_load_balancer: adaptive
```

### Round Robin

**Best for**: Simple, predictable distribution

Distributes requests sequentially across all healthy providers.

```yaml
pool_load_balancer: round_robin
```

### Least Connections

**Best for**: Variable request durations

Routes requests to the provider with the fewest active connections.

```yaml
pool_load_balancer: least_connections
```

### Random

**Best for**: Testing or equally capable providers

Selects a random healthy provider.

```yaml
pool_load_balancer: random
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_provider_pool` | bool | false | Enable provider pool |
| `pool_size` | int | 3 | Maximum number of provider instances (1-10) |
| `pool_load_balancer` | str | adaptive | Load balancing strategy |
| `pool_enable_warmup` | bool | true | Warm up connections on startup |
| `pool_warmup_concurrency` | int | 3 | Concurrent warmup requests (1-10) |
| `pool_health_check_interval` | int | 30 | Health check interval in seconds (5-300) |
| `pool_max_retries` | int | 3 | Maximum retry attempts (1-10) |
| `pool_min_instances` | int | 1 | Minimum healthy instances required |

## Usage Examples

### Example 1: Local Development with Multiple Ollama Instances

```bash
# Start multiple Ollama instances
OLLAMA_HOST=11434 ollama serve &
OLLAMA_HOST=11435 ollama serve &
OLLAMA_HOST=11436 ollama serve &

# Configure Victor to use all instances
export OLLAMA_BASE_URL="http://localhost:11434,http://localhost:11435,http://localhost:11436"
export VICTOR_ENABLE_PROVIDER_POOL=true
export VICTOR_POOL_LOAD_BALANCER=adaptive

# Start Victor
victor chat
```

### Example 2: Production with LMStudio Cluster

```yaml
# ~/.victor/config.yaml
default_provider: lmstudio
default_model: qwen3-coder:30b

lmstudio_base_urls:
  - http://gpu-server-1:1234
  - http://gpu-server-2:1234
  - http://gpu-server-3:1234

enable_provider_pool: true
pool_size: 3
pool_load_balancer: adaptive
pool_enable_warmup: true
pool_health_check_interval: 30
pool_max_retries: 3
```

### Example 3: Testing with Random Load Balancer

```bash
victor chat \
  --pool \
  --pool-size 5 \
  --pool-load-balancer random \
  --no-pool-warmup
```

## Monitoring and Debugging

### View Pool Statistics

The provider pool logs comprehensive statistics:

```
INFO: Creating provider pool with 3 instances using adaptive load balancing
INFO: Provider pool stats: {
  "pool_name": "lmstudio-pool",
  "instances": {
    "total": 3,
    "healthy": 3,
    "accepting_traffic": 3
  },
  "config": {
    "load_balancer": "adaptive",
    "pool_size": 3
  }
}
```

### Debug Mode

Enable debug logging to see detailed pool operations:

```bash
victor chat --pool --log-level DEBUG
```

Output includes:
- Provider selection decisions
- Health check results
- Load balancing scores
- Failover events

### Health Status

Check individual provider health:

```python
import asyncio
from victor.providers.health_monitor import get_health_registry

async def check_health():
    registry = await get_health_registry()
    stats = await registry.get_all_stats()
    print(stats)

asyncio.run(check_health())
```

## Performance Tips

### 1. Enable Warmup for Production

Warmup reduces first-request latency by 40-60%:

```yaml
pool_enable_warmup: true
pool_warmup_concurrency: 3
```

### 2. Tune Health Check Interval

Balance between responsiveness and overhead:

- **Development**: 10-30 seconds (faster detection)
- **Production**: 30-60 seconds (lower overhead)

```yaml
pool_health_check_interval: 30
```

### 3. Adjust Pool Size

Match pool size to available resources:

- **CPU-bound**: 1-2 instances per CPU core
- **I/O-bound**: 3-5 instances per CPU core

```yaml
pool_size: 3
```

### 4. Use Adaptive Load Balancing

Adaptive provides best overall performance in production:

```yaml
pool_load_balancer: adaptive
```

## Troubleshooting

### Pool Not Creating

**Problem**: Provider pool not enabled despite `--pool` flag

**Solutions**:
1. Check that provider has multiple endpoints configured
2. Verify settings: `victor chat --pool --log-level DEBUG`
3. Check logs for: "Provider pool enabled but X has only one endpoint"

### All Providers Unhealthy

**Problem**: All providers marked as unhealthy

**Solutions**:
1. Check provider connectivity: `curl http://localhost:1234`
2. Increase health check interval: `--pool-health-check-interval 60`
3. Check provider logs for errors
4. Verify API keys and authentication

### High Latency

**Problem**: Pool requests slower than single provider

**Solutions**:
1. Reduce pool size: `--pool-size 2`
2. Use round-robin instead of adaptive: `--pool-load-balancer round_robin`
3. Disable warmup: `--no-pool-warmup`
4. Check network latency between Victor and providers

### Uneven Load Distribution

**Problem**: Some providers receive more traffic than others

**Solutions**:
1. Check provider weights (if configured)
2. Use least-connections strategy: `--pool-load-balancer least_connections`
3. Verify all providers are healthy
4. Check for network issues affecting specific providers

## Advanced Usage

### Programmatic Pool Creation

```python
import asyncio
from victor.providers.provider_pool import (
    create_provider_pool,
    ProviderPoolConfig,
)
from victor.providers.load_balancer import LoadBalancerType

async def main():
    # Create multiple provider instances
    providers = {
        "provider-0": OllamaProvider(base_url="http://localhost:11434"),
        "provider-1": OllamaProvider(base_url="http://localhost:11435"),
        "provider-2": OllamaProvider(base_url="http://localhost:11436"),
    }

    # Configure pool
    config = ProviderPoolConfig(
        pool_size=3,
        load_balancer=LoadBalancerType.ADAPTIVE,
        enable_warmup=True,
    )

    # Create pool
    pool = await create_provider_pool(
        name="ollama-pool",
        providers=providers,
        config=config,
    )

    # Use pool
    response = await pool.chat(
        messages=[Message(role="user", content="Hello")],
        model="llama2",
    )

    # Get stats
    stats = pool.get_pool_stats()
    print(stats)

    # Cleanup
    await pool.close()

asyncio.run(main())
```

### Custom Health Checks

```python
from victor.providers.health_monitor import HealthMonitor, HealthCheckConfig

class CustomHealthMonitor(HealthMonitor):
    async def _perform_health_check(self):
        # Custom health check logic
        try:
            # Ping provider endpoint
            response = await self._ping_provider()
            if response.status_code == 200:
                self.record_success(100)
            else:
                self.record_failure()
        except Exception as e:
            self.record_failure()
```

## Best Practices

1. **Start Small**: Begin with 2-3 providers, scale up as needed
2. **Monitor Health**: Regularly check pool stats and health metrics
3. **Test Failover**: Simulate provider failures to test failover logic
4. **Use Adaptive**: Prefer adaptive load balancing for production
5. **Enable Warmup**: Reduce cold start latency in production
6. **Tune Intervals**: Adjust health check intervals based on SLA requirements
7. **Set Timeouts**: Configure appropriate timeouts for provider requests
8. **Monitor Costs**: Track API usage across all pool instances

## Integration with Existing Code

The provider pool is fully backward compatible. When disabled, Victor uses a single provider as before. When enabled, the pool acts as a drop-in replacement:

```python
# No code changes needed - pool acts like a single provider
response = await orchestrator.provider.chat(
    messages=messages,
    model=model,
)

# Streaming also works
async for chunk in orchestrator.provider.stream(
    messages=messages,
    model=model,
):
    print(chunk.content)
```

## Additional Resources

- [Provider Pool Implementation Summary](provider_pool_implementation_summary.md)
- [Provider Pool Architecture](../victor/providers/provider_pool.py)
- [Load Balancer Strategies](../victor/providers/load_balancer.py)
- [Health Monitoring System](../victor/providers/health_monitor.py)
- [Integration Tests](../tests/integration/test_provider_pool_integration.py)

## Support

For issues or questions:
1. Check logs with `--log-level DEBUG`
2. Review pool statistics
3. Verify provider connectivity
4. Check configuration in `~/.victor/config.yaml`
5. Open an issue on GitHub
