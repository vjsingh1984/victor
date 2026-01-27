# Load Testing and Scalability Tests

This directory contains comprehensive load testing and scalability testing infrastructure for Victor AI.

## Overview

The load testing framework provides:

1. **Locust-based HTTP load testing** - Simulate realistic user traffic
2. **Pytest-based scalability tests** - Test system limits and resource usage
3. **Performance benchmarking** - Establish and track performance baselines
4. **Memory leak detection** - Identify memory leaks over long-running sessions
5. **Stress testing** - Find breaking points and test graceful degradation

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -e ".[dev]"

# This installs:
# - locust>=2.0 (load testing)
# - pytest>=8.0 (test framework)
# - psutil>=5.9 (system metrics)
# - aiohttp>=3.9 (async HTTP client)
```

### Running Load Tests

#### Option 1: Quick Pytest Tests

```bash
# Run quick concurrent request test
make load-test-quick

# Or directly with pytest
pytest tests/load_test/test_scalability.py::TestConcurrentRequests::test_10_concurrent_requests -v -s
```

#### Option 2: Locust Web Interface

```bash
# 1. Start the Victor API server
victor serve

# 2. In another terminal, start Locust
make load-test

# 3. Open browser to http://localhost:8089
# 4. Configure number of users and spawn rate
# 5. Start swarming
```

#### Option 3: Headless Load Test

```bash
# Run 5-minute load test with 100 users
make load-test-headless

# This will:
# - Spawn 100 users over 10 seconds
# - Run for 5 minutes
# - Generate HTML report: load-test-report.html
```

#### Option 4: Generate Report

```bash
# Generate timestamped load test report
make load-test-report

# Report saved to: load-test-reports/report_YYYYMMDD_HHMMSS.html
```

### Running Scalability Tests

```bash
# Run all scalability tests (WARNING: takes 30+ minutes)
make scalability-test

# Run specific test category
pytest tests/load_test/test_scalability.py::TestConcurrentRequests -v -s
pytest tests/load_test/test_scalability.py::TestLargeContext -v -s
pytest tests/load_test/test_scalability.py::TestMemoryLeaks -v -s
pytest tests/load_test/test_scalability.py::TestStressTesting -v -s
```

### Running Performance Benchmarks

```bash
# Run performance baseline tests
make benchmark

# Run all benchmarks
make benchmark-all

# Check for performance regressions
make performance-regression-check
```

## Test Files

### load_test_framework.py

Main load testing framework using Locust.

**Key Classes**:
- `VictorLoadTest` - Simulates realistic Victor AI usage patterns
- `StressTestUser` - Aggressive load testing to find breaking points
- `MemoryLeakTestUser` - Long-running sessions to detect memory leaks
- `AsyncLoadTestFramework` - Advanced async testing framework

**Usage**:
```python
from locust import run_single_user
from tests.load_test.load_test_framework import VictorLoadTest

# Run single user test for development
run_single_user(VictorLoadTest)
```

### test_scalability.py

Comprehensive scalability test suite.

**Test Classes**:
- `TestConcurrentRequests` - Concurrent request handling
- `TestLargeContext` - Large conversation context management
- `TestMemoryLeaks` - Memory leak detection
- `TestStressTesting` - Find breaking points
- `TestEndurance` - Long-running stability tests

**Example**:
```bash
# Test concurrent request handling
pytest tests/load_test/test_scalability.py::TestConcurrentRequests -v

# Test with 100 concurrent requests
pytest tests/load_test/test_scalability.py::TestConcurrentRequests::test_100_concurrent_requests -v -s
```

## Test Scenarios

### 1. Concurrent Request Testing

**Purpose**: Measure throughput and latency under concurrent load

**Tests**:
- 10 concurrent requests (quick smoke test)
- 100 concurrent requests (normal load)
- 500 concurrent requests (stress test)
- Throughput degradation analysis

**Metrics**:
- Requests per second
- Response time percentiles (P50, P95, P99)
- Error rate
- Resource utilization (CPU, memory)

### 2. Large Context Testing

**Purpose**: Evaluate performance with large conversation histories

**Tests**:
- 100-turn conversations
- 500-turn conversations
- 1000-turn conversations
- Memory usage tracking
- Context compaction performance

**Metrics**:
- Response time vs conversation length
- Memory growth rate
- Context compaction time
- GC pressure

### 3. Memory Leak Testing

**Purpose**: Detect memory leaks over long-running sessions

**Tests**:
- 1000 requests memory leak detection
- Resource cleanup verification
- ServiceContainer lifecycle testing
- Session cleanup verification

**Metrics**:
- Memory growth per request
- Final vs initial memory
- Container leak detection
- GC effectiveness

### 4. Stress Testing

**Purpose**: Find system breaking points

**Tests**:
- Breaking point identification
- Graceful degradation testing
- Extreme concurrency (1000+ users)
- Error recovery testing

**Metrics**:
- Error rate vs concurrency
- Breaking point user count
- Degradation pattern
- Recovery success rate

### 5. Endurance Testing

**Purpose**: Validate stability over extended periods

**Tests**:
- 1-hour sustained load test
- Memory trend analysis
- Performance stability over time
- Resource leak detection

**Metrics**:
- Performance over time
- Memory growth rate
- Error rate over time
- Long-term stability

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Single Request Latency** | | |
| P50 | <100ms | Median response time |
| P95 | <300ms | 95th percentile |
| P99 | <500ms | 99th percentile |
| **Throughput** | | |
| Requests/second | >100 req/s | With 100 concurrent users |
| Concurrent users | 100 | Recommended max per instance |
| **Memory Usage** | | |
| Per session | <10MB | Average per active session |
| 100 sessions | <1GB | Total for 100 concurrent sessions |
| **Tool Execution** | | |
| Fast tool P50 | <50ms | Median tool execution time |
| Fast tool P99 | <200ms | 99th percentile tool time |
| **Context Compaction** | | |
| 1000 messages | <100ms | Time to compact large context |

## Interpreting Results

### Locust Reports

Locust generates HTML reports with:

1. **Statistics Tab**
   - Total requests
   - Failure rate
   - Response times (min, max, avg, median)
   - Requests per second
   - Response time percentiles

2. **Charts Tab**
   - Response time over time
   - Requests per second over time
   - Failure rate over time

3. **Exceptions Tab**
   - Error types
   - Error counts
   - Error traces

### Key Metrics to Monitor

#### 1. Response Time
- **P50**: Median latency, should be <100ms
- **P95**: 95% of requests complete faster than this
- **P99**: Only 1% of requests should exceed this
- **Trend**: Watch for increasing trends (indicates overload)

#### 2. Throughput
- **Requests/second**: Should scale linearly with users initially
- **Plateau**: Indicates system bottleneck
- **Decline**: Indicates system overload

#### 3. Error Rate
- **Target**: <1% under normal load
- **Acceptable**: 1-5% under stress
- **Critical**: >5% indicates capacity issues

#### 4. Resource Usage
- **CPU**: Should be <70% under normal load
- **Memory**: Should be <80% under normal load
- **Growth**: Watch for continuous memory growth (leaks)

### Identifying Bottlenecks

#### High CPU Usage
- **Symptom**: CPU >80%, latency increasing
- **Cause**: Computationally expensive operations
- **Solution**: Optimize algorithms, add caching

#### High Memory Usage
- **Symptom**: Memory >80%, growth over time
- **Cause**: Memory leaks or inefficient data structures
- **Solution**: Profile memory, fix leaks, optimize context storage

#### High I/O Wait
- **Symptom**: Low CPU, high latency
- **Cause**: Disk or network bottlenecks
- **Solution**: Add caching, use faster storage, optimize queries

#### Connection Timeouts
- **Symptom**: Increased error rate
- **Cause**: Connection pool exhaustion
- **Solution**: Increase pool size, add connection pooling

## Configuration

### Environment Variables

```bash
# API server endpoint
export VICTOR_API_HOST="http://localhost:8000"

# Default provider/model for tests
export VICTOR_PROVIDER="anthropic"
export VICTOR_MODEL="claude-sonnet-4-5"

# Performance tuning
export VICTOR_MAX_CONCURRENT_REQUESTS=150
export VICTOR_CONNECTION_POOL_SIZE=100
export VICTOR_CACHE_SIZE=1000
```

### Locust Configuration

Create `locust.conf`:
```ini
[locust]
headless = false
users = 100
spawn-rate = 10
run-time = 5m
host = http://localhost:8000
```

### Test Customization

Edit `load_test_framework.py` to customize:

```python
class VictorLoadTest(HttpUser):
    # Adjust wait time between requests
    wait_time = between(1, 3)  # 1-3 seconds

    # Add new test scenarios
    @task(2)  # Weight: 2 out of total weight
    def custom_scenario(self):
        payload = {...}
        self.client.post("/api/v1/chat", json=payload)
```

## Troubleshooting

### "Connection refused" Error

**Problem**: Can't connect to API server

**Solution**:
```bash
# Make sure API server is running
victor serve

# Check if port 8000 is available
lsof -i :8000

# Update host if different
export VICTOR_API_HOST="http://your-server:port"
```

### High Memory Usage

**Problem**: Tests consume too much memory

**Solution**:
```bash
# Reduce concurrency
locust -f tests/load_test/load_test_framework.py --users=50

# Run tests sequentially
pytest tests/load_test/test_scalability.py -k "test_10_concurrent" -v
```

### Slow Test Execution

**Problem**: Tests taking too long

**Solution**:
```bash
# Skip slow tests
pytest tests/load_test/ -m "not slow"

# Run specific test only
pytest tests/load_test/test_scalability.py::TestConcurrentRequests::test_10_concurrent_requests -v

# Reduce test duration
# Edit test_scalability.py and reduce num_requests
```

### Import Errors

**Problem**: Module not found errors

**Solution**:
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Verify installation
pip list | grep locust
pip list | grep pytest

# Reinstall if needed
pip install --upgrade locust pytest pytest-asyncio
```

## Best Practices

### Development

1. **Start small**: Begin with 10 concurrent requests
2. **Increase gradually**: Double load each iteration
3. **Monitor continuously**: Watch metrics during tests
4. **Stop on failure**: Don't ignore errors

### Pre-Production

1. **Run full suite**: Execute all scalability tests
2. **Test at scale**: Simulate expected peak load
3. **Long-running tests**: Run endurance tests
4. **Document results**: Save reports for comparison

### Production Monitoring

1. **Set baselines**: Record normal performance metrics
2. **Alert on degradation**: Configure alerting rules
3. **Regular testing**: Run load tests weekly/monthly
4. **Track trends**: Monitor performance over time

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Load Tests

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Start API server
        run: |
          victor serve &
          sleep 10

      - name: Run load tests
        run: |
          make load-test-headless

      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: load-test-reports
          path: load-test-report.html
```

## Resources

- **Main Documentation**: `/Users/vijaysingh/code/codingagent/docs/SCALABILITY_REPORT.md`
- **Locust Documentation**: https://locust.io/
- **pytest Documentation**: https://docs.pytest.org/
- **Performance Tuning**: See CLAUDE.md section on performance optimization

## Support

For issues or questions:
- GitHub Issues: https://github.com/vijayksingh/victor/issues
- Email: singhvjd@gmail.com
