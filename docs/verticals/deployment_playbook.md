# Victor Architecture Deployment Playbook

**Version**: 1.0
**Date**: 2026-03-31
**Environment**: Production
**Target**: Victor API v0.6.0+

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Strategy](#deployment-strategy)
3. [Deployment Procedures](#deployment-procedures)
4. [Validation Steps](#validation-steps)
5. [Rollback Procedures](#rollback-procedures)
6. [Post-Deployment](#post-deployment)

---

## Pre-Deployment Checklist

### Environment Validation

- [ ] **Python Version**: 3.10+ installed
  ```bash
  python --version  # Should be 3.10+
  ```

- [ ] **Dependencies**: All required packages installed
  ```bash
  pip list | grep victor
  # victor-ai: 0.6.0+
  # victor-sdk: 0.6.0+
  ```

- [ ] **Feature Flags**: Configuration file deployed
  ```bash
  ls /etc/victor/feature_flags.yaml
  # Should exist and be valid YAML
  ```

- [ ] **Monitoring**: Prometheus and Grafana operational
  ```bash
  curl http://prometheus:9090/-/healthy  # Should return "Prometheus is Healthy."
  curl http://grafana:3000/api/health  # Should return "OK"
  ```

### Database Validation

- [ ] **Migration**: Database schema migrations applied
  ```bash
  python -m victor.cli migrate --dry-run
  python -m victor.cli migrate --apply
  ```

- [ ] **Backups**: Current database backed up
  ```bash
  python -m victor.cli backup create --before-deployment
  ```

### Configuration Validation

- [ ] **Config Files**: All configuration files valid
  ```bash
  python -m victor.cli config validate
  ```

- [ ] **Secrets**: Environment variables set
  ```bash
  env | grep VICTOR_
  # Should list API keys, database URLs, etc.
  ```

- [ ] **Feature Flags**: Properly configured
  ```bash
  python -m victor.cli feature-flags validate
  ```

### Testing Validation

- [ ] **Unit Tests**: All unit tests pass
  ```bash
  pytest tests/unit/ -v --tb=short
  # Expected: All pass
  ```

- [ ] **Integration Tests**: All integration tests pass
  ```bash
  pytest tests/integration/ -v --tb=short
  # Expected: All pass
  ```

- [ ] **Performance Benchmarks**: Performance targets met
  ```bash
  pytest tests/benchmarks/ -v
  # Entry point scan: < 50ms
  # Dependency resolution: < 10ms
  ```

---

## Deployment Strategy

### Blue-Green Deployment

**Strategy**: Zero-downtime deployment using blue-green switch

**Architecture**:
```
                    ┌─────────────┐
                    │   Load Balancer │
                    └──────┬────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
           ┌────▼────┐          ┌────▼────┐
           │  Blue   │          │  Green  │
           │ (v0.5.x) │          │ (v0.6.0+) │
           └─────────┘          └─────────┘
              │                     │
              │                     │
          ┌───┴────┐          ┌────┴───┐
          │  DB (Read) │  │  DB (Write)  │
          └─────────┘          └─────────┘
```

**Steps**:
1. Deploy new version to Green environment
2. Validate Green environment
3. Run smoke tests on Green
4. Switch load balancer to Green
5. Monitor for issues
6. Rollback to Blue if needed

### Canary Deployment

**Strategy**: Gradual rollout to percentage of traffic

**Steps**:
1. Deploy new version to canary servers
2. Route 10% of traffic to canary
3. Monitor metrics closely
4. Increase traffic gradually (50%, 100%)
5. Rollback if issues detected

---

## Deployment Procedures

### Step 1: Prepare Deployment Artifacts

**Create deployment package**:

```bash
# 1. Create deployment directory
mkdir -p /tmp/victor-deploy-$(date +%Y%m%d)
cd /tmp/victor-deploy-$(date +%Y%m%d)

# 2. Export current database schema
python -m victor.cli backup create --output backup.sql

# 3. Create deployment package
tar -czf victor-0.6.0.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='tests/.pytest_cache' \
  /path/to/victor

# 4. Calculate checksums
sha256sum victor-0.6.0.tar.gz > checksums.txt
```

**Verify deployment package**:

```bash
# Verify integrity
sha256sum -c checksums.txt

# List contents
tar -tzf victor-0.6.0.tar.gz | head -20
```

---

### Step 2: Deploy to Staging Environment

**Deploy to staging first**:

```bash
# 1. SSH to staging server
ssh staging@victor.example.com

# 2. Stop services
sudo systemctl stop victor-api
sudo systemctl stop victor-worker

# 3. Backup current installation
sudo cp -r /opt/victor /opt/victor.backup.$(date +%Y%m%d)

# 4. Extract new version
sudo tar -xzf victor-0.6.0.tar.gz -C /tmp

# 5. Install new version
sudo rsync -av /tmp/victor/ /opt/victor/

# 6. Install dependencies
cd /opt/victor
sudo pip install -e .[prod]

# 7. Run migrations
sudo -u victor python -m victor.cli migrate --apply

# 8. Validate installation
sudo -u victor python -m victor.cli health-check --full

# 9. Start services
sudo systemctl start victor-api
sudo systemctl start victor-worker

# 10. Verify
sudo systemctl status victor-api
sudo systemctl status victor-worker
```

---

### Step 3: Run Staging Validation

**Smoke tests**:

```bash
# 1. Health check
curl http://staging.example.com/health
# Expected: {"status": "healthy"}

# 2. Vertical listing
curl http://staging.example.com/api/v1/verticals
# Expected: List of verticals

# 3. Load vertical
curl -X POST http://staging.example.com/api/v1/verticals/coding/load
# Expected: 200 OK

# 4. Run tools
curl -X POST http://staging.example.com/api/v1/agents/coding/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
# Expected: 200 OK with response
```

**Performance tests**:

```bash
# 1. Entry point scan benchmark
python -m victor.cli benchmark entry-point-scan
# Expected: < 50ms

# 2. Dependency resolution benchmark
python -m victor.cli benchmark dependency-resolution
# Expected: < 10ms

# 3. Full system load test
python -m victor.cli load-test --concurrency=100 --duration=60
# Expected: No errors, P95 latency acceptable
```

---

### Step 4: Deploy to Production

**Blue-Green Deployment**:

```bash
# 1. Deploy to Green environment
ssh green@victor.example.com

# Stop Green services
sudo systemctl stop victor-api
sudo systemctl stop victor-worker

# Backup and deploy
sudo cp -r /opt/victor /opt/victor.backup
sudo rsync -av /tmp/victor/ /opt/victor/

# Install and migrate
cd /opt/victor
sudo pip install -e .[prod]
sudo -u victor python -m victor.cli migrate --apply

# Start Green services
sudo systemctl start victor-api
sudo systemctl start victor-worker

# Verify Green
curl http://green.example.com/health
```

**Validate Green Environment**:

```bash
# Run smoke tests on Green
python -m victor.cli smoke-test --url=http://green.example.com

# Run performance tests on Green
python -m victor.cli benchmark --url=http://green.example.com

# Check metrics
python -m victor.cli monitor --url=http://green.example.com --watch
```

**Switch Load Balancer**:

```bash
# 1. Update load balancer configuration
# (This varies by load balancer)

# For HAProxy:
# Edit /etc/haproxy/haproxy.cfg
# Change backend from blue to green
sudo systemctl reload haproxy

# For NGINX:
# Edit /etc/nginx/nginx.conf
# Change upstream from blue to green
sudo systemctl reload nginx

# 2. Verify switch
curl http://victor.example.com/health
# Should hit Green environment
```

---

### Step 5: Monitor Production

**Monitor key metrics for 1 hour**:

```bash
# Watch metrics
python -m victor.cli monitor --watch --interval=30

# Check logs
sudo journalctl -u victor-api -f
sudo journalctl -u victor-worker -f

# Monitor dashboards
open http://grafana.example.com/d/victor-vertical-architecture
```

**Critical metrics to watch**:
- Vertical loading time (should be ≤ baseline)
- Error rate (should be ≤ baseline)
- Entry point scan duration (should be < 50ms)
- Cache hit rate (should be > 80%)
- Dependency graph depth (should be ≤ 10)

---

## Validation Steps

### Automated Validation

**Run validation suite**:

```bash
python -m victor.cli validate --full
```

This runs:
- Health check
- Configuration validation
- Database migration check
- Feature flag validation
- Smoke tests
- Performance benchmarks

### Manual Validation

**Check critical verticals**:

```bash
# 1. Coding vertical
curl -X POST http://victor.example.com/api/v1/verticals/coding/load
# Expected: 200 OK

# 2. DevOps vertical
curl -X POST http://victor.example.com/api/v1/verticals/devops/load
# Expected: 200 OK

# 3. RAG vertical (if available)
curl -X POST http://victor.example.com/api/v1/verticals/rag/load
# Expected: 200 OK or 404 (if not installed)
```

**Check performance**:

```bash
# Run performance benchmarks
python -m victor.cli benchmark --compare-to-baseline
# Expected: Performance within 10% of baseline
```

---

## Rollback Procedures

### Immediate Rollback

**If critical issues detected**:

```bash
# 1. Switch load balancer back to Blue
# HAProxy:
sudo vim /etc/haproxy/haproxy.cfg
# Change backend from green to blue
sudo systemctl reload haproxy

# NGINX:
sudo vim /etc/nginx/nginx.conf
# Change upstream from green to blue
sudo systemctl reload nginx

# 2. Verify rollback
curl http://victor.example.com/health
# Should hit Blue environment

# 3. Stop Green environment
ssh green@victor.example.com
sudo systemctl stop victor-api
sudo systemctl stop victor-worker

# 4. Notify team
echo "Rolled back to Blue environment due to: [ISSUE]" | \
  mail -s "Victor Rollback" eng-team@example.com
```

### Staged Rollback

**Rollback specific features**:

```bash
# If specific features causing issues, disable them

# Disable namespace isolation
export VICTOR_NAMESPACE_ISOLATION=false

# Disable async caching
export VICTOR_ASYNC_SAFE_CACHING=false

# Disable dependency graph
export VICTOR_EXTENSION_DEPENDENCY_GRAPH=false

# Restart services
sudo systemctl restart victor-api
sudo systemctl restart victor-worker
```

---

## Post-Deployment

### Monitoring and Observation

**Monitor for 24 hours**:

```bash
# Watch metrics dashboard
open http://grafana.example.com/d/victor-vertical-architecture

# Watch logs
sudo journalctl -u victor-api -f | grep -E "(ERROR|WARN)"

# Collect metrics
python -m victor.cli collect-metrics --duration=24h --output=metrics.json
```

### Performance Validation

**Compare to baseline**:

```bash
# Generate performance report
python -m victor.cli report performance \
  --baseline=baseline.json \
  --current=metrics.json \
  --output=performance-report.html
```

### Issue Resolution

**Document and resolve issues**:

```bash
# If issues found:
# 1. Document issue
echo "Issue: [DESCRIPTION]" >> deployment-issues.md
# 2. Create ticket
python -m victor.cli ticket create --type=bug --summary="[ISSUE]"
# 3. Fix issue
# 4. Validate fix
python -m victor.cli validate --fix=[ISSUE_ID]
```

---

## Success Criteria

### Deployment Success

- ✅ All health checks pass
- ✅ All smoke tests pass
- ✅ Performance within 10% of baseline
- ✅ Error rate ≤ baseline
- ✅ No critical alerts for 1 hour
- ✅ Monitoring operational

### Rollback Success

- ✅ Load balancer switched successfully
- ✅ Blue environment operational
- ✅ Green environment stopped cleanly
- ✅ Data integrity verified
- ✅ Services restarted

---

## Communication

### Pre-Deployment

**Notify engineering team**:

```
Subject: Victor Architecture Deployment - [DATE]

Deployment Details:
- Version: 0.6.0
- Environment: Production
- Start Time: [TIME]
- Estimated Duration: 1 hour
- Downtime: None (blue-green deployment)

Rollout Plan:
1. Deploy to Green (30 min)
2. Validate Green (15 min)
3. Switch traffic (5 min)
4. Monitor (10 min)

Rollback Plan:
- Immediate rollback if critical issues
- Load balancer switch to Blue
- All documented procedures

Monitoring:
- Grafana: http://grafana.example.com/d/victor
- Logs: journalctl -u victor-api -f

Contact:
- Deploy Lead: [NAME]
- On-Call: [NAME]
```

### Post-Deployment

**Announce deployment**:

```
Subject: Victor Architecture Deployment Complete ✅

Deployment Summary:
- Version: 0.6.0 deployed successfully
- Environment: Production
- Downtime: None
- Issues: None

Performance Improvements:
- Startup time: 200-500ms faster
- Entry point scan: 31x faster (16ms vs 500ms)
- Dependency resolution: 2-5x faster

Next Steps:
- Monitor for 24 hours
- Remove legacy code (week 5)
- Migrate remaining verticals

Questions? Contact: [TEAM]
```

---

## Emergency Procedures

### Critical Failure

**If system is down**:

```bash
# 1. Immediate rollback to Blue
# Load balancer
sudo vim /etc/haproxy/haproxy.cfg
# Set all backends to blue
sudo systemctl reload haproxy

# 2. Restart Blue services
ssh blue@victor.example.com
sudo systemctl restart victor-api
sudo systemctl restart victor-worker

# 3. Verify
curl http://victor.example.com/health

# 4. Notify team
echo "CRITICAL: System rolled back to Blue" | \
  mail -s "CRITICAL: Victor Rollback" eng-team@example.com \
  --cc management@example.com
```

### Data Recovery

**If database migration failed**:

```bash
# 1. Stop services
sudo systemctl stop victor-api
sudo systemctl stop victor-worker

# 2. Restore database from backup
python -m victor.cli backup restore --from=backup.sql

# 3. Rollback code
sudo rm -rf /opt/victor
sudo mv /opt/victor.backup.$(date +%Y%m%d) /opt/victor

# 4. Start services
sudo systemctl start victor-api
sudo systemctl start victor-worker

# 5. Verify
python -m victor.cli health-check --full
```

---

## Summary

The deployment playbook provides:

- ✅ **Pre-deployment checklist** for validation
- ✅ **Blue-green deployment strategy** for zero downtime
- ✅ **Step-by-step procedures** for deployment
- ✅ **Validation steps** for each stage
- ✅ **Rollback procedures** for safety
- ✅ **Post-deployment monitoring** guidance
- ✅ **Emergency procedures** for critical failures

For rollout timeline, see [Rollout Plan](rollout_plan.md).
For monitoring setup, see [Monitoring Dashboards](monitoring_dashboards.md).
For legacy removal, see [Legacy Code Deprecation](legacy_deprecation.md).
