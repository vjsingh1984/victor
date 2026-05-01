# Provider-Specific Tier Optimization — Rollout Plan

**Status**: Ready for Rollout (2026-04-24)
**Feature**: Provider-specific tool tier assignments
**Impact**: 80% token reduction for edge models, 40% for standard models
**Risk**: Low (feature flag enables instant rollback)

## Pre-Rollout Checklist

### Code Readiness

- ✅ **Phase 1**: Configuration structure (tool_tiers.yaml)
- ✅ **Phase 2**: Tier loading logic (tool_tiers.py)
- ✅ **Phase 3**: Orchestrator integration (6 methods updated)
- ✅ **Phase 4**: Testing (39 unit tests, 654 total tests passing)
- ✅ **Phase 5**: Validation (validate_provider_tiers.py script passing)

### Documentation

- ✅ **Architecture docs**: `tool-usage-data-sources.md` updated
- ✅ **Integration testing**: `provider-tier-integration-testing.md` created
- ✅ **Metrics verification**: `provider-tier-metrics-verification.md` created
- ✅ **Implementation complete**: `provider-specific-tier-optimization-complete.md` created

### Validation

- ✅ **Unit tests**: All 39 provider-specific tier tests passing
- ✅ **Integration tests**: All 654 tests passing (no regressions)
- ✅ **Validation script**: All checks passing (edge/standard/large)
- ✅ **Backward compatibility**: Feature flag enables instant rollback

## Rollout Strategy

### Week 1: Internal Testing (2026-04-24 to 2026-04-30)

**Goal**: Validate with internal team using edge models

**Participants**: 5-10 internal developers

**Configuration**:
```bash
# Enable for internal users only
export VICTOR_TOOL_STRATEGY_V2=true
```

**Test Plan**:
1. Edge models (qwen3.5:2b, gemma:2b)
   - Verify 2 FULL tools (read, shell)
   - Verify 80% token reduction
   - Verify no budget warnings

2. Standard models (qwen2.5:7b, llama:7b)
   - Verify 7 tools (5 FULL + 2 COMPACT)
   - Verify 40% token reduction
   - Verify balanced performance

3. Large models (claude-sonnet-4, gpt-4o)
   - Verify 10 FULL tools
   - Verify no regression
   - Verify full capability

**Success Criteria**:
- ✅ No budget violations
- ✅ No increase in error rate
- ✅ Positive feedback from users
- ✅ Token savings >70% for edge, >35% for standard

**Metrics to Track**:
- Token usage per provider category
- Error rate by category
- User feedback scores
- Latency impact

**Rollback Trigger**:
- Error rate >5% for any category
- Budget violations >1% of requests
- Negative user feedback >50%

### Week 2: Beta Testing (2026-05-01 to 2026-05-07)

**Goal**: Expand to beta users with all provider categories

**Participants**: 50-100 beta users (10% of user base)

**Configuration**:
```bash
# Enable for beta users (10% rollout)
if user_is_beta():
    VICTOR_TOOL_STRATEGY_V2 = true
else:
    VICTOR_TOOL_STRATEGY_V2 = false
```

**Test Plan**:
1. Diverse provider usage
   - 30% edge models
   - 50% standard models
   - 20% large models

2. Real-world scenarios
   - Code generation
   - Debugging
   - File operations
   - Testing

3. Performance validation
   - Latency measurements
   - Token cost tracking
   - User satisfaction

**Success Criteria**:
- ✅ 95%+ requests without budget warnings
- ✅ Error rate <1% per category
- ✅ Average user satisfaction >4/5
- ✅ Token savings meet targets (80% edge, 40% standard)

**Metrics to Track**:
- Request distribution by category
- Tier distribution accuracy
- Token savings achieved
- User feedback themes

**Rollback Trigger**:
- Error rate >2% for any category
- User satisfaction <3/5
- Critical bugs reported

### Week 3: Gradual Rollout (2026-05-08 to 2026-05-14)

**Goal**: Expand to 50% then 100% of users

**Phase 3A: 50% Rollout** (Days 1-3)
```bash
# Enable for 50% of users
if hash(user_id) % 2 == 0:
    VICTOR_TOOL_STRATEGY_V2 = true
else:
    VICTOR_TOOL_STRATEGY_V2 = false
```

**Monitoring**:
- Error rates every hour
- Token usage every 6 hours
- User feedback daily

**Phase 3B: 100% Rollout** (Days 4-7)
```bash
# Enable for all users
VICTOR_TOOL_STRATEGY_V2 = true
```

**Monitoring**:
- Error rates every hour
- Token usage daily
- User satisfaction weekly
- Performance metrics daily

**Success Criteria**:
- ✅ Stable error rates (<1%)
- ✅ Token savings sustained (>75% edge, >35% standard)
- ✅ No performance regression
- ✅ Positive user sentiment

**Rollback Trigger**:
- Error rate >5% for any category
- Performance degradation >20%
- Critical infrastructure issues

### Week 4: Production Monitoring (2026-05-15 to 2026-05-21)

**Goal**: Full production deployment with monitoring

**Configuration**:
```bash
# Feature flag enabled by default
VICTOR_TOOL_STRATEGY_V2 = true  # Default setting
```

**Monitoring Plan**:
1. **Daily Health Checks**
   - Run `daily_health_check.sh`
   - Review dashboard metrics
   - Check error logs

2. **Weekly Reports**
   - Token savings summary
   - Cost reduction analysis
   - User feedback themes
   - Performance trends

3. **Monthly Reviews**
   - ROI analysis
   - Optimization opportunities
   - Tier adjustment needs
   - Feature requests

**Success Criteria**:
- ✅ 30-day stable operation
- ✅ Token savings sustained
- ✅ User adoption >80%
- ✅ No critical issues

## Monitoring Dashboard

### Key Metrics

**Real-Time**:
- Request rate by provider category
- Error rate by provider category
- Budget violation rate
- Token usage rate

**Daily**:
- Token savings achieved
- Cost reduction
- User satisfaction score
- Average latency

**Weekly**:
- Tier distribution accuracy
- Provider category distribution
- A/B test results (if applicable)
- Feature request themes

### Alerting Rules

**Critical Alerts** (page immediately):
- Error rate >5% for any category
- Budget violations >10% of requests
- System-wide outage

**Warning Alerts** (investigate within 1 hour):
- Error rate >2% for any category
- Budget violations >5% of requests
- Performance degradation >20%

**Info Alerts** (review daily):
- Error rate >1% for any category
- Budget violations >1% of requests
- Unusual tier distribution

## Rollback Plan

### Instant Rollback

**Trigger**: Critical issues or user complaints

**Action**:
```bash
# Disable feature flag globally
export VICTOR_TOOL_STRATEGY_V2=false

# Or in config
VICTOR_TOOL_STRATEGY_V2: false
```

**Verification**:
```bash
# Confirm global tiers are used
grep "tool_tokens" ~/.victor/logs/victor.log | tail -10
# Should show 1,250 tokens (global) instead of 250/765/1250 (provider-specific)
```

### Gradual Rollback

**Trigger**: Degraded performance or increased errors

**Action**:
```bash
# Reduce rollout percentage
Week 1: 100% → 50%
Week 2: 50% → 10%
Week 3: 10% → 0% (disabled)
```

### Post-Rollout Analysis

**Investigation**:
- Root cause analysis
- User feedback review
- Metrics examination

**Fixes**:
- Bug fixes
- Tier adjustments
- Documentation updates

**Re-rollout**:
- Follow same phased approach
- Additional testing
- Increased monitoring

## Success Metrics

### Technical Metrics

**Token Savings**:
- Edge: ≥80% reduction (1250 → 250 tokens)
- Standard: ≥38% reduction (1250 → 765 tokens)
- Large: 0% regression (1250 → 1250 tokens)

**Budget Compliance**:
- Edge: ≤12.2% budget utilization (250/2048)
- Standard: ≤9.3% budget utilization (765/8192)
- Large: ≤2.5% budget utilization (1250/50000)

**Error Rate**:
- All categories: <1% error rate
- No critical bugs in production

**Performance**:
- No latency degradation
- No throughput reduction
- Stable resource usage

### Business Metrics

**Cost Reduction**:
- Edge models: 80% reduction in tool token costs
- Standard models: 40% reduction in tool token costs
- Overall: 50%+ reduction in token costs

**User Satisfaction**:
- Average satisfaction ≥4/5
- Positive feedback ≥80%
- Negative feedback ≤5%

**Adoption Rate**:
- 30-day adoption ≥80%
- Feature enabled by default
- No opt-out requests

### Operational Metrics

**Reliability**:
- 99.9% uptime
- No data loss
- No security issues

**Maintainability**:
- Clear documentation
- Automated monitoring
- Easy rollback process

## Communication Plan

### Internal Communication

**Pre-Rollout** (Week 0):
- Engineering team: Feature overview and testing plan
- Support team: Known issues and troubleshooting
- Management: Timeline and success metrics

**During Rollout** (Weeks 1-3):
- Daily status updates
- Weekly progress reports
- Ad-hoc issues and resolutions

**Post-Rollout** (Week 4+):
- Monthly success metrics
- Quarterly reviews
- Annual impact assessment

### External Communication

**Release Notes**:
- Feature announcement
- Benefits description
- Known limitations
- Opt-out instructions

**Documentation Updates**:
- Architecture docs
- Integration guides
- Troubleshooting guides

**User Feedback**:
- In-app prompts
- Feedback forms
- Community discussions

## Risk Mitigation

### Technical Risks

**Risk 1**: Edge model performance degradation
- **Mitigation**: Extensive testing with edge models
- **Fallback**: Revert to global tiers for edge category

**Risk 2**: Budget violations for complex queries
- **Mitigation**: Conservative tier assignments
- **Fallback**: Runtime tool demotion already in place

**Risk 3**: Incorrect tier assignments
- **Mitigation**: Comprehensive unit tests (39 tests)
- **Fallback**: Feature flag for instant rollback

### Operational Risks

**Risk 4**: Increased support burden
- **Mitigation**: Clear documentation and troubleshooting guides
- **Fallback**: Disable feature flag if support burden >20%

**Risk 5**: User confusion
- **Mitigation**: In-app notifications and docs
- **Fallback**: Disable feature flag if confusion >30%

### Business Risks

**Risk 6**: Cost savings not realized
- **Mitigation**: Gradual rollout with measurement
- **Fallback**: Revert to global tiers if savings <50%

## Timeline Summary

| Week | Phase | Rollout % | Key Activities |
|------|-------|-----------|----------------|
| 0 | Preparation | 0% | Code review, documentation, testing |
| 1 | Internal Testing | 1% | Edge model validation, bug fixes |
| 2 | Beta Testing | 10% | Expanded testing, metrics tracking |
| 3 | Gradual Rollout | 50% → 100% | Phased rollout, monitoring |
| 4 | Production | 100% | Full deployment, optimization |
| 5+ | Maintenance | 100% | Ongoing monitoring, improvements |

## References

- **Implementation**: `victor/config/tool_tiers.py`, `victor/agent/orchestrator.py`
- **Testing**: `tests/unit/config/test_provider_tool_tiers.py`, `tests/unit/agent/test_provider_specific_tier_orchestrator.py`
- **Validation**: `python -m victor.scripts.validate_provider_tiers`
- **Integration Testing**: `docs/architecture/provider-tier-integration-testing.md`
- **Metrics Verification**: `docs/architecture/provider-tier-metrics-verification.md`
- **Complete Documentation**: `docs/architecture/provider-specific-tier-optimization-complete.md`
