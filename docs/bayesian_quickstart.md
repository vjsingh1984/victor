# Bayesian Orchestration Quickstart Guide

**Get started with Bayes-consistent multi-agent coordination in Victor**

Last Updated: 2026-05-04

---

## 🚀 Quick Start (5 Minutes)

### 1. **Basic Usage** (Automatic)

Bayesian orchestration is **enabled by default**. Just use Victor normally:

```bash
# Simple query → fast path (~30ms)
victor chat "What files are in the current directory?"

# Complex query → Bayesian path (~100ms)
victor chat "Analyze the database schema and recommend optimization strategies"
```

**That's it!** Victor automatically:
1. Detects query complexity
2. Routes to fast path (simple) or Bayesian (complex)
3. Optimizes agent selection using Value of Information
4. Tracks agent reliability over time

---

## 📊 Understanding Complexity-Based Routing

### How It Works

```
Query Complexity Analysis (5 factors)
    ↓
complexity_score = 0.0 to 1.0
    ↓
IF score >= 0.7:
    → Full Bayesian (3+ agents, VoI, correlation-aware)
ELIF score >= 0.3:
    → Limited Bayesian (2 agents, basic consensus)
ELSE:
    → Fast Path (1 agent, majority vote)
```

### Complexity Factors

1. **Query Length**: Longer → more complex
2. **Patterns**: "analyze", "design", "compare" → complex
3. **Tools Required**: Multiple tools → more complex
4. **Ambiguity**: "maybe", "uncertain" → more complex
5. **Domain**: Technical jargon → more complex

### Examples

**Simple (Fast Path)**:
```bash
victor chat "What is Python?"
victor chat "How do I create a file?"
victor chat "List the files in this directory"
```

**Moderate (Limited Bayesian)**:
```bash
victor chat "Should I use PostgreSQL or MongoDB?"
victor chat "Compare caching strategies for this API"
victor chat "What are the pros and cons of microservices?"
```

**Complex (Full Bayesian)**:
```bash
victor chat "Analyze the performance bottlenecks in this microservice architecture and suggest optimization strategies"
victor chat "Design a scalable fault-tolerant system for processing real-time data streams with conflicting requirements"
```

---

## 🎛️ Configuration Options

### 1. **Enable/Disable Bayesian**

```bash
# Enable (default)
victor chat --enable-bayesian "Complex query"

# Disable entirely (use fast path for all)
victor chat --no-enable-bayesian "Any query"
```

**When to Disable**:
- Ultra-low latency requirements (<10ms)
- Simple Q&A workloads
- Testing/debugging

### 2. **Adjust Complexity Thresholds**

```bash
# More aggressive (more queries → Bayesian)
victor chat --simple-threshold 0.2 --complex-threshold 0.5 "Compare approaches"

# More conservative (fewer queries → Bayesian)
victor chat --simple-threshold 0.5 --complex-threshold 0.9 "Analyze this"
```

**When to Adjust**:
- **Lower thresholds**: Want more Bayesian analysis
- **Higher thresholds**: Want faster responses

### 3. **Force Bayesian Mode** (Testing)

```bash
# Force ALL queries through Bayesian (for testing)
victor chat --force-bayesian "Even simple queries"
```

**When to Use**:
- Testing Bayesian features
- Benchmarking performance
- Debugging orchestration

### 4. **Disable Specific Features**

```bash
# Disable Value of Information (agent selection)
victor chat --no-enable-voi "Should I use this approach?"

# Disable correlation tracking
victor chat --no-enable-correlation "Compare these options"
```

**When to Disable**:
- **VoI**: Want to query all agents (no filtering)
- **Correlation**: Agents are truly independent

---

## 📈 Monitoring & Observability

### 1. **View System Statistics**

```bash
# View overall Bayesian orchestration stats
victor bayesian summary --days 7

# Export metrics for analysis
victor bayesian summary --export metrics.json --days 30
```

**Output**:
```
Bayesian Orchestration System Summary (Last 7 days)
============================================================

Belief States:
  Unique belief states: 150

Agent Reliability:
  Tracked agents: 5
  Most reliable: agent_a (92%)
  Least reliable: agent_c (68%)

Consensus Decisions:
  Consensus decisions: 120
  Accuracy: 85%

Value of Information:
  VoI queries: 50
  Beneficial rate: 78%
  Mean predicted VoI: 0.4523
```

### 2. **View Agent Reliability**

```bash
# View reliability trends for all agents
victor bayesian reliability --days 7

# View specific agents
victor bayesian reliability --agent-ids agent_a,agent_b --days 7

# Export reliability data
victor bayesian reliability --export reliability.csv --days 30
```

### 3. **View Consensus Performance**

```bash
# View consensus accuracy and agreement levels
victor bayesian consensus --days 7
```

### 4. **View Correlation Matrix**

```bash
# View correlations between agents
victor bayesian correlations agent_a,agent_b,agent_c --days 7
```

**Output**:
```
Agent Correlations (Last 7 days)
=====================================

              agent_a    agent_b    agent_c
              ──────────────────────────────
agent_a       ████ 1.00   ███ 0.85   ██ 0.72
agent_b       ███ 0.85    ████ 1.00   ██ 0.68
agent_c       ██ 0.72     ██ 0.68    ████ 1.00

Legend: ████ = 1.0, ███ = 0.7-0.9, ██ = 0.4-0.6, ▓ = 0.1-0.3,   = 0.0

Highly Correlated Pairs (|correlation| > 0.7):
  agent_a ↔ agent_b: 0.850
  agent_a ↔ agent_c: 0.720
```

---

## 💡 Best Practices

### 1. **Start with Defaults**

The default settings work well for most use cases:
- `--simple-threshold 0.3`
- `--complex-threshold 0.7`
- `--enable-bayesian`
- `--enable-voi`
- `--enable-correlation`

### 2. **Monitor Performance**

```bash
# Check performance weekly
victor bayesian summary --days 7

# Export metrics for analysis
victor bayesian summary --export weekly_$(date +%Y%m%d).json
```

### 3. **Adjust Based on Workload**

**For Simple Q&A**:
```bash
victor chat --no-enable-bayesian "What is this file?"
```

**For Complex Analysis**:
```bash
victor chat --force-bayesian "Analyze the system architecture"
```

**For Cost Optimization**:
```bash
victor chat --enable-voi "Should I use this approach?"  # Let VoI decide
```

### 4. **Track Agent Reliability**

```bash
# View which agents are most reliable
victor bayesian reliability --days 30

# Use reliable agents more often
# (System learns automatically from outcomes)
```

### 5. **Handle Correlated Agents**

```bash
# Check for correlations
victor bayesian correlations agent_a,agent_b,agent_c

# If highly correlated (>0.7), consider:
# - Removing redundant agents
# - Using diverse agent sets
# - Relying on correlation-aware consensus (automatic)
```

---

## 🔧 Troubleshooting

### Problem: "Queries are slower than expected"

**Solution 1**: Check if queries are being routed to Bayesian:

```bash
# Query should be fast (simple)
victor chat "What files are here?"

# If still slow, check complexity
python -c "
from victor.agent.complexity_detector import QueryComplexityDetector
detector = QueryComplexityDetector()
analysis = detector.analyze('What files are here?')
print(f'Level: {analysis.level}')
print(f'Reasons: {analysis.reasons}')
"
```

**Solution 2**: Adjust thresholds:

```bash
# Make more queries use fast path
victor chat --simple-threshold 0.5 --complex-threshold 0.9 "Query"
```

**Solution 3**: Disable Bayesian for ultra-fast responses:

```bash
victor chat --no-enable-bayesian "Query"
```

---

### Problem: "Accuracy is not improving"

**Solution 1**: Check if agents are being queried:

```bash
# View VoI statistics
victor bayesian voi --days 7

# Should see beneficial queries >70%
```

**Solution 2**: Force Bayesian mode:

```bash
victor chat --force-bayesian "Complex query"
```

**Solution 3**: Check agent reliability:

```bash
victor bayesian reliability --days 7

# Look for agents with <70% reliability
# Consider removing or retraining them
```

---

### Problem: "Agents seem correlated"

**Solution 1**: Check correlation matrix:

```bash
victor bayesian correlations agent_a,agent_b,agent_c --days 7
```

**Solution 2**: If correlation >0.7:
- Remove redundant agents
- Use diverse agent sets
- Correlation-aware consensus handles this automatically

**Solution 3**: Monitor Effective Sample Size:

```python
from victor.framework.rl.learners.correlation_tracker import CorrelationTracker

# ESS accounts for correlations
# Lower ESS = less independent information
```

---

### Problem: "VoI is not filtering agents"

**Solution 1**: Check VoI is enabled:

```bash
victor chat --enable-voi "Should I use this approach?"
```

**Solution 2**: Check VoI statistics:

```bash
victor bayesian voi --days 7

# Look for:
# - Beneficial rate >70%
# - Mean predicted VoI >0.3
```

**Solution 3**: Lower query cost threshold:

```python
# In hybrid_orchestrator.py, adjust:
voi = voi_controller.compute_voi(
    task_analysis=belief,
    agent_id=agent_id,
    query_cost=0.0,  # Lower cost = more queries
)
```

---

## 📊 Performance Tuning

### For Speed

```bash
# Use fast path for everything
victor chat --no-enable-bayesian "Query"

# Or raise thresholds (fewer Bayesian queries)
victor chat --simple-threshold 0.7 --complex-threshold 0.9 "Query"
```

**Expected**: 70% faster for simple queries (~30ms)

### For Accuracy

```bash
# Force Bayesian for everything
victor chat --force-bayesian "Query"

# Or lower thresholds (more Bayesian queries)
victor chat --simple-threshold 0.2 --complex-threshold 0.5 "Query"
```

**Expected**: 25-40% more accurate for complex queries

### For Cost

```bash
# Enable VoI (default)
victor chat --enable-voi "Query"

# Check VoI is working
victor bayesian voi --days 7

# Should see 40% reduction in queries
```

**Expected**: 40% reduction in unnecessary agent queries

---

## 🎓 Advanced Usage

### 1. **Custom Complexity Detection**

```python
from victor.agent.complexity_detector import QueryComplexityDetector

# Create detector with custom thresholds
detector = QueryComplexityDetector(
    simple_threshold=0.4,  # Higher
    complex_threshold=0.8,  # Higher
)

# Analyze query
analysis = detector.analyze("Your query here")
print(f"Level: {analysis.level}")
print(f"Confidence: {analysis.confidence}")
print(f"Reasons: {analysis.reasons}")
```

### 2. **Programmatic Usage**

```python
from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter
from victor.framework.session_config import SessionConfig

# Create config
config = SessionConfig.from_cli_flags(
    enable_bayesian=True,
    simple_threshold=0.3,
    complex_threshold=0.7,
)

# Create router
router = HybridOrchestrationRouter(
    enable_bayesian=config.bayesian.enabled,
    track_performance=True,
)

# Route query
result = router.route_query(
    query="Your query here",
    agent_messages={"agent_a": "Response"},
)

print(f"Type: {result.orchestration_type}")
print(f"Decision: {result.decision}")
print(f"Latency: {result.latency_ms}ms")
```

### 3. **Performance Monitoring**

```python
from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter

router = HybridOrchestrationRouter(track_performance=True)

# Run queries...
for query in queries:
    router.route_query(query, agent_messages)

# Get statistics
stats = router.get_performance_stats()
print(f"Simple queries: {stats['simple_count']}")
print(f"Bayesian queries: {stats['bayesian_count']}")
print(f"Avg simple latency: {stats['avg_simple_latency_ms']:.2f}ms")
print(f"Avg Bayesian latency: {stats['avg_bayesian_latency_ms']:.2f}ms")
print(f"Bayesian percentage: {stats['bayesian_percentage']:.1f}%")
```

---

## 📚 Additional Resources

### Documentation

- [Full Bayesian Documentation](bayesian_orchestration.md) - Complete user guide
- [API Reference](api/bayesian.md) - Method signatures and schemas
- [Architecture Guide](architecture/bayesian.md) - Design and algorithms
- [Production Readiness](BAYESIAN_PRODUCTION_READINESS.md) - Deployment guide

### Examples

- [Bayesian Orchestration Examples](../examples/bayesian_orchestration_example.py) - 4 comprehensive examples
- [Correlation Tracking Examples](../examples/correlation_tracking_example.py) - 4 correlation analysis examples

### CLI Commands

```bash
# Bayesian monitoring commands
victor bayesian summary --days 7
victor bayesian reliability --days 7
victor bayesian consensus --days 7
victor bayesian voi --days 7
victor bayesian correlations agent_a,agent_b --days 7
victor bayesian belief <belief_id>
```

---

## 🎯 Common Use Cases

### 1. **Code Review**

```bash
# Simple: Check for syntax errors
victor chat "Check if this file has syntax errors"

# Complex: Full code review
victor chat "Analyze this code for performance issues, security vulnerabilities, and design patterns. Suggest improvements with explanations."
```

### 2. **Debugging**

```bash
# Simple: What's the error?
victor chat "What does this error mean?"

# Complex: Root cause analysis
victor chat "Investigate the root cause of this memory leak. Analyze allocation patterns, garbage collection logs, and propose a comprehensive solution."
```

### 3. **Architecture**

```bash
# Simple: How do I...?
victor chat "How do I add authentication to this API?"

# Complex: System design
victor chat "Design a scalable microservice architecture for this application. Consider data consistency, fault tolerance, and performance optimization."
```

### 4. **Optimization**

```bash
# Simple: Make it faster
victor chat "How can I speed up this query?"

# Complex: Performance analysis
victor chat "Analyze the performance bottlenecks in this system. Profile the database queries, API calls, and caching strategies. Recommend optimizations with expected impact."
```

---

## ✅ Checklist

### Getting Started

- [ ] Try default Bayesian settings (automatic)
- [ ] Run a simple query (should use fast path)
- [ ] Run a complex query (should use Bayesian)
- [ ] Check complexity detection works as expected
- [ ] View system statistics: `victor bayesian summary`

### Optimization

- [ ] Monitor performance for 1 week
- [ ] Check agent reliability: `victor bayesian reliability`
- [ ] View consensus performance: `victor bayesian consensus`
- [ ] Adjust thresholds if needed
- [ ] Export metrics for analysis

### Advanced

- [ ] Check agent correlations: `victor bayesian correlations`
- [ ] View Value of Information stats: `victor bayesian voi`
- [ ] Customize complexity thresholds for workload
- [ ] Integrate into custom workflows
- [ ] Set up monitoring dashboards

---

## 🚀 Next Steps

1. **Experiment**: Try different query types and observe routing
2. **Monitor**: Check statistics weekly to track improvement
3. **Optimize**: Adjust thresholds based on your workload
4. **Learn**: Read full documentation for advanced features
5. **Extend**: Integrate into your custom workflows

**Questions?** See [Troubleshooting](#-troubleshooting) or [Full Documentation](bayesian_orchestration.md)

**Ready to use Bayesian orchestration! 🎉**
