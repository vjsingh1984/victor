# Task R2: Architecture Design

## Objective
Design a system architecture for a given set of requirements with trade-off analysis.

## Requirements
Design an architecture for a "Real-Time Chat Application" with:
1. **Backend**: WebSocket server handling 10K concurrent connections
2. **Database**: PostgreSQL for persistence, Redis for caching
3. **Message Broker**: RabbitMQ for async processing
4. **Scaling**: Horizontal scaling with load balancer
5. **Monitoring**: Metrics collection and alerting

The design should include:
- System diagram description
- Component breakdown
- Data flow explanation
- Technology choices with justification
- Failure mode handling
- Scaling strategy

## Input Prompt
```
Design a microservices architecture for a real-time chat application that:
- Supports 10,000 concurrent WebSocket connections
- Persists messages to PostgreSQL
- Caches user sessions in Redis
- Uses RabbitMQ for async notifications
- Can scale horizontally behind a load balancer
- Collects metrics for monitoring

Explain your component choices, data flow, and how you handle failures.
```

## Success Criteria
1. Complete architecture described (all components)
2. Technology choices justified
3. Data flow explained clearly
4. Failure modes addressed
5. Scaling strategy documented
6. Trade-offs discussed (why not alternatives)

## Scoring Rubric
- **5 points**: Comprehensive design, all components justified, trade-offs discussed, realistic
- **4 points**: Complete design with good justification, minor gaps
- **3 points**: Basic architecture with some justification
- **2 points**: High-level overview only, missing key components
- **1 point**: Attempted but incomplete or impractical
- **0 points**: No output or completely unrelated

## Test Environment
- **Allowed tools**: None (reasoning-only task)
- **Timeout**: 120 seconds
- **LLM temperature**: 0.5
- **Max tokens**: 2000

## Validation Steps
1. Check for all required components
2. Verify technology choices are relevant
3. Confirm failure modes are addressed
4. Assess if scaling strategy is sound
5. Human evaluation of architecture quality

## Evaluation Dimensions
| Dimension | Weight | Criteria |
|-----------|--------|----------|
| Completeness | 40% | All required components addressed |
| Technical Soundness | 30% | Choices are practical and appropriate |
| Clarity | 20% | Design is understandable and well-explained |
| Trade-off Analysis | 10% | Alternatives considered and justified |

## Example Passing Response
A passing response would include:
- System diagram (text description is fine)
- Component list: WebSocket server, API gateway, PostgreSQL, Redis, RabbitMQ, etc.
- Data flow: Client → LB → WebSocket Server → Redis/PostgreSQL → RabbitMQ → Worker
- Technology justifications: Why PostgreSQL vs MongoDB, Why Redis vs Memcached
- Failure handling: What happens if Redis crashes, if RabbitMQ fills up
- Scaling: How to add more WebSocket servers, database sharding strategy
- Trade-offs: Why microservices vs monolith, CAP theorem implications

## Notes
- This tests reasoning and planning capabilities
- No single "correct" answer - quality of reasoning matters
- Framework's ability to handle complex, multi-faceted requirements is tested
