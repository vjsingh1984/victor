"""Circuit breaker, retry, rate limiting, and streaming metrics."""

from __future__ import annotations

from pydantic import BaseModel


class ResilienceSettings(BaseModel):
    """Circuit breaker, retry, rate limiting, and streaming metrics."""

    resilience_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_success_threshold: int = 2
    circuit_breaker_timeout: float = 60.0
    circuit_breaker_half_open_max: int = 3
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential_base: float = 2.0
    rate_limiting_enabled: bool = True
    rate_limit_requests_per_minute: int = 50
    rate_limit_tokens_per_minute: int = 50000
    rate_limit_max_concurrent: int = 5
    rate_limit_queue_size: int = 100
    rate_limit_num_workers: int = 3
    streaming_metrics_enabled: bool = True
    streaming_metrics_history_size: int = 1000
