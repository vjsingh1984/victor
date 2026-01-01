# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow Service Lifecycle Management.

Manages external service dependencies for workflows: databases (PostgreSQL, MySQL),
caches (Redis, Memcached), message queues (RabbitMQ, Kafka), and external APIs.

Provides service initialization, health checks, and cleanup hooks.

Example:
    from victor.workflows.services import (
        ServiceManager,
        ServiceConfig,
        PostgresService,
        RedisService,
    )

    # Configure services
    services = [
        ServiceConfig(
            name="db",
            service_type="postgres",
            config={
                "host": "localhost",
                "port": 5432,
                "database": "workflows",
            },
        ),
        ServiceConfig(
            name="cache",
            service_type="redis",
            config={"host": "localhost", "port": 6379},
        ),
    ]

    # Initialize services
    manager = ServiceManager()
    handles = await manager.initialize_services(services)

    # Use in workflow
    db = handles["db"]
    cache = handles["cache"]

    # Cleanup
    await manager.cleanup_services()
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a workflow service.

    Attributes:
        name: Service identifier (used to access in workflow)
        service_type: Type of service (postgres, redis, rabbitmq, etc.)
        config: Service-specific configuration
        required: Whether workflow fails if service unavailable
        init_timeout: Timeout for service initialization (seconds)
        cleanup_timeout: Timeout for service cleanup (seconds)
    """

    name: str
    service_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    init_timeout: float = 30.0
    cleanup_timeout: float = 10.0


class ServiceLifecycle(ABC):
    """Abstract base for service lifecycle management."""

    @abstractmethod
    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize the service, return service handle/client."""
        pass

    @abstractmethod
    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Cleanup/shutdown the service."""
        pass

    @abstractmethod
    async def health_check(self, handle: Any) -> bool:
        """Check if service is healthy."""
        pass


class PostgresService(ServiceLifecycle):
    """Lifecycle manager for PostgreSQL connections."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize PostgreSQL connection pool."""
        logger.info(f"Initializing PostgreSQL service: {config.name}")

        # Use asyncpg if available
        try:
            import asyncpg

            pool = await asyncpg.create_pool(
                host=config.config.get("host", "localhost"),
                port=config.config.get("port", 5432),
                database=config.config.get("database", "postgres"),
                user=config.config.get("user", "postgres"),
                password=config.config.get("password", ""),
                min_size=config.config.get("min_connections", 1),
                max_size=config.config.get("max_connections", 10),
            )
            logger.info(f"PostgreSQL pool created for {config.name}")
            return {"pool": pool, "type": "asyncpg"}
        except ImportError:
            logger.warning("asyncpg not installed, using placeholder")
            return {
                "connection": f"postgres://{config.config.get('host', 'localhost')}",
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close PostgreSQL connections."""
        logger.info(f"Cleaning up PostgreSQL service: {config.name}")
        if handle.get("type") == "asyncpg" and handle.get("pool"):
            await handle["pool"].close()

    async def health_check(self, handle: Any) -> bool:
        """Check PostgreSQL connectivity."""
        if handle.get("type") == "asyncpg" and handle.get("pool"):
            try:
                async with handle["pool"].acquire() as conn:
                    await conn.execute("SELECT 1")
                return True
            except Exception:
                return False
        return True


class RedisService(ServiceLifecycle):
    """Lifecycle manager for Redis connections."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize Redis client."""
        logger.info(f"Initializing Redis service: {config.name}")

        try:
            import redis.asyncio as aioredis

            client = aioredis.Redis(
                host=config.config.get("host", "localhost"),
                port=config.config.get("port", 6379),
                db=config.config.get("db", 0),
                password=config.config.get("password"),
                decode_responses=config.config.get("decode_responses", True),
            )
            logger.info(f"Redis client created for {config.name}")
            return {"client": client, "type": "aioredis"}
        except ImportError:
            logger.warning("redis not installed, using placeholder")
            return {
                "client": f"redis://{config.config.get('host', 'localhost')}",
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close Redis connection."""
        logger.info(f"Cleaning up Redis service: {config.name}")
        if handle.get("type") == "aioredis" and handle.get("client"):
            await handle["client"].close()

    async def health_check(self, handle: Any) -> bool:
        """Check Redis connectivity."""
        if handle.get("type") == "aioredis" and handle.get("client"):
            try:
                await handle["client"].ping()
                return True
            except Exception:
                return False
        return True


class RabbitMQService(ServiceLifecycle):
    """Lifecycle manager for RabbitMQ connections."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize RabbitMQ connection."""
        logger.info(f"Initializing RabbitMQ service: {config.name}")

        try:
            import aio_pika

            connection = await aio_pika.connect_robust(
                host=config.config.get("host", "localhost"),
                port=config.config.get("port", 5672),
                login=config.config.get("user", "guest"),
                password=config.config.get("password", "guest"),
                virtualhost=config.config.get("vhost", "/"),
            )
            channel = await connection.channel()
            logger.info(f"RabbitMQ connection created for {config.name}")
            return {"connection": connection, "channel": channel, "type": "aio_pika"}
        except ImportError:
            logger.warning("aio_pika not installed, using placeholder")
            return {
                "connection": f"amqp://{config.config.get('host', 'localhost')}",
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close RabbitMQ connection."""
        logger.info(f"Cleaning up RabbitMQ service: {config.name}")
        if handle.get("type") == "aio_pika":
            if handle.get("channel"):
                await handle["channel"].close()
            if handle.get("connection"):
                await handle["connection"].close()

    async def health_check(self, handle: Any) -> bool:
        """Check RabbitMQ connectivity."""
        if handle.get("type") == "aio_pika" and handle.get("connection"):
            return not handle["connection"].is_closed
        return True


class KafkaService(ServiceLifecycle):
    """Lifecycle manager for Kafka connections."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize Kafka producer/consumer."""
        logger.info(f"Initializing Kafka service: {config.name}")

        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

            bootstrap_servers = config.config.get("bootstrap_servers", "localhost:9092")

            producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
            await producer.start()

            logger.info(f"Kafka producer created for {config.name}")
            return {"producer": producer, "type": "aiokafka"}
        except ImportError:
            logger.warning("aiokafka not installed, using placeholder")
            return {
                "producer": f"kafka://{config.config.get('bootstrap_servers', 'localhost:9092')}",
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close Kafka connection."""
        logger.info(f"Cleaning up Kafka service: {config.name}")
        if handle.get("type") == "aiokafka" and handle.get("producer"):
            await handle["producer"].stop()

    async def health_check(self, handle: Any) -> bool:
        """Check Kafka connectivity."""
        # Producer health check is implicit - it reconnects automatically
        return True


class HTTPClientService(ServiceLifecycle):
    """Lifecycle manager for HTTP API clients."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize HTTP client session."""
        logger.info(f"Initializing HTTP client service: {config.name}")

        try:
            import httpx

            client = httpx.AsyncClient(
                base_url=config.config.get("base_url", "http://localhost:8080"),
                timeout=config.config.get("timeout", 30.0),
                headers=config.config.get("headers", {}),
            )
            logger.info(f"HTTP client created for {config.name}")
            return {"client": client, "type": "httpx"}
        except ImportError:
            logger.warning("httpx not installed, using placeholder")
            return {
                "base_url": config.config.get("base_url", "http://localhost:8080"),
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close HTTP client session."""
        logger.info(f"Cleaning up HTTP client service: {config.name}")
        if handle.get("type") == "httpx" and handle.get("client"):
            await handle["client"].aclose()

    async def health_check(self, handle: Any) -> bool:
        """Check HTTP client health (optional health endpoint)."""
        return True


class S3Service(ServiceLifecycle):
    """Lifecycle manager for AWS S3 client."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize S3 client."""
        logger.info(f"Initializing S3 service: {config.name}")

        try:
            import aioboto3

            session = aioboto3.Session(
                aws_access_key_id=config.config.get("aws_access_key_id"),
                aws_secret_access_key=config.config.get("aws_secret_access_key"),
                region_name=config.config.get("region_name", "us-east-1"),
            )
            logger.info(f"S3 session created for {config.name}")
            return {"session": session, "type": "aioboto3"}
        except ImportError:
            logger.warning("aioboto3 not installed, using placeholder")
            return {
                "bucket": config.config.get("bucket"),
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """No cleanup needed for S3 (stateless)."""
        logger.info(f"Cleaning up S3 service: {config.name}")

    async def health_check(self, handle: Any) -> bool:
        """S3 is stateless, always healthy."""
        return True


class ServiceManager:
    """Manages service lifecycle for workflow execution.

    Handles initialization, health checks, and cleanup of services
    required by workflows.
    """

    # Registry of lifecycle handlers by service type
    _lifecycle_handlers: Dict[str, Type[ServiceLifecycle]] = {
        "postgres": PostgresService,
        "postgresql": PostgresService,
        "redis": RedisService,
        "rabbitmq": RabbitMQService,
        "kafka": KafkaService,
        "http": HTTPClientService,
        "api": HTTPClientService,
        "s3": S3Service,
    }

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._configs: Dict[str, ServiceConfig] = {}
        self._handlers: Dict[str, ServiceLifecycle] = {}

    @classmethod
    def register_service(
        cls,
        service_type: str,
        handler: Type[ServiceLifecycle],
    ) -> None:
        """Register a lifecycle handler for a service type.

        Args:
            service_type: Service type identifier (e.g., "postgres")
            handler: ServiceLifecycle class to handle this type
        """
        cls._lifecycle_handlers[service_type] = handler

    async def initialize_services(
        self,
        configs: List[ServiceConfig],
    ) -> Dict[str, Any]:
        """Initialize all configured services.

        Args:
            configs: List of service configurations

        Returns:
            Dictionary of service name -> handle

        Raises:
            RuntimeError: If a required service fails to initialize
        """
        for config in configs:
            handler_class = self._lifecycle_handlers.get(config.service_type)
            if not handler_class:
                logger.warning(f"No lifecycle handler for service type: {config.service_type}")
                continue

            handler = handler_class()
            self._handlers[config.name] = handler
            self._configs[config.name] = config

            try:
                handle = await asyncio.wait_for(
                    handler.initialize(config),
                    timeout=config.init_timeout,
                )
                self._services[config.name] = handle
                logger.info(f"Service '{config.name}' initialized successfully")

            except asyncio.TimeoutError:
                msg = f"Service '{config.name}' initialization timed out"
                logger.error(msg)
                if config.required:
                    raise RuntimeError(msg)

            except Exception as e:
                msg = f"Service '{config.name}' initialization failed: {e}"
                logger.error(msg, exc_info=True)
                if config.required:
                    raise RuntimeError(msg)

        return self._services

    async def cleanup_services(self) -> None:
        """Cleanup all initialized services."""
        for name in list(self._services.keys()):
            handle = self._services.get(name)
            handler = self._handlers.get(name)
            config = self._configs.get(name)

            if handler and config and handle:
                try:
                    await asyncio.wait_for(
                        handler.cleanup(handle, config),
                        timeout=config.cleanup_timeout,
                    )
                    logger.info(f"Service '{name}' cleaned up successfully")

                except asyncio.TimeoutError:
                    logger.warning(f"Service '{name}' cleanup timed out")

                except Exception as e:
                    logger.warning(f"Service '{name}' cleanup failed: {e}")

        self._services.clear()
        self._handlers.clear()
        self._configs.clear()

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services.

        Returns:
            Dictionary of service name -> health status
        """
        results = {}
        for name, handle in self._services.items():
            handler = self._handlers.get(name)
            if handler:
                try:
                    results[name] = await handler.health_check(handle)
                except Exception:
                    results[name] = False
            else:
                results[name] = True
        return results

    def get_service(self, name: str) -> Optional[Any]:
        """Get a service handle by name.

        Args:
            name: Service name

        Returns:
            Service handle or None if not found
        """
        return self._services.get(name)


__all__ = [
    # Config
    "ServiceConfig",
    # Lifecycle base
    "ServiceLifecycle",
    # Built-in services
    "PostgresService",
    "RedisService",
    "RabbitMQService",
    "KafkaService",
    "HTTPClientService",
    "S3Service",
    # Manager
    "ServiceManager",
]
