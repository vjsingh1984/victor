"""Focused async-bridge tests for the AWS workflow service provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor.workflows.services.definition import ServiceConfig
from victor.workflows.services.providers import aws as aws_module


def _provider(session: MagicMock) -> aws_module.AWSServiceProvider:
    with patch.object(aws_module, "BOTO3_AVAILABLE", True):
        provider = aws_module.AWSServiceProvider(region="us-east-1")
    provider._session = session
    return provider


@pytest.mark.asyncio
async def test_aws_provider_start_msk_uses_to_thread() -> None:
    kafka = MagicMock()
    kafka.get_bootstrap_brokers.return_value = {
        "BootstrapBrokerString": "broker-1.example:9092,broker-2.example:9092"
    }
    session = MagicMock()
    session.client.return_value = kafka
    session.region_name = "us-east-1"
    provider = _provider(session)

    config = ServiceConfig(
        name="kafka",
        provider="aws_msk",
        aws_cluster_id="arn:aws:kafka:cluster/example",
    )

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch(
        "victor.workflows.services.providers.aws.asyncio.to_thread",
        side_effect=call_to_thread,
    ) as mock_to_thread:
        handle = await provider._start_msk(config)

    assert handle.host == "broker-1.example"
    assert handle.ports[9092] == 9092
    assert (
        handle.connection_info["KAFKA_BOOTSTRAP_SERVERS"]
        == "broker-1.example:9092,broker-2.example:9092"
    )
    mock_to_thread.assert_awaited_once()
    called = mock_to_thread.await_args
    assert called.args[0] is kafka.get_bootstrap_brokers
    assert called.kwargs == {"ClusterArn": "arn:aws:kafka:cluster/example"}


@pytest.mark.asyncio
async def test_aws_provider_start_sqs_uses_to_thread_for_create_queue() -> None:
    sqs = MagicMock()
    sqs.get_queue_url.side_effect = aws_module.ClientError(
        {
            "Error": {
                "Code": "AWS.SimpleQueueService.NonExistentQueue",
                "Message": "missing",
            }
        },
        "GetQueueUrl",
    )
    sqs.create_queue.return_value = {
        "QueueUrl": "https://sqs.us-east-1.amazonaws.com/123/test"
    }

    session = MagicMock()
    session.client.return_value = sqs
    session.region_name = "us-east-1"
    provider = _provider(session)

    config = ServiceConfig(name="test-queue", provider="aws_sqs")

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch(
        "victor.workflows.services.providers.aws.asyncio.to_thread",
        side_effect=call_to_thread,
    ) as mock_to_thread:
        handle = await provider._start_sqs(config)

    assert handle.metadata["queue_name"] == "test-queue"
    assert (
        handle.connection_info["SQS_QUEUE_URL"]
        == "https://sqs.us-east-1.amazonaws.com/123/test"
    )
    assert handle.connection_info["SQS_QUEUE_NAME"] == "test-queue"
    assert mock_to_thread.await_count == 2
    first = mock_to_thread.await_args_list[0]
    second = mock_to_thread.await_args_list[1]
    assert first.args[0] is sqs.get_queue_url
    assert first.kwargs == {"QueueName": "test-queue"}
    assert second.args[0] is sqs.create_queue
    assert second.kwargs == {
        "QueueName": "test-queue",
        "tags": {"victor:managed": "true"},
    }
