"""Tests for CQRS module: CommandBus, QueryBus, Command, Query."""

from dataclasses import dataclass

import pytest

from victor.core.cqrs import (
    Command,
    CommandBus,
    CommandResult,
    Query,
    QueryBus,
    QueryResult,
    clear_handlers,
)


@dataclass
class CreateItemCommand(Command):
    name: str = ""


@dataclass
class GetItemQuery(Query[str]):
    item_id: str = ""


@pytest.fixture(autouse=True)
def cleanup_handlers():
    clear_handlers()
    yield
    clear_handlers()


class TestCommand:
    def test_auto_generates_id(self):
        cmd = CreateItemCommand(name="test")
        assert cmd.command_id
        assert cmd.correlation_id == cmd.command_id

    def test_custom_correlation_id(self):
        cmd = CreateItemCommand(name="test", correlation_id="custom")
        assert cmd.correlation_id == "custom"


class TestCommandBus:
    async def test_register_and_execute(self):
        bus = CommandBus()

        async def handler(cmd):
            return f"created:{cmd.name}"

        bus.register(CreateItemCommand, handler)
        result = await bus.execute(CreateItemCommand(name="widget"))
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.result == "created:widget"

    async def test_no_handler_returns_failure(self):
        bus = CommandBus()
        result = await bus.execute(CreateItemCommand(name="x"))
        assert result.success is False
        assert "No handler" in result.error

    async def test_handler_exception_returns_failure(self):
        bus = CommandBus()

        async def bad_handler(cmd):
            raise ValueError("boom")

        bus.register(CreateItemCommand, bad_handler)
        result = await bus.execute(CreateItemCommand(name="x"))
        assert result.success is False
        assert "boom" in result.error

    async def test_execution_time_recorded(self):
        bus = CommandBus()

        async def handler(cmd):
            return "ok"

        bus.register(CreateItemCommand, handler)
        result = await bus.execute(CreateItemCommand(name="x"))
        assert result.execution_time_ms >= 0


class TestQueryBus:
    async def test_register_and_execute(self):
        bus = QueryBus()

        async def handler(query):
            return f"item-{query.item_id}"

        bus.register(GetItemQuery, handler)
        result = await bus.execute(GetItemQuery(item_id="42"))
        assert isinstance(result, QueryResult)
        assert result.success is True
        assert result.data == "item-42"

    async def test_no_handler_returns_failure(self):
        bus = QueryBus()
        result = await bus.execute(GetItemQuery(item_id="1"))
        assert result.success is False
        assert "No handler" in result.error

    async def test_handler_exception_returns_failure(self):
        bus = QueryBus()

        async def bad_handler(query):
            raise RuntimeError("db error")

        bus.register(GetItemQuery, bad_handler)
        result = await bus.execute(GetItemQuery(item_id="1"))
        assert result.success is False
        assert "db error" in result.error
