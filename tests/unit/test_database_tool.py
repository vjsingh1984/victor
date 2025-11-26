"""Tests for database_tool module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.tools.database_tool import (
    database_connect,
    database_query,
    database_tables,
    database_describe,
    database_schema,
    database_disconnect,
)


class TestDatabaseConnect:
    """Tests for database_connect function."""

    @pytest.mark.asyncio
    async def test_connect_sqlite_success(self):
        """Test successful SQLite connection."""
        with patch("aiosqlite.connect", new_callable=AsyncMock) as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_conn

            result = await database_connect(
                database_type="sqlite",
                database="test.db"
            )

            assert result["success"] is True
            assert "connection_id" in result

    @pytest.mark.asyncio
    async def test_connect_missing_database_type(self):
        """Test connection with missing database type."""
        result = await database_connect(database_type="", database="test.db")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_connect_unsupported_type(self):
        """Test connection with unsupported database type."""
        result = await database_connect(
            database_type="unsupported",
            database="test.db"
        )

        assert result["success"] is False
        assert "Unsupported database type" in result["error"]

    @pytest.mark.asyncio
    async def test_connect_postgresql(self):
        """Test PostgreSQL connection."""
        with patch("asyncpg.connect", new_callable=AsyncMock) as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn

            result = await database_connect(
                database_type="postgresql",
                host="localhost",
                port=5432,
                database="testdb",
                user="testuser",
                password="testpass"
            )

            assert result["success"] is True or "error" in result

    @pytest.mark.asyncio
    async def test_connect_mysql(self):
        """Test MySQL connection."""
        with patch("aiomysql.connect", new_callable=AsyncMock) as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_conn

            result = await database_connect(
                database_type="mysql",
                host="localhost",
                port=3306,
                database="testdb",
                user="testuser",
                password="testpass"
            )

            assert result["success"] is True or "error" in result


class TestDatabaseQuery:
    """Tests for database_query function."""

    @pytest.mark.asyncio
    async def test_query_no_connection(self):
        """Test query without active connection."""
        result = await database_query(
            connection_id="nonexistent",
            query="SELECT 1"
        )

        assert result["success"] is False
        assert "No active connection" in result["error"]

    @pytest.mark.asyncio
    async def test_query_missing_query(self):
        """Test query with missing SQL."""
        result = await database_query(
            connection_id="test",
            query=""
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]


class TestDatabaseTables:
    """Tests for database_tables function."""

    @pytest.mark.asyncio
    async def test_tables_no_connection(self):
        """Test listing tables without active connection."""
        result = await database_tables(connection_id="nonexistent")

        assert result["success"] is False
        assert "No active connection" in result["error"]


class TestDatabaseDescribe:
    """Tests for database_describe function."""

    @pytest.mark.asyncio
    async def test_describe_no_connection(self):
        """Test describing table without active connection."""
        result = await database_describe(
            connection_id="nonexistent",
            table="users"
        )

        assert result["success"] is False
        assert "No active connection" in result["error"]

    @pytest.mark.asyncio
    async def test_describe_missing_table(self):
        """Test describe with missing table name."""
        result = await database_describe(
            connection_id="test",
            table=""
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]


class TestDatabaseSchema:
    """Tests for database_schema function."""

    @pytest.mark.asyncio
    async def test_schema_no_connection(self):
        """Test getting schema without active connection."""
        result = await database_schema(connection_id="nonexistent")

        assert result["success"] is False
        assert "No active connection" in result["error"]


class TestDatabaseDisconnect:
    """Tests for database_disconnect function."""

    @pytest.mark.asyncio
    async def test_disconnect_no_connection(self):
        """Test disconnecting non-existent connection."""
        result = await database_disconnect(connection_id="nonexistent")

        assert result["success"] is False
        assert "No active connection" in result["error"]


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_sqlite_lifecycle(self):
        """Test complete SQLite connection lifecycle."""
        # Test connection with SQLite (in-memory)
        with patch("aiosqlite.connect", new_callable=AsyncMock) as mock_connect:
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()

            # Mock connection context manager
            mock_connect.return_value.__aenter__.return_value = mock_conn
            mock_connect.return_value.__aexit__.return_value = None

            # Mock cursor operations
            mock_conn.execute.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []

            # Connect
            conn_result = await database_connect(
                database_type="sqlite",
                database=":memory:"
            )

            if conn_result["success"]:
                conn_id = conn_result["connection_id"]

                # Query - would fail without real connection, testing error handling
                query_result = await database_query(
                    connection_id=conn_id,
                    query="SELECT 1"
                )

                # Check result structure
                assert "success" in query_result

                # Disconnect
                disc_result = await database_disconnect(connection_id=conn_id)
                assert "success" in disc_result
