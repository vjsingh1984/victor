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

"""Tests for database_tool module."""

import pytest
import tempfile
import os

from victor.tools.database_tool import (
    database_connect,
    database_query,
    database_tables,
    database_describe,
    database_schema,
    database_disconnect,
    set_database_config,
    DANGEROUS_PATTERNS,
    _connections,
    DatabaseTool,
)


class TestSetDatabaseConfig:
    """Tests for set_database_config function."""

    def test_set_config_default(self):
        """Test setting config with defaults."""
        set_database_config()
        # Just verify no exception

    def test_set_config_custom(self):
        """Test setting config with custom values."""
        set_database_config(allow_modifications=True, max_rows=50)
        # Just verify no exception


class TestDatabaseConnect:
    """Tests for database_connect function."""

    @pytest.mark.asyncio
    async def test_connect_sqlite_memory(self):
        """Test SQLite in-memory connection."""
        result = await database_connect(database=":memory:")
        assert result["success"] is True
        assert "connection_id" in result
        assert "sqlite_" in result["connection_id"]
        # Cleanup
        if result["success"]:
            await database_disconnect(result["connection_id"])

    @pytest.mark.asyncio
    async def test_connect_sqlite_file(self):
        """Test SQLite file connection."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            result = await database_connect(database=db_path, db_type="sqlite")
            assert result["success"] is True
            if result["success"]:
                await database_disconnect(result["connection_id"])
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_connect_unsupported_type(self):
        """Test connection with unsupported database type."""
        result = await database_connect(database="test", db_type="mongodb")
        assert result["success"] is False
        assert "Unsupported database type" in result["error"]

    @pytest.mark.asyncio
    async def test_connect_postgresql_no_driver(self):
        """Test PostgreSQL connection without driver."""
        result = await database_connect(
            database="testdb",
            db_type="postgresql",
            host="localhost",
            username="user",
            password="pass",
        )
        # Either succeeds (if psycopg2 is installed) or fails with import error
        assert "success" in result

    @pytest.mark.asyncio
    async def test_connect_mysql_no_driver(self):
        """Test MySQL connection without driver."""
        result = await database_connect(
            database="testdb",
            db_type="mysql",
            host="localhost",
            username="user",
            password="pass",
        )
        # Either succeeds (if mysql-connector is installed) or fails with import error
        assert "success" in result

    @pytest.mark.asyncio
    async def test_connect_sqlserver_no_driver(self):
        """Test SQL Server connection without driver."""
        result = await database_connect(
            database="testdb",
            db_type="sqlserver",
            host="localhost",
            username="user",
            password="pass",
        )
        # Either succeeds (if pyodbc is installed) or fails with import error
        assert "success" in result


class TestDatabaseQuery:
    """Tests for database_query function."""

    @pytest.fixture
    async def sqlite_conn(self):
        """Create SQLite connection for testing."""
        # Reset config
        set_database_config(allow_modifications=False, max_rows=100)

        result = await database_connect(database=":memory:")
        conn_id = result["connection_id"]

        # Create test table
        _connections[conn_id].execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
        """
        )
        _connections[conn_id].execute(
            "INSERT INTO users (name, email) VALUES ('Alice', 'alice@test.com')"
        )
        _connections[conn_id].execute(
            "INSERT INTO users (name, email) VALUES ('Bob', 'bob@test.com')"
        )
        _connections[conn_id].commit()

        yield conn_id

        await database_disconnect(conn_id)

    @pytest.mark.asyncio
    async def test_query_invalid_connection(self):
        """Test query with invalid connection."""
        result = await database_query(connection_id="invalid_conn", sql="SELECT 1")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_query_missing_sql(self, sqlite_conn):
        """Test query with missing SQL."""
        result = await database_query(connection_id=sqlite_conn, sql="")
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_query_select(self, sqlite_conn):
        """Test SELECT query."""
        result = await database_query(connection_id=sqlite_conn, sql="SELECT * FROM users")
        assert result["success"] is True
        assert result["count"] == 2
        assert "columns" in result
        assert "rows" in result
        assert "id" in result["columns"]
        assert "name" in result["columns"]

    @pytest.mark.asyncio
    async def test_query_select_with_limit(self, sqlite_conn):
        """Test SELECT query with limit."""
        result = await database_query(connection_id=sqlite_conn, sql="SELECT * FROM users", limit=1)
        assert result["success"] is True
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_query_dangerous_blocked(self, sqlite_conn):
        """Test dangerous queries are blocked."""
        result = await database_query(connection_id=sqlite_conn, sql="DROP TABLE users")
        assert result["success"] is False
        assert "Modification operations not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_query_insert_blocked(self, sqlite_conn):
        """Test INSERT is blocked by default."""
        result = await database_query(
            connection_id=sqlite_conn,
            sql="INSERT INTO users (name, email) VALUES ('Charlie', 'c@test.com')",
        )
        assert result["success"] is False
        assert "Modification operations not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_query_update_blocked(self, sqlite_conn):
        """Test UPDATE is blocked by default."""
        result = await database_query(
            connection_id=sqlite_conn, sql="UPDATE users SET name = 'Updated' WHERE id = 1"
        )
        assert result["success"] is False
        assert "Modification operations not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_query_delete_blocked(self, sqlite_conn):
        """Test DELETE is blocked by default."""
        result = await database_query(
            connection_id=sqlite_conn, sql="DELETE FROM users WHERE id = 1"
        )
        assert result["success"] is False
        assert "Modification operations not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_query_modifications_allowed(self, sqlite_conn):
        """Test modifications work when allowed."""
        set_database_config(allow_modifications=True, max_rows=100)

        result = await database_query(
            connection_id=sqlite_conn,
            sql="INSERT INTO users (name, email) VALUES ('Charlie', 'c@test.com')",
        )
        assert result["success"] is True
        assert result["rows_affected"] == 1

        # Reset config
        set_database_config(allow_modifications=False, max_rows=100)

    @pytest.mark.asyncio
    async def test_query_invalid_sql(self, sqlite_conn):
        """Test invalid SQL."""
        result = await database_query(connection_id=sqlite_conn, sql="INVALID SQL STATEMENT")
        assert result["success"] is False
        assert "Query failed" in result["error"]


class TestDatabaseTables:
    """Tests for database_tables function."""

    @pytest.fixture
    async def sqlite_conn(self):
        """Create SQLite connection for testing."""
        result = await database_connect(database=":memory:")
        conn_id = result["connection_id"]

        # Create test tables
        _connections[conn_id].execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        _connections[conn_id].execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
        _connections[conn_id].commit()

        yield conn_id

        await database_disconnect(conn_id)

    @pytest.mark.asyncio
    async def test_tables_invalid_connection(self):
        """Test listing tables with invalid connection."""
        result = await database_tables(connection_id="invalid_conn")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_tables_list(self, sqlite_conn):
        """Test listing tables."""
        result = await database_tables(connection_id=sqlite_conn)
        assert result["success"] is True
        assert "tables" in result
        assert "users" in result["tables"]
        assert "orders" in result["tables"]
        assert result["count"] == 2


class TestDatabaseDescribe:
    """Tests for database_describe function."""

    @pytest.fixture
    async def sqlite_conn(self):
        """Create SQLite connection for testing."""
        result = await database_connect(database=":memory:")
        conn_id = result["connection_id"]

        _connections[conn_id].execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                age INTEGER
            )
        """
        )
        _connections[conn_id].commit()

        yield conn_id

        await database_disconnect(conn_id)

    @pytest.mark.asyncio
    async def test_describe_invalid_connection(self):
        """Test describe with invalid connection."""
        result = await database_describe(connection_id="invalid_conn", table="users")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_describe_missing_table(self, sqlite_conn):
        """Test describe with missing table name."""
        result = await database_describe(connection_id=sqlite_conn, table="")
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_describe_table(self, sqlite_conn):
        """Test describing a table."""
        result = await database_describe(connection_id=sqlite_conn, table="users")
        assert result["success"] is True
        assert result["table"] == "users"
        assert result["count"] == 4
        assert len(result["columns"]) == 4

        # Check column details
        col_names = [c["name"] for c in result["columns"]]
        assert "id" in col_names
        assert "name" in col_names
        assert "email" in col_names
        assert "age" in col_names


class TestDatabaseSchema:
    """Tests for database_schema function."""

    @pytest.fixture
    async def sqlite_conn(self):
        """Create SQLite connection for testing."""
        result = await database_connect(database=":memory:")
        conn_id = result["connection_id"]

        _connections[conn_id].execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        _connections[conn_id].execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)"
        )
        _connections[conn_id].commit()

        yield conn_id

        await database_disconnect(conn_id)

    @pytest.mark.asyncio
    async def test_schema_invalid_connection(self):
        """Test schema with invalid connection."""
        result = await database_schema(connection_id="invalid_conn")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_full(self, sqlite_conn):
        """Test getting full schema."""
        result = await database_schema(connection_id=sqlite_conn)
        assert result["success"] is True
        assert "tables" in result
        assert len(result["tables"]) == 2

        table_names = [t["name"] for t in result["tables"]]
        assert "users" in table_names
        assert "orders" in table_names

        # Check columns are included
        for table in result["tables"]:
            assert "columns" in table


class TestDatabaseDisconnect:
    """Tests for database_disconnect function."""

    @pytest.mark.asyncio
    async def test_disconnect_invalid_connection(self):
        """Test disconnect with invalid connection."""
        result = await database_disconnect(connection_id="invalid_conn")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_disconnect_success(self):
        """Test successful disconnect."""
        # Connect first
        conn_result = await database_connect(database=":memory:")
        assert conn_result["success"] is True
        conn_id = conn_result["connection_id"]

        # Disconnect
        result = await database_disconnect(connection_id=conn_id)
        assert result["success"] is True
        assert "Disconnected" in result["message"]

        # Verify connection is removed
        assert conn_id not in _connections


class TestDangerousPatterns:
    """Tests for dangerous SQL pattern detection."""

    def test_dangerous_patterns_list(self):
        """Test dangerous patterns list."""
        assert "DROP DATABASE" in DANGEROUS_PATTERNS
        assert "DROP TABLE" in DANGEROUS_PATTERNS
        assert "TRUNCATE" in DANGEROUS_PATTERNS
        assert "DELETE FROM" in DANGEROUS_PATTERNS
        assert "UPDATE" in DANGEROUS_PATTERNS
        assert "INSERT INTO" in DANGEROUS_PATTERNS


class TestDatabaseToolDeprecation:
    """Tests for deprecated DatabaseTool class."""

    def test_deprecated_class_warning(self):
        """Test deprecation warning for DatabaseTool class."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DatabaseTool()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
