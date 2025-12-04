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

"""Unified database tool for SQL operations and schema inspection.

Consolidates all database operations into a single tool for better token efficiency.
Supports multiple database types:
- SQLite (built-in, no dependencies)
- PostgreSQL (requires psycopg2)
- MySQL (requires mysql-connector-python)
- SQL Server (requires pyodbc)

Features:
- Connect/disconnect to databases
- Execute SQL queries
- Inspect database schemas
- List tables
- Describe table structure
- Safe query execution with validation
"""

import sqlite3
from typing import Any, Dict, List, Optional

from victor.tools.decorators import tool

# Global state
_allow_modifications: bool = False
_max_rows: int = 100
_connections: Dict[str, Any] = {}

# Dangerous SQL patterns that should be blocked
DANGEROUS_PATTERNS = [
    "DROP DATABASE",
    "DROP SCHEMA",
    "TRUNCATE",
    "DELETE FROM",
    "UPDATE",
    "INSERT INTO",
    "ALTER TABLE",
    "CREATE",
    "DROP TABLE",
]


def set_database_config(allow_modifications: bool = False, max_rows: int = 100) -> None:
    """Configure database tool settings.

    Args:
        allow_modifications: Allow INSERT/UPDATE/DELETE operations
        max_rows: Maximum rows to return from queries
    """
    global _allow_modifications, _max_rows
    _allow_modifications = allow_modifications
    _max_rows = max_rows


# Helper functions for db-specific connections
async def _connect_sqlite(database: str) -> Dict[str, Any]:
    """Connect to SQLite database."""
    try:
        conn = sqlite3.connect(database)
        conn.row_factory = sqlite3.Row  # Enable column names
        connection_id = f"sqlite_{id(conn)}"
        _connections[connection_id] = conn

        return {
            "success": True,
            "connection_id": connection_id,
            "message": f"Connected to SQLite database: {database}",
        }
    except Exception as e:
        return {"success": False, "error": f"SQLite connection failed: {str(e)}"}


async def _connect_postgresql(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Connect to PostgreSQL database."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 5432),
            database=kwargs.get("database"),
            user=kwargs.get("username"),
            password=kwargs.get("password"),
        )

        connection_id = f"postgresql_{id(conn)}"
        _connections[connection_id] = conn

        return {
            "success": True,
            "connection_id": connection_id,
            "message": "Connected to PostgreSQL database",
        }
    except ImportError:
        return {
            "success": False,
            "error": "PostgreSQL support requires: pip install psycopg2-binary",
        }
    except Exception as e:
        return {"success": False, "error": f"PostgreSQL connection failed: {str(e)}"}


async def _connect_mysql(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Connect to MySQL database."""
    try:
        import mysql.connector

        conn = mysql.connector.connect(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 3306),
            database=kwargs.get("database"),
            user=kwargs.get("username"),
            password=kwargs.get("password"),
        )

        connection_id = f"mysql_{id(conn)}"
        _connections[connection_id] = conn

        return {
            "success": True,
            "connection_id": connection_id,
            "message": "Connected to MySQL database",
        }
    except ImportError:
        return {
            "success": False,
            "error": "MySQL support requires: pip install mysql-connector-python",
        }
    except Exception as e:
        return {"success": False, "error": f"MySQL connection failed: {str(e)}"}


async def _connect_sqlserver(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Connect to SQL Server database."""
    try:
        import pyodbc

        conn_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={kwargs.get('host', 'localhost')};"
            f"DATABASE={kwargs.get('database')};"
            f"UID={kwargs.get('username')};"
            f"PWD={kwargs.get('password')}"
        )

        conn = pyodbc.connect(conn_string)
        connection_id = f"sqlserver_{id(conn)}"
        _connections[connection_id] = conn

        return {
            "success": True,
            "connection_id": connection_id,
            "message": "Connected to SQL Server database",
        }
    except ImportError:
        return {"success": False, "error": "SQL Server support requires: pip install pyodbc"}
    except Exception as e:
        return {"success": False, "error": f"SQL Server connection failed: {str(e)}"}


async def _do_connect(
    database: str,
    db_type: str,
    host: Optional[str],
    port: Optional[int],
    username: Optional[str],
    password: Optional[str],
) -> Dict[str, Any]:
    """Internal connect handler."""
    if db_type == "sqlite":
        return await _connect_sqlite(database)
    elif db_type == "postgresql":
        return await _connect_postgresql(
            {
                "database": database,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
            }
        )
    elif db_type == "mysql":
        return await _connect_mysql(
            {
                "database": database,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
            }
        )
    elif db_type == "sqlserver":
        return await _connect_sqlserver(
            {"database": database, "host": host, "username": username, "password": password}
        )
    else:
        return {"success": False, "error": f"Unsupported database type: {db_type}"}


async def _do_query(connection_id: str, sql: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Internal query handler."""
    if not connection_id or connection_id not in _connections:
        return {
            "success": False,
            "error": "Invalid or missing connection_id. Use action='connect' first.",
        }

    if not sql:
        return {"success": False, "error": "Missing required parameter: sql"}

    # Check for dangerous patterns
    if not _allow_modifications:
        sql_upper = sql.upper()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in sql_upper:
                return {
                    "success": False,
                    "error": f"Modification operations not allowed: {pattern}. "
                    "Call set_database_config(allow_modifications=True) to enable.",
                }

    try:
        conn = _connections[connection_id]
        cursor = conn.cursor()
        cursor.execute(sql)

        # Check if query returns results
        if cursor.description:
            # SELECT query - fetch results
            columns = [desc[0] for desc in cursor.description]
            query_limit = limit if limit is not None else _max_rows
            rows = cursor.fetchmany(query_limit)

            # Convert to list of dicts
            results = []
            for row in rows:
                if hasattr(row, "keys"):  # SQLite Row object
                    results.append(dict(row))
                else:  # Tuple
                    results.append(dict(zip(columns, row, strict=False)))

            return {
                "success": True,
                "columns": columns,
                "rows": results,
                "count": len(results),
                "limited": len(rows) == query_limit,
            }
        else:
            # Non-SELECT query (INSERT, UPDATE, DELETE)
            conn.commit()
            return {
                "success": True,
                "rows_affected": cursor.rowcount,
                "message": f"Query executed successfully. Rows affected: {cursor.rowcount}",
            }

    except Exception as e:
        return {"success": False, "error": f"Query failed: {str(e)}"}


async def _do_tables(connection_id: str) -> Dict[str, Any]:
    """Internal list tables handler."""
    if not connection_id or connection_id not in _connections:
        return {"success": False, "error": "Invalid or missing connection_id"}

    try:
        conn = _connections[connection_id]
        cursor = conn.cursor()

        if connection_id.startswith("sqlite"):
            sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        elif connection_id.startswith("postgresql"):
            sql = "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        elif connection_id.startswith("mysql"):
            sql = "SHOW TABLES"
        elif connection_id.startswith("sqlserver"):
            sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME"
        else:
            return {"success": False, "error": "Unknown database type"}

        cursor.execute(sql)
        rows = cursor.fetchall()
        tables = [row[0] for row in rows]

        return {"success": True, "tables": tables, "count": len(tables)}

    except Exception as e:
        return {"success": False, "error": f"Failed to list tables: {str(e)}"}


async def _do_describe(connection_id: str, table: str) -> Dict[str, Any]:
    """Internal describe table handler."""
    if not connection_id or connection_id not in _connections:
        return {"success": False, "error": "Invalid or missing connection_id"}

    if not table:
        return {"success": False, "error": "Missing required parameter: table"}

    try:
        conn = _connections[connection_id]
        cursor = conn.cursor()

        if connection_id.startswith("sqlite"):
            cursor.execute(f"PRAGMA table_info({table})")
            rows = cursor.fetchall()
            columns = [
                {
                    "name": row[1],
                    "type": row[2],
                    "nullable": not row[3],
                    "primary_key": bool(row[5]),
                }
                for row in rows
            ]

        elif connection_id.startswith("postgresql"):
            cursor.execute(
                f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
                """
            )
            rows = cursor.fetchall()
            columns = [
                {"name": row[0], "type": row[1], "nullable": row[2] == "YES"} for row in rows
            ]

        elif connection_id.startswith("mysql"):
            cursor.execute(f"DESCRIBE {table}")
            rows = cursor.fetchall()
            columns = [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "primary_key": row[3] == "PRI",
                }
                for row in rows
            ]

        else:
            return {"success": False, "error": "Describe not implemented for this database type"}

        return {"success": True, "table": table, "columns": columns, "count": len(columns)}

    except Exception as e:
        return {"success": False, "error": f"Failed to describe table: {str(e)}"}


async def _do_schema(connection_id: str) -> Dict[str, Any]:
    """Internal get schema handler."""
    if not connection_id or connection_id not in _connections:
        return {"success": False, "error": "Invalid or missing connection_id"}

    # Get list of tables first
    tables_result = await _do_tables(connection_id)
    if not tables_result["success"]:
        return tables_result

    # Get schema for each table
    schema_info: Dict[str, List[Dict[str, Any]]] = {"tables": []}

    try:
        for table in tables_result["tables"]:
            describe_result = await _do_describe(connection_id, table)
            if describe_result["success"]:
                schema_info["tables"].append({"name": table, "columns": describe_result["columns"]})

        return {"success": True, "tables": schema_info["tables"]}

    except Exception as e:
        return {"success": False, "error": f"Schema inspection failed: {str(e)}"}


async def _do_disconnect(connection_id: str) -> Dict[str, Any]:
    """Internal disconnect handler."""
    if not connection_id or connection_id not in _connections:
        return {"success": False, "error": "Invalid or missing connection_id"}

    try:
        conn = _connections[connection_id]
        conn.close()
        del _connections[connection_id]

        return {"success": True, "message": f"Disconnected from database: {connection_id}"}

    except Exception as e:
        return {"success": False, "error": f"Disconnect failed: {str(e)}"}


@tool
async def database(
    action: str,
    database: Optional[str] = None,
    db_type: str = "sqlite",
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    connection_id: Optional[str] = None,
    sql: Optional[str] = None,
    table: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Unified database tool for SQL operations. Supports SQLite, PostgreSQL, MySQL, SQL Server.

    Actions:
    - connect: Connect to a database
    - query: Execute SQL queries
    - tables: List all tables
    - describe: Describe a table's structure
    - schema: Get complete database schema
    - disconnect: Close connection

    Args:
        action: Operation to perform - 'connect', 'query', 'tables', 'describe', 'schema', 'disconnect'.
        database: Database name or path (required for connect).
        db_type: Database type - 'sqlite', 'postgresql', 'mysql', 'sqlserver' (default: 'sqlite').
        host: Database host for remote databases (default: 'localhost').
        port: Database port (defaults vary by db_type).
        username: Database username for authenticated connections.
        password: Database password for authenticated connections.
        connection_id: Connection ID from previous connect (required for query/tables/describe/schema/disconnect).
        sql: SQL query to execute (required for query action).
        table: Table name (required for describe action).
        limit: Maximum rows to return for query (default: 100).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - For connect: connection_id, message
        - For query: columns, rows, count
        - For tables: tables list, count
        - For describe: table name, columns list
        - For schema: tables with their columns
        - error: Error message if failed

    Example:
        # Connect to PostgreSQL
        database(action="connect", database="mydb", db_type="postgresql",
                host="localhost", username="user", password="pass")

        # Query with returned connection_id
        database(action="query", connection_id="postgresql_123",
                sql="SELECT * FROM users LIMIT 10")

        # List tables
        database(action="tables", connection_id="postgresql_123")

        # Describe table
        database(action="describe", connection_id="postgresql_123", table="users")
    """
    action_lower = action.lower().strip()

    if action_lower == "connect":
        if not database:
            return {"success": False, "error": "Missing required parameter: database"}
        return await _do_connect(database, db_type, host, port, username, password)

    elif action_lower == "query":
        if not connection_id:
            return {"success": False, "error": "Missing required parameter: connection_id"}
        if not sql:
            return {"success": False, "error": "Missing required parameter: sql"}
        return await _do_query(connection_id, sql, limit)

    elif action_lower == "tables":
        if not connection_id:
            return {"success": False, "error": "Missing required parameter: connection_id"}
        return await _do_tables(connection_id)

    elif action_lower == "describe":
        if not connection_id:
            return {"success": False, "error": "Missing required parameter: connection_id"}
        if not table:
            return {"success": False, "error": "Missing required parameter: table"}
        return await _do_describe(connection_id, table)

    elif action_lower == "schema":
        if not connection_id:
            return {"success": False, "error": "Missing required parameter: connection_id"}
        return await _do_schema(connection_id)

    elif action_lower == "disconnect":
        if not connection_id:
            return {"success": False, "error": "Missing required parameter: connection_id"}
        return await _do_disconnect(connection_id)

    else:
        return {
            "success": False,
            "error": f"Unknown action: {action}. Valid actions: connect, query, tables, describe, schema, disconnect",
        }
