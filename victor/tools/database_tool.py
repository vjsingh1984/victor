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

"""Database tool for SQL operations and schema inspection.

Supports multiple database types:
- SQLite (built-in, no dependencies)
- PostgreSQL (requires psycopg2)
- MySQL (requires mysql-connector-python)
- SQL Server (requires pyodbc)

Features:
- Execute SQL queries
- Inspect database schemas
- Table information
- Safe query execution with validation
"""

import json
import sqlite3
from typing import Any, Dict, Optional

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
            "message": f"Connected to SQLite database: {database}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"SQLite connection failed: {str(e)}"
        }


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
            "message": "Connected to PostgreSQL database"
        }
    except ImportError:
        return {
            "success": False,
            "error": "PostgreSQL support requires: pip install psycopg2-binary"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"PostgreSQL connection failed: {str(e)}"
        }


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
            "message": "Connected to MySQL database"
        }
    except ImportError:
        return {
            "success": False,
            "error": "MySQL support requires: pip install mysql-connector-python"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"MySQL connection failed: {str(e)}"
        }


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
            "message": "Connected to SQL Server database"
        }
    except ImportError:
        return {
            "success": False,
            "error": "SQL Server support requires: pip install pyodbc"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"SQL Server connection failed: {str(e)}"
        }


@tool
async def database_connect(
    database: str,
    db_type: str = "sqlite",
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> Dict[str, Any]:
    """
    Connect to a database.

    Supports SQLite, PostgreSQL, MySQL, and SQL Server. Returns a connection_id
    that must be used for subsequent operations.

    Args:
        database: Database name or path (for SQLite).
        db_type: Database type - 'sqlite', 'postgresql', 'mysql', or 'sqlserver' (default: 'sqlite').
        host: Database host for remote databases (default: 'localhost').
        port: Database port (defaults vary by db_type).
        username: Database username for authenticated connections.
        password: Database password for authenticated connections.

    Returns:
        Dictionary containing:
        - success: Whether connection succeeded
        - connection_id: ID to use for subsequent operations
        - message: Status message
        - error: Error message if failed
    """
    if db_type == "sqlite":
        return await _connect_sqlite(database)
    elif db_type == "postgresql":
        return await _connect_postgresql({
            "database": database,
            "host": host,
            "port": port,
            "username": username,
            "password": password
        })
    elif db_type == "mysql":
        return await _connect_mysql({
            "database": database,
            "host": host,
            "port": port,
            "username": username,
            "password": password
        })
    elif db_type == "sqlserver":
        return await _connect_sqlserver({
            "database": database,
            "host": host,
            "username": username,
            "password": password
        })
    else:
        return {
            "success": False,
            "error": f"Unsupported database type: {db_type}"
        }


@tool
async def database_query(
    connection_id: str,
    sql: str,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute a SQL query.

    Supports SELECT queries (returns results) and modification queries
    (INSERT/UPDATE/DELETE - if allowed). Read-only by default for safety.

    Args:
        connection_id: Connection ID from database_connect.
        sql: SQL query to execute.
        limit: Maximum rows to return (default: configured max_rows).

    Returns:
        Dictionary containing:
        - success: Whether query succeeded
        - columns: Column names (for SELECT)
        - rows: Query results as list of dicts (for SELECT)
        - count: Number of rows returned
        - rows_affected: Number of rows affected (for modifications)
        - error: Error message if failed
    """
    if not connection_id or connection_id not in _connections:
        return {
            "success": False,
            "error": "Invalid or missing connection_id. Use database_connect first."
        }

    if not sql:
        return {
            "success": False,
            "error": "Missing required parameter: sql"
        }

    # Check for dangerous patterns
    if not _allow_modifications:
        sql_upper = sql.upper()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in sql_upper:
                return {
                    "success": False,
                    "error": f"Modification operations not allowed: {pattern}. Call set_database_config(allow_modifications=True) to enable."
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
                    results.append(dict(zip(columns, row)))

            return {
                "success": True,
                "columns": columns,
                "rows": results,
                "count": len(results),
                "limited": len(rows) == query_limit
            }
        else:
            # Non-SELECT query (INSERT, UPDATE, DELETE)
            conn.commit()
            return {
                "success": True,
                "rows_affected": cursor.rowcount,
                "message": f"Query executed successfully. Rows affected: {cursor.rowcount}"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Query failed: {str(e)}"
        }


@tool
async def database_tables(connection_id: str) -> Dict[str, Any]:
    """
    List all tables in the database.

    Args:
        connection_id: Connection ID from database_connect.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - tables: List of table names
        - count: Number of tables
        - error: Error message if failed
    """
    if not connection_id or connection_id not in _connections:
        return {
            "success": False,
            "error": "Invalid or missing connection_id"
        }

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
            return {
                "success": False,
                "error": "Unknown database type"
            }

        cursor.execute(sql)
        rows = cursor.fetchall()
        tables = [row[0] for row in rows]

        return {
            "success": True,
            "tables": tables,
            "count": len(tables)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to list tables: {str(e)}"
        }


@tool
async def database_describe(connection_id: str, table: str) -> Dict[str, Any]:
    """
    Describe a table's structure.

    Returns column information including names, types, nullable status,
    and primary key information.

    Args:
        connection_id: Connection ID from database_connect.
        table: Name of the table to describe.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - table: Table name
        - columns: List of column info dicts
        - count: Number of columns
        - error: Error message if failed
    """
    if not connection_id or connection_id not in _connections:
        return {
            "success": False,
            "error": "Invalid or missing connection_id"
        }

    if not table:
        return {
            "success": False,
            "error": "Missing required parameter: table"
        }

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
                {"name": row[0], "type": row[1], "nullable": row[2] == "YES"}
                for row in rows
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
            return {
                "success": False,
                "error": "Describe not implemented for this database type"
            }

        return {
            "success": True,
            "table": table,
            "columns": columns,
            "count": len(columns)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to describe table: {str(e)}"
        }


@tool
async def database_schema(connection_id: str) -> Dict[str, Any]:
    """
    Get complete database schema.

    Returns information about all tables and their columns.
    This is a convenience function that combines database_tables
    and database_describe for all tables.

    Args:
        connection_id: Connection ID from database_connect.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - tables: List of table info dicts (name and columns)
        - error: Error message if failed
    """
    if not connection_id or connection_id not in _connections:
        return {
            "success": False,
            "error": "Invalid or missing connection_id"
        }

    # Get list of tables first
    tables_result = await database_tables(connection_id)
    if not tables_result["success"]:
        return tables_result

    # Get schema for each table
    schema_info = {"tables": []}

    try:
        for table in tables_result["tables"]:
            describe_result = await database_describe(connection_id, table)
            if describe_result["success"]:
                schema_info["tables"].append({
                    "name": table,
                    "columns": describe_result["columns"]
                })

        return {
            "success": True,
            "tables": schema_info["tables"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Schema inspection failed: {str(e)}"
        }


@tool
async def database_disconnect(connection_id: str) -> Dict[str, Any]:
    """
    Disconnect from database.

    Closes the connection and removes it from the connection pool.

    Args:
        connection_id: Connection ID from database_connect.

    Returns:
        Dictionary containing:
        - success: Whether disconnection succeeded
        - message: Status message
        - error: Error message if failed
    """
    if not connection_id or connection_id not in _connections:
        return {
            "success": False,
            "error": "Invalid or missing connection_id"
        }

    try:
        conn = _connections[connection_id]
        conn.close()
        del _connections[connection_id]

        return {
            "success": True,
            "message": f"Disconnected from database: {connection_id}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Disconnect failed: {str(e)}"
        }


# Keep class for backward compatibility
class DatabaseTool:
    """Deprecated: Use individual database_* functions instead."""

    def __init__(self, allow_modifications: bool = False, max_rows: int = 100):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "DatabaseTool class is deprecated. Use database_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
        set_database_config(allow_modifications, max_rows)
