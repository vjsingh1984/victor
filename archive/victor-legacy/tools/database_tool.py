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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.base import BaseTool, ToolParameter, ToolResult


class DatabaseTool(BaseTool):
    """Tool for database operations and SQL queries."""

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

    def __init__(
        self,
        allow_modifications: bool = False,
        max_rows: int = 100,
    ):
        """Initialize database tool.

        Args:
            allow_modifications: Allow INSERT/UPDATE/DELETE operations
            max_rows: Maximum rows to return from queries
        """
        super().__init__()
        self.allow_modifications = allow_modifications
        self.max_rows = max_rows
        self.connections: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Get tool name."""
        return "database"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Database operations and SQL queries.

Supports multiple database types (SQLite, PostgreSQL, MySQL, SQL Server).

Operations:
- connect: Connect to database
- query: Execute SQL query
- schema: Get database schema information
- tables: List all tables
- describe: Describe table structure
- disconnect: Close database connection

Example workflows:
1. Connect and query:
   database(operation="connect", db_type="sqlite", database="mydb.db")
   database(operation="query", connection_id="...", sql="SELECT * FROM users")

2. Schema inspection:
   database(operation="tables", connection_id="...")
   database(operation="describe", connection_id="...", table="users")

Safety:
- Read-only by default (no INSERT/UPDATE/DELETE)
- Configurable modification permissions
- Query result limits
- Dangerous pattern detection
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
            [
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation: connect, query, schema, tables, describe, disconnect",
                    required=True,
                ),
                ToolParameter(
                    name="db_type",
                    type="string",
                    description="Database type: sqlite, postgresql, mysql, sqlserver",
                    required=False,
                ),
                ToolParameter(
                    name="database",
                    type="string",
                    description="Database name or path (for SQLite)",
                    required=False,
                ),
                ToolParameter(
                    name="host",
                    type="string",
                    description="Database host (default: localhost)",
                    required=False,
                ),
                ToolParameter(
                    name="port",
                    type="integer",
                    description="Database port",
                    required=False,
                ),
                ToolParameter(
                    name="username",
                    type="string",
                    description="Database username",
                    required=False,
                ),
                ToolParameter(
                    name="password",
                    type="string",
                    description="Database password",
                    required=False,
                ),
                ToolParameter(
                    name="connection_id",
                    type="string",
                    description="Connection ID (returned from connect)",
                    required=False,
                ),
                ToolParameter(
                    name="sql",
                    type="string",
                    description="SQL query to execute",
                    required=False,
                ),
                ToolParameter(
                    name="table",
                    type="string",
                    description="Table name (for describe operation)",
                    required=False,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum rows to return (default: 100)",
                    required=False,
                ),
            ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute database operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with query results or error
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "connect":
                return await self._connect(kwargs)
            elif operation == "query":
                return await self._query(kwargs)
            elif operation == "schema":
                return await self._schema(kwargs)
            elif operation == "tables":
                return await self._tables(kwargs)
            elif operation == "describe":
                return await self._describe(kwargs)
            elif operation == "disconnect":
                return await self._disconnect(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            return ToolResult(success=False, output="", error=f"Database error: {str(e)}")

    async def _connect(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Connect to database.

        Args:
            kwargs: Connection parameters

        Returns:
            Connection result with connection ID
        """
        db_type = kwargs.get("db_type", "sqlite")
        database = kwargs.get("database")

        if not database:
            return ToolResult(
                success=False, output="", error="Missing required parameter: database"
            )

        if db_type == "sqlite":
            return await self._connect_sqlite(database)
        elif db_type == "postgresql":
            return await self._connect_postgresql(kwargs)
        elif db_type == "mysql":
            return await self._connect_mysql(kwargs)
        elif db_type == "sqlserver":
            return await self._connect_sqlserver(kwargs)
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unsupported database type: {db_type}",
            )

    async def _connect_sqlite(self, database: str) -> ToolResult:
        """Connect to SQLite database."""
        try:
            conn = sqlite3.connect(database)
            conn.row_factory = sqlite3.Row  # Enable column names
            connection_id = f"sqlite_{id(conn)}"
            self.connections[connection_id] = conn

            return ToolResult(
                success=True,
                output=f"Connected to SQLite database: {database}\nConnection ID: {connection_id}",
                error="",
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=f"SQLite connection failed: {str(e)}")

    async def _connect_postgresql(self, kwargs: Dict[str, Any]) -> ToolResult:
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
            self.connections[connection_id] = conn

            return ToolResult(
                success=True,
                output=f"Connected to PostgreSQL database\nConnection ID: {connection_id}",
                error="",
            )

        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="PostgreSQL support requires: pip install psycopg2-binary",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"PostgreSQL connection failed: {str(e)}",
            )

    async def _connect_mysql(self, kwargs: Dict[str, Any]) -> ToolResult:
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
            self.connections[connection_id] = conn

            return ToolResult(
                success=True,
                output=f"Connected to MySQL database\nConnection ID: {connection_id}",
                error="",
            )

        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="MySQL support requires: pip install mysql-connector-python",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"MySQL connection failed: {str(e)}")

    async def _connect_sqlserver(self, kwargs: Dict[str, Any]) -> ToolResult:
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
            self.connections[connection_id] = conn

            return ToolResult(
                success=True,
                output=f"Connected to SQL Server database\nConnection ID: {connection_id}",
                error="",
            )

        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="SQL Server support requires: pip install pyodbc",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"SQL Server connection failed: {str(e)}",
            )

    async def _query(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Execute SQL query."""
        connection_id = kwargs.get("connection_id")
        sql = kwargs.get("sql")
        limit = kwargs.get("limit", self.max_rows)

        if not connection_id or connection_id not in self.connections:
            return ToolResult(
                success=False,
                output="",
                error="Invalid or missing connection_id. Use connect operation first.",
            )

        if not sql:
            return ToolResult(success=False, output="", error="Missing required parameter: sql")

        # Check for dangerous patterns
        if not self.allow_modifications:
            sql_upper = sql.upper()
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern in sql_upper:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Modification operations not allowed: {pattern}. Set allow_modifications=True to enable.",
                    )

        try:
            conn = self.connections[connection_id]
            cursor = conn.cursor()

            # Execute query
            cursor.execute(sql)

            # Check if query returns results
            if cursor.description:
                # SELECT query - fetch results
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(limit)

                # Convert to list of dicts
                results = []
                for row in rows:
                    if hasattr(row, "keys"):  # SQLite Row object
                        results.append(dict(row))
                    else:  # Tuple
                        results.append(dict(zip(columns, row)))

                output = {
                    "columns": columns,
                    "rows": results,
                    "count": len(results),
                    "limited": len(rows) == limit,
                }

                return ToolResult(success=True, output=json.dumps(output, indent=2), error="")
            else:
                # Non-SELECT query (INSERT, UPDATE, DELETE)
                conn.commit()
                rowcount = cursor.rowcount

                return ToolResult(
                    success=True,
                    output=f"Query executed successfully. Rows affected: {rowcount}",
                    error="",
                )

        except Exception as e:
            return ToolResult(success=False, output="", error=f"Query failed: {str(e)}")

    async def _schema(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Get database schema information."""
        connection_id = kwargs.get("connection_id")

        if not connection_id or connection_id not in self.connections:
            return ToolResult(success=False, output="", error="Invalid or missing connection_id")

        # Get list of tables first
        tables_result = await self._tables(kwargs)
        if not tables_result.success:
            return tables_result

        # Get schema for each table
        schema_info = {"tables": []}

        try:
            tables_data = json.loads(tables_result.output)
            for table in tables_data["tables"]:
                describe_result = await self._describe(
                    {"connection_id": connection_id, "table": table}
                )
                if describe_result.success:
                    table_info = json.loads(describe_result.output)
                    schema_info["tables"].append({"name": table, "columns": table_info["columns"]})

            return ToolResult(success=True, output=json.dumps(schema_info, indent=2), error="")

        except Exception as e:
            return ToolResult(success=False, output="", error=f"Schema inspection failed: {str(e)}")

    async def _tables(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List all tables in database."""
        connection_id = kwargs.get("connection_id")

        if not connection_id or connection_id not in self.connections:
            return ToolResult(success=False, output="", error="Invalid or missing connection_id")

        try:
            conn = self.connections[connection_id]
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
                return ToolResult(success=False, output="", error="Unknown database type")

            cursor.execute(sql)
            rows = cursor.fetchall()

            tables = [row[0] for row in rows]

            return ToolResult(
                success=True,
                output=json.dumps({"tables": tables, "count": len(tables)}, indent=2),
                error="",
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to list tables: {str(e)}")

    async def _describe(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Describe table structure."""
        connection_id = kwargs.get("connection_id")
        table = kwargs.get("table")

        if not connection_id or connection_id not in self.connections:
            return ToolResult(success=False, output="", error="Invalid or missing connection_id")

        if not table:
            return ToolResult(success=False, output="", error="Missing required parameter: table")

        try:
            conn = self.connections[connection_id]
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
                return ToolResult(
                    success=False,
                    output="",
                    error="Describe not implemented for this database type",
                )

            return ToolResult(
                success=True,
                output=json.dumps(
                    {"table": table, "columns": columns, "count": len(columns)},
                    indent=2,
                ),
                error="",
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to describe table: {str(e)}")

    async def _disconnect(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Disconnect from database."""
        connection_id = kwargs.get("connection_id")

        if not connection_id or connection_id not in self.connections:
            return ToolResult(success=False, output="", error="Invalid or missing connection_id")

        try:
            conn = self.connections[connection_id]
            conn.close()
            del self.connections[connection_id]

            return ToolResult(
                success=True,
                output=f"Disconnected from database: {connection_id}",
                error="",
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=f"Disconnect failed: {str(e)}")

    def __del__(self):
        """Cleanup: close all connections."""
        for conn in self.connections.values():
            try:
                conn.close()
            except:
                pass
