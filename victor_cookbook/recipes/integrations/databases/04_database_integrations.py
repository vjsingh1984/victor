"""Database integration recipes.

Recipes for connecting Victor agents to databases for data storage,
retrieval, and analysis.
"""

RECIPE_CATEGORY = "integrations/databases"
RECIPE_DIFFICULTY = "intermediate"
RECIPE_TIME = "15 minutes"


async def postgres_query_execution():
    """Execute PostgreSQL queries with natural language."""
    from victor import Agent

    agent = Agent.create(
        tools=["shell"],
        temperature=0.3
    )

    # Generate SQL from natural language
    result = await agent.run(
        "Generate a PostgreSQL query to find users who registered in the last 7 days. "
        "Table: users(id, email, created_at, name). "
        "Include: name, email, registration date."
    )

    print("Generated Query:")
    print(result.content)

    # Execute the query
    execute_result = await agent.run(
        "Execute this PostgreSQL query using psql: "
        f"SELECT * FROM users WHERE created_at > NOW() - INTERVAL '7 days' LIMIT 10"
    )

    return execute_result.content


async def sqlite_schema_analysis(db_path: str):
    """Analyze SQLite database schema."""
    from victor import Agent

    agent = Agent.create(
        tools=["read", "python"],
        vertical="dataanalysis",
        temperature=0.3
    )

    result = await agent.run(
        f"""Analyze the SQLite database at {db_path}.

        Provide:
        1. List all tables
        2. Schema for each table
        3. Relationships between tables
        4. Row counts per table
        5. Index information"""
    )

    return result.content


async def mongo_query_generator():
    """Generate MongoDB queries from natural language."""
    from victor import Agent

    agent = Agent.create(temperature=0.3)

    result = await agent.run(
        "Generate a MongoDB query to find documents in the 'users' collection "
        "where the 'status' field is 'active' and 'last_login' was within 30 days. "
        "Sort by 'last_login' descending. Limit to 10 results."
    )

    return result.content


async def redis_cache_strategy():
    """Design Redis caching strategy for an API."""
    from victor import Agent

    agent = Agent.create(temperature=0.3)

    result = await agent.run(
        """Design a Redis caching strategy for a REST API.

        API endpoints:
        - GET /users/{id} - Get user by ID
        - GET /users - List users with pagination
        - POST /users - Create user
        - PUT /users/{id} - Update user
        - DELETE /users/{id} - Delete user

        Requirements:
        - Cache frequently accessed users
        - Invalidate cache on updates/deletes
        - Set appropriate TTL values
        - Use hash tags for efficient invalidation

        Provide:
        1. Key naming conventions
        2. TTL values for different operations
        3. Cache invalidation strategy
        4. Example Redis commands"""
    )

    return result.content


async def etl_pipeline():
    """Design ETL pipeline for data migration."""
    from victor import Agent

    agent = Agent.create(
        tools=["python", "read", "write"],
        temperature=0.3
    )

    result = await agent.run(
        """Design an ETL pipeline to migrate data from PostgreSQL to MongoDB.

        Source (PostgreSQL):
        - Table: users(id, email, name, created_at, updated_at)
        - Table: orders(id, user_id, total, status, created_at)

        Target (MongoDB):
        - Collection: users
        - Collection: orders

        Requirements:
        - Handle schema differences
        - Transform dates to ISO8601
        - Handle relational data (orders -> users)
        - Batch processing (1000 records at a time)
        - Error handling and logging
        - Progress tracking

        Provide:
        1. Extraction code
        2. Transformation logic
        3. Loading strategy
        4. Error handling
        """
    )

    return result.content


async def data_validation_rules():
    """Generate data validation rules."""
    from victor import Agent

    agent = Agent.create(temperature=0.2)

    result = await agent.run(
        """Generate Python data validation rules for user registration data.

        Fields to validate:
        - email (required, valid format, unique)
        - password (required, min 8 chars, contains number+letter+special)
        - age (required, 18+, integer)
        - country (required, valid ISO country code)
        - phone (optional, E.164 format)

        Provide:
        1. Validation functions
        2. Error messages
        3. Pydantic schema
        4. Unit tests
        """
    )

    return result.content


async def database_backup_strategy():
    """Design database backup strategy."""
    from victor import Agent

    agent = Agent.create(temperature=0.3)

    result = await agent.run(
        """Design a comprehensive backup strategy for a production PostgreSQL database.

        Requirements:
        - Database size: ~100GB
        - RTO: 15 minutes (Recovery Time Objective)
        - RPO: 1 hour (Recovery Point Objective)
        - Budget constraint: $500/month for backup storage
        - Compliance: GDPR compliant (data in EU)

        Provide:
        1. Backup tools and setup
        2. Backup schedule (daily, weekly, monthly)
        3. Storage strategy (on-premises + cloud)
        4. Encryption requirements
        5. Testing procedures
        6. Disaster recovery plan
        """
    )

    return result.content


async def full_text_search_setup():
    """Setup full-text search with PostgreSQL."""
    from victor import Agent

    agent = Agent.create(
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        """Set up full-text search in PostgreSQL for documents.

        Table: documents(id, title, content, created_at, author_id)

        Requirements:
        - Search in title and content
        - Support phrase search with quotes
        - Rank results by relevance
        - Handle stemming (English language)
        - Support faceted search by author, date
        - Performance: < 100ms for 1M documents

        Provide:
        1. SQL commands to set up FTS
        2. Index creation
        3. Query examples
        4. Optimization tips
        """
    )

    return result.content


async def demo_database_integrations():
    """Demonstrate database integrations."""
    print("=== Database Integration Recipes ===\n")

    print("1. MongoDB Query Generator:")
    result = await mongo_query_generator()
    print(result)
    print()

    print("2. Redis Cache Strategy:")
    result = await redis_cache_strategy()
    print(result[:400] + "...")
    print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_database_integrations())
