#!/usr/bin/env python3
"""Air-gapped Codebase Semantic Search Example.

This demonstrates Victor's air-gapped codebase indexing and search:
- Works 100% offline (no network required)
- Local embeddings via sentence-transformers
- Local vector store via LanceDB
- Fast performance (~8ms per embedding)
- Production-ready scalability

Perfect for:
- Privacy-sensitive codebases
- Offline development
- Air-gapped environments
- Solo developers
- Small teams

Install dependencies:
    pip install lancedb sentence-transformers

No external servers required!
"""

import asyncio

from victor.storage.vector_stores import EmbeddingConfig, EmbeddingRegistry


async def main():
    """Demo air-gapped codebase search."""

    print("🔒 Air-gapped Codebase Semantic Search Demo")
    print("=" * 60)
    print()

    # ========================================
    # Configuration (Air-gapped Mode)
    # ========================================
    config = EmbeddingConfig(
        # Vector Store: LanceDB (local, disk-based, fast)
        vector_store="lancedb",
        persist_directory="~/.victor/embeddings/airgapped_demo",
        distance_metric="cosine",
        # Embedding Model: sentence-transformers (local, offline)
        embedding_model_type="sentence-transformers",
        embedding_model_name="all-MiniLM-L12-v2",  # 384-dim, 120MB, ~8ms
        # Alternative fast option: "all-MiniLM-L6-v2"  # 384-dim, 80MB, ~5ms
        # Alternative quality option: "all-mpnet-base-v2"  # 768-dim, 420MB, ~15ms
        # LanceDB-specific configuration
        extra_config={
            "table_name": "airgapped_codebase",
            "dimension": 384,  # Match model dimension
            "batch_size": 32,  # Adjust based on available RAM
        },
    )

    print(f"📦 Vector Store: {config.vector_store}")
    print(f"🤖 Embedding Model: {config.embedding_model_name} ({config.embedding_model_type})")
    print(f"📁 Storage: {config.persist_directory}")
    print("🌐 Network Required: NO (100% offline)")
    print()

    # Initialize provider via registry (auto-detects installed backends)
    provider = EmbeddingRegistry.create(config)
    await provider.initialize()

    print()
    print("=" * 60)
    print("📚 Indexing Code Snippets")
    print("=" * 60)
    print()

    # ========================================
    # Index Sample Code
    # ========================================
    code_snippets = [
        {
            "id": "auth_login",
            "content": """
def authenticate_user(username: str, password: str) -> Optional[User]:
    '''Authenticate user with username and password.

    Args:
        username: User's username
        password: User's password (will be hashed)

    Returns:
        User object if authenticated, None otherwise
    '''
    user = db.query(User).filter_by(username=username).first()
    if user and verify_password(password, user.password_hash):
        return user
    return None
""",
            "metadata": {
                "file_path": "src/auth/login.py",
                "symbol_name": "authenticate_user",
                "line_number": 15,
                "language": "python",
            },
        },
        {
            "id": "auth_jwt",
            "content": """
def create_jwt_token(user_id: int, expires_delta: timedelta = None) -> str:
    '''Create JWT token for authenticated user.

    Args:
        user_id: User ID to encode in token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token string
    '''
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + (expires_delta or timedelta(hours=24))
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
""",
            "metadata": {
                "file_path": "src/auth/jwt.py",
                "symbol_name": "create_jwt_token",
                "line_number": 8,
                "language": "python",
            },
        },
        {
            "id": "db_connection",
            "content": """
async def get_database_connection(pool: ConnectionPool) -> AsyncConnection:
    '''Get database connection from pool with retry logic.

    Args:
        pool: Database connection pool

    Returns:
        Active database connection

    Raises:
        ConnectionError: If connection fails after retries
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = await pool.acquire()
            await conn.ping()  # Verify connection
            return conn
        except Exception as e:
            if attempt == max_retries - 1:
                raise ConnectionError(f"Failed to connect: {e}")
            await asyncio.sleep(1)
""",
            "metadata": {
                "file_path": "src/database/connection.py",
                "symbol_name": "get_database_connection",
                "line_number": 22,
                "language": "python",
            },
        },
        {
            "id": "cache_decorator",
            "content": """
def cache_result(ttl: int = 3600):
    '''Decorator to cache function results in Redis.

    Args:
        ttl: Time to live in seconds (default: 1 hour)

    Returns:
        Decorated function with caching
    '''
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
            result = await func(*args, **kwargs)
            await redis.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
""",
            "metadata": {
                "file_path": "src/utils/cache.py",
                "symbol_name": "cache_result",
                "line_number": 5,
                "language": "python",
            },
        },
        {
            "id": "api_handler",
            "content": """
@app.post("/api/users")
async def create_user(user: UserCreate, db: Session = Depends(get_db)) -> User:
    '''Create new user account.

    Args:
        user: User creation data
        db: Database session

    Returns:
        Created user object

    Raises:
        HTTPException: If username already exists
    '''
    existing = db.query(User).filter_by(username=user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(
        username=user.username,
        email=user.email,
        password_hash=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
""",
            "metadata": {
                "file_path": "src/api/users.py",
                "symbol_name": "create_user",
                "line_number": 12,
                "language": "python",
            },
        },
    ]

    # Index documents (batch operation)
    await provider.index_documents(code_snippets)

    print()
    print("=" * 60)
    print("🔍 Semantic Search Queries")
    print("=" * 60)
    print()

    # ========================================
    # Search Examples
    # ========================================
    queries = [
        "how to authenticate users with username and password",
        "JWT token generation for API authentication",
        "database connection with retry logic",
        "cache decorator for Redis",
        "create new user account endpoint",
    ]

    for query in queries:
        print(f"📝 Query: '{query}'")
        print("-" * 60)

        results = await provider.search_similar(query, limit=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.symbol_name} (score: {result.score:.3f})")
            print(f"     📄 {result.file_path}:{result.line_number}")
            print(f"     {result.content.strip()[:100]}...")
            print()

        if not results:
            print("  ❌ No results found")
            print()

    # ========================================
    # Statistics
    # ========================================
    print()
    print("=" * 60)
    print("📊 Index Statistics")
    print("=" * 60)
    print()

    stats = await provider.get_stats()
    print(f"  Provider: {stats['provider']}")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Embedding model: {stats['embedding_model_name']} ({stats['embedding_model_type']})")
    print(f"  Dimension: {stats['dimension']}")
    print(f"  Distance metric: {stats['distance_metric']}")
    print(f"  Storage: {stats['persist_directory']}")
    print()

    # Clean up
    await provider.close()

    print("=" * 60)
    print("✅ Demo Complete!")
    print()
    print("Key Benefits:")
    print("  ✅ 100% offline (no network required)")
    print("  ✅ Fast (~8ms per embedding)")
    print("  ✅ Privacy-preserving (data stays local)")
    print("  ✅ Production-ready (LanceDB scales to millions)")
    print("  ✅ Low memory footprint (disk-based storage)")
    print()
    print(f"Indexed data persisted to: {config.persist_directory}")
    print("Run this script again to reuse the index (no re-indexing needed)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
