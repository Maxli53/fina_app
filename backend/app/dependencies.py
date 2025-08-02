"""
Application dependencies
"""
import os
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/financedb")
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_redis_client() -> redis.Redis:
    """Get Redis client"""
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        await client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        # Return None or a mock client for development
        return None