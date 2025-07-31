from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import logging

from app.api import health, data, analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Financial Analysis Platform API...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Python path: {os.getenv('PYTHONPATH', 'not set')}")
    
    # Check environment variables
    db_url = os.getenv('DATABASE_URL')
    redis_url = os.getenv('REDIS_URL')
    if db_url:
        logger.info(f"Database URL configured: {db_url.split('@')[0]}@***")
    if redis_url:
        logger.info(f"Redis URL configured: {redis_url}")
    
    yield
    # Shutdown
    logger.info("Shutting down Financial Analysis Platform API...")


app = FastAPI(
    title="Financial Time Series Analysis Platform",
    description="Advanced quantitative analysis platform integrating IDTxl, ML, and Neural Networks",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])


@app.get("/")
async def root():
    return {
        "message": "Financial Time Series Analysis Platform API",
        "version": "0.1.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)