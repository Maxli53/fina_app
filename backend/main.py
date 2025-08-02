from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import logging
import asyncio

from app.api import health, data, analysis, strategy, trading, system, ai_advisor
from app.services.system_orchestrator import SystemOrchestrator
from app.dependencies import get_db_session, get_redis_client

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
    
    # Initialize System Orchestrator
    try:
        logger.info("Initializing System Orchestrator...")
        
        # Get database and redis connections
        db = await get_db_session()
        redis = await get_redis_client()
        
        # Create orchestrator config
        config = {
            "use_gpu": True,
            "health_check_interval": 30,
            "risk_check_interval": 60,
            "data_pipeline_interval": 1,
            "websocket_port": 8765,
            "risk_limits": {
                "var_95_limit": 10000,
                "max_daily_loss": 5000,
                "max_concentration": 0.30
            }
        }
        
        # Create and initialize orchestrator
        orchestrator = SystemOrchestrator(db, redis, config)
        await orchestrator.initialize()
        
        # Set orchestrator in system API
        system.set_orchestrator(orchestrator)
        
        # Start orchestrator in background
        asyncio.create_task(orchestrator.start())
        
        logger.info("System Orchestrator initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize System Orchestrator: {e}")
        # Continue without orchestrator for now
    
    yield
    
    # Shutdown
    logger.info("Shutting down Financial Analysis Platform API...")
    
    # Stop orchestrator if running
    try:
        if 'orchestrator' in locals():
            await orchestrator.stop()
    except Exception as e:
        logger.error(f"Error stopping orchestrator: {e}")


app = FastAPI(
    title="Financial Time Series Analysis Platform",
    description="Advanced quantitative analysis platform integrating IDTxl, ML, and Neural Networks",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "ws://localhost:8765"],  # React dev server and WebSocket
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(strategy.router, prefix="/api/strategy", tags=["strategy"])
app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
app.include_router(system.router, tags=["system"])
app.include_router(ai_advisor.router, tags=["ai-advisor"])


@app.get("/")
async def root():
    return {
        "message": "Financial Time Series Analysis Platform API",
        "version": "0.1.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)